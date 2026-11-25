"""probe: reduce_scatter / all_gather vs all_reduce on HBM, 8 ranks, prefill-block sized ([128, 40, 512] bf16)."""
import os, sys
import numpy as np
import nki
import nki.isa as nisa
import nki.language as nl
import nki.collectives as ncc
from nki.collectives import ReplicaGroup
import ml_dtypes

W = 8


@nki.jit
def chk(x, rg, T=512):
    """x [128, 40, T] bf16 -> (all_reduce, reduce_scatter of token shards, all_gather of the shards)"""
    P, HT, _ = x.shape
    TS = T // W
    hb = nl.ndarray((P, HT, T), dtype=x.dtype, buffer=nl.shared_hbm, name="cc_hb")
    hr = nl.ndarray((P, HT, T), dtype=x.dtype, buffer=nl.shared_hbm, name="cc_hr")
    h8 = nl.ndarray((W, P, HT, TS), dtype=x.dtype, buffer=nl.shared_hbm, name="cc_h8")
    rs = nl.ndarray((1, P, HT, TS), dtype=x.dtype, buffer=nl.shared_hbm, name="cc_rs")
    ag = nl.ndarray((W, P, HT, TS), dtype=x.dtype, buffer=nl.shared_hbm, name="cc_ag")
    nisa.dma_copy(dst=hb, src=x)
    for r in range(W):
        nisa.dma_copy(dst=h8[r], src=x[:, :, r * TS:(r + 1) * TS])
    ncc.all_reduce(srcs=[hb], dsts=[hr], op=nl.add, replica_group=rg)
    ncc.reduce_scatter(srcs=[h8], dsts=[rs], op=nl.add, replica_group=rg, collective_dim=0)
    ncc.all_gather(srcs=[rs], dsts=[ag], replica_group=rg, collective_dim=0)
    o1 = nl.ndarray((P, HT, T), dtype=x.dtype, buffer=nl.shared_hbm)
    o2 = nl.ndarray((P, HT, TS), dtype=x.dtype, buffer=nl.shared_hbm)
    o3 = nl.ndarray((W, P, HT, TS), dtype=x.dtype, buffer=nl.shared_hbm)
    nisa.dma_copy(dst=o1, src=hr)
    nisa.dma_copy(dst=o2, src=rs[0])
    nisa.dma_copy(dst=o3, src=ag)
    rk = ncc.rank_id()
    ti = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.sbuf)
    nisa.register_store(ti, rk)
    o4 = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.shared_hbm)
    nisa.dma_copy(dst=o4, src=ti)
    # own shard of x through a register offset: x viewed [128, 40, 8, TS], index rank on dim 2
    sh = nl.ndarray((P, HT, TS), dtype=x.dtype, buffer=nl.sbuf)
    rk2 = nisa.register_alloc()
    nisa.register_load(rk2, ti)
    nisa.dma_copy(dst=sh, src=x.reshape((P, HT, W, TS)).ap(pattern=[[HT * T, P], [T, HT], [1, TS]], offset=0, scalar_offset=rk2, indirect_dim=2))
    o5 = nl.ndarray((P, HT, TS), dtype=x.dtype, buffer=nl.shared_hbm)
    nisa.dma_copy(dst=o5, src=sh)
    pid = nl.program_id(0)
    rp = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=rp, data=ti, op0=nl.multiply, operand0=2, op1=nl.add, operand1=pid)
    rr = nisa.register_alloc()
    nisa.register_load(rr, rp)
    HC = TS // 2
    s6 = nl.ndarray((P, HT, HC), dtype=x.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=s6, src=x.reshape((P, HT, 2 * W, HC)).ap(pattern=[[HT * T, P], [T, HT], [1, HC]], offset=0, scalar_offset=rr, indirect_dim=2))
    o6 = nl.ndarray((2, P, HT, HC), dtype=x.dtype, buffer=nl.shared_hbm)
    nisa.dma_copy(dst=o6[pid], src=s6)
    return o1, o2, o3, o4, o5, o6


@nki.jit
def bench(x, rg, mode, K, T=512):
    P, HT, _ = x.shape
    TS = T // W
    out = nl.ndarray((1, 1), dtype=nl.float32, buffer=nl.shared_hbm)
    hb = nl.ndarray((P, HT, T), dtype=x.dtype, buffer=nl.shared_hbm, name="cc_hb")
    h8 = nl.ndarray((W, P, HT, TS), dtype=x.dtype, buffer=nl.shared_hbm, name="cc_h8")
    nisa.dma_copy(dst=hb, src=x)
    for r in range(W):
        nisa.dma_copy(dst=h8[r], src=x[:, :, r * TS:(r + 1) * TS])
    for i in range(K):
        if mode == 0:
            hr = nl.ndarray((P, HT, T), dtype=x.dtype, buffer=nl.shared_hbm, name=f"cc_hr{i}")
            ncc.all_reduce(srcs=[hb], dsts=[hr], op=nl.add, replica_group=rg)
        elif mode == 1:
            rs = nl.ndarray((1, P, HT, TS), dtype=x.dtype, buffer=nl.shared_hbm, name=f"cc_rs{i}")
            ncc.reduce_scatter(srcs=[h8], dsts=[rs], op=nl.add, replica_group=rg, collective_dim=0)
        else:
            ag = nl.ndarray((W, P, HT, TS), dtype=x.dtype, buffer=nl.shared_hbm, name=f"cc_ag{i}")
            ncc.all_gather(srcs=[h8[0:1]], dsts=[ag], replica_group=rg, collective_dim=0)
    t = nl.ndarray((1, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(dst=t, value=0.0)
    nisa.dma_copy(dst=out, src=t)
    return out


def run_chk(T=512):
    rng = np.random.default_rng(0)
    xs = [rng.standard_normal((128, 40, T), dtype=np.float32).astype(ml_dtypes.bfloat16) for _ in range(W)]
    outs = chk[2](xs, ReplicaGroup([list(range(W))]), T)
    TS = T // W
    res = []
    for r in range(W):
        o1, o2, o3, o4, o5, o6 = [np.asarray(a) for a in outs[r]]
        okr = int(o4.ravel()[0])
        oksh = [k for k in range(512 - 31) if np.array_equal(o5.view(np.uint16), np.asarray(xs[r])[:, :, k:k + TS].view(np.uint16))]
        ar_sl = o1[:, :, r * TS:(r + 1) * TS]
        eq = np.array_equal(ar_sl.view(np.uint16), o2.view(np.uint16))
        nd = int((ar_sl.view(np.uint16) != o2.view(np.uint16)).sum())
        ag_full = np.concatenate([o3[k] for k in range(W)], axis=2)
        eqg = np.array_equal(ag_full.view(np.uint16), o1.view(np.uint16))
        o6k = [[k for k in range(512 - 31) if np.array_equal(o6[c].view(np.uint16), np.asarray(xs[r])[:, :, k:k + TS // 2].view(np.uint16))] for c in range(2)]
        res.append((r, eq, nd, eqg, okr, oksh, o6k))
    ref = sum(x.astype(np.float32) for x in xs)
    err = float(np.abs(np.asarray(outs[0][0]).astype(np.float32) - ref).max())
    return {"per_rank": res, "ar_err_vs_fp32": err}


def run_bench(mode, K, T=512):
    xs = [np.zeros((128, 40, T), dtype=ml_dtypes.bfloat16) for _ in range(W)]
    bench[2](xs, ReplicaGroup([list(range(W))]), mode, K, T)
    return {}
