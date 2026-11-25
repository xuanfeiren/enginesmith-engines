"""probe (round 008): do collectives on two replica groups with the same ranks in a different order run on two CC
streams (concurrently)? Semantics of reduce_scatter / all_gather / all_reduce on a permuted group."""
import numpy as np
import nki
import nki.isa as nisa
import nki.language as nl
import nki.collectives as ncc
from nki.collectives import ReplicaGroup
import ml_dtypes

W = 8
PERM = [1, 0, 3, 2, 5, 4, 7, 6]


@nki.jit
def sem(x, rgA, rgB):
    """x [W, 128, 64] bf16 per rank: RS / AG / AR on group B (permuted)"""
    Q, P, F = x.shape
    h = nl.ndarray((Q, P, F), dtype=x.dtype, buffer=nl.shared_hbm, name="s_h")
    nisa.dma_copy(dst=h, src=x)
    rs = nl.ndarray((1, P, F), dtype=x.dtype, buffer=nl.shared_hbm, name="s_rs")
    ncc.reduce_scatter(srcs=[h], dsts=[rs], op=nl.add, replica_group=rgB, collective_dim=0)
    ag = nl.ndarray((Q, P, F), dtype=x.dtype, buffer=nl.shared_hbm, name="s_ag")
    ncc.all_gather(srcs=[h[0:1]], dsts=[ag], replica_group=rgB, collective_dim=0)
    ar = nl.ndarray((Q, P, F), dtype=x.dtype, buffer=nl.shared_hbm, name="s_ar")
    ncc.all_reduce(srcs=[h], dsts=[ar], op=nl.add, replica_group=rgB, collective_dim=0) if False else ncc.all_reduce(srcs=[h], dsts=[ar], op=nl.add, replica_group=rgB)
    arA = nl.ndarray((Q, P, F), dtype=x.dtype, buffer=nl.shared_hbm, name="s_arA")
    ncc.all_reduce(srcs=[h], dsts=[arA], op=nl.add, replica_group=rgA)
    o1 = nl.ndarray((P, F), dtype=x.dtype, buffer=nl.shared_hbm)
    o2 = nl.ndarray((Q, P, F), dtype=x.dtype, buffer=nl.shared_hbm)
    o3 = nl.ndarray((Q, P, F), dtype=x.dtype, buffer=nl.shared_hbm)
    o4 = nl.ndarray((Q, P, F), dtype=x.dtype, buffer=nl.shared_hbm)
    nisa.dma_copy(dst=o1, src=rs[0])
    nisa.dma_copy(dst=o2, src=ag)
    nisa.dma_copy(dst=o3, src=ar)
    nisa.dma_copy(dst=o4, src=arA)
    return o1, o2, o3, o4


@nki.jit
def par(x, rgA, rgB, mode, K):
    """K independent all-reduces of x [128, 40, 256] bf16 (2.6 MB): mode 0 all on group A, 1 alternating A / B"""
    P, HT, T = x.shape
    hb = nl.ndarray((P, HT, T), dtype=x.dtype, buffer=nl.shared_hbm, name="p_hb")
    nisa.dma_copy(dst=hb, src=x)
    for i in range(K):
        hr = nl.ndarray((P, HT, T), dtype=x.dtype, buffer=nl.shared_hbm, name=f"p_hr{i}")
        g = rgA
        if mode == 1 and i % 2 == 1:
            g = rgB
        ncc.all_reduce(srcs=[hb], dsts=[hr], op=nl.add, replica_group=g)
    out = nl.ndarray((1, 1), dtype=nl.float32, buffer=nl.shared_hbm)
    t = nl.ndarray((1, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(dst=t, value=0.0)
    nisa.dma_copy(dst=out, src=t)
    return out


def run_sem():
    rng = np.random.default_rng(0)
    xs = [rng.standard_normal((W, 128, 64), dtype=np.float32).astype(ml_dtypes.bfloat16) for _ in range(W)]
    outs = sem[2](xs, ReplicaGroup([list(range(W))]), ReplicaGroup([PERM]))
    res = []
    for r in range(W):
        o1, o2, o3, o4 = [np.asarray(a) for a in outs[r]]
        u = lambda a: a.view(np.uint16)
        # which shard index k did RS give rank r?  sum over ranks of xs[*][k]
        rsk = [k for k in range(W) if np.allclose(o1.astype(np.float32), sum(np.asarray(xs[q][k]).astype(np.float32) for q in range(W)), atol=0.1)]
        # AG slot j holds which rank's x[0]?
        agj = [[q for q in range(W) if np.array_equal(u(o2[j]), u(np.asarray(xs[q][0])))] for j in range(W)]
        res.append((r, rsk, agj, bool(np.array_equal(u(o3), u(o4)))))
    return res


def run_par(mode, K):
    xs = [np.zeros((128, 40, 256), dtype=ml_dtypes.bfloat16) for _ in range(W)]
    par[2](xs, ReplicaGroup([list(range(W))]), ReplicaGroup([PERM]), mode, K)
    return "ok"
