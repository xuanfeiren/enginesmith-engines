"""probe (round 008): cost of SP-sized collectives, single vs coalesced (two tensors in one op), queued back to back"""
import numpy as np
import nki
import nki.isa as nisa
import nki.language as nl
import nki.collectives as ncc
from nki.collectives import ReplicaGroup
import ml_dtypes

W = 8


@nki.jit
def bench(x, rg, mode, K):
    """x [16, 128, 40, 32] bf16 (an SP block slot-major buffer, 5.2 MB). mode 0: K RS; 1: K/2 coalesced RS of 2;
    2: K AG of [2,128,40,32]; 3: K/2 coalesced AG of 2"""
    Q, P, HT, HC = x.shape
    hb = nl.ndarray((Q, P, HT, HC), dtype=x.dtype, buffer=nl.shared_hbm, name="c_hb")
    hb2 = nl.ndarray((Q, P, HT, HC), dtype=x.dtype, buffer=nl.shared_hbm, name="c_hb2")
    nisa.dma_copy(dst=hb, src=x)
    nisa.dma_copy(dst=hb2, src=x)
    xa = nl.ndarray((2, P, HT, HC), dtype=x.dtype, buffer=nl.shared_hbm, name="c_xa")
    xa2 = nl.ndarray((2, P, HT, HC), dtype=x.dtype, buffer=nl.shared_hbm, name="c_xa2")
    nisa.dma_copy(dst=xa, src=x[0:2])
    nisa.dma_copy(dst=xa2, src=x[2:4])
    for i in range(K):
        if mode == 0:
            r = nl.ndarray((2, P, HT, HC), dtype=x.dtype, buffer=nl.shared_hbm, name=f"c_r{i}")
            ncc.reduce_scatter(srcs=[hb], dsts=[r], op=nl.add, replica_group=rg, collective_dim=0)
        elif mode == 1 and i % 2 == 0:
            r = nl.ndarray((2, P, HT, HC), dtype=x.dtype, buffer=nl.shared_hbm, name=f"c_r{i}")
            r2 = nl.ndarray((2, P, HT, HC), dtype=x.dtype, buffer=nl.shared_hbm, name=f"c_rr{i}")
            ncc.reduce_scatter(srcs=[hb, hb2], dsts=[r, r2], op=nl.add, replica_group=rg, collective_dim=0)
        elif mode == 2:
            g = nl.ndarray((Q, P, HT, HC), dtype=x.dtype, buffer=nl.shared_hbm, name=f"c_g{i}")
            ncc.all_gather(srcs=[xa], dsts=[g], replica_group=rg, collective_dim=0)
        elif mode == 3 and i % 2 == 0:
            g = nl.ndarray((Q, P, HT, HC), dtype=x.dtype, buffer=nl.shared_hbm, name=f"c_g{i}")
            g2 = nl.ndarray((Q, P, HT, HC), dtype=x.dtype, buffer=nl.shared_hbm, name=f"c_gg{i}")
            ncc.all_gather(srcs=[xa, xa2], dsts=[g, g2], replica_group=rg, collective_dim=0)
    out = nl.ndarray((1, 1), dtype=nl.float32, buffer=nl.shared_hbm)
    t = nl.ndarray((1, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(dst=t, value=0.0)
    nisa.dma_copy(dst=out, src=t)
    return out


def run(mode, K):
    xs = [np.zeros((16, 128, 40, 32), dtype=ml_dtypes.bfloat16) for _ in range(W)]
    bench[2](xs, ReplicaGroup([list(range(W))]), mode, K)
    return "ok"
