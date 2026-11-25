"""probe (round 008): reduce_scatter of an SP block [16, 128, 40, 32] bf16 vs two reduce_scatters of hidden-tile halves
(tiles {0-9, 20-29} and {10-19, 30-39}): bitwise equal per element?"""
import numpy as np
import nki
import nki.isa as nisa
import nki.language as nl
import nki.collectives as ncc
from nki.collectives import ReplicaGroup
import ml_dtypes

W = 8
G0 = list(range(0, 10)) + list(range(20, 30))
G1 = list(range(10, 20)) + list(range(30, 40))


@nki.jit
def k(x, x0, x1, rg):
    Q, P, HT, HC = x.shape
    hb = nl.ndarray((Q, P, HT, HC), dtype=x.dtype, buffer=nl.shared_hbm, name="h_hb")
    h0 = nl.ndarray((Q, P, HT // 2, HC), dtype=x.dtype, buffer=nl.shared_hbm, name="h_h0")
    h1 = nl.ndarray((Q, P, HT // 2, HC), dtype=x.dtype, buffer=nl.shared_hbm, name="h_h1")
    nisa.dma_copy(dst=hb, src=x)
    nisa.dma_copy(dst=h0, src=x0)
    nisa.dma_copy(dst=h1, src=x1)
    r = nl.ndarray((2, P, HT, HC), dtype=x.dtype, buffer=nl.shared_hbm, name="h_r")
    r0 = nl.ndarray((2, P, HT // 2, HC), dtype=x.dtype, buffer=nl.shared_hbm, name="h_r0")
    r1 = nl.ndarray((2, P, HT // 2, HC), dtype=x.dtype, buffer=nl.shared_hbm, name="h_r1")
    ncc.reduce_scatter(srcs=[hb], dsts=[r], op=nl.add, replica_group=rg, collective_dim=0)
    ncc.reduce_scatter(srcs=[h0], dsts=[r0], op=nl.add, replica_group=rg, collective_dim=0)
    ncc.reduce_scatter(srcs=[h1], dsts=[r1], op=nl.add, replica_group=rg, collective_dim=0)
    o = nl.ndarray((2, P, HT, HC), dtype=x.dtype, buffer=nl.shared_hbm)
    o0 = nl.ndarray((2, P, HT // 2, HC), dtype=x.dtype, buffer=nl.shared_hbm)
    o1 = nl.ndarray((2, P, HT // 2, HC), dtype=x.dtype, buffer=nl.shared_hbm)
    nisa.dma_copy(dst=o, src=r)
    nisa.dma_copy(dst=o0, src=r0)
    nisa.dma_copy(dst=o1, src=r1)
    return o, o0, o1


def run():
    rng = np.random.default_rng(1)
    xs = [(rng.standard_normal((16, 128, 40, 32), dtype=np.float32) * 3).astype(ml_dtypes.bfloat16) for _ in range(W)]
    x0 = [np.ascontiguousarray(x[:, :, G0, :]) for x in xs]
    x1 = [np.ascontiguousarray(x[:, :, G1, :]) for x in xs]
    outs = k[2](xs, x0, x1, ReplicaGroup([list(range(W))]))
    res = []
    for r in range(W):
        o, o0, o1 = [np.asarray(a) for a in outs[r]]
        a = o.view(np.uint16)
        eq0 = np.array_equal(a[:, :, G0, :], o0.view(np.uint16))
        eq1 = np.array_equal(a[:, :, G1, :], o1.view(np.uint16))
        res.append((r, eq0, eq1, int((a[:, :, G0, :] != o0.view(np.uint16)).sum())))
    return res
