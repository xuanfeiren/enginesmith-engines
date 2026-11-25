"""probe: nkilib attention_cte split into two exact sub-problems (block 0 alone; block 1 with block 0 as prior) vs one
call over the whole sequence: bitwise?"""
import numpy as np, ml_dtypes
import nki
import nki.isa as nisa
import nki.language as nl
from nkilib.core.attention.attention_cte import _attention_cte

SCALE = 0.08838834764831845


@nki.jit
def att(q, k, v, H, MODE):
    NQ, D, S = q.shape
    NK = k.shape[0]
    bf = nl.bfloat16
    o1 = nl.ndarray((NQ, D, S), dtype=bf, buffer=nl.shared_hbm)
    o2 = nl.ndarray((NQ, D, H), dtype=bf, buffer=nl.shared_hbm)
    o3 = nl.ndarray((NQ, D, S - H), dtype=bf, buffer=nl.shared_hbm)
    sq = nl.ndarray((NQ, D, S), dtype=bf, buffer=nl.shared_hbm, name="sq")
    sk = nl.ndarray((NK, D, S), dtype=bf, buffer=nl.shared_hbm, name="sk")
    sv = nl.ndarray((NK, S, D), dtype=bf, buffer=nl.shared_hbm, name="sv")
    so = nl.ndarray((NQ, D, S), dtype=bf, buffer=nl.shared_hbm, name="so")
    q0 = nl.ndarray((NQ, D, H), dtype=bf, buffer=nl.shared_hbm, name="q0")
    k0 = nl.ndarray((NK, D, H), dtype=bf, buffer=nl.shared_hbm, name="k0")
    v0 = nl.ndarray((NK, H, D), dtype=bf, buffer=nl.shared_hbm, name="v0")
    so0 = nl.ndarray((NQ, D, H), dtype=bf, buffer=nl.shared_hbm, name="so0")
    q1 = nl.ndarray((NQ, D, S - H), dtype=bf, buffer=nl.shared_hbm, name="q1")
    k1 = nl.ndarray((NK, D, S - H), dtype=bf, buffer=nl.shared_hbm, name="k1")
    v1 = nl.ndarray((NK, S - H, D), dtype=bf, buffer=nl.shared_hbm, name="v1")
    so1 = nl.ndarray((NQ, D, S - H), dtype=bf, buffer=nl.shared_hbm, name="so1")
    pl = nl.ndarray((1,), dtype=nl.int32, buffer=nl.shared_hbm, name="pl")
    pid = nl.program_id(0)
    if True:
        for i in range(NQ):
            t = nl.ndarray((D, S), dtype=bf, buffer=nl.sbuf)
            nisa.dma_copy(dst=t, src=q[i])
            nisa.dma_copy(dst=sq[i], src=t)
            nisa.dma_copy(dst=q0[i], src=t[:, 0:H])
            nisa.dma_copy(dst=q1[i], src=t[:, H:S])
        for i in range(NK):
            t = nl.ndarray((D, S), dtype=bf, buffer=nl.sbuf)
            nisa.dma_copy(dst=t, src=k[i])
            nisa.dma_copy(dst=sk[i], src=t)
            nisa.dma_copy(dst=k0[i], src=t[:, 0:H])
            nisa.dma_copy(dst=k1[i], src=t[:, H:S])
            for j in range(S // 128):
                tv = nl.ndarray((128, D), dtype=bf, buffer=nl.sbuf)
                nisa.dma_copy(dst=tv, src=v[i, j * 128:(j + 1) * 128, :])
                nisa.dma_copy(dst=sv[i, j * 128:(j + 1) * 128, :], src=tv)
                if j * 128 < H:
                    nisa.dma_copy(dst=v0[i, j * 128:(j + 1) * 128, :], src=tv)
                else:
                    nisa.dma_copy(dst=v1[i, j * 128 - H:(j + 1) * 128 - H, :], src=tv)
    nisa.core_barrier(data=sq, cores=(0, 1))
    nisa.core_barrier(data=sk, cores=(0, 1))
    nisa.core_barrier(data=sv, cores=(0, 1))
    _attention_cte(q=sq, k=sk, v=sv, scale=SCALE, causal_mask=True, tp_q=False, tp_k=False, tp_out=True, out_o_hbm=so)
    _attention_cte(q=q0, k=k0, v=v0, scale=SCALE, causal_mask=True, tp_q=False, tp_k=False, tp_out=True, out_o_hbm=so0)
    nisa.core_barrier(data=so, cores=(0, 1))
    nisa.core_barrier(data=so0, cores=(0, 1))
    nisa.core_barrier(data=so1, cores=(0, 1))
    for i in range(NQ):
        t = nl.ndarray((D, S), dtype=bf, buffer=nl.sbuf)
        nisa.dma_copy(dst=t, src=so[i])
        nisa.dma_copy(dst=o1[i], src=t)
        t2 = nl.ndarray((D, H), dtype=bf, buffer=nl.sbuf)
        nisa.dma_copy(dst=t2, src=so0[i])
        nisa.dma_copy(dst=o2[i], src=t2)
        t3 = nl.ndarray((D, S - H), dtype=bf, buffer=nl.sbuf)
        nisa.dma_copy(dst=t3, src=so1[i])
        nisa.dma_copy(dst=o3[i], src=t3)
    return o1, o2, o3


def run(S, H):
    rng = np.random.default_rng(0)
    q = (rng.standard_normal((8, 128, S)) * 2).astype(ml_dtypes.bfloat16)
    k = (rng.standard_normal((2, 128, S)) * 2).astype(ml_dtypes.bfloat16)
    v = rng.standard_normal((2, S, 128)).astype(ml_dtypes.bfloat16)
    o1, o2, o3 = [np.asarray(x) for x in att[2](q, k, v, H, 0)]
    u = lambda a: a.view(np.uint16)
    d0 = int((u(o1[:, :, :H]) != u(o2)).sum()); d1 = int((u(o1[:, :, H:]) != u(o3)).sum())
    e0 = float(np.abs(o1[:, :, :H].astype(np.float32) - o2.astype(np.float32)).max())
    e1 = float(np.abs(o1[:, :, H:].astype(np.float32) - o3.astype(np.float32)).max())
    return {"S": S, "H": H, "diff_c0": d0, "diff_c1": d1, "maxerr0": e0, "maxerr1": e1, "n": o2.size, "absmax": float(np.abs(o1.astype(np.float32)).max())}
