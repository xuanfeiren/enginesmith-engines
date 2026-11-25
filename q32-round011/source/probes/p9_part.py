"""probe (round 009): elementwise ops across partition offsets (psum rows 64-127 -> sbuf rows 0-63)"""
import numpy as np, nki, nki.isa as nisa, nki.language as nl, ml_dtypes

@nki.jit
def k(w, x, MODE):
    out = nl.ndarray((64, 512), dtype=nl.float32, buffer=nl.shared_hbm)
    ws = nl.ndarray((128, 128), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.dma_copy(dst=ws, src=w)
    xs = nl.ndarray((128, 512), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.dma_copy(dst=xs, src=x)
    ps = nl.ndarray((128, 512), dtype=nl.float32, buffer=nl.psum)
    nisa.nc_matmul(dst=ps, stationary=ws, moving=xs)
    sg = nl.ndarray((64, 512), dtype=nl.float32, buffer=nl.sbuf)
    nisa.activation(dst=sg, op=nl.silu, data=ps[0:64, :])
    o = nl.ndarray((64, 512), dtype=nl.float32, buffer=nl.sbuf)
    if MODE == 0:     # direct: data2 on psum partitions 64-127
        nisa.tensor_tensor(dst=o, data1=sg, data2=ps[64:128, :], op=nl.multiply)
    elif MODE == 1:   # copy psum 64-127 -> sbuf rows 0-63 by an engine, then multiply
        u = nl.ndarray((64, 512), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=u, src=ps[64:128, :])
        nisa.tensor_tensor(dst=o, data1=sg, data2=u, op=nl.multiply)
    else:             # sbuf copy at 64-127, DMA shift to 0-63
        u2 = nl.ndarray((128, 512), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=u2[64:128, :], src=ps[64:128, :])
        u = nl.ndarray((64, 512), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=u, src=u2[64:128, :])
        nisa.tensor_tensor(dst=o, data1=sg, data2=u, op=nl.multiply)
    nisa.dma_copy(dst=out, src=o)
    return out


def run(MODE):
    rng = np.random.default_rng(1)
    w = (rng.standard_normal((128, 128), dtype=np.float32) * 0.1).astype(ml_dtypes.bfloat16)
    x = rng.standard_normal((128, 512), dtype=np.float32).astype(ml_dtypes.bfloat16)
    o = np.asarray(k(w, x, MODE))
    wf = w.astype(np.float32); xf = x.astype(np.float32)
    p = wf.T @ xf
    g = p[0:64]; u = p[64:128]
    ref = g / (1 + np.exp(-g)) * u
    return float(np.abs(o - ref).max()), float(np.abs(ref).max())
