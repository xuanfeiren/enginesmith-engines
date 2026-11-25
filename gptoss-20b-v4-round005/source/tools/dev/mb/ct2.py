# variant e: column tile c computes outputs [gate 90c..90c+90 | up 90c..] over all kt (no cross-tile sum);
# 2 fp32 transposes per expert -> gate / up [90, 4] (partition p, column c = index 90c + p)
import os, sys
import numpy as np, ml_dtypes
import nki, nki.language as nl, nki.isa as nisa

MODE = sys.argv[1]; REP = int(sys.argv[2])
HT = 24; GU = 720; NE = 4; Q = 90; C4 = 180
F = HT * GU


@nki.jit
def k(x, w):
    out = nl.ndarray((REP * NE, Q, 8), dtype=nl.float32, buffer=nl.shared_hbm)
    xs = nl.ndarray((128, HT), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.dma_copy(dst=xs, src=x)
    x32 = nl.ndarray((128, HT, 32), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.memset(dst=x32, value=0.0)
    for kt in range(HT):
        nisa.tensor_copy(dst=x32[0:128, kt, 0:1], src=xs[0:128, kt:kt + 1])
    ws = []
    for e in range(NE):
        t = nl.ndarray((128, F), dtype=nl.float8_e5m2, buffer=nl.sbuf)
        nisa.dma_copy(dst=t, src=w[e])
        ws.append(t)
    for r in range(REP):
        for e in range(NE):
            gs = ws[e]
            acc = nl.ndarray((128, 512), dtype=nl.float32, buffer=nl.psum)
            for kt in range(HT):
                for c in range(4):
                    nisa.nc_matmul(dst=acc[32 * c:32 * c + 32, 0:C4], stationary=x32[0:128, kt, 0:32],
                                   moving=gs[0:128, kt * GU + c * C4:kt * GU + (c + 1) * C4],
                                   tile_position=(0, 32 * c), tile_size=(128, 32))
            r4 = nl.ndarray((128, C4), dtype=nl.float32, buffer=nl.sbuf)
            if MODE == "e":
                nisa.tensor_copy(dst=r4, src=acc[0:128, 0:C4])
            else:
                nisa.activation(dst=r4, op=nl.copy, data=acc[0:128, 0:C4])
            tp = nl.ndarray((Q, 2, 128), dtype=nl.float32, buffer=nl.psum)
            for j in range(2):
                nisa.nc_transpose(dst=tp[0:Q, j, 0:128], data=r4[0:128, j * Q:(j + 1) * Q])
            gu = nl.ndarray((Q, 2, 4), dtype=nl.float32, buffer=nl.sbuf)
            tpv = tp.reshape((Q, 2, 4, 32))
            nisa.tensor_copy(dst=gu, src=tpv[0:Q, 0:2, 0:4, 0])
            nisa.dma_copy(dst=out[r * NE + e], src=gu.reshape((Q, 8)))
    return out


rng = np.random.default_rng(0)
x = rng.standard_normal((128, HT)).astype(ml_dtypes.bfloat16)
w = (rng.standard_normal((NE, 128, F)) * 0.05).astype(ml_dtypes.float8_e5m2)
res = np.asarray(k(x, w))
xf = x.astype(np.float32); wf = w.astype(np.float32).reshape(NE, 128, HT, GU)
ref = np.einsum("pk,epkn->en", xf, wf).reshape(NE, 4, 2, Q)  # column index = c*180 + j*90 + p
got = res.reshape(REP, NE, Q, 2, 4)[0].transpose(0, 3, 2, 1)  # [e, c, j, p]
print("max abs err", float(np.abs(got - ref).max()), "ref max", float(np.abs(ref).max()))
