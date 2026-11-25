# micro-benchmark: decode gate/up GEMVs (4 experts x 24 kt x 720 outputs, fp8 weights, bf16 x) with and without
# TensorE column tiling; output in column layout [120, 6] per expert. usage: python ct.py MODE REP
import os, sys
import numpy as np, ml_dtypes
import nki, nki.language as nl, nki.isa as nisa

MODE = sys.argv[1]; REP = int(sys.argv[2])
HT = 24; GU = 720; GH = 360; NE = 4; P = 120; GT = 6
F = HT * GU


@nki.jit
def k(x, w):
    out = nl.ndarray((REP * NE, P, GT), dtype=nl.float32, buffer=nl.shared_hbm)
    xs = nl.ndarray((128, HT), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.dma_copy(dst=xs, src=x)
    x32 = nl.ndarray((128, HT, 32), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.memset(dst=x32, value=0.0)
    for kt in range(HT):
        nisa.tensor_copy(dst=x32[0:128, kt, 0:1], src=xs[0:128, kt:kt + 1])
    sel = nl.ndarray((128, 1), dtype=nl.float32, buffer=nl.sbuf)  # 1 at partitions 0, 32, 64, 96
    nisa.memset(dst=sel, value=0.0)
    for c in range(4):
        nisa.memset(dst=sel[32 * c:32 * c + 1, 0:1], value=1.0)
    ws = []
    for e in range(NE):
        t = nl.ndarray((128, F), dtype=nl.float8_e5m2, buffer=nl.sbuf)
        nisa.dma_copy(dst=t, src=w[e])
        ws.append(t)
    for r in range(REP):
        for e in range(NE):
            gs = ws[e]
            acc = []
            for hh in range(2):
                acc.append(nl.ndarray((128, 512), dtype=nl.float32, buffer=nl.psum))
            col = nl.ndarray((P, GT), dtype=nl.float32, buffer=nl.psum)
            if MODE == "a":
                for kt in range(HT):
                    for hh in range(2):
                        nisa.nc_matmul(dst=acc[hh][0:1, 0:GH], stationary=xs[0:128, kt:kt + 1],
                                       moving=gs[0:128, kt * GU + hh * GH:kt * GU + (hh + 1) * GH])
                row = nl.ndarray((1, GU), dtype=nl.float32, buffer=nl.sbuf)
                for hh in range(2):
                    nisa.tensor_copy(dst=row[0:1, hh * GH:(hh + 1) * GH], src=acc[hh][0:1, 0:GH])
                for t in range(GT):
                    nisa.nc_transpose(dst=col[0:P, t:t + 1], data=row[0:1, t * P:(t + 1) * P])
            else:
                for kt in range(HT):
                    c = kt % 4
                    for hh in range(2):
                        if MODE == "b":
                            nisa.nc_matmul(dst=acc[hh][32 * c:32 * c + 32, 0:GH], stationary=x32[0:128, kt, 0:32],
                                           moving=gs[0:128, kt * GU + hh * GH:kt * GU + (hh + 1) * GH],
                                           tile_position=(0, 32 * c), tile_size=(128, 32))
                        else:
                            nisa.nc_matmul(dst=acc[hh][32 * c:32 * c + 1, 0:GH], stationary=xs[0:128, kt:kt + 1],
                                           moving=gs[0:128, kt * GU + hh * GH:kt * GU + (hh + 1) * GH],
                                           tile_position=(0, 32 * c), tile_size=(128, 32))
                r4 = nl.ndarray((128, GU), dtype=nl.float32, buffer=nl.sbuf)
                if MODE == "b":
                    for hh in range(2):
                        nisa.tensor_copy(dst=r4[0:128, hh * GH:(hh + 1) * GH], src=acc[hh][0:128, 0:GH])
                else:
                    nisa.memset(dst=r4, value=0.0)
                    for c in range(4):
                        for hh in range(2):
                            nisa.tensor_copy(dst=r4[32 * c:32 * c + 1, hh * GH:(hh + 1) * GH], src=acc[hh][32 * c:32 * c + 1, 0:GH])
                for t in range(GT):
                    nisa.nc_matmul(dst=col[0:P, t:t + 1], stationary=r4[0:128, t * P:(t + 1) * P], moving=sel)
            cs = nl.ndarray((P, GT), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_copy(dst=cs, src=col)
            nisa.dma_copy(dst=out[r * NE + e], src=cs)
    return out


rng = np.random.default_rng(0)
x = rng.standard_normal((128, HT)).astype(ml_dtypes.bfloat16)
w = (rng.standard_normal((NE, 128, F)) * 0.05).astype(ml_dtypes.float8_e5m2)
res = np.asarray(k(x, w))
xf = x.astype(np.float32); wf = w.astype(np.float32).reshape(NE, 128, HT, GU)
ref = np.einsum("pk,epkn->en", xf, wf)  # [e, n], n = t * 120 + p
got = res.reshape(REP, NE, P, GT)[0].transpose(0, 2, 1).reshape(NE, GU)
print("max abs err", float(np.abs(got - ref).max()), "ref max", float(np.abs(ref).max()), "rel", float(np.abs(got - ref).max() / np.abs(ref).max()))
