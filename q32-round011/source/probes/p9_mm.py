"""probe (round 009): does reusing a stationary tile over 2 moving blocks save the LDWEIGHTS time?"""
import numpy as np, nki, nki.isa as nisa, nki.language as nl, ml_dtypes

NT, KT, REP = 6, 40, 8

@nki.jit
def k(w, x, MODE):
    out = nl.ndarray((128, 1024), dtype=nl.float32, buffer=nl.shared_hbm)
    ws = nl.ndarray((128, NT * KT * 128), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.dma_copy(dst=ws, src=w)
    xs = nl.ndarray((128, KT, 1024), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.dma_copy(dst=xs, src=x)
    acc = nl.ndarray((128, 1024), dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(dst=acc, value=0.0)
    for r in range(REP):
        for nt in range(NT):
            if MODE == 0 or MODE >= 2:
                NB = 512 if MODE == 0 else (256 if MODE == 2 else 128)
                for b in range(1024 // NB):
                    ps = nl.ndarray((128, NB), dtype=nl.float32, buffer=nl.psum)
                    for kt in range(KT):
                        c = (nt * KT + kt) * 128
                        nisa.nc_matmul(dst=ps, stationary=ws[:, c:c + 128], moving=xs[:, kt, b * NB:(b + 1) * NB])
                    nisa.tensor_tensor(dst=acc[:, b * NB:(b + 1) * NB], data1=acc[:, b * NB:(b + 1) * NB], data2=ps, op=nl.add)
            elif MODE == 1:
                p0 = nl.ndarray((128, 512), dtype=nl.float32, buffer=nl.psum)
                p1 = nl.ndarray((128, 512), dtype=nl.float32, buffer=nl.psum)
                for kt in range(KT):
                    c = (nt * KT + kt) * 128
                    nisa.nc_matmul(dst=p0, stationary=ws[:, c:c + 128], moving=xs[:, kt, 0:512])
                    nisa.nc_matmul(dst=p1, stationary=ws[:, c:c + 128], moving=xs[:, kt, 512:1024])
                nisa.tensor_tensor(dst=acc[:, 0:512], data1=acc[:, 0:512], data2=p0, op=nl.add)
                nisa.tensor_tensor(dst=acc[:, 512:1024], data1=acc[:, 512:1024], data2=p1, op=nl.add)
    nisa.dma_copy(dst=out, src=acc)
    return out


@nki.jit
def k2(w, x, MODE):
    # MODE 4: column tiling, two M=64 halves in parallel; MODE 5: same halves without tiling; MODE 6: row tiling K=64 x2
    out = nl.ndarray((128, 1024), dtype=nl.float32, buffer=nl.shared_hbm)
    ws = nl.ndarray((128, NT * KT * 128), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.dma_copy(dst=ws, src=w)
    xs = nl.ndarray((128, KT, 1024), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.dma_copy(dst=xs, src=x)
    acc = nl.ndarray((128, 1024), dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(dst=acc, value=0.0)
    for r in range(REP):
        for nt in range(NT):
            for b in range(2):
                if MODE == 6:
                    pa = nl.ndarray((128, 512), dtype=nl.float32, buffer=nl.psum)
                    pb = nl.ndarray((128, 512), dtype=nl.float32, buffer=nl.psum)
                    for kt in range(KT):
                        c = (nt * KT + kt) * 128
                        nisa.nc_matmul(dst=pa, stationary=ws[0:64, c:c + 128], moving=xs[0:64, kt, b * 512:(b + 1) * 512], tile_position=(0, 0), tile_size=(64, 128))
                        nisa.nc_matmul(dst=pb, stationary=ws[64:128, c:c + 128], moving=xs[64:128, kt, b * 512:(b + 1) * 512], tile_position=(64, 0), tile_size=(64, 128))
                    nisa.tensor_tensor(dst=acc[:, b * 512:(b + 1) * 512], data1=acc[:, b * 512:(b + 1) * 512], data2=pa, op=nl.add)
                    nisa.tensor_tensor(dst=acc[:, b * 512:(b + 1) * 512], data1=acc[:, b * 512:(b + 1) * 512], data2=pb, op=nl.add)
                else:
                    pa = nl.ndarray((64, 512), dtype=nl.float32, buffer=nl.psum)
                    pb = nl.ndarray((64, 512), dtype=nl.float32, buffer=nl.psum)
                    for kt in range(KT):
                        c = (nt * KT + kt) * 128
                        if MODE == 4:
                            nisa.nc_matmul(dst=pa, stationary=ws[:, c:c + 64], moving=xs[:, kt, b * 512:(b + 1) * 512], tile_position=(0, 0), tile_size=(128, 64))
                            nisa.nc_matmul(dst=pb, stationary=ws[:, c + 64:c + 128], moving=xs[:, kt, b * 512:(b + 1) * 512], tile_position=(0, 64), tile_size=(128, 64))
                        else:
                            nisa.nc_matmul(dst=pa, stationary=ws[:, c:c + 64], moving=xs[:, kt, b * 512:(b + 1) * 512])
                            nisa.nc_matmul(dst=pb, stationary=ws[:, c + 64:c + 128], moving=xs[:, kt, b * 512:(b + 1) * 512])
                    nisa.tensor_tensor(dst=acc[0:64, b * 512:(b + 1) * 512], data1=acc[0:64, b * 512:(b + 1) * 512], data2=pa, op=nl.add)
                    nisa.tensor_tensor(dst=acc[64:128, b * 512:(b + 1) * 512], data1=acc[64:128, b * 512:(b + 1) * 512], data2=pb, op=nl.add)
    nisa.dma_copy(dst=out, src=acc)
    return out


def run(MODE):
    rng = np.random.default_rng(1)
    w = (rng.standard_normal((128, NT * KT * 128), dtype=np.float32) * 0.05).astype(ml_dtypes.bfloat16)
    x = rng.standard_normal((128, KT, 1024), dtype=np.float32).astype(ml_dtypes.bfloat16)
    o = np.asarray(k(w, x, MODE) if MODE < 4 else k2(w, x, MODE))
    return float(np.abs(o).sum())
