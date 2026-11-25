# make per-rank input npy files for a step NEFF: small tensors real-ish, big ones random small bf16 (shared file)
import sys, os, numpy as np, ml_dtypes
sys.path.insert(0, sys.argv[1] + "/kernels")
import targets as T
out = sys.argv[2]; pos = int(sys.argv[3]); os.makedirs(out, exist_ok=True)
rng = np.random.default_rng(0)
DT = {"bf16": ml_dtypes.bfloat16, "f32": np.float32, "i32": np.int32}
lines = [[] for _ in range(T.W)]
for n in T.STEP_ORDER:
    shp, dt = T.STEP_SPEC[n]
    f = os.path.join(out, n + ".npy")
    if n == "pos": a = np.full(shp, pos, np.int32)
    elif n == "plen": a = np.full(shp, 1, np.int32)
    elif n in ("prompt", "toks"): a = np.full(shp, 1000, np.int32)
    elif n == "rkb": a = None
    elif n in ("ln1", "ln2", "qn", "kn", "normf"): a = np.ones(shp, np.float32)
    elif n in ("cos", "sin"): a = rng.uniform(-1, 1, shp).astype(np.float32)
    else:
        a = (rng.standard_normal(shp, dtype=np.float32) * 0.02).astype(DT[dt])
    if a is not None and not os.path.exists(f): np.save(f, a)
    for r in range(T.W):
        if n == "rkb":
            fr = os.path.join(out, f"rkb{r}.npy"); np.save(fr, np.full(shp, r * 2 * T.NVL * 128, np.float32)); lines[r] += [n, fr]
        else: lines[r] += [n, f]
open(os.path.join(out, "multi.txt"), "w").write("\n".join(" ".join(l) for l in lines) + "\n")
