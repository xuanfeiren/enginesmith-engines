"""Test decode_layer on 4 ranks (NKI's documented run-once route) against the numpy reference.
usage: NKI_ARTIFACTS_DIR=... python test_layer.py LAYER POS B"""
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "kernels"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import layout as L  # noqa: E402
from layout import bf16, rb  # noqa: E402

MODEL = "/mnt/nvme1/es-ds/models/gpt-oss-20b"
MAXPOS = 4608
l, pos, B = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
ck = L.Ckpt(MODEL)
sliding = ck.cfg["layer_types"][l] == "sliding_attention"
if sliding:
    B = 128
T = MAXPOS
rng = np.random.default_rng(0)
emb = ck.get("model.embed_tokens.weight")
h = rb(emb[12345].astype(np.float32) * 1.0 + rng.standard_normal(L.H).astype(np.float32) * 0.5)
kcache = rb(rng.standard_normal((8, pos, 64)) * 1.0)
vcache = rb(rng.standard_normal((8, pos, 64)) * 1.0)
cs = L.rope_table(MODEL, MAXPOS)
t0 = time.time()
ref, kn, vn, info = L.ref_layer(ck, l, h, pos, kcache, vcache, cs)
print("ref done %.1fs order" % (time.time() - t0), info["order"], info["w"])

import nki  # noqa: E402
from nki.collectives import ReplicaGroup  # noqa: E402
import model as M  # noqa: E402

args = {k: [] for k in ["h", "pos", "ln1", "wqkv", "bqkv", "cs", "rmat", "sinks", "kc", "vc", "kcw", "vcw", "wo2", "bo", "ln2", "wr",
                        "br", "wgu", "bgu", "wd", "bd"]}
for r in range(4):
    w = L.layer_rank_arrays(ck, l, r)
    # the decode kernels read 128-partition copies (rows 120..127 zero) of wqkv / wgu / wd (src/engine.cpp)
    for k_, F_ in (("wgu", 6 * L.HT * L.P), ("wd", 3 * L.HT * L.P)):
        v = w[k_].reshape(L.E, 2, L.P, F_)
        w[k_] = np.concatenate([v, np.zeros((L.E, 2, 128 - L.P, F_), v.dtype)], 2).reshape(L.E, -1)
    w["wqkv"] = np.concatenate([w["wqkv"], np.zeros((2, 128 - L.P, w["wqkv"].shape[2]), w["wqkv"].dtype)], 1)
    kc = np.zeros((2, 64, T), bf16)
    vc = np.zeros((2, T, 64), bf16)
    for c in range(2):
        g = 2 * r + c
        for t in range(pos):
            kc[c, :, t] = kcache[g, t]
            vc[c, t, :] = vcache[g, t]
    a = dict(h=h.astype(bf16).reshape(1, -1), pos=np.array([[pos]], np.int32), cs=cs, rmat=L.rmat(), kc=kc[None], vc=vc[None], kcw=kc[None].copy(), vcw=vc[None].copy(), **w)
    for k in args:
        args[k].append(a[k])
t0 = time.time()
outs = M.decode_layer[2](*[args[k] for k in args], rg=ReplicaGroup([[0, 1, 2, 3]]), sliding=sliding, B=B)
print("kernel %.1fs" % (time.time() - t0))
for r in range(4):
    o = np.asarray(outs[r]).astype(np.float32).reshape(-1)
    err = np.abs(o - ref)
    print(f"rank {r}: max err {err.max():.4g} mean err {err.mean():.4g} |ref| mean {np.abs(ref).mean():.4g} "
          f"rel {np.linalg.norm(o - ref) / np.linalg.norm(ref):.3g}")
h1 = info["h1"]
print("h1 vs h2 delta (scale of moe+attn):", np.linalg.norm(ref - h), np.linalg.norm(h1 - h))
