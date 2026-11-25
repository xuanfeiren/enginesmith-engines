"""Test prefill_layer (chunk at p0) on 4 ranks against ref_layer_seq. usage: test_prefill.py LAYER P0 KB"""
import os, sys, time
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "kernels"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import layout as L
from layout import bf16, rb
MODEL = "/mnt/nvme1/es-ds/models/gpt-oss-20b"
MAXPOS = 4608
l, p0, KB = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
SC = 256
S = p0 + SC
ck = L.Ckpt(MODEL)
sliding = ck.cfg["layer_types"][l] == "sliding_attention"
rng = np.random.default_rng(1)
emb = ck.get("model.embed_tokens.weight")
X = rb(emb[rng.integers(0, 200000, S)].astype(np.float32))
cs = L.rope_table(MODEL, MAXPOS)
t0 = time.time()
ref, K, V = L.ref_layer_seq(ck, l, X, cs)
print("ref %.1fs" % (time.time() - t0))
import nki
from nki.collectives import ReplicaGroup
import prefill as PF
names = ["hT", "pos", "ln1", "wqkv", "bqkv", "csT", "rmat", "sinks", "kc", "vc", "wo", "bo", "ln2", "wr", "br",
         "wgu", "bgu", "wd", "bd", "xch"]
args = {k: [] for k in names}
hT = np.zeros((L.H, MAXPOS), bf16)
hT[:, :S] = X.T.astype(bf16)
for r in range(4):
    w = L.layer_rank_arrays(ck, l, r)
    kc = np.zeros((2, 64, MAXPOS), bf16); vc = np.zeros((2, MAXPOS, 64), bf16)
    for c in range(2):
        g = 2 * r + c
        kc[c, :, :p0] = K[g, :p0].T; vc[c, :p0, :] = V[g, :p0]
    a = dict(hT=hT.copy(), pos=np.array([[p0]], np.int32), csT=np.ascontiguousarray(cs.T), rmat=L.rmat(), kc=kc, vc=vc,
             xch=np.zeros((2, 2, L.P, L.HT * SC), np.float32), **w)
    for k in names:
        args[k].append(a[k])
t0 = time.time()
outs = PF.prefill_layer[2](*[args[k] for k in names], rg=ReplicaGroup([[0, 1, 2, 3]]), sliding=sliding, KB=KB, dbg=True)
print("kernel %.1fs" % (time.time() - t0), type(outs))

for r in range(4):
    _, o, ok = outs[r]
    o = np.asarray(o).astype(np.float32).reshape(L.P, L.HT, SC).transpose(1, 0, 2).reshape(L.H, SC).T  # [SC, H]
    rr = ref[p0:p0 + SC]
    err = np.abs(o - rr)
    kk = np.asarray(ok).astype(np.float32).T
    print(f"rank {r}: max err {err.max():.4g} mean {err.mean():.4g} rel {np.linalg.norm(o - rr) / np.linalg.norm(rr):.3g}"
          f" | k rel {np.linalg.norm(kk - K[2 * r, p0:p0 + SC]) / np.linalg.norm(K[2 * r, p0:p0 + SC]):.3g}")
    worst = np.argsort(-np.linalg.norm(o - rr, axis=1) / np.linalg.norm(rr, axis=1))[:3]
    print("   worst tokens", worst, (np.linalg.norm(o - rr, axis=1) / np.linalg.norm(rr, axis=1))[worst])
