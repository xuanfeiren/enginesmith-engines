# debug the layer-0 prefill MoE: compare xtm, rti, rec and hT with a numpy reference
import sys, json, os, subprocess, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import layout as L
from layout import rb
MODEL = "/mnt/nvme1/es-ds/models/gpt-oss-20b"; MAXPOS = 4608; H = 2880; E = 32
eng, S = sys.argv[1], int(sys.argv[2])
rng = np.random.default_rng(1)
toks = rng.integers(0, 200000, S)
env = dict(os.environ, ENGINE_PF_LAYERS="1")
p = subprocess.Popen([eng, "--model", MODEL], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=open("/tmp/es-dev/test_l0.err", "w"), env=env)
def req(o):
    p.stdin.write((json.dumps(o) + "\n").encode()); p.stdin.flush(); return json.loads(p.stdout.readline())
def rd(name, off, n, ty, rank=0):
    out = []
    CH = 1 << 20
    for o in range(off, off + n, CH):
        out += req({"id": 3, "op": "read", "name": name, "offset": o, "n": min(CH, off + n - o), "type": ty, "rank": rank})["values"]
    return np.array(out, np.float32)
req({"id": 1, "op": "hello"})
print(req({"id": 2, "op": "generate", "prompt": [int(t) for t in toks], "max_tokens": 1})["ok"])
xtm = rd("xtm", 0, S * H, "bf16").reshape(S, H)
rti = rd("rti", 0, S * 8, "f32").reshape(S, 8)
rec = rd("rec", 0, E * MAXPOS * 4, "f32").reshape(E, MAXPOS, 4)
hT = rd("hT", 0, H * MAXPOS, "bf16").reshape(H, MAXPOS)[:, :S].T
yb = rd("ybuf", 0, 4 * MAXPOS * H, "f32").reshape(4, MAXPOS, H)[:, :S]
p.stdin.close(); p.wait()
ck = L.Ckpt(MODEL); pre = "model.layers.0."
emb = ck.get("model.embed_tokens.weight")
X = rb(emb[toks].astype(np.float32))
cs = L.rope_table(MODEL, MAXPOS)
ref, K, V = L.ref_layer_seq(ck, 0, X, cs)
# reference intermediates: h1 from the kernel's xtm is enough to check routing
def norm(Xm, w):
    v = (Xm * Xm).mean(1, keepdims=True); return rb(w[None, :] * (Xm / np.sqrt(v + 1e-5)))
lg = rb(xtm @ ck.f32(pre + "mlp.router.weight").T + ck.f32(pre + "mlp.router.bias"))
sel = np.array([sorted(range(E), key=lambda j: (-lg[t, j], j))[:4] for t in range(S)])
print("sel agree (engine vs ref from engine xtm):", (rti[:, :4] == sel).mean())
cnt = np.array([(sel == e).sum() for e in range(E)])
print("counts ref", cnt[:8], "...")
for e in range(3):
    rows = rec[e, :cnt[e] + 2]
    exp_t = np.nonzero((sel == e).any(1))[0]
    print("expert", e, "rec tokens ok:", np.array_equal(rows[:cnt[e], 0], exp_t), "next rows", rows[cnt[e]:cnt[e] + 2])
    slot = np.array([np.nonzero(sel[t] == e)[0][0] for t in exp_t])
    print("   ydst ok:", np.array_equal(rows[:cnt[e], 1], slot * MAXPOS + exp_t), " w ok:", np.allclose(rows[:cnt[e], 2], rti[exp_t, 4 + slot]))
# ybuf vs expert outputs (rank 0 partial: vcores 0,1 sum not available; check full-sum after all-reduce instead)
d = np.linalg.norm(hT - ref, axis=1) / np.linalg.norm(ref - X, axis=1)
print("h2 rel err median %.4g max %.4g" % (np.median(d), d.max()))
print("ybuf slot norms", [float(np.linalg.norm(yb[i])) for i in range(4)])
np.save("/tmp/es-dev/dbg.npy", dict(xtm=xtm, rti=rti, hT=hT, ref=ref, X=X, sel=sel, yb=yb), allow_pickle=True)
