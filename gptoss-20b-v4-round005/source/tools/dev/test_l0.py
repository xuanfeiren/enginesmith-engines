# usage: test_l0.py ENGINE_WRAPPER S [LAYERS]: prefill S tokens through the first LAYERS layers (ENGINE_PF_LAYERS) and
# compare hT with the numpy reference (layer 0 only when LAYERS=1)
import sys, json, os, subprocess, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "kernels"))
import layout as L
from layout import rb
MODEL = "/mnt/nvme1/es-ds/models/gpt-oss-20b"; MAXPOS = 4608
eng, S = sys.argv[1], int(sys.argv[2])
rng = np.random.default_rng(1)
toks = rng.integers(0, 200000, S)
env = dict(os.environ, ENGINE_PF_LAYERS="1")
p = subprocess.Popen([eng, "--model", MODEL], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=open(os.environ.get("ES_ERR", "/dev/null"), "w"), env=env)
def req(o):
    p.stdin.write((json.dumps(o) + "\n").encode()); p.stdin.flush(); return json.loads(p.stdout.readline())
req({"id": 1, "op": "hello"})
print(req({"id": 2, "op": "generate", "prompt": [int(t) for t in toks], "max_tokens": 1})["ok"])
vals = []
for kt in range(L.H):  # row kt of hT: positions 0..S-1
    vals.append(req({"id": 3, "op": "read", "name": "hT", "offset": kt * MAXPOS, "n": S, "type": "bf16"})["values"])
p.stdin.close(); p.wait()
o = np.array(vals, np.float32).T  # [S, H]
ck = L.Ckpt(MODEL)
emb = ck.get("model.embed_tokens.weight")
X = rb(emb[toks].astype(np.float32))
cs = L.rope_table(MODEL, MAXPOS)
ref, K, V = L.ref_layer_seq(ck, 0, X, cs)
d = np.linalg.norm(o - ref, axis=1) / np.linalg.norm(ref - X, axis=1)
print("rel err of (h2) vs |h2 - x| per token: median %.4g max %.4g at %d" % (np.median(d), d.max(), d.argmax()))
if os.environ.get("ES_D"): np.save(os.environ["ES_D"], d)
print("worst 5", np.argsort(-d)[:5], np.sort(d)[-5:])
