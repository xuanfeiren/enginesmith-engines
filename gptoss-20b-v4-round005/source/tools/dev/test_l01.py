# usage: test_l01.py ENGINE S: prefill S tokens through layers 0-1, compare hT with the numpy reference chain
import sys, json, os, subprocess, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import layout as L
from layout import rb
MODEL = "/mnt/nvme1/es-ds/models/gpt-oss-20b"; MAXPOS = 4608
eng, S = sys.argv[1], int(sys.argv[2])
rng = np.random.default_rng(1); toks = rng.integers(0, 200000, S)
env = dict(os.environ, ENGINE_PF_LAYERS="2", ENGINE_NO_WARMUP="1")
p = subprocess.Popen([eng, "--model", MODEL], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=open("/tmp/es-dev/test_l01.err", "w"), env=env)
def req(o):
    p.stdin.write((json.dumps(o) + "\n").encode()); p.stdin.flush(); return json.loads(p.stdout.readline())
req({"id": 1, "op": "hello"})
print(req({"id": 2, "op": "generate", "prompt": [int(t) for t in toks], "max_tokens": 1})["ok"])
vals = []
for kt in range(L.H):
    vals.append(req({"id": 3, "op": "read", "name": "hT", "offset": kt * MAXPOS, "n": S, "type": "bf16"})["values"])
p.stdin.close(); p.wait()
o = np.array(vals, np.float32).T
ck = L.Ckpt(MODEL); emb = ck.get("model.embed_tokens.weight")
X = rb(emb[toks].astype(np.float32)); cs = L.rope_table(MODEL, MAXPOS)
r0, _, _ = L.ref_layer_seq(ck, 0, X, cs)
r1, _, _ = L.ref_layer_seq(ck, 1, r0, cs)
d = np.linalg.norm(o - r1, axis=1) / np.linalg.norm(r1 - r0, axis=1)
print("layer-1 rel err per token: median %.4g p99 %.4g max %.4g" % (np.median(d), np.percentile(d, 99), d.max()))
