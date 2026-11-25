"""Test head (and embed, inc) on 4 ranks against numpy."""
import os, sys, time
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "kernels"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import layout as L
from layout import bf16, rb
MODEL = "/mnt/nvme1/es-ds/models/gpt-oss-20b"
ck = L.Ckpt(MODEL)
W = ck.get("lm_head.weight")
rng = np.random.default_rng(int(sys.argv[1]) if len(sys.argv) > 1 else 0)
emb = ck.get("model.embed_tokens.weight")
h = rb(emb[777].astype(np.float32) * 20 + rng.standard_normal(L.H).astype(np.float32) * 3)
if os.environ.get("HFILE"):
    import json; h = rb(np.array(json.load(open(os.environ["HFILE"]))["values"], np.float32))
tok, lg = L.ref_head(ck, h, W)
srt = np.sort(lg)[::-1]
print("ref argmax", tok, "top2", srt[:3], "ties at max:", int((lg == lg.max()).sum()), np.nonzero(lg == lg.max())[0][:5])
import nki
from nki.collectives import ReplicaGroup
import model as M
pos = 37
TOKLEN = 4616
args = {k: [] for k in ["h", "pos", "toks", "lnf", "wlm", "lmbias", "lmscore"]}
for r in range(4):
    a = dict(h=h.astype(bf16).reshape(1, -1), pos=np.array([[pos]], np.int32), toks=np.zeros((1, TOKLEN), np.int32),
             **L.head_rank_arrays(ck, r, W))
    for k in args:
        args[k].append(a[k])
t0 = time.time()
outs = M.head[2](*[args[k] for k in args], rg=ReplicaGroup([[0, 1, 2, 3]]))
print("kernel %.1fs" % (time.time() - t0))
print("pos_out", [np.asarray(o[0]).reshape(-1)[0] for o in outs], "tok", [np.asarray(o[1]).reshape(-1)[0] for o in outs])
print("toks[pos+1] per rank", [a[0, pos + 1] for a in args["toks"]], "ref", tok)
