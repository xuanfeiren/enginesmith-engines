import sys, json, os, subprocess, numpy as np
eng, S = sys.argv[1], int(sys.argv[2])
env = dict(os.environ, ENGINE_PF_LAYERS="1", ENGINE_NO_WARMUP="1")
p = subprocess.Popen([eng, "--model", "/mnt/nvme1/es-ds/models/gpt-oss-20b"], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=open("/tmp/es-dev/dbg.err", "w"), env=env)
def req(o):
    p.stdin.write((json.dumps(o) + "\n").encode()); p.stdin.flush(); return json.loads(p.stdout.readline())
def rd(name, off, n, ty):
    return np.array(req({"id": 3, "op": "read", "name": name, "offset": off, "n": n, "type": ty})["values"], np.float32)
req({"id": 1, "op": "hello"})
rng = np.random.default_rng(1); toks = rng.integers(0, 200000, S)
print(req({"id": 2, "op": "generate", "prompt": [int(t) for t in toks], "max_tokens": 1})["ok"])
kc = rd("L0_kc", 0, 2 * 64 * 4608, "bf16").reshape(2, 64, 4608)[:, :, :S]
vc = rd("L0_vc", 0, 2 * 4608 * 64, "bf16").reshape(2, 4608, 64)[:, :S]
xtm = rd("xtm", 0, S * 2880, "bf16").reshape(S, 2880)
rti = rd("rti", 0, S * 8, "f32").reshape(S, 8)
print("kc nan", np.isnan(kc).sum(), "vc nan", np.isnan(vc).sum(), "xtm nan rows", np.isnan(xtm).any(1).sum(), "rti nan", np.isnan(rti).sum())
print("kc absmax", np.nanmax(np.abs(kc)), "vc absmax", np.nanmax(np.abs(vc)))
r = np.nonzero(np.isnan(xtm).any(1))[0]; print("nan rows", r[:20]); print("nan cols in row", np.isnan(xtm[r[0]]).sum() if len(r) else 0)
p.stdin.close(); p.wait()
