# prefill a nonce prompt of PLEN tokens through LAYERS layers and save rti (rank 0) as npy
import os
import sys, json, subprocess
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import numpy as np, os
eng, pl, layers, out = sys.argv[1], int(sys.argv[2]), sys.argv[3], sys.argv[4]
env = dict(os.environ, ENGINE_PF_LAYERS=layers)
p = subprocess.Popen([eng, "--model", "/mnt/nvme1/es-ds/models/gpt-oss-20b"], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=open("/tmp/es-dev/dump.err", "w"), env=env)
def req(o):
    p.stdin.write((json.dumps(o) + "\n").encode()); p.stdin.flush(); return json.loads(p.stdout.readline())
req({"id": 1, "op": "hello"})
prompt = [((i * 2654435761) & 0xFFFFFFFF) % 201088 for i in range(pl)]
print(req({"id": 2, "op": "generate", "prompt": prompt, "max_tokens": 1})["ok"])
v = req({"id": 3, "op": "read", "name": "rti", "offset": 0, "n": 4608 * 8, "type": "f32"})["values"]
np.save(out, np.array(v, np.float32).reshape(4608, 8))
p.stdin.close(); p.wait()
r = np.array(v, np.float32).reshape(4608, 8)[:pl, :4].astype(int)
c = np.bincount(r.ravel(), minlength=32); print("counts", sorted(c)[::-1][:10], "min", c.min(), "blocks", int(np.ceil(c / 128).sum()))
