# usage: bench.py ENGINE PLEN what[:layer[:chunk]] ... : prefill a PLEN prompt, then time n executions of each NEFF
import sys, json, subprocess
MODEL = "/mnt/nvme1/es-ds/models/gpt-oss-20b"
p = subprocess.Popen([sys.argv[1], "--model", MODEL], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=open("/tmp/es-dev/bench.err", "w"))
def req(o):
    p.stdin.write((json.dumps(o) + "\n").encode()); p.stdin.flush(); return json.loads(p.stdout.readline())
req({"id": 1, "op": "hello"})
pl = int(sys.argv[2])
prompt = [((i * 2654435761) & 0xFFFFFFFF) % 201088 for i in range(pl)]
for r in range(2):
    print("generate", req({"id": 2, "op": "generate", "prompt": prompt, "max_tokens": 1}).get("ok"))
for w in sys.argv[3:]:
    f = w.split(":"); o = {"id": 3, "op": "bench", "what": f[0], "n": 20}
    if len(f) > 1: o["layer"] = int(f[1])
    if len(f) > 2: o["chunk"] = int(f[2])
    print(w, req(o))
p.stdin.close(); p.wait()
