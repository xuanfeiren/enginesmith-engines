# usage: drive.py ENGINE_WRAPPER plen ndecode   (runs hello, generate(P,1), generate((),n)); prints times
import os
import sys, json
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "brief", "skill", "scripts"))
from engine_client import Engine, load_launch, nonce_prompt, add_launch_args
import argparse
ap = argparse.ArgumentParser(); ap.add_argument("engine"); ap.add_argument("plen", type=int); ap.add_argument("n", type=int)
ap.add_argument("--reps", type=int, default=1)
add_launch_args(ap); a = ap.parse_args()
launch = load_launch(a)
e = Engine(a.engine, launch)
for r in range(a.reps):
    p = nonce_prompt(a.plen, 1000 * r + 7, 201088)
    t, ms = e.generate(p, 1)
    t2, ms2 = e.generate([], a.n)
    print(json.dumps({"prefill_ms": ms, "decode_ms": ms2, "tokens": t + t2}), flush=True)
e.close()
print("stderr:", e.stderr_path)
