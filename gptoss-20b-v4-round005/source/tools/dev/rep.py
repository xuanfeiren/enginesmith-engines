# rep.py ENGINE plen n reps: the same prompt repeatedly; prints whether every run gives the same tokens
import os, sys, json
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "brief", "skill", "scripts"))
from engine_client import Engine, load_launch, nonce_prompt, add_launch_args
import argparse
ap = argparse.ArgumentParser(); ap.add_argument("engine"); ap.add_argument("plen", type=int); ap.add_argument("n", type=int); ap.add_argument("reps", type=int)
add_launch_args(ap); a = ap.parse_args()
e = Engine(a.engine, load_launch(a))
outs = []
for r in range(a.reps):
    p = nonce_prompt(a.plen, 7, 201088)
    t, ms = e.generate(p, a.n)
    outs.append(t)
e.close()
for o in outs: print(o[:40])
print("all same:", all(o == outs[0] for o in outs))
