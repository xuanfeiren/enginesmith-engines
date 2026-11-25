import os
import sys, json, subprocess
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "brief", "skill", "scripts"))
from engine_client import Engine, load_launch, nonce_prompt, add_launch_args
import argparse
ap = argparse.ArgumentParser(); ap.add_argument("engine"); add_launch_args(ap); a = ap.parse_args()
e = Engine(a.engine, load_launch(a))
p = nonce_prompt(200, 5, 201088)
t, ms = e.generate(p, 4000); print("200+4000:", len(t), "tokens", round(ms), "ms, min/max id", min(t), max(t))
t2, ms2 = e.generate([], 407); print("continuation 407:", len(t2), round(ms2), "ms")
t3, ms3 = e.generate([], 1); print("continuation 1 (last slot):", len(t3))
try:
    e.generate([], 1); print("past the end: accepted?!")
except Exception as ex: print("past the end refused:", str(ex)[:120])
t4, _ = e.generate(nonce_prompt(4500, 9, 201088), 100); print("4500+100:", len(t4))
e.close()
