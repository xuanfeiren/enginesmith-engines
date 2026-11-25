"""pf.py LENS ENGINE...: prefill-only timing (generate(P, 1)), 3 reps per length"""
import sys, os, random
sys.path.insert(0, "/home/ubuntu/es-wave2/tasks/q32/runs/q32-20261106-022117/work/008/engine/tests")
from compare import Eng
lens = [int(x) for x in sys.argv[1].split(",")]
rng = random.Random(7)
for spec in sys.argv[2:]:
    e = Eng(spec); e.call(op="hello")
    out = []
    for L in lens:
        ts = []
        for _ in range(3):
            pr = [rng.randrange(151936) for _ in range(L)]
            a, t = e.call(op="generate", prompt=pr, max_tokens=1, reset=True); ts.append(t * 1e3)
        out.append(f"{L}: " + " ".join(f"{x:.1f}" for x in ts))
    e.p.kill(); e.p.wait()
    print(spec, " | ".join(out), flush=True)
