"""pf.py LENS N ENGINE... : prefill-only generate(P,1) on the same random prompts (N per length), median ms + tokens"""
import sys, os, random, statistics
sys.path.insert(0, "/home/ubuntu/es-wave2/tasks/q32/runs/q32-20261106-022117/work/010/engine/tests")
from compare import Eng
lens = [int(x) for x in sys.argv[1].split(",")]; N = int(sys.argv[2])
rng = random.Random(int(os.environ.get("SEED", "7")))
prompts = {L: [[rng.randrange(151936) for _ in range(L)] for _ in range(N)] for L in lens}
ref = None
for spec in sys.argv[3:]:
    e = Eng(spec); e.call(op="hello")
    e.call(op="generate", prompt=prompts[lens[0]][0], max_tokens=2, reset=True)
    res = []; toks = []
    for L in lens:
        ts = []
        for pr in prompts[L]:
            a, t = e.call(op="generate", prompt=pr, max_tokens=1, reset=True); ts.append(t * 1e3); toks.append(a["tokens"][0])
        res.append(f"p{L} med {statistics.median(ts):.2f} [{' '.join(f'{x:.1f}' for x in ts)}]")
    e.p.kill(); e.p.wait()
    if ref is None: ref = toks
    print(spec, ' | '.join(res), '| same', sum(a == b for a, b in zip(toks, ref)), '/', len(ref), flush=True)
