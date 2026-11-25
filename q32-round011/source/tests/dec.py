"""dec.py LENS N ENGINE... : for each engine (sequential, CORES env), prefill+decode on the same random prompts; prints times, checks tokens equal to first engine"""
import sys, os, random, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from compare import Eng
lens = [int(x) for x in sys.argv[1].split(",")]; N = int(sys.argv[2])
rng = random.Random(int(os.environ.get("SEED", "1")))
prompts = [[rng.randrange(151936) for _ in range(L)] for L in lens]
ref = None
for spec in sys.argv[3:]:
    e = Eng(spec); e.call(op="hello")
    e.call(op="generate", prompt=prompts[0][:300], max_tokens=4, reset=True)  # warmup
    toks = []; pt = []; dt = []
    for pr in prompts:
        a, t1 = e.call(op="generate", prompt=pr, max_tokens=1, reset=True)
        b, t2 = e.call(op="generate", prompt=[], max_tokens=N, reset=False)
        toks.append(a["tokens"] + b["tokens"]); pt.append(t1 * 1e3); dt.append(t2 * 1e3)
    e.close()
    if ref is None: ref = toks
    same = [next((i for i in range(len(x)) if x[i] != y[i]), None) for x, y in zip(toks, ref)]
    print(spec, "prefill", " ".join(f"{x:.1f}" for x in pt), "| decode", " ".join(f"{x:.1f}" for x in dt), "| diverge", same, flush=True)
