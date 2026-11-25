"""sweep.py LENS N BASE VARIANT_SPEC... : each engine in turn on the same random prompts; survives failures"""
import sys, os, random
sys.path.insert(0, "/home/ubuntu/es-wave2/tasks/q32/runs/q32-20261106-022117/work/008/engine/tests")
from compare import Eng
lens = [int(x) for x in sys.argv[1].split(",")]; N = int(sys.argv[2])
rng = random.Random(int(os.environ.get("SEED", "7")))
prompts = [[rng.randrange(151936) for _ in range(L)] for L in lens]
ref = None
for spec in sys.argv[3:]:
    e = None
    try:
        e = Eng(spec); e.call(op="hello")
        e.call(op="generate", prompt=prompts[0][:300], max_tokens=4, reset=True)
        toks = []; pt = []; dt = []
        for pr in prompts:
            a, t1 = e.call(op="generate", prompt=pr, max_tokens=1, reset=True)
            b, t2 = e.call(op="generate", prompt=[], max_tokens=N, reset=False)
            toks.append(a["tokens"] + b["tokens"]); pt.append(t1 * 1e3); dt.append(t2 * 1e3)
        if ref is None: ref = toks
        same = [next((i for i in range(len(x)) if x[i] != y[i]), None) for x, y in zip(toks, ref)]
        print(spec.split(":", 1)[-1], "prefill", " ".join(f"{x:.1f}" for x in pt), "| decode", " ".join(f"{x:.1f}" for x in dt), "| diverge", same, flush=True)
    except Exception as x:
        print(spec.split(":", 1)[-1], "FAILED", str(x)[-200:].replace("\n", " "), flush=True)
    finally:
        try:
            e.p.kill(); e.p.wait()
        except Exception:
            pass
print("DONE", flush=True)
