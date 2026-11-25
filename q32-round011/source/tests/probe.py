"""quick probe: python tests/probe.py ENGINE[:ENV=V] CORES LEN[,LEN] [N]: generate(P,1) only, then (P, N) separately"""
import sys, random, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from compare import Eng
e = Eng(sys.argv[1], sys.argv[2]); e.call(op="hello")
N = int(sys.argv[4]) if len(sys.argv) > 4 else 1
rng = random.Random(0)
for L in [int(x) for x in sys.argv[3].split(",")]:
    pr = [rng.randrange(151936) for _ in range(L)]
    try:
        r, t = e.call(op="generate", prompt=pr, max_tokens=N, reset=True)
        print(L, r["tokens"][:16], f"{t*1e3:.0f} ms", flush=True)
    except Exception as ex:
        print(L, "ERR", str(ex)[-300:], flush=True)
e.close()
