import sys, random
sys.path.insert(0, sys.argv[1] + "/tests")
from compare import Eng
e = Eng(sys.argv[2]); e.call(op="hello")
rng = random.Random(3); p = [rng.randrange(151936) for _ in range(200)]
a, t1 = e.call(op="generate", prompt=p, max_tokens=4000, reset=True)
b, t2 = e.call(op="generate", prompt=[], max_tokens=408, reset=False)
print("got", len(a["tokens"]), len(b["tokens"]), f"{t1:.1f}s {t2:.1f}s")
try:
    c, _ = e.call(op="generate", prompt=[], max_tokens=1, reset=False); print("extra:", c)
except Exception as ex:
    print("extra refused:", str(ex)[:200])
e.close()
