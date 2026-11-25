# timing_build.py WORKDIR ENVVAR S[,S..] : build pf<S>_layer with a timing-only env switch under a dummy key, into WORKDIR/target/neff
import os, sys, shutil
wd, var, Ss = sys.argv[1], sys.argv[2], [int(x) for x in sys.argv[3].split(",")]
sys.path.insert(0, os.path.join(wd, 'kernels'))
import targets, build_neffs
os.environ[var] = '1'
ts = [t for t in targets.TARGETS if t['kind'] == 'prefill' and t['STAGE'] == 2 and t['S'] in Ss]
from concurrent.futures import ThreadPoolExecutor
def go(a):
    i, t = a
    t = dict(t, TIMINGX=var)
    r = build_neffs.build_one(t, ['0-7', '8-15'][i % 2], 62000 + i * 8 + (os.getpid() % 50) * 16, os.path.join(wd, 'target/neffcache'))
    shutil.copy(r[1], os.path.join(wd, f"target/neff/{t['name']}.neff")); print(r[0], round(r[2]), flush=True)
with ThreadPoolExecutor(2) as ex: list(ex.map(go, enumerate(ts)))
