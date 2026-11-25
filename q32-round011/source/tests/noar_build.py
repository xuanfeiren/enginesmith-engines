# timing-only build: prefill layer NEFFs with PF_NOAR=1 under a dummy key (NOARX), copied into target/neff; relink by hand.
import os, sys, shutil
sys.path.insert(0, 'kernels')
import targets, build_neffs
os.environ['PF_NOAR'] = '1'
ts = [t for t in targets.TARGETS if t['kind'] == 'prefill' and t['STAGE'] == 2 and t['S'] in (256, 1024, 4096)]
from concurrent.futures import ThreadPoolExecutor
def go(a):
    i, t = a
    t = dict(t, NOARX=1)
    r = build_neffs.build_one(t, ['0-7', '8-15'][i % 2], 62000 + i * 8, os.path.abspath('target/neffcache'))
    shutil.copy(r[1], f"target/neff/{t['name']}.neff"); print(r[0], r[2], flush=True)
with ThreadPoolExecutor(2) as ex: list(ex.map(go, enumerate(ts)))
