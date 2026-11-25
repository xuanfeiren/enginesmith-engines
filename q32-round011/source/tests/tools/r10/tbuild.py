# tbuild.py WORKDIR CORES S[,S..] VAR=VAL... : build pf<S>_layer of WORKDIR's source with experiment env vars (dummy key), into WORKDIR/target/neff
import os, sys, shutil
wd, cores, Ss = sys.argv[1], sys.argv[2], [int(x) for x in sys.argv[3].split(",")]
env = dict(a.split("=", 1) for a in sys.argv[4:])
os.environ.update(env)
sys.path.insert(0, os.path.join(wd, 'kernels'))
import targets, build_neffs
ts = [t for t in targets.TARGETS if t['kind'] == 'prefill' and t['STAGE'] == 2 and t['S'] in Ss]
for i, t in enumerate(ts):
    t = dict(t, TIMINGX=env)
    r = build_neffs.build_one(t, cores, 62000 + i * 8 + (os.getpid() % 50) * 16, os.path.join(wd, 'target/neffcache'))
    shutil.copy(r[1], os.path.join(wd, f"target/neff/{t['name']}.neff")); print(r[0], round(r[2]), flush=True)
