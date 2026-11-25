"""runp.py NAME CORES 'module:func' args-json : compile+run a probe through nki's documented route in a fresh process,
artifacts in /tmp/q32w8/build/NAME (kernel.neff), prints the pickled result"""
import os, sys, json, pickle, subprocess, time, shutil
name, cores, target = sys.argv[1], sys.argv[2], sys.argv[3]
args = json.loads(sys.argv[4]) if len(sys.argv) > 4 else []
art = f"/tmp/q32w8/build/{name}"
shutil.rmtree(art, ignore_errors=True); os.makedirs(art)
env = {**os.environ, "NKI_ARTIFACTS_DIR": art, "NEURON_PLATFORM_TARGET_OVERRIDE": "trn2", "NEURON_RT_VISIBLE_CORES": cores,
       "NEURON_RT_ROOT_COMM_ID": f"localhost:{51000 + os.getpid() % 2000}", "PYTHONPATH": "/tmp/q32w8/probes"}
mod, fn = target.split(":")
res = os.path.join(art, "result.pkl")
code = f"import pickle, sys; import {mod} as m; r = m.{fn}(*{args!r}); pickle.dump(r, open({res!r}, 'wb'))"
t0 = time.time()
p = subprocess.run([sys.executable, "-c", code], env=env, cwd=art, capture_output=True, text=True)
print(f"{name}: rc {p.returncode} in {time.time() - t0:.0f} s")
if p.returncode:
    print(p.stdout[-3000:], p.stderr[-5000:])
else:
    print(pickle.load(open(res, "rb")))
