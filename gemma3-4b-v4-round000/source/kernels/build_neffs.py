"""Build time only: compile every NKI kernel variant to a NEFF through NKI's documented route.

For each variant a fresh process sets NKI_ARTIFACTS_DIR before importing nki, then calls the @nki.jit kernel with
numpy inputs of the right shapes (zeros): nki compiles it (neuronx-cc), leaves kernel.neff in that directory and
then tries to run it once. The run is not needed for the build; when no NeuronCore is free it fails after the NEFF
was written, which is fine. NEFFs are cached under the output directory, keyed by the kernel sources and the name.

usage: build_neffs.py OUT_DIR [--jobs N] [names...]
"""
import concurrent.futures as cf
import hashlib
import os
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SOURCES = ["gemma.py", "pre2s.py", "pre3.py", "variants.py", "build_neffs.py"]


def source_key():
    h = hashlib.sha256()
    for f in SOURCES:
        with open(os.path.join(HERE, f), "rb") as fh:
            h.update(fh.read())
    return h.hexdigest()[:16]


def worker(name, art):
    """Runs in a fresh process with NKI_ARTIFACTS_DIR=art."""
    sys.path.insert(0, HERE)
    import variants
    args, shapes, call = variants.all_variants()[name]
    mod = os.path.join(art, f"wrap_{name}.py")
    with open(mod, "w") as f:
        f.write("import nki\nfrom gemma import *\nfrom pre2s import *\nfrom pre3 import *\n\n\n@nki.jit\n"
                f"def k_{name}({', '.join(args)}):\n    return {call}\n")
    sys.path.insert(0, art)
    import importlib
    m = importlib.import_module(f"wrap_{name}")
    inputs = variants.zeros_for(shapes)
    try:
        getattr(m, f"k_{name}")[2](**{a: inputs[a] for a in args})
    except Exception as e:  # the run after compilation may fail (no free core); the NEFF is what we need
        if not os.path.exists(os.path.join(art, "kernel.neff")):
            raise
        print(f"[{name}] run skipped/failed after compile: {type(e).__name__}", file=sys.stderr)


def build_one(name, out, key):
    dst = os.path.join(out, f"{name}.neff")
    stamp = os.path.join(out, f"{name}.key")
    if os.path.exists(dst) and os.path.exists(stamp) and open(stamp).read() == key:
        return name, "cached"
    art = os.path.abspath(os.path.join(out, "build", name))
    shutil.rmtree(art, ignore_errors=True)
    os.makedirs(art)
    env = dict(os.environ)
    env["NKI_ARTIFACTS_DIR"] = art
    env["NEURON_PLATFORM_TARGET_OVERRIDE"] = "trn2"
    log = open(os.path.join(art, "build.log"), "w")
    r = subprocess.run([sys.executable, os.path.abspath(__file__), "--worker", name, art], env=env, stdout=log,
                       stderr=subprocess.STDOUT)
    neff = os.path.join(art, "kernel.neff")
    if not os.path.exists(neff):
        raise RuntimeError(f"{name}: no NEFF (exit {r.returncode}), see {art}/build.log")
    shutil.copy(neff, dst)
    with open(stamp, "w") as f:
        f.write(key)
    return name, "built"


def main():
    if sys.argv[1] == "--worker":
        worker(sys.argv[2], sys.argv[3])
        return
    out = os.path.abspath(sys.argv[1])
    rest = sys.argv[2:]
    jobs = 8
    if rest[:1] == ["--jobs"]:
        jobs = int(rest[1])
        rest = rest[2:]
    sys.path.insert(0, HERE)
    import variants
    names = rest or variants.used_variants()
    os.makedirs(out, exist_ok=True)
    key = source_key()
    with cf.ThreadPoolExecutor(jobs) as ex:
        futs = [ex.submit(build_one, n, out, key) for n in names]
        failed = 0
        for f in cf.as_completed(futs):
            try:
                n, st = f.result()
                print(f"  {n}: {st}", flush=True)
            except Exception as e:
                failed += 1
                print(f"  FAILED {e}", flush=True)
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
