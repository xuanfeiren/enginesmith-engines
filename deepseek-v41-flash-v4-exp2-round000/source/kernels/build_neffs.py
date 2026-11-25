"""Build time: compile every NEFF of specs.NEFFS through NKI's documented route (NKI_ARTIFACTS_DIR set before
importing nki, then calling the @nki.jit kernel once with numpy arrays on a free NeuronCore).

Usage: python build_neffs.py OUT_DIR CACHE_DIR [names...]
Each kernel is built in its own process (own artifacts directory, own NeuronCore from NEFF_BUILD_CORES).
NEFFs are cached in CACHE_DIR keyed by the hash of the kernel sources and the argument specs.
"""
import hashlib
import os
import shutil
import subprocess
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))


def src_hash(name):
    import specs
    h = hashlib.sha256()
    for f in sorted(os.listdir(HERE)):
        if f.endswith('.py') and f != 'build_neffs.py':
            h.update(f.encode())
            h.update(open(os.path.join(HERE, f), 'rb').read())
    mod, fn, args = specs.NEFFS[name]
    h.update(repr((name, mod, fn, [(a, s, str(d)) for a, s, d in args])).encode())
    return h.hexdigest()[:24]


def child(name, out_path):
    """runs inside the per-kernel process"""
    import importlib
    import numpy as np
    import specs
    mod, fn, args = specs.NEFFS[name]
    k = getattr(importlib.import_module(mod), fn)
    arrays = [np.zeros(shape, dtype) for _, shape, dtype in args]
    k[int(os.environ.get('NEFF_LNC', '2'))](*arrays)
    art = os.environ['NKI_ARTIFACTS_DIR']
    shutil.copy(os.path.join(art, 'kernel.neff'), out_path)


def main():
    if len(sys.argv) >= 2 and sys.argv[1] == '--child':
        sys.path.insert(0, HERE)
        child(sys.argv[2], sys.argv[3])
        return
    out_dir, cache_dir = os.path.abspath(sys.argv[1]), os.path.abspath(sys.argv[2])
    sys.path.insert(0, HERE)
    import specs
    names = sys.argv[3:] or list(specs.NEFFS)
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(cache_dir, exist_ok=True)
    cores = [int(c) for c in os.environ.get('NEFF_BUILD_CORES', '0,1,2,3,4,5,6,7').split(',')]
    todo = []
    for n in names:
        cached = os.path.join(cache_dir, f'{n}-{src_hash(n)}.neff')
        if os.path.exists(cached):
            shutil.copy(cached, os.path.join(out_dir, n + '.neff'))
        else:
            todo.append((n, cached))
    print(f'neffs: {len(names) - len(todo)} cached, {len(todo)} to build', flush=True)
    running = []
    free = list(cores)
    failed = []
    tries = {}
    t0 = time.time()
    while todo or running:
        while todo and free:
            n, cached = todo.pop(0)
            core = free.pop(0)
            art = tempfile.mkdtemp(prefix=f'nki_{n}_')
            env = dict(os.environ, NKI_ARTIFACTS_DIR=art, NEURON_RT_VISIBLE_CORES=str(core),
                       NEURON_PLATFORM_TARGET_OVERRIDE='trn2')
            log = open(os.path.join(cache_dir, f'{n}.log'), 'w')
            p = subprocess.Popen([sys.executable, os.path.abspath(__file__), '--child', n, cached + '.tmp'],
                                 env=env, stdout=log, stderr=subprocess.STDOUT, cwd=HERE)
            running.append((p, n, cached, core, art, log))
        time.sleep(0.5)
        for item in list(running):
            p, n, cached, core, art, log = item
            if p.poll() is None:
                continue
            running.remove(item)
            free.append(core)
            log.close()
            shutil.rmtree(art, ignore_errors=True)
            if p.returncode == 0 and os.path.exists(cached + '.tmp'):
                os.replace(cached + '.tmp', cached)
                shutil.copy(cached, os.path.join(out_dir, n + '.neff'))
                print(f'  built {n} ({time.time() - t0:.0f}s)', flush=True)
            else:
                tries[n] = tries.get(n, 0) + 1
                if tries[n] < 3:      # transient failures (a busy NeuronCore) get retried
                    print(f'  retry {n} (attempt {tries[n] + 1})', flush=True)
                    todo.append((n, cached))
                else:
                    failed.append(n)
                    print(f'  FAILED {n}: see {cache_dir}/{n}.log', flush=True)
    if failed:
        sys.exit(f'neff build failed: {failed}')


if __name__ == '__main__':
    main()
