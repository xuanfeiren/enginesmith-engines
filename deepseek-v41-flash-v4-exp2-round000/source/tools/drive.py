"""lab driver: one engine process, split requests at several lengths, tokens + times to JSON.
usage: drive.py ENGINE OUT.json [--lens 256,1024,4096] [--offsets 0] [--decode 64] [--reps 1]"""
import argparse, json, sys, time, os
sys.path.insert(0, os.environ.get('SKILL_SCRIPTS', os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../brief/skill/scripts')))
import engine_client as ec
ap = argparse.ArgumentParser()
ap.add_argument('engine'); ap.add_argument('out')
ap.add_argument('--lens', default='256,1024,4096'); ap.add_argument('--offsets', default='1000')
ap.add_argument('--decode', type=int, default=64); ap.add_argument('--reps', type=int, default=1)
ap.add_argument('--warm', type=int, default=1)
a = ap.parse_args()
L = ec.Launch('/mnt/nvme1/DeepSeek-V4.1-Flash', 129280)
res = []
with ec.gpu_lock():
    t = time.time()
    e = ec.Engine(a.engine, L)
    print(f'load {time.time() - t:.1f}s', flush=True)
    try:
        if a.warm:
            e.split(ec.nonce_prompt(64, 77777, L.vocab), 8)
        for rep in range(a.reps):
            for off in [int(x) for x in a.offsets.split(',')]:
                for n in [int(x) for x in a.lens.split(',')]:
                    p = ec.nonce_prompt(n, off + rep * 100000 + n, L.vocab)
                    toks, pm, dm = e.split(p, a.decode)
                    print(f'len {n} off {off} rep {rep}: prefill {pm:.1f} ms, decode {dm:.1f} ms ({dm / a.decode:.2f}/tok)', flush=True)
                    res.append({'len': n, 'off': off, 'rep': rep, 'tokens': toks, 'prefill_ms': pm, 'decode_ms': dm})
    finally:
        e.close()
        os.system(f'cp {e.stderr_path} {a.out}.stderr')
json.dump(res, open(a.out, 'w'))
