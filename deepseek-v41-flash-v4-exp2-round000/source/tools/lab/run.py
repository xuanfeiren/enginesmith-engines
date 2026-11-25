"""run one engine over a prompt set (split requests), dump tokens. usage: run.py ENGINE OUT.json [--set std|rep] """
import argparse, json, sys, time, os, random
sys.path.insert(0, '/mnt/nvme1/es-ds/tasks/deepseek-v41-flash/runs/deepseek-v41-flash-20261109-070642/work/008/brief/skill/scripts')
import engine_client as ec
ap = argparse.ArgumentParser(); ap.add_argument('engine'); ap.add_argument('out')
ap.add_argument('--set', default='std'); ap.add_argument('--reps', type=int, default=1)
ap.add_argument('--seed', type=int, default=1)
a = ap.parse_args()
L = ec.Launch('/mnt/nvme1/DeepSeek-V4.1-Flash', 129280)
rng = random.Random(a.seed)
if a.set == 'std':
    lens = [256, 1024, 4096, 256, 1, 5, 64, 65, 128, 129, 255, 257, 511, 513, 1023, 1025, 2048, 2049, 4097, 256, 300]
elif a.set == 'eval':
    lens = [256, 1024, 4096] * 3
else:
    lens = [int(x) for x in a.set.split(',')]
prompts = [[rng.randrange(1000, 100000) for _ in range(n)] for n in lens]
res = []
with ec.gpu_lock():
    t = time.time(); e = ec.Engine(a.engine, L); print(f'load {time.time()-t:.1f}s', flush=True)
    try:
        e.split(ec.nonce_prompt(64, 77777, L.vocab), 8)
        for rep in range(a.reps):
            for i, p in enumerate(prompts):
                toks, pm, dm = e.split(p, 64)
                print(f'{i} len {len(p)} rep {rep}: prefill {pm:.1f} decode {dm:.1f}', flush=True)
                res.append({'i': i, 'len': len(p), 'rep': rep, 'tokens': toks, 'pm': pm, 'dm': dm})
    finally:
        e.close(); os.system(f'cp {e.stderr_path} {a.out}.stderr')
json.dump(res, open(a.out, 'w'))
