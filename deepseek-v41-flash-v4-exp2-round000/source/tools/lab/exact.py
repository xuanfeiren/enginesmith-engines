"""replay the harness sequence of round 003's evaluation (seed, round r) on one engine. usage: exact.py ENGINE OUT.json [--round 1] [--reps 1]"""
import argparse, json, sys, os, time
sys.path.insert(0, '/mnt/nvme1/es-ds/src')
sys.path.insert(0, '/mnt/nvme1/es-ds/tasks/deepseek-v41-flash/runs/deepseek-v41-flash-20261109-070642/work/005/brief/skill/scripts')
from enginesmith import workload
from enginesmith.evaluate import WARMUP_LENGTH
import engine_client as ec
ap = argparse.ArgumentParser(); ap.add_argument('engine'); ap.add_argument('out')
ap.add_argument('--rounds', default='1'); ap.add_argument('--reps', type=int, default=1)
ap.add_argument('--cases', default='256,1024,4096')
a = ap.parse_args()
seed = bytes.fromhex('121088dca7bd3f739cccccced241266a4b8223f623d8e159a3110f6f09674cf3')
V = 129280
def off(n, r): return workload.draw_offset(seed, f'nonce/{r}/p{n}', n, workload.NONCE)
wl = min(WARMUP_LENGTH, 4096)
warm = workload.prompt(wl, V, workload.draw_offset(seed, 'warmup/0', wl, workload.NONCE))
L = ec.Launch('/mnt/nvme1/DeepSeek-V4.1-Flash', V)
res = []
with ec.gpu_lock():
    for rep in range(a.reps):
        e = ec.Engine(a.engine, L)
        try:
            e.generate(warm, 2)
            for r in [int(x) for x in a.rounds.split(',')]:
                for n in [int(x) for x in a.cases.split(',')]:
                    toks, pm, dm = e.split(workload.prompt(n, V, off(n, r)), 64)
                    print(rep, r, n, pm, dm, toks, flush=True)
                    res.append({'rep': rep, 'round': r, 'len': n, 'tokens': toks})
        finally:
            e.close()
json.dump(res, open(a.out, 'w'))
