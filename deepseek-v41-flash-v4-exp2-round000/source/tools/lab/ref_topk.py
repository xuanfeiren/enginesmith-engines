import json, subprocess, sys
sys.path.insert(0, '/mnt/nvme1/es-ds/src')
from enginesmith import workload
seed = bytes.fromhex('121088dca7bd3f739cccccced241266a4b8223f623d8e159a3110f6f09674cf3')
n = 256; r = 1
prompt = workload.prompt(n, 129280, workload.draw_offset(seed, f'nonce/{r}/p{n}', n, workload.NONCE))
toks = [x for x in json.load(open('out/x003.json')) if x['round'] == 1][0]['tokens']
p = subprocess.Popen(['/mnt/nvme1/es-ds/tasks/deepseek-v41-flash/reference.sh', '--model', '/mnt/nvme1/DeepSeek-V4.1-Flash'],
                     stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=open('out/ref.err', 'w'), cwd='/tmp')
def call(o):
    p.stdin.write((json.dumps(o) + '\n').encode()); p.stdin.flush()
    return json.loads(p.stdout.readline())
print(call({'id': 1, 'op': 'hello'}), flush=True)
rep = call({'id': 2, 'op': 'topk_logits', 'prompt': prompt, 'tokens': toks, 'k': 20})
json.dump({'prompt': prompt, 'tokens': toks, 'rows': rep['rows']}, open('out/ref_topk_r1p256.json', 'w'))
p.stdin.close(); p.wait()
print('done', flush=True)
