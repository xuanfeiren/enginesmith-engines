import json, sys, collections, numpy as np, pickle, os
f = sys.argv[1]; lo = float(sys.argv[2]); hi = float(sys.argv[3]); B = float(sys.argv[4]) if len(sys.argv) > 4 else 5
pk = f + '.pkl'
if os.path.exists(pk):
    d = pickle.load(open(pk, 'rb'))
else:
    j = json.load(open(f))
    d = dict(ins=[{k: i.get(k) for k in ('timestamp', 'duration', 'label', 'opcode', 'nki_source_location', 'subgraph')} for i in j['instruction']],
             dma=[{k: x.get(k) for k in ('timestamp', 'duration', 'transfer_size', 'subgraph', 'queue_name')} for x in j['dma']],
             cc=j['cc_ops'], ham=j['ham'])
    pickle.dump(d, open(pk, 'wb'))
sg = 'sg01'
ins = [i for i in d['ins'] if i['subgraph'] == sg]
t0 = min(i['timestamp'] for i in ins)
nb = int((hi - lo) / B) + 1
eng = collections.defaultdict(lambda: np.zeros(nb)); src = collections.defaultdict(collections.Counter)
for i in ins:
    s = (i['timestamp'] - t0) / 1000
    if not (lo <= s < hi): continue
    b = int((s - lo) / B)
    eng[i['label']][b] += i['duration'] / 1000
    if i['label'] == 'TensorMatrix':
        src[b][(i['nki_source_location'] or '?').split('/')[-1]] += i['duration'] / 1000
dmab = np.zeros(nb)
for x in d['dma']:
    if x['subgraph'] != sg: continue
    s = (x['timestamp'] - t0) / 1000
    if lo <= s < hi: dmab[int((s - lo) / B)] += x.get('transfer_size') or 0
lim = ['' for _ in range(nb)]
for h in d['ham']:
    if h['subgraph'] != sg: continue
    a = (h['timestamp'] - t0) / 1000; e = a + h['duration'] / 1000
    for b in range(nb):
        bs = lo + b * B
        if a < bs + B and e > bs and h['k'] < h['n']: lim[b] = '*'
ccs = collections.defaultdict(list)
for c in d['cc']:
    s = (c['timestamp'] - t0) / 1000
    if lo <= s < hi: ccs[int((s - lo) / B)].append(f"{c.get('operation')} {c['duration']/1000:.0f}us")
labs = sorted(eng)
print('t(us) thr ' + ' '.join(l[:6] for l in labs) + '  DMA GB/s  top TE src')
for b in range(nb):
    top = ', '.join(f"{k}:{v:.1f}" for k, v in src[b].most_common(2))
    print(f"{lo + b * B:7.0f} {lim[b]:1s} " + ' '.join(f"{eng[l][b] / B:6.2f}" for l in labs) + f" {dmab[b] / (B * 1e3):7.0f}  {top} {' '.join(ccs[b])}")
