import pickle, sys, numpy as np, collections
d = pickle.load(open(sys.argv[1], 'rb'))
cc = sorted(d['cc'], key=lambda c: c['timestamp'])
i = int(sys.argv[2]); lo = cc[i]['timestamp'] - float(sys.argv[3]) * 1e3; hi = cc[i]['timestamp'] + float(sys.argv[4]) * 1e3
t0 = cc[i]['timestamp']
print('AR', cc[i]['operation'], 'trig', (cc[i]['cc_trigger'] - t0) / 1e3, 'dur', cc[i]['duration'] / 1e3)
ev = []
for x in d['ins']:
    if lo <= x['timestamp'] <= hi and x['label'] not in ('Tensor', 'TensorMatrix'):
        ev.append((x['timestamp'], x['subgraph'] + ' ' + x['label'], x['opcode'], (x['nki_source_location'] or '').split('/')[-1], x['duration'], x['evt_wait_time']))
# tensor engine: summarize per 1us: count matmuls per subgraph
tens = collections.defaultdict(list)
for x in d['ins']:
    if lo <= x['timestamp'] <= hi and x['label'] == 'TensorMatrix':
        tens[x['subgraph']].append((x['timestamp'], (x['nki_source_location'] or '').split('/')[-1]))
for sg in tens:
    L = sorted(tens[sg]); 
    # group consecutive by source line
    grp = []
    for t, s in L:
        if grp and grp[-1][2] == s and t - grp[-1][1] < 2000: grp[-1][1] = t; grp[-1][3] += 1
        else: grp.append([t, t, s, 1])
    for g in grp: ev.append((g[0], sg + ' Tensor', f'x{g[3]} until {(g[1]-t0)/1e3:.1f}', g[2], 0, 0))
for e in sorted(ev):
    print(f'{(e[0]-t0)/1e3:8.2f} {e[1]:12s} {str(e[2]):28s} {str(e[3]):28s} dur {e[4]/1e3:.2f} wait {(e[5] or 0)/1e3:.2f}')
# dma by variable / queue in the window
dq = collections.defaultdict(lambda: [1e18, 0, 0])
for x in d['dma']:
    if lo <= x['timestamp'] <= hi:
        k = (x['subgraph'], x['variable'], x['dma_queue'])
        a = dq[k]; a[0] = min(a[0], x['timestamp']); a[1] = max(a[1], x['timestamp'] + x['duration']); a[2] += x['transfer_size'] or 0
for k, a in sorted(dq.items(), key=lambda kv: kv[1][0]):
    print('DMA', k, f'{(a[0]-t0)/1e3:.1f} .. {(a[1]-t0)/1e3:.1f}', a[2])
