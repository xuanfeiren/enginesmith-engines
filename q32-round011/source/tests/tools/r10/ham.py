import json, sys, numpy as np
d = json.load(open(sys.argv[1]))
sg = 'sg01'
ham = sorted([h for h in d['ham'] if h['subgraph'] == sg], key=lambda h: h['timestamp'])
ins = [i for i in d['instruction'] if i['subgraph'] == sg]
t0 = min(i['timestamp'] for i in ins)
te = sorted([(i['timestamp'], i['timestamp'] + i['duration']) for i in ins if i.get('label') == 'TensorMatrix'])
T1 = max(e for s, e in te)
# busy array 1us resolution
n = int((T1 - t0) / 1000) + 2
busy = np.zeros(n)
for s, e in te:
    a = (s - t0) / 1000; b = (e - t0) / 1000
    ia, ib = int(a), int(b)
    if ia == ib: busy[ia] += b - a
    else:
        busy[ia] += ia + 1 - a; busy[ia+1:ib] += 1; busy[ib] += b - ib
busy = np.minimum(busy, 1)
print('TE busy total us', busy.sum(), 'span', n)
lim = np.ones(n)
for h in ham:
    a = int((h['timestamp'] - t0) / 1000); b = int((h['timestamp'] + h['duration'] - t0) / 1000)
    a = max(a, 0); b = min(b, n)
    if b > a: lim[a:b] = h['k'] / h['n']
print('limit<1 us', (lim < 1).sum(), 'TE busy under limit', busy[lim < 1].sum(), 'TE busy at full', busy[lim == 1].sum())
types = {}
for h in ham: types[(h['type'], h['k'], h['n'])] = types.get((h['type'], h['k'], h['n']), 0) + h['duration']
print(types)
# print first 40 ham events relative with TE busy before / during
for h in ham[int(sys.argv[2]) if len(sys.argv) > 2 else 0:][:60]:
    a = int((h['timestamp'] - t0) / 1000); b = a + int(h['duration'] / 1000)
    if a < 0: continue
    pre = busy[max(0, a - 20):a].mean() if a > 0 else 0
    dur = busy[a:b].mean() if b > a else 0
    print(f"{a:8d} +{h['duration']/1000:7.1f} k{h['k']}/{h['n']} t{h['type']}  TE busy prev20 {pre:.2f} during {dur:.2f}")
