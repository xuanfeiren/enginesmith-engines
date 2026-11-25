import pickle, sys, numpy as np
d = pickle.load(open(sys.argv[1], 'rb'))
sg = 'sg01'
ham = sorted([h for h in d['ham'] if h['subgraph'] == sg], key=lambda h: h['timestamp'])
ins = [i for i in d['ins'] if i['subgraph'] == sg]
t0 = min(i['timestamp'] for i in ins)
te = [(i['timestamp'], i['timestamp'] + i['duration']) for i in ins if i.get('label') == 'TensorMatrix']
n = int((max(e for s, e in te) - t0) / 1000) + 2
busy = np.zeros(n)
for s, e in te:
    a = (s - t0) / 1000; b = (e - t0) / 1000; ia, ib = int(a), int(b)
    if ia == ib: busy[ia] += b - a
    else: busy[ia] += ia + 1 - a; busy[ia+1:ib] += 1; busy[ib] += b - ib
busy = np.minimum(busy, 1)
lo, hi = float(sys.argv[2]), float(sys.argv[3])
for h in ham:
    a = int((h['timestamp'] - t0) / 1000); b = a + int(h['duration'] / 1000)
    if a < lo or a > hi: continue
    pre = busy[max(0, a - 10):a].mean() if a > 0 else 0
    dur = busy[a:b].mean() if b > a else 0
    print(f"{a:8d} +{h['duration']/1000:7.1f} k{h['k']}/{h['n']}  TE prev10 {pre:.2f} during {dur:.2f}")
