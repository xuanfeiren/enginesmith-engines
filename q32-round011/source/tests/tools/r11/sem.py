import pickle, sys
d = pickle.load(open(sys.argv[1], 'rb'))
cc = sorted(d['cc'], key=lambda c: c['timestamp'])
i = int(sys.argv[2]); t0 = cc[i]['timestamp']; lo = t0 - float(sys.argv[3]) * 1e3; hi = t0 + float(sys.argv[4]) * 1e3
ids = sys.argv[5].split(',')
print(d['sem'][0])
for s in sorted(d['sem'], key=lambda s: s['timestamp']):
    if lo <= s['timestamp'] <= hi and s['id'] in ids:
        print(f"{(s['timestamp']-t0)/1e3:8.2f} id {s['id']} {s['update_operation_type']} {s['value']} src {s['update_source']} {s.get('subgraph')}")
