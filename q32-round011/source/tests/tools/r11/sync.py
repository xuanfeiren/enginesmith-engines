import pickle, sys
d = pickle.load(open(sys.argv[1], 'rb'))
cc = sorted(d['cc'], key=lambda c: c['timestamp'])
i = int(sys.argv[2]); t0 = cc[i]['timestamp']; lo = t0 - float(sys.argv[3]) * 1e3; hi = t0 + float(sys.argv[4]) * 1e3
for x in sorted(d['ins'], key=lambda x: x['timestamp']):
    if lo <= x['timestamp'] <= hi and x['label'] == 'Sync':
        print(f"{(x['timestamp']-t0)/1e3:8.2f} {x['opcode']:18s} {(x['nki_source_location'] or '').split('/')[-1]:22s} dur {x['duration']/1e3:5.2f} | {(x['operands'] or '')[:150]}")
