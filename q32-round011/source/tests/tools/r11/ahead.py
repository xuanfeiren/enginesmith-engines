import pickle, sys
d = pickle.load(open(sys.argv[1], 'rb'))
cc = sorted(d['cc'], key=lambda c: c['timestamp'])
for i in [41, 61, 81]:
    t0 = cc[i]['timestamp']
    # last TensorE matmul before t0 (down GEMV end), first after t0 + 5us (next layer)
    mm = sorted(x['timestamp'] for x in d['ins'] if x['label'] == 'TensorMatrix' and t0 - 60e3 < x['timestamp'] < t0 + 60e3)
    before = [t for t in mm if t < t0]; after = [t for t in mm if t > t0]
    tend = before[-1]; tnext = after[0]
    # bytes transferred on sg01 between tend and tnext (weights queues: all but 0/11 small)
    tot = sum((x['transfer_size'] or 0) for x in d['dma'] if x['subgraph'] == 'sg01' and tend <= x['timestamp'] < tnext)
    print(i, f'down GEMV end {(tend-t0)/1e3:.1f}  next matmul {(tnext-t0)/1e3:.1f}  bytes in between {tot/1e6:.2f} MB = {tot/128/1024:.0f} KB/partition')
