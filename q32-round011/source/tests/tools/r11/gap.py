import pickle, sys, numpy as np, collections
d = pickle.load(open(sys.argv[1], 'rb'))
cc = sorted(d['cc'], key=lambda c: c['timestamp'])
i = int(sys.argv[2]); t0 = cc[i]['timestamp']
for sg in ('sg00', 'sg01'):
    xs = sorted((x['timestamp'], x['timestamp'] + x['duration'], x['transfer_size'], x['dma_queue'], x['dma_engine']) for x in d['dma'] if x['subgraph'] == sg and abs(x['timestamp'] - t0) < 40e3)
    # print packets per 1us bucket by queue
    b = collections.defaultdict(lambda: collections.Counter())
    for s, e, sz, q, en in xs:
        b[int((s - t0) // 1000)][q] += sz
    for k in sorted(b):
        print(sg, k, dict(b[k]))
