import pickle, sys, collections
d = pickle.load(open(sys.argv[1], 'rb'))
cc = sorted(d['cc'], key=lambda c: c['timestamp'])
i = int(sys.argv[2]); t0 = cc[i]['timestamp']; lo = t0 - float(sys.argv[3]) * 1e3; hi = t0 + float(sys.argv[4]) * 1e3
sg = sys.argv[5]
# per engine: list of (start,end,size) merged into transfers per 0.5us
eng = collections.defaultdict(list)
for x in d['dma']:
    if x['subgraph'] == sg and lo <= x['timestamp'] <= hi:
        eng[(x['dma_engine'], x['dma_queue'])].append((x['timestamp'], x['timestamp'] + x['duration'], x['transfer_size'] or 0))
for k in sorted(eng, key=lambda k: (int(k[0]), k[1])):
    L = sorted(eng[k])
    # segments of activity: merge if gap < 300ns
    segs = []
    for s, e, z in L:
        if segs and s - segs[-1][1] < 300: segs[-1][1] = max(segs[-1][1], e); segs[-1][2] += z
        else: segs.append([s, e, z])
    print(k, ' '.join(f'[{(a-t0)/1e3:.1f},{(b-t0)/1e3:.1f}]{c//1024}K' for a, b, c in segs))
