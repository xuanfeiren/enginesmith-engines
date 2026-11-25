import json, sys, collections
d = json.load(open(sys.argv[1]))
B = int(sys.argv[2]) if len(sys.argv) > 2 else 20000
t0 = min(i['timestamp'] for i in d['instruction'])
by = collections.defaultdict(lambda: collections.Counter())
for x in d['dma']:
    b = (x['timestamp'] - t0) // B
    by[b][(x['subgraph'], x.get('variable', '?'))] += x['transfer_size']
ins = [i for i in d['instruction'] if i.get('execution_unit') == 'compute']
src = collections.defaultdict(lambda: collections.Counter())
for i in ins:
    loc = i.get('nki_source_location', '?').split('/')[-1]
    src[(i['timestamp'] - t0) // B][(i['subgroup'][:3], loc)] += i['duration']
nb = max(by) + 1
for b in range(int(nb)):
    tot = sum(by[b].values())
    top = by[b].most_common(3)
    ts = src[b].most_common(3)
    print(f"{b*B/1000:6.0f}us {tot/1e6:6.2f}MB {tot/B:5.0f}GB/s | " + ' '.join(f"{s[-1]}:{v}:{n/1e6:.1f}" for (s, v), n in top) + ' || ' + ' '.join(f"{e}:{l}" for (e, l), v in ts))
