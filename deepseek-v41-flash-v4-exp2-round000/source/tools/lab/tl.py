import json, sys, collections
d = json.load(open(sys.argv[1]))
B = int(sys.argv[2]) if len(sys.argv) > 2 else 10000
sg = sys.argv[3] if len(sys.argv) > 3 else 'sg00'
ins = [i for i in d['instruction'] if i['subgraph'] == sg]
t0 = min(i['timestamp'] for i in d['instruction'])
busy = collections.defaultdict(lambda: collections.Counter())
src = collections.defaultdict(lambda: collections.Counter())
for i in ins:
    b = (i['timestamp'] - t0) // B
    e = i['subgroup'][:3]
    busy[b][e] += i['duration']
    loc = i.get('nki_source_location', '?').split('/')[-1]
    src[b][(e, loc)] += i['duration']
dm = collections.Counter()
for x in d['dma']:
    if x['subgraph'] == sg: dm[(x['timestamp'] - t0) // B] += x['transfer_size']
for b in range(int(max(busy)) + 1):
    s = ' '.join(f"{e}{busy[b][e]*100//B:3d}" for e in ['Ten', 'Vec', 'Sca', 'GpS', 'Syn'])
    top = ' '.join(f"{e}:{l}:{v*100//B}" for (e, l), v in src[b].most_common(4))
    print(f"{b*B/1000:6.0f}us {dm[b]/B:5.0f}GB/s {s} | {top}")
