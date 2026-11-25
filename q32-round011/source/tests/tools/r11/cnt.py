import pickle, sys, collections, re
d = pickle.load(open(sys.argv[1], 'rb'))
cc = sorted(d['cc'], key=lambda c: c['timestamp'])
a = cc[60]['timestamp']; b = cc[62]['timestamp']
c = collections.Counter(); tm = collections.Counter()
for x in d['ins']:
    if a <= x['timestamp'] < b and x['label'] == 'Sync' and x['subgraph'] == 'sg01':
        m = re.search(r'src_elem_size=(\d+)', x['operands'] or '')
        k = (x['opcode'], m.group(1) if m else '-', (x['nki_source_location'] or '').split('/')[-1])
        c[k] += 1; tm[k] += x['duration']
for k in sorted(c): print(k, c[k], f'{tm[k]/1e3:.1f} us')
print('layer us', (b - a) / 1e3, 'sync busy', sum(tm.values()) / 1e3)
