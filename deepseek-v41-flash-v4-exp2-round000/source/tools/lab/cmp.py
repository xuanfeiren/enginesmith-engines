import json, sys
A = json.load(open(sys.argv[1])); B = json.load(open(sys.argv[2]))
ka = {(r['i'], r['rep']): r for r in A}
nd = 0
for r in B:
    k = (r['i'], r['rep'] if (r['i'], r['rep']) in ka else 0)
    a = ka.get(k)
    if not a: continue
    if a['tokens'] != r['tokens']:
        nd += 1
        f = next(j for j in range(len(a['tokens'])) if a['tokens'][j] != r['tokens'][j])
        print('DIFF', r['i'], r['len'], 'rep', r['rep'], 'first step', f)
print('diffs', nd, 'of', len(B))
