import sys, collections, statistics as S
rows = [l.split() for l in open(sys.argv[1])]
rows = [(int(a), int(b), int(c), float(d)) for a, b, c, d in rows]
starts = [i for i, r in enumerate(rows) if r[0] == 0 and r[1] == 0 and r[2] == 0]
rows = rows[starts[-1]:]
ev = collections.defaultdict(dict)
for st, pos, e, t in rows: ev[(st, pos)][e] = t
for st in [int(x) for x in sys.argv[2:]]:
    ps = sorted(p for (s, p) in ev if s == st)
    ps = ps[len(ps)//4:3*len(ps)//4]
    d = collections.defaultdict(list)
    for i in range(1, len(ps)):
        a, b = ev[(st, ps[i-1])], ev[(st, ps[i])]
        for k1, k2 in [(0,1),(1,2),(2,3)]:
            if k1 in b and k2 in b: d[f'{k1}->{k2}'].append((b[k2]-b[k1])*1e3)
        if 3 in a and 0 in b: d['out->next arrival'].append((b[0]-a[3])*1e3)
        if 0 in a and 0 in b: d['period'].append((b[0]-a[0])*1e3)
    print(st, {k: round(S.mean(v), 3) for k, v in d.items()})
