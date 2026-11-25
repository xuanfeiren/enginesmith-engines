import sys, collections
rows = [l.split() for l in open(sys.argv[1])]
rows = [(int(a), int(b), int(c), float(d)) for a, b, c, d in rows]
# last request: find the last occurrence of pos 0 arrival at stage 0
starts = [i for i, r in enumerate(rows) if r[0] == 0 and r[1] == 0 and r[2] == 0]
rows = rows[starts[-1]:]
t0 = rows[0][3]
ev = collections.defaultdict(dict)
for st, pos, e, t in rows:
    ev[(st, pos)][e] = (t - t0) * 1e3
def stage_stats(st, a_ev, d_ev):
    poss = sorted(p for (s, p) in ev if s == st)
    poss = [p for p in poss if a_ev in ev[(st, p)] and d_ev in ev[(st, p)]]
    mid = poss[len(poss)//4: 3*len(poss)//4]
    idle = []; gap = []; lat = []
    for i in range(1, len(mid)):
        p, q = mid[i-1], mid[i]
        idle.append(ev[(st, q)][a_ev] - ev[(st, p)][d_ev])   # launch of next - done of prev (ms)
        gap.append(ev[(st, q)][d_ev] - ev[(st, p)][d_ev])
        lat.append(ev[(st, q)].get(3, 0) - ev[(st, q)].get(0, 0))
    import statistics as S
    return S.mean(gap), S.mean(idle), S.mean(lat)
for l in range(40):
    g, i, la = stage_stats(l, 1, 2)
    gm, im, lm = stage_stats(-2000 - l, 11, 12)
    print(f"layer {l:2d}: A done-gap {g:.3f} launch-after-prev-done {i:+.3f} stage lat {la:.3f} | M done-gap {gm:.3f} launch-after-prev-done {im:+.3f} lat {lm:.3f}")
