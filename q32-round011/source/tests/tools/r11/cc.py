import pickle, sys, numpy as np, collections
d = pickle.load(open(sys.argv[1], 'rb'))
cc = sorted(d['cc'], key=lambda c: c['timestamp'])
print(len(cc), collections.Counter((c['operation'], c['output_size']) for c in cc))
# subgraphs: sg00 = core0, sg01 = core1?
print(collections.Counter(i['subgraph'] for i in d['ins']))
t = [(c['cc_trigger']/1e3, c['timestamp']/1e3, c['duration']/1e3) for c in cc]
for i in range(20, 30):
    print(i, ' trig %.1f start %.1f dur %.1f  delay %.1f' % (t[i][0], t[i][1], t[i][2], t[i][1]-t[i][0]))
# layer period: every 2 AR
ar = [x for x in t if True]
per = [ar[i+2][1]-ar[i][1] for i in range(0, len(ar)-3, 2)]
print('period med', np.median(per), 'mean', np.mean(per))
durs_a = [ar[i][2] for i in range(0, 128, 2)]; durs_m = [ar[i][2] for i in range(1, 128, 2)]
print('o_proj AR dur med', np.median(durs_a), 'down AR dur med', np.median(durs_m))
print('trigger delay med a', np.median([ar[i][1]-ar[i][0] for i in range(0,128,2)]), 'm', np.median([ar[i][1]-ar[i][0] for i in range(1,128,2)]))
