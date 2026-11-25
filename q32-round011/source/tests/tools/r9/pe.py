import pickle, sys, collections
d=pickle.load(open(sys.argv[1],'rb'))
ins=[i for i in d['ins'] if i['subgraph']=='sg01']
t0=min(i['timestamp'] for i in ins); t1=max(i['timestamp']+i['duration'] for i in ins)
print('span us', (t1-t0)/1e3, 'n', len(ins), 'keys', list(ins[0].keys()))
by=collections.defaultdict(lambda:[0,0])
for i in ins:
    b=by[i['label']]; b[0]+=1; b[1]+=i['duration']
for k,v in sorted(by.items(), key=lambda x:-x[1][1]): print(k, v[0], v[1]/1e3)
pe=[i for i in ins if i['label']=='TensorMatrix']
h=collections.defaultdict(lambda:[0,0])
for i in pe:
    k=(i.get('opcode') or i.get('name') or '')[:40], (i['nki_source_location'] or '').split('/')[-1]
    h[k][0]+=1; h[k][1]+=i['duration']
for k,v in sorted(h.items(), key=lambda x:-x[1][1])[:25]: print(k, v[0], round(v[1]/1e3,1), 'avg ns', round(v[1]/v[0]))
