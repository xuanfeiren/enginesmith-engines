import pickle, numpy as np, sys, collections
d=pickle.load(open(sys.argv[1],'rb'))
ins=[i for i in d['ins'] if i['subgraph']=='sg01']
t0=min(i['timestamp'] for i in ins); t1=max(i['timestamp']+i['duration'] for i in ins)
dma=[x for x in d['dma'] if x['subgraph']=='sg01']
n=int((t1-t0)/1e3)+2; b=np.zeros(n)
for x in dma:
    s=(x['timestamp']-t0)/1e3; e=s+max(x['duration'],1)/1e3; a=int(s); z=int(e); sz=x['transfer_size'] or 0
    if z==a: b[a]+=sz
    else:
        w=sz/(e-s); b[a]+=w*(a+1-s); b[a+1:z]+=w; b[z]+=w*(e-z)
g=b/1e3
print('span', (t1-t0)/1e3, 'low<150 us', (g<150).sum())
cc=sorted(d['cc'],key=lambda c:c['timestamp'])
low=g<150
for c in cc[20:26]:
    s=int((c['timestamp']-t0)/1e3); e=s+int(c['duration']/1e3)
    print(c.get('operation'), s, round(c['duration']/1e3,1), 'low', low[max(0,s-15):e+20].sum())
if len(sys.argv)>2:
    s0=int(sys.argv[2])
    for lo in range(s0-30, s0+30, 3):
        ii=collections.Counter()
        for i in ins:
            t=(i['timestamp']-t0)/1e3
            if lo<=t<lo+3 and i['label'] in ('Sync','GpSimd'): ii[(i['label'],(i['nki_source_location'] or '?').split('/')[-1], i['opcode'])]+=1
        print(lo, round(g[lo:lo+3].mean()), dict(ii))
