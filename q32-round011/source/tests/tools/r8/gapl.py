import pickle, sys
d=pickle.load(open(sys.argv[1],'rb')); lo,hi=float(sys.argv[2])*1e3,float(sys.argv[3])*1e3
ins=sorted([i for i in d['ins'] if i['subgraph']=='sg01' and i['label']=='TensorMatrix'], key=lambda i:i['timestamp'])
t0=min(i['timestamp'] for i in d['ins'] if i['subgraph']=='sg01')
end=None; tot=0; ev=[]
for i in ins:
    t=i['timestamp']-t0
    if lo<=t<hi and end is not None and t-end>3000:
        tot+=t-end; ev.append((end, f"gap {(t-end)/1000:6.1f} next {(i['nki_source_location'] or '').split('/')[-1]}"))
    end=max(end or 0,t+i['duration'])
for c in sorted(d['cc'], key=lambda c:c['timestamp']):
    t=c['timestamp']-t0
    if lo<=t<hi: ev.append((t, f"   CC {c['operation']} dur {c['duration']/1000:.0f} -> ends {(t+c['duration'])/1000:.1f}"))
for t,s in sorted(ev): print(f"{t/1000:9.1f} {s}")
print('gaps total', tot/1e3)
