import pickle, sys, collections
d=pickle.load(open(sys.argv[1],'rb'))
print(collections.Counter(i['subgraph'] for i in d['ins']))
for sg in sorted(set(i['subgraph'] for i in d['ins'])):
  for lab in ['TensorMatrix','Tensor']:
    ins=sorted([i for i in d['ins'] if i['subgraph']==sg and i['label']==lab], key=lambda i:i['timestamp'])
    if not ins: continue
    t0=ins[0]['timestamp']; t1=max(i['timestamp']+i['duration'] for i in ins)
    busy=0; end=-1; st=None
    for i in ins:
        s=i['timestamp']; e=s+i['duration']
        if s>end:
            if st is not None: busy+=end-st
            st=s; end=e
        else: end=max(end,e)
    busy+=end-st
    # print a dense sample: 10 consecutive instructions
    mid=len(ins)//2
    samp=[(round((i['timestamp']-ins[mid]['timestamp'])/1),i['duration'],i['opcode']) for i in ins[mid:mid+8]]
    print(sg, lab, 'span', (t1-t0)/1e3, 'busy-union', busy/1e3, samp)
