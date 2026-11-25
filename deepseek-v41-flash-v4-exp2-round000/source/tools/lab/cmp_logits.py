import json, sys, numpy as np
L=np.fromfile(sys.argv[1],dtype=np.float32).reshape(-1,128,1010)[2:].transpose(0,2,1).reshape(-1,129280)
R=json.load(open(sys.argv[2]))
for s,row in enumerate(R['rows']):
    ids=[a for a,b in row]; v=np.array([b for a,b in row]); ours=L[s,ids]; d=ours-v
    top=int(L[s].argmax())
    flag = '' if top==ids[0] else ' <-- ours %d (ref rank %s) ourlogit %.2f'%(top, ids.index(top)+1 if top in ids else '>20', L[s,top])
    print(s, 'ref', ids[:3], np.round(v[:3],2), 'ours@', np.round(ours[:3],2), 'maxd %.2f'%np.abs(d).max(), flag)
