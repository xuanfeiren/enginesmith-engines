import json, struct, numpy as np, sys
d='/mnt/nvme1/DeepSeek-V4.1-Flash/'
idx=json.load(open(d+'model.safetensors.index.json'))['weight_map']
def load(name):
    f=d+idx[name]; b=open(f,'rb'); n=struct.unpack('<Q',b.read(8))[0]; h=json.loads(b.read(n)); m=h[name]
    s,e=m['data_offsets']; b.seek(8+n+s); return np.frombuffer(b.read(e-s),dtype=np.uint8).reshape(m['shape'])
for L in (0, 4, 20):
  for w in ('wq_a','wq_b','wkv','wo_a','wo_b'):
    a=load(f'layers.{L}.attn.{w}.weight'); R,C=a.shape
    ex=(a>>3)&15; man=a&7
    big=(ex>=15)&(man>0) | (ex==15) # values >=256: exp field 15 -> 2^8 (e4m3fn bias 7: exp 15 -> 2^8)
    big=(ex==15)
    tiny=(ex<=1)&((man&1)==1)&~((ex==0)&(man==0))
    tiny=(ex<=1)&((man&1)==1)
    bb=big.reshape(R//32,32,C//32,32).any(axis=(1,3)); tb=tiny.reshape(R//32,32,C//32,32).any(axis=(1,3))
    print(L,w,a.shape,'big frac %.4f tiny frac %.2e'%(big.mean(),tiny.mean()),'blocks: big %.3f tiny %.3f both %.4f'%(bb.mean(),tb.mean(),(bb&tb).mean()))
