"""Reference per-layer dump at chosen decode positions: block outputs, ffn_pre, attn outputs, routes.
usage: python ref_layers.py OUT.pt POS[,POS...]  (prompt/tokens from out/ref_topk_r1p256.json)"""
import sys, json, torch
sys.path.insert(0, '/mnt/nvme1/ds-ref')
import deepseek_cpu as R
from pathlib import Path
out, wanted = sys.argv[1], set(int(x) for x in sys.argv[2].split(','))
d = json.load(open('out/ref_topk_r1p256.json')); prompt, toks = d['prompt'], d['tokens']
ref = R.Reference(Path('/mnt/nvme1/DeepSeek-V4.1-Flash'), 48)
m = ref.model
cur = {'pos': None}; rec = {}
def H(name, fn):
    def hook(mod, inp, outp):
        p = cur['pos']
        if p in wanted: rec.setdefault(p, {})[name] = fn(inp, outp)
    return hook
for i, b in enumerate(m.layers):
    b.register_forward_hook(H(f'L{i}.out', lambda i_, o: (o[0].detach().clone(), o[1].detach().clone())))
    b.register_forward_hook(H(f'L{i}.in', lambda i_, o: i_[0].detach().clone()))
    b.attn.register_forward_hook(H(f'L{i}.attn', lambda i_, o: o.detach().clone()))
    b.ffn.gate.register_forward_hook(H(f'L{i}.gate', lambda i_, o: (o[0].detach().clone(), o[1].detach().clone())))
    b.ffn.register_forward_hook(H(f'L{i}.ffn', lambda i_, o: (i_[0].detach().clone(), o.detach().clone())))
    if b.engram is not None:
        b.engram.register_forward_hook(H(f'L{i}.engram', lambda i_, o: o.detach().clone()))
import types
for i, b in enumerate(m.layers):
    def mk(i, b, f0):
        cnt = {'n': 0}
        def hp(self, x, residual, post, comb):
            y = f0(x, residual, post, comb)
            p = cur['pos']
            if p in wanted:
                k = f'L{i}.hcpost{len([1 for kk in rec.get(p, {}) if kk.startswith(f"L{i}.hcpost")])}'
                rec.setdefault(p, {})[k] = y.detach().clone()
            return y
        return types.MethodType(hp, b)
    b.hc_post = mk(i, b, b.hc_post)
orig = ref._forward
def fwd(ids, start_pos):
    cur['pos'] = start_pos if len(ids) == 1 else -1
    r = orig(ids, start_pos)
    if cur['pos'] in wanted: rec[cur['pos']]['logits'] = r.clone()
    return r
ref._forward = fwd
need = max(wanted) - len(prompt) + 2
for _ in ref._teacher_force(prompt, toks[:need]): pass
torch.save(rec, out); print('saved', list(rec.keys()))
