import sys, torch, numpy as np, struct
rec = torch.load('out/ref_layers2.pt')
data = open('out/dump_r1p256.bin', 'rb').read()
ours = {}; o = 0
while o < len(data):
    l, pos, n = struct.unpack_from('<iii', data, o); o += 12
    if l >= 2000 and pos >= 300: ours[(l, pos)] = data[o:o + n]
    o += n
def bf(b): return torch.from_numpy((np.frombuffer(b, dtype=np.uint16).astype(np.uint32) << 16).view(np.float32))
D = 5120
def rel(a, b): return ((a - b).norm() / b.norm()).item()
for pos in sorted(rec):
    R = rec[pos]; print('=== pos', pos)
    for l in range(0, 12):
        s = f'{l:2d}'
        if (5000 + l, pos) in ours:
            s += ' eng_in %.4f eng_out %.4f' % (rel(bf(ours[(4000 + l, pos)]), R[f'L{l-1}.out'][0].float().reshape(-1)), rel(bf(ours[(5000 + l, pos)]), R[f'L{l}.in'].float().reshape(-1)))
        h2 = bf(ours[(3000 + l, pos)]); s += ' h2 %.4f' % rel(h2, R[f'L{l}.hcpost0'].float().reshape(-1))
        x = bf(ours[(2000 + l, pos)][:D * 2]); s += ' xffn %.4f' % rel(x, R[f'L{l}.ffn'][0].float().reshape(-1))
        s += '  |ref attn| %.3g |ref in| %.3g' % (R[f'L{l}.attn'].float().norm(), R[f'L{l}.in'].float().norm())
        print(s)
