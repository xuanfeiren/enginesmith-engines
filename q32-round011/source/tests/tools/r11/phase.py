import pickle, sys, numpy as np, collections
d = pickle.load(open(sys.argv[1], 'rb'))
cc = sorted(d['cc'], key=lambda c: c['timestamp'])
dma = d['dma']
print(collections.Counter((x['subgraph'], x['is_cc_dma']) for x in dma))
T = int(max(x['timestamp'] + x['duration'] for x in dma) / 1e3) + 2
def rate(sel):
    b = np.zeros(T * 4)  # 250 ns buckets
    for x in sel:
        s = x['timestamp'] / 250; e = s + max(x['duration'], 1) / 250; sz = x['transfer_size'] or 0
        a = int(s); z = int(e)
        if z == a: b[a] += sz
        else:
            w = sz / (e - s); b[a] += w * (a + 1 - s); b[a + 1:z] += w; b[z] += w * (e - z)
    return b / 250  # bytes/ns = GB/s
for sg in ('sg00', 'sg01'):
    g = rate([x for x in dma if x['subgraph'] == sg and x['is_cc_dma'] == 'no'])
    print(sg, 'mean GB/s over span', g[:T*4].mean())
    # phase relative to down AR start (odd cc idx), layers 4..60
    W0, W1 = -170, 60
    acc = np.zeros((W1 - W0) * 4); n = 0
    for i in range(1 + 8, 128 - 8, 2):
        s = int(cc[i]['timestamp'] / 250)
        acc += g[s + W0 * 4:s + W1 * 4]; n += 1
    acc /= n
    for k in range(0, len(acc), 8):
        us = W0 + k / 4
        print(f'{sg} {us:7.1f} {acc[k:k+8].mean():6.0f}', '#' * int(acc[k:k+8].mean() / 10))
