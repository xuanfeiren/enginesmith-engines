"""compare two DSV4_DUMP files record by record: first differing (layer, pos)"""
import sys, struct, collections
def recs(fn):
    out = []
    with open(fn, 'rb') as f:
        while True:
            h = f.read(12)
            if len(h) < 12: break
            l, p, n = struct.unpack('<iii', h)
            out.append((l, p, f.read(n)))
    return out
a, b = recs(sys.argv[1]), recs(sys.argv[2])
print(len(a), len(b), 'records')
# group by (layer, pos) with occurrence index (decode records have pos -1)
def keyed(r):
    c = collections.Counter(); d = {}
    for l, p, x in r:
        k = (l, p, c[(l, p)]); c[(l, p)] += 1; d[k] = x
    return d
A, B = keyed(a), keyed(b)
diff = [k for k in A if k in B and A[k] != B[k]]
print('common', len(set(A) & set(B)), 'differ', len(diff))
for k in sorted(diff, key=lambda k: (k[2], k[1], k[0]))[:10]: 
    x, y = A[k], B[k]
    nd = sum(1 for i in range(0, len(x), 2) if x[i:i+2] != y[i:i+2])
    print('  diff', k, 'bf16 words differing', nd, 'of', len(x)//2)
