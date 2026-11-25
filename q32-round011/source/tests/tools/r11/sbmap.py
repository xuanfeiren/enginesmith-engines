import pickle, re, sys, collections
d = pickle.load(open(sys.argv[1], 'rb'))
cc = sorted(d['cc'], key=lambda c: c['timestamp'])
L = int(sys.argv[2])  # layer index: o_proj AR = cc[2L], down AR = cc[2L+1]
a = cc[2 * L - 1]['timestamp'] - 5e3; b = cc[2 * L + 1]['timestamp'] + 40e3
t0 = cc[2 * L + 1]['timestamp']
names = {}
rows = []
for x in sorted(d['ins'], key=lambda x: x['timestamp']):
    if a <= x['timestamp'] <= b and x['label'] == 'Sync' and x['opcode'] == 'DMA_DIRECT2D':
        o = x['operands'] or ''
        g = lambda k: (re.search(k + r'=(\S+)', o) or [None, None])[1]
        dst = g('dst_addr_imm'); ti = g('src_table_index'); so = g('src_table_offset_imm'); es = g('dst_elem_size')
        wt = re.search(r'^(\S+ \(\S+\)>=\d+)', o)
        if dst is None: continue
        off = int(dst, 16) & 0x3ffff
        rows.append(((x['timestamp'] - t0) / 1e3, ti, int(so, 16) if so else -1, off, int(es), x['duration'] / 1e3, wt.group(1) if wt else ''))
for r in rows:
    print(f'{r[0]:8.1f} tab {r[1]:>3} src {r[2]:#10x} dst {r[3]/1024:7.1f}K..{(r[3]+r[4])/1024:7.1f}K size {r[4]/1024:5.1f}K dur {r[5]:5.2f} {r[6]}')
