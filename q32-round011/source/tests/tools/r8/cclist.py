import json, sys
d = json.load(open(sys.argv[1]))
cc = sorted(d['cc_ops'], key=lambda c: c['timestamp']); t0 = cc[0]['timestamp']
for c in cc: print(c['operation'], c['algorithm'], c.get('stream_id'), c.get('stream_index'), (c['timestamp'] - t0) // 1000, c['duration'] // 1000)
