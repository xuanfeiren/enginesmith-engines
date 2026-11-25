import pickle, sys, collections
d = pickle.load(open(sys.argv[1], "rb")); sg = "sg01"
ins = [i for i in d["ins"] if i["subgraph"] == sg]
t0 = min(i["timestamp"] for i in ins)
dm = [x for x in d["dma"] if x.get("subgraph") == sg]
print(len(dm), dm[0])
for a, b in [(int(x.split(':')[0]), int(x.split(':')[1])) for x in sys.argv[2:]]:
    w = [x for x in dm if a * 1000 <= x["timestamp"] - t0 < b * 1000]
    by = sum(x["transfer_size"] or 0 for x in w); du = sum(x["duration"] for x in w)
    sz = collections.Counter(x["transfer_size"] for x in w).most_common(4)
    print(f"{a}-{b} us: n={len(w)} GB/s={by / (b - a) / 1000:.0f} avg_inflight={du / ((b - a) * 1000):.1f} mean_dur_ns={du / max(1, len(w)):.0f} sizes={sz}")
