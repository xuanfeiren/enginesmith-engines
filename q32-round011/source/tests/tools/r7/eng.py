import pickle, sys, collections, statistics
d = pickle.load(open(sys.argv[1], "rb")); lo, hi = float(sys.argv[2]) * 1000, float(sys.argv[3]) * 1000
labs = sys.argv[4].split(",") if len(sys.argv) > 4 else ["Sync", "Vector", "Scalar", "GpSimd"]
ins = [i for i in d["ins"] if i["subgraph"] == "sg01"]
t0 = min(i["timestamp"] for i in ins)
c = collections.defaultdict(list)
for i in ins:
    if lo <= i["timestamp"] - t0 < hi and i["label"] in labs:
        c[(i["label"], i["opcode"], (i["nki_source_location"] or "").split("/")[-1])].append(i["duration"])
for k, v in sorted(c.items(), key=lambda kv: -sum(kv[1]))[:16]:
    print(k, "n", len(v), "sum us", sum(v) // 1000, "median ns", statistics.median(v))
