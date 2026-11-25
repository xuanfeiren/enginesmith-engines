import pickle, sys, collections
d = pickle.load(open(sys.argv[1], "rb")); lo, hi = float(sys.argv[2]) * 1000, float(sys.argv[3]) * 1000
th = float(sys.argv[4]) * 1000 if len(sys.argv) > 4 else 2000
ins = sorted([i for i in d["ins"] if i["subgraph"] == "sg01" and i["label"] == "TensorMatrix"], key=lambda i: i["timestamp"])
t0 = min(i["timestamp"] for i in d["ins"] if i["subgraph"] == "sg01")
end = None; tot = collections.Counter(); n = collections.Counter()
for i in ins:
    t = i["timestamp"] - t0
    if lo <= t < hi and end is not None and t - end > th:
        k = (i["nki_source_location"] or "").split("/")[-1]
        tot[k] += t - end; n[k] += 1
    end = max(end or 0, t + i["duration"])
for k, v in tot.most_common(12): print(k, "gaps", n[k], "us", v // 1000)
print("---")
end=None
for i in ins:
    t = i["timestamp"] - t0
    if lo <= t < hi and end is not None and t - end > th * 5:
        print(f"{end/1000:9.1f} gap {(t-end)/1000:6.1f} next {(i['nki_source_location'] or '').split('/')[-1]}")
    end = max(end or 0, t + i["duration"])
