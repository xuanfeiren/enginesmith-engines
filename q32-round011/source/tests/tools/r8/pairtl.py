import pickle, sys, collections
d = pickle.load(open(sys.argv[1], "rb")); a, b = int(sys.argv[2]), int(sys.argv[3]); B = 5000
t0 = min(i["timestamp"] for i in d["ins"] if i["subgraph"] == "sg01")
by = {sg: collections.Counter() for sg in ("sg00", "sg01")}
for x in d["dma"]:
    by[x["subgraph"]][(x["timestamp"] - t0) // B] += x["transfer_size"] or 0
src = collections.defaultdict(collections.Counter)
for i in d["ins"]:
    if i["subgraph"] == "sg01" and i["label"] == "TensorMatrix":
        src[(i["timestamp"] - t0) // B][(i["nki_source_location"] or "").split("/")[-1]] += 1
for k in range(a * 1000 // B, b * 1000 // B):
    s0, s1 = by["sg00"][k] / B, by["sg01"][k] / B
    print(f"{k * B // 1000:6d} {s0:5.0f} {s1:5.0f} {s0 + s1:5.0f}  " + " ".join(f"{n}:{c}" for n, c in src[k].most_common(2)))
