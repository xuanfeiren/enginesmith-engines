import pickle, sys
d = pickle.load(open(sys.argv[1], "rb")); lo, hi = float(sys.argv[2]) * 1000, float(sys.argv[3]) * 1000
labs = sys.argv[4].split(",") if len(sys.argv) > 4 else None
ins = [i for i in d["ins"] if i["subgraph"] == "sg01"]
t0 = min(i["timestamp"] for i in ins)
for i in sorted(ins, key=lambda i: i["timestamp"]):
    t = i["timestamp"] - t0
    if lo <= t < hi and (labs is None or i["label"] in labs):
        print(f'{t/1000:9.2f} {i["duration"]/1000:8.2f} {i["label"]:12s} {i["opcode"]:22s} {(i["nki_source_location"] or "").split("/")[-1]}')
