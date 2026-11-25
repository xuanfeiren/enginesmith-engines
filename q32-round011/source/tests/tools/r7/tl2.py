import pickle, sys, collections
d = pickle.load(open(sys.argv[1], "rb")); B = int(sys.argv[2]) * 1000; sg = sys.argv[3] if len(sys.argv) > 3 else "sg01"
ins = [i for i in d["ins"] if i["subgraph"] == sg]
t0 = min(i["timestamp"] for i in ins); T = max(i["timestamp"] + i["duration"] for i in ins) - t0
nb = T // B + 1
eng = collections.defaultdict(lambda: [0] * nb)
src = collections.defaultdict(collections.Counter)
for i in ins:
    t, du = i["timestamp"] - t0, i["duration"]
    e = i["label"]
    src[e][(t // B, (i["nki_source_location"] or "").split("/")[-1])] += du
    while du > 0 and t // B < nb:
        b = t // B; take = min(du, (b + 1) * B - t); eng[e][b] += take; du -= take; t += take
dma = [0] * nb
for x in d["dma"]:
    if x.get("subgraph") != sg: continue
    b = (x["timestamp"] - t0) // B
    if 0 <= b < nb: dma[b] += x.get("transfer_size") or 0
keys = sorted(eng)
print("T us", T // 1000)
print("t_us  " + " ".join(f"{k[:6]:>6}" for k in keys) + "  GB/s  top-PE-src")
for b in range(nb):
    pe = [k for k in keys if k.startswith("TensorM")]
    top = src[pe[0]] if pe else {}
    tops = sorted([(v, s) for (bb, s), v in top.items() if bb == b], reverse=True)[:2]
    print(f"{b * B // 1000:5d} " + " ".join(f"{eng[k][b] / B * 100:6.0f}" for k in keys) + f"  {dma[b] / B:6.0f}  " + " ".join(s for v, s in tops))
for c in d["cc"]:
    if c["subgraph"] == "sg00" or True:
        print(c["operation"], (c["timestamp"] - t0) // 1000, c["duration"] // 1000, c["input_size"], c.get("subgraph"))
