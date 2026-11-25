import json, sys, collections
d = json.load(open(sys.argv[1])); B = int(sys.argv[2]) if len(sys.argv) > 2 else 10000  # bucket ns
sg = sys.argv[3] if len(sys.argv) > 3 else "sg00"
T = d["metadata"][0]["last_hw_timestamp"]
nb = T // B + 1
eng = collections.defaultdict(lambda: [0] * nb)
labs = collections.Counter()
for ins in d["instruction"]:
    if ins["subgraph"] != sg: continue
    e = ins["label"]; labs[e] += 1
    t0, du = ins["timestamp"], ins["duration"]
    t = t0
    while du > 0 and t // B < nb:
        b = t // B; take = min(du, (b + 1) * B - t)
        eng[e][b] += take; du -= take; t += take
dma = [0] * nb
for x in d["dma"]:
    if x.get("subgraph") != sg: continue
    dma[min(nb - 1, x["timestamp"] // B)] += x.get("transfer_size", 0)
print("labels", dict(labs))
print("cc_ops:", [(c["subgraph"], c["operation"], c["timestamp"] // 1000, c["duration"] // 1000, c["input_size"]) for c in d["cc_ops"]])
print("ham:", [(h["subgraph"], h["timestamp"] // 1000, h["duration"] // 1000, h["type"]) for h in d.get("ham", [])][:40])
print("core_barriers:", [(c["subgraph"], c["start"] // 1000, c["duration"] // 1000) for c in d.get("core_barriers", []) if c["subgraph"] == sg])
keys = sorted(eng)
print("t_us  " + " ".join(f"{k[:6]:>6}" for k in keys) + "  GB/s")
for b in range(nb):
    print(f"{b * B // 1000:5d} " + " ".join(f"{eng[k][b] / B * 100:6.0f}" for k in keys) + f"  {dma[b] / B:6.0f}")
