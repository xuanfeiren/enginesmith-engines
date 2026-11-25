# timeline from neuron-explorer ntff.json: per 10us bin, busy fraction per engine and DMA bytes (sg00)
import json, sys, collections
d = json.load(open(sys.argv[1])); BIN = int(sys.argv[2]) if len(sys.argv) > 2 else 10000
sg = sys.argv[3] if len(sys.argv) > 3 else "sg00"
busy = collections.defaultdict(lambda: collections.defaultdict(int))
end = 0
for i in d["instruction"]:
    if i["subgraph"] != sg: continue
    t0, du = i["timestamp"], i["duration"]; eng = i["subgroup"]
    end = max(end, t0 + du)
    t = t0
    while t < t0 + du:
        b = t // BIN; nb = (b + 1) * BIN; seg = min(nb, t0 + du) - t
        busy[eng][b] += seg; t += seg
dmab = collections.defaultdict(int)
for x in d["dma"]:
    if x["subgraph"] != sg: continue
    dmab[x["timestamp"] // BIN] += x.get("transfer_size", 0)
engs = sorted(busy)
print("bin(us) " + " ".join(f"{e[:6]:>6}" for e in engs) + "  DMA_MB  GB/s")
for b in range(end // BIN + 1):
    print(f"{b*BIN/1000:6.0f}  " + " ".join(f"{busy[e][b]/BIN:6.2f}" for e in engs) + f"  {dmab[b]/1e6:6.2f}  {dmab[b]/BIN:5.0f}")
for c in d["cc_ops"]:
    if c["subgraph"] == sg: print("cc", c["operation"], c["timestamp"] / 1000, c["duration"] / 1000)
for c in d["core_barriers"]:
    if c["subgraph"] == sg: print("barrier", c["start"] / 1000, (c["end"] - c["start"]) / 1000)
