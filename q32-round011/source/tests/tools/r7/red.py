import json, pickle, sys, collections
d = json.load(open(sys.argv[1]))
sgs = collections.Counter(i["subgraph"] for i in d["instruction"])
print(sgs)
keep = ("timestamp", "duration", "label", "opcode", "nki_source_location", "subgraph")
ins = [{k: i.get(k) for k in keep} for i in d["instruction"]]
print(d["instruction"][1000])
pickle.dump(dict(ins=ins, dma=[{k: x.get(k) for k in ("timestamp", "duration", "transfer_size", "subgraph")} for x in d["dma"]],
                 cc=d["cc_ops"], meta=d["metadata"]), open(sys.argv[2], "wb"))
print(d["cc_ops"][0])
