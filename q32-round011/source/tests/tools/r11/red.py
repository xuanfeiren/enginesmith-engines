import json, pickle, sys
d = json.load(open(sys.argv[1]))
keep = ("timestamp", "duration", "label", "opcode", "nki_source_location", "subgraph", "evt_wait_time", "operands", "engine_instance_idx")
ins = [{k: i.get(k) for k in keep} for i in d["instruction"]]
dma = [{k: x.get(k) for k in ("timestamp", "duration", "transfer_size", "subgraph", "dma_queue", "semaphore_id", "variable", "is_cc_dma", "dma_engine", "queue_type", "source", "dest")} for x in d["dma"]]
pickle.dump(dict(ins=ins, dma=dma, cc=d["cc_ops"], meta=d["metadata"], sem=d["semaphore_update"], cb=d["core_barriers"]), open(sys.argv[2], "wb"))
