import pickle, sys, collections
for f in sys.argv[1:]:
    d = pickle.load(open(f, "rb"))
    t0 = min(i["timestamp"] for i in d["ins"] if i["subgraph"] == "sg01")
    B = 5000
    tot = collections.Counter()
    for x in d["dma"]:
        tot[(x["timestamp"] - t0) // B] += x["transfer_size"] or 0
    lo, hi = 600, 2600   # 3 .. 13 ms
    r = sorted(tot[b] / B for b in range(lo, hi))   # bytes/ns = GB/s (both cores)
    n = len(r)
    print(f, "mean", round(sum(r) / n), "p10", round(r[n // 10]), "p50", round(r[n // 2]), "p90", round(r[9 * n // 10]),
          "low(<400) buckets", sum(1 for x in r if x < 400), "of", n)
