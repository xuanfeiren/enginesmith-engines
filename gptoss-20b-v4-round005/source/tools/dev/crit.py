import json, sys
d = json.load(open(sys.argv[1])); a = float(sys.argv[2]) * 1000; b = float(sys.argv[3]) * 1000
sg = sys.argv[4] if len(sys.argv) > 4 else "sg01"
ins = [i for i in d["instruction"] if i["subgraph"] == sg]
t0 = min(i["timestamp"] for i in ins)
ins = sorted([i for i in ins if a <= i["timestamp"] - t0 < b], key=lambda i: i["timestamp"])
# compress runs of same engine+opcode
runs = []
for i in ins:
    e = i["subgroup"][:3]; t = i["timestamp"] - t0
    if runs and runs[-1][0] == e and runs[-1][1] == i["opcode"] and t - runs[-1][3] < 1500:
        r = runs[-1]; r[3] = t + i["duration"]; r[4] += 1
    else:
        runs.append([e, i["opcode"], t, t + i["duration"], 1, (i.get("operands") or "")[:70]])
for r in runs:
    print(f"{r[2]/1000:8.2f} {r[3]/1000:8.2f} {r[0]} {r[1][:16]:16} x{r[4]:<3} {r[5]}")
