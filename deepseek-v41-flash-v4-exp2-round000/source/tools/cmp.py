import json, sys
a, b = json.load(open(sys.argv[1])), json.load(open(sys.argv[2]))
bad = 0
for x, y in zip(a, b):
    same = x['tokens'] == y['tokens']
    first = next((i for i, (p, q) in enumerate(zip(x['tokens'], y['tokens'])) if p != q), None)
    print(x['len'], x['off'], x['rep'], 'SAME' if same else f'DIFF at {first}', f"pf {x['prefill_ms']:.0f}/{y['prefill_ms']:.0f} dec {x['decode_ms']:.0f}/{y['decode_ms']:.0f}")
    bad += not same
print('bad', bad)
