"""Dev only: compare the numpy reference (layout.ref_layer_seq + ref_head) with HF transformers (CPU, bf16) on a token
sequence; prints per-position top tokens and per-layer hidden differences. usage: check_ref.py tok,tok,..."""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import layout as L  # noqa: E402
from layout import rb  # noqa: E402

MODEL = "/mnt/nvme1/es-ds/models/gpt-oss-20b"
toks = [int(t) for t in sys.argv[1].split(",")]
S = len(toks)
from transformers import AutoModelForCausalLM, Mxfp4Config  # noqa: E402
m = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, quantization_config=Mxfp4Config(dequantize=True)).eval()
with torch.inference_mode():
    out = m(torch.tensor([toks]), output_hidden_states=True)
hf_h = [h[0].float().numpy() for h in out.hidden_states]
hf_lg = out.logits[0].float().numpy()
ck = L.Ckpt(MODEL)
cs = L.rope_table(MODEL, 64)
X = rb(ck.get("model.embed_tokens.weight")[toks].astype(np.float32))
print("embed diff", np.abs(X - hf_h[0]).max())
nlayers = int(os.environ.get("NL", "24"))
for l in range(nlayers):
    X, _, _ = L.ref_layer_seq(ck, l, X, cs)
    d = np.linalg.norm(X - hf_h[l + 1], axis=1) / np.linalg.norm(hf_h[l + 1], axis=1)
    print(f"layer {l}: rel diff per pos", np.round(d, 4))
if nlayers == 24:
    W = ck.get("lm_head.weight")
    for t in range(S):
        tok, lg = L.ref_head(ck, X[t], W)
        print(t, "ref", tok, "hf", int(hf_lg[t].argmax()), "hf top5", np.argsort(-hf_lg[t])[:5])
