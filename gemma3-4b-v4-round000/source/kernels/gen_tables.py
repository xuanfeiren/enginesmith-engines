"""Build time only: RoPE cos/sin tables for Gemma 3, computed with torch exactly as transformers'
Gemma3RotaryEmbedding does (fp32 inverse frequencies, fp32 position product, cos/sin in fp32, cast to bf16).

Writes OUT: local cos, local sin, global cos, global sin, each [128, S] bf16 (row d = rotary pair d; the two
halves of the 256 rotary dims share it), little-endian, concatenated.
"""
import sys

import numpy as np
import torch

S = 8192
DIM = 256


def tables(theta, factor):
    inv_freq = 1.0 / (theta ** (torch.arange(0, DIM, 2, dtype=torch.int64).to(dtype=torch.float) / DIM))
    inv_freq = inv_freq / factor
    pos = torch.arange(S)[None, :]
    inv_e = inv_freq[None, :, None].float().expand(1, -1, 1)
    pos_e = pos[:, None, :].float()
    freqs = (inv_e @ pos_e).transpose(1, 2)              # [1, S, 128]
    emb = torch.cat((freqs, freqs), dim=-1)
    cos = (emb.cos() * 1.0).to(torch.bfloat16)[0]
    sin = (emb.sin() * 1.0).to(torch.bfloat16)[0]
    assert torch.equal(cos[:, :128], cos[:, 128:]) and torch.equal(sin[:, :128], sin[:, 128:])
    f = lambda t: t[:, :128].T.contiguous().view(torch.int16).numpy().view(np.uint16)
    return f(cos), f(sin)


def main():
    lc, ls = tables(10000.0, 1.0)
    gc, gs = tables(1000000.0, 8.0)
    with open(sys.argv[1], "wb") as fh:
        for t in (lc, ls, gc, gs):
            fh.write(np.ascontiguousarray(t).tobytes())


if __name__ == "__main__":
    main()
