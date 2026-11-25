"""Reference (numpy) implementation of the host-side weight re-layout; the C++ host does the same.
Tests use it to feed real weights to single kernels."""
import numpy as np


def tiles(W):
    """W [N, K] (HF out x in) -> [128, (N/128)*(K/128)*128], tile (nt, kt) row-major,
    out[p, (nt*KTn + kt)*128 + j] = W[nt*128 + j, kt*128 + p]."""
    N, K = W.shape
    return np.ascontiguousarray(W.reshape(N // 128, 128, K // 128, 128).transpose(3, 0, 2, 1)).reshape(128, -1)


def layer_weights(get, l):
    """get(name) -> numpy array (bf16 as ml_dtypes); returns dict of kernel inputs for layer l."""
    p = f"language_model.model.layers.{l}."
    wq, wk, wv = get(p + "self_attn.q_proj.weight"), get(p + "self_attn.k_proj.weight"), get(p + "self_attn.v_proj.weight")
    wo = get(p + "self_attn.o_proj.weight")
    wg, wu, wd = get(p + "mlp.gate_proj.weight"), get(p + "mlp.up_proj.weight"), get(p + "mlp.down_proj.weight")
    out = {}
    out["wqkv"] = np.stack([tiles(np.concatenate([wq[1024 * c:1024 * (c + 1)], wk[512 * c:512 * (c + 1)],
                                                  wv[512 * c:512 * (c + 1)]])) for c in range(2)])
    wos = []
    for c in range(2):
        cols = []
        for kt in range(8):
            dt, hl = kt // 4, kt % 4
            f0 = (4 * c + hl) * 256 + dt * 128
            cols.append(wo[:, f0:f0 + 128])
        wos.append(tiles(np.concatenate(cols, axis=1)))
    out["wo"] = np.stack(wos)
    gus = []
    for c in range(2):
        rows = []
        for nt in range(40):
            r0 = 5120 * c + nt * 128
            rows.append(wg[r0:r0 + 128])
            rows.append(wu[r0:r0 + 128])
        gus.append(tiles(np.concatenate(rows)))
    out["wgu"] = np.stack(gus)
    out["wd"] = np.stack([tiles(wd[:, 5120 * c:5120 * (c + 1)]) for c in range(2)])
    out["lnw"] = np.stack([get(p + n + ".weight") for n in
                           ["input_layernorm", "post_attention_layernorm", "pre_feedforward_layernorm",
                            "post_feedforward_layernorm"]])
    out["qkw"] = np.stack([get(p + "self_attn.q_norm.weight"), get(p + "self_attn.k_norm.weight")])
    return out


def rope_tables(theta, factor, S):
    """cos, sin [128, S] as bf16 bits (uint16), computed exactly like HF Gemma3RotaryEmbedding (torch, fp32)."""
    import torch
    inv = 1.0 / (theta ** (torch.arange(0, 256, 2, dtype=torch.int64).to(dtype=torch.float) / 256))
    inv = inv / factor
    pos = torch.arange(S, dtype=torch.float)[None, None, :]
    fr = (inv[None, :, None].float() @ pos).transpose(1, 2)[0]        # [S, 128]
    emb = torch.cat((fr, fr), dim=-1)
    cos, sin = emb.cos().to(torch.bfloat16), emb.sin().to(torch.bfloat16)
    return (cos[:, :128].T.contiguous().view(torch.int16).numpy().view(np.uint16),
            sin[:, :128].T.contiguous().view(torch.int16).numpy().view(np.uint16))


def head_weights(emb, bf16):
    """emb [262208, 2560] -> embT [2, 128, 1025*20*128], lmask, lids [2, 128, 1025] fp32."""
    V = emb.shape[0]
    VPC = 131200
    embT, mask, ids = [], [], []
    for c in range(2):
        rows = emb[VPC * c:VPC * (c + 1)]
        if rows.shape[0] < VPC:
            rows = np.concatenate([rows, np.zeros((VPC - rows.shape[0], emb.shape[1]), emb.dtype)])
        embT.append(tiles(rows))
        gid = VPC * c + np.arange(1025)[None, :] * 128 + np.arange(128)[:, None]
        mask.append(np.where(gid < V, 0.0, -1e30).astype(np.float32))
        ids.append((1048576.0 - gid).astype(np.float32))
    return np.stack(embT), np.stack(mask), np.stack(ids)
