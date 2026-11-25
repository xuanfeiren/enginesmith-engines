"""Build-/test-time only: numpy versions of the engine's weight layouts and a bf16-emulating numpy reference.

The C++ engine (src/engine.cpp) produces the same layouts at load time; tools/test_*.py check kernels against
the reference here.
"""
import json
import math
import os

import ml_dtypes
import numpy as np

bf16 = ml_dtypes.bfloat16
f8 = ml_dtypes.float8_e5m2  # expert weights (exact, see src/engine.cpp mx_row)
H, P, HT, D, NQ, E = 2880, 120, 24, 64, 8, 32
I = 2880
FP4 = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0], np.float32)


def rb(x):
    """round to bf16, back to fp32"""
    return np.asarray(x, np.float32).astype(bf16).astype(np.float32)


class Ckpt:
    def __init__(self, d):
        from safetensors import safe_open
        self.d = d
        self.idx = json.load(open(os.path.join(d, "model.safetensors.index.json")))["weight_map"]
        self.files = {}
        self.cfg = json.load(open(os.path.join(d, "config.json")))

    def get(self, k):
        from safetensors import safe_open
        f = self.idx[k]
        if f not in self.files:
            self.files[f] = safe_open(os.path.join(self.d, f), "np")
        t = self.files[f].get_slice(k)
        a = t[:]
        if t.get_dtype() == "BF16":
            a = a.view(bf16) if a.dtype == np.uint16 else a
        return a

    def f32(self, k):
        a = self.get(k)
        return a.astype(np.float32)

    def experts(self, l, name):
        """dequantized [E, rows, cols] fp32 (rows = output features)"""
        blk = self.get(f"model.layers.{l}.mlp.experts.{name}_blocks")  # [E, R, G, 16] u8
        sc = self.get(f"model.layers.{l}.mlp.experts.{name}_scales").astype(np.int32) - 127  # [E, R, G]
        lo = FP4[blk & 0x0F]
        hi = FP4[blk >> 4]
        v = np.stack([lo, hi], axis=-1).reshape(*blk.shape[:-1], 32)
        v = v * np.ldexp(np.float32(1.0), sc)[..., None]
        return v.reshape(blk.shape[0], blk.shape[1], -1).astype(np.float32)


def rope_table(cfg, maxpos):
    """[maxpos, 128] fp32: cos (64, the 32 freqs twice) | sin (64), each rounded to bf16 as HF does."""
    import torch
    from transformers import AutoConfig
    from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS
    c = AutoConfig.from_pretrained(cfg)
    inv_freq, scale = ROPE_INIT_FUNCTIONS["yarn"](c, None)
    pos = torch.arange(maxpos, dtype=torch.float32)
    freqs = (inv_freq[None, :, None].float() @ pos[None, None, :]).transpose(1, 2)[0]  # [maxpos, 32]
    cos = (freqs.cos() * scale).to(torch.bfloat16).float().numpy()
    sin = (freqs.sin() * scale).to(torch.bfloat16).float().numpy()
    return np.concatenate([cos, cos, sin, sin], axis=1).astype(np.float32)


def rmat():
    R = np.zeros((D, D), np.float32)
    for d in range(32):
        R[d, d + 32] = -1.0
        R[d + 32, d] = 1.0
    return R.T.astype(bf16)  # stationary = R^T


def col(v):
    return np.ascontiguousarray(np.asarray(v, np.float32).reshape(HT, P).T)


def layer_rank_arrays(ck, l, rank):
    """per-rank kernel inputs for layer l (weights only)."""
    pre = f"model.layers.{l}."
    qw, kw, vw = (ck.get(pre + f"self_attn.{n}_proj.weight") for n in "qkv")
    qb, kb, vb = (ck.f32(pre + f"self_attn.{n}_proj.bias") for n in "qkv")
    ow = ck.get(pre + "self_attn.o_proj.weight")
    out = {}
    wqkv, bqkv, wo, sinks = [], [], [], []
    sk = ck.f32(pre + "self_attn.sinks")
    for c in range(2):
        g = 2 * rank + c
        Wc = np.concatenate([qw[512 * g:512 * g + 512], kw[64 * g:64 * g + 64], vw[64 * g:64 * g + 64]], 0)
        wqkv.append(Wc.reshape(10, 64, HT, P).transpose(3, 2, 0, 1).reshape(P, -1))  # [p][kt][t][n]
        bc = np.concatenate([qb[512 * g:512 * g + 512], kb[64 * g:64 * g + 64], vb[64 * g:64 * g + 64]])
        bqkv.append(bc.reshape(10, 64).T)
        wo.append(ow[:, 512 * g:512 * g + 512].reshape(HT, P, NQ, D).transpose(3, 2, 0, 1).reshape(D, -1))
        sinks.append(sk[8 * g:8 * g + 8].reshape(8, 1))
    out["wqkv"] = np.ascontiguousarray(np.stack(wqkv)).astype(bf16)
    out["bqkv"] = np.ascontiguousarray(np.stack(bqkv)).astype(np.float32)
    out["wo"] = np.ascontiguousarray(np.stack(wo)).astype(bf16)
    out["wo2"] = np.ascontiguousarray(np.stack([w.reshape(D, 4, 2, H).transpose(2, 0, 1, 3).reshape(128, -1) for w in wo])).astype(bf16)
    out["sinks"] = np.ascontiguousarray(np.stack(sinks)).astype(np.float32)
    out["bo"] = col(ck.f32(pre + "self_attn.o_proj.bias"))
    out["ln1"] = col(ck.f32(pre + "input_layernorm.weight"))
    out["ln2"] = col(ck.f32(pre + "post_attention_layernorm.weight"))
    wr = ck.get(pre + "mlp.router.weight")
    out["wr"] = np.ascontiguousarray(wr.reshape(E, HT, P).transpose(2, 1, 0).reshape(P, -1)).astype(bf16)
    out["br"] = ck.f32(pre + "mlp.router.bias").reshape(1, E)
    GU = ck.experts(l, "gate_up_proj")  # [E, 5760, 2880]
    DN = ck.experts(l, "down_proj")     # [E, 2880, 2880]
    gub = ck.f32(pre + "mlp.experts.gate_up_proj_bias")  # [E, 5760]
    wgu = np.zeros((E, 2, P, 6 * HT * P), f8)
    bgu = np.zeros((E, 2, P, 6), np.float32)
    wd = np.zeros((E, 2, P, 3 * HT * P), f8)
    for c in range(2):
        g = 2 * rank + c
        j = 360 * g + np.arange(360)
        rows = np.concatenate([2 * j, 2 * j + 1])  # (t, n) order: gate tiles 0..2 then up tiles 3..5
        sel = GU[:, rows, :]  # [E, 720, 2880]
        wgu[:, c] = sel.reshape(E, 6, P, HT, P).transpose(0, 4, 3, 1, 2).reshape(E, P, -1).astype(f8)  # [p][kt][t][n]
        bgu[:, c] = gub[:, rows].reshape(E, 6, P).transpose(0, 2, 1)
        dsel = DN[:, :, 360 * g:360 * g + 360]  # [E, 2880 h, 360 i]
        wd[:, c] = dsel.reshape(E, HT, P, 3, P).transpose(0, 4, 3, 1, 2).reshape(E, P, -1).astype(f8)
    out["wgu"] = wgu.reshape(E, -1)
    out["bgu"] = bgu.reshape(E, -1)
    out["wd"] = wd.reshape(E, -1)
    out["bd"] = ck.f32(pre + "mlp.experts.down_proj_bias")
    return out


# ---------------------------------------------------------------- numpy reference (HF bf16 rounding points)

def ref_rmsnorm(x, w):
    x = x.astype(np.float32)
    v = (x * x).mean()
    return rb(w * (x / np.sqrt(v + 1e-5)))


def ref_layer(ck, l, h, pos, kcache, vcache, cs):
    """h [H] (bf16 values); kcache/vcache [8, pos, 64] float (bf16 values) of earlier positions (already
    roped). Returns h_out and the new k, v [8, 64]."""
    pre = f"model.layers.{l}."
    sliding = ck.cfg["layer_types"][l] == "sliding_attention"
    xn = ref_rmsnorm(h, ck.f32(pre + "input_layernorm.weight"))
    q = rb(ck.f32(pre + "self_attn.q_proj.weight") @ xn + ck.f32(pre + "self_attn.q_proj.bias")).reshape(64, 64)
    k = rb(ck.f32(pre + "self_attn.k_proj.weight") @ xn + ck.f32(pre + "self_attn.k_proj.bias")).reshape(8, 64)
    v = rb(ck.f32(pre + "self_attn.v_proj.weight") @ xn + ck.f32(pre + "self_attn.v_proj.bias")).reshape(8, 64)
    cos, sin = cs[pos, :64], cs[pos, 64:]

    def rope(x):
        x1, x2 = x[:, :32], x[:, 32:]
        return rb(np.concatenate([x1 * cos[:32] - x2 * sin[:32], x2 * cos[32:] + x1 * sin[32:]], 1))
    q, k = rope(q), rope(k)
    K = np.concatenate([kcache, k[:, None, :]], 1)  # [8, pos+1, 64]
    V = np.concatenate([vcache, v[:, None, :]], 1)
    t0 = max(0, pos - 127) if sliding else 0
    sinks = ck.f32(pre + "self_attn.sinks")
    out = np.zeros((64, 64), np.float32)
    for hq in range(64):
        kv = hq // 8
        s = rb(K[kv, t0:] @ q[hq]) * 0.125
        m = max(s.max(), sinks[hq])
        e = np.exp(s - m)
        tot = e.sum() + np.exp(sinks[hq] - m)
        p = rb(e / tot)
        out[hq] = rb(p @ V[kv, t0:])
    attn = rb(ck.f32(pre + "self_attn.o_proj.weight") @ out.reshape(-1) + ck.f32(pre + "self_attn.o_proj.bias"))
    h1 = rb(h + attn)
    xn2 = ref_rmsnorm(h1, ck.f32(pre + "post_attention_layernorm.weight"))
    lg = rb(ck.f32(pre + "mlp.router.weight") @ xn2 + ck.f32(pre + "mlp.router.bias"))
    order = sorted(range(E), key=lambda j: (-lg[j], j))[:4]
    tv = lg[order]
    w = rb(np.exp(tv - tv[0]) / np.exp(tv - tv[0]).sum())
    GU = ck.experts(l, "gate_up_proj")
    DN = ck.experts(l, "down_proj")
    gub = ck.f32(pre + "mlp.experts.gate_up_proj_bias")
    dnb = ck.f32(pre + "mlp.experts.down_proj_bias")
    moe = np.zeros(H, np.float32)
    for i, e in enumerate(order):
        gu = rb(GU[e] @ xn2 + gub[e])
        gate, up = gu[0::2], gu[1::2]
        gate = np.minimum(gate, 7.0)
        up = np.clip(up, -7.0, 7.0)
        glu = rb(gate * rb(1 / (1 + np.exp(-rb(gate * 1.702)))))
        inter = rb(rb(up + 1) * glu)
        y = rb(DN[e] @ inter + dnb[e])
        moe = moe + rb(y * w[i])
    h2 = rb(h1 + rb(moe))
    return h2, k, v, dict(order=order, w=w, h1=h1)


NVT, LM_G, VOCAB, BIG = 200, 4, 201088, 1048576.0


def head_rank_arrays(ck, rank, W=None):
    if W is None:
        W = ck.get("lm_head.weight")
    NG = NVT // LM_G
    wl, lb, ls = [], [], []
    for c in range(2):
        g = 2 * rank + c
        base = g * NVT * 128
        Wg = np.zeros((NVT * 128, H), bf16)
        n = max(0, min(VOCAB - base, NVT * 128))
        Wg[:n] = W[base:base + n]
        Wp = np.zeros((NVT * 128, 23 * 128), bf16)
        Wp[:, :H] = Wg
        wl.append(Wp.reshape(NG, LM_G, 128, 23, 128).transpose(0, 4, 1, 3, 2).reshape(NG, -1))  # 128-row tiles
        ids = base + np.arange(NVT)[None, :] * 128 + np.arange(128)[:, None]
        lb.append(np.where(ids < VOCAB, 0.0, -3e38).astype(np.float32))
        ls.append((BIG - ids).astype(np.float32))
    return dict(wlm=np.ascontiguousarray(np.concatenate(wl)), lmbias=np.stack(lb), lmscore=np.stack(ls),
                lnf=col(ck.f32("model.norm.weight")))


def ref_head(ck, h, W=None):
    if W is None:
        W = ck.get("lm_head.weight")
    xn = ref_rmsnorm(h, ck.f32("model.norm.weight"))
    lg = rb(W.astype(np.float32) @ xn)
    return int(np.argmax(lg)), lg


def ref_layer_seq(ck, l, X, cs):
    """Prefill reference for one layer over tokens at positions 0..S-1. X [S, H] (bf16 values).
    Returns (out [S, H], K [8, S, 64], V [8, S, 64])."""
    pre = f"model.layers.{l}."
    S = X.shape[0]
    sliding = ck.cfg["layer_types"][l] == "sliding_attention"
    def norm(Xm, w):
        v = (Xm * Xm).mean(1, keepdims=True)
        return rb(w[None, :] * (Xm / np.sqrt(v + 1e-5)))
    xn = norm(X, ck.f32(pre + "input_layernorm.weight"))
    q = rb(xn @ ck.f32(pre + "self_attn.q_proj.weight").T + ck.f32(pre + "self_attn.q_proj.bias")).reshape(S, 64, 64)
    k = rb(xn @ ck.f32(pre + "self_attn.k_proj.weight").T + ck.f32(pre + "self_attn.k_proj.bias")).reshape(S, 8, 64)
    v = rb(xn @ ck.f32(pre + "self_attn.v_proj.weight").T + ck.f32(pre + "self_attn.v_proj.bias")).reshape(S, 8, 64)
    cos, sin = cs[:S, None, :32], cs[:S, None, 64:96]
    def rope(x):
        x1, x2 = x[..., :32], x[..., 32:]
        return rb(np.concatenate([x1 * cos - x2 * sin, x2 * cos + x1 * sin], -1))
    q, k = rope(q), rope(k)
    sinks = ck.f32(pre + "self_attn.sinks")
    qi = np.arange(S)[:, None]; kj = np.arange(S)[None, :]
    valid = kj <= qi
    if sliding:
        valid &= kj > qi - 128
    out = np.zeros((S, 64, 64), np.float32)
    for hq in range(64):
        kv = hq // 8
        s = rb(q[:, hq] @ k[:, kv].T) * 0.125
        s = np.where(valid, s, -np.inf)
        m = np.maximum(s.max(1), sinks[hq])
        e = np.exp(s - m[:, None])
        tot = e.sum(1) + np.exp(sinks[hq] - m)
        p = rb(e / tot[:, None])
        out[:, hq] = rb(p @ v[:, kv])
    attn = rb(out.reshape(S, -1) @ ck.f32(pre + "self_attn.o_proj.weight").T + ck.f32(pre + "self_attn.o_proj.bias"))
    h1 = rb(X + attn)
    xn2 = norm(h1, ck.f32(pre + "post_attention_layernorm.weight"))
    lg = rb(xn2 @ ck.f32(pre + "mlp.router.weight").T + ck.f32(pre + "mlp.router.bias"))
    GU = ck.experts(l, "gate_up_proj"); DN = ck.experts(l, "down_proj")
    gub = ck.f32(pre + "mlp.experts.gate_up_proj_bias"); dnb = ck.f32(pre + "mlp.experts.down_proj_bias")
    moe = np.zeros((S, H), np.float32)
    sel = [sorted(range(E), key=lambda j: (-lg[t, j], j))[:4] for t in range(S)]
    W = np.zeros((S, E), np.float32)
    for t in range(S):
        tv = lg[t, sel[t]]
        w = rb(np.exp(tv - tv[0]) / np.exp(tv - tv[0]).sum())
        W[t, sel[t]] = w
    for e in range(E):
        toks = np.nonzero(W[:, e])[0]
        if len(toks) == 0:
            continue
        gu = rb(xn2[toks] @ GU[e].T + gub[e])
        gate, up = np.minimum(gu[:, 0::2], 7.0), np.clip(gu[:, 1::2], -7.0, 7.0)
        glu = rb(gate * rb(1 / (1 + np.exp(-rb(gate * 1.702)))))
        inter = rb(rb(up + 1) * glu)
        y = rb(inter @ DN[e].T + dnb[e])
        moe[toks] += rb(y * W[toks, e:e + 1])
    return rb(h1 + rb(moe)), k.transpose(1, 0, 2), v.transpose(1, 0, 2)
