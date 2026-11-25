"""Kernels: KV compressor of the kv-source layers (2, 8, 14: ratio 2; 20: ratio 1) for one decode token, with the
indexer's key path (wk, k_norm, RoPE, fp4 quant) and the compressed-KV path (RoPE, fp4 quant), cache writes.

Ratio 2 runs as two variants chosen by the host from the position parity: `store` (pos even: keep kv/score of slot
0 in the fp32 state) and `emit` (pos odd: slot 1 is the current token, softmax-pool both slots, then the rest).
"""
import nki
import nki.language as nl
import nki.isa as nisa

import lib
import blocks as B
from lib import P, F32, BF16, I32, sb, ps

HD = 512
ID = 128


def gemv_f32(xc, wt, N, K):
    """y = W x with fp32 output kept unrounded (F.linear in fp32 of bf16-valued operands). C [128, N/128] fp32."""
    return lib.gemv_c(xc, wt, N, K, out_dtype=F32)


def finish_sb(latent_r, wk, knw, roperc, ident):
    """latent_r [4, 128] bf16 -> (index key [1, 128], compressed kv [4, 128]) bf16 SBUF rows."""
    lc = lib.r_to_c(latent_r, ident)
    k = lib.gemv_c(lc, wk, ID, HD)
    kr = lib.c_to_r(k, ident)
    g = sb((1, P), BF16)
    nisa.dma_copy(dst=g, src=knw)
    ones1 = lib.ones_sb(1, 1)
    kn = lib.rmsnorm_r(kr, g, ID, B.EPS, ones1)
    cs1 = B.load_rope_r(roperc, 1)
    B.rope_r_row(kn, 0, 1, cs1)
    kq = lib.fp4_act_quant_r(kn, 32, False)
    lr = sb((4, P), BF16)
    nisa.tensor_copy(dst=lr, src=latent_r)
    cs4 = B.load_rope_r(roperc, 4, 3, 4)
    B.rope_r_row(lr, 0, 4, cs4)
    lq = lib.fp4_act_quant_r(lr, 16, True)
    return kq, lq


def pool2_sb(kv0, sc0, kv1, sc1):
    """pool2 with slot 0 from SBUF (the previous token of a multi-token group)."""
    m = sb((P, 4), F32)
    nisa.tensor_tensor(dst=m, data1=sc0, data2=sc1, op=nl.maximum)
    e0 = sb((P, 4), F32)
    nisa.tensor_tensor(dst=e0, data1=sc0, data2=m, op=nl.subtract)
    lib.exp_precise(e0, e0)
    e1 = sb((P, 4), F32)
    nisa.tensor_tensor(dst=e1, data1=sc1, data2=m, op=nl.subtract)
    lib.exp_precise(e1, e1)
    den = sb((P, 4), F32)
    nisa.tensor_tensor(dst=den, data1=e0, data2=e1, op=nl.add)
    rd = sb((P, 4), F32)
    lib.recip_precise(rd, den)
    nisa.tensor_tensor(dst=e0, data1=e0, data2=rd, op=nl.multiply)
    nisa.tensor_tensor(dst=e1, data1=e1, data2=rd, op=nl.multiply)
    a = sb((P, 4), F32)
    nisa.tensor_tensor(dst=a, data1=kv0, data2=e0, op=nl.multiply)
    b = sb((P, 4), F32)
    nisa.tensor_tensor(dst=b, data1=kv1, data2=e1, op=nl.multiply)
    pooled = sb((P, 4), BF16)
    nisa.tensor_tensor(dst=pooled, data1=a, data2=b, op=nl.add)
    return pooled


def finish_sb_r(latent_r, wk, knw, roperc, ident):
    return finish_sb(latent_r, wk, knw, roperc, ident)


def pool2(kv1, sc1, state):
    """ratio 2: softmax over the 2 slots (slot 0 from the fp32 state, slot 1 = this token) and weighted sum.
    Returns bf16 C [128, 4]."""
    kv0 = sb((P, 4), F32)
    nisa.dma_copy(dst=kv0, src=state[0])
    sc0 = sb((P, 4), F32)
    nisa.dma_copy(dst=sc0, src=state[1])
    m = sb((P, 4), F32)
    nisa.tensor_tensor(dst=m, data1=sc0, data2=sc1, op=nl.maximum)
    e0 = sb((P, 4), F32)
    nisa.tensor_tensor(dst=e0, data1=sc0, data2=m, op=nl.subtract)
    lib.exp_precise(e0, e0)
    e1 = sb((P, 4), F32)
    nisa.tensor_tensor(dst=e1, data1=sc1, data2=m, op=nl.subtract)
    lib.exp_precise(e1, e1)
    den = sb((P, 4), F32)
    nisa.tensor_tensor(dst=den, data1=e0, data2=e1, op=nl.add)
    rd = sb((P, 4), F32)
    lib.recip_precise(rd, den)
    nisa.tensor_tensor(dst=e0, data1=e0, data2=rd, op=nl.multiply)
    nisa.tensor_tensor(dst=e1, data1=e1, data2=rd, op=nl.multiply)
    a = sb((P, 4), F32)
    nisa.tensor_tensor(dst=a, data1=kv0, data2=e0, op=nl.multiply)
    b = sb((P, 4), F32)
    nisa.tensor_tensor(dst=b, data1=kv1, data2=e1, op=nl.multiply)
    pooled = sb((P, 4), BF16)
    nisa.tensor_tensor(dst=pooled, data1=a, data2=b, op=nl.add)
    return pooled


def finish(latent_r, wk, knw, roperc, ident):
    """latent_r [4, 128] bf16 (compressor output, pre-RoPE).  Returns the new rows as HBM outputs:
    the index key (k_norm(wk(latent)), RoPE, fp4 E8M0/32) [1, 128] and the compressed kv (RoPE, fp4 E4M3/16)
    [4, 128]; the host copies them into the caches (no dynamic-offset DMA in kernels)."""
    lc = lib.r_to_c(latent_r, ident)                               # C [128, 4]
    k = lib.gemv_c(lc, wk, ID, HD)                                  # C [128, 1] bf16
    kr = lib.c_to_r(k, ident)                                       # [1, 128]
    g = sb((1, P), BF16)
    nisa.dma_copy(dst=g, src=knw)
    ones1 = lib.ones_sb(1, 1)
    kn = lib.rmsnorm_r(kr, g, ID, B.EPS, ones1)
    cs1 = B.load_rope_r(roperc, 1)
    B.rope_r_row(kn, 0, 1, cs1)
    kq = lib.fp4_act_quant_r(kn, 32, False)
    lr = sb((4, P), BF16)
    nisa.tensor_copy(dst=lr, src=latent_r)
    cs4 = B.load_rope_r(roperc, 4, 3, 4)
    B.rope_r_row(lr, 0, 4, cs4)
    lq = lib.fp4_act_quant_r(lr, 16, True)
    k_out = nl.ndarray((1, P), dtype=BF16, buffer=nl.shared_hbm)
    nisa.dma_copy(dst=k_out, src=kq)
    c_out = nl.ndarray((4, P), dtype=BF16, buffer=nl.shared_hbm)
    nisa.dma_copy(dst=c_out, src=lq)
    return k_out, c_out


@nki.jit
def compress1(xn, wkv, cnw, wk, knw, roperc):
    """ratio 1: latent = norm(wkv(x)) (bf16 linear)."""
    ident = lib.identity_sb(P)
    xr = sb((B.NR, P), BF16)
    nisa.dma_copy(dst=xr, src=xn)
    xc = lib.r_to_c(xr, ident)
    kv = lib.gemv_c(xc, wkv, HD, B.D)                              # bf16 C [128, 4]
    kvr = lib.c_to_r(kv, ident)
    g = sb((4, P), BF16)
    nisa.dma_copy(dst=g, src=cnw)
    ones4 = lib.ones_sb(4, 4)
    lat = lib.rmsnorm_r(kvr, g, HD, B.EPS, ones4)
    return finish(lat, wk, knw, roperc, ident)


def kv_score(xn, wkv, wgate, ident):
    xr = sb((B.NR, P), BF16)
    nisa.dma_copy(dst=xr, src=xn)
    xc = lib.r_to_c(xr, ident)
    kv = gemv_f32(xc, wkv, HD, B.D)                                # fp32 C [128, 4]
    sc = gemv_f32(xc, wgate, HD, B.D)
    return kv, sc


@nki.jit
def compress2_store(xn, wkv, wgate, state):
    """ratio 2, even position: state[0] = kv, state[1] = score (slot 0), fp32 C layout [128, 4] each."""
    ident = lib.identity_sb(P)
    kv, sc = kv_score(xn, wkv, wgate, ident)
    nisa.dma_copy(dst=state[0], src=kv)
    nisa.dma_copy(dst=state[1], src=sc)
    o = nl.ndarray((1, 8), dtype=F32, buffer=nl.shared_hbm)
    t = sb((1, 8), F32)
    nisa.memset(dst=t, value=0.0)
    nisa.dma_copy(dst=o, src=t)
    return o


@nki.jit
def compress2_emit(xn, wkv, wgate, state, cnw, wk, knw, roperc):
    """ratio 2, odd position: pool slot 0 (state) and slot 1 (this token) with a softmax over the slots."""
    ident = lib.identity_sb(P)
    kv1, sc1 = kv_score(xn, wkv, wgate, ident)
    kv0 = sb((P, 4), F32)
    nisa.dma_copy(dst=kv0, src=state[0])
    sc0 = sb((P, 4), F32)
    nisa.dma_copy(dst=sc0, src=state[1])
    m = sb((P, 4), F32)
    nisa.tensor_tensor(dst=m, data1=sc0, data2=sc1, op=nl.maximum)
    e0 = sb((P, 4), F32)
    nisa.tensor_tensor(dst=e0, data1=sc0, data2=m, op=nl.subtract)
    lib.exp_precise(e0, e0)
    e1 = sb((P, 4), F32)
    nisa.tensor_tensor(dst=e1, data1=sc1, data2=m, op=nl.subtract)
    lib.exp_precise(e1, e1)
    den = sb((P, 4), F32)
    nisa.tensor_tensor(dst=den, data1=e0, data2=e1, op=nl.add)
    rd = sb((P, 4), F32)
    lib.recip_precise(rd, den)
    nisa.tensor_tensor(dst=e0, data1=e0, data2=rd, op=nl.multiply)
    nisa.tensor_tensor(dst=e1, data1=e1, data2=rd, op=nl.multiply)
    a = sb((P, 4), F32)
    nisa.tensor_tensor(dst=a, data1=kv0, data2=e0, op=nl.multiply)
    b = sb((P, 4), F32)
    nisa.tensor_tensor(dst=b, data1=kv1, data2=e1, op=nl.multiply)
    pooled = sb((P, 4), BF16)
    nisa.tensor_tensor(dst=pooled, data1=a, data2=b, op=nl.add)
    pr = lib.c_to_r(pooled, ident)                                  # [4, 128] bf16
    g = sb((4, P), BF16)
    nisa.dma_copy(dst=g, src=cnw)
    ones4 = lib.ones_sb(4, 4)
    lat = lib.rmsnorm_r(pr, g, HD, B.EPS, ones4)
    return finish(lat, wk, knw, roperc, ident)
