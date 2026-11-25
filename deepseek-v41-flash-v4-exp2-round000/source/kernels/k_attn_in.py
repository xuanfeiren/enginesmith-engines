"""Kernel: attention input stage of one layer for one decode token (the host copies the kv row output into the
window ring; the kernel has no dynamic-offset DMA: such DMAs could read their offset before it was written).

hc_mixes(attn) + hc_pre(pre_mix) + attn_norm + wq_a + q_norm + wq_b + RoPE(q) + wkv + kv_norm + RoPE(kv) +
act_quant(kv) + window-cache write at slot pos % 128.
"""
import nki
import nki.language as nl
import nki.isa as nisa

import lib
import blocks as B
from lib import P, F32, BF16, I32, sb, ps

QL = 1280
NH = 64
HD = 512
WIN = 128


@nki.jit
def attn_in(h, premix, fnT, hsc, hbs, anw, wqa, qnw, wqb, wkv, kvw, rotP, ropec, roper):
    ident = lib.identity_sb(P)
    ones40 = lib.ones_sb(B.NR, B.NR)
    ones1 = lib.ones_sb(1, B.NR)

    h_r = B.load_h_r(h)
    mix = B.hc_mixes(h_r, fnT, hsc, hbs, ident, ones40)
    mix_out = nl.ndarray((1, 24), dtype=F32, buffer=nl.shared_hbm)
    nisa.dma_copy(dst=mix_out, src=mix)

    pm = B.bcast_row(premix, B.NR, 4)
    x = B.hc_pre(h_r, pm)
    g = sb((B.NR, P), BF16)
    nisa.dma_copy(dst=g, src=anw)
    xn = lib.rmsnorm_r(x, g, B.D, B.EPS, ones40)
    xn_out = nl.ndarray((B.NR, P), dtype=BF16, buffer=nl.shared_hbm)
    nisa.dma_copy(dst=xn_out, src=xn)

    xq = lib.act_quant_r(xn)
    xc = lib.r_to_c(xq, ident)
    # q path
    qa = lib.gemv_c(xc, wqa, QL, B.D)                       # C [128, 10]
    qa_r = lib.c_to_r(qa, ident)                              # [10, 128]
    gq = sb((QL // P, P), BF16)
    nisa.dma_copy(dst=gq, src=qnw)
    ones10 = lib.ones_sb(QL // P, QL // P)
    qr = lib.rmsnorm_r(qa_r, gq, QL, B.EPS, ones10)
    qr_out = nl.ndarray((QL // P, P), dtype=BF16, buffer=nl.shared_hbm)
    nisa.dma_copy(dst=qr_out, src=qr)
    qrq = lib.act_quant_r(qr)
    qrc = lib.r_to_c(qrq, ident)
    q = lib.gemv_c(qrc, wqb, NH * HD, QL)                     # C [128, 256], column = h*4 + c
    rot = sb((P, P), BF16)
    nisa.dma_copy(dst=rot, src=rotP)
    cosc, sinc = B.load_rope_c(ropec)
    qv = q[:, 3:NH * 4:4]
    B.rope_c_cols(qv, rot, cosc, sinc, qv)
    q_out = nl.ndarray((P, NH * 4), dtype=BF16, buffer=nl.shared_hbm)
    nisa.dma_copy(dst=q_out, src=q)
    # kv path
    kv = lib.gemv_c(xc, wkv, HD, B.D)                        # C [128, 4]
    kv_r = lib.c_to_r(kv, ident)                              # [4, 128]
    gk = sb((4, P), BF16)
    nisa.dma_copy(dst=gk, src=kvw)
    ones4 = lib.ones_sb(4, 4)
    kvn = lib.rmsnorm_r(kv_r, gk, HD, B.EPS, ones4)
    cs = B.load_rope_r(roper, 4, 3, 4)
    B.rope_r_row(kvn, 0, 4, cs)
    kvq = lib.act_quant_r(kvn)
    kv_out = nl.ndarray((4, P), dtype=BF16, buffer=nl.shared_hbm)
    nisa.dma_copy(dst=kv_out, src=kvq)
    return mix_out, xn_out, qr_out, q_out, kv_out
