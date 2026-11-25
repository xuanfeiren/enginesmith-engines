"""The M-token prefill MoE with every distinct routed expert read once (moe_impl_m's arithmetic per token).

A group of M prompt tokens routes to M*6 (token, expert) pairs but only ~16 (M=4) / ~26 (M=8) distinct experts.
Here each of S expert *slots* (one distinct expert each, ascending expert id; the host binds it) is applied to all M
tokens at once: the moving operand of every weight tile holds the M tokens' block-masked x ([128 k, 4M]), so each
weight tile is loaded and made stationary once per slot instead of once per pair.  The routing weight of token m in
slot s is rw[s, m] (0 when the token is not routed to that expert: its column then computes exact zeros, and adding
them to the token's fp32 accumulator changes nothing).  Each token's accumulator receives its experts in ascending
expert id order, the order of the official MoE.forward (and of the host-sorted moe4 / decode paths), so the
values are those of moe_impl_m with the token's experts sorted.  Per column every instruction is that of
k_moe.gemv_blockscaled / expert_up / expert_down: the same fp32 PSUM block sums, the same products with the block
scales, the same 160- / 72-element reductions, swiglu, act_quant per token row.
"""
import nki
import nki.language as nl
import nki.isa as nisa

import lib
import blocks as B
import k_batch as KB
import k_moe as KM
from lib import P, F32, BF16, sb, ps

FI = KM.FI
KT_D = KM.KT_D
NT_I = KM.NT_I
WBUF = KM.WBUF
SBUFN = KM.SBUFN


def block_x_m(xm, bmt, M, kt):
    """xm [128, M, KT] bf16 (C layout per token) -> X [128, KT, 4M]: X[p, k, 4m + j] = xm[p, m, k] if p // 32 == j"""
    x4 = sb((P, kt, M, 4), BF16)
    nisa.tensor_tensor(dst=x4, data1=xm.ap(pattern=[[M * kt, P], [1, kt], [kt, M], [0, 4]], offset=0),
                       data2=bmt.ap(pattern=[[4, P], [0, kt], [0, M], [1, 4]], offset=0), op=nl.multiply)
    return x4.reshape((P, kt, 4 * M))


def gemv_bs_m(x4m, M, w_hbm, s_hbm, N, K, out, rings, wbase, sbase):
    """out [128, N/128, M] fp32: per token m, column m of k_moe.gemv_blockscaled (moving x4m [128, KT, 4M]).
    Returns the number of weight tiles used."""
    KT, NT, NB = K // P, N // P, K // 32
    W = 4 * M
    kpb = 512 // W                      # kt per PSUM bank
    nbank = (KT + kpb - 1) // kpb
    n0, n1 = lib.lnc_split(NT)
    sring = rings[1][sbase % SBUFN]
    SROW = sring.shape[1]
    lib.wdma(sring[:, 0:(n1 - n0) * NB], s_hbm[:, n0 * NB:n1 * NB])
    for nt in range(n0, n1):
        w = rings[0][(wbase + nt - n0) % WBUF][:, 0:KT * P]
        lib.wdma(w, w_hbm[:, nt * KT * P:(nt + 1) * KT * P])
        prod = sb((P, M * NB), F32)          # [P, M, KT, 4]: each token's block products contiguous, kt-major
        for bi in range(nbank):
            k0 = bi * kpb
            nk = min(KT, k0 + kpb) - k0
            pt = nl.ndarray((P, 512), dtype=F32, buffer=nl.psum)
            for i in range(nk):
                kt = k0 + i
                nisa.nc_matmul(dst=pt[:, i * W:(i + 1) * W], stationary=w[:, kt * P:(kt + 1) * P],
                               moving=x4m[:, kt, :], accumulate=False)
            nisa.tensor_tensor(dst=prod.ap(pattern=[[M * NB, P], [4, nk], [NB, M], [1, 4]], offset=k0 * 4),
                               data1=pt.ap(pattern=[[512, P], [W, nk], [4, M], [1, 4]], offset=0),
                               data2=sring.ap(pattern=[[SROW, P], [4, nk], [0, M], [1, 4]],
                                              offset=(nt - n0) * NB + k0 * 4),
                               op=nl.multiply)
        nisa.tensor_reduce(dst=out[:, nt, :], op=nl.add, data=prod.reshape((P, M, NB)), axis=2)
    if nl.num_programs() == 2 and NT % 2 == 0 and NT >= 2:
        lib.exchange_halves(out.reshape((P, NT * M)), NT * M)
    return n1 - n0


def perm_copy(dst, src, A, Bn):
    """dst [P, Bn, A] = src [P, A, Bn] transposed in the free dims (one copy; dtype conversion as tensor_copy)"""
    nisa.tensor_copy(dst=dst.ap(pattern=[[A * Bn, P], [1, A], [A, Bn]], offset=0),
                     src=src.ap(pattern=[[A * Bn, P], [Bn, A], [1, Bn]], offset=0))


def slot_up(x4m, M, w1, s1, w3, s3, wrow, bmt, ident, rings, wc, gc):
    """W1 / W3 of one slot for the M tokens, swiglu (* the tokens' routing weights), act_quant per token, block
    mask: the down projection's moving operand [128, NT_I, 4M].  Returns (a4m, tiles used)."""
    g1 = sb((P, NT_I, M), F32)
    t = gemv_bs_m(x4m, M, w1, s1, FI, B.D, g1, rings, wc, gc)
    g3 = sb((P, NT_I, M), F32)
    t = t + gemv_bs_m(x4m, M, w3, s3, FI, B.D, g3, rings, wc + t, gc + 1)
    g1b = sb((P, M * NT_I), BF16)
    perm_copy(g1b, g1, NT_I, M)
    g3b = sb((P, M * NT_I), BF16)
    perm_copy(g3b, g3, NT_I, M)
    n = M * NT_I
    gf = sb((P, n), F32)
    nisa.tensor_scalar(dst=gf, data=g1b, op0=nl.minimum, operand0=KM.SWIGLU)
    uf = sb((P, n), F32)
    nisa.tensor_scalar(dst=uf, data=g3b, op0=nl.minimum, operand0=KM.SWIGLU, op1=nl.maximum,
                       operand1=-KM.SWIGLU)
    sg = sb((P, n), F32)
    lib.sigmoid_precise(sg, gf)
    a = sb((P, n), F32)
    nisa.tensor_tensor(dst=a, data1=gf, data2=sg, op=nl.multiply)
    nisa.tensor_tensor(dst=a, data1=a, data2=uf, op=nl.multiply)
    ab = sb((P, M, NT_I), BF16)
    nisa.tensor_tensor(dst=ab, data1=a.reshape((P, M, NT_I)), data2=wrow.reshape((P, M, 1)).broadcast(2, NT_I),
                       op=nl.multiply)
    acm = sb((P, M, NT_I), BF16)
    TPC = min(M, P // NT_I)          # tokens per transpose (rows NT_I each)
    for m0 in range(0, M, TPC):
        m1 = min(M, m0 + TPC)
        ar = lib.c_to_r(ab.reshape((P, M * NT_I))[:, m0 * NT_I:m1 * NT_I], ident)
        aq = lib.act_quant_r(ar)
        lib.transpose_to(acm.reshape((P, M * NT_I))[:, m0 * NT_I:m1 * NT_I], aq, ident)
    return block_x_m(acm, bmt, M, NT_I), t


def slot_down(a4m, M, w2, s2, rings, wc, gc):
    """W2 of one slot: y2b [128, M, 40] bf16 (per token moe_impl's y2b)"""
    y2 = sb((P, KT_D, M), F32)
    t = gemv_bs_m(a4m, M, w2, s2, B.D, FI, y2, rings, wc, gc)
    y2b = sb((P, M, KT_D), BF16)
    perm_copy(y2b, y2, KT_D, M)
    return y2b, t


def moeu_impl(M, S, x, h2, rw, bm, we, se, s2in, fnTf, hscf, hbsf):
    """x [M, 40, 128] bf16, h2 [M, 4, 5120], rw [S, M] fp32 (slot s, token m), we/se: the S slots' expert tensors.
    Outputs h_out [M, 4, 5120] bf16 and the ffn mixes [M, 24] fp32, as moe_impl_m."""
    ident = lib.identity_sb(P)
    ones1 = lib.ones_sb(1, P)
    ones40 = lib.ones_sb(1, B.NR)
    bmt = sb((P, 4), BF16)
    nisa.dma_copy(dst=bmt, src=bm)
    xqm = sb((P, M, KT_D), BF16)
    xqcs = []
    for m in range(M):     # per token on its owner physical core (lib.tok_mine), then exchanged
        if not lib.tok_mine(m, M):
            xqcs.append(sb((P, KT_D), BF16))
            continue
        xr = sb((B.NR, P), BF16)
        nisa.dma_copy(dst=xr, src=x[m])
        xq = lib.act_quant_r(xr)
        xqcs.append(lib.r_to_c(xq, ident))
    lib.exchange_tok(xqcs, M)
    for m in range(M):
        nisa.tensor_copy(dst=xqm[:, m, :], src=xqcs[m])
    x4m = block_x_m(xqm, bmt, M, KT_D)
    # the routing weights of all slots, on every partition (exact: ones matmul as in moe_impl_m)
    wall = sb((1, S * M), F32)
    nisa.dma_copy(dst=wall, src=rw.reshape((1, S * M)))
    wb = B.bcast_parts(wall, P, ones1)
    yacc = sb((P, M, KT_D), F32)
    nisa.memset(dst=yacc, value=0.0)
    h2rs = []
    for m in range(M):
        h2rs.append(B.load_h_r(h2[m]))
    fnF = B.load_fn(fnTf)
    identM = KB.ident_f32(M)
    ones40s = lib.ones_sb(B.NR, B.NR)
    fmst = KB.hc_mixes_m_start(h2rs, fnF, hscf, hbsf, ident, ones40s, identM)
    nsk = 0
    rings = KM.make_rings(we[0].dtype)
    wc = 0
    gc = 0
    for s in range(S):
        w1, s1, w3, s3, w2, s2 = KM.expert_views(we[s], se[s])
        a4, t = slot_up(x4m, M, w1, s1, w3, s3, wb[:, s * M:(s + 1) * M], bmt, ident, rings, wc, gc)
        wc = wc + t
        gc = gc + 2
        if nsk < KB.SK_ITERS:
            KB.hc_mixes_m_iter(fmst)
            nsk = nsk + 1
        if s > 0:
            y2b, t = slot_down(a4p, M, w2p, s2p, rings, wc, gc)
            wc = wc + t
            gc = gc + 1
            nisa.tensor_tensor(dst=yacc, data1=yacc, data2=y2b, op=nl.add)
        a4p = a4
        w2p = w2
        s2p = s2
    y2b, t = slot_down(a4p, M, w2p, s2p, rings, wc, gc)
    nisa.tensor_tensor(dst=yacc, data1=yacc, data2=y2b, op=nl.add)
    # shared expert: computed by the attention half (k_layer.shared_m, the same instructions), bf16 C layout
    s2s = []
    for m in range(M):
        if not lib.tok_mine(m, M):
            s2s.append(None)
            continue
        t2 = sb((P, KT_D), BF16)
        nisa.dma_copy(dst=t2, src=s2in[m])
        s2s.append(t2)
    while nsk < KB.SK_ITERS:
        KB.hc_mixes_m_iter(fmst)
        nsk = nsk + 1
    fmix_all = KB.hc_mixes_m_finish(fmst)
    fmo = nl.ndarray((M, 24), dtype=F32, buffer=nl.shared_hbm)
    nisa.dma_copy(dst=fmo, src=fmix_all)
    h_out = nl.ndarray((M, 4, B.D), dtype=BF16, buffer=nl.shared_hbm)
    for m in range(M):
        if not lib.tok_mine(m, M):    # the other physical core finishes and stores this token
            continue
        ya = sb((P, KT_D), F32)
        nisa.tensor_tensor(dst=ya, data1=yacc[:, m, :], data2=s2s[m], op=nl.add)
        yb = sb((P, KT_D), BF16)
        nisa.tensor_copy(dst=yb, src=ya)
        yr = lib.c_to_r(yb, ident)
        mixrow = KB.row_of(fmix_all, m, 24)
        mixb = B.bcast_parts(mixrow, B.NR, ones40)
        h_r = B.load_h_r(h2[m])
        ho = B.hc_post(yr, h_r, mixb)
        B.store_h_r(h_out[m], ho)
    return h_out, fmo


def generate(path, variants):
    """k_moeu_gen.py: one @nki.jit wrapper per (M, S) with explicit arguments"""
    lines = ['"""generated by k_moeu.generate: the deduplicated M-token MoE variants"""', 'import nki',
             'import k_moeu as KU', '']
    for name, (M, S) in variants.items():
        es = [f'e_{s}' for s in range(S)]
        cs = [f'c_{s}' for s in range(S)]
        args = ['x', 'h2', 'rw', 'bm'] + es + cs + ['s2in', 'fnTf', 'hscf', 'hbsf']
        lines += ['@nki.jit', f'def {name}({", ".join(args)}):',
                  f'    return KU.moeu_impl({M}, {S}, x, h2, rw, bm, [{", ".join(es)}], [{", ".join(cs)}], s2in, '
                  'fnTf, hscf, hbsf)', '', '']
    src = '\n'.join(lines)
    import os
    if not os.path.exists(path) or open(path).read() != src:
        open(path, 'w').write(src)
