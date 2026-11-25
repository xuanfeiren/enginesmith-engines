"""Kernel: MoE FFN of one layer for one decode token, plus hc_post.

Gate (fp32 scores of the unquantized input, sqrt(softplus), +bias, top-6 by max8), routing weights, 6 routed
experts whose weights are selected with dynamic DMA offsets, the shared expert, sum, hc_post(ffn).

Routed expert weights are the published fp4 values re-encoded at load as legacy fp8 bytes (value * 2^-6, exact),
laid out per expert as [128 p, nt, kt, 128 j]; their E8M0 scales (times 2^6) as bf16 [N, K/32] per expert.  The
GEMV keeps the official per-32-block structure: each 32-row block of a [128 k, 128 n] weight tile is contracted
separately (x masked to one block per moving column), then the fp32 block sums are scaled and added.
"""
import nki
import nki.language as nl
import nki.isa as nisa

import lib
import blocks as B
import k_batch as KB
from lib import P, F32, BF16, I32, U32, sb, ps

NE = 384
TOPK = 6
FI = 2304          # expert intermediate
KT_D = B.D // P    # 40
NT_I = FI // P     # 18
SWIGLU = 10.0
ROUTE_SCALE = 1.5
WBUF = 6           # weight tile buffers shared by the block-scaled GEMVs of a kernel
SBUFN = 3          # scale buffers


def block_x(xc, bm):
    """xc C [128, KT] bf16 -> X4 [128, KT, 4]: X4[p, kt, j] = xc[p, kt] if p // 32 == j else 0."""
    kt = xc.shape[1]
    x4 = sb((P, kt, 4), BF16)
    nisa.tensor_tensor(dst=x4, data1=xc.reshape((P, kt, 1)).broadcast(2, 4), data2=bm.reshape((P, 1, 4)).broadcast(1, kt),
                       op=nl.multiply)
    return x4


def make_rings(dtype):
    """weight tile and scale buffers shared by all block-scaled GEMVs of a kernel: the weight DMAs of the next tiles
    (also of the next matrix / expert) proceed while the TensorE works on earlier ones (buffers allocated per GEMV
    call were reused at the same SBUF address, which serialised each call's first DMAs behind the previous call)"""
    wr = []
    for _ in range(WBUF):
        wr.append(sb((P, KT_D * P), dtype))
    sr = []
    for _ in range(SBUFN):
        sr.append(sb((P, NT_I * (B.D // 32)), BF16))
    return wr, sr


def gemv_blockscaled(x4, w_hbm, s_hbm, N, K, out, rings=None, wbase=0, sbase=0):
    """out C [128, N/128] (fp32 SBUF) = sum_b s[n, b] * (sum_{k in b} x[k] w[n, k]).
    w_hbm: one expert's [128, NT*KT*128] fp8 (legacy), s_hbm: its [128, NT*NB] bf16 scales (bound by the host).
    rings (make_rings): buffer i of the weight ring holds this call's tile wbase + i, scale buffer sbase.
    Returns the number of weight tiles used."""
    KT, NT, NB = K // P, N // P, K // 32
    n0, n1 = lib.lnc_split(NT)
    # the scales are laid out per partition (s[p, nt*NB + b] = scale of row nt*128 + p, block b): this core's
    # scales arrive in one DMA of (n1 - n0) * NB contiguous values per partition (per-n-tile loads of NB values
    # per partition made the scale descriptors as numerous as the weight descriptors)
    if rings is None:
        sall = sb((P, (n1 - n0) * NB), BF16)
    else:
        sall = rings[1][sbase % SBUFN][:, 0:(n1 - n0) * NB]
    lib.wdma(sall, s_hbm[:, n0 * NB:n1 * NB])     # on the sync engine (the scalar engine's queue also computes)
    for nt in range(n0, n1):
        if rings is None:
            w = sb((P, KT * P), w_hbm.dtype)
        else:
            w = rings[0][(wbase + nt - n0) % WBUF][:, 0:KT * P]
        lib.wdma(w, w_hbm[:, nt * KT * P:(nt + 1) * KT * P])
        s = sall[:, (nt - n0) * NB:(nt - n0 + 1) * NB]
        pt = ps((P, NB))
        for kt in range(KT):
            nisa.nc_matmul(dst=pt[:, kt * 4:(kt + 1) * 4], stationary=w[:, kt * P:(kt + 1) * P], moving=x4[:, kt, :],
                           accumulate=False)
        prod = sb((P, NB), F32)
        nisa.tensor_tensor(dst=prod, data1=pt, data2=s, op=nl.multiply)
        nisa.tensor_reduce(dst=out[:, nt:nt + 1], op=nl.add, data=prod, axis=1)
    lib.exchange_halves(out, NT)
    return n1 - n0


def expert_up(x4, w1, s1, w3, s3, wcol, bmt, ident, rings, wc, gc):
    """W1 / W3 GEMVs, swiglu (* routing weight), act_quant, block mask: the down projection's moving operand.
    Returns (a4, tiles used)."""
    g1 = sb((P, NT_I), F32)
    t = gemv_blockscaled(x4, w1, s1, FI, B.D, g1, rings, wc, gc)
    g3 = sb((P, NT_I), F32)
    t = t + gemv_blockscaled(x4, w3, s3, FI, B.D, g3, rings, wc + t, gc + 1)
    g1b = sb((P, NT_I), BF16)
    nisa.tensor_copy(dst=g1b, src=g1)
    g3b = sb((P, NT_I), BF16)
    nisa.tensor_copy(dst=g3b, src=g3)
    a = swiglu(g1b, g3b, wcol, NT_I)
    ar = lib.c_to_r(a, ident)
    aq = lib.act_quant_r(ar)
    ac = lib.r_to_c(aq, ident)
    return block_x(ac, bmt), t


def expert_down(a4, w2, s2, rings, wc, gc):
    """the down projection: y2b C [128, 40] bf16 (moe_impl's y2b).  Returns (y2b, tiles used)."""
    y2 = sb((P, KT_D), F32)
    t = gemv_blockscaled(a4, w2, s2, B.D, FI, y2, rings, wc, gc)
    y2b = sb((P, KT_D), BF16)
    nisa.tensor_copy(dst=y2b, src=y2)
    return y2b, t


def swiglu(g_c, u_c, wcol, n):
    """act = silu(min(gate, 10)) * clamp(up, -10, 10) [* w]  (fp32), then bf16.  g_c/u_c: [128, n] bf16."""
    gf = sb((P, n), F32)
    nisa.tensor_scalar(dst=gf, data=g_c, op0=nl.minimum, operand0=SWIGLU)
    uf = sb((P, n), F32)
    nisa.tensor_scalar(dst=uf, data=u_c, op0=nl.minimum, operand0=SWIGLU, op1=nl.maximum, operand1=-SWIGLU)
    sg = sb((P, n), F32)
    lib.sigmoid_precise(sg, gf)
    a = sb((P, n), F32)
    nisa.tensor_tensor(dst=a, data1=gf, data2=sg, op=nl.multiply)      # silu = x * sigmoid(x)
    nisa.tensor_tensor(dst=a, data1=a, data2=uf, op=nl.multiply)
    out = sb((P, n), BF16)
    if wcol is None:
        nisa.tensor_copy(dst=out, src=a)
    else:
        nisa.tensor_scalar(dst=out, data=a, op0=nl.multiply, operand0=wcol)
    return out


def gate_scores(xc, wg, gbias, M):
    """router scores of M tokens (xc [128, 40, M] bf16, C layout per token): sqrt(softplus(x . Wg)) and that + bias,
    as rows [M, 384] fp32.  The GEMV and the transcendentals run with the experts on the partitions ([128, 3, M]:
    expert t*128 + p), so each instruction of the long softplus / sqrt chains covers 3*M values per partition instead
    of 384 on one; the same dot products (fp32 PSUM, kt order) and element-wise operations; the rows come back by
    (bit-exact) TensorE transposes."""
    NTE = NE // P
    gpT = ps((P, NTE * M))
    for t in range(NTE):
        for kt in range(KT_D):
            nisa.nc_matmul(dst=gpT[:, t * M:(t + 1) * M], stationary=wg[:, kt * NE + t * P:kt * NE + (t + 1) * P],
                           moving=xc[:, kt, :])
    g0 = sb((P, NTE * M), F32)
    nisa.tensor_copy(dst=g0, src=gpT)
    sp = sb((P, NTE * M), F32)
    lib.softplus_precise(sp, g0)
    scT = sb((P, NTE * M), F32)
    lib.sqrt_precise(scT, sp)
    gb = sb((P, NTE, M), F32)
    for m in range(M):
        nisa.dma_copy(dst=gb[:, :, m], src=gbias.ap(pattern=[[1, P], [P, NTE]], offset=0))
    bT = sb((P, NTE * M), F32)
    nisa.tensor_tensor(dst=bT, data1=scT, data2=gb.reshape((P, NTE * M)), op=nl.add)
    rows = []
    for src in (scT, bT):
        pt = ps((M, NE), F32)
        for t in range(NTE):
            nisa.nc_transpose(dst=pt[:, t * P:(t + 1) * P], data=src[:, t * M:(t + 1) * M])
        r = sb((M, NE), F32)
        nisa.tensor_copy(dst=r, src=pt)
        rows.append(r)
    return rows[0], rows[1]


def gate_route(xr, wgT, gbias, ident, wg_sb=None):
    """MoE gate for one token: returns (idx [1, 8] int32 SBUF, wts [1, 8] fp32 SBUF); entries 0..5 used.
    scores = linear(x.float(), W.float()); sqrt(softplus); top-6 of scores + bias; weights from unbiased scores,
    normalized (+1e-20), times 1.5."""
    xc_raw = lib.r_to_c(xr, ident)                            # C [128, 40]
    if wg_sb is None:
        wg = sb((P, KT_D * NE), BF16)
        nisa.dma_copy(dst=wg, src=wgT)
    else:
        wg = wg_sb
    sc, biased = gate_scores(xc_raw.reshape((P, KT_D, 1)), wg, gbias, 1)
    top = sb((1, 8), F32)
    nisa.max8(dst=top, src=biased)
    tidx = sb((1, 8), U32)
    nisa.nc_find_index8(dst=tidx, data=biased, vals=top)
    tidx_i = sb((1, 8), I32)
    nisa.tensor_copy(dst=tidx_i, src=tidx)
    tidx_f = sb((1, 8), F32)
    nisa.tensor_copy(dst=tidx_f, src=tidx_i)
    io = sb((1, NE), I32)
    nisa.iota(dst=io, pattern=[[1, NE]], offset=0, channel_multiplier=0)
    iof = sb((1, NE), F32)
    nisa.tensor_copy(dst=iof, src=io)
    wts = sb((1, 8), F32)
    nisa.memset(dst=wts, value=0.0)
    for j in range(TOPK):
        oh = sb((1, NE), F32)
        nisa.tensor_scalar(dst=oh, data=iof, op0=nl.equal, operand0=tidx_f[:, j:j + 1])
        nisa.tensor_tensor(dst=oh, data1=oh, data2=sc, op=nl.multiply)
        nisa.tensor_reduce(dst=wts[:, j:j + 1], op=nl.add, data=oh, axis=1)
    wsum = sb((1, 1), F32)
    nisa.tensor_reduce(dst=wsum, op=nl.add, data=wts[:, 0:TOPK], axis=1)
    nisa.tensor_scalar(dst=wsum, data=wsum, op0=nl.add, operand0=1e-20)
    rws = sb((1, 1), F32)
    B.recip_precise(rws, wsum)
    nisa.tensor_scalar(dst=wts[:, 0:TOPK], data=wts[:, 0:TOPK], op0=nl.multiply, operand0=rws)
    nisa.tensor_scalar(dst=wts[:, 0:TOPK], data=wts[:, 0:TOPK], op0=nl.multiply, operand0=ROUTE_SCALE)
    return tidx_i, wts


def ffn_mixes(h2, fnTf, hscf, hbsf, ident):
    """the ffn hc_mixes of a decode token from its h2 [4, 5120] bf16 (HBM) -> SBUF [1, 24] fp32 and an HBM copy"""
    h_r = B.load_h_r(h2)
    ones40 = lib.ones_sb(B.NR, B.NR)
    fmix = B.hc_mixes(h_r, fnTf, hscf, hbsf, ident, ones40)
    fmix_out = nl.ndarray((1, 24), dtype=F32, buffer=nl.shared_hbm)
    nisa.dma_copy(dst=fmix_out, src=fmix)
    return fmix, fmix_out


def moe_impl(x, h2, fmix, rw, bm, w1, s1, w3, s3, w2, s2, ws1, ws3, ws2, fnTf=None, hscf=None, hbsf=None):
    """The 6 routed experts arrive as separate inputs (w1[j] ... s2[j]): the host binds the selected experts' weight
    slices after reading the router output of attn_out, so no DMA has a run-time offset."""
    ident = lib.identity_sb(P)
    ones1 = lib.ones_sb(1, P)
    xr = sb((B.NR, P), BF16)
    nisa.dma_copy(dst=xr, src=x)
    bmt = sb((P, 4), BF16)
    nisa.dma_copy(dst=bmt, src=bm)
    wts = sb((1, 8), F32)
    nisa.dma_copy(dst=wts, src=rw)
    wb = B.bcast_parts(wts, P, ones1)                         # [128, 8] routing weights on every partition
    # ---------------- quantized input, block-masked moving operand
    xq = lib.act_quant_r(xr)
    xqc = lib.r_to_c(xq, ident)                                # C [128, 40]
    x4 = block_x(xqc, bmt)
    yacc = sb((P, KT_D), F32)
    nisa.memset(dst=yacc, value=0.0)
    # software-pipelined: expert j's W1 / W3 are issued before expert j-1's W2 (the TensorE works while the
    # swiglu / act_quant of j-1 run); the y2b are added in expert order as before
    rings = make_rings(w1[0].dtype)
    wc = 0
    gc = 0
    for j in range(TOPK):
        a4, t = expert_up(x4, w1[j], s1[j], w3[j], s3[j], wb[:, j:j + 1], bmt, ident, rings, wc, gc)
        wc = wc + t
        gc = gc + 2
        if j > 0:
            y2b, t = expert_down(a4p, w2[j - 1], s2[j - 1], rings, wc, gc)
            wc = wc + t
            gc = gc + 1
            nisa.tensor_tensor(dst=yacc, data1=yacc, data2=y2b, op=nl.add)
        a4p = a4
    y2b, t = expert_down(a4p, w2[TOPK - 1], s2[TOPK - 1], rings, wc, gc)
    nisa.tensor_tensor(dst=yacc, data1=yacc, data2=y2b, op=nl.add)
    yrout = sb((P, KT_D), BF16)
    nisa.tensor_copy(dst=yrout, src=yacc)
    # ---------------- shared expert (bf16 weights)
    s1 = lib.gemv_c(xqc, ws1, FI, B.D)
    s3 = lib.gemv_c(xqc, ws3, FI, B.D)
    sa = swiglu(s1, s3, None, NT_I)
    sar = lib.c_to_r(sa, ident)
    saq = lib.act_quant_r(sar)
    sac = lib.r_to_c(saq, ident)
    s2 = lib.gemv_c(sac, ws2, B.D, FI)                         # bf16 C [128, 40]
    nisa.tensor_tensor(dst=yacc, data1=yacc, data2=s2, op=nl.add)
    yb = sb((P, KT_D), BF16)
    nisa.tensor_copy(dst=yb, src=yacc)
    yr = lib.c_to_r(yb, ident)                                 # R [40, 128]
    # ---------------- hc_post(ffn)
    fmix_out = None
    if fnTf is not None:      # the ffn mixes are computed here from h2 (the attention half no longer does)
        mixrow, fmix_out = ffn_mixes(h2, fnTf, hscf, hbsf, ident)
    else:
        mixrow = sb((1, 24), F32)
        nisa.dma_copy(dst=mixrow, src=fmix)
    h_r = B.load_h_r(h2)
    ones40 = lib.ones_sb(1, B.NR)
    mixb = B.bcast_parts(mixrow, B.NR, ones40)
    ho = B.hc_post(yr, h_r, mixb)
    h_out = nl.ndarray((4, B.D), dtype=BF16, buffer=nl.shared_hbm)
    B.store_h_r(h_out, ho)
    if fmix_out is not None:
        return h_out, fmix_out
    return h_out


def expert_lists(es, cs):
    """per-expert (we, se) tensors -> the six lists of views moe_impl takes"""
    L6 = [[], [], [], [], [], []]
    for i in range(len(es)):
        v = expert_views(es[i], cs[i])
        for k in range(6):
            L6[k].append(v[k])
    return L6


@nki.jit
def moe(x, h2, rw, bm, e_0, e_1, e_2, e_3, e_4, e_5, c_0, c_1, c_2, c_3, c_4, c_5, ws1, ws3, ws2, fnTf, hscf, hbsf):
    """outputs: h_out, fmix (the ffn mixes, computed here from h2)"""
    L6 = expert_lists([e_0, e_1, e_2, e_3, e_4, e_5], [c_0, c_1, c_2, c_3, c_4, c_5])
    return moe_impl(x, h2, None, rw, bm, L6[0], L6[1], L6[2], L6[3], L6[4], L6[5], ws1, ws3, ws2, fnTf, hscf, hbsf)


def moe_impl_m(M, x, h2, fmix, rw, bm, w1, s1, w3, s3, w2, s2, ws1, ws3, ws2, fnTf=None, hscf=None, hbsf=None):
    """moe_impl for M tokens: w1[m][j] ... are the host-bound expert slices of token m; the shared expert's weight
    tiles are loaded once for all tokens.  Per-token arithmetic is that of moe_impl."""
    ident = lib.identity_sb(P)
    ones1 = lib.ones_sb(1, P)
    ones40 = lib.ones_sb(1, B.NR)
    bmt = sb((P, 4), BF16)
    nisa.dma_copy(dst=bmt, src=bm)
    xqcs, yaccs, x4s, wbs = [], [], [], []
    for m in range(M):
        xr = sb((B.NR, P), BF16)
        nisa.dma_copy(dst=xr, src=x[m])
        wts = sb((1, 8), F32)
        nisa.dma_copy(dst=wts, src=rw[m:m + 1, :])
        wbs.append(B.bcast_parts(wts, P, ones1))
        xq = lib.act_quant_r(xr)
        xqc = lib.r_to_c(xq, ident)
        xqcs.append(xqc)
        x4s.append(block_x(xqc, bmt))
        yacc = sb((P, KT_D), F32)
        nisa.memset(dst=yacc, value=0.0)
        yaccs.append(yacc)
    # the ffn mixes of the M tokens (k_batch.hc_mixes_m), their Sinkhorn iterations interleaved with the expert
    # pairs below (each iteration is a few tiny VectorE / ScalarE instructions)
    fmst = None
    if fnTf is not None:
        h2rs = []
        for m in range(M):
            h2rs.append(B.load_h_r(h2[m]))
        fnF = B.load_fn(fnTf)
        identM = KB.ident_f32(M)
        ones40s = lib.ones_sb(B.NR, B.NR)
        fmst = KB.hc_mixes_m_start(h2rs, fnF, hscf, hbsf, ident, ones40s, identM)
    nsk = 0
    # software-pipelined over the M*6 (token, expert) pairs: pair i's W1 / W3 before pair i-1's W2; each token's
    # y2b are added in rank order as before
    rings = make_rings(w1[0][0].dtype)
    wc = 0
    gc = 0
    for m in range(M):
        for j in range(TOPK):
            a4, t = expert_up(x4s[m], w1[m][j], s1[m][j], w3[m][j], s3[m][j], wbs[m][:, j:j + 1], bmt, ident, rings,
                              wc, gc)
            wc = wc + t
            gc = gc + 2
            if fmst is not None and nsk < KB.SK_ITERS:
                KB.hc_mixes_m_iter(fmst)
                nsk = nsk + 1
            if m * TOPK + j > 0:
                y2b, t = expert_down(a4p, w2[mp][jp], s2[mp][jp], rings, wc, gc)
                wc = wc + t
                gc = gc + 1
                nisa.tensor_tensor(dst=yaccs[mp], data1=yaccs[mp], data2=y2b, op=nl.add)
            a4p = a4
            mp = m
            jp = j
    y2b, t = expert_down(a4p, w2[mp][jp], s2[mp][jp], rings, wc, gc)
    nisa.tensor_tensor(dst=yaccs[mp], data1=yaccs[mp], data2=y2b, op=nl.add)
    # shared expert, weight tiles shared by the M tokens
    s1s = lib.gemv_cm(xqcs, ws1, FI, B.D)
    s3s = lib.gemv_cm(xqcs, ws3, FI, B.D)
    sacs = []
    for m in range(M):
        sa = swiglu(s1s[m], s3s[m], None, NT_I)
        sar = lib.c_to_r(sa, ident)
        saq = lib.act_quant_r(sar)
        sacs.append(lib.r_to_c(saq, ident))
    s2s = lib.gemv_cm(sacs, ws2, B.D, FI)
    fmo = None
    if fmst is not None:
        while nsk < KB.SK_ITERS:
            KB.hc_mixes_m_iter(fmst)
            nsk = nsk + 1
        fmix_all = KB.hc_mixes_m_finish(fmst)
        fmo = nl.ndarray((M, 24), dtype=F32, buffer=nl.shared_hbm)
        nisa.dma_copy(dst=fmo, src=fmix_all)
    h_out = nl.ndarray((M, 4, B.D), dtype=BF16, buffer=nl.shared_hbm)
    for m in range(M):
        nisa.tensor_tensor(dst=yaccs[m], data1=yaccs[m], data2=s2s[m], op=nl.add)
        yb = sb((P, KT_D), BF16)
        nisa.tensor_copy(dst=yb, src=yaccs[m])
        yr = lib.c_to_r(yb, ident)
        h_r = B.load_h_r(h2[m])
        if fmst is not None:
            mixrow = KB.row_of(fmix_all, m, 24)
        else:
            mixrow = sb((1, 24), F32)
            nisa.dma_copy(dst=mixrow, src=fmix[m:m + 1, :])
        mixb = B.bcast_parts(mixrow, B.NR, ones40)
        ho = B.hc_post(yr, h_r, mixb)
        B.store_h_r(h_out[m], ho)
    if fmo is not None:
        return h_out, fmo
    return h_out


# ================================================================================ decode: experts spread over cores
# One routed expert (expert1), the shared expert (shared1) and the sum + hc_post (combine): the same per-expert
# arithmetic as moe_impl, split so that the 6 routed experts of a decode token run on 6 NeuronCores at once.
W13E = FI * B.D          # fp8 bytes of one expert matrix (w1, w3, w2 alike)
S13E = FI * (B.D // 32)   # bf16 scales of one matrix (s1, s3, s2 alike: 128 x 2880)


def expert_views(we, se):
    """one expert's tensors -> (w1, s1, w3, s3, w2, s2) views: we [3 * W13E] fp8 = w1 | w3 | w2 (each [128, ...]
    tiled), se [3 * S13E] bf16 = s1 | s3 | s2 (each [128, 2880], per partition).  The host binds 2 tensors per
    expert instead of 6 (the schedule call costs ~0.6 us per bound tensor)."""
    w = []
    sv = []
    for m in range(3):
        w.append(we[m * W13E:(m + 1) * W13E].reshape((P, W13E // P)))
        sv.append(se[m * S13E:(m + 1) * S13E].reshape((P, S13E // P)))
    return w[0], sv[0], w[1], sv[1], w[2], sv[2]


def x4_of(x, bmt, ident):
    xr = sb((B.NR, P), BF16)
    nisa.dma_copy(dst=xr, src=x)
    xq = lib.act_quant_r(xr)
    xqc = lib.r_to_c(xq, ident)
    return xqc, block_x(xqc, bmt)


@nki.jit
def expert1(x, rw, bm, we, se):
    w1, s1, w3, s3, w2, s2 = expert_views(we, se)
    return expert1_impl(x, rw, bm, w1, s1, w3, s3, w2, s2)


def expert1_impl(x, rw, bm, w1, s1, w3, s3, w2, s2):
    """y2b = bf16(W2 act_quant(swiglu(W1 x, W3 x) * rw[0])) as a C tile [128, 40] bf16 (moe_impl's y2b)."""
    ident = lib.identity_sb(P)
    ones1 = lib.ones_sb(1, P)
    bmt = sb((P, 4), BF16)
    nisa.dma_copy(dst=bmt, src=bm)
    wts = sb((1, 8), F32)
    nisa.dma_copy(dst=wts, src=rw)
    wb = B.bcast_parts(wts, P, ones1)
    xqc, x4 = x4_of(x, bmt, ident)
    rings = make_rings(w1.dtype)     # the W2 tiles stream in while the swiglu / act_quant run
    a4, t = expert_up(x4, w1, s1, w3, s3, wb[:, 0:1], bmt, ident, rings, 0, 0)
    y2b, t2 = expert_down(a4, w2, s2, rings, t, 2)
    out = nl.ndarray((P, KT_D), dtype=BF16, buffer=nl.shared_hbm)
    nisa.dma_copy(dst=out, src=y2b)
    return out


@nki.jit
def shared1(x, ws1, ws3, ws2, h2, fnTf, hscf, hbsf):
    """the shared expert of moe_impl: s2 as a C tile [128, 40] bf16; then the ffn mixes of h2 (fmix [1, 24])."""
    ident = lib.identity_sb(P)
    xr = sb((B.NR, P), BF16)
    nisa.dma_copy(dst=xr, src=x)
    xq = lib.act_quant_r(xr)
    xqc = lib.r_to_c(xq, ident)
    s1 = lib.gemv_c(xqc, ws1, FI, B.D)
    s3 = lib.gemv_c(xqc, ws3, FI, B.D)
    sa = swiglu(s1, s3, None, NT_I)
    sar = lib.c_to_r(sa, ident)
    saq = lib.act_quant_r(sar)
    sac = lib.r_to_c(saq, ident)
    s2 = lib.gemv_c(sac, ws2, B.D, FI)
    out = nl.ndarray((P, KT_D), dtype=BF16, buffer=nl.shared_hbm)
    nisa.dma_copy(dst=out, src=s2)
    _, fmix_out = ffn_mixes(h2, fnTf, hscf, hbsf, ident)
    return out, fmix_out


@nki.jit
def combine(y6, s2, h2, fmix):
    """moe_impl's tail: yacc = sum_j y2b_j (fp32, j = 0..5), + s2, bf16, hc_post(ffn) -> h_out [4, 5120] bf16."""
    ho = combine_core(y6, s2, h2, fmix)
    h_out = nl.ndarray((4, B.D), dtype=BF16, buffer=nl.shared_hbm)
    B.store_h_r(h_out, ho)
    return h_out


def combine_core(y6, s2, h2, fmix):
    """combine without the store: the residual as an SBUF tile [40, 4, 128] bf16 (load_h_r layout)"""
    ident = lib.identity_sb(P)
    yacc = sb((P, KT_D), F32)
    nisa.memset(dst=yacc, value=0.0)
    for j in range(TOPK):
        yj = sb((P, KT_D), BF16)
        nisa.dma_copy(dst=yj, src=y6[j])
        nisa.tensor_tensor(dst=yacc, data1=yacc, data2=yj, op=nl.add)
    sv = sb((P, KT_D), BF16)
    nisa.dma_copy(dst=sv, src=s2)
    nisa.tensor_tensor(dst=yacc, data1=yacc, data2=sv, op=nl.add)
    yb = sb((P, KT_D), BF16)
    nisa.tensor_copy(dst=yb, src=yacc)
    yr = lib.c_to_r(yb, ident)
    h_r = B.load_h_r(h2)
    mixrow = sb((1, 24), F32)
    nisa.dma_copy(dst=mixrow, src=fmix)
    ones40 = lib.ones_sb(1, B.NR)
    mixb = B.bcast_parts(mixrow, B.NR, ones40)
    return B.hc_post(yr, h_r, mixb)
