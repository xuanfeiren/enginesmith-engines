"""Kernel: the whole attention half of one layer for one decode token, in one NEFF.

attn_in (hc_mixes, hc_pre, norm, q / kv projections, RoPE, act_quant) -> [compressor of a kv source] ->
[indexer of an index source, when more than 512 compressed positions exist] -> sparse attention (window ring +
compressed rows, dense over a static bucket with a mask) -> wo_a / wo_b -> hc_post -> ffn hc_mixes / hc_pre /
ffn_norm -> MoE router.  The routed experts run in a second NEFF (`k_moe.moe`) whose expert weights the host binds
from the router output.

Rows this token adds to HBM caches (window ring slot, compressed row, index key) are patched into the SBUF copies
the kernel reads, so no HBM write of this kernel is read back by it; the host writes the compressed rows / index
keys into the caches afterwards.  No DMA has a run-time offset.
"""
import nki
import nki.language as nl
import nki.isa as nisa

import lib
import blocks as B
import k_attn_out as AO
import k_compress as KC
import k_indexer as KI
import k_moe as KM
import k_batch as KB
from lib import P, F32, BF16, I32, sb, ps

QL = 1280
NH = 64
HD = 512
WIN = 128
ID = 128
NEG = -1.0e30
PRE_QB = 12        # wq_b tiles prefetched during the q_a norm


def ones_row_sel(n, r):
    """[n, 128] bf16 selector: row r all ones, other rows zero (built with iota, no partition-offset writes)."""
    io = sb((n, P), I32)
    nisa.iota(dst=io, pattern=[[0, P]], offset=0, channel_multiplier=1)        # value = partition
    iof = sb((n, P), F32)
    nisa.tensor_copy(dst=iof, src=io)
    s = sb((n, P), BF16)
    nisa.tensor_scalar(dst=s, data=iof, op0=nl.equal, operand0=float(r))
    return s


def bcast_vec(x_r):
    """R tile [n, 128] bf16 -> SBUF [128, n*128] bf16, every partition holding the whole vector."""
    n = x_r.shape[0]
    out = sb((P, n * P), BF16)
    for r in range(n):
        s = ones_row_sel(n, r)
        pt = ps((P, P))
        nisa.nc_matmul(dst=pt, stationary=s, moving=x_r)
        nisa.tensor_copy(dst=out[:, r * P:(r + 1) * P], src=pt)
    return out


def attn_in_core(h, premix, fnT, hsc, hbs, anw, wqa, qnw, wqb, wkv, kvw, rotP, ropec, roper, ident, ones40, pk=None,
                 late_mix=False):
    if pk is not None:   # packed: the previous layer's combine (y6, s2, h2, fmix) runs here first
        h_r = KM.combine_core(pk[0], pk[1], pk[2], pk[3])
    else:
        h_r = B.load_h_r(h)
    # the attention mixes (pre / post / comb) are used only after the attention (hc_post, the ffn's hc_pre): with
    # late_mix the caller computes them later, off the critical path to the q projection (same instructions)
    mix = None if late_mix else B.hc_mixes(h_r, fnT, hsc, hbs, ident, ones40)
    pm = B.bcast_row(premix, B.NR, 4)
    x = B.hc_pre(h_r, pm)
    g = sb((B.NR, P), BF16)
    nisa.dma_copy(dst=g, src=anw)
    xn = lib.rmsnorm_r(x, g, B.D, B.EPS, ones40)
    xq = lib.act_quant_r(xn)
    xc = lib.r_to_c(xq, ident)
    qa = lib.gemv_c(xc, wqa, QL, B.D)
    pre_qb = lib.prefetch(wqb, NH * HD, QL, PRE_QB)
    qa_r = lib.c_to_r(qa, ident)
    gq = sb((QL // P, P), BF16)
    nisa.dma_copy(dst=gq, src=qnw)
    ones10 = lib.ones_sb(QL // P, QL // P)
    qr = lib.rmsnorm_r(qa_r, gq, QL, B.EPS, ones10)
    qrq = lib.act_quant_r(qr)
    qrc = lib.r_to_c(qrq, ident)
    if late_mix:   # the attention mixes, issued ahead of the wq_b stream (they are needed only at hc_post)
        mix = B.hc_mixes(h_r, fnT, hsc, hbs, ident, ones40)
    q = lib.gemv_c(qrc, wqb, NH * HD, QL, pre=pre_qb)         # C [128, 256], column = h*4 + c
    rot = sb((P, P), BF16)
    nisa.dma_copy(dst=rot, src=rotP)
    cosc, sinc = B.load_rope_c(ropec)
    qv = q[:, 3:NH * 4:4]
    B.rope_c_cols(qv, rot, cosc, sinc, qv)
    kv = lib.gemv_c(xc, wkv, HD, B.D)
    kv_r = lib.c_to_r(kv, ident)
    gk = sb((4, P), BF16)
    nisa.dma_copy(dst=gk, src=kvw)
    ones4 = lib.ones_sb(4, 4)
    kvn = lib.rmsnorm_r(kv_r, gk, HD, B.EPS, ones4)
    cs = B.load_rope_r(roper, 4, 3, 4)
    B.rope_r_row(kvn, 0, 4, cs)
    kvq = lib.act_quant_r(kvn)
    return h_r, mix, xn, qr, q, rot, cosc, sinc, kvq


def compress_core(xn, mode, ratio, cwkv, cwg, state, cnw, wk, knw, roperc, ident):
    """mode 'store' (ratio 2, even position: keep slot 0) or 'emit'.  Returns (k [1,128], c [4,128]) bf16 SBUF
    rows for 'emit', None for 'store'."""
    xc = lib.r_to_c(xn, ident)
    if ratio == 1:
        kv = lib.gemv_c(xc, cwkv, HD, B.D)
        kvr = lib.c_to_r(kv, ident)
        g = sb((4, P), BF16)
        nisa.dma_copy(dst=g, src=cnw)
        ones4 = lib.ones_sb(4, 4)
        lat = lib.rmsnorm_r(kvr, g, HD, B.EPS, ones4)
        return KC.finish_sb(lat, wk, knw, roperc, ident)
    kv1 = KC.gemv_f32(xc, cwkv, HD, B.D)
    sc1 = KC.gemv_f32(xc, cwg, HD, B.D)
    if mode == 'store':
        nisa.dma_copy(dst=state[0], src=kv1)
        nisa.dma_copy(dst=state[1], src=sc1)
        return None
    pooled = KC.pool2(kv1, sc1, state)
    pr = lib.c_to_r(pooled, ident)
    g = sb((4, P), BF16)
    nisa.dma_copy(dst=g, src=cnw)
    ones4 = lib.ones_sb(4, 4)
    lat = lib.rmsnorm_r(pr, g, HD, B.EPS, ones4)
    return KC.finish_sb(lat, wk, knw, roperc, ident)


def sel_to_mask(sel, T, ident):
    """sel [128, J] fp32 0/1 (t = j*128 + p) -> additive mask [64, T] fp32 (0 selected, -1e30 not)."""
    J = T // P
    selT = sb((J, P), F32)
    pt = ps((J, P), F32)
    nisa.nc_transpose(dst=pt, data=sel)
    nisa.tensor_copy(dst=selT, src=pt)                         # [J, 128]: row j = positions 128j..128j+127
    selb = sb((J, P), BF16)
    nisa.tensor_copy(dst=selb, src=selT)
    m = sb((NH, T), F32)
    for j in range(J):
        s = ones_row_sel(J, j)
        pr = ps((NH, P))
        nisa.nc_matmul(dst=pr, stationary=s[:, 0:NH], moving=selb)
        # (sel - 1) * 1e30
        nisa.tensor_scalar(dst=m[:, j * P:(j + 1) * P], data=pr, op0=nl.subtract, operand0=1.0, op1=nl.multiply,
                           operand1=-NEG)
    return m


def load_crank(crank_hbm, NCH):
    """host-written list positions of the compressed keys [1, T] -> SBUF [128, NCH] (key j*128 + p)"""
    t = sb((P, NCH), F32)
    nisa.dma_copy(dst=t, src=crank_hbm.ap(pattern=[[1, P], [P, NCH]], offset=0))
    return t


def load_crows(ckv, T, cpatch):
    NCH = T // P
    crows = sb((P, NCH, HD), BF16)
    nisa.dma_copy(dst=crows, src=ckv.ap(pattern=[[HD, P], [P * HD, NCH], [1, HD]], offset=0))
    if cpatch is not None:
        for pi in range(len(cpatch)):
            KI.KL_patch(crows, cpatch[pi][0], cpatch[pi][1], NCH, HD)
    return crows


def layer_impl(h, premix, posv, fnT, hsc, hbs, anw, wqa, qnw, wqb, wkv, kvw, rotP, ropec, roper,
               win, sink, woa, wob, fnTf, hscf, hbsf, fnw, wgT, gbias,
               ckv, cmask, cwkv, cwg, state, cnw, wk, knw, roperc, iwqb, iwp, kc, lstrict,
               ratio, T, comp, index, wrank=None, lvalid=None, pk=None):
    ident = lib.identity_sb(P)
    ones40 = lib.ones_sb(B.NR, B.NR)
    ones1 = lib.ones_sb(1, B.NR)
    onesP = lib.ones_sb(1, P)
    pv = sb((1, 8), I32)
    nisa.dma_copy(dst=pv, src=posv)
    h_r, mix, xn, qr, q, rot, cosc, sinc, kvq = attn_in_core(h, premix, fnT, hsc, hbs, anw, wqa, qnw, wqb, wkv,
                                                             kvw, rotP, ropec, roper, ident, ones40, pk,
                                                             late_mix=True)
    outs = []
    newk = None
    newc = None
    if comp:
        res = compress_core(xn, comp, ratio, cwkv, cwg, state, cnw, wk, knw, roperc, ident)
        if res is not None:
            newk, newc = res
            cp = nl.ndarray((1, 5 * P), dtype=BF16, buffer=nl.shared_hbm)     # k (128) | c (512)
            nisa.dma_copy(dst=cp[:, 0:P], src=newk)
            nisa.dma_copy(dst=cp.ap(pattern=[[P, 4], [1, P]], offset=P), src=newc)
            outs.append(cp)
    crow_f = None
    if newk is not None:
        crf = sb((1, 1), F32)
        nisa.tensor_copy(dst=crf, src=pv[:, B.PV_CROW:B.PV_CROW + 1])
        crow_f = B.bcast_parts(crf, P, onesP)                   # [128, 1]
    mask = None
    if index:
        kpatch = None
        if newk is not None:
            kpatch = []
            kpatch.append((bcast_vec(newk), crow_f))
        sel = KI.indexer_core(qr, xn, pv, iwqb, iwp, rot, cosc, sinc, kc, lstrict, T, ident, kpatch)
        so = nl.ndarray((P, T // P), dtype=F32, buffer=nl.shared_hbm)
        nisa.dma_copy(dst=so, src=sel)
        outs.append(so)
        mask = KI.sel_listpos(sel, T // P, lstrict)
    cpatch = None
    if newc is not None:
        cpatch = []
        cpatch.append((bcast_vec(newc), crow_f))
    kvb = bcast_vec(kvq)                                       # [128, 512] this token's window row
    wrows = sb((WIN, HD), BF16)
    nisa.dma_copy(dst=wrows, src=win)
    AO.ring_patch(wrows, kvb, pv)
    nisa.dma_copy(dst=win, src=wrows)
    crows = None
    crank = None
    NCH = T // P
    if ratio:
        crows = load_crows(ckv, T, cpatch)
        if mask is not None:
            crank = mask
        else:
            crank = load_crank(cmask, NCH)
    pre_oa = AO.prefetch_woa(woa)
    o = AO.attn_core_list(q, wrows, wrank, lvalid, sink, roper, crows, crank, NCH, ratio == 0)
    wg = sb((P, 40 * 384), BF16)                               # router weights, loaded under wo_b
    lib.wdma(wg, wgT)
    xa = AO.out_proj(o, woa, wob, ident, pre_oa=pre_oa)
    mixb = B.bcast_parts(mix, B.NR, ones1)
    h2 = B.hc_post(xa, h_r, mixb)
    h2_out = nl.ndarray((4, B.D), dtype=BF16, buffer=nl.shared_hbm)
    B.store_h_r(h2_out, h2)
    # the ffn mixes (B.hc_mixes(h2, fnTf, ...)) are needed only after the MoE and by the next layer: shared1 / moe
    # compute them from h2, so the router follows hc_post directly
    x2 = B.hc_pre(h2, mixb, 0)
    g = sb((B.NR, P), BF16)
    nisa.dma_copy(dst=g, src=fnw)
    xf = lib.rmsnorm_r(x2, g, B.D, B.EPS, ones40)
    xf_out = nl.ndarray((B.NR, P), dtype=BF16, buffer=nl.shared_hbm)
    nisa.dma_copy(dst=xf_out, src=xf)
    ridx, rw = KM.gate_route(xf, wgT, gbias, ident, wg_sb=wg)
    ridx_out = nl.ndarray((1, 8), dtype=I32, buffer=nl.shared_hbm)
    nisa.dma_copy(dst=ridx_out, src=ridx)
    rw_out = nl.ndarray((1, 8), dtype=F32, buffer=nl.shared_hbm)
    nisa.dma_copy(dst=rw_out, src=rw)
    res = [h2_out, xf_out, ridx_out, rw_out]
    for o in outs:
        res.append(o)
    return tuple(res)


# ================================================================================================ M tokens
def layer_impl_m(M, h, premix, posv, fnT, hsc, hbs, anw, wqa, qnw, wqb, wkv, kvw, rotP, ropec, roper,
                 win, sink, woa, wob, fnTf, hscf, hbsf, fnw, wgT, gbias,
                 ckv, cmask, cwkv, cwg, state, cnw, wk, knw, roperc, iwqb, iwp, kc, lstrict,
                 ratio, T, comp, index, wrank=None, lvalid=None, shared=None):
    """The attention half for M consecutive prompt tokens (positions p0 .. p0+M-1, p0 a multiple of M, M even).
    Weight tiles of the shared GEMVs are loaded once for all M tokens; every per-token computation (and its
    rounding) is that of layer_impl.  Token m sees the window rows, compressed rows and index keys of tokens < m
    through SBUF patches.  comp: None or 'pairs' (ratio 2: even tokens store, odd tokens emit) or 'emit' (ratio 1).
    Inputs are stacked per token on the first dimension; outputs too."""
    ident = lib.identity_sb(P)
    ones40 = lib.ones_sb(B.NR, B.NR)
    ones1 = lib.ones_sb(1, B.NR)
    onesP = lib.ones_sb(1, P)
    fnA = B.load_fn(fnT)
    rot = sb((P, P), BF16)
    nisa.dma_copy(dst=rot, src=rotP)
    pvs, h_rs, mixs, xns, xcs = [], [], [], [], []
    for m in range(M):
        pv = sb((1, 8), I32)
        nisa.dma_copy(dst=pv, src=posv[m:m + 1, :])
        pvs.append(pv)
        h_r = B.load_h_r(h[m])
        h_rs.append(h_r)
        if not lib.tok_mine(m, M):    # computed by the other physical core, exchanged below
            xns.append(sb((B.NR, P), BF16))
            xcs.append(sb((P, B.NR), BF16))
            continue
        pm = B.bcast_row(premix[m:m + 1, :], B.NR, 4)
        x = B.hc_pre(h_r, pm)
        g = sb((B.NR, P), BF16)
        nisa.dma_copy(dst=g, src=anw)
        xn = lib.rmsnorm_r(x, g, B.D, B.EPS, ones40)
        xns.append(xn)
        xq = lib.act_quant_r(xn)
        xcs.append(lib.r_to_c(xq, ident))
    lib.exchange_tok(xcs, M)
    lib.exchange_tok(xns, M)
    qas = lib.gemv_cm(xcs, wqa, QL, B.D)
    identM = KB.ident_f32(M)
    # the M tokens' attention mixes at once; their Sinkhorn iterations run between the wo_a groups (out_proj_m)
    mixst = KB.hc_mixes_m_start(h_rs, fnA, hsc, hbs, ident, ones40, identM)
    qrs, qrcs = [], []
    for m in range(M):
        if not lib.tok_mine(m, M):
            qrs.append(sb((QL // P, P), BF16))
            qrcs.append(sb((P, QL // P), BF16))
            continue
        qa_r = lib.c_to_r(qas[m], ident)
        gq = sb((QL // P, P), BF16)
        nisa.dma_copy(dst=gq, src=qnw)
        ones10 = lib.ones_sb(QL // P, QL // P)
        qr = lib.rmsnorm_r(qa_r, gq, QL, B.EPS, ones10)
        qrs.append(qr)
        qrq = lib.act_quant_r(qr)
        qrcs.append(lib.r_to_c(qrq, ident))
    lib.exchange_tok(qrcs, M)
    lib.exchange_tok(qrs, M)
    qs = lib.gemv_cm(qrcs, wqb, NH * HD, QL)
    coss, sins = [], []
    for m in range(M):
        if not lib.tok_mine(m, M):    # q RoPE, indexer and attention of this token run on the other core
            coss.append(None)
            sins.append(None)
            continue
        cosc, sinc = B.load_rope_c(ropec[m])
        coss.append(cosc)
        sins.append(sinc)
        qv = qs[m][:, 3:NH * 4:4]
        B.rope_c_cols(qv, rot, cosc, sinc, qv)
    kvs = lib.gemv_cm(xcs, wkv, HD, B.D)
    kvbs = []
    kvqs = []
    for m in range(M):
        if not lib.tok_mine(m, M):
            kvqs.append(sb((4, P), BF16))
            continue
        kv_r = lib.c_to_r(kvs[m], ident)
        gk = sb((4, P), BF16)
        nisa.dma_copy(dst=gk, src=kvw)
        ones4 = lib.ones_sb(4, 4)
        kvn = lib.rmsnorm_r(kv_r, gk, HD, B.EPS, ones4)
        cs = B.load_rope_r(roper[m], 4, 3, 4)
        B.rope_r_row(kvn, 0, 4, cs)
        kvq = lib.act_quant_r(kvn)
        kvqs.append(kvq)
    lib.exchange_tok(kvqs, M)
    for m in range(M):
        kvbs.append(bcast_vec(kvqs[m]))
    # ---- compressor: new rows per token (None where the token adds none)
    newk = []
    newc = []
    crows_f = []
    kbs = []
    cbs = []
    masks = []
    for m in range(M):
        newk.append(None)
        newc.append(None)
        crows_f.append(None)
        kbs.append(None)
        cbs.append(None)
        masks.append(None)
    outs = []
    if comp:
        xncs = []
        lats = []
        for m in range(M):
            xncs.append(lib.r_to_c(xns[m], ident))
            lats.append(None)
        if ratio == 1:
            kvc = lib.gemv_cm(xncs, cwkv, HD, B.D)
            for m in range(M):
                kvr = lib.c_to_r(kvc[m], ident)
                g = sb((4, P), BF16)
                nisa.dma_copy(dst=g, src=cnw)
                ones4 = lib.ones_sb(4, 4)
                lats[m] = lib.rmsnorm_r(kvr, g, HD, B.EPS, ones4)
        else:
            kv1 = lib.gemv_cm(xncs, cwkv, HD, B.D, out_dtype=F32)
            sc1 = lib.gemv_cm(xncs, cwg, HD, B.D, out_dtype=F32)
            for m in range(1, M, 2):
                pooled = KC.pool2_sb(kv1[m - 1], sc1[m - 1], kv1[m], sc1[m])
                pr = lib.c_to_r(pooled, ident)
                g = sb((4, P), BF16)
                nisa.dma_copy(dst=g, src=cnw)
                ones4 = lib.ones_sb(4, 4)
                lats[m] = lib.rmsnorm_r(pr, g, HD, B.EPS, ones4)
        cp = nl.ndarray((M, 5 * P), dtype=BF16, buffer=nl.shared_hbm)     # per token: k (128) | c (512)
        for m in range(M):
            if lats[m] is not None:
                k1, c1 = KC.finish_sb_r(lats[m], wk, knw, roperc[m], ident)
                newk[m] = k1
                newc[m] = c1
                nisa.dma_copy(dst=cp[m:m + 1, 0:P], src=k1)
                nisa.dma_copy(dst=cp[m].ap(pattern=[[P, 4], [1, P]], offset=P), src=c1)
        outs.append(cp)
    for m in range(M):
        if newk[m] is not None:
            crf = sb((1, 1), F32)
            nisa.tensor_copy(dst=crf, src=pvs[m][:, B.PV_CROW:B.PV_CROW + 1])
            crows_f[m] = B.bcast_parts(crf, P, onesP)
    for m in range(M):
        if newk[m] is not None:
            kbs[m] = bcast_vec(newk[m])
            cbs[m] = bcast_vec(newc[m])
    # ---- indexer (per token; q projection shared)
    if index:
        iqin = []
        for m in range(M):
            iqin.append(KI.indexer_q_in(qrs[m], ident))
        iqs = lib.gemv_cm(iqin, iwqb, 32 * 128, QL)
        wp = sb((P, B.NR * 32), BF16)
        nisa.dma_copy(dst=wp, src=iwp)
        selo = nl.ndarray((M, P, T // P), dtype=F32, buffer=nl.shared_hbm)
        for m in range(M):
            kp = []
            for i in range(m + 1):
                if kbs[i] is not None:
                    kp.append((kbs[i], crows_f[i]))
            if len(kp) == 0:
                kp = None
            if lib.tok_mine(m, M):     # the token's owner core only (its attention below needs the mask)
                sel = KI.indexer_core(qrs[m], xns[m], pvs[m], iwqb, iwp, rot, coss[m], sins[m], kc, lstrict, T, ident,
                                      kp, q_pre=iqs[m], wp_sb=wp)
                nisa.dma_copy(dst=selo[m], src=sel)
                masks[m] = KI.sel_listpos(sel, T // P, lstrict)
        outs.append(selo)
    # ---- attention per token
    wrows = sb((WIN, HD), BF16)
    nisa.dma_copy(dst=wrows, src=win)
    os_ = []
    for m in range(M):
        AO.ring_patch(wrows, kvbs[m], pvs[m])     # both cores: every token sees the rows of the earlier ones
        if not lib.tok_mine(m, M):
            os_.append(None)                       # computed by the other physical core (out_proj_m exchanges)
            continue
        cp_ = []
        for i in range(m + 1):
            if cbs[i] is not None:
                cp_.append((cbs[i], crows_f[i]))
        if len(cp_) == 0:
            cp_ = None
        crows = None
        crank = None
        NCH = T // P
        if ratio:
            crows = load_crows(ckv, T, cp_)
            if masks[m] is not None:
                crank = masks[m]
            else:
                crank = load_crank(cmask[m:m + 1, :], NCH)
        o = AO.attn_core_list(qs[m], wrows, wrank[m:m + 1, :], lvalid[m:m + 1, :], sink, roper[m], crows, crank, NCH,
                              ratio == 0)
        os_.append(o)
    nisa.dma_copy(dst=win, src=wrows)
    xas = AO.out_proj_m(os_, woa, wob, ident, mixst=mixst, nsk=KB.SK_ITERS)
    mix_all = KB.hc_mixes_m_finish(mixst)
    # ---- hc_post, ffn mixes, hc_pre, ffn_norm, router per token
    wg = sb((P, 40 * 384), BF16)
    nisa.dma_copy(dst=wg, src=wgT)
    h2o = nl.ndarray((M, 4, B.D), dtype=BF16, buffer=nl.shared_hbm)
    xfo = nl.ndarray((M, B.NR, P), dtype=BF16, buffer=nl.shared_hbm)
    rio = nl.ndarray((M, 8), dtype=I32, buffer=nl.shared_hbm)
    rwo = nl.ndarray((M, 8), dtype=F32, buffer=nl.shared_hbm)
    h2s, xfs = [], []
    for m in range(M):
        if not lib.tok_mine(m, M):      # the owner core computes and stores this token's rows; xf comes by sendrecv
            xfs.append(sb((B.NR, P), BF16))
            continue
        mixb = B.bcast_parts(KB.row_of(mix_all, m, 24), B.NR, ones1)
        h2 = B.hc_post(xas[m], h_rs[m], mixb)
        B.store_h_r(h2o[m], h2)
        x2 = B.hc_pre(h2, mixb, 0)
        g = sb((B.NR, P), BF16)
        nisa.dma_copy(dst=g, src=fnw)
        xf = lib.rmsnorm_r(x2, g, B.D, B.EPS, ones40)
        nisa.dma_copy(dst=xfo[m], src=xf)
        h2s.append(h2)
        xfs.append(xf)
    lib.exchange_tok(xfs, M)   # both cores run the (partition-batched) router on all tokens' ffn inputs
    # the router of the M tokens at once (row m = token m); their ffn mixes are computed by moe4 (k_moe.moe_impl_m),
    # interleaved with its expert stream
    ridx, rw = KB.gate_route_m(xfs, wg, gbias, ident)
    nisa.dma_copy(dst=rio, src=ridx)
    nisa.dma_copy(dst=rwo, src=rw)
    res = [h2o, xfo, rio, rwo]
    for o in outs:
        res.append(o)
    if shared is not None:
        res.append(shared_m(xfs, shared[0], shared[1], shared[2], ident, M))
    return tuple(res)


def shared_m(xfs, ws1, ws3, ws2, ident, M):
    """The group's shared expert, exactly as k_moeu.moeu_impl computed it from the same bf16 ffn inputs
    (act_quant per token, gemv_cm of the M tokens, swiglu, act_quant, gemv_cm); per-token steps on the owner core.
    Returns s2o [M, 128, 40] bf16 (C layout per token)."""
    xqcs = []
    for m in range(M):
        if not lib.tok_mine(m, M):
            xqcs.append(sb((P, B.NR), BF16))
            continue
        xq = lib.act_quant_r(xfs[m])
        xqcs.append(lib.r_to_c(xq, ident))
    lib.exchange_tok(xqcs, M)
    s1s = lib.gemv_cm(xqcs, ws1, KM.FI, B.D)
    s3s = lib.gemv_cm(xqcs, ws3, KM.FI, B.D)
    sacs = []
    for m in range(M):
        if not lib.tok_mine(m, M):
            sacs.append(sb((P, KM.NT_I), BF16))
            continue
        sa = KM.swiglu(s1s[m], s3s[m], None, KM.NT_I)
        sar = lib.c_to_r(sa, ident)
        saq = lib.act_quant_r(sar)
        sacs.append(lib.r_to_c(saq, ident))
    lib.exchange_tok(sacs, M)
    s2s = lib.gemv_cm(sacs, ws2, B.D, KM.FI)
    s2o = nl.ndarray((M, P, B.NR), dtype=BF16, buffer=nl.shared_hbm)
    for m in range(M):
        if lib.tok_mine(m, M):
            nisa.dma_copy(dst=s2o[m], src=s2s[m])
    return s2o
