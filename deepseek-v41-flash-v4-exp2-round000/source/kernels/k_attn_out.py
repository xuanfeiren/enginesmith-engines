"""Kernel: attention core + output projection + hc_post + FFN input stage of one layer for one decode token.

sparse_attn over the window ring (+ up to 512 gathered compressed rows), inverse RoPE, act_quant, wo_a (grouped),
act_quant, wo_b, hc_post(attn), hc_mixes(ffn), hc_pre(attn_pre), ffn_norm.
"""
import nki
import nki.language as nl
import nki.isa as nisa

import lib
import k_batch as KB
import blocks as B
import k_moe as KM
import k_indexer as KI
from lib import P, F32, BF16, I32, sb, ps

NH = 64
HD = 512
WIN = 128
NG = 8
OL = 1024
NSEL = 512
NC = NSEL // P
SCALE = float(HD ** -0.5)
NEG = -1.0e30


def tr_to(dst, src):
    """dst [f, n] (SBUF) = src [n, f]^T through PSUM (exact)."""
    n, f = src.shape[0], src.shape[1]
    pt = ps((f, n), src.dtype)
    nisa.nc_transpose(dst=pt, data=src)
    nisa.tensor_copy(dst=dst, src=pt)


def attn_core(qc, win, kvr, pv, sink, roper, ckv, cmask, T):
    kvb = sb((WIN, HD), BF16)
    nisa.dma_copy(dst=kvb, src=kvr.ap(pattern=[[0, WIN], [1, HD]], offset=0))
    return attn_core_sb(qc, win, kvb, pv, sink, roper, ckv, cmask, None, None, T)


def ring_patch(wrows, kvb, pv):
    """window ring rows [128 slots, 512]: put this token's kv row (kvb: the row on every partition) into slot
    pos % 128 (pv[0, 1]) with a predicated copy (exact)."""
    slot_f = sb((1, 1), F32)
    nisa.tensor_copy(dst=slot_f, src=pv[:, 1:2])
    ones_w = lib.ones_sb(1, WIN)
    slotb = B.bcast_parts(slot_f, WIN, ones_w)            # [128, 1]
    pio = sb((WIN, 1), I32)
    nisa.iota(dst=pio, pattern=[[0, 1]], offset=0, channel_multiplier=1)   # value = partition index
    piof = sb((WIN, 1), F32)
    nisa.tensor_copy(dst=piof, src=pio)
    pred = sb((WIN, 1), nl.uint8)
    nisa.tensor_scalar(dst=pred, data=piof, op0=nl.equal, operand0=slotb)
    nisa.tensor_copy_predicated(dst=wrows, src=kvb, predicate=pred.broadcast(1, HD))


def attn_core_sb(qc, win, kvb, pv, sink, roper, ckv, cmask, mask_sb, cpatch, T, wrows_sb=None):
    """sparse_attn for one query over the window ring and, when T > 0, the compressed rows [0, T) (a static
    bucket) with the host-written additive mask cmask [1, T] (0 = selected, -1e30 = not).  Dense over the bucket:
    the same softmax set as the reference's gather of the selected rows.  Returns o (bf16 [64, 512])."""
    pos_sb = sb((1, 1), I32)
    nisa.tensor_copy(dst=pos_sb, src=pv[:, 0:1])
    if wrows_sb is not None:      # the caller loaded and patched the ring (multi-token kernels)
        wrows = wrows_sb
    else:
        wrows = sb((WIN, HD), BF16)
        nisa.dma_copy(dst=wrows, src=win)
        ring_patch(wrows, kvb, pv)
        nisa.dma_copy(dst=win, src=wrows)
    # window validity: slot s valid iff s <= pos
    io = sb((NH, WIN), I32)
    nisa.iota(dst=io, pattern=[[1, WIN]], offset=0, channel_multiplier=0)
    iof = sb((NH, WIN), F32)
    nisa.tensor_copy(dst=iof, src=io)
    posf = sb((1, 1), F32)
    nisa.tensor_copy(dst=posf, src=pos_sb)
    ones1 = lib.ones_sb(1, NH)
    posb = B.bcast_parts(posf, NH, ones1)                 # [64, 1]
    wmask = sb((NH, WIN), F32)
    nisa.tensor_scalar(dst=wmask, data=iof, op0=nl.greater, operand0=posb, op1=nl.multiply,
                       operand1=NEG)
    # scores, window part: q^T chunk c = qc[:, c::4]  ([128 d, 64 h]);  K^T chunk = transpose of the rows
    sw = ps((NH, WIN))
    for c in range(4):
        kT = sb((P, WIN), BF16)
        tr_to(kT, wrows[:, c * P:(c + 1) * P])
        nisa.nc_matmul(dst=sw, stationary=qc[:, c:NH * 4:4], moving=kT)
    s_w = sb((NH, WIN), F32)
    nisa.scalar_tensor_tensor(dst=s_w, data=sw, op0=nl.multiply, operand0=SCALE, op1=nl.add, operand1=wmask)
    m = sb((NH, 1), F32)
    nisa.tensor_reduce(dst=m, op=nl.maximum, data=s_w, axis=1)
    if T:
        NCH = T // P
        crows = sb((P, NCH, HD), BF16)                    # crows[p, j, :] = ckv[j*128 + p]
        nisa.dma_copy(dst=crows, src=ckv.ap(pattern=[[HD, P], [P * HD, NCH], [1, HD]], offset=0))
        if cpatch is not None:                            # the compressed rows this token (and its group) add
            for pi in range(len(cpatch)):
                KI.KL_patch(crows, cpatch[pi][0], cpatch[pi][1], NCH, HD)
        if mask_sb is not None:
            cm = mask_sb
        else:
            cm = sb((NH, T), F32)
            nisa.dma_copy(dst=cm, src=cmask.ap(pattern=[[0, NH], [1, T]], offset=0))
        s_c = sb((NH, T), F32)
        CH = 512
        for t0 in range(0, T, CH):
            scp = ps((NH, CH))
            for c in range(4):
                gT = sb((P, CH), BF16)
                for jj in range(CH // P):
                    j = t0 // P + jj
                    tr_to(gT[:, jj * P:(jj + 1) * P], crows[:, j, c * P:(c + 1) * P])
                nisa.nc_matmul(dst=scp, stationary=qc[:, c:NH * 4:4], moving=gT)
            nisa.scalar_tensor_tensor(dst=s_c[:, t0:t0 + CH], data=scp, op0=nl.multiply, operand0=SCALE, op1=nl.add,
                                      operand1=cm[:, t0:t0 + CH])
        m2 = sb((NH, 1), F32)
        nisa.tensor_reduce(dst=m2, op=nl.maximum, data=s_c, axis=1)
        nisa.tensor_tensor(dst=m, data1=m, data2=m2, op=nl.maximum)
    negm = sb((NH, 1), F32)
    nisa.tensor_scalar(dst=negm, data=m, op0=nl.multiply, operand0=-1.0)
    # p = exp(s - m), row sums
    p_w = sb((NH, WIN), F32)
    sum_w = sb((NH, 1), F32)
    d_w = sb((NH, WIN), F32)
    nisa.tensor_scalar(dst=d_w, data=s_w, op0=nl.add, operand0=negm)
    lib.exp_precise(p_w, d_w)                          # exact exp, as the reference (was the table exp)
    nisa.tensor_reduce(dst=sum_w, op=nl.add, data=p_w, axis=1)
    pb_w = sb((NH, WIN), BF16)
    nisa.tensor_copy(dst=pb_w, src=p_w)
    tot = sb((NH, 1), F32)
    nisa.tensor_copy(dst=tot, src=sum_w)
    acc = ps((NH, HD))
    pT = sb((WIN, NH), BF16)
    tr_to(pT, pb_w)
    nisa.nc_matmul(dst=acc, stationary=pT, moving=wrows)
    if T:
        p_c = sb((NH, T), F32)
        sum_c = sb((NH, 1), F32)
        d_c = sb((NH, T), F32)
        nisa.tensor_scalar(dst=d_c, data=s_c, op0=nl.add, operand0=negm)
        lib.exp_precise(p_c, d_c)                      # exact exp, as the reference (was the table exp)
        nisa.tensor_reduce(dst=sum_c, op=nl.add, data=p_c, axis=1)
        pb_c = sb((NH, T), BF16)
        nisa.tensor_copy(dst=pb_c, src=p_c)
        nisa.tensor_tensor(dst=tot, data1=tot, data2=sum_c, op=nl.add)
        for j in range(T // P):
            pTc = sb((P, NH), BF16)
            tr_to(pTc, pb_c[:, j * P:(j + 1) * P])
            nisa.nc_matmul(dst=acc, stationary=pTc, moving=crows[:, j, :])
    # denominator += exp(sink - m)
    sk = sb((NH, 1), F32)
    nisa.dma_copy(dst=sk, src=sink.ap(pattern=[[1, NH], [1, 1]], offset=0))
    es = sb((NH, 1), F32)
    dsk = sb((NH, 1), F32)
    nisa.tensor_tensor(dst=dsk, data1=sk, data2=negm, op=nl.add)
    lib.exp_precise(es, dsk)                           # exact exp, as the reference (was the table exp)
    nisa.tensor_tensor(dst=tot, data1=tot, data2=es, op=nl.add)
    rden = sb((NH, 1), F32)
    B.recip_precise(rden, tot)
    o = sb((NH, HD), BF16)
    nisa.tensor_scalar(dst=o, data=acc, op0=nl.multiply, operand0=rden)
    # inverse RoPE on the last 64 dims (row layout, all heads)
    cs = B.load_rope_r(roper, NH)
    B.rope_r_row(o[:, HD - P:HD], 0, NH, cs, inverse=True)
    return o


PRE_OA = 8    # wo_a tiles (of this core's first group) prefetched before the attention core
PRE_OB = 2    # wo_b tiles prefetched before wo_a


def prefetch_woa(woa):
    GK = HD * NH // NG
    per = OL // P * (GK // P) * P
    g0, g1 = lib.lnc_split(NG)
    return lib.prefetch(woa[:, g0 * per:(g0 + 1) * per], OL, GK, PRE_OA, split=False)


def out_proj(o, woa, wob, ident, pre_oa=None):
    """act_quant(o) per 32 -> wo_a (8 groups, bf16 weights) -> act_quant -> wo_b.  Returns R [40, 128] bf16."""
    pre_ob = lib.prefetch(wob, B.D, NG * OL, PRE_OB) if pre_oa is not None else None
    oq = lib.act_quant_r(o)                                    # [64, 512] bf16
    # oc[p, h, c] = oq[h, c*128 + p]
    oc = sb((P, NH, 4), BF16)
    for c in range(4):
        tmp = sb((P, NH), BF16)
        tr_to(tmp, oq[:, c * P:(c + 1) * P])
        nisa.tensor_copy(dst=oc[:, :, c], src=tmp)
    oflat = oc.reshape((P, NH * 4))                            # column (h*4 + c)
    GK = HD * NH // NG                                          # 4096 per group
    KT = GK // P                                                # 32
    o2 = sb((P, NG * (OL // P)), BF16)                          # C layout of [8192]
    per = OL // P * KT * P                                      # elements per partition per group
    g0, g1 = lib.lnc_split(NG)                                  # groups split over the two physical cores
    for g in range(g0, g1):
        yg = lib.gemv_c(oflat[:, g * KT:(g + 1) * KT], woa[:, g * per:(g + 1) * per], OL, GK, split=False,
                        pre=pre_oa if g == g0 else None)
        nisa.tensor_copy(dst=o2[:, g * (OL // P):(g + 1) * (OL // P)], src=yg)
    lib.exchange_halves(o2, NG * (OL // P))
    o2r = lib.c_to_r(o2, ident)                                 # [64, 128]
    o2q = lib.act_quant_r(o2r)
    o2c = lib.r_to_c(o2q, ident)
    x = lib.gemv_c(o2c, wob, B.D, NG * OL, pre=pre_ob)          # C [128, 40]
    return lib.c_to_r(x, ident)                                 # R [40, 128]


def out_proj_m(os_, woa, wob, ident, mixst=None, nsk=0):
    """out_proj for several tokens, the wo_a / wo_b weight tiles loaded once (per-token arithmetic unchanged).
    mixst: a k_batch.hc_mixes_m_start state whose nsk Sinkhorn iterations are issued between the wo_a groups
    (they overlap the weight stream instead of idling it)."""
    M = len(os_)
    oflats = []
    tsplit = False
    for o in os_:
        if o is None:     # token attended by the other physical core (k_layer: lib.tok_mine)
            tsplit = True
            oflats.append(sb((P, NH * 4), BF16))
            continue
        oq = lib.act_quant_r(o)
        oc = sb((P, NH, 4), BF16)
        for c in range(4):
            tmp = sb((P, NH), BF16)
            tr_to(tmp, oq[:, c * P:(c + 1) * P])
            nisa.tensor_copy(dst=oc[:, :, c], src=tmp)
        oflats.append(oc.reshape((P, NH * 4)))
    if tsplit:
        # core 0 owns tokens 0 .. M/2-1 and runs wo_a groups 0-3 (columns 0..127 = heads 0-31), core 1 the rest:
        # each sends the other core's half of its own tokens and receives its half of the other tokens
        pid = nl.program_id(0)
        HC_ = NH * 4 // 2
        for k in range(M // 2):
            a = k
            b = k + M // 2
            if pid == 0:
                nisa.sendrecv(src=oflats[a][:, HC_:2 * HC_], dst=oflats[b][:, 0:HC_], send_to_rank=1,
                              recv_from_rank=1, pipe_id=0)
            else:
                nisa.sendrecv(src=oflats[b][:, 0:HC_], dst=oflats[a][:, HC_:2 * HC_], send_to_rank=0,
                              recv_from_rank=0, pipe_id=0)
    GK = HD * NH // NG
    KT = GK // P
    o2s = []
    for m in range(M):
        o2s.append(sb((P, NG * (OL // P)), BF16))
    per = OL // P * KT * P
    g0, g1 = lib.lnc_split(NG)
    done = 0
    for g in range(g0, g1):
        ins = []
        for m in range(M):
            ins.append(oflats[m][:, g * KT:(g + 1) * KT])
        ygs = lib.gemv_cm(ins, woa[:, g * per:(g + 1) * per], OL, GK, split=False)
        for m in range(M):
            nisa.tensor_copy(dst=o2s[m][:, g * (OL // P):(g + 1) * (OL // P)], src=ygs[m])
        if mixst is not None:
            want = (nsk * (g - g0 + 1)) // (g1 - g0)
            while done < want:
                KB.hc_mixes_m_iter(mixst)
                done = done + 1
    if mixst is not None:
        while done < nsk:
            KB.hc_mixes_m_iter(mixst)
            done = done + 1
    o2cs = []
    for m in range(M):
        lib.exchange_halves(o2s[m], NG * (OL // P))
        o2r = lib.c_to_r(o2s[m], ident)
        o2q = lib.act_quant_r(o2r)
        o2cs.append(lib.r_to_c(o2q, ident))
    xs = lib.gemv_cm(o2cs, wob, B.D, NG * OL)
    res = []
    for m in range(M):
        res.append(lib.c_to_r(xs[m], ident))
    return res


def attn_out_impl(h, mix, posv, qc_hbm, win, kvr, sink, roper, ckv, cmask, woa, wob, fnT, hsc, hbs, fnw, wgT, gbias, T):
    ident = lib.identity_sb(P)
    ones40 = lib.ones_sb(B.NR, B.NR)
    ones1 = lib.ones_sb(1, B.NR)
    pv = sb((1, 8), I32)
    nisa.dma_copy(dst=pv, src=posv)
    qc = sb((P, NH * 4), BF16)
    nisa.dma_copy(dst=qc, src=qc_hbm)
    o = attn_core(qc, win, kvr, pv, sink, roper, ckv, cmask, T)
    o_out = nl.ndarray((NH, HD), dtype=BF16, buffer=nl.shared_hbm)
    nisa.dma_copy(dst=o_out, src=o)
    xa = out_proj(o, woa, wob, ident)
    # hc_post with the attention mixes
    h_r = B.load_h_r(h)
    mixrow = sb((1, 24), F32)
    nisa.dma_copy(dst=mixrow, src=mix)
    mixb = B.bcast_parts(mixrow, B.NR, ones1)
    h2 = B.hc_post(xa, h_r, mixb)
    h2_out = nl.ndarray((4, B.D), dtype=BF16, buffer=nl.shared_hbm)
    B.store_h_r(h2_out, h2)
    fmix = B.hc_mixes(h2, fnT, hsc, hbs, ident, ones40)
    fmix_out = nl.ndarray((1, 24), dtype=F32, buffer=nl.shared_hbm)
    nisa.dma_copy(dst=fmix_out, src=fmix)
    x = B.hc_pre(h2, mixb, 0)
    g = sb((B.NR, P), BF16)
    nisa.dma_copy(dst=g, src=fnw)
    xn = lib.rmsnorm_r(x, g, B.D, B.EPS, ones40)
    xn_out = nl.ndarray((B.NR, P), dtype=BF16, buffer=nl.shared_hbm)
    nisa.dma_copy(dst=xn_out, src=xn)
    xa_out = nl.ndarray((B.NR, P), dtype=BF16, buffer=nl.shared_hbm)
    nisa.dma_copy(dst=xa_out, src=xa)
    ridx, rw = KM.gate_route(xn, wgT, gbias, ident)
    ridx_out = nl.ndarray((1, 8), dtype=I32, buffer=nl.shared_hbm)
    nisa.dma_copy(dst=ridx_out, src=ridx)
    rw_out = nl.ndarray((1, 8), dtype=F32, buffer=nl.shared_hbm)
    nisa.dma_copy(dst=rw_out, src=rw)
    return h2_out, fmix_out, xn_out, ridx_out, rw_out, xa_out, o_out


@nki.jit
def attn_out_w(h, mix, posv, qc, win, kvr, sink, roper, woa, wob, fnT, hsc, hbs, fnw, wgT, gbias):
    return attn_out_impl(h, mix, posv, qc, win, kvr, sink, roper, None, None, woa, wob, fnT, hsc, hbs, fnw, wgT, gbias,
                         0)


@nki.jit
def attn_out_c(h, mix, posv, qc, win, kvr, sink, roper, ckv, cmask, woa, wob, fnT, hsc, hbs, fnw, wgT, gbias):
    return attn_out_impl(h, mix, posv, qc, win, kvr, sink, roper, ckv, cmask, woa, wob, fnT, hsc, hbs, fnw, wgT, gbias,
                         cmask.shape[1])


LL = 640           # attention list entries: window (<= 128) then the selected compressed keys (<= 512)
NBLK = LL // 64


def onehot_rows(rank_col, n_list, iota_l):
    """[128 keys, n_list] bf16 one-hot: key p -> list position rank_col[p, 0] (-1: none).  iota_l: [128, n_list] f32."""
    g = sb((P, n_list), BF16)
    nisa.tensor_scalar(dst=g, data=iota_l, op0=nl.equal, operand0=rank_col)
    return g


def attn_core_list(qc, wrows, wrank_hbm, lvalid_hbm, sink, roper, crows, crank_sb, NCH, win_only):
    """sparse_attn for one query exactly as the reference computes it (deepseek_cpu.sparse_attn): the keys are put
    in the reference's list order (window entries, then the selected compressed keys ascending) by one-hot
    matmuls (exact), then an online softmax over blocks of 64 list entries: running max per block, P rounded to
    bf16 relative to its block's running max, fp32 rescaling, the sink in the denominator.
    wrows: [128 slots, 512] bf16 (patched ring); wrank_hbm: [1, 128] list position per slot (-1 none);
    lvalid_hbm: [1, LL] 1/0; crows: [128, NCH, 512] bf16 compressed rows (key j*128 + p) or None;
    crank_sb: [128, NCH] fp32 list position per compressed key (-1 none)."""
    nl_ = P if win_only else LL
    nb = nl_ // 64
    io = sb((P, nl_), I32)
    nisa.iota(dst=io, pattern=[[1, nl_]], offset=0, channel_multiplier=0)
    iol = sb((P, nl_), F32)
    nisa.tensor_copy(dst=iol, src=io)
    wr = sb((P, 1), F32)
    nisa.dma_copy(dst=wr, src=wrank_hbm.ap(pattern=[[1, P], [1, 1]], offset=0))
    gw = onehot_rows(wr, nl_, iol)
    gcs = []
    if crows is not None:
        for j in range(NCH):
            gcs.append(onehot_rows(crank_sb[:, j:j + 1], nl_, iol))
    # list-ordered K/V rows: vl[lc] = [128 list entries, 512]
    vls = []
    for lc in range(nl_ // P):
        pt = ps((P, HD))
        first = True
        if lc == 0:
            nisa.nc_matmul(dst=pt, stationary=gw[:, 0:P], moving=wrows, accumulate=False)
            first = False
        if crows is not None and (lc > 0 or NCH * P <= 512):
            for j in range(NCH):
                nisa.nc_matmul(dst=pt, stationary=gcs[j][:, lc * P:(lc + 1) * P], moving=crows[:, j, :],
                               accumulate=not first)
                first = False
        v = sb((P, HD), BF16)
        nisa.tensor_copy(dst=v, src=pt)
        vls.append(v)
    nlc = len(vls)
    # scores over the list: s[h, i] = q_h . v_i * scale
    s = sb((NH, nl_), F32)
    for c0 in range(0, nlc, 4):
        c1 = min(nlc, c0 + 4)
        sp = ps((NH, (c1 - c0) * P))
        for c in range(4):
            vt = sb((P, (c1 - c0) * P), BF16)
            for lc in range(c0, c1):
                tr_to(vt[:, (lc - c0) * P:(lc - c0 + 1) * P], vls[lc][:, c * P:(c + 1) * P])
            nisa.nc_matmul(dst=sp, stationary=qc[:, c:NH * 4:4], moving=vt)
        nisa.tensor_scalar(dst=s[:, c0 * P:c1 * P], data=sp, op0=nl.multiply, operand0=SCALE)
    # invalid list entries -> NEG
    lv = sb((NH, nl_), F32)
    nisa.dma_copy(dst=lv, src=lvalid_hbm.ap(pattern=[[0, NH], [1, nl_]], offset=0))
    pen = sb((NH, nl_), F32)
    nisa.tensor_scalar(dst=pen, data=lv, op0=nl.subtract, operand0=1.0, op1=nl.multiply, operand1=-NEG)
    nisa.tensor_tensor(dst=s, data1=s, data2=pen, op=nl.add)
    # block maxima, running max (starts at -1e30, as the reference's)
    bm = sb((NH, nb), F32)
    nisa.tensor_reduce(dst=bm, op=nl.maximum, data=s.reshape((NH, nb, 64)), axis=2)
    nisa.tensor_scalar(dst=bm, data=bm, op0=nl.maximum, operand0=NEG)
    ones_nb = lib.ones_sb(NH, nb)
    rm = sb((NH, nb), F32)
    nisa.tensor_tensor_scan(dst=rm, data0=ones_nb, data1=bm, initial=NEG, op0=nl.multiply, op1=nl.maximum)
    # p = exp(s - m_block); invalid entries: s = NEG - ... -> 0
    d = sb((NH, nl_), F32)
    nisa.tensor_tensor(dst=d.reshape((NH, nb, 64)), data1=s.reshape((NH, nb, 64)),
                       data2=rm.reshape((NH, nb, 1)).broadcast(2, 64), op=nl.subtract)
    p = sb((NH, nl_), F32)
    lib.exp_precise(p, d)                              # Q11 test: precise exp instead of the table
    nisa.tensor_tensor(dst=p, data1=p, data2=lv, op=nl.multiply)
    pb = sb((NH, nl_), BF16)
    nisa.tensor_copy(dst=pb, src=p)
    # fac_b = exp(m_b - m_final); p' = bf16(p) * fac (fp32); den = sum p * fac + exp(sink - m_final)
    mf = rm[:, nb - 1:nb]
    negmf = sb((NH, 1), F32)
    nisa.tensor_scalar(dst=negmf, data=mf, op0=nl.multiply, operand0=-1.0)
    fac = sb((NH, nb), F32)
    dm = sb((NH, nb), F32)
    nisa.tensor_scalar(dst=dm, data=rm, op0=nl.add, operand0=negmf)
    lib.exp_precise(fac, dm)
    pf = sb((NH, nl_), F32)
    nisa.tensor_tensor(dst=pf.reshape((NH, nb, 64)), data1=pb.reshape((NH, nb, 64)),
                       data2=fac.reshape((NH, nb, 1)).broadcast(2, 64), op=nl.multiply)
    bs = sb((NH, nb), F32)
    nisa.tensor_reduce(dst=bs, op=nl.add, data=p.reshape((NH, nb, 64)), axis=2)
    nisa.tensor_tensor(dst=bs, data1=bs, data2=fac, op=nl.multiply)
    den = sb((NH, 1), F32)
    nisa.tensor_reduce(dst=den, op=nl.add, data=bs, axis=1)
    sk = sb((NH, 1), F32)
    nisa.dma_copy(dst=sk, src=sink.ap(pattern=[[1, NH], [1, 1]], offset=0))
    es = sb((NH, 1), F32)
    dsk = sb((NH, 1), F32)
    nisa.tensor_tensor(dst=dsk, data1=sk, data2=negmf, op=nl.add)
    lib.exp_precise(es, dsk)
    nisa.tensor_tensor(dst=den, data1=den, data2=es, op=nl.add)
    # acc = p' @ V (fp32 matmul: p' is not a bf16 value)
    acc = ps((NH, HD))
    for lc in range(nlc):
        pt = ps((P, NH), F32)
        nisa.nc_transpose(dst=pt, data=pf[:, lc * P:(lc + 1) * P])
        pT = sb((P, NH), F32)
        nisa.tensor_copy(dst=pT, src=pt)
        vf = sb((P, HD), F32)
        nisa.tensor_copy(dst=vf, src=vls[lc])
        nisa.nc_matmul(dst=acc, stationary=pT, moving=vf)
    rden = sb((NH, 1), F32)
    B.recip_precise(rden, den)
    o = sb((NH, HD), BF16)
    nisa.tensor_scalar(dst=o, data=acc, op0=nl.multiply, operand0=rden)
    cs = B.load_rope_r(roper, NH)
    B.rope_r_row(o[:, HD - P:HD], 0, NH, cs, inverse=True)
    return o
