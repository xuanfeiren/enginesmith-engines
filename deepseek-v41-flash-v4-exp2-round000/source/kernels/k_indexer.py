"""Kernel: the sparse-attention indexer of an index-source layer for one decode token, when more than 512
compressed positions exist.  Writes the selected compressed rows (ascending) to an int32 [512] output.

Scores as the official code rounds them: bf16(q_h . k_t), relu, * bf16 weight -> bf16, sum over heads (fp32) ->
bf16.  Top-512 exactly as CUDA's radix select: every key above the 512th key, then equal keys by lowest index.
Keys are the radix order of the bf16 bit patterns.  T (compile-time) bounds the compressed positions read.
"""
import nki
import nki.language as nl
import nki.isa as nisa

import lib
import blocks as B
from lib import P, F32, BF16, I32, U32, sb, ps

NIH = 32
IHD = 128
QL = 1280
K_SEL = 512


def floor_half(dst, x):
    """dst = floor(x) for x = n/2, 0 <= x < 2^21 (fp32): RNE of x - 0.25 via the 2^23 magic number."""
    t = sb(x.shape, F32)
    nisa.tensor_scalar(dst=t, data=x, op0=nl.add, operand0=8388608.0 - 0.25)
    nisa.tensor_scalar(dst=dst, data=t, op0=nl.subtract, operand0=8388608.0)


def excl_prefix(dst_ps, x, ones_pp, lstrict, ones_row_j):
    """dst_ps [128, J] (PSUM fp32) = number of x == 1 entries before (p, j) in the order t = j*128 + p.
    x: [128, J] fp32 0/1 tile."""
    J = x.shape[1]
    inc = sb((P, J), F32)
    nisa.tensor_tensor_scan(dst=inc, data0=ones_row_j, data1=x, initial=0.0, op0=nl.multiply, op1=nl.add)
    a = sb((P, J), F32)
    nisa.tensor_tensor(dst=a, data1=inc, data2=x, op=nl.subtract)          # exclusive prefix along j
    nisa.nc_matmul(dst=dst_ps, stationary=ones_pp, moving=a, accumulate=False)   # sum over all partitions, j' < j
    nisa.nc_matmul(dst=dst_ps, stationary=lstrict, moving=x, accumulate=True)    # same column, p' < p


def indexer_impl(qr, xn, posv, wqb, wpT, rotP, ropec, kc, lstrict_hbm, T):
    ident = lib.identity_sb(P)
    pv = sb((1, 8), I32)
    nisa.dma_copy(dst=pv, src=posv)
    qrr = sb((QL // P, P), BF16)
    nisa.dma_copy(dst=qrr, src=qr)
    xr = sb((B.NR, P), BF16)
    nisa.dma_copy(dst=xr, src=xn)
    rot = sb((P, P), BF16)
    nisa.dma_copy(dst=rot, src=rotP)
    cosc, sinc = B.load_rope_c(ropec)
    sel = indexer_core(qrr, xr, pv, wqb, wpT, rot, cosc, sinc, kc, lstrict_hbm, T, ident, None)
    out = nl.ndarray((P, T // P), dtype=F32, buffer=nl.shared_hbm)
    nisa.dma_copy(dst=out, src=sel)
    return out


def indexer_q_in(qrr, ident):
    """the wq_b input: act_quant(qr) in C layout"""
    qq = lib.act_quant_r(qrr)
    return lib.r_to_c(qq, ident)


def indexer_core(qrr, xr, pv, wqb, wpT, rot, cosc, sinc, kc, lstrict_hbm, T, ident, kpatch, q_pre=None, wp_sb=None):
    """SBUF inputs (qr R [10,128], xn R [40,128], pv, rotation + cos/sin columns).  kpatch: list of (row broadcast
    [128, 128], crow [128,1] fp32) replacing index-key rows (the keys this token and earlier ones of its group add),
    or None.  q_pre: wq_b(act_quant(qr)) already computed (C [128, 32]).
    Returns the selection [128, T/128] fp32 (1.0 selected; position j*128 + p)."""
    J = T // P
    # ---- q: wq_b(qr) (fp8 Linear), RoPE, fp4 quant per 32 (E8M0)
    if q_pre is None:
        qc_in = indexer_q_in(qrr, ident)
        q = lib.gemv_c(qc_in, wqb, NIH * IHD, QL)                     # C [128, 32]: column = head
    else:
        q = sb((P, NIH), BF16)
        nisa.tensor_copy(dst=q, src=q_pre)
    B.rope_c_cols(q, rot, cosc, sinc, q)
    qr_ = lib.c_to_r(q, ident)                                         # [32, 128]
    qf = lib.fp4_act_quant_r(qr_, 32, False)
    qT = lib.r_to_c(qf, ident)                                         # [128 d, 32 h] bf16
    xc = lib.r_to_c(xr, ident)
    if wp_sb is None:
        wp = sb((P, B.NR * NIH), BF16)
        nisa.dma_copy(dst=wp, src=wpT)
    else:
        wp = wp_sb
    # ---- head weights: weights_proj(x) * (128^-0.5 * 32^-0.5) as bf16, broadcast over partitions
    wps = ps((1, NIH))
    for kt in range(B.NR):
        nisa.nc_matmul(dst=wps, stationary=xc[:, kt:kt + 1], moving=wp[:, kt * NIH:(kt + 1) * NIH])
    wrow = sb((1, NIH), BF16)
    nisa.tensor_copy(dst=wrow, src=wps)                                # F.linear in bf16 -> bf16
    wrow2 = sb((1, NIH), F32)
    nisa.tensor_scalar(dst=wrow2, data=wrow, op0=nl.multiply, operand0=float(IHD ** -0.5 * NIH ** -0.5))
    wrb = sb((1, NIH), BF16)
    nisa.tensor_copy(dst=wrb, src=wrow2)
    ones1 = lib.ones_sb(1, P)
    wf = sb((1, NIH), F32)
    nisa.tensor_copy(dst=wf, src=wrb)
    wb = B.bcast_parts(wf, P, ones1)                                   # [128, 32]
    # ---- scores in column layout: key[p, j] for t = j*128 + p
    score = sb((P, J), F32)
    G = 16                                                             # 16 blocks of 128 per PSUM tile
    for g0 in range(0, J, G):
        g1 = min(J, g0 + G)
        rows = sb((P, g1 - g0, IHD), BF16)
        nisa.dma_copy(dst=rows, src=kc.ap(pattern=[[IHD, P], [P * IHD, g1 - g0], [1, IHD]], offset=g0 * P * IHD))
        if kpatch is not None:
            # replace rows crow (in this group: index crow - g0*128) by the new keys
            for pi in range(len(kpatch)):
                cr2 = sb((P, 1), F32)
                nisa.tensor_scalar(dst=cr2, data=kpatch[pi][1], op0=nl.subtract, operand0=float(g0 * P))
                KL_patch(rows, kpatch[pi][0], cr2, g1 - g0, IHD)
        sp = ps((P, (g1 - g0) * NIH))
        for jj in range(g1 - g0):
            kT = sb((P, P), BF16)
            pt = ps((P, P), BF16)
            nisa.nc_transpose(dst=pt, data=rows[:, jj, :])
            nisa.tensor_copy(dst=kT, src=pt)
            nisa.nc_matmul(dst=sp[:, jj * NIH:(jj + 1) * NIH], stationary=kT, moving=qT, accumulate=False)
        n = (g1 - g0) * NIH
        sbf = sb((P, n), BF16)
        nisa.tensor_copy(dst=sbf, src=sp)                              # bf16(q . k)
        sw = sb((P, n), BF16)
        nisa.tensor_scalar(dst=sw, data=sbf, op0=nl.maximum, operand0=0.0)
        sw2 = sb((P, n), BF16)
        nisa.tensor_tensor(dst=sw2.reshape((P, g1 - g0, NIH)), data1=sw.reshape((P, g1 - g0, NIH)),
                           data2=wb.reshape((P, 1, NIH)).broadcast(1, g1 - g0), op=nl.multiply)
        tot = sb((P, g1 - g0), F32)
        nisa.tensor_reduce(dst=tot, op=nl.add, data=sw2.reshape((P, g1 - g0, NIH)), axis=2)
        totb = sb((P, g1 - g0), BF16)
        nisa.tensor_copy(dst=totb, src=tot)
        nisa.tensor_copy(dst=score[:, g0:g1], src=totb)
    # ---- radix keys of the bf16 scores; invalid positions (t >= T_c) -> -1
    sbf16 = sb((P, J), BF16)
    nisa.tensor_copy(dst=sbf16, src=score)
    u16 = sb((P, J), nl.uint16)
    nisa.tensor_copy(dst=u16, src=sbf16.view(nl.uint16))
    uf = sb((P, J), F32)
    nisa.tensor_copy(dst=uf, src=u16)
    neg = sb((P, J), F32)
    nisa.tensor_scalar(dst=neg, data=uf, op0=nl.greater_equal, operand0=32768.0)
    t2 = sb((P, J), F32)
    nisa.tensor_scalar(dst=t2, data=uf, op0=nl.multiply, operand0=2.0, op1=nl.subtract, operand1=32767.0)
    nisa.tensor_tensor(dst=t2, data1=t2, data2=neg, op=nl.multiply)
    key = sb((P, J), F32)
    nisa.tensor_scalar(dst=key, data=uf, op0=nl.add, operand0=32768.0)
    nisa.tensor_tensor(dst=key, data1=key, data2=t2, op=nl.subtract)
    io = sb((P, J), I32)
    nisa.iota(dst=io, pattern=[[P, J]], offset=0, channel_multiplier=1)    # t = j*128 + p
    tf = sb((P, J), F32)
    nisa.tensor_copy(dst=tf, src=io)
    tcf = sb((1, 1), F32)
    nisa.tensor_copy(dst=tcf, src=pv[:, B.PV_TC:B.PV_TC + 1])
    tcb = B.bcast_parts(tcf, P, ones1)                                 # [128, 1]
    valid = sb((P, J), F32)
    nisa.tensor_scalar(dst=valid, data=tf, op0=nl.less, operand0=tcb)
    # key = valid ? key : -1   ==  (key + 1) * valid - 1
    nisa.tensor_scalar(dst=key, data=key, op0=nl.add, operand0=1.0)
    nisa.tensor_tensor(dst=key, data1=key, data2=valid, op=nl.multiply)
    nisa.tensor_scalar(dst=key, data=key, op0=nl.subtract, operand0=1.0)
    # ---- binary search: largest v with count(key >= v) >= 512
    ones_pp = lib.ones_sb(P, P)
    lo = sb((P, 1), F32)
    nisa.memset(dst=lo, value=0.0)
    hi = sb((P, 1), F32)
    nisa.memset(dst=hi, value=65535.0)
    for _ in range(16):
        s_ = sb((P, 1), F32)
        nisa.tensor_tensor(dst=s_, data1=lo, data2=hi, op=nl.add)
        nisa.tensor_scalar(dst=s_, data=s_, op0=nl.add, operand0=1.0, op1=nl.multiply, operand1=0.5)
        mid = sb((P, 1), F32)
        floor_half(mid, s_)
        ge = sb((P, J), F32)
        nisa.tensor_scalar(dst=ge, data=key, op0=nl.greater_equal, operand0=mid)
        cnt = sb((P, 1), F32)
        nisa.tensor_reduce(dst=cnt, op=nl.add, data=ge, axis=1)
        tot = lib.allsum_partitions(cnt, ones_pp)
        cond = sb((P, 1), F32)
        nisa.tensor_scalar(dst=cond, data=tot, op0=nl.greater_equal, operand0=float(K_SEL))
        # lo += cond * (mid - lo);  hi = mid - 1 + cond * (hi - mid + 1)
        d1 = sb((P, 1), F32)
        nisa.tensor_tensor(dst=d1, data1=mid, data2=lo, op=nl.subtract)
        nisa.tensor_tensor(dst=d1, data1=d1, data2=cond, op=nl.multiply)
        nisa.tensor_tensor(dst=lo, data1=lo, data2=d1, op=nl.add)
        d2 = sb((P, 1), F32)
        nisa.tensor_tensor(dst=d2, data1=hi, data2=mid, op=nl.subtract)
        nisa.tensor_scalar(dst=d2, data=d2, op0=nl.add, operand0=1.0)
        nisa.tensor_tensor(dst=d2, data1=d2, data2=cond, op=nl.multiply)
        nisa.tensor_scalar(dst=hi, data=mid, op0=nl.subtract, operand0=1.0)
        nisa.tensor_tensor(dst=hi, data1=hi, data2=d2, op=nl.add)
    # ---- selection: key > v, plus the first `need` keys == v in index order
    gt = sb((P, J), F32)
    nisa.tensor_scalar(dst=gt, data=key, op0=nl.greater, operand0=lo)
    eq = sb((P, J), F32)
    nisa.tensor_scalar(dst=eq, data=key, op0=nl.equal, operand0=lo)
    cg = sb((P, 1), F32)
    nisa.tensor_reduce(dst=cg, op=nl.add, data=gt, axis=1)
    ngt = lib.allsum_partitions(cg, ones_pp)
    need = sb((P, 1), F32)
    nisa.tensor_scalar(dst=need, data=ngt, op0=nl.multiply, operand0=-1.0, op1=nl.add, operand1=float(K_SEL))
    lst = sb((P, P), F32)
    nisa.dma_copy(dst=lst, src=lstrict_hbm)
    onesj = lib.ones_sb(P, J)
    rk = ps((P, J))
    excl_prefix(rk, eq, ones_pp, lst, onesj)
    take = sb((P, J), F32)
    nisa.tensor_scalar(dst=take, data=rk, op0=nl.less, operand0=need)
    nisa.tensor_tensor(dst=take, data1=take, data2=eq, op=nl.multiply)
    sel = sb((P, J), F32)
    nisa.tensor_tensor(dst=sel, data1=gt, data2=take, op=nl.add)
    return sel


def sel_listpos(sel, J, lstrict_hbm):
    """List position of each selected key in the reference's attention list (128 window entries first, then the
    selected compressed keys ascending): 128 + rank; -1 for keys not selected.  [128, J] fp32 (key j*128 + p)."""
    ones_pp = lib.ones_sb(P, P)
    lst = sb((P, P), F32)
    nisa.dma_copy(dst=lst, src=lstrict_hbm)
    onesj = lib.ones_sb(P, J)
    rk = ps((P, J))
    excl_prefix(rk, sel, ones_pp, lst, onesj)
    t = sb((P, J), F32)
    nisa.tensor_scalar(dst=t, data=rk, op0=nl.add, operand0=129.0)
    lp = sb((P, J), F32)
    nisa.tensor_tensor(dst=lp, data1=t, data2=sel, op=nl.multiply)
    nisa.tensor_scalar(dst=lp, data=lp, op0=nl.subtract, operand0=1.0)
    return lp


def sel_blocks(sel, J, lstrict_hbm):
    """Block index of each selected key in the reference's attention list: the window fills list entries 0..127,
    the selected compressed keys follow in ascending position order, blocks are 64 entries: 2 + rank // 64;
    -1 for keys not selected.  sel / result: [128, J] fp32 (position j*128 + p)."""
    ones_pp = lib.ones_sb(P, P)
    lst = sb((P, P), F32)
    nisa.dma_copy(dst=lst, src=lstrict_hbm)
    onesj = lib.ones_sb(P, J)
    rk = ps((P, J))
    excl_prefix(rk, sel, ones_pp, lst, onesj)
    # floor(rank / 64) with integer ops (rank is an exact integer in fp32)
    rf = sb((P, J), F32)
    nisa.tensor_copy(dst=rf, src=rk)
    ri = sb((P, J), I32)
    nisa.tensor_copy(dst=ri, src=rf)
    nisa.tensor_scalar(dst=ri, data=ri, op0=nl.right_shift, operand0=6)
    t = sb((P, J), F32)
    nisa.tensor_copy(dst=t, src=ri)
    nisa.tensor_scalar(dst=t, data=t, op0=nl.add, operand0=3.0)                     # 2 + floor + 1
    # blk = sel ? (2 + floor) : -1  ==  sel * (2 + floor + 1) - 1
    blk = sb((P, J), F32)
    nisa.tensor_tensor(dst=blk, data1=t, data2=sel, op=nl.multiply)
    nisa.tensor_scalar(dst=blk, data=blk, op0=nl.subtract, operand0=1.0)
    return blk


def KL_patch(rows, newrow_b, crow_f, J, width):
    """rows [128, J, width]: replace row crow (at (crow % 128, crow // 128)) by newrow_b [128, width]."""
    io = sb((P, J), I32)
    nisa.iota(dst=io, pattern=[[P, J]], offset=0, channel_multiplier=1)   # value = p + 128 j
    iof = sb((P, J), F32)
    nisa.tensor_copy(dst=iof, src=io)
    pred = sb((P, J), nl.uint8)
    nisa.tensor_scalar(dst=pred, data=iof, op0=nl.equal, operand0=crow_f)
    nisa.tensor_copy_predicated(dst=rows, src=newrow_b.reshape((P, 1, width)).broadcast(1, J),
                                predicate=pred.reshape((P, J, 1)).broadcast(2, width))


@nki.jit
def indexer_1024(qr, xn, posv, wqb, wpT, rotP, ropec, kc, lstrict):
    return indexer_impl(qr, xn, posv, wqb, wpT, rotP, ropec, kc, lstrict, 1024)


@nki.jit
def indexer_2048(qr, xn, posv, wqb, wpT, rotP, ropec, kc, lstrict):
    return indexer_impl(qr, xn, posv, wqb, wpT, rotP, ropec, kc, lstrict, 2048)


@nki.jit
def indexer_4096(qr, xn, posv, wqb, wpT, rotP, ropec, kc, lstrict):
    return indexer_impl(qr, xn, posv, wqb, wpT, rotP, ropec, kc, lstrict, 4096)


@nki.jit
def indexer_8192(qr, xn, posv, wqb, wpT, rotP, ropec, kc, lstrict):
    return indexer_impl(qr, xn, posv, wqb, wpT, rotP, ropec, kc, lstrict, 8192)
