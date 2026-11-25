"""Batched (M-token) versions of the per-token serial parts of the 4-token prefill kernel: the hyper-connection
mixes (fp32 projection + sigmoid + 20-iteration Sinkhorn) and the MoE router.  Every element is computed with the
instructions of the M=1 versions (blocks.hc_mixes, k_moe.gate_route); the M tokens sit on M partitions instead
of running one after another.  Reductions are per partition (per token), and every matmul output element is the
same dot product, so the values are those of the per-token code."""
import nki.language as nl
import nki.isa as nisa

import lib
import blocks as B
import k_moe as KM
from lib import P, F32, BF16, I32, U32, sb, ps

NE = 384
TOPK = 6
KT_D = B.D // P
ROUTE_SCALE = 1.5


def ident_f32(M):
    t = sb((M, M), F32)
    nisa.memset(dst=t, value=0.0)
    one = sb((M, M), F32)
    nisa.memset(dst=one, value=1.0)
    nisa.affine_select(dst=t, pattern=[[-1, M]], channel_multiplier=1, on_true_tile=one, on_false_value=0.0,
                       cmp_op=nl.equal)
    return t


def diag_col(full, M, identM):
    """full [>=M, M] fp32 (row-broadcast values) -> [M, 1] with entry m = full[m, m] (x*1 + 0s: exact)"""
    t = sb((M, M), F32)
    nisa.tensor_tensor(dst=t, data1=full[0:M, :], data2=identM, op=nl.multiply)
    d = sb((M, 1), F32)
    nisa.tensor_reduce(dst=d, op=nl.add, data=t, axis=1)
    return d


def div_rows_m(c3, den, M):
    r = sb((M, 4), F32)
    nisa.reciprocal(dst=r, data=den)
    nisa.tensor_tensor(dst=c3, data1=c3, data2=r.reshape((M, 4, 1)).broadcast(2, 4), op=nl.multiply)


def sk_colnorm_m(cT, M):
    csum = sb((M, 4), F32)
    nisa.tensor_reduce(dst=csum, op=nl.add, data=cT, axis=2)
    nisa.tensor_scalar(dst=csum, data=csum, op0=nl.add, operand0=B.HC_EPS)
    div_rows_m(cT, csum, M)


def sk_rownorm_m(c3, M):
    rs_ = sb((M, 4), F32)
    nisa.tensor_reduce(dst=rs_, op=nl.add, data=c3, axis=2)
    nisa.tensor_scalar(dst=rs_, data=rs_, op0=nl.add, operand0=B.HC_EPS)
    div_rows_m(c3, rs_, M)


def hc_mixes_m_start(h_rs, fn, scale_hbm, base_hbm, ident, ones40, identM):
    """hc_mixes_m up to the Sinkhorn iterations: returns the state (out, c, c3, cT)"""
    M = len(h_rs)
    NR, HC, KT = B.NR, B.HC, B.HCD // P
    xc = sb((P, KT, M), F32)
    ss = sb((NR, M), F32)
    for m in range(M):
        h_r = h_rs[m]
        for i in range(HC):
            pt = ps((P, NR), BF16)
            nisa.nc_transpose(dst=pt, data=h_r[:, i, :])
            nisa.tensor_copy(dst=xc[:, i * NR:(i + 1) * NR, m], src=pt)
        hf = sb((NR, HC * P), F32)
        nisa.tensor_copy(dst=hf, src=h_r.reshape((NR, HC * P)))
        sq = sb((NR, HC * P), F32)
        nisa.activation_reduce(dst=sq, op=nl.square, data=hf, reduce_op=nl.add, reduce_res=ss[:, m:m + 1])
    pt = ps((NR, M))
    nisa.nc_matmul(dst=pt, stationary=ones40, moving=ss)
    tot = sb((NR, M), F32)
    nisa.tensor_copy(dst=tot, src=pt)
    var = sb((NR, M), F32)
    nisa.tensor_scalar(dst=var, data=tot, op0=nl.multiply, operand0=1.0 / B.HCD, op1=nl.add, operand1=B.EPS)
    rs = sb((NR, M), F32)
    lib.rsqrt_precise(rs, var)
    rsd = diag_col(rs, M, identM)
    acc = ps((M, 24))
    for kt in range(KT):
        nisa.nc_matmul(dst=acc, stationary=xc[:, kt, :], moving=fn[:, kt * 24:(kt + 1) * 24])
    mixes = sb((M, 24), F32)
    nisa.tensor_scalar(dst=mixes, data=acc, op0=nl.multiply, operand0=rsd)
    sc = sb((M, 3), F32)
    nisa.dma_copy(dst=sc, src=scale_hbm.ap(pattern=[[0, M], [1, 3]], offset=0))
    bs = sb((M, 24), F32)
    nisa.dma_copy(dst=bs, src=base_hbm.ap(pattern=[[0, M], [1, 24]], offset=0))
    out = sb((M, 24), F32)
    t = sb((M, 24), F32)
    nisa.tensor_scalar(dst=t[:, 0:4], data=mixes[:, 0:4], op0=nl.multiply, operand0=sc[:, 0:1])
    nisa.tensor_tensor(dst=t[:, 0:4], data1=t[:, 0:4], data2=bs[:, 0:4], op=nl.add)
    lib.sigmoid_precise(out[:, 0:4], t[:, 0:4])
    nisa.tensor_scalar(dst=out[:, 0:4], data=out[:, 0:4], op0=nl.add, operand0=B.HC_EPS)
    nisa.tensor_scalar(dst=t[:, 4:8], data=mixes[:, 4:8], op0=nl.multiply, operand0=sc[:, 1:2])
    nisa.tensor_tensor(dst=t[:, 4:8], data1=t[:, 4:8], data2=bs[:, 4:8], op=nl.add)
    lib.sigmoid_precise(out[:, 4:8], t[:, 4:8])
    nisa.tensor_scalar(dst=out[:, 4:8], data=out[:, 4:8], op0=nl.multiply, operand0=2.0)
    c = sb((M, 16), F32)
    nisa.tensor_scalar(dst=c, data=mixes[:, 8:24], op0=nl.multiply, operand0=sc[:, 2:3])
    nisa.tensor_tensor(dst=c, data1=c, data2=bs[:, 8:24], op=nl.add)
    c3 = c.reshape((M, 4, 4))
    cT = c3.permute((0, 2, 1))
    rmax = sb((M, 4), F32)
    nisa.tensor_reduce(dst=rmax, op=nl.maximum, data=c3, axis=2)
    nisa.tensor_tensor(dst=c3, data1=c3, data2=rmax.reshape((M, 4, 1)).broadcast(2, 4), op=nl.subtract)
    lib.exp_precise(c, c)
    rsum = sb((M, 4), F32)
    nisa.tensor_reduce(dst=rsum, op=nl.add, data=c3, axis=2)
    div_rows_m(c3, rsum, M)
    nisa.tensor_scalar(dst=c, data=c, op0=nl.add, operand0=B.HC_EPS)
    sk_colnorm_m(cT, M)
    return out, c, c3, cT


def hc_mixes_m(h_rs, fn, scale_hbm, base_hbm, ident, ones40, identM):
    """blocks.hc_mixes for M tokens: h_rs list of [40, 4, 128] bf16, fn the SBUF [128, 160*24] fp32 projection.
    Returns [M, 24] fp32 (row m = token m)."""
    st = hc_mixes_m_start(h_rs, fn, scale_hbm, base_hbm, ident, ones40, identM)
    for _ in range(SK_ITERS):
        hc_mixes_m_iter(st)
    return hc_mixes_m_finish(st)


SK_ITERS = 20 - 1   # the Sinkhorn (row, column) iterations after the first column normalisation


def hc_mixes_m_iter(st):
    """one Sinkhorn (row, column) iteration of hc_mixes_m_start's state (callers may interleave these with other
    work: the instructions and their order on the data are those of hc_mixes_m)"""
    out, c, c3, cT = st
    M = out.shape[0]
    sk_rownorm_m(c3, M)
    sk_colnorm_m(cT, M)


def hc_mixes_m_finish(st):
    out, c, c3, cT = st
    nisa.tensor_copy(dst=out[:, 8:24], src=c)
    return out


def row_of(t, m, n):
    """row m of an SBUF [M, n] tile as a [1, n] tile at partition 0 (SBUF -> SBUF DMA: a byte copy)"""
    r = sb((1, n), F32)
    nisa.dma_copy(dst=r, src=t[m:m + 1, :])
    return r


def gate_route_m(xrs, wg, gbias, ident):
    """k_moe.gate_route for M tokens (xrs: list of R [40, 128] bf16, wg the SBUF router weights).
    Returns (idx [M, 8] int32, wts [M, 8] fp32), row m = token m."""
    M = len(xrs)
    xc = sb((P, KT_D, M), BF16)
    for m in range(M):
        pt = ps((P, B.NR), BF16)
        nisa.nc_transpose(dst=pt, data=xrs[m])
        nisa.tensor_copy(dst=xc[:, :, m], src=pt)
    sc, biased = KM.gate_scores(xc, wg, gbias, M)
    top = sb((M, 8), F32)
    nisa.max8(dst=top, src=biased)
    tidx = sb((M, 8), U32)
    nisa.nc_find_index8(dst=tidx, data=biased, vals=top)
    tidx_i = sb((M, 8), I32)
    nisa.tensor_copy(dst=tidx_i, src=tidx)
    tidx_f = sb((M, 8), F32)
    nisa.tensor_copy(dst=tidx_f, src=tidx_i)
    io = sb((M, NE), I32)
    nisa.iota(dst=io, pattern=[[1, NE]], offset=0, channel_multiplier=0)
    iof = sb((M, NE), F32)
    nisa.tensor_copy(dst=iof, src=io)
    wts = sb((M, 8), F32)
    nisa.memset(dst=wts, value=0.0)
    for j in range(TOPK):
        oh = sb((M, NE), F32)
        nisa.tensor_scalar(dst=oh, data=iof, op0=nl.equal, operand0=tidx_f[:, j:j + 1])
        nisa.tensor_tensor(dst=oh, data1=oh, data2=sc, op=nl.multiply)
        nisa.tensor_reduce(dst=wts[:, j:j + 1], op=nl.add, data=oh, axis=1)
    wsum = sb((M, 1), F32)
    nisa.tensor_reduce(dst=wsum, op=nl.add, data=wts[:, 0:TOPK], axis=1)
    nisa.tensor_scalar(dst=wsum, data=wsum, op0=nl.add, operand0=1e-20)
    rws = sb((M, 1), F32)
    nisa.reciprocal(dst=rws, data=wsum)
    nisa.tensor_scalar(dst=wts[:, 0:TOPK], data=wts[:, 0:TOPK], op0=nl.multiply, operand0=rws)
    nisa.tensor_scalar(dst=wts[:, 0:TOPK], data=wts[:, 0:TOPK], op0=nl.multiply, operand0=ROUTE_SCALE)
    return tidx_i, wts
