"""Building blocks of the DeepSeek-V4.1-Flash layer kernels (NKI, traced at build time).

Semantics follow the official inference/model.py and the CPU reference kernels (deepseek_cpu.py).
"""
import nki.language as nl
import nki.isa as nisa

import lib
from lib import P, F32, BF16, I32, sb, ps

D = 5120
HC = 4
HCD = HC * D
NR = D // P          # 40 rows of 128 for a hidden vector
EPS = 1e-20
HC_EPS = 1e-6


def load_reg(val_sb11):
    """register from a [1,1] int32 SBUF tile"""
    r = nisa.register_alloc()
    nisa.register_load(r, val_sb11)
    return r


# posv layout (int32 [1, 8], written by the host for every step)
PV_POS, PV_SLOT, PV_CPOS, PV_CROW, PV_TC = 0, 1, 2, 3, 4


def pv_off(pv, i):
    """[1,1] SBUF view of entry i of the position vector, usable as a DMA scalar_offset."""
    return pv.ap(pattern=[[8, 1], [1, 1]], offset=i)


def load_h_r(h_hbm):
    """h [4, 5120] bf16 in HBM -> SBUF [40, 4, 128] with t[r, i, f] = h[i, r*128 + f]."""
    t = sb((NR, HC, P), BF16)
    nisa.dma_copy(dst=t, src=h_hbm.ap(pattern=[[P, NR], [D, HC], [1, P]], offset=0))
    return t


def store_h_r(h_hbm, t):
    nisa.dma_copy(dst=h_hbm.ap(pattern=[[P, NR], [D, HC], [1, P]], offset=0), src=t)


def bcast_row(src_hbm_row, nparts, n, dtype=F32):
    """[1, n] HBM row -> [nparts, n] SBUF (same row on every partition)."""
    t = sb((nparts, n), dtype)
    nisa.dma_copy(dst=t, src=src_hbm_row.ap(pattern=[[0, nparts], [1, n]], offset=0))
    return t


# ------------------------------------------------------------------------------------------ hyper-connections
def recip_precise(dst, a):
    """dst = 1/a (fp32): VectorE reciprocal + one Newton step."""
    nisa.reciprocal(dst=dst, data=a)   # VectorE reciprocal is correctly rounded (measured: 0 ulp)


def div_rows(c3, den):
    """c3 [1, 4, 4] view /= den [1, 4] broadcast over the last dim."""
    r = sb(den.shape)
    recip_precise(r, den)
    nisa.tensor_tensor(dst=c3, data1=c3, data2=r.reshape((1, 4, 1)).broadcast(2, 4), op=nl.multiply)


def sk_colnorm(cT):
    csum = sb((1, 4), F32)
    nisa.tensor_reduce(dst=csum, op=nl.add, data=cT, axis=2)        # sum over i, per column j
    nisa.tensor_scalar(dst=csum, data=csum, op0=nl.add, operand0=HC_EPS)
    div_rows(cT, csum)


def sk_rownorm(c3):
    rs_ = sb((1, 4), F32)
    nisa.tensor_reduce(dst=rs_, op=nl.add, data=c3, axis=2)
    nisa.tensor_scalar(dst=rs_, data=rs_, op0=nl.add, operand0=HC_EPS)
    div_rows(c3, rs_)


def load_fn(fnT_hbm):
    fn = sb((P, (HCD // P) * 24), F32)
    nisa.dma_copy(dst=fn, src=fnT_hbm)
    return fn


def hc_mixes(h_r, fnT_hbm, scale_hbm, base_hbm, ident, ones40, fn_sb=None):
    """Block.hc_mixes + hc_split_sinkhorn.  h_r: [40, 4, 128] bf16.
    fnT_hbm: [128, 160*24] fp32 with fnT[p, kt*24 + n] = fn[n, kt*128 + p] (kt = i*40 + r).
    Returns an SBUF [1, 24] fp32 tile: pre[0:4], post[4:8], comb[8:24] (comb[i][j] at 8 + 4i + j)."""
    KT = HCD // P  # 160
    # x_flat in C layout (fp32): xc[p, i*40 + r] = h[i, r*128 + p]
    xc = sb((P, KT), F32)
    for i in range(HC):
        pt = ps((P, NR), BF16)
        nisa.nc_transpose(dst=pt, data=h_r[:, i, :])
        nisa.tensor_copy(dst=xc[:, i * NR:(i + 1) * NR], src=pt)
    # rsqrt(mean(x^2) + eps) over all 20480 values
    hf = sb((NR, HC * P), F32)
    nisa.tensor_copy(dst=hf, src=h_r.reshape((NR, HC * P)))
    sq = sb((NR, HC * P), F32)
    ss = sb((NR, 1), F32)
    nisa.activation_reduce(dst=sq, op=nl.square, data=hf, reduce_op=nl.add, reduce_res=ss)
    tot = lib.allsum_partitions(ss, ones40)
    var = sb((NR, 1), F32)
    nisa.tensor_scalar(dst=var, data=tot, op0=nl.multiply, operand0=1.0 / HCD, op1=nl.add, operand1=EPS)
    rs = sb((NR, 1), F32)
    lib.rsqrt_precise(rs, var)
    # mixes [1, 24] = x . fn^T  (fp32 matmul)
    if fn_sb is None:
        fn = load_fn(fnT_hbm)
    else:
        fn = fn_sb
    acc = ps((1, 24))
    for kt in range(KT):
        nisa.nc_matmul(dst=acc, stationary=xc[:, kt:kt + 1], moving=fn[:, kt * 24:(kt + 1) * 24])
    mixes = sb((1, 24), F32)
    nisa.tensor_scalar(dst=mixes, data=acc, op0=nl.multiply, operand0=rs[0:1, 0:1])
    sc = sb((1, 3), F32)
    nisa.dma_copy(dst=sc, src=scale_hbm)
    bs = sb((1, 24), F32)
    nisa.dma_copy(dst=bs, src=base_hbm)
    out = sb((1, 24), F32)
    t = sb((1, 24), F32)
    # pre = sigmoid(m*s0 + b) + eps
    nisa.tensor_scalar(dst=t[:, 0:4], data=mixes[:, 0:4], op0=nl.multiply, operand0=sc[:, 0:1])
    nisa.tensor_tensor(dst=t[:, 0:4], data1=t[:, 0:4], data2=bs[:, 0:4], op=nl.add)
    lib.sigmoid_precise(out[:, 0:4], t[:, 0:4])
    nisa.tensor_scalar(dst=out[:, 0:4], data=out[:, 0:4], op0=nl.add, operand0=HC_EPS)
    # post = 2 * sigmoid(m*s1 + b)
    nisa.tensor_scalar(dst=t[:, 4:8], data=mixes[:, 4:8], op0=nl.multiply, operand0=sc[:, 1:2])
    nisa.tensor_tensor(dst=t[:, 4:8], data1=t[:, 4:8], data2=bs[:, 4:8], op=nl.add)
    lib.sigmoid_precise(out[:, 4:8], t[:, 4:8])
    nisa.tensor_scalar(dst=out[:, 4:8], data=out[:, 4:8], op0=nl.multiply, operand0=2.0)
    # comb = softmax_rows(m*s2 + b) + eps, then column norm, then 19 x (row, column)
    c = sb((1, 16), F32)
    nisa.tensor_scalar(dst=c, data=mixes[:, 8:24], op0=nl.multiply, operand0=sc[:, 2:3])
    nisa.tensor_tensor(dst=c, data1=c, data2=bs[:, 8:24], op=nl.add)
    c3 = c.reshape((1, 4, 4))           # [i, j]
    cT = c3.permute((0, 2, 1))          # [j, i] view
    rmax = sb((1, 4), F32)
    nisa.tensor_reduce(dst=rmax, op=nl.maximum, data=c3, axis=2)
    nisa.tensor_tensor(dst=c3, data1=c3, data2=rmax.reshape((1, 4, 1)).broadcast(2, 4), op=nl.subtract)
    lib.exp_precise(c, c)
    rsum = sb((1, 4), F32)
    nisa.tensor_reduce(dst=rsum, op=nl.add, data=c3, axis=2)
    div_rows(c3, rsum)
    nisa.tensor_scalar(dst=c, data=c, op0=nl.add, operand0=HC_EPS)

    sk_colnorm(cT)
    for _ in range(20 - 1):
        sk_rownorm(c3)
        sk_colnorm(cT)
    nisa.tensor_copy(dst=out[:, 8:24], src=c)
    return out


def bcast_parts(row_sb, nparts, ones_1n):
    """[1, n] fp32 SBUF row -> [nparts, n] (TensorE: ones[1, nparts]^T @ row)."""
    n = row_sb.shape[1]
    pt = ps((nparts, n))
    nisa.nc_matmul(dst=pt, stationary=ones_1n, moving=row_sb)
    t = sb((nparts, n), F32)
    nisa.tensor_copy(dst=t, src=pt)
    return t


def hc_pre(h_r, mix_b, col0=0, out_dtype=BF16):
    """y = sum_i mix[i] * h[i] (fp32) -> bf16.  mix_b: [40, >=4] fp32 (same on all partitions), weights at
    columns col0..col0+3.  Returns R tile [40, 128]."""
    acc = sb((NR, P), F32)
    nisa.tensor_scalar(dst=acc, data=h_r[:, 0, :], op0=nl.multiply, operand0=mix_b[:, col0:col0 + 1])
    for i in range(1, HC):
        nisa.scalar_tensor_tensor(dst=acc, data=h_r[:, i, :], op0=nl.multiply, operand0=mix_b[:, col0 + i:col0 + i + 1],
                                  op1=nl.add, operand1=acc)
    out = sb((NR, P), out_dtype)
    nisa.tensor_copy(dst=out, src=acc)
    return out


def hc_post(x_r, res_r, mix_b):
    """y[j] = post[j]*x + sum_i comb[i,j]*res[i]  (fp32) -> bf16 [40, 4, 128].
    mix_b: [40, 24] (pre 0:4, post 4:8, comb 8:24)."""
    xf = sb((NR, P), F32)
    nisa.tensor_copy(dst=xf, src=x_r)
    y = sb((NR, HC, P), BF16)
    for j in range(HC):
        s = sb((NR, P), F32)
        nisa.tensor_scalar(dst=s, data=res_r[:, 0, :], op0=nl.multiply, operand0=mix_b[:, 8 + j:9 + j])
        for i in range(1, HC):
            k = 8 + 4 * i + j
            nisa.scalar_tensor_tensor(dst=s, data=res_r[:, i, :], op0=nl.multiply, operand0=mix_b[:, k:k + 1],
                                      op1=nl.add, operand1=s)
        nisa.scalar_tensor_tensor(dst=y[:, j, :], data=xf, op0=nl.multiply, operand0=mix_b[:, 4 + j:5 + j],
                                  op1=nl.add, operand1=s)
    return y


# ------------------------------------------------------------------------------------------ RoPE
def rope_c_cols(xc_cols, rotP, cosc, sinc, out_cols):
    """RoPE on a C-layout chunk whose partitions 64..127 hold the rope dims (pairs on adjacent partitions).
    xc_cols: [128, n] bf16 view; rotP: [128,128] bf16 pair-swap matrix; cosc/sinc: [128,1] fp32 (cos=1, sin=0
    on partitions < 64; sinc is -sin on even, +sin on odd). Writes bf16 into out_cols (may alias)."""
    n = xc_cols.shape[1]
    pt = ps((P, n))
    nisa.nc_matmul(dst=pt, stationary=rotP, moving=xc_cols)
    a = sb((P, n), F32)
    nisa.tensor_scalar(dst=a, data=xc_cols, op0=nl.multiply, operand0=cosc)
    nisa.scalar_tensor_tensor(dst=out_cols, data=pt, op0=nl.multiply, operand0=sinc, op1=nl.add, operand1=a)


def rope_r_row(x_tile, r0, r1, cs, inverse=False):
    """RoPE on the last 64 elements of rows r0..r1-1 of an R tile x_tile [n, 128] (in place), pairs adjacent on
    the free dim.  cs: [n, 64] fp32 tile (cos 0:32, sin 32:64) valid on rows r0..r1-1."""
    n = x_tile.shape[0]
    xe = x_tile[r0:r1, 64:128:2]
    xo = x_tile[r0:r1, 65:128:2]
    cos = cs[r0:r1, 0:32]
    sin = cs[r0:r1, 32:64]
    a_ = sb((n, 32), F32)
    b_ = sb((n, 32), F32)
    e2_ = sb((n, 32), F32)
    o2_ = sb((n, 32), F32)
    a, b, e2, o2 = a_[r0:r1], b_[r0:r1], e2_[r0:r1], o2_[r0:r1]
    nisa.tensor_tensor(dst=a, data1=xe, data2=cos, op=nl.multiply)
    nisa.tensor_tensor(dst=b, data1=xo, data2=sin, op=nl.multiply)
    if inverse:   # x * conj(f): re = xe*cos + xo*sin, im = xo*cos - xe*sin
        nisa.tensor_tensor(dst=e2, data1=a, data2=b, op=nl.add)
    else:         # re = xe*cos - xo*sin, im = xe*sin + xo*cos
        nisa.tensor_tensor(dst=e2, data1=a, data2=b, op=nl.subtract)
    nisa.tensor_tensor(dst=a, data1=xe, data2=sin, op=nl.multiply)
    nisa.tensor_tensor(dst=b, data1=xo, data2=cos, op=nl.multiply)
    if inverse:
        nisa.tensor_tensor(dst=o2, data1=b, data2=a, op=nl.subtract)
    else:
        nisa.tensor_tensor(dst=o2, data1=a, data2=b, op=nl.add)
    nisa.tensor_copy(dst=xe, src=e2)
    nisa.tensor_copy(dst=xo, src=o2)


def load_rope_c(ropec_hbm):
    """per-partition cos/sin columns [128,1] of the current position (host-written ropec [128, 2] fp32:
    cos | sin, cos = 1 and sin = 0 on partitions < 64; sin is -sin on even partitions)."""
    rc = sb((P, 2), F32)
    nisa.dma_copy(dst=rc, src=ropec_hbm)
    return rc[:, 0:1], rc[:, 1:2]


def load_rope_r(roper_hbm, nparts, r0=0, r1=None):
    """[nparts, 64] tile whose rows r0..r1-1 hold the host-written (cos | sin) row [1, 64] of the position;
    identity rotation (cos 1, sin 0) on the other rows."""
    if r1 is None:
        r1 = nparts
    cs = sb((nparts, 64), F32)
    if r0 > 0 or r1 < nparts:   # engines cannot start at odd partitions: rotate whole tiles
        nisa.memset(dst=cs[:, 0:32], value=1.0)
        nisa.memset(dst=cs[:, 32:64], value=0.0)
    nisa.dma_copy(dst=cs[r0:r1, :], src=roper_hbm.ap(pattern=[[0, r1 - r0], [1, 64]], offset=0))
    return cs
