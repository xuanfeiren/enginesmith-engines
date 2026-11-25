"""NKI helper functions shared by the DeepSeek-V4.1-Flash engine kernels (build time only).

Layouts used throughout (P = 128 partitions):
  R layout ("rows"):    a vector v[N] as an SBUF tile [N/128, 128], R[r, f] = v[r*128 + f]  (N/128 <= 128)
  C layout ("columns"): a vector v[N] as an SBUF tile [128, N/128], C[p, j] = v[j*128 + p]
GEMV weights (bf16, pre-scaled at load) live in HBM as wt[p, nt, kt, j] = W[nt*128 + j, kt*128 + p]
(one contiguous run per partition), consumed as stationary [128 k, 128 n] tiles with x in C layout moving.
"""
import nki
import nki.language as nl
import nki.isa as nisa

P = 128
F32 = nl.float32
BF16 = nl.bfloat16
I32 = nl.int32
U32 = nl.uint32


def sb(shape, dtype=F32, name=None):
    if name is None:
        return nl.ndarray(shape, dtype=dtype, buffer=nl.sbuf)
    return nl.ndarray(shape, dtype=dtype, buffer=nl.sbuf, name=name)


def ps(shape, dtype=F32):
    """PSUM tile padded to one full 2 KiB bank per partition (a tile straddling two banks corrupts matmuls)."""
    assert len(shape) == 2
    per_bank = 512 if dtype == F32 else 1024
    assert shape[1] <= per_bank, shape
    t = nl.ndarray((shape[0], per_bank), dtype=dtype, buffer=nl.psum)
    return t[:, 0:shape[1]]


# ----------------------------------------------------------------------------------------------- constants
def ones_sb(n, m, dtype=F32):
    t = sb((n, m), dtype)
    nisa.memset(dst=t, value=1.0)
    return t


def identity_sb(n, dtype=BF16):
    """[n, n] identity in SBUF (for TensorE transposes / permutations)."""
    ident = sb((n, n), dtype)
    nisa.memset(dst=ident, value=0.0)
    # affine_select: keep 1 where (p - f) == 0
    one = sb((n, n), dtype)
    nisa.memset(dst=one, value=1.0)
    nisa.affine_select(dst=ident, pattern=[[-1, n]], channel_multiplier=1, on_true_tile=one,
                       on_false_value=0.0, cmp_op=nl.equal)
    return ident


# ----------------------------------------------------------------------------------------------- transposes
def transpose_to(dst_sb, src_sb, ident):
    """dst [F, Pn] = src [Pn, F]^T via TensorE (exact). F, Pn <= 128."""
    pn, f = src_sb.shape[0], src_sb.shape[1]
    pt = ps((f, pn), src_sb.dtype)
    nisa.nc_transpose(dst=pt, data=src_sb)
    nisa.tensor_copy(dst=dst_sb, src=pt)


def r_to_c(r_sb, ident, dtype=BF16):
    """R [NR, 128] -> C [128, NR]."""
    nr = r_sb.shape[0]
    c = sb((P, nr), dtype)
    transpose_to(c, r_sb, ident)
    return c


def c_to_r(c_sb, ident, dtype=BF16):
    """C [128, NC] -> R [NC, 128] (NC <= 128)."""
    nc = c_sb.shape[1]
    r = sb((nc, P), dtype)
    transpose_to(r, c_sb, ident)
    return r


# ----------------------------------------------------------------------------------------------- reductions
def allsum_partitions(col_sb, ones_nn):
    """col [n, 1] fp32 -> [n, 1] fp32 holding sum over the n partitions on every partition (TensorE)."""
    n = col_sb.shape[0]
    pt = ps((n, 1))
    nisa.nc_matmul(dst=pt, stationary=ones_nn, moving=col_sb)
    out = sb((n, 1))
    nisa.tensor_copy(dst=out, src=pt)
    return out


def rsqrt_precise(dst, a):
    """dst = 1/sqrt(a) in fp32: GpSimd rsqrt + one Newton step (a: [n,1] fp32)."""
    y = sb(a.shape)
    nisa.tensor_scalar(dst=y, data=a, op0=nl.rsqrt, operand0=0.0, engine=nisa.engine.gpsimd)
    # y = y * (1.5 - 0.5 * a * y * y)
    t = sb(a.shape)
    nisa.tensor_tensor(dst=t, data1=y, data2=y, op=nl.multiply)
    nisa.tensor_tensor(dst=t, data1=t, data2=a, op=nl.multiply)
    nisa.tensor_scalar(dst=t, data=t, op0=nl.multiply, operand0=-0.5, op1=nl.add, operand1=1.5)
    nisa.tensor_tensor(dst=dst, data1=y, data2=t, op=nl.multiply)


def rmsnorm_r(x_r, w_r, n_total, eps, ones_nn, out_dtype=BF16):
    """RMSNorm over all elements of an R-layout tile x_r [NR, 128] (any float dtype).
    torch: x = x.float(); x * rsqrt(mean(x^2) + eps); (w * x).to(bf16).  w_r: [NR, 128] (fp32/bf16)."""
    nr, f = x_r.shape
    xf = sb((nr, f))
    nisa.tensor_copy(dst=xf, src=x_r)
    sq = sb((nr, f))
    ss = sb((nr, 1))
    nisa.activation_reduce(dst=sq, op=nl.square, data=xf, reduce_op=nl.add, reduce_res=ss)
    tot = allsum_partitions(ss, ones_nn)
    var = sb((nr, 1))
    nisa.tensor_scalar(dst=var, data=tot, op0=nl.multiply, operand0=1.0 / n_total, op1=nl.add, operand1=eps)
    rs = sb((nr, 1))
    rsqrt_precise(rs, var)
    y = sb((nr, f))
    nisa.tensor_scalar(dst=y, data=xf, op0=nl.multiply, operand0=rs)
    out = sb((nr, f), out_dtype)
    nisa.tensor_tensor(dst=out, data1=y, data2=w_r, op=nl.multiply)
    return out


# ----------------------------------------------------------------------------------------------- quantization
def pow2_ceil(dst_f32, src_f32):
    """dst = 2^ceil(log2(src)) for positive normal fp32 src (bit exact: (bits + 0x7FFFFF) & 0xFF800000)."""
    shp = src_f32.shape
    t = sb(shp, I32)
    nisa.tensor_scalar(dst=t, data=src_f32.view(I32), op0=nl.add, operand0=0x7FFFFF, engine=nisa.engine.gpsimd)
    nisa.tensor_scalar(dst=dst_f32.view(I32), data=t, op0=nl.bitwise_and, operand0=-8388608)  # 0xFF800000


def round_e4m3fn(dst, v):
    """dst = v rounded to the fp8 e4m3fn grid, round to nearest even (|v| <= 448 assumed), in fp32.
    q = 2^(max(floor(log2|v|), -6) - 3); r = (v + 1.5*2^23*q) - 1.5*2^23*q."""
    shp = v.shape
    e = sb(shp, I32)
    nisa.tensor_scalar(dst=e, data=v.view(I32), op0=nl.bitwise_and, operand0=0x7F800000)
    q = sb(shp, I32)
    nisa.tensor_scalar(dst=q, data=e, op0=nl.maximum, operand0=121 << 23, op1=nl.subtract, operand1=3 << 23,
                       engine=nisa.engine.gpsimd)
    c = sb(shp)
    nisa.tensor_scalar(dst=c, data=q.view(F32), op0=nl.multiply, operand0=12582912.0)
    t = sb(shp)
    nisa.tensor_tensor(dst=t, data1=v, data2=c, op=nl.add)
    nisa.tensor_tensor(dst=dst, data1=t, data2=c, op=nl.subtract)


def act_quant_r(x_r, out_dtype=BF16, block=32):
    """act_quant(x, 32, ue8m0) quant + dequant (inplace=True semantics) on an R tile [NR, 128].
    Returns an R tile with fp8(x/s)*s values (exact in bf16)."""
    nr, f = x_r.shape
    nb = f // block
    xf = sb((nr, f))
    nisa.tensor_copy(dst=xf, src=x_r)
    ax = sb((nr, f))
    nisa.tensor_scalar(dst=ax, data=xf, op0=nl.abs, operand0=0.0)
    amax = sb((nr, nb))
    nisa.tensor_reduce(dst=amax, op=nl.maximum, data=ax.reshape((nr, nb, block)), axis=2)
    s0 = sb((nr, nb))
    nisa.tensor_scalar(dst=s0, data=amax, op0=nl.maximum, operand0=1e-4, op1=nl.multiply, operand1=1.0 / 448.0)
    s = sb((nr, nb))
    pow2_ceil(s, s0)
    inv = sb((nr, nb))
    nisa.reciprocal(dst=inv, data=s)
    v = sb((nr, f))
    nisa.tensor_tensor(dst=v.reshape((nr, nb, block)), data1=xf.reshape((nr, nb, block)),
                       data2=inv.reshape((nr, nb, 1)).broadcast(2, block), op=nl.multiply)
    nisa.tensor_scalar(dst=v, data=v, op0=nl.minimum, operand0=448.0, op1=nl.maximum, operand1=-448.0)
    r = sb((nr, f))
    round_e4m3fn(r, v)
    out = sb((nr, f), out_dtype)
    nisa.tensor_tensor(dst=out.reshape((nr, nb, block)), data1=r.reshape((nr, nb, block)),
                       data2=s.reshape((nr, nb, 1)).broadcast(2, block), op=nl.multiply)
    return out


# ----------------------------------------------------------------------------------------------- GEMV
WEIGHT_DMA = 'hwdge'


def wdma(dst, src):
    """weight streams: hardware DGE on the Sync engine (which computes nothing), unless configured otherwise"""
    if WEIGHT_DMA == 'hwdge':
        nisa.dma_copy(dst=dst, src=src, dge_mode=nisa.dge_mode.hwdge, engine=nisa.engine.sync)
    else:
        nisa.dma_copy(dst=dst, src=src)


def lnc_split(NT):
    """(first, last) output tile of this physical core, or (0, NT) when the work is not split.
    Under LNC=2 (kernel[2]) each program takes one half of an even tile count; the halves are exchanged with
    sendrecv (`exchange_halves`).  The per-tile arithmetic does not change, so results are bitwise those of LNC=1."""
    if nl.num_programs() == 2 and NT % 2 == 0 and NT >= 2:
        pid = nl.program_id(0)
        h = NT // 2
        return pid * h, (pid + 1) * h
    return 0, NT


def tok_mine(m, M):
    """Under LNC=2, is token m (of an M-token group, M even) computed by this physical core?  Core 0 takes tokens
    0 .. M/2-1, core 1 the rest.  Per-token work that both cores used to repeat (indexer, attention core) runs on the
    owner only; its instructions do not change, so the results are bitwise the same."""
    if nl.num_programs() == 2 and M % 2 == 0 and M >= 2:
        return (2 * m) // M == nl.program_id(0)
    return True


def tok_split(M):
    return nl.num_programs() == 2 and M % 2 == 0 and M >= 2


def exchange_tok(tiles, M):
    """tiles: per-token SBUF tiles of one shape; each core filled those of its own tokens (tok_mine).  After the
    call both cores hold all of them (exact copies)."""
    if not tok_split(M):
        return
    pid = nl.program_id(0)
    for k in range(M // 2):
        a = k
        b = k + M // 2
        if pid == 0:
            nisa.sendrecv(src=tiles[a], dst=tiles[b], send_to_rank=1, recv_from_rank=1, pipe_id=0)
        else:
            nisa.sendrecv(src=tiles[b], dst=tiles[a], send_to_rank=0, recv_from_rank=0, pipe_id=0)


def exchange_halves(y, NT):
    """y [*, NT] SBUF tile whose own half was written by this core: receive the other half."""
    if nl.num_programs() == 2 and NT % 2 == 0 and NT >= 2:
        pid = nl.program_id(0)
        h = NT // 2
        nisa.sendrecv(src=y[:, pid * h:(pid + 1) * h], dst=y[:, (1 - pid) * h:(2 - pid) * h], send_to_rank=1 - pid,
                      recv_from_rank=1 - pid, pipe_id=0)


def prefetch(wt_hbm, N, K, count, split=True):
    """issue the weight DMAs of the first `count` output tiles (of this core) of a later gemv_c now, so that the
    weight stream continues through the compute before it.  Returns {nt: tile} for gemv_c(pre=...)."""
    KT, NT = K // P, N // P
    n0, n1 = lnc_split(NT) if split else (0, NT)
    pre = []
    for nt in range(n0, min(n1, n0 + count)):
        w = sb((P, KT * P), wt_hbm.dtype)
        wdma(w, wt_hbm[:, nt * KT * P:(nt + 1) * KT * P])
        pre.append(w)
    return pre


def gemv_c(x_c, wt_hbm, N, K, out_dtype=BF16, nt_group=8, split=True, pre=None):
    """y = W x.  x_c: C tile [128, K/128] (bf16), wt_hbm: [128, NT*KT*128] bf16 (wt[p, nt, kt, j]).
    Returns y as a C tile [128, N/128] of out_dtype (fp32 accumulate).  Split over the two physical cores.
    pre: weight tiles already loaded by prefetch() (same values, same arithmetic)."""
    KT, NT = K // P, N // P
    y = sb((P, NT), out_dtype)
    n0, n1 = lnc_split(NT) if split else (0, NT)
    npre = 0 if pre is None else len(pre)
    for g0 in range(n0, n1, nt_group):
        g1 = min(n1, g0 + nt_group)
        acc = ps((P, g1 - g0))
        for nt in range(g0, g1):
            if nt - n0 < npre:
                w = pre[nt - n0]
            else:
                w = sb((P, KT * P), wt_hbm.dtype)
                wdma(w, wt_hbm[:, nt * KT * P:(nt + 1) * KT * P])
            for kt in range(KT):
                nisa.nc_matmul(dst=acc[:, nt - g0:nt - g0 + 1], stationary=w[:, kt * P:(kt + 1) * P],
                               moving=x_c[:, kt:kt + 1])
        nisa.tensor_copy(dst=y[:, g0:g1], src=acc)
    if split:
        exchange_halves(y, NT)
    return y


def round_e2m1(dst, v):
    """dst = v rounded to the fp4 e2m1 grid {0, .5, 1, 1.5, 2, 3, 4, 6}, round to nearest even (|v| <= 6), fp32.
    q = 2^(max(floor(log2|v|), 0) - 1); r = (v + 1.5*2^23*q) - 1.5*2^23*q."""
    shp = v.shape
    e = sb(shp, I32)
    nisa.tensor_scalar(dst=e, data=v.view(I32), op0=nl.bitwise_and, operand0=0x7F800000)
    q = sb(shp, I32)
    nisa.tensor_scalar(dst=q, data=e, op0=nl.maximum, operand0=127 << 23, op1=nl.subtract, operand1=1 << 23,
                       engine=nisa.engine.gpsimd)
    c = sb(shp)
    nisa.tensor_scalar(dst=c, data=q.view(F32), op0=nl.multiply, operand0=12582912.0)
    t = sb(shp)
    nisa.tensor_tensor(dst=t, data1=v, data2=c, op=nl.add)
    nisa.tensor_tensor(dst=dst, data1=t, data2=c, op=nl.subtract)


def recip_precise(dst, a):
    """dst = 1/a (fp32): VectorE reciprocal + one Newton step."""
    nisa.reciprocal(dst=dst, data=a)   # VectorE reciprocal is correctly rounded (measured: 0 ulp)


def fp4_act_quant_r(x_r, block, e4m3_scale, out_dtype=BF16):
    """fp4_act_quant(x, block, inplace=True) on an R tile [n, f] (blocks along the free dim).
    E8M0 scales (indexer, block 32): s = pow2_ceil(max(amax, 6*2^-126) * (1/6)).
    E4M3 scales (compressed KV, block 16): s = e4m3fn(min(max(amax, 6*2^-9) / 6, 448)).
    Returns fp4(x/s) * s (exact in bf16)."""
    nr, f = x_r.shape
    nb = f // block
    xf = sb((nr, f))
    nisa.tensor_copy(dst=xf, src=x_r)
    ax = sb((nr, f))
    nisa.tensor_scalar(dst=ax, data=xf, op0=nl.abs, operand0=0.0)
    amax = sb((nr, nb))
    nisa.tensor_reduce(dst=amax, op=nl.maximum, data=ax.reshape((nr, nb, block)), axis=2)
    s = sb((nr, nb))
    if e4m3_scale:
        s0 = sb((nr, nb))
        nisa.tensor_scalar(dst=s0, data=amax, op0=nl.maximum, operand0=6.0 * 2.0 ** -9, op1=nl.multiply,
                           operand1=1.0 / 6.0)
        nisa.tensor_scalar(dst=s0, data=s0, op0=nl.minimum, operand0=448.0)
        round_e4m3fn(s, s0)
    else:
        s0 = sb((nr, nb))
        nisa.tensor_scalar(dst=s0, data=amax, op0=nl.maximum, operand0=6.0 * 2.0 ** -126, op1=nl.multiply,
                           operand1=1.0 / 6.0)
        pow2_ceil(s, s0)
    inv = sb((nr, nb))
    recip_precise(inv, s)
    v = sb((nr, f))
    nisa.tensor_tensor(dst=v.reshape((nr, nb, block)), data1=xf.reshape((nr, nb, block)),
                       data2=inv.reshape((nr, nb, 1)).broadcast(2, block), op=nl.multiply)
    nisa.tensor_scalar(dst=v, data=v, op0=nl.minimum, operand0=6.0, op1=nl.maximum, operand1=-6.0)
    r = sb((nr, f))
    round_e2m1(r, v)
    out = sb((nr, f), out_dtype)
    nisa.tensor_tensor(dst=out.reshape((nr, nb, block)), data1=r.reshape((nr, nb, block)),
                       data2=s.reshape((nr, nb, 1)).broadcast(2, block), op=nl.multiply)
    return out


def gemv_cm(xs, wt_hbm, N, K, out_dtype=BF16, split=True):
    """gemv_c for several tokens at once: each weight tile is loaded once and multiplied by the M token columns
    (moving operand [128 k, M]); per column the arithmetic (and fp32 accumulation order) is that of gemv_c.
    xs: list of C tiles [128, K/128].  Returns a list of C tiles [128, N/128]."""
    M = len(xs)
    KT, NT = K // P, N // P
    xm = sb((P, KT, M), xs[0].dtype)
    for m in range(M):
        nisa.tensor_copy(dst=xm[:, :, m], src=xs[m])
    ym = sb((P, NT, M), out_dtype)
    n0, n1 = lnc_split(NT) if split else (0, NT)
    G = max(1, min(8, 512 // M))
    for g0 in range(n0, n1, G):
        g1 = min(n1, g0 + G)
        acc = ps((P, (g1 - g0) * M))
        for nt in range(g0, g1):
            w = sb((P, KT * P), wt_hbm.dtype)
            wdma(w, wt_hbm[:, nt * KT * P:(nt + 1) * KT * P])
            i = nt - g0
            for kt in range(KT):
                nisa.nc_matmul(dst=acc[:, i * M:(i + 1) * M], stationary=w[:, kt * P:(kt + 1) * P],
                               moving=xm[:, kt, :])
        nisa.tensor_copy(dst=ym[:, g0:g1, :], src=acc.reshape((P, g1 - g0, M)))
    if split and nl.num_programs() == 2 and NT % 2 == 0 and NT >= 2:
        exchange_halves(ym.reshape((P, NT * M)), NT * M)
    ys = []
    for m in range(M):
        y = sb((P, NT), out_dtype)
        nisa.tensor_copy(dst=y, src=ym[:, :, m])
        ys.append(y)
    return ys


# ---------------------------------------------------------------------------------------- precise transcendentals
# The ScalarE activation tables are ~1e-6..1e-5 relative (measured: exp 1.1e-5, sqrt 6.9e-6, softplus 3.1e-6,
# sigmoid 2.4e-6 worst case); the reference computes these to ~1 ulp.  These versions use VectorE fp32 arithmetic.
_LOG2E = 1.4426950408889634
_LN2_HI = 0.693359375
_LN2_LO = -2.12194440e-4
_MAGIC = 12582912.0          # 1.5 * 2^23: x + MAGIC - MAGIC rounds x to an integer (RNE)


def exp_precise(dst, x):
    """dst = exp(x) in fp32 to ~1 ulp (x clamped to [-87, 88]: exp(-87) ~ 1.6e-38 stands for 0)."""
    shp = x.shape
    xc = sb(shp, F32)
    nisa.tensor_scalar(dst=xc, data=x, op0=nl.maximum, operand0=-87.0, op1=nl.minimum, operand1=88.0)
    kf = sb(shp, F32)
    nisa.tensor_scalar(dst=kf, data=xc, op0=nl.multiply, operand0=_LOG2E, op1=nl.add, operand1=_MAGIC)
    nisa.tensor_scalar(dst=kf, data=kf, op0=nl.subtract, operand0=_MAGIC)
    r = sb(shp, F32)
    nisa.scalar_tensor_tensor(dst=r, data=kf, op0=nl.multiply, operand0=-_LN2_HI, op1=nl.add, operand1=xc)
    nisa.scalar_tensor_tensor(dst=r, data=kf, op0=nl.multiply, operand0=-_LN2_LO, op1=nl.add, operand1=r)
    # e^r, |r| <= 0.35: Horner degree 7
    acc = sb(shp, F32)
    nisa.tensor_scalar(dst=acc, data=r, op0=nl.multiply, operand0=1.0 / 5040.0, op1=nl.add, operand1=1.0 / 720.0)
    for c in (1.0 / 120.0, 1.0 / 24.0, 1.0 / 6.0, 0.5, 1.0, 1.0):
        nisa.tensor_tensor(dst=acc, data1=acc, data2=r, op=nl.multiply)
        nisa.tensor_scalar(dst=acc, data=acc, op0=nl.add, operand0=c)
    # 2^k: exponent bits
    ki = sb(shp, I32)
    nisa.tensor_copy(dst=ki, src=kf)
    nisa.tensor_scalar(dst=ki, data=ki, op0=nl.add, operand0=127, engine=nisa.engine.gpsimd)
    nisa.tensor_scalar(dst=ki, data=ki, op0=nl.left_shift, operand0=23)
    nisa.tensor_tensor(dst=dst, data1=acc, data2=ki.view(F32), op=nl.multiply)


def sigmoid_precise(dst, x):
    """1 / (1 + exp(-x)) (fp32, ~1 ulp)."""
    nx = sb(x.shape, F32)
    nisa.tensor_scalar(dst=nx, data=x, op0=nl.multiply, operand0=-1.0)
    e = sb(x.shape, F32)
    exp_precise(e, nx)
    nisa.tensor_scalar(dst=e, data=e, op0=nl.add, operand0=1.0)
    nisa.reciprocal(dst=dst, data=e)


def log_precise(dst, u):
    """natural log of u > 0 (normal fp32), ~1 ulp: u = 2^e * m, m in [sqrt(1/2), sqrt(2)), atanh series."""
    shp = u.shape
    bits = u.view(I32)
    e = sb(shp, I32)
    nisa.tensor_scalar(dst=e, data=bits, op0=nl.right_shift, operand0=23, op1=nl.bitwise_and, operand1=255)
    mb = sb(shp, I32)
    nisa.tensor_scalar(dst=mb, data=bits, op0=nl.bitwise_and, operand0=0x007FFFFF, op1=nl.bitwise_or, operand1=0x3F800000)
    m = mb.view(F32)                                       # [1, 2)
    big = sb(shp, F32)
    nisa.tensor_scalar(dst=big, data=m, op0=nl.greater, operand0=1.4142135623730951)   # 1.0 / 0.0
    # m = m * (1 - big/2) ; e += big
    sc = sb(shp, F32)
    nisa.tensor_scalar(dst=sc, data=big, op0=nl.multiply, operand0=-0.5, op1=nl.add, operand1=1.0)
    mm = sb(shp, F32)
    nisa.tensor_tensor(dst=mm, data1=m, data2=sc, op=nl.multiply)
    ef = sb(shp, F32)
    nisa.tensor_copy(dst=ef, src=e)
    nisa.tensor_scalar(dst=ef, data=ef, op0=nl.subtract, operand0=127.0)
    nisa.tensor_tensor(dst=ef, data1=ef, data2=big, op=nl.add)
    f = sb(shp, F32)
    nisa.tensor_scalar(dst=f, data=mm, op0=nl.subtract, operand0=1.0)
    den = sb(shp, F32)
    nisa.tensor_scalar(dst=den, data=f, op0=nl.add, operand0=2.0)
    rd = sb(shp, F32)
    nisa.reciprocal(dst=rd, data=den)
    s = sb(shp, F32)
    nisa.tensor_tensor(dst=s, data1=f, data2=rd, op=nl.multiply)
    s2 = sb(shp, F32)
    nisa.tensor_tensor(dst=s2, data1=s, data2=s, op=nl.multiply)
    acc = sb(shp, F32)
    nisa.tensor_scalar(dst=acc, data=s2, op0=nl.multiply, operand0=2.0 / 11.0, op1=nl.add, operand1=2.0 / 9.0)
    for c in (2.0 / 7.0, 2.0 / 5.0, 2.0 / 3.0, 2.0):
        nisa.tensor_tensor(dst=acc, data1=acc, data2=s2, op=nl.multiply)
        nisa.tensor_scalar(dst=acc, data=acc, op0=nl.add, operand0=c)
    nisa.tensor_tensor(dst=acc, data1=acc, data2=s, op=nl.multiply)          # log(m)
    t = sb(shp, F32)
    nisa.tensor_scalar(dst=t, data=ef, op0=nl.multiply, operand0=_LN2_LO)
    nisa.tensor_tensor(dst=acc, data1=acc, data2=t, op=nl.add)
    nisa.scalar_tensor_tensor(dst=dst, data=ef, op0=nl.multiply, operand0=_LN2_HI, op1=nl.add, operand1=acc)


def softplus_precise(dst, x):
    """torch F.softplus (beta 1, threshold 20): x if x > 20 else log1p(exp(x)) (log1p via log(u) * z / (u - 1))."""
    shp = x.shape
    z = sb(shp, F32)
    exp_precise(z, x)
    u = sb(shp, F32)
    nisa.tensor_scalar(dst=u, data=z, op0=nl.add, operand0=1.0)
    lu = sb(shp, F32)
    log_precise(lu, u)
    um1 = sb(shp, F32)
    nisa.tensor_scalar(dst=um1, data=u, op0=nl.subtract, operand0=1.0)
    # where u == 1 the ratio z / (u - 1) is 1 (log1p(z) = z): use max(um1, tiny) and fix with z
    um1c = sb(shp, F32)
    nisa.tensor_scalar(dst=um1c, data=um1, op0=nl.maximum, operand0=1e-38)
    r = sb(shp, F32)
    nisa.reciprocal(dst=r, data=um1c)
    lp = sb(shp, F32)
    nisa.tensor_tensor(dst=lp, data1=lu, data2=z, op=nl.multiply)
    nisa.tensor_tensor(dst=lp, data1=lp, data2=r, op=nl.multiply)
    small = sb(shp, F32)
    nisa.tensor_scalar(dst=small, data=um1, op0=nl.equal, operand0=0.0)       # 1 where u == 1
    # lp = small ? z : lp
    dz = sb(shp, F32)
    nisa.tensor_tensor(dst=dz, data1=z, data2=lp, op=nl.subtract)
    nisa.tensor_tensor(dst=dz, data1=dz, data2=small, op=nl.multiply)
    nisa.tensor_tensor(dst=lp, data1=lp, data2=dz, op=nl.add)
    # x > 20 -> x
    gt = sb(shp, F32)
    nisa.tensor_scalar(dst=gt, data=x, op0=nl.greater, operand0=20.0)
    dx = sb(shp, F32)
    nisa.tensor_tensor(dst=dx, data1=x, data2=lp, op=nl.subtract)
    nisa.tensor_tensor(dst=dx, data1=dx, data2=gt, op=nl.multiply)
    nisa.tensor_tensor(dst=dst, data1=lp, data2=dx, op=nl.add)


def sqrt_precise(dst, a):
    """sqrt(a), a >= 0: ScalarE estimate + one Newton step (y + a/y) / 2 with the exact reciprocal."""
    y = sb(a.shape, F32)
    nisa.activation(dst=y, op=nl.sqrt, data=a)
    yc = sb(a.shape, F32)
    nisa.tensor_scalar(dst=yc, data=y, op0=nl.maximum, operand0=1e-30)
    r = sb(a.shape, F32)
    nisa.reciprocal(dst=r, data=yc)
    q = sb(a.shape, F32)
    nisa.tensor_tensor(dst=q, data1=a, data2=r, op=nl.multiply)
    nisa.tensor_tensor(dst=q, data1=q, data2=y, op=nl.add)
    nisa.tensor_scalar(dst=dst, data=q, op0=nl.multiply, operand0=0.5)
