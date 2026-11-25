"""Kernels: embedding lookup, Engram (layers 1 and 14), LM head with argmax."""
import nki
import nki.language as nl
import nki.isa as nisa

import lib
import blocks as B
from lib import P, F32, BF16, I32, U32, sb, ps

VOCAB = 129280
NVT = VOCAB // P          # 1010
NHASH = 24
EHD = 256
EK = NHASH * EHD          # 6144
EN = 5 * B.D              # 25600


@nki.jit
def embed(tok, table):
    """h[4, 5120] = 4 copies of table[tok] (bf16)."""
    t = sb((1, 1), I32)
    nisa.dma_copy(dst=t, src=tok)
    row = sb((B.NR, P), BF16)
    nisa.dma_copy(dst=row, src=table.ap(pattern=[[P, B.NR], [1, P]], offset=0, scalar_offset=t, indirect_dim=0))
    h = sb((B.NR, B.HC, P), BF16)
    for i in range(B.HC):
        nisa.tensor_copy(dst=h[:, i, :], src=row)
    out = nl.ndarray((B.HC, B.D), dtype=BF16, buffer=nl.shared_hbm)
    B.store_h_r(out, h)
    return out


def decode_e4m3fn(dst_f32, b_u8):
    """exact OCP e4m3fn decode of raw bytes (as integers): (-1)^s * (8*[e>0] + m) * 2^(max(e,1) - 10)."""
    shp = b_u8.shape
    bi = sb(shp, I32)
    nisa.tensor_copy(dst=bi, src=b_u8)
    e = sb(shp, I32)
    nisa.tensor_scalar(dst=e, data=bi, op0=nl.right_shift, operand0=3, op1=nl.bitwise_and, operand1=15)
    m = sb(shp, I32)
    nisa.tensor_scalar(dst=m, data=bi, op0=nl.bitwise_and, operand0=7)
    s = sb(shp, I32)
    nisa.tensor_scalar(dst=s, data=bi, op0=nl.right_shift, operand0=7)
    # mantissa with hidden bit: m + 8*min(e,1)
    hb = sb(shp, I32)
    nisa.tensor_scalar(dst=hb, data=e, op0=nl.minimum, operand0=1, op1=nl.multiply, operand1=8,
                       engine=nisa.engine.gpsimd)
    mm = sb(shp, I32)
    nisa.tensor_tensor(dst=mm, data1=m, data2=hb, op=nl.add, engine=nisa.engine.gpsimd)
    # 2^(max(e,1) - 10) as fp32 bits: (max(e,1) + 117) << 23
    pw = sb(shp, I32)
    nisa.tensor_scalar(dst=pw, data=e, op0=nl.maximum, operand0=1, op1=nl.add, operand1=117,
                       engine=nisa.engine.gpsimd)
    nisa.tensor_scalar(dst=pw, data=pw, op0=nl.left_shift, operand0=23)
    mf = sb(shp, F32)
    nisa.tensor_copy(dst=mf, src=mm)
    nisa.tensor_tensor(dst=mf, data1=mf, data2=pw.view(F32), op=nl.multiply)
    # sign: * (1 - 2s)
    sf = sb(shp, F32)
    nisa.tensor_copy(dst=sf, src=s)
    nisa.tensor_scalar(dst=sf, data=sf, op0=nl.multiply, operand0=-2.0, op1=nl.add, operand1=1.0)
    nisa.tensor_tensor(dst=dst_f32, data1=mf, data2=sf, op=nl.multiply)


def engram_ev(rows, rscale, ident, m=0):
    """the act_quant'd embedding rows of token m (rows [M, 24, 256] / rscale [M, 24, 8] raw bytes, or [24, ...])
    as a C tile [128, 48] bf16"""
    NRE = EK // P                                                  # 48 rows of 128
    rb = sb((NRE, P), nl.uint8)
    nisa.dma_copy(dst=rb, src=rows.ap(pattern=[[P, NRE], [1, P]], offset=m * EK))
    vals = sb((NRE, P), F32)
    decode_e4m3fn(vals, rb)
    sbytes = sb((NRE, 4), nl.uint8)
    nisa.dma_copy(dst=sbytes, src=rscale.ap(pattern=[[4, NRE], [1, 4]], offset=m * NHASH * 8))
    si = sb((NRE, 4), I32)
    nisa.tensor_copy(dst=si, src=sbytes)
    nisa.tensor_scalar(dst=si, data=si, op0=nl.left_shift, operand0=23)
    ev = sb((NRE, P), BF16)
    nisa.tensor_tensor(dst=ev.reshape((NRE, 4, 32)), data1=vals.reshape((NRE, 4, 32)),
                       data2=si.view(F32).reshape((NRE, 4, 1)).broadcast(2, 32), op=nl.multiply)
    evq = lib.act_quant_r(ev)
    return lib.r_to_c(evq, ident)                                  # C [128, 48]


@nki.jit
def engram4(h, rows, rscale, wkv, qw, kw):
    """Engram.forward for a 4-token prefill group: the kv GEMV with each weight tile loaded once for the 4 tokens
    (lib.gemv_cm: per token the arithmetic of gemv_c), then each token's gate.  h [4, 4, 5120] bf16,
    rows [4, 24, 256] / rscale [4, 24, 8] raw bytes."""
    ident = lib.identity_sb(P)
    evcs = []
    for m in range(4):
        evcs.append(engram_ev(rows, rscale, ident, m))
    kvs = lib.gemv_cm(evcs, wkv, EN, EK)
    o = nl.ndarray((4, B.HC, B.D), dtype=BF16, buffer=nl.shared_hbm)
    for m in range(4):
        engram_gate_core(h[m], kvs[m], qw, kw, o=o[m])
    return o


def engram_kv_core(rows, rscale, wkv):
    """the token-only part of Engram: kv = wkv(act_quant(embedding rows)) as a C tile [128, 200] bf16"""
    ident = lib.identity_sb(P)
    ones40 = lib.ones_sb(B.NR, B.NR)
    NRE = EK // P                                                  # 48 rows of 128
    rb = sb((NRE, P), nl.uint8)
    nisa.dma_copy(dst=rb, src=rows.ap(pattern=[[P, NRE], [1, P]], offset=0))
    vals = sb((NRE, P), F32)
    decode_e4m3fn(vals, rb)
    # scales: row r of R covers hash col r // 2, elements (r % 2)*128 .. +128 = scale blocks (r%2)*4 .. +4
    sbytes = sb((NRE, 4), nl.uint8)
    nisa.dma_copy(dst=sbytes, src=rscale.ap(pattern=[[4, NRE], [1, 4]], offset=0))
    si = sb((NRE, 4), I32)
    nisa.tensor_copy(dst=si, src=sbytes)
    nisa.tensor_scalar(dst=si, data=si, op0=nl.left_shift, operand0=23)
    ev = sb((NRE, P), BF16)
    nisa.tensor_tensor(dst=ev.reshape((NRE, 4, 32)), data1=vals.reshape((NRE, 4, 32)),
                       data2=si.view(F32).reshape((NRE, 4, 1)).broadcast(2, 32), op=nl.multiply)
    # wkv: Linear fp8 -> act_quant + GEMV
    evq = lib.act_quant_r(ev)
    evc = lib.r_to_c(evq, ident)                                   # C [128, 48]
    kv = lib.gemv_c(evc, wkv, EN, EK)                              # C [128, 200] bf16
    return kv


def engram_gate_core(h, kv, qw, kw, o=None):
    """the rest of Engram.forward: gate from h and the keys, h + gate * value -> [4, 5120] bf16 (HBM)"""
    ident = lib.identity_sb(P)
    ones40 = lib.ones_sb(B.NR, B.NR)
    h_r = B.load_h_r(h)
    hf = sb((B.NR, B.HC, P), F32)
    nisa.tensor_copy(dst=hf, src=h_r)
    qwt = B.load_h_r(qw)
    kwt = B.load_h_r(kw)
    wf = sb((B.NR, B.HC, P), F32)
    nisa.tensor_tensor(dst=wf, data1=qwt, data2=kwt, op=nl.multiply)
    val = lib.c_to_r(kv[:, 4 * B.NR:5 * B.NR], ident)              # [40, 128] bf16
    valf = sb((B.NR, P), F32)
    nisa.tensor_copy(dst=valf, src=val)
    out = sb((B.NR, B.HC, P), BF16)
    for i in range(B.HC):
        key = lib.c_to_r(kv[:, i * B.NR:(i + 1) * B.NR], ident)     # [40, 128] bf16
        keyf = sb((B.NR, P), F32)
        nisa.tensor_copy(dst=keyf, src=key)
        # rstd = rsqrt(mean(h^2) + eps) * rsqrt(mean(key^2) + eps)
        sq = sb((B.NR, P), F32)
        s1 = sb((B.NR, 1), F32)
        nisa.activation_reduce(dst=sq, op=nl.square, data=hf[:, i, :], reduce_op=nl.add, reduce_res=s1)
        s2 = sb((B.NR, 1), F32)
        nisa.activation_reduce(dst=sq, op=nl.square, data=keyf, reduce_op=nl.add, reduce_res=s2)
        t1 = lib.allsum_partitions(s1, ones40)
        t2 = lib.allsum_partitions(s2, ones40)
        v1 = sb((B.NR, 1), F32)
        nisa.tensor_scalar(dst=v1, data=t1, op0=nl.multiply, operand0=1.0 / B.D, op1=nl.add, operand1=B.EPS)
        v2 = sb((B.NR, 1), F32)
        nisa.tensor_scalar(dst=v2, data=t2, op0=nl.multiply, operand0=1.0 / B.D, op1=nl.add, operand1=B.EPS)
        r1 = sb((B.NR, 1), F32)
        lib.rsqrt_precise(r1, v1)
        r2 = sb((B.NR, 1), F32)
        lib.rsqrt_precise(r2, v2)
        rs = sb((B.NR, 1), F32)
        nisa.tensor_tensor(dst=rs, data1=r1, data2=r2, op=nl.multiply)
        # dot = sum(h * w * key) * rstd * dim^-0.5
        p = sb((B.NR, P), F32)
        nisa.tensor_tensor(dst=p, data1=hf[:, i, :], data2=wf[:, i, :], op=nl.multiply)
        nisa.tensor_tensor(dst=p, data1=p, data2=keyf, op=nl.multiply)
        ps_ = sb((B.NR, 1), F32)
        nisa.tensor_reduce(dst=ps_, op=nl.add, data=p, axis=1)
        dot = lib.allsum_partitions(ps_, ones40)
        nisa.tensor_tensor(dst=dot, data1=dot, data2=rs, op=nl.multiply)
        nisa.tensor_scalar(dst=dot, data=dot, op0=nl.multiply, operand0=float(B.D ** -0.5))
        # gate = sigmoid(sign(dot) * sqrt(max(|dot|, 1e-6)))
        ad = sb((B.NR, 1), F32)
        nisa.tensor_scalar(dst=ad, data=dot, op0=nl.abs, operand0=0.0)
        nisa.tensor_scalar(dst=ad, data=ad, op0=nl.maximum, operand0=1e-6)
        sq1 = sb((B.NR, 1), F32)
        lib.sqrt_precise(sq1, ad)
        sg = sb((B.NR, 1), F32)
        nisa.tensor_scalar(dst=sg, data=dot, op0=nl.greater_equal, operand0=0.0, op1=nl.multiply, operand1=2.0)
        nisa.tensor_scalar(dst=sg, data=sg, op0=nl.subtract, operand0=1.0)
        nisa.tensor_tensor(dst=sq1, data1=sq1, data2=sg, op=nl.multiply)
        gt = sb((B.NR, 1), F32)
        lib.sigmoid_precise(gt, sq1)
        nisa.scalar_tensor_tensor(dst=out[:, i, :], data=valf, op0=nl.multiply, operand0=gt, op1=nl.add,
                                  operand1=hf[:, i, :])
    if o is None:
        o = nl.ndarray((B.HC, B.D), dtype=BF16, buffer=nl.shared_hbm)
    B.store_h_r(o, out)
    return o




@nki.jit
def engram(h, rows, rscale, wkv, qw, kw):
    """Engram.forward for one token.  rows: [24, 256] raw e4m3fn bytes (uint8) of the gathered table rows,
    rscale: [24, 8] raw E8M0 bytes, wkv: tiled bf16 [25600, 6144], qw/kw: [4, 5120] bf16."""
    kv = engram_kv_core(rows, rscale, wkv)
    return engram_gate_core(h, kv, qw, kw)


@nki.jit
def engram_kv(rows, rscale, wkv):
    """decode: engram_kv_core alone (launched when the step starts; it depends on the tokens only)"""
    kv = engram_kv_core(rows, rscale, wkv)
    o = nl.ndarray((P, 200), dtype=BF16, buffer=nl.shared_hbm)
    nisa.dma_copy(dst=o, src=kv)
    return o


@nki.jit
def engram_gate(h, kvh, qw, kw):
    kv = sb((P, 200), BF16)
    nisa.dma_copy(dst=kv, src=kvh)
    return engram_gate_core(h, kv, qw, kw)


@nki.jit
def head(h, premix, nw, wt):
    """logits = linear(norm(hc_pre(h, pre_mix)).float(), head) in fp32; returns argmax (ties -> lowest id)."""
    ident = lib.identity_sb(P)
    ones40 = lib.ones_sb(B.NR, B.NR)
    h_r = B.load_h_r(h)
    pm = B.bcast_row(premix, B.NR, 4)
    x = B.hc_pre(h_r, pm)
    g = sb((B.NR, P), BF16)
    nisa.dma_copy(dst=g, src=nw)
    xn = lib.rmsnorm_r(x, g, B.D, B.EPS, ones40)
    xc = lib.r_to_c(xn, ident)
    lg = lib.gemv_c(xc, wt, VOCAB, B.D, out_dtype=F32)             # C [128, 1010] fp32, id = c*128 + p
    # per-partition max, then the lowest id reaching it: score = (lg == m_p) * (2^20 - id)
    mp = sb((P, 1), F32)
    nisa.tensor_reduce(dst=mp, op=nl.maximum, data=lg, axis=1)
    io = sb((P, NVT), I32)
    nisa.iota(dst=io, pattern=[[P, NVT]], offset=0, channel_multiplier=1)
    iof = sb((P, NVT), F32)
    nisa.tensor_copy(dst=iof, src=io)
    big = sb((P, NVT), F32)
    nisa.tensor_scalar(dst=big, data=iof, op0=nl.multiply, operand0=-1.0, op1=nl.add, operand1=1048576.0)
    eq = sb((P, NVT), F32)
    nisa.tensor_scalar(dst=eq, data=lg, op0=nl.equal, operand0=mp)
    nisa.tensor_tensor(dst=eq, data1=eq, data2=big, op=nl.multiply)
    sp = sb((P, 1), F32)
    nisa.tensor_reduce(dst=sp, op=nl.maximum, data=eq, axis=1)
    both = sb((P, 2), F32)
    nisa.tensor_copy(dst=both[:, 0:1], src=mp)
    nisa.tensor_copy(dst=both[:, 1:2], src=sp)
    pt = ps((2, P), F32)
    nisa.nc_transpose(dst=pt, data=both)
    rows = sb((2, P), F32)
    nisa.tensor_copy(dst=rows, src=pt)
    M = sb((1, 1), F32)
    nisa.tensor_reduce(dst=M, op=nl.maximum, data=rows[0:1, :], axis=1)
    # second row must be compared at partition 0: copy via DMA-free path (transpose gives both rows at p 0/1)
    r2 = sb((1, P), F32)
    pt2 = ps((1, P), F32)
    nisa.nc_transpose(dst=pt2, data=both[:, 1:2])
    nisa.tensor_copy(dst=r2, src=pt2)
    msk = sb((1, P), F32)
    nisa.tensor_scalar(dst=msk, data=rows[0:1, :], op0=nl.equal, operand0=M)
    nisa.tensor_tensor(dst=msk, data1=msk, data2=r2, op=nl.multiply)
    best = sb((1, 1), F32)
    nisa.tensor_reduce(dst=best, op=nl.maximum, data=msk, axis=1)
    nisa.tensor_scalar(dst=best, data=best, op0=nl.multiply, operand0=-1.0, op1=nl.add, operand1=1048576.0)
    tid = sb((1, 1), I32)
    nisa.tensor_copy(dst=tid, src=best)
    out = nl.ndarray((1, 1), dtype=I32, buffer=nl.shared_hbm)
    nisa.dma_copy(dst=out, src=tid)
    lo = nl.ndarray((P, NVT), dtype=F32, buffer=nl.shared_hbm)
    nisa.dma_copy(dst=lo, src=lg)
    return out, lo


# ------------------------------------------------------------------------------ LM head split over 5 cores (decode)
HPARTS = 5
NVT_P = NVT // HPARTS     # 202 tiles of 128 ids per part
VP = NVT_P * P            # 25856 ids per part


@nki.jit
def head_part(h, premix, nw, wt):
    """head over ids [part*VP, (part+1)*VP) (wt: the part's rows, tiled): (max logit, lowest local id reaching it)
    as fp32 [1, 2].  The logits are those of `head` (same GEMV, same accumulation order)."""
    ident = lib.identity_sb(P)
    ones40 = lib.ones_sb(B.NR, B.NR)
    h_r = B.load_h_r(h)
    pm = B.bcast_row(premix, B.NR, 4)
    x = B.hc_pre(h_r, pm)
    g = sb((B.NR, P), BF16)
    nisa.dma_copy(dst=g, src=nw)
    xn = lib.rmsnorm_r(x, g, B.D, B.EPS, ones40)
    xc = lib.r_to_c(xn, ident)
    lg = lib.gemv_c(xc, wt, VP, B.D, out_dtype=F32)              # C [128, 202] fp32, local id = c*128 + p
    mp = sb((P, 1), F32)
    nisa.tensor_reduce(dst=mp, op=nl.maximum, data=lg, axis=1)
    io = sb((P, NVT_P), I32)
    nisa.iota(dst=io, pattern=[[P, NVT_P]], offset=0, channel_multiplier=1)
    iof = sb((P, NVT_P), F32)
    nisa.tensor_copy(dst=iof, src=io)
    big = sb((P, NVT_P), F32)
    nisa.tensor_scalar(dst=big, data=iof, op0=nl.multiply, operand0=-1.0, op1=nl.add, operand1=1048576.0)
    eq = sb((P, NVT_P), F32)
    nisa.tensor_scalar(dst=eq, data=lg, op0=nl.equal, operand0=mp)
    nisa.tensor_tensor(dst=eq, data1=eq, data2=big, op=nl.multiply)
    sp = sb((P, 1), F32)
    nisa.tensor_reduce(dst=sp, op=nl.maximum, data=eq, axis=1)
    both = sb((P, 2), F32)
    nisa.tensor_copy(dst=both[:, 0:1], src=mp)
    nisa.tensor_copy(dst=both[:, 1:2], src=sp)
    pt = ps((2, P), F32)
    nisa.nc_transpose(dst=pt, data=both)
    rows = sb((2, P), F32)
    nisa.tensor_copy(dst=rows, src=pt)
    M = sb((1, 1), F32)
    nisa.tensor_reduce(dst=M, op=nl.maximum, data=rows[0:1, :], axis=1)
    r2 = sb((1, P), F32)
    pt2 = ps((1, P), F32)
    nisa.nc_transpose(dst=pt2, data=both[:, 1:2])
    nisa.tensor_copy(dst=r2, src=pt2)
    msk = sb((1, P), F32)
    nisa.tensor_scalar(dst=msk, data=rows[0:1, :], op0=nl.equal, operand0=M)
    nisa.tensor_tensor(dst=msk, data1=msk, data2=r2, op=nl.multiply)
    best = sb((1, 1), F32)
    nisa.tensor_reduce(dst=best, op=nl.maximum, data=msk, axis=1)
    res = sb((1, 2), F32)
    nisa.tensor_copy(dst=res[:, 0:1], src=M)
    nisa.tensor_scalar(dst=res[:, 1:2], data=best, op0=nl.multiply, operand0=-1.0, op1=nl.add, operand1=1048576.0)
    out = nl.ndarray((1, 2), dtype=F32, buffer=nl.shared_hbm)
    nisa.dma_copy(dst=out, src=res)
    return out


@nki.jit
def head_final(c5):
    """c5 [1, 10] fp32: (max, local id) of the 5 parts -> the token (largest logit, ties to the lowest id)."""
    c = sb((1, 2 * HPARTS), F32)
    nisa.dma_copy(dst=c, src=c5)
    v = c[:, 0:2 * HPARTS:2]
    ids = sb((1, HPARTS), F32)
    for p in range(HPARTS):
        nisa.tensor_scalar(dst=ids[:, p:p + 1], data=c[:, 2 * p + 1:2 * p + 2], op0=nl.add, operand0=float(p * VP))
    M = sb((1, 1), F32)
    nisa.tensor_reduce(dst=M, op=nl.maximum, data=v, axis=1)
    sc = sb((1, HPARTS), F32)
    nisa.tensor_scalar(dst=sc, data=ids, op0=nl.multiply, operand0=-1.0, op1=nl.add, operand1=1048576.0)
    eq = sb((1, HPARTS), F32)
    nisa.tensor_scalar(dst=eq, data=v, op0=nl.equal, operand0=M)
    nisa.tensor_tensor(dst=eq, data1=eq, data2=sc, op=nl.multiply)
    best = sb((1, 1), F32)
    nisa.tensor_reduce(dst=best, op=nl.maximum, data=eq, axis=1)
    nisa.tensor_scalar(dst=best, data=best, op0=nl.multiply, operand0=-1.0, op1=nl.add, operand1=1048576.0)
    tid = sb((1, 1), I32)
    nisa.tensor_copy(dst=tid, src=best)
    out = nl.ndarray((1, 1), dtype=I32, buffer=nl.shared_hbm)
    nisa.dma_copy(dst=out, src=tid)
    return out
