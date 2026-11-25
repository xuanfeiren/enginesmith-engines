"""Tensor-parallel layered prefill for Qwen3-14B (W ranks x 2 physical cores = 2W virtual cores), adapted from lab 1's
prefill_g.py (Qwen3-8B, one logical core). Three kinds of NEFF for each prompt bucket S: STAGE 1 embedding, STAGE 2
one layer (executed once per layer on per-layer tensor slices), STAGE 3 final norm + LM head + argmax on token plen-1
(the vocabulary split over the 2W virtual cores, step_tp._lm_argmax_tp).

Per virtual core (rank r, physical core pid): q heads (2r+pid)*NQL.., kv heads (2r+pid)*NKL.., intermediate tiles
(2r+pid)*ITL.. -- the decode step's split and weight layouts (step_tp.py), so prefill and decode share every tensor.
Attention is local: nkilib attention_cte over the rank's 2*NQL q heads, split over the two cores as our QKV split.
o_proj and down are K-split: each core writes a partial h for all hidden tiles to s_p[pid]; after a barrier core pid
adds s_p[0] + s_p[1] for its half of the hidden tiles, all-reduces that over the ranks (SBUF, per physical core:
the same physical core of every rank holds the same half) and adds it to h.
Scratch inputs (caller-allocated, shared by the two cores): s_h [128, HT, S] fp32, s_q [2NQL, 128, S],
s_k [2NKL, 128, S], s_v [2NKL, S, 128], s_o [2NQL, 128, S] bf16, s_p [2, 128, HT, S] fp32.
"""
import nki
import nki.language as nl
import nki.isa as nisa
from layer_decode import _consts, _rms_col, D, EPS
import nki.collectives as ncc
from step_tp import _lm_argmax_tp, VPERM, ITF, IH
from nkilib.core.attention.attention_cte import _attention_cte
from prefill_sp import layer_sp, head_sp

SCALE = 0.08838834764831845
import os as _os
SLAB_HW = int(_os.environ.get("PG_SLAB_HW", "1"))


def _hd(dst, src):
    nisa.dma_copy(dst=dst, src=src, dge_mode=nisa.dge_mode.hwdge, engine=nisa.engine.sync)


SQ = [0]   # round 004 experiment: non-weight DMAs with static descriptors (dge none), off the Sync weight queue


def _hs1(dst, src):
    if SQ[0] >= 1:
        nisa.dma_copy(dst=dst, src=src, dge_mode=nisa.dge_mode.none)
    else:
        _hd(dst, src)


def _hs2(dst, src):
    if SQ[0] >= 2:
        nisa.dma_copy(dst=dst, src=src, dge_mode=nisa.dge_mode.none)
    else:
        _hd(dst, src)


AQ = [0]   # round 004 experiment: bitmask of DMA classes moved to the Scalar engine's hwdge queue (1 = all-reduce
# input writes, 2 = SwiGLU activation exchange, 4 = all-reduce finishes); level 1 static fallback as before


def _hq(dst, src, bit):
    if AQ[0] & bit:
        nisa.dma_copy(dst=dst, src=src, dge_mode=nisa.dge_mode.hwdge, engine=nisa.engine.scalar)
    elif SQ[0] >= (2 if bit == 2 else 1):
        nisa.dma_copy(dst=dst, src=src, dge_mode=nisa.dge_mode.none)
    else:
        _hd(dst, src)


def _hqb(dst, src):
    _hq(dst, src, 1)


def _hqx(dst, src):
    _hq(dst, src, 2)


def _hqf(dst, src):
    _hq(dst, src, 4)


NCQ = [0]    # round 006 experiment: queue of the chunked h loads (0 Sync, 1 Scalar, 2 alternating)
NCHK = [1]   # round 006 (NCH): the norms' h load + square in NCHK[0] chunks of hidden tiles (pipelined DMA -> Scalar ->
# TensorE; the same per-element ops, the sum-of-squares matmuls in the same kt order)


def _load_sq(h, t0, n, HT, hc, sq, mm=None):
    """hc = h[:, :, t0:t0+n] (HBM -> SBUF) and sq = square(hc), in NCHK[0] chunks of hidden tiles; with mm = (ss, onesb)
    the sum-of-squares matmuls of each chunk follow it"""
    NC = NCHK[0]
    CT = HT // NC
    for c in range(NC):
        if NCQ[0] == 1 or (NCQ[0] == 2 and c % 2 == 1):   # round 006: on the Scalar engine's hwdge queue (all / odd)
            nisa.dma_copy(dst=hc[:, c * CT:(c + 1) * CT, :], src=h[:, nl.ds(c * CT, CT), nl.ds(t0, n)],
                          dge_mode=nisa.dge_mode.hwdge, engine=nisa.engine.scalar)
        else:
            _hs2(hc[:, c * CT:(c + 1) * CT, :], h[:, nl.ds(c * CT, CT), nl.ds(t0, n)])
        nisa.activation(dst=sq[:, c * CT:(c + 1) * CT, :], op=nl.square, data=hc[:, c * CT:(c + 1) * CT, :])
        if mm is not None:
            for kt in range(c * CT, (c + 1) * CT):
                nisa.nc_matmul(dst=mm[0], stationary=mm[1], moving=sq[:, kt, :])


def _norm_blk(h, t0, n, w_sb, onesb, HT, xo, xo0):
    """RMSNorm of tokens [t0, t0+n) of h [128, HT, S] fp32 (HBM) into xo[:, :, xo0:xo0+n] bf16 (SBUF)."""
    hc = nl.ndarray((128, HT, n), dtype=nl.float32, buffer=nl.sbuf)
    sq = nl.ndarray((128, HT, n), dtype=nl.bfloat16, buffer=nl.sbuf)
    ss = nl.ndarray((128, n), dtype=nl.float32, buffer=nl.psum)
    if NCHK[0] > 1:
        _load_sq(h, t0, n, HT, hc, sq, (ss, onesb))
    else:
        _hs2(hc, h[:, :, nl.ds(t0, n)])
        nisa.activation(dst=sq, op=nl.square, data=hc)
        for kt in range(HT):
            nisa.nc_matmul(dst=ss, stationary=onesb, moving=sq[:, kt, :])
    rstd = nl.ndarray((128, n), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=rstd, data=ss, op0=nl.multiply, operand0=1.0 / (HT * 128), op1=nl.add, operand1=EPS)
    nisa.tensor_scalar(dst=rstd, data=rstd, op0=nl.rsqrt, operand0=0.0, engine=nisa.engine.gpsimd)
    for kt in range(HT):
        if NSPL[0] and kt % NSPL[0] == NSPL[0] - 1:   # round 004 experiment: (h * w) on the Scalar engine (fp32),
            # * rstd on GpSimd: the same two fp32 products and one bf16 rounding as the Vector engine's fused op
            tw = nl.ndarray((128, n), dtype=nl.float32, buffer=nl.sbuf)
            nisa.activation(dst=tw, op=nl.copy, data=hc[:, kt, :], scale=w_sb[:, kt:kt + 1])
            nisa.tensor_tensor(dst=xo[:, kt, xo0:xo0 + n], data1=tw, data2=rstd, op=nl.multiply,
                               engine=nisa.engine.gpsimd)
        else:
            nisa.scalar_tensor_tensor(dst=xo[:, kt, xo0:xo0 + n], data=hc[:, kt, :], op0=nl.multiply,
                                      operand0=w_sb[:, kt:kt + 1], op1=nl.multiply, operand1=rstd)


NSPL = [0]


def _norm_pre(h, t0, n, HT):
    """first half of _norm_blk (round 005, PN=2): load h and square it (no TensorE work)"""
    hc = nl.ndarray((128, HT, n), dtype=nl.float32, buffer=nl.sbuf)
    sq = nl.ndarray((128, HT, n), dtype=nl.bfloat16, buffer=nl.sbuf)
    if NCHK[0] > 1:
        _load_sq(h, t0, n, HT, hc, sq)
    else:
        _hs2(hc, h[:, :, nl.ds(t0, n)])
        nisa.activation(dst=sq, op=nl.square, data=hc)
    return hc, sq


def _norm_post(hc, sq, n, w_sb, onesb, HT, xo, xo0):
    """second half of _norm_blk: the same instructions in the same order"""
    ss = nl.ndarray((128, n), dtype=nl.float32, buffer=nl.psum)
    for kt in range(HT):
        nisa.nc_matmul(dst=ss, stationary=onesb, moving=sq[:, kt, :])
    rstd = nl.ndarray((128, n), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=rstd, data=ss, op0=nl.multiply, operand0=1.0 / (HT * 128), op1=nl.add, operand1=EPS)
    nisa.tensor_scalar(dst=rstd, data=rstd, op0=nl.rsqrt, operand0=0.0, engine=nisa.engine.gpsimd)
    for kt in range(HT):
        nisa.scalar_tensor_tensor(dst=xo[:, kt, xo0:xo0 + n], data=hc[:, kt, :], op0=nl.multiply,
                                  operand0=w_sb[:, kt:kt + 1], op1=nl.multiply, operand1=rstd)


def _slab(W, nt, KT):
    """weight slab: on a hardware DGE queue triggered from the Sync engine (no compute there), see FINDINGS 4"""
    t = nl.ndarray((128, KT * 128), dtype=W.dtype, buffer=nl.sbuf)
    if SLAB_HW:
        nisa.dma_copy(dst=t, src=W[:, nl.ds(nt * KT * 128, KT * 128)], dge_mode=nisa.dge_mode.hwdge,
                      engine=nisa.engine.sync)
    else:
        nisa.dma_copy(dst=t, src=W[:, nl.ds(nt * KT * 128, KT * 128)])
    return t


def _combine(h, s_p, pid, t0, n, HT):
    """h[:, own half, t0:t0+n] += s_p[0] + s_p[1] (own half of the hidden tiles)."""
    HH = HT // 2
    a = nl.ndarray((128, HH, n), dtype=nl.float32, buffer=nl.sbuf)
    _hd(a, h[:, nl.ds(pid * HH, HH), nl.ds(t0, n)])
    b = nl.ndarray((128, HH, n), dtype=nl.float32, buffer=nl.sbuf)
    _hd(b, s_p[0, :, nl.ds(pid * HH, HH), nl.ds(t0, n)])
    c = nl.ndarray((128, HH, n), dtype=nl.float32, buffer=nl.sbuf)
    _hd(c, s_p[1, :, nl.ds(pid * HH, HH), nl.ds(t0, n)])
    nisa.tensor_tensor(dst=a, data1=a, data2=b, op=nl.add)
    nisa.tensor_tensor(dst=a, data1=a, data2=c, op=nl.add)
    _hd(h[:, nl.ds(pid * HH, HH), nl.ds(t0, n)], a)


NOAR = int(_os.environ.get("PF_NOAR", "0"))
NSP_LATE = 1   # NSP: finish block b's attention all-reduce just before MLP block b (4096: 480 -> 441 ms)
NOATT = int(_os.environ.get("PF_NOATT", "0"))   # timing experiment only: skip the cross-rank all-reduce


def _combine_tp(h, s_p, pid, t0, n, HT, rg, W, CCN=128, CCBF=0):
    """h[:, own half, t0:t0+n] += sum over ranks of (s_p[0] + s_p[1]) (own half of the hidden tiles), in sub-chunks of
    CCN tokens (an SBUF all-reduce message of [128, HT/2 * CCN]; fp32, or bf16 with CCBF=1)."""
    HH = HT // 2
    NC = CCN
    if n < NC:
        NC = n
    cdt = nl.float32
    if CCBF:
        cdt = nl.bfloat16
    for sc in range(n // NC):
        u0 = t0 + sc * NC
        b = nl.ndarray((128, HH, NC), dtype=nl.float32, buffer=nl.sbuf)
        _hd(b, s_p[0, :, nl.ds(pid * HH, HH), nl.ds(u0, NC)])
        c = nl.ndarray((128, HH, NC), dtype=nl.float32, buffer=nl.sbuf)
        _hd(c, s_p[1, :, nl.ds(pid * HH, HH), nl.ds(u0, NC)])
        sm = nl.ndarray((128, HH * NC), dtype=cdt, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=sm.reshape((128, HH, NC)), data1=b, data2=c, op=nl.add)
        red = sm
        if W > 1 and not NOAR:
            red = nl.ndarray((128, HH * NC), dtype=cdt, buffer=nl.sbuf)
            ncc.all_reduce(srcs=[sm], dsts=[red], op=nl.add, replica_group=rg)
        a = nl.ndarray((128, HH, NC), dtype=nl.float32, buffer=nl.sbuf)
        _hd(a, h[:, nl.ds(pid * HH, HH), nl.ds(u0, NC)])
        nisa.tensor_tensor(dst=a, data1=a, data2=red.reshape((128, HH, NC)), op=nl.add)
        _hd(h[:, nl.ds(pid * HH, HH), nl.ds(u0, NC)], a)


def _combine_tp_hbm(h, s_p, pid, S, HT, rg, hb, hr, CCN, CCBF=0):
    """h[:, own half, :S] += sum over ranks of (s_p[0] + s_p[1]): each physical core writes its half of the hidden
    tiles of the two-core sum to the internal HBM buffer hb [NCH, 128, HT, CCN] (one slot per chunk, no reuse within a combine) (fp32, or bf16 with CCBF=1), one HBM
    all_reduce over the ranks per CCN tokens (runs once per logical core), both cores read their half of hr back.
    HBM collectives across the two devices: 62 GB/s at 16 MB vs 8.4 GB/s for 1 MB SBUF messages (FINDINGS 1)."""
    HH = HT // 2
    SUB = 256
    if CCN < SUB:
        SUB = CCN
    cdt = nl.float32
    if CCBF:
        cdt = nl.bfloat16
    for ch in range(S // CCN):
        c0 = ch * CCN
        for sc in range(CCN // SUB):
            u0 = c0 + sc * SUB
            b = nl.ndarray((128, HH, SUB), dtype=nl.float32, buffer=nl.sbuf)
            _hd(b, s_p[0, :, nl.ds(pid * HH, HH), nl.ds(u0, SUB)])
            c = nl.ndarray((128, HH, SUB), dtype=nl.float32, buffer=nl.sbuf)
            _hd(c, s_p[1, :, nl.ds(pid * HH, HH), nl.ds(u0, SUB)])
            sm = nl.ndarray((128, HH, SUB), dtype=cdt, buffer=nl.sbuf)
            nisa.tensor_tensor(dst=sm, data1=b, data2=c, op=nl.add)
            _hd(hb[ch, :, nl.ds(pid * HH, HH), nl.ds(sc * SUB, SUB)], sm)
        nisa.core_barrier(data=hb, cores=(0, 1))
        src = hb
        if not NOAR:
            ncc.all_reduce(srcs=[hb[ch]], dsts=[hr[ch]], op=nl.add, replica_group=rg)
            src = hr
        for sc in range(CCN // SUB):
            u0 = c0 + sc * SUB
            a = nl.ndarray((128, HH, SUB), dtype=nl.float32, buffer=nl.sbuf)
            _hd(a, h[:, nl.ds(pid * HH, HH), nl.ds(u0, SUB)])
            r = nl.ndarray((128, HH, SUB), dtype=cdt, buffer=nl.sbuf)
            _hd(r, src[ch, :, nl.ds(pid * HH, HH), nl.ds(sc * SUB, SUB)])
            nisa.tensor_tensor(dst=a, data1=a, data2=r, op=nl.add)
            _hd(h[:, nl.ds(pid * HH, HH), nl.ds(u0, SUB)], a)


def _hbm_launch(s_p, pid, c0, n, HT, rg, hb, hr, ch, CCBF=0):
    """first half of _combine_tp_hbm for tokens [c0, c0+n) into slot ch: two-core sum of the own half -> hb[ch],
    barrier, HBM all_reduce hb[ch] -> hr[ch]. Returns the buffer to read the result from."""
    HH = HT // 2
    SUB = 256
    if n < SUB:
        SUB = n
    cdt = nl.float32
    if CCBF:
        cdt = nl.bfloat16
    for sc in range(n // SUB):
        u0 = c0 + sc * SUB
        b = nl.ndarray((128, HH, SUB), dtype=nl.float32, buffer=nl.sbuf)
        _hd(b, s_p[0, :, nl.ds(pid * HH, HH), nl.ds(u0, SUB)])
        c = nl.ndarray((128, HH, SUB), dtype=nl.float32, buffer=nl.sbuf)
        _hd(c, s_p[1, :, nl.ds(pid * HH, HH), nl.ds(u0, SUB)])
        sm = nl.ndarray((128, HH, SUB), dtype=cdt, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=sm, data1=b, data2=c, op=nl.add)
        _hd(hb[ch, :, nl.ds(pid * HH, HH), nl.ds(sc * SUB, SUB)], sm)
    nisa.core_barrier(data=hb, cores=(0, 1))
    if NOAR:
        return hb
    ncc.all_reduce(srcs=[hb[ch]], dsts=[hr[ch]], op=nl.add, replica_group=rg)
    return hr


def _hbm_finish(h, pid, c0, n, HT, src, ch, CCBF=0):
    """second half: h[:, own half, c0:c0+n] += src[ch] (own half)"""
    HH = HT // 2
    SUB = 256
    if n < SUB:
        SUB = n
    cdt = nl.float32
    if CCBF:
        cdt = nl.bfloat16
    for sc in range(n // SUB):
        u0 = c0 + sc * SUB
        a = nl.ndarray((128, HH, SUB), dtype=nl.float32, buffer=nl.sbuf)
        _hqf(a, h[:, nl.ds(pid * HH, HH), nl.ds(u0, SUB)])
        r = nl.ndarray((128, HH, SUB), dtype=cdt, buffer=nl.sbuf)
        _hqf(r, src[ch, :, nl.ds(pid * HH, HH), nl.ds(sc * SUB, SUB)])
        nisa.tensor_tensor(dst=a, data1=a, data2=r, op=nl.add)
        _hqf(h[:, nl.ds(pid * HH, HH), nl.ds(u0, SUB)], a)


def _finish_norm(h, pid, t0, n, HT, src, ch, CCBF, w_sb, onesb, xo, NSUB):
    """FN (round 004): _hbm_finish fused with the following _norm_blk. Each core adds the all-reduce result src[ch] to
    ALL hidden tiles of h[:, :, t0:t0+n] in SBUF (the other core computes the same sums) and normalises the sums
    straight from SBUF into xo: no h write -> core barrier -> h read round trip on the critical path. h itself is not
    written here (the other core may still read it): _hbm_finish2 adds both all-reduce results at the MLP finish.
    Same per-element operations as _hbm_finish + _norm_blk + _hbm_finish (bitwise identical)."""
    HH = HT // 2
    cdt = nl.float32
    if CCBF:
        cdt = nl.bfloat16
    for sc in range(n // NSUB):
        u0 = t0 + sc * NSUB
        hc = nl.ndarray((128, HT, NSUB), dtype=nl.float32, buffer=nl.sbuf)
        r = nl.ndarray((128, HT, NSUB), dtype=cdt, buffer=nl.sbuf)
        sq = nl.ndarray((128, HT, NSUB), dtype=nl.bfloat16, buffer=nl.sbuf)
        ss = nl.ndarray((128, NSUB), dtype=nl.float32, buffer=nl.psum)
        if NCHK[0] > 1:   # NCH (round 006): load / add / square / sum of squares pipelined over hidden-tile chunks
            NC = NCHK[0]
            CT = HT // NC
            for c in range(NC):
                _hd(hc[:, c * CT:(c + 1) * CT, :], h[:, nl.ds(c * CT, CT), nl.ds(u0, NSUB)])
                _hd(r[:, c * CT:(c + 1) * CT, :], src[ch, :, nl.ds(c * CT, CT), nl.ds(sc * NSUB, NSUB)])
                nisa.tensor_tensor(dst=hc[:, c * CT:(c + 1) * CT, :], data1=hc[:, c * CT:(c + 1) * CT, :],
                                   data2=r[:, c * CT:(c + 1) * CT, :], op=nl.add)
                nisa.activation(dst=sq[:, c * CT:(c + 1) * CT, :], op=nl.square, data=hc[:, c * CT:(c + 1) * CT, :])
                for kt in range(c * CT, (c + 1) * CT):
                    nisa.nc_matmul(dst=ss, stationary=onesb, moving=sq[:, kt, :])
        else:
            _hd(hc, h[:, :, nl.ds(u0, NSUB)])
            _hd(r, src[ch, :, :, nl.ds(sc * NSUB, NSUB)])
            for hf in range(2):
                nisa.tensor_tensor(dst=hc[:, nl.ds(hf * HH, HH), :], data1=hc[:, nl.ds(hf * HH, HH), :],
                                   data2=r[:, nl.ds(hf * HH, HH), :], op=nl.add)
            nisa.activation(dst=sq, op=nl.square, data=hc)
            for kt in range(HT):
                nisa.nc_matmul(dst=ss, stationary=onesb, moving=sq[:, kt, :])
        rstd = nl.ndarray((128, NSUB), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=rstd, data=ss, op0=nl.multiply, operand0=1.0 / (HT * 128), op1=nl.add, operand1=EPS)
        nisa.tensor_scalar(dst=rstd, data=rstd, op0=nl.rsqrt, operand0=0.0, engine=nisa.engine.gpsimd)
        for kt in range(HT):
            nisa.scalar_tensor_tensor(dst=xo[:, kt, sc * NSUB:(sc + 1) * NSUB], data=hc[:, kt, :], op0=nl.multiply,
                                      operand0=w_sb[:, kt:kt + 1], op1=nl.multiply, operand1=rstd)


HSCH = [1]   # HS: hidden-tile chunks of the residual add / square / sum of squares (pipelined; same per-element ops)


def _norm_sb(hs, t0, n, w_sb, onesb, HT, xo, xo0):
    """HS (round 006): _norm_blk with h already in SBUF (hs [128, HT, S] fp32): the same instructions minus the load
    (the square in HSCH[0] chunks of hidden tiles, the sum-of-squares matmuls in the same kt order)"""
    sq = nl.ndarray((128, HT, n), dtype=nl.bfloat16, buffer=nl.sbuf)
    ss = nl.ndarray((128, n), dtype=nl.float32, buffer=nl.psum)
    NC = HSCH[0]
    CT = HT // NC
    for c in range(NC):
        nisa.activation(dst=sq[:, c * CT:(c + 1) * CT, :], op=nl.square, data=hs[:, c * CT:(c + 1) * CT, t0:t0 + n])
        for kt in range(c * CT, (c + 1) * CT):
            nisa.nc_matmul(dst=ss, stationary=onesb, moving=sq[:, kt, :])
    rstd = nl.ndarray((128, n), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=rstd, data=ss, op0=nl.multiply, operand0=1.0 / (HT * 128), op1=nl.add, operand1=EPS)
    nisa.tensor_scalar(dst=rstd, data=rstd, op0=nl.rsqrt, operand0=0.0, engine=nisa.engine.gpsimd)
    for kt in range(HT):
        nisa.scalar_tensor_tensor(dst=xo[:, kt, xo0:xo0 + n], data=hs[:, kt, t0:t0 + n], op0=nl.multiply,
                                  operand0=w_sb[:, kt:kt + 1], op1=nl.multiply, operand1=rstd)


def _sb_add(hs, t0, n, HT, src, ch, CCBF, NSUB):
    """HS: hs[:, :, t0:t0+n] += src[ch] (all hidden tiles, in SBUF; the other core computes the same sums)"""
    HH = HT // 2
    cdt = nl.float32
    if CCBF:
        cdt = nl.bfloat16
    NC = HSCH[0]
    if NC < 2:
        NC = 2
    CT = HT // NC
    for sc in range(n // NSUB):
        u0 = t0 + sc * NSUB
        for c in range(NC):
            r = nl.ndarray((128, CT, NSUB), dtype=cdt, buffer=nl.sbuf)
            _hd(r, src[ch, :, nl.ds(c * CT, CT), nl.ds(sc * NSUB, NSUB)])
            nisa.tensor_tensor(dst=hs[:, nl.ds(c * CT, CT), u0:u0 + NSUB], data1=hs[:, nl.ds(c * CT, CT), u0:u0 + NSUB],
                               data2=r, op=nl.add)


def _hbm_finish2(h, pid, c0, n, HT, src_a, src_m, ch, CCBF=0):
    """FN: h[:, own half, c0:c0+n] = (h + src_a[ch]) + src_m[ch] (own half), the two finishes in one pass"""
    HH = HT // 2
    SUB = 256
    if n < SUB:
        SUB = n
    cdt = nl.float32
    if CCBF:
        cdt = nl.bfloat16
    for sc in range(n // SUB):
        u0 = c0 + sc * SUB
        a = nl.ndarray((128, HH, SUB), dtype=nl.float32, buffer=nl.sbuf)
        _hqf(a, h[:, nl.ds(pid * HH, HH), nl.ds(u0, SUB)])
        ra = nl.ndarray((128, HH, SUB), dtype=cdt, buffer=nl.sbuf)
        _hqf(ra, src_a[ch, :, nl.ds(pid * HH, HH), nl.ds(sc * SUB, SUB)])
        rm = nl.ndarray((128, HH, SUB), dtype=cdt, buffer=nl.sbuf)
        _hqf(rm, src_m[ch, :, nl.ds(pid * HH, HH), nl.ds(sc * SUB, SUB)])
        nisa.tensor_tensor(dst=a, data1=a, data2=ra, op=nl.add)
        nisa.tensor_tensor(dst=a, data1=a, data2=rm, op=nl.add)
        _hqf(h[:, nl.ds(pid * HH, HH), nl.ds(u0, SUB)], a)


def _combine_launch(s_p, pid, t0, n, HT, rg, W):
    """first half of _combine_tp: (s_p[0] + s_p[1]) own half -> all-reduce over the ranks; returns the pending
    (u0, NC, reduced tile) list. The consumer (_combine_finish) is issued later in program order, so the in-order
    engines keep working on the next block while the all-reduce is in flight."""
    HH = HT // 2
    NC = 128
    if n < NC:
        NC = n
    pend = []
    for sc in range(n // NC):
        u0 = t0 + sc * NC
        b = nl.ndarray((128, HH, NC), dtype=nl.float32, buffer=nl.sbuf)
        _hd(b, s_p[0, :, nl.ds(pid * HH, HH), nl.ds(u0, NC)])
        c = nl.ndarray((128, HH, NC), dtype=nl.float32, buffer=nl.sbuf)
        _hd(c, s_p[1, :, nl.ds(pid * HH, HH), nl.ds(u0, NC)])
        sm = nl.ndarray((128, HH * NC), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=sm.reshape((128, HH, NC)), data1=b, data2=c, op=nl.add)
        red = sm
        if W > 1 and not NOAR:
            red = nl.ndarray((128, HH * NC), dtype=nl.float32, buffer=nl.sbuf)
            ncc.all_reduce(srcs=[sm], dsts=[red], op=nl.add, replica_group=rg)
        pend.append((u0, NC, red))
    return pend


def _combine_finish(h, pid, HT, pend):
    HH = HT // 2
    for i in range(len(pend)):
        u0 = pend[i][0]
        NC = pend[i][1]
        red = pend[i][2]
        a = nl.ndarray((128, HH, NC), dtype=nl.float32, buffer=nl.sbuf)
        _hd(a, h[:, nl.ds(pid * HH, HH), nl.ds(u0, NC)])
        nisa.tensor_tensor(dst=a, data1=a, data2=red.reshape((128, HH, NC)), op=nl.add)
        _hd(h[:, nl.ds(pid * HH, HH), nl.ds(u0, NC)], a)


def _pg(ids, plen, toks, embed, ln1, ln2, qn, kn, wqkv, wo, wgu, wd, normf, wlm, rot, cosT, sinT, kcache, vcache,
        rkb, s_h, s_q, s_k, s_v, s_o, s_p, STAGE, NQL, NKL, ITL, W, rg, OVL=0, TEB=256, CCN=128, CCBF=0, CCHBM=0, NSP=0, SB=0, KVS=0, LPN=1, PFG=0, PFQ=0, FN=0, PMF=0, HALF=0, SQL=0, ILV=0, QW=0, QNS=0, AQL=0, NSP2=0, PN=0, PNA=0, XN=0, FN0=0, HS=0, HSC=1, NCH=1, NCQL=0, NCH2=0, SP=0, SPL=-1, SPN=0, SPT=0, SPO=0, SPA=0, SPD=0, SPH=0, SPHN=0, SPP=0, SPQ=0, GCH=1, SOC=1, SPS=1, SPG=0, KTP=0, KTQ=0, MP=0):
    NCQ[0] = NCQL
    SQ[0] = SQL
    HSCH[0] = HSC
    NCHK[0] = NCH
    AQ[0] = AQL
    NSPL[0] = NSP2
    S = ids.shape[0]
    if HALF:
        assert NSP and PFG == 0
    HT = embed.shape[1] // 128
    HID = HT * 128
    NQ = 2 * NQL
    pid = nl.program_id(0)
    TB = 512
    if S < 512:
        TB = S
    TE = TEB                  # block for gate/up + down (act block [128, ITL, TE] bf16)
    if S < TE:
        TE = S
    if S <= SB:               # small buckets: blocks of 128 tokens, so a block's all-reduce overlaps the next block
        TB = 128
        TE = 128
    NSUB = 256                # norm sub-chunk
    if TB < 256:
        NSUB = TB
    ones, ident, idf, ones16 = _consts()
    onesb = nl.ndarray((128, 128), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.memset(dst=onesb, value=1.0)
    pl_sb = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.sbuf)
    nisa.dma_copy(dst=pl_sb, src=plen)
    pos_out = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.shared_hbm)
    h = s_h

    if STAGE == 1:
        NB = S // 128
        for bi in range(NB // 2):
            b = 2 * bi + pid
            idx = nl.ndarray((128, 1), dtype=nl.uint32, buffer=nl.sbuf)
            nisa.dma_copy(dst=idx, src=ids[nl.ds(b * 128, 128), :])
            er = nl.ndarray((128, HID), dtype=nl.bfloat16, buffer=nl.sbuf)
            nisa.dma_copy(dst=er, src=embed.ap(pattern=[[HID, 128], [1, HID]], offset=0, vector_offset=idx, indirect_dim=0))
            et = nl.ndarray((128, HT, 128), dtype=nl.float32, buffer=nl.sbuf)
            for kt in range(HT):
                tp = nl.ndarray((128, 128), dtype=nl.float32, buffer=nl.psum)
                nisa.nc_matmul(dst=tp, stationary=er[:, kt * 128:(kt + 1) * 128], moving=ident)
                nisa.tensor_copy(dst=et[:, kt, :], src=tp)
            _hd(h[:, :, nl.ds(b * 128, 128)], et)
        nisa.dma_copy(dst=pos_out, src=pl_sb)
        return pos_out

    if STAGE == 2 and SP:   # round 007: sequence-parallel residual (prefill_sp.py)
        assert S % 512 == 0
        layer_sp(h, ln1, ln2, qn, kn, wqkv, wo, wgu, wd, rot, cosT, sinT, kcache, vcache, s_q, s_k, s_v, s_o,
                 NQL, NKL, ITL, W, rg, onesb, ident, VPERM, SPL, SPN, SPT, SPO, SPA, LPN, SPD, SPH, SPHN, SPP, SPQ, GCH, SOC, SPS, SPG, KTP, KTQ, MP)
        nisa.dma_copy(dst=pos_out, src=pl_sb)
        return pos_out

    if STAGE == 2:
        rot_sb = nl.ndarray((128, 128), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.dma_copy(dst=rot_sb, src=rot)
        NQKV = NQL + 2 * NKL
        if CCHBM:
            CN = CCN
            if S < CN:
                CN = S
            cdt = nl.float32
            if CCBF:
                cdt = nl.bfloat16
            CA = CN
            CM = CN
            if OVL == 3 or NSP:   # one HBM all-reduce per token block, launched right after the block's partial
                CA = TB
                CM = TE
            hb_a = nl.ndarray((S // CA, 128, HT, CA), dtype=cdt, buffer=nl.shared_hbm, name="pf_hb_a")
            hr_a = nl.ndarray((S // CA, 128, HT, CA), dtype=cdt, buffer=nl.shared_hbm, name="pf_hr_a")
            hb_m = nl.ndarray((S // CM, 128, HT, CM), dtype=cdt, buffer=nl.shared_hbm, name="pf_hb_m")
            hr_m = nl.ndarray((S // CM, 128, HT, CM), dtype=cdt, buffer=nl.shared_hbm, name="pf_hr_m")
            if NSP:
                act_x = nl.ndarray((2, 128, ITL, S), dtype=nl.bfloat16, buffer=nl.shared_hbm, name="pf_act_x")
            if XN:
                x1x = nl.ndarray((128, HT, S), dtype=nl.bfloat16, buffer=nl.shared_hbm, name="pf_x1x")

        hs = None
        if HS:   # round 006 (HS): the residual stays in SBUF for all LPN layers (one block: S == TB == TE, FN)
            assert S == TB and S == TE and FN and NSP and not PN and not PMF
            hs = nl.ndarray((128, HT, S), dtype=nl.float32, buffer=nl.sbuf)
            _hd(hs, h[:, :, :])
        pre_qkv = None
        pre_o = None
        for l in range(LPN):   # LPN layers per execution (per-layer tensors are slices of LPN layers)
            NCHK[0] = NCH   # phase A norms
            pend_a = []
            pend_m = []
            pre_o_in = pre_o
            w1 = nl.ndarray((128, HT), dtype=nl.float32, buffer=nl.sbuf)
            nisa.dma_copy(dst=w1, src=ln1[l])
            w2 = nl.ndarray((128, HT), dtype=nl.float32, buffer=nl.sbuf)
            nisa.dma_copy(dst=w2, src=ln2[l])
            qnc = nl.ndarray((128, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.dma_copy(dst=qnc, src=qn[l])
            knc = nl.ndarray((128, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.dma_copy(dst=knc, src=kn[l])

            # ===== A: norm1 + QKV + q/k norm + RoPE + cache writes
            qw = None
            if QW and S // TB > 1:   # round 004: the QKV slabs loaded once per layer, kept in SBUF for all token blocks
                qw = []
                for nt in range(NQKV):
                    qw.append(_slab(wqkv[l, pid], nt, HT))
            NSA = NSUB
            if QNS and QNS < NSA:
                NSA = QNS
            nx1 = None
            for blk in range(S // TB):
                t0 = blk * TB
                if nx1 is not None:   # PNA: normalised during the previous block's QKV
                    x1 = nx1
                    nx1 = None
                elif XN >= 2 and TB == 2 * NSA and blk > 0:
                    x1 = nx1
                elif XN and TB == 2 * NSA:   # round 005 (XN): core pid normalises tokens [t0 + pid NSA, +NSA) only;
                    # the two halves are exchanged through HBM (bf16, the same values the other core would compute)
                    xo = nl.ndarray((128, HT, NSA), dtype=nl.bfloat16, buffer=nl.sbuf)
                    _norm_blk(h, t0 + pid * NSA, NSA, w1, onesb, HT, xo, 0)
                    _hs2(x1x[:, :, nl.ds(t0 + pid * NSA, NSA)], xo)
                    nisa.core_barrier(data=x1x, cores=(0, 1))
                    x1 = nl.ndarray((128, HT, TB), dtype=nl.bfloat16, buffer=nl.sbuf)
                    _hs2(x1, x1x[:, :, nl.ds(t0, TB)])
                elif hs is not None:
                    x1 = nl.ndarray((128, HT, TB), dtype=nl.bfloat16, buffer=nl.sbuf)
                    for sb in range(TB // NSA):
                        _norm_sb(hs, t0 + sb * NSA, NSA, w1, onesb, HT, x1, sb * NSA)
                else:
                    x1 = nl.ndarray((128, HT, TB), dtype=nl.bfloat16, buffer=nl.sbuf)
                    for sb in range(TB // NSA):
                        _norm_blk(h, t0 + sb * NSA, NSA, w1, onesb, HT, x1, sb * NSA)
                xq = None
                if XN >= 2 and TB == 2 * NSA and blk + 1 < S // TB:   # round 005 (XN=2): the next block's own-half
                    # norm1 + exchange pipelined into this block's QKV tiles
                    xq = 0
                    xcur = _norm_pre(h, t0 + TB + pid * NSA, NSA, HT)
                    xon = nl.ndarray((128, HT, NSA), dtype=nl.bfloat16, buffer=nl.sbuf)
                pa_q = None
                NPA = TB // NSA
                if PNA and blk + 1 < S // TB:   # round 005 (PNA): the next block's norm1, its TensorE part
                    # spread over this block's QKV tiles
                    nx1 = nl.ndarray((128, HT, TB), dtype=nl.bfloat16, buffer=nl.sbuf)
                    pa_q = 0
                    pa_cur = _norm_pre(h, t0 + TB, NSA, HT)
                cs = nl.ndarray((128, TB), dtype=nl.float32, buffer=nl.sbuf)
                _hd(cs, cosT[:, nl.ds(t0, TB)])
                sn = nl.ndarray((128, TB), dtype=nl.float32, buffer=nl.sbuf)
                _hd(sn, sinT[:, nl.ds(t0, TB)])
                nts = []
                for nt in range(NQKV):
                    nts.append(nt)
                if KVS:   # core 0 projects k, core 1 projects v (the rank's kv head, replicated on both cores) and
                    # writes both cores' copies (s_k / s_v [2], kcache / vcache [l, 0 / 1]); round 004
                    assert NKL == 1
                    nts = []
                    for nt in range(NQL):
                        nts.append(nt)
                    nts.append(NQL + pid)
                for qi_ in range(len(nts)):
                    nt = nts[qi_]
                    if xq is not None and xq == 0 and qi_ == len(nts) // 3:
                        _norm_post(xcur[0], xcur[1], NSA, w1, onesb, HT, xon, 0)
                        _hs2(x1x[:, :, nl.ds(t0 + TB + pid * NSA, NSA)], xon)
                        xq = 1
                    if xq is not None and xq == 1 and qi_ == (2 * len(nts)) // 3:
                        nisa.core_barrier(data=x1x, cores=(0, 1))
                        nx1 = nl.ndarray((128, HT, TB), dtype=nl.bfloat16, buffer=nl.sbuf)
                        _hs2(nx1, x1x[:, :, nl.ds(t0 + TB, TB)])
                        xq = 2
                    if pa_q is not None and pa_q < NPA and qi_ == (len(nts) * (pa_q + 1)) // (NPA + 1):
                        _norm_post(pa_cur[0], pa_cur[1], NSA, w1, onesb, HT, nx1, pa_q * NSA)
                        pa_q += 1
                        if pa_q < NPA:
                            pa_cur = _norm_pre(h, t0 + TB + pa_q * NSA, NSA, HT)
                    if qw is not None:
                        wt = qw[nt]
                    elif blk == 0 and pre_qkv is not None:
                        wt = pre_qkv[qi_]
                    else:
                        wt = _slab(wqkv[l, pid], nt, HT)
                    ps = nl.ndarray((128, TB), dtype=nl.float32, buffer=nl.psum)
                    for kt in range(HT):
                        nisa.nc_matmul(dst=ps, stationary=wt[:, kt * 128:(kt + 1) * 128], moving=x1[:, kt, :])
                    if nt < NQL + NKL:
                        sq = nl.ndarray((128, TB), dtype=nl.bfloat16, buffer=nl.sbuf)
                        nisa.activation(dst=sq, op=nl.square, data=ps)
                        ss = nl.ndarray((128, TB), dtype=nl.float32, buffer=nl.psum)
                        nisa.nc_matmul(dst=ss, stationary=onesb, moving=sq)
                        rs = nl.ndarray((128, TB), dtype=nl.float32, buffer=nl.sbuf)
                        nisa.tensor_scalar(dst=rs, data=ss, op0=nl.multiply, operand0=1.0 / D, op1=nl.add, operand1=EPS)
                        nisa.tensor_scalar(dst=rs, data=rs, op0=nl.rsqrt, operand0=0.0, engine=nisa.engine.gpsimd)
                        yb = nl.ndarray((128, TB), dtype=nl.bfloat16, buffer=nl.sbuf)
                        if nt < NQL:
                            nisa.scalar_tensor_tensor(dst=yb, data=ps, op0=nl.multiply, operand0=qnc, op1=nl.multiply, operand1=rs)
                        else:
                            nisa.scalar_tensor_tensor(dst=yb, data=ps, op0=nl.multiply, operand0=knc, op1=nl.multiply, operand1=rs)
                        rp = nl.ndarray((128, TB), dtype=nl.float32, buffer=nl.psum)
                        nisa.nc_matmul(dst=rp, stationary=rot_sb, moving=yb)
                        xc = nl.ndarray((128, TB), dtype=nl.float32, buffer=nl.sbuf)
                        nisa.tensor_tensor(dst=xc, data1=yb, data2=cs, op=nl.multiply)
                        xs = nl.ndarray((128, TB), dtype=nl.float32, buffer=nl.sbuf)
                        nisa.tensor_tensor(dst=xs, data1=rp, data2=sn, op=nl.multiply)
                        ro = nl.ndarray((128, TB), dtype=nl.bfloat16, buffer=nl.sbuf)
                        nisa.tensor_tensor(dst=ro, data1=xc, data2=xs, op=nl.add, engine=nisa.engine.gpsimd)
                        if nt < NQL:
                            _hs2(s_q[pid * NQL + nt, :, nl.ds(t0, TB)], ro)
                        else:
                            j = nt - NQL
                            if KVS:
                                for c in range(2):
                                    _hs2(s_k[c * NKL + j, :, nl.ds(t0, TB)], ro)
                                    _hs2(kcache[l, c, j, :, nl.ds(t0, TB)], ro)
                            else:
                                _hs2(s_k[pid * NKL + j, :, nl.ds(t0, TB)], ro)
                                _hs2(kcache[l, pid, j, :, nl.ds(t0, TB)], ro)
                    else:
                        j = nt - NQL - NKL
                        vb = nl.ndarray((128, TB), dtype=nl.bfloat16, buffer=nl.sbuf)
                        nisa.activation(dst=vb, op=nl.copy, data=ps)
                        vr = nl.ndarray((128, TB // 128, D), dtype=nl.bfloat16, buffer=nl.sbuf)
                        for tb in range(TB // 128):
                            tp = nl.ndarray((128, D), dtype=nl.float32, buffer=nl.psum)
                            nisa.nc_matmul(dst=tp, stationary=vb[:, tb * 128:(tb + 1) * 128], moving=ident)
                            nisa.tensor_copy(dst=vr[:, tb, :], src=tp)
                        cl = [pid]
                        if KVS:
                            cl = [0, 1]
                        for tb in range(TB // 128):
                            for c in cl:
                                _hs2(s_v[c * NKL + j, nl.ds(t0 + tb * 128, 128), :], vr[:, tb, :])
                                if VPERM:
                                    VJ = vcache.shape[2] // 128
                                    NKD = vcache.shape[3]
                                    _hs2(vcache[l, c].ap(pattern=[[VJ * NKD, 128], [1, D]],
                                                        offset=((t0 + tb * 128) // 128) * NKD + j * D), vr[:, tb, :])
                                else:
                                    _hs2(vcache[l, c, nl.ds(t0 + tb * 128, 128), j * D:(j + 1) * D], vr[:, tb, :])
                assert pa_q is None or pa_q == NPA
                assert xq is None or xq == 2
            nisa.core_barrier(data=s_q, cores=(0, 1))
            nisa.core_barrier(data=s_k, cores=(0, 1))
            nisa.core_barrier(data=s_v, cores=(0, 1))
            # ===== B: attention (nkilib), heads split over the cores as our QKV split
            if NOATT == 0:   # NOATT: timing experiment only
                _attention_cte(q=s_q, k=s_k, v=s_v, scale=SCALE, causal_mask=True, tp_q=False, tp_k=False, tp_out=True,
                               out_o_hbm=s_o)
            nisa.core_barrier(data=s_o, cores=(0, 1))
            # ===== C: o_proj K-split (own heads) -> partial h for all hidden tiles -> s_p[pid]
            pend = None
            HH = HT // 2
            LATE = 0
            if NSP and TB == TE:
                LATE = NSP_LATE
            if NSP:   # N-split over the two physical cores: core pid computes hidden tiles [pid*HH, pid*HH+HH) over all
                # 2*NQL heads of the rank (wo[l, 0] and wo[l, 1] hold the K rows of the two cores' heads) and writes the
                # rank sum directly (bf16) into the all-reduce buffer: no fp32 partial exchange through s_p
                for blk in range(S // TB * (1 - LATE)):   # (NSP and LATE: o_proj blocks run in the merged loop below)
                    t0 = blk * TB
                    oc = nl.ndarray((128, 2 * NQL, TB), dtype=nl.bfloat16, buffer=nl.sbuf)
                    for hh in range(2 * NQL):
                        _hs2(oc[:, hh, :], s_o[hh, :, nl.ds(t0, TB)])
                    for j in range(HH):
                        nt = pid * HH + j
                        w0 = _slab(wo[l, 0], nt, NQL)
                        w1 = _slab(wo[l, 1], nt, NQL)
                        ps = nl.ndarray((128, TB), dtype=nl.float32, buffer=nl.psum)
                        for kt in range(NQL):
                            nisa.nc_matmul(dst=ps, stationary=w0[:, kt * 128:(kt + 1) * 128], moving=oc[:, kt, :])
                        for kt in range(NQL):
                            nisa.nc_matmul(dst=ps, stationary=w1[:, kt * 128:(kt + 1) * 128], moving=oc[:, NQL + kt, :])
                        po = nl.ndarray((128, TB), dtype=hb_a.dtype, buffer=nl.sbuf)
                        nisa.activation(dst=po, op=nl.copy, data=ps)
                        _hqb(hb_a[blk, :, nt, :], po)
                    nisa.core_barrier(data=hb_a, cores=(0, 1))
                    if NOAR:
                        pend_a.append(hb_a)
                    else:
                        ncc.all_reduce(srcs=[hb_a[blk]], dsts=[hr_a[blk]], op=nl.add, replica_group=rg)
                        pend_a.append(hr_a)
            pre_gu = []
            for i in range(PFG):   # gate/up slabs of the first MLP block stream in during the attention all-reduce
                pre_gu.append((_slab(wgu[l, pid], i, HT), _slab(wgu[l, pid], ITL + i, HT)))
            if NSP:
                if not LATE:
                    for blk in range(S // TB):
                        _hbm_finish(h, pid, blk * TB, TB, HT, pend_a[blk], blk, CCBF)
            for blk in range(S // TB * (1 - NSP)):
                t0 = blk * TB
                oc = nl.ndarray((128, NQL, TB), dtype=nl.bfloat16, buffer=nl.sbuf)
                for hh in range(NQL):
                    _hd(oc[:, hh, :], s_o[pid * NQL + hh, :, nl.ds(t0, TB)])
                for nt in range(HT):
                    wt = _slab(wo[l, pid], nt, NQL)
                    ps = nl.ndarray((128, TB), dtype=nl.float32, buffer=nl.psum)
                    for kt in range(NQL):
                        nisa.nc_matmul(dst=ps, stationary=wt[:, kt * 128:(kt + 1) * 128], moving=oc[:, kt, :])
                    po = nl.ndarray((128, TB), dtype=nl.float32, buffer=nl.sbuf)
                    nisa.activation(dst=po, op=nl.copy, data=ps)
                    _hd(s_p[pid, :, nt, nl.ds(t0, TB)], po)
                if OVL == 3:   # HBM all-reduce of this block in flight while the next block computes; finish after the loop
                    nisa.core_barrier(data=s_p, cores=(0, 1))
                    pend_a.append(_hbm_launch(s_p, pid, t0, TB, HT, rg, hb_a, hr_a, blk, CCBF))
                if OVL == 1:   # combine this block now: its all-reduce can overlap the next block's GEMMs
                    nisa.core_barrier(data=s_p, cores=(0, 1))
                    _combine_tp(h, s_p, pid, t0, TB, HT, rg, W)
                if OVL == 2:   # software-pipelined: launch this block's all-reduce, consume the previous block's
                    nisa.core_barrier(data=s_p, cores=(0, 1))
                    nxt = _combine_launch(s_p, pid, t0, TB, HT, rg, W)
                    if pend is not None:
                        _combine_finish(h, pid, HT, pend)
                    pend = nxt
            if OVL == 2:
                _combine_finish(h, pid, HT, pend)
                pend = None
            if OVL == 3 and not NSP:
                for blk in range(S // TB):
                    _hbm_finish(h, pid, blk * TB, TB, HT, pend_a[blk], blk, CCBF)
            if not OVL and not NSP:
                nisa.core_barrier(data=s_p, cores=(0, 1))
                if CCHBM:
                    _combine_tp_hbm(h, s_p, pid, S, HT, rg, hb_a, hr_a, CN, CCBF)
                else:
                    for blk in range(S // TB):
                        _combine_tp(h, s_p, pid, blk * TB, TB, HT, rg, W, CCN, CCBF)
            if not (NSP and LATE):   # (NSP and LATE: after the last o_proj block, in the merged loop)
                nisa.core_barrier(data=h, cores=(0, 1))
            # ===== D+E: norm2 + gate/up (own intermediate tiles) + SiLU*up + down K-split -> partial -> combine
            if NCH2 > 0:   # round 006: chunks of the MLP-side norms (PN / FN0 / LATE norm2), default NCH
                NCHK[0] = NCH2
            if ILV and not (NSP and LATE):
                ILV = 0
            NBL = S // TB
            LAG = NBL
            if ILV:   # round 004: o_proj block b, then MLP block b - ILV (its attention all-reduce has had time)
                LAG = ILV
            NIT = S // TE + LAG
            if not (NSP and LATE):
                NIT = S // TE
                LAG = 0
            nx2 = None
            for it in range(NIT):
                if NSP and LATE and it < NBL:
                    blk = it
                    t0 = blk * TB
                    oc = nl.ndarray((128, 2 * NQL, TB), dtype=nl.bfloat16, buffer=nl.sbuf)
                    for hh in range(2 * NQL):
                        _hs2(oc[:, hh, :], s_o[hh, :, nl.ds(t0, TB)])
                    for j in range(HH):
                        nt = pid * HH + j
                        if blk == 0 and pre_o_in is not None:
                            w0 = pre_o_in[j][0]
                            w1 = pre_o_in[j][1]
                        else:
                            w0 = _slab(wo[l, 0], nt, NQL)
                            w1 = _slab(wo[l, 1], nt, NQL)
                        ps = nl.ndarray((128, TB), dtype=nl.float32, buffer=nl.psum)
                        for kt in range(NQL):
                            nisa.nc_matmul(dst=ps, stationary=w0[:, kt * 128:(kt + 1) * 128], moving=oc[:, kt, :])
                        for kt in range(NQL):
                            nisa.nc_matmul(dst=ps, stationary=w1[:, kt * 128:(kt + 1) * 128], moving=oc[:, NQL + kt, :])
                        po = nl.ndarray((128, TB), dtype=hb_a.dtype, buffer=nl.sbuf)
                        nisa.activation(dst=po, op=nl.copy, data=ps)
                        _hqb(hb_a[blk, :, nt, :], po)
                    nisa.core_barrier(data=hb_a, cores=(0, 1))
                    if NOAR:
                        pend_a.append(hb_a)
                    else:
                        ncc.all_reduce(srcs=[hb_a[blk]], dsts=[hr_a[blk]], op=nl.add, replica_group=rg)
                        pend_a.append(hr_a)
                    if it == NBL - 1 and not ILV and hs is None:
                        nisa.core_barrier(data=h, cores=(0, 1))
                if it >= LAG:
                    blk = it - LAG
                    t0 = blk * TE
                    x2 = nl.ndarray((128, HT, TE), dtype=nl.bfloat16, buffer=nl.sbuf)
                    if nx2 is not None:   # PN: computed during the previous MLP block
                        x2 = nx2
                        nx2 = None
                    elif hs is not None:   # HS: attention all-reduce added to the SBUF residual, norm2 from there
                        _sb_add(hs, t0, TE, HT, pend_a[blk], blk, CCBF, NSUB)
                        for sb in range(TE // NSUB):
                            _norm_sb(hs, t0 + sb * NSUB, NSUB, w2, onesb, HT, x2, sb * NSUB)
                    elif LATE and (FN or (FN0 and blk == 0)):   # fused finish + norm2 of this block (round 004;
                        # FN0, round 005: the first MLP block only)
                        _finish_norm(h, pid, t0, TE, HT, pend_a[blk], blk, CCBF, w2, onesb, x2, NSUB)
                    else:
                        if LATE:   # finish this block's attention all-reduce only now: the later ones overlap the MLP blocks
                            _hbm_finish(h, pid, t0, TE, HT, pend_a[blk], blk, CCBF)
                            nisa.core_barrier(data=h, cores=(0, 1))
                        for sb in range(TE // NSUB):
                            _norm_blk(h, t0 + sb * NSUB, NSUB, w2, onesb, HT, x2, sb * NSUB)
                    act = nl.ndarray((128, ITL, TE), dtype=nl.bfloat16, buffer=nl.sbuf)
                    for i in range(ITL * HALF):   # HALF layout (step_tp._gu_half_slabs): ITF full tiles + an IH-row half
                        UB = ITF * HT * 128 + HT * IH
                        if i < ITF:
                            wg = nl.ndarray((128, HT * 128), dtype=wgu.dtype, buffer=nl.sbuf)
                            _hd(wg, wgu[l, pid][:, nl.ds(i * HT * 128, HT * 128)])
                            wu = nl.ndarray((128, HT * 128), dtype=wgu.dtype, buffer=nl.sbuf)
                            _hd(wu, wgu[l, pid][:, nl.ds(UB + i * HT * 128, HT * 128)])
                            pg = nl.ndarray((128, TE), dtype=nl.float32, buffer=nl.psum)
                            for kt in range(HT):
                                nisa.nc_matmul(dst=pg, stationary=wg[:, kt * 128:(kt + 1) * 128], moving=x2[:, kt, :])
                            pu = nl.ndarray((128, TE), dtype=nl.float32, buffer=nl.psum)
                            for kt in range(HT):
                                nisa.nc_matmul(dst=pu, stationary=wu[:, kt * 128:(kt + 1) * 128], moving=x2[:, kt, :])
                            sg = nl.ndarray((128, TE), dtype=nl.float32, buffer=nl.sbuf)
                            nisa.activation(dst=sg, op=nl.silu, data=pg)
                            nisa.tensor_tensor(dst=act[:, i, :], data1=sg, data2=pu, op=nl.multiply)
                        else:
                            wg = nl.ndarray((128, HT * IH), dtype=wgu.dtype, buffer=nl.sbuf)
                            _hd(wg, wgu[l, pid][:, nl.ds(ITF * HT * 128, HT * IH)])
                            wu = nl.ndarray((128, HT * IH), dtype=wgu.dtype, buffer=nl.sbuf)
                            _hd(wu, wgu[l, pid][:, nl.ds(UB + ITF * HT * 128, HT * IH)])
                            pg = nl.ndarray((IH, TE), dtype=nl.float32, buffer=nl.psum)
                            for kt in range(HT):
                                nisa.nc_matmul(dst=pg, stationary=wg[:, kt * IH:(kt + 1) * IH], moving=x2[:, kt, :])
                            pu = nl.ndarray((IH, TE), dtype=nl.float32, buffer=nl.psum)
                            for kt in range(HT):
                                nisa.nc_matmul(dst=pu, stationary=wu[:, kt * IH:(kt + 1) * IH], moving=x2[:, kt, :])
                            sg = nl.ndarray((IH, TE), dtype=nl.float32, buffer=nl.sbuf)
                            nisa.activation(dst=sg, op=nl.silu, data=pg)
                            nisa.tensor_tensor(dst=act[0:IH, i, :], data1=sg, data2=pu, op=nl.multiply)
                    NFU = ITL - MP   # round 009 (MP): 12 full gate / up pairs + the packed [gate half | up half] slab
                    assert not (MP and (HALF or len(pre_gu) or not NSP))
                    for i in range(NFU * (1 - HALF)):
                        if blk == 0 and i < len(pre_gu):
                            wg = pre_gu[i][0]
                        else:
                            wg = _slab(wgu[l, pid], i, HT)
                        pg = nl.ndarray((128, TE), dtype=nl.float32, buffer=nl.psum)
                        for kt in range(HT):
                            nisa.nc_matmul(dst=pg, stationary=wg[:, kt * 128:(kt + 1) * 128], moving=x2[:, kt, :])
                        if blk == 0 and i < len(pre_gu):
                            wu = pre_gu[i][1]
                        else:
                            wu = _slab(wgu[l, pid], NFU + i, HT)
                        pu = nl.ndarray((128, TE), dtype=nl.float32, buffer=nl.psum)
                        for kt in range(HT):
                            nisa.nc_matmul(dst=pu, stationary=wu[:, kt * 128:(kt + 1) * 128], moving=x2[:, kt, :])
                        sg = nl.ndarray((128, TE), dtype=nl.float32, buffer=nl.sbuf)
                        nisa.activation(dst=sg, op=nl.silu, data=pg)
                        nisa.tensor_tensor(dst=act[:, i, :], data1=sg, data2=pu, op=nl.multiply)
                    if MP:   # packed slab: psum rows 0-63 = this core's gate half, 64-127 its up half
                        wp = _slab(wgu[l, pid], 2 * NFU, HT)
                        pp = nl.ndarray((128, TE), dtype=nl.float32, buffer=nl.psum)
                        for kt in range(HT):
                            nisa.nc_matmul(dst=pp, stationary=wp[:, kt * 128:(kt + 1) * 128], moving=x2[:, kt, :])
                        sgh = nl.ndarray((64, TE), dtype=nl.float32, buffer=nl.sbuf)
                        nisa.activation(dst=sgh, op=nl.silu, data=pp[0:64, :])
                        nisa.tensor_tensor(dst=act[0:64, NFU, :], data1=sgh, data2=pp[64:128, :], op=nl.multiply)
                    pn_q = None
                    if PN and not FN and blk + 1 < S // TE:   # round 005 (PN): the next MLP block's attention
                        # finish + norm2 now, so VectorE runs it during this block's down projection
                        nx2 = nl.ndarray((128, HT, TE), dtype=nl.bfloat16, buffer=nl.sbuf)
                        n0 = (blk + 1) * TE
                        if LATE:
                            _hbm_finish(h, pid, n0, TE, HT, pend_a[blk + 1], blk + 1, CCBF)
                            nisa.core_barrier(data=h, cores=(0, 1))
                        if PN >= 2:   # load + square now, the TensorE part interleaved with the down loop below
                            pn_q = 0   # sub-chunks of nx2 done
                            pn_cur = _norm_pre(h, n0, NSUB, HT)
                        else:
                            for sb in range(TE // NSUB):
                                _norm_blk(h, n0 + sb * NSUB, NSUB, w2, onesb, HT, nx2, sb * NSUB)
                    if NSP:   # N-split down: exchange the SwiGLU activations through HBM, core pid computes its half of
                        # the hidden tiles over all 2*ITL intermediate tiles of the rank, bf16 straight into the all-reduce buffer
                        if MP:   # 12 tiles + the half (rows 0-63 of slot 12); oth slot 12 = the combo tile (rows 0-63
                            # core 0's half, 64-127 core 1's)
                            _hqx(act_x[pid, :, 0:NFU, nl.ds(t0, TE)], act[:, 0:NFU, :])
                            _hqx(act_x[pid, 0:64, NFU, nl.ds(t0, TE)], act[0:64, NFU, :])
                            nisa.core_barrier(data=act_x, cores=(0, 1))
                            oth = nl.ndarray((128, ITL, TE), dtype=nl.bfloat16, buffer=nl.sbuf)
                            _hqx(oth[:, 0:NFU, :], act_x[1 - pid, :, 0:NFU, nl.ds(t0, TE)])
                            _hqx(oth[0:64, NFU, :], act_x[0, 0:64, NFU, nl.ds(t0, TE)])
                            _hqx(oth[64:128, NFU, :], act_x[1, 0:64, NFU, nl.ds(t0, TE)])
                        else:
                            _hqx(act_x[pid, :, :, nl.ds(t0, TE)], act)
                            nisa.core_barrier(data=act_x, cores=(0, 1))
                            oth = nl.ndarray((128, ITL, TE), dtype=nl.bfloat16, buffer=nl.sbuf)
                            _hqx(oth, act_x[1 - pid, :, :, nl.ds(t0, TE)])
                        if HALF:   # region B (the half k-tile) of this core's output tiles, from both cores' slices
                            BO = HT * ITF * 128 + pid * HH * 128
                            wbm = nl.ndarray((IH, HH * 128), dtype=wd.dtype, buffer=nl.sbuf)
                            _hd(wbm, wd[l, pid][0:IH, nl.ds(BO, HH * 128)])
                            wbx = nl.ndarray((IH, HH * 128), dtype=wd.dtype, buffer=nl.sbuf)
                            _hd(wbx, wd[l, 1 - pid][0:IH, nl.ds(BO, HH * 128)])
                        for j in range(HH * HALF):
                            nt = pid * HH + j
                            wm = nl.ndarray((128, ITF * 128), dtype=wd.dtype, buffer=nl.sbuf)
                            _hd(wm, wd[l, pid][:, nl.ds(nt * ITF * 128, ITF * 128)])
                            wx = nl.ndarray((128, ITF * 128), dtype=wd.dtype, buffer=nl.sbuf)
                            _hd(wx, wd[l, 1 - pid][:, nl.ds(nt * ITF * 128, ITF * 128)])
                            ps = nl.ndarray((128, TE), dtype=nl.float32, buffer=nl.psum)
                            for kt in range(ITF):
                                nisa.nc_matmul(dst=ps, stationary=wm[:, kt * 128:(kt + 1) * 128], moving=act[:, kt, :])
                            nisa.nc_matmul(dst=ps, stationary=wbm[:, j * 128:(j + 1) * 128], moving=act[0:IH, ITF, :])
                            for kt in range(ITF):
                                nisa.nc_matmul(dst=ps, stationary=wx[:, kt * 128:(kt + 1) * 128], moving=oth[:, kt, :])
                            nisa.nc_matmul(dst=ps, stationary=wbx[:, j * 128:(j + 1) * 128], moving=oth[0:IH, ITF, :])
                            po = nl.ndarray((128, TE), dtype=hb_m.dtype, buffer=nl.sbuf)
                            nisa.activation(dst=po, op=nl.copy, data=ps)
                            _hqb(hb_m[blk, :, nt, :], po)
                        for j in range(HH * (1 - HALF)):
                            if pn_q is not None and pn_q < TE // NSUB and j == (HH * (pn_q + 1)) // (TE // NSUB + 1):
                                _norm_post(pn_cur[0], pn_cur[1], NSUB, w2, onesb, HT, nx2, pn_q * NSUB)
                                pn_q += 1
                                if pn_q < TE // NSUB:
                                    pn_cur = _norm_pre(h, (blk + 1) * TE + pn_q * NSUB, NSUB, HT)
                            nt = pid * HH + j
                            wm = _slab(wd[l, pid], nt, ITL)
                            if MP:   # 12 own + 12 other tiles + the combo (its weights in slot 12 of both cores' slices)
                                wx = nl.ndarray((128, ITL * 128), dtype=wd.dtype, buffer=nl.sbuf)
                                _hd(wx[:, 0:NFU * 128], wd[l, 1 - pid][:, nl.ds(nt * ITL * 128, NFU * 128)])
                            else:
                                wx = _slab(wd[l, 1 - pid], nt, ITL)
                            ps = nl.ndarray((128, TE), dtype=nl.float32, buffer=nl.psum)
                            for kt in range(NFU):
                                nisa.nc_matmul(dst=ps, stationary=wm[:, kt * 128:(kt + 1) * 128], moving=act[:, kt, :])
                            for kt in range(NFU):
                                nisa.nc_matmul(dst=ps, stationary=wx[:, kt * 128:(kt + 1) * 128], moving=oth[:, kt, :])
                            if MP:
                                nisa.nc_matmul(dst=ps, stationary=wm[:, NFU * 128:(NFU + 1) * 128], moving=oth[:, NFU, :])
                            po = nl.ndarray((128, TE), dtype=hb_m.dtype, buffer=nl.sbuf)
                            nisa.activation(dst=po, op=nl.copy, data=ps)
                            _hqb(hb_m[blk, :, nt, :], po)
                        assert pn_q is None or pn_q == TE // NSUB
                        nisa.core_barrier(data=hb_m, cores=(0, 1))
                        if NOAR:
                            pend_m.append(hb_m)
                        else:
                            ncc.all_reduce(srcs=[hb_m[blk]], dsts=[hr_m[blk]], op=nl.add, replica_group=rg)
                            pend_m.append(hr_m)
                        if PMF and blk >= PMF:   # finish MLP block blk - PMF's all-reduce now, not all in the tail
                            fb = blk - PMF       # (round 005: PMF = lag in blocks; round 004 measured lag 1 only)
                            if LATE and (FN or (FN0 and fb == 0)):   # (round 006: FN0's block 0 too)
                                _hbm_finish2(h, pid, fb * TE, TE, HT, pend_a[fb], pend_m[fb], fb, CCBF)
                            else:
                                _hbm_finish(h, pid, fb * TE, TE, HT, pend_m[fb], fb, CCBF)
                    for nt in range(HT * (1 - NSP)):
                        wt = _slab(wd[l, pid], nt, ITL)
                        ps = nl.ndarray((128, TE), dtype=nl.float32, buffer=nl.psum)
                        for kt in range(ITL):
                            nisa.nc_matmul(dst=ps, stationary=wt[:, kt * 128:(kt + 1) * 128], moving=act[:, kt, :])
                        po = nl.ndarray((128, TE), dtype=nl.float32, buffer=nl.sbuf)
                        nisa.activation(dst=po, op=nl.copy, data=ps)
                        _hd(s_p[pid, :, nt, nl.ds(t0, TE)], po)
                    if OVL == 3 and not NSP:
                        nisa.core_barrier(data=s_p, cores=(0, 1))
                        pend_m.append(_hbm_launch(s_p, pid, t0, TE, HT, rg, hb_m, hr_m, blk, CCBF))
                    if OVL == 1:
                        nisa.core_barrier(data=s_p, cores=(0, 1))
                        _combine_tp(h, s_p, pid, t0, TE, HT, rg, W)
                    if OVL == 2:
                        nisa.core_barrier(data=s_p, cores=(0, 1))
                        nxt = _combine_launch(s_p, pid, t0, TE, HT, rg, W)
                        if pend is not None:
                            _combine_finish(h, pid, HT, pend)
                        pend = nxt
            if OVL == 2:
                _combine_finish(h, pid, HT, pend)
                pend = None
            pre_qkv = None
            pre_o = None
            if PFQ and l + 1 < LPN:   # the next layer's QKV slabs stream in during the MLP all-reduce
                pre_qkv = []
                for nt in nts:
                    pre_qkv.append(_slab(wqkv[l + 1, pid], nt, HT))
                if PFQ >= 2:   # and its o_proj slabs (both cores' K rows, own hidden tiles)
                    pre_o = []
                    for j in range(HH):
                        pre_o.append((_slab(wo[l + 1, 0], pid * HH + j, NQL), _slab(wo[l + 1, 1], pid * HH + j, NQL)))
            if hs is not None:   # HS: the MLP all-reduce added in SBUF; h written once, after the last layer
                _sb_add(hs, 0, S, HT, pend_m[0], 0, CCBF, NSUB)
                if l == LPN - 1:
                    HH = HT // 2
                    _hd(h[:, nl.ds(pid * HH, HH), :], hs[:, nl.ds(pid * HH, HH), :])
                    nisa.core_barrier(data=h, cores=(0, 1))
            elif OVL == 3 or NSP:
                for blk in range(S // TE):
                    if not (PMF and NSP and blk < S // TE - PMF):
                        if LATE and (FN or (FN0 and blk == 0)):
                            _hbm_finish2(h, pid, blk * TE, TE, HT, pend_a[blk], pend_m[blk], blk, CCBF)
                        else:
                            _hbm_finish(h, pid, blk * TE, TE, HT, pend_m[blk], blk, CCBF)
            if not OVL and not NSP:
                nisa.core_barrier(data=s_p, cores=(0, 1))
                if CCHBM:
                    _combine_tp_hbm(h, s_p, pid, S, HT, rg, hb_m, hr_m, CN, CCBF)
                else:
                    for blk in range(S // TB):
                        _combine_tp(h, s_p, pid, blk * TB, TB, HT, rg, W, CCN, CCBF)
            if hs is None:
                nisa.core_barrier(data=h, cores=(0, 1))
        nisa.dma_copy(dst=pos_out, src=pl_sb)
        return pos_out

    # STAGE 3: LM head on token plen - 1
    lm1 = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=lm1, data=pl_sb, op0=nl.add, operand0=-1)
    lreg = nisa.register_alloc()
    nisa.register_load(lreg, lm1)
    hl = nl.ndarray((128, HT), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=hl, src=h.ap(pattern=[[HT * S, 128], [S, HT]], offset=0, scalar_offset=lreg, indirect_dim=2))
    wf = nl.ndarray((128, HT), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=wf, src=normf)
    NVL = wlm.shape[2] // (HT * 128)
    if SP:   # round 007: only the owner rank of token plen - 1 holds its residual
        hl = head_sp(hl, plen, rkb, NVL, W, rg)
    xf = _rms_col(hl, wf, HT, HID, ones, nl.bfloat16)
    tokf = _lm_argmax_tp(xf, wlm[pid], pid, rkb, 1, 2, HT, NVL, rg, W)
    tok_i = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=tok_i, src=tokf)
    preg = nisa.register_alloc()
    nisa.register_load(preg, pl_sb)
    nisa.dma_copy(dst=toks.ap(pattern=[[1, 1], [1, 1]], offset=0, scalar_offset=preg, indirect_dim=0), src=tok_i)
    nisa.dma_copy(dst=pos_out, src=pl_sb)
    return pos_out



@nki.jit
def prefill_tp(ids, plen, toks, embed, ln1, ln2, qn, kn, wqkv, wo, wgu, wd, wgp, wdp, normf, wlm, rot, cosT, sinT, kcache, vcache,
               rkb, s_h, s_q, s_k, s_v, s_o, s_p, STAGE, NQL, NKL, ITL, W, rg, OVL=0, TEB=256, CCN=128, CCBF=0, CCHBM=0, NSP=0, SB=0, KVS=0, LPN=1, PFG=0, PFQ=0, FN=0, PMF=0, HALF=0, SQL=0, ILV=0, QW=0, QNS=0, AQL=0, NSP2=0, PN=0, PNA=0, XN=0, FN0=0, HS=0, HSC=1, NCH=1, NCQL=0, NCH2=0, SP=0, SPL=-1, SPN=0, SPT=0, SPO=0, SPA=0, SPD=0, SPH=0, SPHN=0, SPP=0, SPQ=0, GCH=1, SOC=1, SPS=1, SPG=0, KTP=0, KTQ=0, MP=0):
    if MP:   # round 009: the SP path reads the unpadded prefill MLP copies
        wgu = wgp
        wd = wdp
    return _pg(ids, plen, toks, embed, ln1, ln2, qn, kn, wqkv, wo, wgu, wd, normf, wlm, rot, cosT, sinT, kcache, vcache,
               rkb, s_h, s_q, s_k, s_v, s_o, s_p, STAGE, NQL, NKL, ITL, W, rg, OVL, TEB, CCN, CCBF, CCHBM, NSP, SB, KVS, LPN, PFG, PFQ, FN, PMF, HALF, SQL, ILV, QW, QNS, AQL, NSP2, PN, PNA, XN, FN0, HS, HSC, NCH, NCQL, NCH2, SP, SPL, SPN, SPT, SPO, SPA, SPD, SPH, SPHN, SPP, SPQ, GCH, SOC, SPS, SPG, KTP, KTQ, MP)
