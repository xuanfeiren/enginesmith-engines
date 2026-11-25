"""NKI kernels for Gemma-3-4B (text) on one logical NeuronCore (LNC=2: two physical cores).

Layouts (all fixed at compile time):
  * hidden state in HBM: h[D, T] bf16, feature-major (row = feature). In SBUF: x[p, kt, t] = h[kt*128 + p, t].
  * weights: per matrix and core a [2, 128, NTILES*128] bf16 array, tile i = (nt, kt) in row-major order,
    w[c, p, i*128 + j] = W_c[nt*128 + j, kt*128 + p]  (W_c is the core's slice of the HF [out, in] matrix).
    Tile (nt, kt) is the stationary operand of the TensorE; the activation column(s) x[:, kt, :] are moving.
  * K cache transposed: kc[g, d, s] (4 KV heads x 256 x S); V cache natural: vc[s, g*256 + d].
  * RoPE tables cos/sin [128, S] bf16 (the two halves of the 256 rotary dims share a table row).

Work split over the two physical cores (pid = nl.program_id(0)):
  * q/k/v projections by heads: core c owns q heads 4c..4c+3 and KV heads 2c, 2c+1 (and their cache).
  * attention for the core's own heads.
  * o_proj and down_proj along K (each core its own attention features / intermediate half), the fp32
    partial sums exchanged with nisa.sendrecv and added (a + b == b + a, so both cores hold the same bits).
  * gate/up along N (the core's half of the intermediate features).
  * norms and residual adds redundantly on both cores; each core writes half of the output rows.
"""
import nki
import nki.language as nl
import nki.isa as nisa

D = 2560
KT = 20
HD = 256
S = 8192            # cache positions (max context)
IC = 5120           # intermediate features per core
ICT = 40
EPS = 1e-6
VPC = 131200        # vocabulary rows per core (262208 padded to 2 x 1025 x 128)
VT = 1025
NEG = -1.0e30
BIGID = 1048576.0   # 2^20 > vocab: argmax scores are BIGID - id
DEC_LOCAL_B = 1024  # sliding window (decode local layers)
DEC_LOCAL_SMALL = True   # decode at global bucket 512: local windows of 512 keys
QKC = 256           # prefill attention: keys per score matmul (512-wide moving tiles run at 2/3 rate)
RING_DEC = (24, 16)  # decode weight ring: slots, tiles per slot (4 KB per partition per slot)
RING_PRE = 12        # prefill weight ring slots
WSL = 16
DR_ENGINE = nisa.engine.vector
RS_PIPES = 1        # sendrecv pipe ids the reduce-scatter pieces rotate over
AG_PIPES = 1


class EngCfg(nl.NKIObject):
    """Trace-time engine choices that differ between the single-stream and the two-stream prefill (set by the
    group bodies before they emit anything)."""

    def __init__(self):
        self.rs_add = nisa.engine.vector   # reduce-scatter add (own + received partial sums)


ENG = EngCfg()


def set_two_stream(two):
    """Two streams: the reduce-scatter add on the (otherwise idle) GpSimd, so that it does not queue behind the
    other stream's VectorE work (p2s512A1280_4096K 27.75 -> 27.53 ms); single stream: VectorE (GpSimd measured
    0.2% slower there)."""
    ENG.rs_add = nisa.engine.gpsimd if two else nisa.engine.vector

bf16 = nl.bfloat16
fp32 = nl.float32
i32 = nl.int32


# ----------------------------------------------------------------------------------------------- helpers

def hdma(dst, src):
    """DMA on the Sync engine's hardware DGE queue."""
    nisa.dma_copy(dst=dst, src=src, dge_mode=nisa.dge_mode.hwdge, engine=nisa.engine.sync)


def adma(dst, src):
    """DMA on the Scalar engine's hardware DGE queue (keeps the Sync queue for the weight stream)."""
    nisa.dma_copy(dst=dst, src=src, dge_mode=nisa.dge_mode.hwdge, engine=nisa.engine.scalar)


def make_ident(n, dtype):
    """n x n identity in SBUF (dtype)."""
    io = nl.ndarray((n, n), dtype=fp32, buffer=nl.sbuf)
    nisa.iota(dst=io, pattern=[[-1, n]], offset=0, channel_multiplier=1)   # value = p - j
    idt = nl.ndarray((n, n), dtype=dtype, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=idt, data=io, op0=nl.equal, operand0=0.0)
    return idt


def make_identm():
    """128 x 128 identity as a uint8 predicate."""
    io = nl.ndarray((128, 128), dtype=fp32, buffer=nl.sbuf)
    nisa.iota(dst=io, pattern=[[-1, 128]], offset=0, channel_multiplier=1)   # value = p - j
    m = nl.ndarray((128, 128), dtype=nl.uint8, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=m, data=io, op0=nl.equal, operand0=0.0)
    return m


def make_ones():
    ones = nl.ndarray((128, 128), dtype=fp32, buffer=nl.sbuf)
    nisa.memset(dst=ones, value=1.0)
    return ones


def load_norm_w(src_ap, n):
    """(1 + w) in fp32 for n vectors of D, SBUF [128, n, KT]. src_ap: HBM view [n, D]."""
    wsb = nl.ndarray((128, n, KT), dtype=bf16, buffer=nl.sbuf)
    nisa.dma_copy(dst=wsb, src=src_ap.ap(pattern=[[n * KT, 128], [KT, n], [1, KT]], offset=0))
    w1 = nl.ndarray((128, n, KT), dtype=fp32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=w1, data=wsb, op0=nl.add, operand0=1.0)
    return w1


def rms_stats(sq, nkt, T, n, ones):
    """sq: fp32 [128, >=nkt, T] squares (tiles 0..nkt-1). Returns rstd [128, T] fp32 on every partition."""
    ss = nl.ndarray((128, T), dtype=fp32, buffer=nl.psum)
    for kt in range(nkt):
        nisa.nc_matmul(dst=ss, stationary=ones, moving=sq[:, kt, :], accumulate=(kt > 0))
    r1 = nl.ndarray((128, T), dtype=fp32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=r1, data=ss, op0=nl.multiply, operand0=1.0 / n, op1=nl.add, operand1=EPS)
    r2 = nl.ndarray((128, T), dtype=fp32, buffer=nl.sbuf)
    nisa.activation(dst=r2, op=nl.sqrt, data=r1)
    rstd = nl.ndarray((128, T), dtype=fp32, buffer=nl.sbuf)
    nisa.reciprocal(dst=rstd, data=r2)
    return rstd


def rmsnorm(x, w1, widx, T, ones):
    """Gemma RMSNorm of x [128, KT, T] bf16 with (1 + w) = w1[:, widx, :]; returns bf16 [128, KT, T]."""
    sq = nl.ndarray((128, KT, T), dtype=fp32, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=sq, data1=x, data2=x, op=nl.multiply)
    rstd = rms_stats(sq, KT, T, D, ones)
    y = nl.ndarray((128, KT, T), dtype=bf16, buffer=nl.sbuf)
    for kt in range(KT):
        tmp = nl.ndarray((128, T), dtype=fp32, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=tmp, data1=x[:, kt, :], data2=rstd, op=nl.multiply)
        nisa.tensor_scalar(dst=y[:, kt, :], data=tmp, op0=nl.multiply, operand0=w1[:, widx, kt:kt + 1])
    return y


def head_norm(x, t0, nh, w1h, T, ones, y):
    """Per-head RMSNorm over 256 dims (tiles t0 + 2h, t0 + 2h + 1) of x [128, *, T]; w1h [128, 2] fp32
    (1 + w) for the two 128-halves; writes y[:, t0 + 2h + dt, :] bf16."""
    for h in range(nh):
        sq = nl.ndarray((128, 2, T), dtype=fp32, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=sq, data1=x[:, t0 + 2 * h:t0 + 2 * h + 2, :], data2=x[:, t0 + 2 * h:t0 + 2 * h + 2, :],
                           op=nl.multiply)
        rstd = rms_stats(sq, 2, T, HD, ones)
        for dt in range(2):
            tmp = nl.ndarray((128, T), dtype=fp32, buffer=nl.sbuf)
            nisa.tensor_tensor(dst=tmp, data1=x[:, t0 + 2 * h + dt, :], data2=rstd, op=nl.multiply)
            nisa.tensor_scalar(dst=y[:, t0 + 2 * h + dt, :], data=tmp, op0=nl.multiply, operand0=w1h[:, dt:dt + 1])


def rope(y, t0, nh, cs, sn, T, out):
    """rotate-half RoPE with HF's bf16 rounding: lo' = bf(bf(lo*c) - bf(hi*s)), hi' = bf(bf(hi*c) + bf(lo*s))."""
    for h in range(nh):
        lo = y[:, t0 + 2 * h, :]
        hi = y[:, t0 + 2 * h + 1, :]
        a = nl.ndarray((128, T), dtype=bf16, buffer=nl.sbuf)
        b = nl.ndarray((128, T), dtype=bf16, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=a, data1=lo, data2=cs, op=nl.multiply)
        nisa.tensor_tensor(dst=b, data1=hi, data2=sn, op=nl.multiply)
        nisa.tensor_tensor(dst=out[:, t0 + 2 * h, :], data1=a, data2=b, op=nl.subtract)
        c = nl.ndarray((128, T), dtype=bf16, buffer=nl.sbuf)
        d = nl.ndarray((128, T), dtype=bf16, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=c, data1=hi, data2=cs, op=nl.multiply)
        nisa.tensor_tensor(dst=d, data1=lo, data2=sn, op=nl.multiply)
        nisa.tensor_tensor(dst=out[:, t0 + 2 * h + 1, :], data1=c, data2=d, op=nl.add)


class WRing(nl.NKIObject):
    """Weight slabs streamed through one SBUF ring of `nbuf` slots of `sl` tiles each, all DMAs on the Sync
    engine's hardware DGE queue (which runs nothing else). Slab i waits only for slot i % nbuf to be consumed, so
    the stream runs ahead of the matmuls by up to nbuf slabs, across GEMMs (and layers)."""

    def __init__(self, nbuf, sl):
        self.nbuf = nbuf
        self.sl = sl
        self.i = 0
        self.bufs = []
        for k in range(nbuf):
            self.bufs.append(nl.ndarray((128, sl * 128), dtype=bf16, buffer=nl.sbuf))

    def load(self, src, n):
        slot = self.i % self.nbuf
        self.i += 1
        b = self.bufs[slot]
        nisa.dma_copy(dst=b[:, 0:n * 128], src=src, dge_mode=nisa.dge_mode.hwdge, engine=nisa.engine.sync)
        return b


class AR(nl.NKIObject):
    """All-reduce (sum over the two cores) of a GEMM's fp32 output, piece by piece as the GEMM produces it; the
    sum is written as bf16 to ob (same rounding as allreduce2 + cast)."""

    def __init__(self, ob, pid, th=0, w1=None, widx=0):
        self.ob = ob
        self.pid = pid
        self.th = th      # > 0: reduce-scatter over tokens (core pid keeps tokens [pid*th, (pid+1)*th))
        self.xw = None    # NORM_SPLIT: ob * (1 + w) in fp32 per piece, for the post norm (rmsnorm_res)
        self.w1 = w1
        self.widx = widx
        if NORM_SPLIT and w1 is not None:
            self.xw = nl.ndarray((128, KT, th), dtype=fp32, buffer=nl.sbuf)


def ar_piece(ar, part, g0, gn, T):
    if ar.th > 0:
        # reduce-scatter: send the other core's tokens, receive the other core's partial sums of ours; the sum
        # (own + other, the same fp32 add as the all-reduce) is written for our tokens only
        th = ar.th
        mine = ar.pid * th
        theirs = (1 - ar.pid) * th
        other = nl.ndarray((128, gn, th), dtype=fp32, buffer=nl.sbuf)
        nisa.sendrecv(src=part[:, g0:g0 + gn, theirs:theirs + th], dst=other, send_to_rank=1 - ar.pid,
                      recv_from_rank=1 - ar.pid, pipe_id=(g0 // 2) % RS_PIPES)
        nisa.tensor_tensor(dst=ar.ob[:, g0:g0 + gn, :], data1=part[:, g0:g0 + gn, mine:mine + th], data2=other,
                           op=nl.add, engine=ENG.rs_add)
        if ar.xw is not None:
            for kt in range(g0, g0 + gn):
                nisa.tensor_scalar(dst=ar.xw[:, kt, :], data=ar.ob[:, kt, :], op0=nl.multiply,
                                   operand0=ar.w1[:, ar.widx, kt:kt + 1], engine=nisa.engine.vector)
        return
    other = nl.ndarray((128, gn, T), dtype=fp32, buffer=nl.sbuf)
    nisa.sendrecv(src=part[:, g0:g0 + gn, :], dst=other, send_to_rank=1 - ar.pid, recv_from_rank=1 - ar.pid, pipe_id=0)
    nisa.tensor_tensor(dst=ar.ob[:, g0:g0 + gn, :], data1=part[:, g0:g0 + gn, :], data2=other, op=nl.add)


def rmsnorm_defer(x, w1, widx, th, pid, identf, onesf, ngrp, identm):
    """Deferred Gemma RMSNorm for a GEMM input: returns [u, rstd] with u = bf16(x * (1 + w)) [128, KT, 2*th] and
    rstd [128, 2*th] fp32 (1/rms per token, on every partition), both all-gathered over the two cores (each core
    computes its th tokens). The consumer applies rstd to the GEMM output (W^T (u * rstd) = rstd * W^T u)."""
    mine = pid * th
    theirs = (1 - pid) * th
    u = nl.ndarray((128, KT, 2 * th), dtype=bf16, buffer=nl.sbuf)
    for kt in range(KT):
        if kt % 2 == 0:
            nisa.tensor_scalar(dst=u[:, kt, mine:mine + th], data=x[:, kt, :], op0=nl.multiply,
                               operand0=w1[:, widx, kt:kt + 1], engine=nisa.engine.vector)
        else:
            nisa.activation(dst=u[:, kt, mine:mine + th], op=nl.copy, data=x[:, kt, :], scale=w1[:, widx, kt:kt + 1])
        if (kt + 1) % ngrp == 0 or kt == KT - 1:
            k0 = (kt // ngrp) * ngrp
            nisa.sendrecv(src=u[:, k0:kt + 1, mine:mine + th], dst=u[:, k0:kt + 1, theirs:theirs + th],
                          send_to_rank=1 - pid, recv_from_rank=1 - pid, pipe_id=(kt // ngrp) % AG_PIPES)
    d = DeferRS(pid, th, onesf)
    d.drs = gram_rstd_a(x, 0, KT, th, D, identf, onesf)
    return [u, d]


class DeferRS(nl.NKIObject):
    """rstd of a deferred norm: statistics emitted with the norm, broadcast + all-gather emitted (defer_finish) just
    before the first evacuation that needs it, so the TensorE runs the GEMM meanwhile."""

    def __init__(self, pid, th, onesf):
        self.pid = pid
        self.th = th
        self.ones = onesf
        self.drs = None
        self.done = False
        self.rstd = nl.ndarray((128, 2 * th), dtype=fp32, buffer=nl.sbuf)


def defer_finish(d):
    if d.done:
        return
    mine = d.pid * d.th
    theirs = (1 - d.pid) * d.th
    gram_rstd_b(d.drs, d.ones, d.rstd, mine)
    nisa.sendrecv(src=d.rstd[:, mine:mine + d.th], dst=d.rstd[:, theirs:theirs + d.th], send_to_rank=1 - d.pid,
                  recv_from_rank=1 - d.pid, pipe_id=0)
    d.done = True


def evac_scaled(dst, ps, d, gn, T):
    """dst [128, gn, T] = ps * rstd (per token, broadcast over the gn tiles), on the VectorE; d: DeferRS."""
    defer_finish(d)
    nisa.tensor_tensor(dst=dst, data1=ps, data2=d.rstd.ap(pattern=[[T, 128], [0, gn], [1, T]], offset=0),
                       op=nl.multiply)


def rmsnorm_ag(x, w1, widx, th, pid, identf, onesf, ngrp, identm=None):
    """Gemma RMSNorm of this core's tokens x [128, KT, th] (bf16), all-gathered over the two cores: returns y
    [128, KT, 2*th] bf16 holding both cores' tokens (ours scaled here, theirs received in pieces of ngrp k-tiles).
    Per token the same arithmetic as rmsnormT."""
    mine = pid * th
    theirs = (1 - pid) * th
    y = nl.ndarray((128, KT, 2 * th), dtype=bf16, buffer=nl.sbuf)
    rstd = nl.ndarray((128, th), dtype=fp32, buffer=nl.sbuf)
    gram_rstd(x, 0, KT, th, D, identf, onesf, rstd, identm)
    for kt in range(KT):
        nisa.scalar_tensor_tensor(dst=y[:, kt, mine:mine + th], data=x[:, kt, :], op0=nl.multiply,
                                  operand0=w1[:, widx, kt:kt + 1], op1=nl.multiply, operand1=rstd)
        if (kt + 1) % ngrp == 0 or kt == KT - 1:
            k0 = (kt // ngrp) * ngrp
            nisa.sendrecv(src=y[:, k0:kt + 1, mine:mine + th], dst=y[:, k0:kt + 1, theirs:theirs + th],
                          send_to_rank=1 - pid, recv_from_rank=1 - pid, pipe_id=0)
    return y


NORM_SPLIT = True   # post norms: x * (1 + w) on the VectorE as each reduce-scatter piece lands, * rstd + residual add batched
RES_GRP = 5         # k-tiles per batched scaling / residual-add instruction (rmsnorm_res)


def gemm(w, x, NT, NK, T, dst, G, SL, ring=None, evac=None, ar=None):
    """dst[:, nt, :] = sum_kt W(nt, kt)^T x[:, kt, :]  (cast to dst's dtype).
    w: HBM view [128, NT*NK*128] of the core's tiles; x: SBUF [128, NK, T] bf16.
    G: n-tiles per PSUM tile (G*T <= 512); SL: tiles per weight DMA (the ring's slab size when ring is given)."""
    ntiles = NT * NK
    if ring is not None:
        SL = ring.sl
    ps = None
    for s0 in range(0, ntiles, SL):
        n = min(SL, ntiles - s0)
        if ring is not None:
            wsb = ring.load(w[:, s0 * 128:(s0 + n) * 128], n)
        else:
            wsb = nl.ndarray((128, SL * 128), dtype=bf16, buffer=nl.sbuf)
            nisa.dma_copy(dst=wsb[:, 0:n * 128], src=w[:, s0 * 128:(s0 + n) * 128])
        for j in range(n):
            i = s0 + j
            nt = i // NK
            kt = i % NK
            g0 = (nt // G) * G
            if kt == 0 and nt == g0:
                ps = nl.ndarray((128, G, T), dtype=fp32, buffer=nl.psum)
            nisa.nc_matmul(dst=ps[:, nt - g0, :], stationary=wsb[:, j * 128:(j + 1) * 128], moving=x[:, kt, :])
            if kt == NK - 1 and (nt == g0 + G - 1 or nt == NT - 1):
                gn = nt - g0 + 1
                if evac is None:
                    nisa.tensor_copy(dst=dst[:, g0:g0 + gn, :], src=ps[:, 0:gn, :])
                else:
                    nisa.tensor_copy(dst=dst[:, g0:g0 + gn, :], src=ps[:, 0:gn, :], engine=evac)
                if ar is not None:
                    ar_piece(ar, dst, g0, gn, T)


def allreduce2(part, nk, T, pid, small):
    """part: fp32 [128, nk, T] partial sums of this core; returns part + other core's part (fp32)."""
    other = nl.ndarray((128, nk, T), dtype=fp32, buffer=nl.sbuf)
    if small:
        nisa.sendrecv(src=part, dst=other, send_to_rank=1 - pid, recv_from_rank=1 - pid, pipe_id=0,
                      dma_engine=nisa.dma_engine.gpsimd_dma)
    else:
        nisa.sendrecv(src=part, dst=other, send_to_rank=1 - pid, recv_from_rank=1 - pid, pipe_id=0)
    tot = nl.ndarray((128, nk, T), dtype=fp32, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=tot, data1=part, data2=other, op=nl.add)
    return tot


def window_bias(R, B, thr, cm, local, dst):
    """dst [R, B] bf16 = 0 where key j is visible to row r, NEG elsewhere.  v = j - cm*r; visible iff
    v <= thr (causal) and, for local layers, v >= thr - 1023.  thr: [R, 1] fp32."""
    io = nl.ndarray((R, B), dtype=fp32, buffer=nl.sbuf)
    nisa.iota(dst=io, pattern=[[1, B]], offset=0, channel_multiplier=-cm)
    m = nl.ndarray((R, B), dtype=fp32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=m, data=io, op0=nl.subtract, operand0=thr, op1=nl.greater, operand1=0.0)
    if local:
        m2 = nl.ndarray((R, B), dtype=fp32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=m2, data=io, op0=nl.subtract, operand0=thr, op1=nl.less, operand1=-1023.5)
        nisa.tensor_tensor(dst=m, data1=m, data2=m2, op=nl.add)
    nisa.tensor_scalar(dst=dst, data=m, op0=nl.multiply, operand0=NEG)


def softmax_rows(sc, R, B):
    """sc fp32 [R, B] (scores + bias). Returns (P bf16 [R, B] = exp((s - max)/16), sum fp32 [R, 1])."""
    mx = nl.ndarray((R, 1), dtype=fp32, buffer=nl.sbuf)
    nisa.tensor_reduce(dst=mx, op=nl.maximum, data=sc, axis=1)
    nmx = nl.ndarray((R, 1), dtype=fp32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=nmx, data=mx, op0=nl.multiply, operand0=-0.0625)
    p = nl.ndarray((R, B), dtype=bf16, buffer=nl.sbuf)
    sm = nl.ndarray((R, 1), dtype=fp32, buffer=nl.sbuf)
    nisa.activation(dst=p, op=nl.exp, data=sc, bias=nmx, scale=0.0625, reduce_op=nl.add, reduce_res=sm,
                    reduce_cmd=nisa.reduce_cmd.reset_reduce)
    return p, sm


# ----------------------------------------------------------------------------------------------- layer

def layer_body(h, pos, wqkv, wo, wgu, wd, lnw, qkw, kc, vc, cos, sin, T, B, local, dbg=False):
    pid = nl.program_id(0)
    small = T == 1
    R = 4 if T == 1 else 128
    ones = make_ones()
    ring = WRing(RING_DEC[0], RING_DEC[1]) if small else None

    # ---- scalars: pos, window start = max(0, pos + T - B), thr = pos - start
    p_i = nl.ndarray((R, 1), dtype=i32, buffer=nl.sbuf)
    nisa.dma_copy(dst=p_i, src=pos.ap(pattern=[[0, R], [1, 1]], offset=0))
    p_f = nl.ndarray((R, 1), dtype=fp32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=p_f, src=p_i)
    st_f = nl.ndarray((R, 1), dtype=fp32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=st_f, data=p_f, op0=nl.add, operand0=float(T - B), op1=nl.maximum, operand1=0.0)
    thr = nl.ndarray((R, 1), dtype=fp32, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=thr, data1=p_f, data2=st_f, op=nl.subtract)
    st_i = nl.ndarray((R, 1), dtype=i32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=st_i, src=st_f)
    preg = nisa.register_alloc()
    nisa.register_load(preg, p_i[0:1, 0:1])
    sreg = nisa.register_alloc()
    nisa.register_load(sreg, st_i[0:1, 0:1])

    # ---- inputs
    x = nl.ndarray((128, KT, T), dtype=bf16, buffer=nl.sbuf)
    nisa.dma_copy(dst=x, src=h.ap(pattern=[[T, 128], [128 * T, KT], [1, T]], offset=0))
    w1 = load_norm_w(lnw, 4)
    qk_sb = nl.ndarray((128, 2, 2), dtype=bf16, buffer=nl.sbuf)
    nisa.dma_copy(dst=qk_sb, src=qkw.ap(pattern=[[4, 128], [2, 2], [1, 2]], offset=0))
    qk1 = nl.ndarray((128, 2, 2), dtype=fp32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=qk1, data=qk_sb, op0=nl.add, operand0=1.0)
    cs = nl.ndarray((128, T), dtype=bf16, buffer=nl.sbuf)
    nisa.dma_copy(dst=cs, src=cos.ap(pattern=[[S, 128], [1, T]], offset=0, scalar_offset=preg, indirect_dim=1))
    sn = nl.ndarray((128, T), dtype=bf16, buffer=nl.sbuf)
    nisa.dma_copy(dst=sn, src=sin.ap(pattern=[[S, 128], [1, T]], offset=0, scalar_offset=preg, indirect_dim=1))

    # ---- attention block
    xn = rmsnorm(x, w1, 0, T, ones)
    qkv = nl.ndarray((128, 16, T), dtype=bf16, buffer=nl.sbuf)
    gemm(wqkv[pid], xn, 16, KT, T, qkv, 16 if small else 1, WSL, ring)
    qn = nl.ndarray((128, 12, T), dtype=bf16, buffer=nl.sbuf)
    head_norm(qkv, 0, 4, qk1[:, 0, :], T, ones, qn)
    head_norm(qkv, 8, 2, qk1[:, 1, :], T, ones, qn)
    qr = nl.ndarray((128, 12, T), dtype=bf16, buffer=nl.sbuf)
    rope(qn, 0, 6, cs, sn, T, qr)

    # cache writes (this core's KV heads 2c, 2c+1)
    kbase = (2 * pid) * HD * S
    if small:
        nisa.dma_copy(dst=kc.ap(pattern=[[S, 128], [128 * S, 4]], offset=kbase, scalar_offset=preg, indirect_dim=2),
                      src=qr[:, 8:12, 0])
        nisa.dma_copy(dst=vc.ap(pattern=[[1, 128], [128, 4]], offset=512 * pid, scalar_offset=preg, indirect_dim=0),
                      src=qkv[:, 12:16, 0])
    else:
        nisa.dma_copy(dst=kc.ap(pattern=[[S, 128], [128 * S, 4], [1, T]], offset=kbase, scalar_offset=preg,
                                indirect_dim=2), src=qr[:, 8:12, :])
        ident = make_ident(128, bf16)
        vn = nl.ndarray((128, T // 128, 512), dtype=bf16, buffer=nl.sbuf)
        for tb in range(T // 128):
            for vt in range(4):
                pst = nl.ndarray((128, 128), dtype=fp32, buffer=nl.psum)
                nisa.nc_matmul(dst=pst, stationary=qkv[:, 12 + vt, tb * 128:(tb + 1) * 128], moving=ident)
                nisa.tensor_copy(dst=vn[:, tb, vt * 128:(vt + 1) * 128], src=pst)
        nisa.dma_copy(dst=vc.ap(pattern=[[1024, 128], [128 * 1024, T // 128], [1, 512]], offset=512 * pid,
                                scalar_offset=preg, indirect_dim=0), src=vn)

    attn = nl.ndarray((128, 8, T), dtype=bf16, buffer=nl.sbuf)
    if small:
        attention_decode(qr, kc, vc, pid, sreg, thr, B, local, attn)
    else:
        attention_prefill(qr, kc, vc, pid, sreg, thr, T, B, local, ident, attn)

    part = nl.ndarray((128, KT, T), dtype=fp32, buffer=nl.sbuf)
    gemm(wo[pid], attn, KT, 8, T, part, KT if small else 1, WSL, ring)
    tot = allreduce2(part, KT, T, pid, small)
    ob = nl.ndarray((128, KT, T), dtype=bf16, buffer=nl.sbuf)
    nisa.tensor_copy(dst=ob, src=tot)
    on = rmsnorm(ob, w1, 1, T, ones)
    h1 = nl.ndarray((128, KT, T), dtype=bf16, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=h1, data1=x, data2=on, op=nl.add)

    # ---- MLP block
    xn2 = rmsnorm(h1, w1, 2, T, ones)
    gu = nl.ndarray((128, 2 * ICT, T), dtype=bf16, buffer=nl.sbuf)
    gemm(wgu[pid], xn2, 2 * ICT, KT, T, gu, 2 * ICT if small else 1, WSL, ring)
    act = nl.ndarray((128, ICT, T), dtype=bf16, buffer=nl.sbuf)
    for nt in range(ICT):
        gl_ = nl.ndarray((128, T), dtype=bf16, buffer=nl.sbuf)
        nisa.activation(dst=gl_, op=nl.gelu_apprx_tanh, data=gu[:, 2 * nt, :])
        nisa.tensor_tensor(dst=act[:, nt, :], data1=gl_, data2=gu[:, 2 * nt + 1, :], op=nl.multiply)
    part2 = nl.ndarray((128, KT, T), dtype=fp32, buffer=nl.sbuf)
    gemm(wd[pid], act, KT, ICT, T, part2, KT if small else 1, WSL, ring)
    tot2 = allreduce2(part2, KT, T, pid, small)
    db = nl.ndarray((128, KT, T), dtype=bf16, buffer=nl.sbuf)
    nisa.tensor_copy(dst=db, src=tot2)
    dn = rmsnorm(db, w1, 3, T, ones)
    h2 = nl.ndarray((128, KT, T), dtype=bf16, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=h2, data1=h1, data2=dn, op=nl.add)

    out = nl.ndarray((D, T), dtype=bf16, buffer=nl.shared_hbm)
    half = KT // 2
    nisa.dma_copy(dst=out.ap(pattern=[[T, 128], [128 * T, half], [1, T]], offset=half * pid * 128 * T),
                  src=h2[:, half * pid:half * (pid + 1), :])
    if dbg:
        d1 = nl.ndarray((2, 128, 16, T), dtype=bf16, buffer=nl.shared_hbm)
        nisa.dma_copy(dst=d1[pid], src=qkv)
        d2 = nl.ndarray((2, 128, 12, T), dtype=bf16, buffer=nl.shared_hbm)
        nisa.dma_copy(dst=d2[pid], src=qr)
        d3 = nl.ndarray((2, 128, 8, T), dtype=bf16, buffer=nl.shared_hbm)
        nisa.dma_copy(dst=d3[pid], src=attn)
        d4 = nl.ndarray((2, 128, KT, T), dtype=bf16, buffer=nl.shared_hbm)
        nisa.dma_copy(dst=d4[pid], src=h1)
        return out, d1, d2, d3, d4
    return out


def attention_decode(qr, kc, vc, pid, sreg, thr, B, local, attn):
    """One query token, this core's 4 q heads (rows) and 2 KV heads. qr [128, 12, 1]: q tiles 0..7 (hl*2+dt)."""
    R = 4
    nch = (B + 511) // 512
    nb = B // 128
    # q per KV head with the other heads' columns zero, so both KV heads accumulate into one score tile
    qm = nl.ndarray((128, 2, 2, R), dtype=bf16, buffer=nl.sbuf)
    nisa.memset(dst=qm, value=0.0)
    for gl in range(2):
        for dt in range(2):
            for j in range(2):
                hl = 2 * gl + j
                nisa.tensor_copy(dst=qm[:, gl, dt, hl:hl + 1], src=qr[:, hl * 2 + dt, :])
    bias = nl.ndarray((R, B), dtype=bf16, buffer=nl.sbuf)
    window_bias(R, B, thr, 0, local, bias)
    sc = nl.ndarray((R, B), dtype=fp32, buffer=nl.sbuf)
    for ch in range(nch):
        w = min(512, B - ch * 512)
        ps = nl.ndarray((R, 512), dtype=fp32, buffer=nl.psum)
        i = 0
        for gl in range(2):
            g = 2 * pid + gl
            for dt in range(2):
                ksb = nl.ndarray((128, 512), dtype=bf16, buffer=nl.sbuf)
                adma(dst=ksb[:, 0:w], src=kc.ap(pattern=[[S, 128], [1, w]], offset=(g * HD + dt * 128) * S + ch * 512,
                                                         scalar_offset=sreg, indirect_dim=2))
                nisa.nc_matmul(dst=ps[:, 0:w], stationary=qm[:, gl, dt, :], moving=ksb[:, 0:w], accumulate=(i > 0))
                i += 1
        nisa.tensor_tensor(dst=sc[:, ch * 512:ch * 512 + w], data1=ps[:, 0:w], data2=bias[:, ch * 512:ch * 512 + w],
                           op=nl.add)
    p, sm = softmax_rows(sc, R, B)
    ident4 = make_ident(R, bf16)
    pt = nl.ndarray((128, nb, R), dtype=bf16, buffer=nl.sbuf)
    for b0 in range(0, nb, 64):
        n = min(64, nb - b0)
        pst = nl.ndarray((128, 64, R), dtype=fp32, buffer=nl.psum)
        for b in range(n):
            kb = b0 + b
            nisa.nc_matmul(dst=pst[:, b, :], stationary=p[:, kb * 128:(kb + 1) * 128], moving=ident4)
        nisa.tensor_copy(dst=pt[:, b0:b0 + n, :], src=pst[:, 0:n, :])
    po = nl.ndarray((R, 512), dtype=fp32, buffer=nl.psum)
    for ch in range(nch):
        nbc = min(4, nb - ch * 4)
        vsb = nl.ndarray((128, 4, 512), dtype=bf16, buffer=nl.sbuf)
        adma(dst=vsb[:, 0:nbc, :], src=vc.ap(pattern=[[1024, 128], [128 * 1024, nbc], [1, 512]],
                                                      offset=ch * 512 * 1024 + 512 * pid, scalar_offset=sreg, indirect_dim=0))
        for b in range(nbc):
            kb = ch * 4 + b
            nisa.nc_matmul(dst=po, stationary=pt[:, kb, :], moving=vsb[:, b, :], accumulate=(kb > 0))
    # rows 0,1 use KV head 0 (cols 0..255), rows 2,3 KV head 1 (cols 256..511)
    io = nl.ndarray((R, 1), dtype=fp32, buffer=nl.sbuf)
    nisa.iota(dst=io, pattern=[[0, 1]], offset=0, channel_multiplier=1)
    m1 = nl.ndarray((R, 1), dtype=fp32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=m1, data=io, op0=nl.greater_equal, operand0=2.0)
    m0 = nl.ndarray((R, 1), dtype=fp32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=m0, data=io, op0=nl.less, operand0=2.0)
    t0 = nl.ndarray((R, HD), dtype=fp32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=t0, data=po[:, 0:HD], op0=nl.multiply, operand0=m0)
    t1a = nl.ndarray((R, HD), dtype=fp32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=t1a, data=po[:, HD:2 * HD], op0=nl.multiply, operand0=m1)
    t1 = nl.ndarray((R, HD), dtype=fp32, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=t1, data1=t1a, data2=t0, op=nl.add)
    rs = nl.ndarray((R, 1), dtype=fp32, buffer=nl.sbuf)
    nisa.reciprocal(dst=rs, data=sm)
    o = nl.ndarray((R, HD), dtype=bf16, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=o, data=t1, op0=nl.multiply, operand0=rs)
    for dt in range(2):
        pso = nl.ndarray((128, R), dtype=fp32, buffer=nl.psum)
        nisa.nc_matmul(dst=pso, stationary=o[:, dt * 128:(dt + 1) * 128], moving=ident4)
        nisa.tensor_copy(dst=attn[:, dt * 4:(dt + 1) * 4, 0], src=pso)


def attention_prefill(qr, kc, vc, pid, sreg, thr, T, B, local, ident, attn):
    """T query tokens (128 per q tile), this core's 4 q heads, keys [start, start + B)."""
    nch = (B + 511) // 512
    nb = B // 128
    nqt = T // 128
    biases = []
    for qt in range(nqt):
        thq = nl.ndarray((128, 1), dtype=fp32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=thq, data=thr, op0=nl.add, operand0=float(qt * 128))
        bq = nl.ndarray((128, B), dtype=bf16, buffer=nl.sbuf)
        window_bias(128, B, thq, 1, local, bq)
        biases.append(bq)
    for hl in range(4):
        gl = hl // 2
        g = 2 * pid + gl
        for qt in range(nqt):
            sc = nl.ndarray((128, B), dtype=fp32, buffer=nl.sbuf)
            for ch in range(nch):
                w = min(512, B - ch * 512)
                ps = nl.ndarray((128, 512), dtype=fp32, buffer=nl.psum)
                for dt in range(2):
                    ksb = nl.ndarray((128, 512), dtype=bf16, buffer=nl.sbuf)
                    nisa.dma_copy(dst=ksb[:, 0:w], src=kc.ap(pattern=[[S, 128], [1, w]],
                                                             offset=(g * HD + dt * 128) * S + ch * 512,
                                                             scalar_offset=sreg, indirect_dim=2))
                    nisa.nc_matmul(dst=ps[:, 0:w], stationary=qr[:, hl * 2 + dt, qt * 128:(qt + 1) * 128],
                                   moving=ksb[:, 0:w], accumulate=(dt > 0))
                nisa.tensor_tensor(dst=sc[:, ch * 512:ch * 512 + w], data1=ps[:, 0:w],
                                   data2=biases[qt][:, ch * 512:ch * 512 + w], op=nl.add)
            p, sm = softmax_rows(sc, 128, B)
            pt = nl.ndarray((128, nb, 128), dtype=bf16, buffer=nl.sbuf)
            for b0 in range(0, nb, 4):
                n = min(4, nb - b0)
                pst = nl.ndarray((128, 4, 128), dtype=fp32, buffer=nl.psum)
                for b in range(n):
                    kb = b0 + b
                    nisa.nc_matmul(dst=pst[:, b, :], stationary=p[:, kb * 128:(kb + 1) * 128], moving=ident)
                nisa.tensor_copy(dst=pt[:, b0:b0 + n, :], src=pst[:, 0:n, :])
            po = nl.ndarray((128, HD), dtype=fp32, buffer=nl.psum)
            for ch in range(nch):
                nbc = min(4, nb - ch * 4)
                vsb = nl.ndarray((128, 4, HD), dtype=bf16, buffer=nl.sbuf)
                nisa.dma_copy(dst=vsb[:, 0:nbc, :], src=vc.ap(pattern=[[1024, 128], [128 * 1024, nbc], [1, HD]],
                                                              offset=ch * 512 * 1024 + g * HD, scalar_offset=sreg,
                                                              indirect_dim=0))
                for b in range(nbc):
                    kb = ch * 4 + b
                    nisa.nc_matmul(dst=po, stationary=pt[:, kb, :], moving=vsb[:, b, :], accumulate=(kb > 0))
            rs = nl.ndarray((128, 1), dtype=fp32, buffer=nl.sbuf)
            nisa.reciprocal(dst=rs, data=sm)
            o = nl.ndarray((128, HD), dtype=bf16, buffer=nl.sbuf)
            nisa.tensor_scalar(dst=o, data=po, op0=nl.multiply, operand0=rs)
            for dt in range(2):
                pso = nl.ndarray((128, 128), dtype=fp32, buffer=nl.psum)
                nisa.nc_matmul(dst=pso, stationary=o[:, dt * 128:(dt + 1) * 128], moving=ident)
                nisa.tensor_copy(dst=attn[:, dt * 4 + hl, qt * 128:(qt + 1) * 128], src=pso)


# ----------------------------------------------------------------------------------------------- embed / head

def embed_dec_body(toks, pos, emb):
    pid = nl.program_id(0)
    p_i = nl.ndarray((1, 1), dtype=i32, buffer=nl.sbuf)
    nisa.dma_copy(dst=p_i, src=pos)
    preg = nisa.register_alloc()
    nisa.register_load(preg, p_i)
    t_i = nl.ndarray((1, 1), dtype=i32, buffer=nl.sbuf)
    nisa.dma_copy(dst=t_i, src=toks.ap(pattern=[[1, 1], [1, 1]], offset=0, scalar_offset=preg, indirect_dim=0))
    treg = nisa.register_alloc()
    nisa.register_load(treg, t_i)
    e = nl.ndarray((128, KT), dtype=bf16, buffer=nl.sbuf)
    nisa.dma_copy(dst=e, src=emb.ap(pattern=[[1, 128], [128, KT]], offset=0, scalar_offset=treg, indirect_dim=0))
    y = nl.ndarray((128, KT), dtype=bf16, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=y, data=e, op0=nl.multiply, operand0=50.5)
    out = nl.ndarray((D, 1), dtype=bf16, buffer=nl.shared_hbm)
    if pid == 0:
        nisa.dma_copy(dst=out.ap(pattern=[[1, 128], [128, KT]], offset=0), src=y)
    return out


def embed_pre_body(toks, pos, emb, T):
    """Tokens toks[pos .. pos + T) -> h [D, T]; core c embeds tokens 128c .. 128c + 127 of the chunk."""
    pid = nl.program_id(0)
    p_i = nl.ndarray((1, 1), dtype=i32, buffer=nl.sbuf)
    nisa.dma_copy(dst=p_i, src=pos)
    preg = nisa.register_alloc()
    nisa.register_load(preg, p_i)
    ident = make_ident(128, bf16)
    out = nl.ndarray((D, T), dtype=bf16, buffer=nl.shared_hbm)
    for tb in range(T // 256):
        t0 = (2 * tb + pid) * 128
        idx = nl.ndarray((128, 1), dtype=i32, buffer=nl.sbuf)
        nisa.dma_copy(dst=idx, src=toks.ap(pattern=[[1, 128], [1, 1]], offset=t0, scalar_offset=preg, indirect_dim=0))
        rows = nl.ndarray((128, D), dtype=bf16, buffer=nl.sbuf)
        nisa.dma_copy(dst=rows, src=emb.ap(pattern=[[D, 128], [1, D]], offset=0, vector_offset=idx, indirect_dim=0))
        y = nl.ndarray((128, D), dtype=bf16, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=y, data=rows, op0=nl.multiply, operand0=50.5)
        hT = nl.ndarray((128, KT, 128), dtype=bf16, buffer=nl.sbuf)
        for kt in range(KT):
            pst = nl.ndarray((128, 128), dtype=fp32, buffer=nl.psum)
            nisa.nc_matmul(dst=pst, stationary=y[:, kt * 128:(kt + 1) * 128], moving=ident)
            nisa.tensor_copy(dst=hT[:, kt, :], src=pst)
        nisa.dma_copy(dst=out.ap(pattern=[[T, 128], [128 * T, KT], [1, 128]], offset=t0), src=hT)
    return out


def head_body(h, col, pos, pnext, lnf, embT, lmask, lids, toks, T):
    """Final norm of h[:, col], LM head over this core's vocabulary half, argmax (ties: lowest id) across both
    cores; writes toks[pos + 1] and pnext = pos + 1."""
    pid = nl.program_id(0)
    ones = make_ones()
    c_i = nl.ndarray((1, 1), dtype=i32, buffer=nl.sbuf)
    nisa.dma_copy(dst=c_i, src=col)
    creg = nisa.register_alloc()
    nisa.register_load(creg, c_i)
    p_i = nl.ndarray((1, 1), dtype=i32, buffer=nl.sbuf)
    nisa.dma_copy(dst=p_i, src=pos)
    preg = nisa.register_alloc()
    nisa.register_load(preg, p_i)

    x = nl.ndarray((128, KT, 1), dtype=bf16, buffer=nl.sbuf)
    nisa.dma_copy(dst=x, src=h.ap(pattern=[[T, 128], [128 * T, KT], [1, 1]], offset=0, scalar_offset=creg, indirect_dim=1))
    w1 = load_norm_w(lnf, 1)
    xn = rmsnorm(x, w1, 0, 1, ones)
    lgb = nl.ndarray((128, VT, 1), dtype=bf16, buffer=nl.sbuf)
    gemm(embT[pid], xn, VT, KT, 1, lgb, 128, 32)
    mk = nl.ndarray((128, VT), dtype=fp32, buffer=nl.sbuf)
    nisa.dma_copy(dst=mk, src=lmask[pid])
    ids = nl.ndarray((128, VT), dtype=fp32, buffer=nl.sbuf)
    nisa.dma_copy(dst=ids, src=lids[pid])
    lg = nl.ndarray((128, VT), dtype=fp32, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=lg, data1=lgb[:, :, 0], data2=mk, op=nl.add)
    mp = nl.ndarray((128, 1), dtype=fp32, buffer=nl.sbuf)
    nisa.tensor_reduce(dst=mp, op=nl.maximum, data=lg, axis=1)
    eq = nl.ndarray((128, VT), dtype=fp32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=eq, data=lg, op0=nl.equal, operand0=mp)
    cand = nl.ndarray((128, VT), dtype=fp32, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=cand, data1=eq, data2=ids, op=nl.multiply)
    cp = nl.ndarray((128, 1), dtype=fp32, buffer=nl.sbuf)
    nisa.tensor_reduce(dst=cp, op=nl.maximum, data=cand, axis=1)
    # partitions -> one row
    mrp = nl.ndarray((1, 128), dtype=fp32, buffer=nl.psum)
    nisa.nc_transpose(dst=mrp, data=mp)
    mr = nl.ndarray((1, 128), dtype=fp32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=mr, src=mrp)
    crp = nl.ndarray((1, 128), dtype=fp32, buffer=nl.psum)
    nisa.nc_transpose(dst=crp, data=cp)
    cr = nl.ndarray((1, 128), dtype=fp32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=cr, src=crp)
    mine = nl.ndarray((1, 2), dtype=fp32, buffer=nl.sbuf)
    nisa.tensor_reduce(dst=mine[:, 0:1], op=nl.maximum, data=mr, axis=1)
    eqr = nl.ndarray((1, 128), dtype=fp32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=eqr, data=mr, op0=nl.equal, operand0=mine[:, 0:1])
    crr = nl.ndarray((1, 128), dtype=fp32, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=crr, data1=eqr, data2=cr, op=nl.multiply)
    best = nl.ndarray((1, 1), dtype=fp32, buffer=nl.sbuf)
    nisa.tensor_reduce(dst=best, op=nl.maximum, data=crr, axis=1)
    nisa.tensor_scalar(dst=mine[:, 1:2], data=best, op0=nl.subtract, operand0=BIGID, reverse0=True)  # id = BIG - best
    oth = nl.ndarray((1, 2), dtype=fp32, buffer=nl.sbuf)
    nisa.sendrecv(src=mine, dst=oth, send_to_rank=1 - pid, recv_from_rank=1 - pid, pipe_id=0,
                  dma_engine=nisa.dma_engine.gpsimd_dma)
    # symmetric combine: larger value wins; equal values -> lower id
    gt = nl.ndarray((1, 1), dtype=fp32, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=gt, data1=mine[:, 0:1], data2=oth[:, 0:1], op=nl.greater)
    lt = nl.ndarray((1, 1), dtype=fp32, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=lt, data1=oth[:, 0:1], data2=mine[:, 0:1], op=nl.greater)
    mn = nl.ndarray((1, 1), dtype=fp32, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=mn, data1=mine[:, 1:2], data2=oth[:, 1:2], op=nl.minimum)
    d1 = nl.ndarray((1, 1), dtype=fp32, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=d1, data1=mine[:, 1:2], data2=mn, op=nl.subtract)
    d2 = nl.ndarray((1, 1), dtype=fp32, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=d2, data1=oth[:, 1:2], data2=mn, op=nl.subtract)
    e1 = nl.ndarray((1, 1), dtype=fp32, buffer=nl.sbuf)
    nisa.scalar_tensor_tensor(dst=e1, data=d1, op0=nl.multiply, operand0=gt, op1=nl.add, operand1=mn)
    fin = nl.ndarray((1, 1), dtype=fp32, buffer=nl.sbuf)
    nisa.scalar_tensor_tensor(dst=fin, data=d2, op0=nl.multiply, operand0=lt, op1=nl.add, operand1=e1)
    tok = nl.ndarray((1, 1), dtype=i32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=tok, src=fin)
    pn = nl.ndarray((1, 1), dtype=i32, buffer=nl.sbuf)
    p_f = nl.ndarray((1, 1), dtype=fp32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=p_f, src=p_i)
    p_f1 = nl.ndarray((1, 1), dtype=fp32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=p_f1, data=p_f, op0=nl.add, operand0=1.0)
    nisa.tensor_copy(dst=pn, src=p_f1)
    out = nl.ndarray((1, 2), dtype=i32, buffer=nl.shared_hbm)
    if pid == 0:
        nisa.dma_copy(dst=toks.ap(pattern=[[1, 1], [1, 1]], offset=1, scalar_offset=preg, indirect_dim=0), src=tok)
        nisa.dma_copy(dst=pnext, src=pn)
        nisa.dma_copy(dst=out[:, 0:1], src=tok)
        nisa.dma_copy(dst=out[:, 1:2], src=pn)
    return out


# ----------------------------------------------------------------------------------------------- decode groups
# One NEFF runs several decoder layers of a decode step (the first group also the embedding, the last one the LM
# head and argmax). The hidden state stays in SBUF between layers ([128, KT, 1] column layout), every weight streams
# through one SBUF ring on the Sync engine's hardware DGE queue, and the K/V window of each layer is loaded at the
# top of the layer from the cache *without* the current position (masked; its K/V come from SBUF), so nothing on the
# Sync queue waits for compute and the weight stream runs ahead across GEMMs and layers.

def bc(t, pstride, pattern, offset=0):
    """Access-pattern view of an SBUF/PSUM tensor: partition stride pstride, free pattern [[stride, n], ...]."""
    return t.ap(pattern=[[pstride, 128]] + pattern, offset=offset)


def rmsnorm1(x, w1row, onesr):
    """Gemma RMSNorm of one column x [128, KT] bf16 (view); w1row [128, KT] fp32 = (1 + w). Returns [128, KT, 1]
    bf16. Sum of squares as x^T x per k-tile on the TensorE (bf16 products exact, fp32 accumulation), statistics on
    one partition, broadcast to all partitions by a K=1 fp32 matmul with a row of ones (exact)."""
    ss = nl.ndarray((1, 1), dtype=fp32, buffer=nl.psum)
    for kt in range(KT):
        nisa.nc_matmul(dst=ss, stationary=x[:, kt:kt + 1], moving=x[:, kt:kt + 1], accumulate=(kt > 0))
    r1 = nl.ndarray((1, 1), dtype=fp32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=r1, data=ss, op0=nl.multiply, operand0=1.0 / D, op1=nl.add, operand1=EPS)
    r2 = nl.ndarray((1, 1), dtype=fp32, buffer=nl.sbuf)
    nisa.activation(dst=r2, op=nl.sqrt, data=r1)
    rs = nl.ndarray((1, 1), dtype=fp32, buffer=nl.sbuf)
    nisa.reciprocal(dst=rs, data=r2)
    pb = nl.ndarray((128, 1), dtype=fp32, buffer=nl.psum)
    nisa.nc_matmul(dst=pb, stationary=onesr, moving=rs)
    rstd = nl.ndarray((128, 1), dtype=fp32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=rstd, src=pb)
    tmp = nl.ndarray((128, KT), dtype=fp32, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=tmp, data1=x, data2=bc(rstd, 1, [[0, KT]]), op=nl.multiply)
    y = nl.ndarray((128, KT, 1), dtype=bf16, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=y[:, :, 0], data1=tmp, data2=w1row, op=nl.multiply)
    return y


def window_bias2(R, B, thr, local, dst):
    """dst [R, B] bf16 = 0 where window index j is a visible *cached* key, NEG elsewhere: j < thr (the current
    position thr itself comes from SBUF) and, for local layers, j >= thr - 1023.  thr: [R, 1] fp32."""
    io = nl.ndarray((R, B), dtype=fp32, buffer=nl.sbuf)
    nisa.iota(dst=io, pattern=[[1, B]], offset=0, channel_multiplier=0)
    m = nl.ndarray((R, B), dtype=fp32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=m, data=io, op0=nl.subtract, operand0=thr, op1=nl.greater, operand1=-0.5)
    if local:
        m2 = nl.ndarray((R, B), dtype=fp32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=m2, data=io, op0=nl.subtract, operand0=thr, op1=nl.less, operand1=-1023.5)
        nisa.tensor_tensor(dst=m, data1=m, data2=m2, op=nl.add)
    nisa.tensor_scalar(dst=dst, data=m, op0=nl.multiply, operand0=NEG)


def dec_window(p_f, B):
    """start = max(0, pos + 1 - B) as a register, thr = pos - start [4, 1] fp32 (p_f: pos [4, 1] fp32)."""
    st_f = nl.ndarray((4, 1), dtype=fp32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=st_f, data=p_f, op0=nl.add, operand0=float(1 - B), op1=nl.maximum, operand1=0.0)
    thr = nl.ndarray((4, 1), dtype=fp32, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=thr, data1=p_f, data2=st_f, op=nl.subtract)
    st_i = nl.ndarray((4, 1), dtype=i32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=st_i, src=st_f)
    sreg = nisa.register_alloc()
    nisa.register_load(sreg, st_i[0:1, 0:1])
    return sreg, thr


def dec_layer(x, wqkv, wo, wgu, wd, lnw, qkw, kc, vc, cs, sn, preg, sreg, bias, B, pid, ring, ones, ident, ident4,
              sel, kvbuf, onesr):
    """One decoder layer for one token. x [128, KT, 1] bf16 (SBUF). Returns the layer output [128, KT, 1] bf16."""
    R = 4
    nb = B // 128
    nch = (B + 511) // 512
    # ---- K/V window of the cache; no dependency on this step's compute. kvbuf: persistent buffers (so that the
    # compiler does not hand their space to other tensors, which would order the next layer's load behind them)
    if kvbuf is None:
        kw = nl.ndarray((128, 4, B), dtype=bf16, buffer=nl.sbuf)
        vw = nl.ndarray((128, nb, 512), dtype=bf16, buffer=nl.sbuf)
    else:
        kw = kvbuf[0]
        vw = kvbuf[1]
    hdma(dst=kw, src=kc.ap(pattern=[[S, 128], [128 * S, 4], [1, B]], offset=(2 * pid) * HD * S, scalar_offset=sreg,
                           indirect_dim=2))
    hdma(dst=vw, src=vc.ap(pattern=[[1024, 128], [128 * 1024, nb], [1, 512]], offset=512 * pid, scalar_offset=sreg,
                           indirect_dim=0))
    # ---- small per-layer weights
    wsb = nl.ndarray((128, 4, KT), dtype=bf16, buffer=nl.sbuf)
    nisa.dma_copy(dst=wsb, src=lnw.ap(pattern=[[4 * KT, 128], [KT, 4], [1, KT]], offset=0))
    w1 = nl.ndarray((128, 4, KT), dtype=fp32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=w1, data=wsb, op0=nl.add, operand0=1.0)
    qk_sb = nl.ndarray((128, 2, 2), dtype=bf16, buffer=nl.sbuf)
    nisa.dma_copy(dst=qk_sb, src=qkw.ap(pattern=[[4, 128], [2, 2], [1, 2]], offset=0))
    qk1 = nl.ndarray((128, 2, 2), dtype=fp32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=qk1, data=qk_sb, op0=nl.add, operand0=1.0)
    w12 = nl.ndarray((128, 12), dtype=fp32, buffer=nl.sbuf)       # (1 + w) of q/k tile 2h + dt
    nisa.tensor_copy(dst=bc(w12, 12, [[2, 4], [1, 2]]), src=bc(qk1, 4, [[0, 4], [1, 2]]))
    nisa.tensor_copy(dst=bc(w12, 12, [[2, 2], [1, 2]], 8), src=bc(qk1, 4, [[0, 2], [1, 2]], 2))

    # ---- attention block
    xn = rmsnorm1(x[:, :, 0], w1[:, 0, :], onesr)
    qkv = nl.ndarray((128, 16, 1), dtype=bf16, buffer=nl.sbuf)
    gemm(wqkv[pid], xn, 16, KT, 1, qkv, 16, 16, ring)
    # q/k norms, 6 heads at once: per head q^T q over its 2 tiles on the TensorE, statistics on one partition,
    # broadcast back with a K=1 ones matmul
    ssp = nl.ndarray((1, 6), dtype=fp32, buffer=nl.psum)
    for h in range(6):
        for dt in range(2):
            t = 2 * h + dt
            nisa.nc_matmul(dst=ssp[:, h:h + 1], stationary=qkv[:, t, :], moving=qkv[:, t, :])
    r1 = nl.ndarray((1, 6), dtype=fp32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=r1, data=ssp, op0=nl.multiply, operand0=1.0 / HD, op1=nl.add, operand1=EPS)
    r2 = nl.ndarray((1, 6), dtype=fp32, buffer=nl.sbuf)
    nisa.activation(dst=r2, op=nl.sqrt, data=r1)
    rs6 = nl.ndarray((1, 6), dtype=fp32, buffer=nl.sbuf)
    nisa.reciprocal(dst=rs6, data=r2)
    pb6 = nl.ndarray((128, 6), dtype=fp32, buffer=nl.psum)
    nisa.nc_matmul(dst=pb6, stationary=onesr, moving=rs6)
    rstd = nl.ndarray((128, 6), dtype=fp32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=rstd, src=pb6)
    tq = nl.ndarray((128, 12), dtype=fp32, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=bc(tq, 12, [[2, 6], [1, 2]]), data1=bc(qkv, 16, [[2, 6], [1, 2]]),
                       data2=bc(rstd, 6, [[1, 6], [0, 2]]), op=nl.multiply)
    qn = nl.ndarray((128, 12), dtype=bf16, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=qn, data1=tq, data2=w12, op=nl.multiply)
    # RoPE, 6 heads at once: lo = even tiles, hi = odd tiles (HF's bf16 roundings, as in rope)
    lo = bc(qn, 12, [[2, 6]], 0)
    hi = bc(qn, 12, [[2, 6]], 1)
    c6 = bc(cs, 1, [[0, 6]])
    s6 = bc(sn, 1, [[0, 6]])
    ra = nl.ndarray((128, 6), dtype=bf16, buffer=nl.sbuf)
    rb = nl.ndarray((128, 6), dtype=bf16, buffer=nl.sbuf)
    rc = nl.ndarray((128, 6), dtype=bf16, buffer=nl.sbuf)
    rd = nl.ndarray((128, 6), dtype=bf16, buffer=nl.sbuf)
    qr = nl.ndarray((128, 12), dtype=bf16, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=ra, data1=lo, data2=c6, op=nl.multiply)
    nisa.tensor_tensor(dst=rb, data1=hi, data2=s6, op=nl.multiply)
    nisa.tensor_tensor(dst=bc(qr, 12, [[2, 6]], 0), data1=ra, data2=rb, op=nl.subtract)
    nisa.tensor_tensor(dst=rc, data1=hi, data2=c6, op=nl.multiply)
    nisa.tensor_tensor(dst=rd, data1=lo, data2=s6, op=nl.multiply)
    nisa.tensor_tensor(dst=bc(qr, 12, [[2, 6]], 1), data1=rc, data2=rd, op=nl.add)
    # cache writes at pos (this core's KV heads 2c, 2c+1); GpSimd queue, since they wait for compute
    adma(dst=kc.ap(pattern=[[S, 128], [128 * S, 4]], offset=(2 * pid) * HD * S, scalar_offset=preg,
                   indirect_dim=2), src=qr[:, 8:12])

    # ---- attention (rows = this core's 4 q heads; KV head gl serves rows 2gl, 2gl+1)
    qm = nl.ndarray((128, 2, 2, R), dtype=bf16, buffer=nl.sbuf)
    nisa.memset(dst=qm, value=0.0)
    for gl in range(2):
        # qm[:, gl, dt, 2gl + j] = q tile (2gl + j)*2 + dt
        nisa.tensor_copy(dst=bc(qm, 16, [[R, 2], [1, 2]], gl * 2 * R + 2 * gl), src=bc(qr, 12, [[1, 2], [2, 2]], 4 * gl))
    sc = nl.ndarray((R, B), dtype=fp32, buffer=nl.sbuf)
    for ch in range(nch):
        w = min(512, B - ch * 512)
        ps = nl.ndarray((R, 512), dtype=fp32, buffer=nl.psum)
        i = 0
        for gl in range(2):
            for dt in range(2):
                nisa.nc_matmul(dst=ps[:, 0:w], stationary=qm[:, gl, dt, :], moving=kw[:, 2 * gl + dt, ch * 512:ch * 512 + w],
                               accumulate=(i > 0))
                i += 1
        nisa.tensor_tensor(dst=sc[:, ch * 512:ch * 512 + w], data1=ps[:, 0:w], data2=bias[:, ch * 512:ch * 512 + w],
                           op=nl.add)
    psc = nl.ndarray((R, 1), dtype=fp32, buffer=nl.psum)
    i = 0
    for gl in range(2):
        for dt in range(2):
            t = 8 + 2 * gl + dt
            nisa.nc_matmul(dst=psc, stationary=qm[:, gl, dt, :], moving=qr[:, t:t + 1], accumulate=(i > 0))
            i += 1
    scur = nl.ndarray((R, 1), dtype=fp32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=scur, src=psc)
    mxw = nl.ndarray((R, 1), dtype=fp32, buffer=nl.sbuf)
    nisa.tensor_reduce(dst=mxw, op=nl.maximum, data=sc, axis=1)
    mx = nl.ndarray((R, 1), dtype=fp32, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=mx, data1=mxw, data2=scur, op=nl.maximum)
    nmx = nl.ndarray((R, 1), dtype=fp32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=nmx, data=mx, op0=nl.multiply, operand0=-0.0625)
    p = nl.ndarray((R, B), dtype=bf16, buffer=nl.sbuf)
    sm = nl.ndarray((R, 1), dtype=fp32, buffer=nl.sbuf)
    nisa.activation(dst=p, op=nl.exp, data=sc, bias=nmx, scale=0.0625, reduce_op=nl.add, reduce_res=sm,
                    reduce_cmd=nisa.reduce_cmd.reset_reduce)
    pc = nl.ndarray((R, 1), dtype=bf16, buffer=nl.sbuf)
    smc = nl.ndarray((R, 1), dtype=fp32, buffer=nl.sbuf)
    nisa.activation(dst=pc, op=nl.exp, data=scur, bias=nmx, scale=0.0625, reduce_op=nl.add, reduce_res=smc,
                    reduce_cmd=nisa.reduce_cmd.reset_reduce)
    pt = nl.ndarray((128, nb, R), dtype=bf16, buffer=nl.sbuf)
    for b0 in range(0, nb, 64):
        n = min(64, nb - b0)
        pst = nl.ndarray((128, 64, R), dtype=fp32, buffer=nl.psum)
        for b in range(n):
            kb = b0 + b
            nisa.nc_matmul(dst=pst[:, b, :], stationary=p[:, kb * 128:(kb + 1) * 128], moving=ident4)
        nisa.tensor_copy(dst=pt[:, b0:b0 + n, :], src=pst[:, 0:n, :])
    # the current position as one more 128-row block with a single non-zero row: P row pc^T, V row v
    pcb = nl.ndarray((128, R), dtype=bf16, buffer=nl.sbuf)
    nisa.memset(dst=pcb, value=0.0)
    vcb = nl.ndarray((128, 512), dtype=bf16, buffer=nl.sbuf)
    nisa.memset(dst=vcb, value=0.0)
    pp = nl.ndarray((1, R), dtype=fp32, buffer=nl.psum)
    nisa.nc_matmul(dst=pp, stationary=pc, moving=ident4)
    nisa.tensor_copy(dst=pcb[0:1, :], src=pp)
    pv = nl.ndarray((1, 512), dtype=fp32, buffer=nl.psum)
    for vt in range(4):
        nisa.nc_matmul(dst=pv[:, vt * 128:(vt + 1) * 128], stationary=qkv[:, 12 + vt, :], moving=ident)
    nisa.tensor_copy(dst=vcb[0:1, :], src=pv)
    # V cache row at pos from the token-major copy (one contiguous 1 KB write)
    nisa.dma_copy(dst=vc.ap(pattern=[[1024, 1], [1, 512]], offset=512 * pid, scalar_offset=preg, indirect_dim=0),
                  src=vcb[0:1, :])
    po = nl.ndarray((R, 512), dtype=fp32, buffer=nl.psum)
    for kb in range(nb):
        nisa.nc_matmul(dst=po, stationary=pt[:, kb, :], moving=vw[:, kb, :], accumulate=(kb > 0))
    nisa.nc_matmul(dst=po, stationary=pcb, moving=vcb, accumulate=True)
    # rows 0,1 use KV head 0 (cols 0..255), rows 2,3 KV head 1 (cols 256..511)
    t0 = nl.ndarray((R, HD), dtype=fp32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=t0, data=po[:, 0:HD], op0=nl.multiply, operand0=sel[:, 0:1])
    t1a = nl.ndarray((R, HD), dtype=fp32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=t1a, data=po[:, HD:2 * HD], op0=nl.multiply, operand0=sel[:, 1:2])
    t1 = nl.ndarray((R, HD), dtype=fp32, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=t1, data1=t1a, data2=t0, op=nl.add)
    tot = nl.ndarray((R, 1), dtype=fp32, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=tot, data1=sm, data2=smc, op=nl.add)
    rs = nl.ndarray((R, 1), dtype=fp32, buffer=nl.sbuf)
    nisa.reciprocal(dst=rs, data=tot)
    o = nl.ndarray((R, HD), dtype=bf16, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=o, data=t1, op0=nl.multiply, operand0=rs)
    attn = nl.ndarray((128, 8, 1), dtype=bf16, buffer=nl.sbuf)
    for dt in range(2):
        pso = nl.ndarray((128, R), dtype=fp32, buffer=nl.psum)
        nisa.nc_matmul(dst=pso, stationary=o[:, dt * 128:(dt + 1) * 128], moving=ident4)
        nisa.tensor_copy(dst=attn[:, dt * 4:(dt + 1) * 4, 0], src=pso)

    part = nl.ndarray((128, KT, 1), dtype=fp32, buffer=nl.sbuf)
    gemm(wo[pid], attn, KT, 8, 1, part, KT, 16, ring)
    tot1 = allreduce2(part, KT, 1, pid, True)
    ob = nl.ndarray((128, KT), dtype=bf16, buffer=nl.sbuf)
    nisa.tensor_copy(dst=ob, src=tot1[:, :, 0])
    on = rmsnorm1(ob, w1[:, 1, :], onesr)
    h1 = nl.ndarray((128, KT, 1), dtype=bf16, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=h1, data1=x, data2=on, op=nl.add)

    # ---- MLP block
    xn2 = rmsnorm1(h1[:, :, 0], w1[:, 2, :], onesr)
    gu = nl.ndarray((128, 2 * ICT, 1), dtype=bf16, buffer=nl.sbuf)
    gemm(wgu[pid], xn2, 2 * ICT, KT, 1, gu, 2 * ICT, 16, ring)
    gl_ = nl.ndarray((128, ICT), dtype=bf16, buffer=nl.sbuf)
    nisa.activation(dst=gl_, op=nl.gelu_apprx_tanh, data=bc(gu, 2 * ICT, [[2, ICT]], 0))
    act = nl.ndarray((128, ICT, 1), dtype=bf16, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=act[:, :, 0], data1=gl_, data2=bc(gu, 2 * ICT, [[2, ICT]], 1), op=nl.multiply)
    part2 = nl.ndarray((128, KT, 1), dtype=fp32, buffer=nl.sbuf)
    gemm(wd[pid], act, KT, ICT, 1, part2, KT, 16, ring)
    tot2 = allreduce2(part2, KT, 1, pid, True)
    db = nl.ndarray((128, KT), dtype=bf16, buffer=nl.sbuf)
    nisa.tensor_copy(dst=db, src=tot2[:, :, 0])
    dn = rmsnorm1(db, w1[:, 3, :], onesr)
    h2 = nl.ndarray((128, KT, 1), dtype=bf16, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=h2, data1=h1, data2=dn, op=nl.add)
    return h2


DEC_NKV = 1                   # local K/V window buffers (the load of layer l+1 only waits for layer l's attention)
DEC_SBUF_KB = 160
DEC_SL = 64                   # weight tiles per decode ring slot
DEC_KV_PERSIST_MAX = 4608     # global windows up to this size get a persistent SBUF buffer
DEC_MULTI_PERSIST_MAX = 1536  # ... in the multi-step NEFFs (dgA2x4608 with it: 30.3 ms per 2 steps, without: 25.5)


def dec_ring_slots(B):
    kv = DEC_NKV * 16 * DEC_LOCAL_B               # local K + V window buffers, bytes per partition
    if B <= DEC_KV_PERSIST_MAX:
        kv += 16 * B
    else:
        kv += 16 * 2048
    return max(6, min(28, (DEC_SBUF_KB * 1024 - kv) // 4096))


def argmax_tail(lgb, lmask, lids, toks, pnext, p_i, preg, pid, final=True, tok_out=None):
    """lgb [128, VT, 1] bf16 logits of this core's vocabulary half -> toks[pos + 1], pnext = pos + 1 (ties: lowest id)."""
    mk = nl.ndarray((128, VT), dtype=fp32, buffer=nl.sbuf)
    nisa.dma_copy(dst=mk, src=lmask[pid])
    ids = nl.ndarray((128, VT), dtype=fp32, buffer=nl.sbuf)
    nisa.dma_copy(dst=ids, src=lids[pid])
    lg = nl.ndarray((128, VT), dtype=fp32, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=lg, data1=lgb[:, :, 0], data2=mk, op=nl.add)
    mp = nl.ndarray((128, 1), dtype=fp32, buffer=nl.sbuf)
    nisa.tensor_reduce(dst=mp, op=nl.maximum, data=lg, axis=1)
    eq = nl.ndarray((128, VT), dtype=fp32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=eq, data=lg, op0=nl.equal, operand0=mp)
    cand = nl.ndarray((128, VT), dtype=fp32, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=cand, data1=eq, data2=ids, op=nl.multiply)
    cp = nl.ndarray((128, 1), dtype=fp32, buffer=nl.sbuf)
    nisa.tensor_reduce(dst=cp, op=nl.maximum, data=cand, axis=1)
    mrp = nl.ndarray((1, 128), dtype=fp32, buffer=nl.psum)
    nisa.nc_transpose(dst=mrp, data=mp)
    mr = nl.ndarray((1, 128), dtype=fp32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=mr, src=mrp)
    crp = nl.ndarray((1, 128), dtype=fp32, buffer=nl.psum)
    nisa.nc_transpose(dst=crp, data=cp)
    cr = nl.ndarray((1, 128), dtype=fp32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=cr, src=crp)
    mine = nl.ndarray((1, 2), dtype=fp32, buffer=nl.sbuf)
    nisa.tensor_reduce(dst=mine[:, 0:1], op=nl.maximum, data=mr, axis=1)
    eqr = nl.ndarray((1, 128), dtype=fp32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=eqr, data=mr, op0=nl.equal, operand0=mine[:, 0:1])
    crr = nl.ndarray((1, 128), dtype=fp32, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=crr, data1=eqr, data2=cr, op=nl.multiply)
    best = nl.ndarray((1, 1), dtype=fp32, buffer=nl.sbuf)
    nisa.tensor_reduce(dst=best, op=nl.maximum, data=crr, axis=1)
    nisa.tensor_scalar(dst=mine[:, 1:2], data=best, op0=nl.subtract, operand0=BIGID, reverse0=True)
    oth = nl.ndarray((1, 2), dtype=fp32, buffer=nl.sbuf)
    nisa.sendrecv(src=mine, dst=oth, send_to_rank=1 - pid, recv_from_rank=1 - pid, pipe_id=0,
                  dma_engine=nisa.dma_engine.gpsimd_dma)
    gt = nl.ndarray((1, 1), dtype=fp32, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=gt, data1=mine[:, 0:1], data2=oth[:, 0:1], op=nl.greater)
    lt = nl.ndarray((1, 1), dtype=fp32, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=lt, data1=oth[:, 0:1], data2=mine[:, 0:1], op=nl.greater)
    mn = nl.ndarray((1, 1), dtype=fp32, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=mn, data1=mine[:, 1:2], data2=oth[:, 1:2], op=nl.minimum)
    d1 = nl.ndarray((1, 1), dtype=fp32, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=d1, data1=mine[:, 1:2], data2=mn, op=nl.subtract)
    d2 = nl.ndarray((1, 1), dtype=fp32, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=d2, data1=oth[:, 1:2], data2=mn, op=nl.subtract)
    e1 = nl.ndarray((1, 1), dtype=fp32, buffer=nl.sbuf)
    nisa.scalar_tensor_tensor(dst=e1, data=d1, op0=nl.multiply, operand0=gt, op1=nl.add, operand1=mn)
    fin = nl.ndarray((1, 1), dtype=fp32, buffer=nl.sbuf)
    nisa.scalar_tensor_tensor(dst=fin, data=d2, op0=nl.multiply, operand0=lt, op1=nl.add, operand1=e1)
    tok = nl.ndarray((1, 1), dtype=i32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=tok, src=fin)
    if tok_out is not None:
        tok_out.append(tok)
    if not final:
        if pid == 0:
            nisa.dma_copy(dst=toks.ap(pattern=[[1, 1], [1, 1]], offset=1, scalar_offset=preg, indirect_dim=0), src=tok)
        return None
    pn = nl.ndarray((1, 1), dtype=i32, buffer=nl.sbuf)
    p_f = nl.ndarray((1, 1), dtype=fp32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=p_f, src=p_i[0:1, 0:1])
    p_f1 = nl.ndarray((1, 1), dtype=fp32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=p_f1, data=p_f, op0=nl.add, operand0=1.0)
    nisa.tensor_copy(dst=pn, src=p_f1)
    out = nl.ndarray((1, 2), dtype=i32, buffer=nl.shared_hbm)
    if pid == 0:
        nisa.dma_copy(dst=toks.ap(pattern=[[1, 1], [1, 1]], offset=1, scalar_offset=preg, indirect_dim=0), src=tok)
        nisa.dma_copy(dst=pnext, src=pn)
        nisa.dma_copy(dst=out[:, 0:1], src=tok)
        nisa.dma_copy(dst=out[:, 1:2], src=pn)
    return out


def dec_group_body(h, toks, emb, pos, pnext, cosl, sinl, cosg, sing, lws, kinds, B, first, last, lnf, embT, lmask,
                   lids, col=None, TH=1):
    """Decoder layers lws[i] (kinds[i] 'L' local or 'G' global with bucket B) for one token at position pos.
    first: the input is the embedding of toks[pos], else h [D, 1]. last: ends with the final norm, LM head and
    argmax (writes toks[pos + 1] and pnext), else returns h [D, 1]."""
    pid = nl.program_id(0)
    R = 4
    ones = make_ones()
    onesr = nl.ndarray((1, 128), dtype=fp32, buffer=nl.sbuf)
    nisa.memset(dst=onesr, value=1.0)
    ident = make_ident(128, bf16)
    ident4 = make_ident(R, bf16)
    nbuf = dec_ring_slots(B if 'G' in kinds else 0)
    # local window: every position below a global bucket B < 1024 fits in a window of B keys (start 0)
    LB = min(DEC_LOCAL_B, B) if ('G' in kinds and DEC_LOCAL_SMALL) else DEC_LOCAL_B
    ring = WRing(nbuf * 16 // DEC_SL, DEC_SL)
    kvL = []
    for k in range(DEC_NKV):
        kvL.append([nl.ndarray((128, 4, LB), dtype=bf16, buffer=nl.sbuf),
                    nl.ndarray((128, LB // 128, 512), dtype=bf16, buffer=nl.sbuf)])
    kvG = None
    if 'G' in kinds and B <= DEC_KV_PERSIST_MAX:
        kvG = [nl.ndarray((128, 4, B), dtype=bf16, buffer=nl.sbuf), nl.ndarray((128, B // 128, 512), dtype=bf16,
                                                                               buffer=nl.sbuf)]
    nloc = 0
    # scalars
    p_i = nl.ndarray((R, 1), dtype=i32, buffer=nl.sbuf)
    nisa.dma_copy(dst=p_i, src=pos.ap(pattern=[[0, R], [1, 1]], offset=0))
    p_f = nl.ndarray((R, 1), dtype=fp32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=p_f, src=p_i)
    preg = nisa.register_alloc()
    nisa.register_load(preg, p_i[0:1, 0:1])
    io = nl.ndarray((R, 1), dtype=fp32, buffer=nl.sbuf)
    nisa.iota(dst=io, pattern=[[0, 1]], offset=0, channel_multiplier=1)
    sel = nl.ndarray((R, 2), dtype=fp32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=sel[:, 0:1], data=io, op0=nl.less, operand0=2.0)
    nisa.tensor_scalar(dst=sel[:, 1:2], data=io, op0=nl.greater_equal, operand0=2.0)
    sregL = None
    sregG = None
    biasL = None
    biasG = None
    csL = None
    snL = None
    csG = None
    snG = None
    if 'L' in kinds:
        sregL, thrL = dec_window(p_f, LB)
        biasL = nl.ndarray((R, LB), dtype=bf16, buffer=nl.sbuf)
        window_bias2(R, LB, thrL, True, biasL)
        csL = nl.ndarray((128, 1), dtype=bf16, buffer=nl.sbuf)
        nisa.dma_copy(dst=csL, src=cosl.ap(pattern=[[S, 128], [1, 1]], offset=0, scalar_offset=preg, indirect_dim=1))
        snL = nl.ndarray((128, 1), dtype=bf16, buffer=nl.sbuf)
        nisa.dma_copy(dst=snL, src=sinl.ap(pattern=[[S, 128], [1, 1]], offset=0, scalar_offset=preg, indirect_dim=1))
    if 'G' in kinds:
        sregG, thrG = dec_window(p_f, B)
        biasG = nl.ndarray((R, B), dtype=bf16, buffer=nl.sbuf)
        window_bias2(R, B, thrG, False, biasG)
        csG = nl.ndarray((128, 1), dtype=bf16, buffer=nl.sbuf)
        nisa.dma_copy(dst=csG, src=cosg.ap(pattern=[[S, 128], [1, 1]], offset=0, scalar_offset=preg, indirect_dim=1))
        snG = nl.ndarray((128, 1), dtype=bf16, buffer=nl.sbuf)
        nisa.dma_copy(dst=snG, src=sing.ap(pattern=[[S, 128], [1, 1]], offset=0, scalar_offset=preg, indirect_dim=1))
    # input
    x = nl.ndarray((128, KT, 1), dtype=bf16, buffer=nl.sbuf)
    if first:
        t_i = nl.ndarray((1, 1), dtype=i32, buffer=nl.sbuf)
        nisa.dma_copy(dst=t_i, src=toks.ap(pattern=[[1, 1], [1, 1]], offset=0, scalar_offset=preg, indirect_dim=0))
        treg = nisa.register_alloc()
        nisa.register_load(treg, t_i)
        e = nl.ndarray((128, KT), dtype=bf16, buffer=nl.sbuf)
        nisa.dma_copy(dst=e, src=emb.ap(pattern=[[1, 128], [128, KT]], offset=0, scalar_offset=treg, indirect_dim=0))
        nisa.tensor_scalar(dst=x[:, :, 0], data=e, op0=nl.multiply, operand0=50.5)
    elif col is not None:
        # column col of a prefill hidden state h [D, TH] (the prompt's last token)
        c_i = nl.ndarray((1, 1), dtype=i32, buffer=nl.sbuf)
        nisa.dma_copy(dst=c_i, src=col)
        creg = nisa.register_alloc()
        nisa.register_load(creg, c_i)
        nisa.dma_copy(dst=x, src=h.ap(pattern=[[TH, 128], [128 * TH, KT], [1, 1]], offset=0, scalar_offset=creg,
                                      indirect_dim=1))
    else:
        nisa.dma_copy(dst=x, src=h.ap(pattern=[[1, 128], [128, KT], [1, 1]], offset=0))
    for li in range(len(kinds)):
        lw = lws[li]
        if kinds[li] == 'L':
            x = dec_layer(x, lw[0], lw[1], lw[2], lw[3], lw[4], lw[5], lw[6], lw[7], csL, snL, preg, sregL, biasL,
                          LB, pid, ring, ones, ident, ident4, sel, kvL[nloc % DEC_NKV], onesr)
            nloc += 1
        else:
            x = dec_layer(x, lw[0], lw[1], lw[2], lw[3], lw[4], lw[5], lw[6], lw[7], csG, snG, preg, sregG, biasG,
                          B, pid, ring, ones, ident, ident4, sel, kvG, onesr)
    if last:
        wf = nl.ndarray((128, KT), dtype=bf16, buffer=nl.sbuf)
        nisa.dma_copy(dst=wf, src=lnf.ap(pattern=[[KT, 128], [1, KT]], offset=0))
        wf1 = nl.ndarray((128, KT), dtype=fp32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=wf1, data=wf, op0=nl.add, operand0=1.0)
        xn = rmsnorm1(x[:, :, 0], wf1, onesr)
        lgb = nl.ndarray((128, VT, 1), dtype=bf16, buffer=nl.sbuf)
        gemm(embT[pid], xn, VT, KT, 1, lgb, 128, 16, ring)
        return argmax_tail(lgb, lmask, lids, toks, pnext, p_i, preg, pid)
    out = nl.ndarray((D, 1), dtype=bf16, buffer=nl.shared_hbm)
    half = KT // 2
    nisa.dma_copy(dst=out.ap(pattern=[[1, 128], [128, half], [1, 1]], offset=half * pid * 128),
                  src=x[:, half * pid:half * (pid + 1), :])
    return out


def dec_multi_body(toks, emb, pos, pnext, cosl, sinl, cosg, sing, lws, kinds, B, nsteps, lnf, embT, lmask, lids):
    """nsteps decode steps (embedding, layers lws, final norm, LM head, argmax) for positions pos .. pos+nsteps-1 in
    one NEFF; step s > 0 embeds the token step s - 1 chose (from SBUF, both cores computed it). Per step the same
    arithmetic as dec_group_body. Writes toks[pos + 1 .. pos + nsteps] and pnext = pos + nsteps."""
    pid = nl.program_id(0)
    R = 4
    ones = make_ones()
    onesr = nl.ndarray((1, 128), dtype=fp32, buffer=nl.sbuf)
    nisa.memset(dst=onesr, value=1.0)
    ident = make_ident(128, bf16)
    ident4 = make_ident(R, bf16)
    nbuf = dec_ring_slots(B if 'G' in kinds else 0)
    # local window: every position below a global bucket B < 1024 fits in a window of B keys (start 0)
    LB = min(DEC_LOCAL_B, B) if ('G' in kinds and DEC_LOCAL_SMALL) else DEC_LOCAL_B
    ring = WRing(nbuf * 16 // DEC_SL, DEC_SL)
    kvL = []
    for k in range(DEC_NKV):
        kvL.append([nl.ndarray((128, 4, LB), dtype=bf16, buffer=nl.sbuf),
                    nl.ndarray((128, LB // 128, 512), dtype=bf16, buffer=nl.sbuf)])
    kvG = None
    if 'G' in kinds and B <= DEC_MULTI_PERSIST_MAX:
        kvG = [nl.ndarray((128, 4, B), dtype=bf16, buffer=nl.sbuf), nl.ndarray((128, B // 128, 512), dtype=bf16,
                                                                               buffer=nl.sbuf)]
    io = nl.ndarray((R, 1), dtype=fp32, buffer=nl.sbuf)
    nisa.iota(dst=io, pattern=[[0, 1]], offset=0, channel_multiplier=1)
    sel = nl.ndarray((R, 2), dtype=fp32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=sel[:, 0:1], data=io, op0=nl.less, operand0=2.0)
    nisa.tensor_scalar(dst=sel[:, 1:2], data=io, op0=nl.greater_equal, operand0=2.0)
    wf = nl.ndarray((128, KT), dtype=bf16, buffer=nl.sbuf)
    nisa.dma_copy(dst=wf, src=lnf.ap(pattern=[[KT, 128], [1, KT]], offset=0))
    wf1 = nl.ndarray((128, KT), dtype=fp32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=wf1, data=wf, op0=nl.add, operand0=1.0)
    p_i0 = nl.ndarray((R, 1), dtype=i32, buffer=nl.sbuf)
    nisa.dma_copy(dst=p_i0, src=pos.ap(pattern=[[0, R], [1, 1]], offset=0))
    p_f0 = nl.ndarray((R, 1), dtype=fp32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=p_f0, src=p_i0)
    toklist = []
    out = None
    nloc = 0
    for step in range(nsteps):
        if step == 0:
            p_i = p_i0
            p_f = p_f0
        else:
            p_f = nl.ndarray((R, 1), dtype=fp32, buffer=nl.sbuf)
            nisa.tensor_scalar(dst=p_f, data=p_f0, op0=nl.add, operand0=float(step))
            p_i = nl.ndarray((R, 1), dtype=i32, buffer=nl.sbuf)
            nisa.tensor_copy(dst=p_i, src=p_f)
        preg = nisa.register_alloc()
        nisa.register_load(preg, p_i[0:1, 0:1])
        sregL = None
        sregG = None
        biasL = None
        biasG = None
        csL = None
        snL = None
        csG = None
        snG = None
        if 'L' in kinds:
            sregL, thrL = dec_window(p_f, LB)
            biasL = nl.ndarray((R, LB), dtype=bf16, buffer=nl.sbuf)
            window_bias2(R, LB, thrL, True, biasL)
            csL = nl.ndarray((128, 1), dtype=bf16, buffer=nl.sbuf)
            nisa.dma_copy(dst=csL, src=cosl.ap(pattern=[[S, 128], [1, 1]], offset=0, scalar_offset=preg, indirect_dim=1))
            snL = nl.ndarray((128, 1), dtype=bf16, buffer=nl.sbuf)
            nisa.dma_copy(dst=snL, src=sinl.ap(pattern=[[S, 128], [1, 1]], offset=0, scalar_offset=preg, indirect_dim=1))
        if 'G' in kinds:
            sregG, thrG = dec_window(p_f, B)
            biasG = nl.ndarray((R, B), dtype=bf16, buffer=nl.sbuf)
            window_bias2(R, B, thrG, False, biasG)
            csG = nl.ndarray((128, 1), dtype=bf16, buffer=nl.sbuf)
            nisa.dma_copy(dst=csG, src=cosg.ap(pattern=[[S, 128], [1, 1]], offset=0, scalar_offset=preg, indirect_dim=1))
            snG = nl.ndarray((128, 1), dtype=bf16, buffer=nl.sbuf)
            nisa.dma_copy(dst=snG, src=sing.ap(pattern=[[S, 128], [1, 1]], offset=0, scalar_offset=preg, indirect_dim=1))
        treg = nisa.register_alloc()
        if step == 0:
            t_i = nl.ndarray((1, 1), dtype=i32, buffer=nl.sbuf)
            nisa.dma_copy(dst=t_i, src=toks.ap(pattern=[[1, 1], [1, 1]], offset=0, scalar_offset=preg, indirect_dim=0))
            nisa.register_load(treg, t_i)
        else:
            t_prev = toklist[step - 1]
            nisa.register_load(treg, t_prev)
        e = nl.ndarray((128, KT), dtype=bf16, buffer=nl.sbuf)
        nisa.dma_copy(dst=e, src=emb.ap(pattern=[[1, 128], [128, KT]], offset=0, scalar_offset=treg, indirect_dim=0))
        x = nl.ndarray((128, KT, 1), dtype=bf16, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=x[:, :, 0], data=e, op0=nl.multiply, operand0=50.5)
        for li in range(len(kinds)):
            lw = lws[li]
            if kinds[li] == 'L':
                x = dec_layer(x, lw[0], lw[1], lw[2], lw[3], lw[4], lw[5], lw[6], lw[7], csL, snL, preg, sregL, biasL,
                              LB, pid, ring, ones, ident, ident4, sel, kvL[nloc % DEC_NKV], onesr)
                nloc += 1
            else:
                x = dec_layer(x, lw[0], lw[1], lw[2], lw[3], lw[4], lw[5], lw[6], lw[7], csG, snG, preg, sregG, biasG,
                              B, pid, ring, ones, ident, ident4, sel, kvG, onesr)
        xn = rmsnorm1(x[:, :, 0], wf1, onesr)
        lgb = nl.ndarray((128, VT, 1), dtype=bf16, buffer=nl.sbuf)
        gemm(embT[pid], xn, VT, KT, 1, lgb, 128, 16, ring)
        out = argmax_tail(lgb, lmask, lids, toks, pnext, p_i, preg, pid, step == nsteps - 1, toklist)
    return out


# ----------------------------------------------------------------------------------------------- prefill v2
# A chunk of T tokens (feature-major [128, KT, T]) through one layer: weights through the Sync-queue ring, norms
# as a few large instructions (squares summed over the k-tiles on the VectorE, then one fp32 ones-matmul for the
# partition sum), q/k norms and RoPE for all heads at once, and attention whose keys are the cached window
# [start, start + B - T) (loaded once per KV head, no dependency on this chunk) followed by the chunk's own keys
# taken from SBUF.

def gram_rstd(x, t0, nt, T, n, identf, onesf, dst, identm=None, doff=0):
    """dst [128, T] fp32 = 1/sqrt(mean of squares + eps) over the n features in tiles t0 .. t0+nt-1 of x [128, *, T]
    (bf16), on every partition. Sums of squares from the diagonal of the Gram matrix accumulated on the TensorE
    (bf16 products are exact in fp32), statistics per token on [128, 1], broadcast back with a ones-matmul of
    diag(rstd) (exact)."""
    for b in range(T // 128):
        g = nl.ndarray((128, 128), dtype=fp32, buffer=nl.psum)
        for kt in range(nt):
            blk = x[:, t0 + kt, b * 128:(b + 1) * 128]
            nisa.nc_matmul(dst=g, stationary=blk, moving=blk, accumulate=(kt > 0))
        dg = nl.ndarray((128, 128), dtype=fp32, buffer=nl.sbuf)
        ss = nl.ndarray((128, 1), dtype=fp32, buffer=nl.sbuf)
        if identm is not None:
            # short chain: sqrt(ss/n + eps) in one activation, diag(1/r) on the VectorE
            nisa.tensor_tensor(dst=dg, data1=g, data2=identf, op=nl.multiply)
            nisa.tensor_reduce(dst=ss, op=nl.add, data=dg, axis=1)
            r2 = nl.ndarray((128, 1), dtype=fp32, buffer=nl.sbuf)
            nisa.activation(dst=r2, op=nl.sqrt, data=ss, scale=1.0 / n, bias=EPS)
            rs = nl.ndarray((128, 1), dtype=fp32, buffer=nl.sbuf)
            nisa.reciprocal(dst=rs, data=r2)
            dr = nl.ndarray((128, 128), dtype=fp32, buffer=nl.sbuf)
            nisa.tensor_scalar(dst=dr, data=identf, op0=nl.multiply, operand0=rs, engine=DR_ENGINE)
            pb = nl.ndarray((128, 128), dtype=fp32, buffer=nl.psum)
            nisa.nc_matmul(dst=pb, stationary=onesf, moving=dr)
            nisa.tensor_copy(dst=dst[:, doff + b * 128:doff + (b + 1) * 128], src=pb, engine=nisa.engine.scalar)
            continue
        nisa.tensor_tensor(dst=dg, data1=g, data2=identf, op=nl.multiply)
        nisa.tensor_reduce(dst=ss, op=nl.add, data=dg, axis=1)
        r1 = nl.ndarray((128, 1), dtype=fp32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=r1, data=ss, op0=nl.multiply, operand0=1.0 / n, op1=nl.add, operand1=EPS)
        r2 = nl.ndarray((128, 1), dtype=fp32, buffer=nl.sbuf)
        nisa.activation(dst=r2, op=nl.sqrt, data=r1)
        rs = nl.ndarray((128, 1), dtype=fp32, buffer=nl.sbuf)
        nisa.reciprocal(dst=rs, data=r2)
        dr = nl.ndarray((128, 128), dtype=fp32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=dr, data=identf, op0=nl.multiply, operand0=rs)
        pb = nl.ndarray((128, 128), dtype=fp32, buffer=nl.psum)
        nisa.nc_matmul(dst=pb, stationary=onesf, moving=dr)
        nisa.tensor_copy(dst=dst[:, b * 128:(b + 1) * 128], src=pb, engine=nisa.engine.scalar)


def gram_rstd_a(x, t0, nt, T, n, identf, onesf):
    """First half of gram_rstd's short chain (Gram matmuls and the per-token statistics): returns diag(rstd) per
    128-token block ([128, 128] fp32 SBUF tiles); gram_rstd_b broadcasts them. Same arithmetic as gram_rstd."""
    drs = []
    for b in range(T // 128):
        g = nl.ndarray((128, 128), dtype=fp32, buffer=nl.psum)
        for kt in range(nt):
            blk = x[:, t0 + kt, b * 128:(b + 1) * 128]
            nisa.nc_matmul(dst=g, stationary=blk, moving=blk, accumulate=(kt > 0))
        dg = nl.ndarray((128, 128), dtype=fp32, buffer=nl.sbuf)
        ss = nl.ndarray((128, 1), dtype=fp32, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=dg, data1=g, data2=identf, op=nl.multiply)
        nisa.tensor_reduce(dst=ss, op=nl.add, data=dg, axis=1)
        r2 = nl.ndarray((128, 1), dtype=fp32, buffer=nl.sbuf)
        nisa.activation(dst=r2, op=nl.sqrt, data=ss, scale=1.0 / n, bias=EPS)
        rs = nl.ndarray((128, 1), dtype=fp32, buffer=nl.sbuf)
        nisa.reciprocal(dst=rs, data=r2)
        dr = nl.ndarray((128, 128), dtype=fp32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=dr, data=identf, op0=nl.multiply, operand0=rs, engine=DR_ENGINE)
        drs.append(dr)
    return drs


def gram_rstd_b(drs, onesf, dst, doff=0):
    for b in range(len(drs)):
        pb = nl.ndarray((128, 128), dtype=fp32, buffer=nl.psum)
        nisa.nc_matmul(dst=pb, stationary=onesf, moving=drs[b])
        nisa.tensor_copy(dst=dst[:, doff + b * 128:doff + (b + 1) * 128], src=pb, engine=nisa.engine.scalar)


def rmsnorm_res(x, res, w1, widx, T, identf, onesf, identm, xw=None):
    """res + RMSNorm(x) per k-tile (the scaling of tile kt, then its residual add), so that a following Gram over
    the sum can run tile by tile. Same arithmetic as rmsnormT + one tensor_tensor add.
    xw given (NORM_SPLIT): x * (1 + w) in fp32, computed per reduce-scatter piece (ar_piece) while the GEMM runs;
    the scaling by rstd and the residual add then run RES_GRP k-tiles per instruction (bitwise the same as the
    fused scalar_tensor_tensor, which computes x * w and * rstd in fp32 with one rounding at the end; checked on
    device)."""
    if xw is not None:
        rstd = nl.ndarray((128, T), dtype=fp32, buffer=nl.sbuf)
        gram_rstd(x, 0, KT, T, D, identf, onesf, rstd, identm)
        y = nl.ndarray((128, KT, T), dtype=bf16, buffer=nl.sbuf)
        h = nl.ndarray((128, KT, T), dtype=bf16, buffer=nl.sbuf)
        for k0 in range(0, KT, RES_GRP):
            n = min(RES_GRP, KT - k0)
            nisa.tensor_tensor(dst=y[:, k0:k0 + n, :], data1=xw[:, k0:k0 + n, :],
                               data2=bc(rstd, T, [[0, n], [1, T]]), op=nl.multiply)
            nisa.tensor_tensor(dst=h[:, k0:k0 + n, :], data1=res[:, k0:k0 + n, :], data2=y[:, k0:k0 + n, :],
                               op=nl.add)
        return h
    rstd = nl.ndarray((128, T), dtype=fp32, buffer=nl.sbuf)
    gram_rstd(x, 0, KT, T, D, identf, onesf, rstd, identm)
    y = nl.ndarray((128, KT, T), dtype=bf16, buffer=nl.sbuf)
    h = nl.ndarray((128, KT, T), dtype=bf16, buffer=nl.sbuf)
    for kt in range(KT):
        nisa.scalar_tensor_tensor(dst=y[:, kt, :], data=x[:, kt, :], op0=nl.multiply, operand0=w1[:, widx, kt:kt + 1],
                                  op1=nl.multiply, operand1=rstd)
        nisa.tensor_tensor(dst=h[:, kt, :], data1=res[:, kt, :], data2=y[:, kt, :], op=nl.add)
    return h


def rmsnormT(x, w1, widx, nw, T, identf, onesf, identm=None):
    """Gemma RMSNorm of x [128, KT, T] bf16 with (1 + w) = w1[:, widx, :] (w1 [128, nw, KT] fp32). bf16 out,
    y = (x * w) * rstd per k-tile (one fused instruction each)."""
    rstd = nl.ndarray((128, T), dtype=fp32, buffer=nl.sbuf)
    gram_rstd(x, 0, KT, T, D, identf, onesf, rstd, identm)
    y = nl.ndarray((128, KT, T), dtype=bf16, buffer=nl.sbuf)
    for kt in range(KT):
        nisa.scalar_tensor_tensor(dst=y[:, kt, :], data=x[:, kt, :], op0=nl.multiply, operand0=w1[:, widx, kt:kt + 1],
                                  op1=nl.multiply, operand1=rstd)
    return y


def gemm_gu(w, x, T, act, ring, rs=None):
    """Gate/up GEMM (tile pairs nt = (gate, up) of intermediate tile nt) fused with act = gelu(bf(gate)) * bf(up)."""
    NK = KT
    ps = None
    ntiles = 2 * ICT * NK
    SL = ring.sl
    for s0 in range(0, ntiles, SL):
        n = min(SL, ntiles - s0)
        wsb = ring.load(w[:, s0 * 128:(s0 + n) * 128], n)
        for j in range(n):
            i = s0 + j
            nt = i // NK
            kt = i % NK
            if kt == 0 and nt % 2 == 0:
                ps = nl.ndarray((128, 2, T), dtype=fp32, buffer=nl.psum)
            nisa.nc_matmul(dst=ps[:, nt % 2, :], stationary=wsb[:, j * 128:(j + 1) * 128], moving=x[:, kt, :])
            if kt == NK - 1 and nt % 2 == 1:
                g2 = nl.ndarray((128, 2, T), dtype=bf16, buffer=nl.sbuf)
                if rs is not None:
                    evac_scaled(g2, ps, rs, 2, T)
                else:
                    nisa.tensor_copy(dst=g2, src=ps, engine=nisa.engine.scalar)
                gl_ = nl.ndarray((128, T), dtype=bf16, buffer=nl.sbuf)
                nisa.activation(dst=gl_, op=nl.gelu_apprx_tanh, data=g2[:, 0, :])
                nisa.tensor_tensor(dst=act[:, nt // 2, :], data1=gl_, data2=g2[:, 1, :], op=nl.multiply)


class PStream(nl.NKIObject):
    """Per-chunk constants of a prefill stream: position registers, window scalars, RoPE rows, masks."""

    def __init__(self, pos, T, BL, BG, kinds, cosl, sinl, cosg, sing, off=0, lazy_g=False, masks=False, p0=None):
        self.T = T
        self.ps = None if p0 is None else p0 + off   # static stream start position, when the NEFF fixes it
        self.masks = masks
        p_i0 = nl.ndarray((128, 1), dtype=i32, buffer=nl.sbuf)
        nisa.dma_copy(dst=p_i0, src=pos.ap(pattern=[[0, 128], [1, 1]], offset=0))
        p_f = nl.ndarray((128, 1), dtype=fp32, buffer=nl.sbuf)
        if off == 0:
            nisa.tensor_copy(dst=p_f, src=p_i0)
            p_i = p_i0
        else:
            p_f0 = nl.ndarray((128, 1), dtype=fp32, buffer=nl.sbuf)
            nisa.tensor_copy(dst=p_f0, src=p_i0)
            nisa.tensor_scalar(dst=p_f, data=p_f0, op0=nl.add, operand0=float(off))
            p_i = nl.ndarray((128, 1), dtype=i32, buffer=nl.sbuf)
            nisa.tensor_copy(dst=p_i, src=p_f)
        self.p_f = p_f
        self.preg = nisa.register_alloc()
        nisa.register_load(self.preg, p_i[0:1, 0:1])
        # current-chunk mask: key j' visible to query t iff j' <= t
        nqt = T // 128
        self.bcur = []
        self.pcur = []
        for qt in range(nqt):
            io = nl.ndarray((128, T), dtype=fp32, buffer=nl.sbuf)
            nisa.iota(dst=io, pattern=[[1, T]], offset=0, channel_multiplier=-1)
            m = nl.ndarray((128, T), dtype=fp32, buffer=nl.sbuf)
            nisa.tensor_scalar(dst=m, data=io, op0=nl.greater, operand0=128.0 * qt + 0.5, op1=nl.multiply,
                               operand1=NEG)
            b = nl.ndarray((128, T), dtype=bf16, buffer=nl.sbuf)
            nisa.tensor_copy(dst=b, src=m)
            self.bcur.append(b)
            pm = nl.ndarray((128, T), dtype=nl.uint8, buffer=nl.sbuf)
            nisa.tensor_scalar(dst=pm, data=io, op0=nl.less_equal, operand0=128.0 * qt + 0.5)
            self.pcur.append(pm)
        self.L = None
        self.G = None
        if 'L' in kinds:
            self.L = self.window(p_f, BL, True, cosl, sinl)
        self.BG = BG
        self.cosg = cosg
        self.sing = sing
        if 'G' in kinds and not lazy_g:
            self.G = self.window(p_f, BG, False, cosg, sing)

    def window(self, p_f, B, local, cos, sin):
        """[sreg, B, old-key masks per q tile, cos rows, sin rows] for a bucket B (start = max(0, pos + T - B))."""
        T = self.T
        nold = B - T
        st_f = nl.ndarray((128, 1), dtype=fp32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=st_f, data=p_f, op0=nl.add, operand0=float(T - B), op1=nl.maximum, operand1=0.0)
        thr = nl.ndarray((128, 1), dtype=fp32, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=thr, data1=p_f, data2=st_f, op=nl.subtract)
        st_i = nl.ndarray((128, 1), dtype=i32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=st_i, src=st_f)
        sreg = nisa.register_alloc()
        nisa.register_load(sreg, st_i[0:1, 0:1])
        bold = []
        if nold > 0:
            for qt in range(T // 128):
                io = nl.ndarray((128, nold), dtype=fp32, buffer=nl.sbuf)
                nisa.iota(dst=io, pattern=[[1, nold]], offset=0, channel_multiplier=0)
                m = nl.ndarray((128, nold), dtype=fp32, buffer=nl.sbuf)
                nisa.tensor_scalar(dst=m, data=io, op0=nl.subtract, operand0=thr, op1=nl.greater, operand1=-0.5)
                if local:
                    io2 = nl.ndarray((128, nold), dtype=fp32, buffer=nl.sbuf)
                    nisa.iota(dst=io2, pattern=[[1, nold]], offset=0, channel_multiplier=-1)
                    m2 = nl.ndarray((128, nold), dtype=fp32, buffer=nl.sbuf)
                    nisa.tensor_scalar(dst=m2, data=io2, op0=nl.subtract, operand0=thr, op1=nl.less,
                                       operand1=128.0 * qt - 1023.5)
                    nisa.tensor_tensor(dst=m, data1=m, data2=m2, op=nl.add)
                if self.masks:
                    b = nl.ndarray((128, nold), dtype=nl.uint8, buffer=nl.sbuf)
                    nisa.tensor_scalar(dst=b, data=m, op0=nl.less, operand0=0.5)
                else:
                    b = nl.ndarray((128, nold), dtype=bf16, buffer=nl.sbuf)
                    nisa.tensor_scalar(dst=b, data=m, op0=nl.multiply, operand0=NEG)
                bold.append(b)
        cs = nl.ndarray((128, T), dtype=bf16, buffer=nl.sbuf)
        nisa.dma_copy(dst=cs, src=cos.ap(pattern=[[S, 128], [1, T]], offset=0, scalar_offset=self.preg, indirect_dim=1))
        sn = nl.ndarray((128, T), dtype=bf16, buffer=nl.sbuf)
        nisa.dma_copy(dst=sn, src=sin.ap(pattern=[[S, 128], [1, T]], offset=0, scalar_offset=self.preg, indirect_dim=1))
        return [sreg, B, bold, cs, sn, local]


def pre_layer(x, wqkv, wo, wgu, wd, lnw, qkw, kc, vc, st, win, pid, ring, ones, ident, identf):
    """One decoder layer for a chunk of T tokens at positions pos .. pos + T - 1. x [128, KT, T] bf16 (SBUF)."""
    T = st.T
    sreg = win[0]
    B = win[1]
    bold = win[2]
    cs = win[3]
    sn = win[4]
    nold = B - T
    nqt = T // 128
    nbo = nold // 128
    nb = B // 128
    # ---- small per-layer weights
    w1 = load_norm_w(lnw, 4)
    qk_sb = nl.ndarray((128, 2, 2), dtype=bf16, buffer=nl.sbuf)
    nisa.dma_copy(dst=qk_sb, src=qkw.ap(pattern=[[4, 128], [2, 2], [1, 2]], offset=0))
    qk1 = nl.ndarray((128, 2, 2), dtype=fp32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=qk1, data=qk_sb, op0=nl.add, operand0=1.0)
    w12 = nl.ndarray((128, 12), dtype=fp32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=bc(w12, 12, [[2, 4], [1, 2]]), src=bc(qk1, 4, [[0, 4], [1, 2]]))
    nisa.tensor_copy(dst=bc(w12, 12, [[2, 2], [1, 2]], 8), src=bc(qk1, 4, [[0, 2], [1, 2]], 2))

    # ---- norm + QKV
    xn = rmsnormT(x, w1, 0, 4, T, identf, ones)
    qkv = nl.ndarray((128, 16, T), dtype=bf16, buffer=nl.sbuf)
    gemm(wqkv[pid], xn, 16, KT, T, qkv, 2, 16, ring, nisa.engine.scalar)
    # cached K/V window of this core's KV heads (no dependency on this chunk; ACT queue)
    kos = []
    vos = []
    if nold > 0:
        for gl in range(2):
            ko = nl.ndarray((128, 2, nold), dtype=bf16, buffer=nl.sbuf)
            adma(dst=ko, src=kc.ap(pattern=[[S, 128], [128 * S, 2], [1, nold]], offset=(2 * pid + gl) * HD * S,
                                   scalar_offset=sreg, indirect_dim=2))
            vo = nl.ndarray((128, nbo, HD), dtype=bf16, buffer=nl.sbuf)
            adma(dst=vo, src=vc.ap(pattern=[[1024, 128], [128 * 1024, nbo], [1, HD]], offset=(2 * pid + gl) * HD,
                                   scalar_offset=sreg, indirect_dim=0))
            kos.append(ko)
            vos.append(vo)
    # ---- q/k norms: per head (2 tiles) Gram statistics, then (x * w) * rstd per tile
    qn = nl.ndarray((128, 12, T), dtype=bf16, buffer=nl.sbuf)
    for h in range(6):
        rh = nl.ndarray((128, T), dtype=fp32, buffer=nl.sbuf)
        gram_rstd(qkv, 2 * h, 2, T, HD, identf, ones, rh)
        for dt in range(2):
            t = 2 * h + dt
            nisa.scalar_tensor_tensor(dst=qn[:, t, :], data=qkv[:, t, :], op0=nl.multiply, operand0=w12[:, t:t + 1],
                                      op1=nl.multiply, operand1=rh)
    # ---- RoPE (6 heads at once)
    lo = bc(qn, 12 * T, [[2 * T, 6], [1, T]], 0)
    hi = bc(qn, 12 * T, [[2 * T, 6], [1, T]], T)
    c6 = bc(cs, T, [[0, 6], [1, T]])
    s6 = bc(sn, T, [[0, 6], [1, T]])
    qr = nl.ndarray((128, 12, T), dtype=bf16, buffer=nl.sbuf)
    ra = nl.ndarray((128, 6, T), dtype=bf16, buffer=nl.sbuf)
    rb = nl.ndarray((128, 6, T), dtype=bf16, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=ra, data1=lo, data2=c6, op=nl.multiply)
    nisa.tensor_tensor(dst=rb, data1=hi, data2=s6, op=nl.multiply)
    nisa.tensor_tensor(dst=bc(qr, 12 * T, [[2 * T, 6], [1, T]], 0), data1=ra, data2=rb, op=nl.subtract)
    rc = nl.ndarray((128, 6, T), dtype=bf16, buffer=nl.sbuf)
    rd = nl.ndarray((128, 6, T), dtype=bf16, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=rc, data1=hi, data2=c6, op=nl.multiply)
    nisa.tensor_tensor(dst=rd, data1=lo, data2=s6, op=nl.multiply)
    nisa.tensor_tensor(dst=bc(qr, 12 * T, [[2 * T, 6], [1, T]], T), data1=rc, data2=rd, op=nl.add)
    # ---- V to token-major, cache writes (GpSimd queue)
    vn = nl.ndarray((128, nqt, 512), dtype=bf16, buffer=nl.sbuf)
    for tb in range(nqt):
        pst = nl.ndarray((128, 4, 128), dtype=fp32, buffer=nl.psum)
        for vt in range(4):
            nisa.nc_matmul(dst=pst[:, vt, :], stationary=qkv[:, 12 + vt, tb * 128:(tb + 1) * 128], moving=ident)
        nisa.tensor_copy(dst=vn[:, tb, :], src=pst.ap(pattern=[[512, 128], [1, 512]], offset=0))
    nisa.dma_copy(dst=kc.ap(pattern=[[S, 128], [128 * S, 4], [1, T]], offset=(2 * pid) * HD * S, scalar_offset=st.preg,
                            indirect_dim=2), src=qr[:, 8:12, :])
    nisa.dma_copy(dst=vc.ap(pattern=[[1024, 128], [128 * 1024, nqt], [1, 512]], offset=512 * pid,
                            scalar_offset=st.preg, indirect_dim=0), src=vn)

    # ---- attention: this core's 4 q heads; KV head gl serves q heads 2gl, 2gl+1
    attn = nl.ndarray((128, 8, T), dtype=bf16, buffer=nl.sbuf)
    for hl in range(4):
        gl = hl // 2
        for qt in range(nqt):
            sc = nl.ndarray((128, B), dtype=fp32, buffer=nl.sbuf)
            for c0 in range(0, nold, QKC):
                w = min(QKC, nold - c0)
                ps = nl.ndarray((128, QKC), dtype=fp32, buffer=nl.psum)
                for dt in range(2):
                    nisa.nc_matmul(dst=ps[:, 0:w], stationary=qr[:, hl * 2 + dt, qt * 128:(qt + 1) * 128],
                                   moving=kos[gl][:, dt, c0:c0 + w], accumulate=(dt > 0))
                nisa.tensor_tensor(dst=sc[:, c0:c0 + w], data1=ps[:, 0:w], data2=bold[qt][:, c0:c0 + w], op=nl.add)
            ps = nl.ndarray((128, T), dtype=fp32, buffer=nl.psum)
            for dt in range(2):
                nisa.nc_matmul(dst=ps, stationary=qr[:, hl * 2 + dt, qt * 128:(qt + 1) * 128],
                               moving=qr[:, 8 + 2 * gl + dt, :], accumulate=(dt > 0))
            nisa.tensor_tensor(dst=sc[:, nold:B], data1=ps, data2=st.bcur[qt], op=nl.add)
            p, sm = softmax_rows(sc, 128, B)
            pt = nl.ndarray((128, nb, 128), dtype=bf16, buffer=nl.sbuf)
            for b0 in range(0, nb, 4):
                n = min(4, nb - b0)
                pst = nl.ndarray((128, 4, 128), dtype=fp32, buffer=nl.psum)
                for b in range(n):
                    kb = b0 + b
                    nisa.nc_matmul(dst=pst[:, b, :], stationary=p[:, kb * 128:(kb + 1) * 128], moving=ident)
                nisa.tensor_copy(dst=pt[:, b0:b0 + n, :], src=pst[:, 0:n, :])
            po = nl.ndarray((128, HD), dtype=fp32, buffer=nl.psum)
            for kb in range(nb):
                if kb < nbo:
                    mv = vos[gl][:, kb, :]
                else:
                    mv = vn[:, kb - nbo, gl * HD:(gl + 1) * HD]
                nisa.nc_matmul(dst=po, stationary=pt[:, kb, :], moving=mv, accumulate=(kb > 0))
            rs = nl.ndarray((128, 1), dtype=fp32, buffer=nl.sbuf)
            nisa.reciprocal(dst=rs, data=sm)
            o = nl.ndarray((128, HD), dtype=bf16, buffer=nl.sbuf)
            nisa.tensor_scalar(dst=o, data=po, op0=nl.multiply, operand0=rs)
            pso = nl.ndarray((128, 2, 128), dtype=fp32, buffer=nl.psum)
            for dt in range(2):
                nisa.nc_matmul(dst=pso[:, dt, :], stationary=o[:, dt * 128:(dt + 1) * 128], moving=ident)
            nisa.tensor_copy(dst=bc(attn, 8 * T, [[4 * T, 2], [1, 128]], hl * T + qt * 128), src=pso)

    # ---- o_proj + residual
    part = nl.ndarray((128, KT, T), dtype=fp32, buffer=nl.sbuf)
    ob = nl.ndarray((128, KT, T), dtype=bf16, buffer=nl.sbuf)
    gemm(wo[pid], attn, KT, 8, T, part, 2, 16, ring, nisa.engine.scalar, AR(ob, pid))
    on = rmsnormT(ob, w1, 1, 4, T, identf, ones)
    h1 = nl.ndarray((128, KT, T), dtype=bf16, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=h1, data1=x, data2=on, op=nl.add)
    # ---- MLP
    xn2 = rmsnormT(h1, w1, 2, 4, T, identf, ones)
    act = nl.ndarray((128, ICT, T), dtype=bf16, buffer=nl.sbuf)
    gemm_gu(wgu[pid], xn2, T, act, ring)
    part2 = nl.ndarray((128, KT, T), dtype=fp32, buffer=nl.sbuf)
    db = nl.ndarray((128, KT, T), dtype=bf16, buffer=nl.sbuf)
    gemm(wd[pid], act, KT, ICT, T, part2, 2, 16, ring, nisa.engine.scalar, AR(db, pid))
    dn = rmsnormT(db, w1, 3, 4, T, identf, ones)
    h2 = nl.ndarray((128, KT, T), dtype=bf16, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=h2, data1=h1, data2=dn, op=nl.add)
    return h2


def pre2_body(h, pos, wqkv, wo, wgu, wd, lnw, qkw, kc, vc, cos, sin, T, B, local):
    """Single-layer prefill NEFF (same interface as layer_body)."""
    pid = nl.program_id(0)
    ones = make_ones()
    ident = make_ident(128, bf16)
    ring = WRing(RING_PRE, 16)
    kinds = 'L' if local else 'G'
    st = PStream(pos, T, B, B, kinds, cos, sin, cos, sin)
    x = nl.ndarray((128, KT, T), dtype=bf16, buffer=nl.sbuf)
    nisa.dma_copy(dst=x, src=h.ap(pattern=[[T, 128], [128 * T, KT], [1, T]], offset=0))
    win = st.L if local else st.G
    identf = make_ident(128, fp32)
    h2 = pre_layer(x, wqkv, wo, wgu, wd, lnw, qkw, kc, vc, st, win, pid, ring, ones, ident, identf)
    out = nl.ndarray((D, T), dtype=bf16, buffer=nl.shared_hbm)
    half = KT // 2
    nisa.dma_copy(dst=out.ap(pattern=[[T, 128], [128 * T, half], [1, T]], offset=half * pid * 128 * T),
                  src=h2[:, half * pid:half * (pid + 1), :])
    return out


def pre_group_body(h, pos, cosl, sinl, cosg, sing, lws, kinds, BL, BG, T):
    """Prefill layers lws[i] (kinds[i] 'L' with window bucket BL, 'G' with bucket BG) for one chunk of T tokens
    at positions pos .. pos + T - 1; h [D, T] in, [D, T] out, the hidden state in SBUF between layers."""
    pid = nl.program_id(0)
    ones = make_ones()
    ident = make_ident(128, bf16)
    identf = make_ident(128, fp32)
    ring = WRing(RING_PRE, 16)
    st = PStream(pos, T, BL, BG, kinds, cosl, sinl, cosg, sing)
    x = nl.ndarray((128, KT, T), dtype=bf16, buffer=nl.sbuf)
    nisa.dma_copy(dst=x, src=h.ap(pattern=[[T, 128], [128 * T, KT], [1, T]], offset=0))
    for li in range(len(kinds)):
        lw = lws[li]
        win = st.L
        if kinds[li] == 'G':
            win = st.G
        x = pre_layer(x, lw[0], lw[1], lw[2], lw[3], lw[4], lw[5], lw[6], lw[7], st, win, pid, ring, ones, ident,
                      identf)
    out = nl.ndarray((D, T), dtype=bf16, buffer=nl.shared_hbm)
    half = KT // 2
    nisa.dma_copy(dst=out.ap(pattern=[[T, 128], [128 * T, half], [1, T]], offset=half * pid * 128 * T),
                  src=x[:, half * pid:half * (pid + 1), :])
    return out
