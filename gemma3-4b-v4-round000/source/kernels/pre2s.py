"""Two-stream prefill: one NEFF processes a chunk of 2T tokens as two streams A (positions pos .. pos+T-1) and B
(pos+T .. pos+2T-1) through several layers. Both streams share one weight stream (each slab is used by A and, a few
slabs later, by B), and the program interleaves the streams phase by phase, so that the serial vector/scalar work of
one stream (norms, softmax, exchanges) overlaps the TensorE work of the other. Arithmetic per stream is the same as
pre_layer in gemma.py.
"""
import nki
import nki.language as nl
import nki.isa as nisa
from gemma import *


class GAcc(nl.NKIObject):
    """PSUM accumulation state of one stream inside a GEMM."""

    def __init__(self):
        self.ps = None
        self.qk = None      # [S, w12, C]: emit q/k norms per head as the qkv GEMM evacuates it
        self.rs = None      # deferred norm: rstd [128, T] applied at the evacuation (v tiles of qkv, gate/up)
        self.pend = None


class SState(nl.NKIObject):
    """Activations of one prefill stream (all [128, tiles, T] SBUF tensors)."""

    def __init__(self, st, T):
        self.st = st
        self.T = T
        self.th = T // 2      # tokens of this stream whose residual / norms this core owns
        self.x = None
        self.xn = None
        self.qkv = None
        self.qr = None
        self.vn = None
        self.kos = None
        self.vos = None
        self.attn = None
        self.h1 = None
        self.xn2 = None
        self.act = None
        self.part = None
        self.ob = None
        self.win = None
        self.rsn1 = None     # deferred norms: rstd of the input norm / pre-feedforward norm
        self.rsn2 = None
        self.obxw = None     # NORM_SPLIT: ob * (1 + w) of the post norm, computed per reduce-scatter piece
        self.defer = DEFER_NORM


class Consts(nl.NKIObject):
    def __init__(self, pid, ring):
        self.pid = pid
        self.ring = ring
        self.ones = make_ones()
        self.ident = make_ident(128, bf16)
        self.identf = make_ident(128, fp32)
        self.identm = make_identm()


def mm_slab(acc, wsb, s0, n, NK, NT, T, x, dst, G, evac, ar=None):
    """Matmuls of one weight slab (tiles s0 .. s0+n-1 of a (nt, kt) row-major tile list) for one stream."""
    for j in range(n):
        i = s0 + j
        nt = i // NK
        kt = i % NK
        g0 = (nt // G) * G
        if kt == 0 and nt == g0:
            acc.ps = nl.ndarray((128, G, T), dtype=fp32, buffer=nl.psum)
        nisa.nc_matmul(dst=acc.ps[:, nt - g0, :], stationary=wsb[:, j * 128:(j + 1) * 128], moving=x[:, kt, :])
        if kt == NK - 1 and (nt == g0 + G - 1 or nt == NT - 1):
            gn = nt - g0 + 1
            if acc.rs is not None and g0 >= 12:
                evac_scaled(dst[:, g0:g0 + gn, :], acc.ps[:, 0:gn, :], acc.rs, gn, T)
            else:
                nisa.tensor_copy(dst=dst[:, g0:g0 + gn, :], src=acc.ps[:, 0:gn, :], engine=evac)
            if ar is not None:
                ar_piece(ar, dst, g0, gn, T)
            if acc.qk is not None:
                qk_hook(acc, g0, T)


class Job(nl.NKIObject):
    """A GEMM over one pass of weight slabs for streams A and B (B's matmuls of slab s follow A's of slab s + lag).
    lead() emits all of A and the first ns - lag slabs of B; tail() the remaining B slabs, so that A's next phase
    can be emitted in between (lag < ring slots)."""

    def __init__(self, w, xA, xB, NT, NK, T, dA, dB, G, ring, lag, evac, gu, arA=None, arB=None):
        self.w = w
        self.xA = xA
        self.xB = xB
        self.NT = NT
        self.NK = NK
        self.T = T
        self.dA = dA
        self.dB = dB
        self.G = G
        self.ring = ring
        self.evac = evac
        self.gu = gu
        self.arA = arA
        self.arB = arB
        self.ntiles = NT * NK
        self.SL = ring.sl
        self.ns = (self.ntiles + self.SL - 1) // self.SL
        self.lag = min(lag, self.ns)
        self.bufs = []
        self.aA = GAcc()
        self.aB = GAcc()
        self.nfullB = None
        self.lagB = 0
        self.bstop = self.ns

    def slab(self, acc, sb, x, d, ar):
        s0 = sb * self.SL
        n = min(self.SL, self.ntiles - s0)
        if self.gu:
            gu_slab(acc, self.bufs[sb], s0, n, self.T, x, d)
        else:
            mm_slab(acc, self.bufs[sb], s0, n, self.NK, self.NT, self.T, x, d, self.G, self.evac, ar)

    def lead(self):
        for step in range(self.ns):
            s0 = step * self.SL
            n = min(self.SL, self.ntiles - s0)
            self.bufs.append(self.ring.load(self.w[:, s0 * 128:(s0 + n) * 128], n))
            self.slab(self.aA, step, self.xA, self.dA, self.arA)
            if step >= self.lag and step - self.lag < self.bstop:
                self.slab(self.aB, step - self.lag, self.xB, self.dB, self.arB)

    def tail(self):
        for sb in range(max(self.ns - self.lag, 0), self.bstop):
            self.slab(self.aB, sb, self.xB, self.dB, self.arB)
        if self.nfullB is not None:
            rs_split_groups(self.bufs, self.SL, self.xB, self.NT // self.G, self.NK, self.G, self.evac, self.arB,
                            self.lagB, self.nfullB)

    def split_b(self, nfull, lag):
        """Stream B's PSUM groups nfull .. of this (reduce-scatter) GEMM run in tail() with the other core's
        tokens first (rs_split_groups); B's earlier groups run slab by slab as before."""
        self.nfullB = nfull
        self.lagB = lag
        self.bstop = nfull * self.G * self.NK // self.SL


def rs_split_groups(bufs, SL, x, NG, NK, G, evac, ar, lag, g_first):
    """PSUM groups g_first .. NG-1 of a reduce-scatter GEMM whose weight slabs are already loaded (bufs, by slab):
    the other core's tokens of each group first (partial sums sent at once), this core's tokens `lag` groups later.
    Same per-column arithmetic as mm_slab + ar_piece."""
    th = ar.th
    pid = ar.pid
    mine = pid * th
    theirs = (1 - pid) * th
    tpg = G * NK
    others = []
    for t in range(g_first, NG + lag):
        if t < NG:
            ps = nl.ndarray((128, G, th), dtype=fp32, buffer=nl.psum)
            for j in range(tpg):
                i = t * tpg + j
                b = bufs[i // SL]
                jj = i % SL
                nisa.nc_matmul(dst=ps[:, j // NK, :], stationary=b[:, jj * 128:(jj + 1) * 128],
                               moving=x[:, j % NK, theirs:theirs + th])
            pr = nl.ndarray((128, G, th), dtype=fp32, buffer=nl.sbuf)
            nisa.tensor_copy(dst=pr, src=ps, engine=evac)
            oth = nl.ndarray((128, G, th), dtype=fp32, buffer=nl.sbuf)
            nisa.sendrecv(src=pr, dst=oth, send_to_rank=1 - pid, recv_from_rank=1 - pid, pipe_id=0)
            others.append(oth)
        g = t - lag
        if g >= g_first:
            ps2 = nl.ndarray((128, G, th), dtype=fp32, buffer=nl.psum)
            for j in range(tpg):
                i = g * tpg + j
                b = bufs[i // SL]
                jj = i % SL
                nisa.nc_matmul(dst=ps2[:, j // NK, :], stationary=b[:, jj * 128:(jj + 1) * 128],
                               moving=x[:, j % NK, mine:mine + th])
            pm = nl.ndarray((128, G, th), dtype=fp32, buffer=nl.sbuf)
            nisa.tensor_copy(dst=pm, src=ps2, engine=evac)
            g0 = g * G
            nisa.tensor_tensor(dst=ar.ob[:, g0:g0 + G, :], data1=pm, data2=others[g - g_first], op=nl.add)
            if ar.xw is not None:
                for kt in range(g0, g0 + G):
                    nisa.tensor_scalar(dst=ar.xw[:, kt, :], data=ar.ob[:, kt, :], op0=nl.multiply,
                                       operand0=ar.w1[:, ar.widx, kt:kt + 1], engine=nisa.engine.vector)


def gu_slab(acc, wsb, s0, n, T, x, act):
    NK = KT
    for j in range(n):
        i = s0 + j
        nt = i // NK
        kt = i % NK
        if kt == 0 and nt % 2 == 0:
            acc.ps = nl.ndarray((128, 2, T), dtype=fp32, buffer=nl.psum)
        nisa.nc_matmul(dst=acc.ps[:, nt % 2, :], stationary=wsb[:, j * 128:(j + 1) * 128], moving=x[:, kt, :])
        if kt == NK - 1 and nt % 2 == 1:
            g2 = nl.ndarray((128, 2, T), dtype=bf16, buffer=nl.sbuf)
            if acc.rs is not None:
                evac_scaled(g2, acc.ps, acc.rs, 2, T)
            else:
                nisa.tensor_copy(dst=g2, src=acc.ps, engine=nisa.engine.scalar)
            gl_ = nl.ndarray((128, T), dtype=bf16, buffer=nl.sbuf)
            nisa.activation(dst=gl_, op=nl.gelu_apprx_tanh, data=g2[:, 0, :])
            nisa.tensor_tensor(dst=act[:, nt // 2, :], data1=gl_, data2=g2[:, 1, :], op=nl.multiply)


def gemm_rs(w, x, NT, NK, G, ring, evac, ar, lag, nfull=0, part=None):
    """o_proj / down GEMM of a single stream with a reduce-scatter over tokens (ar.th > 0), remote tokens first:
    per PSUM group of G n-tiles, the matmuls for the other core's tokens (moving columns theirs) run first and their
    fp32 partial sums are sent at once; the matmuls for this core's tokens of the group run `lag` groups later, so
    the exchange overlaps the rest of the GEMM. Per column the same matmuls, accumulation order, evacuation and fp32
    add as gemm() + ar_piece()."""
    th = ar.th
    pid = ar.pid
    mine = pid * th
    theirs = (1 - pid) * th
    SL = ring.sl
    tpg = G * NK
    NG = NT // G
    spg = (tpg + SL - 1) // SL
    bufs = []
    others = []
    for t in range(nfull):
        while len(bufs) * SL < (t + 1) * tpg:
            s0 = len(bufs) * SL
            bufs.append(ring.load(w[:, s0 * 128:(s0 + SL) * 128], SL))
        ps = nl.ndarray((128, G, 2 * th), dtype=fp32, buffer=nl.psum)
        for j in range(tpg):
            i = t * tpg + j
            b = bufs[i // SL]
            jj = i % SL
            nisa.nc_matmul(dst=ps[:, j // NK, :], stationary=b[:, jj * 128:(jj + 1) * 128], moving=x[:, j % NK, :])
        nisa.tensor_copy(dst=part[:, t * G:(t + 1) * G, :], src=ps, engine=evac)
        ar_piece(ar, part, t * G, G, 2 * th)
        others.append(None)
    for t in range(nfull, NG + lag):
        if t < NG:
            while len(bufs) * SL < (t + 1) * tpg:
                s0 = len(bufs) * SL
                bufs.append(ring.load(w[:, s0 * 128:(s0 + SL) * 128], SL))
            ps = nl.ndarray((128, G, th), dtype=fp32, buffer=nl.psum)
            for j in range(tpg):
                i = t * tpg + j
                b = bufs[i // SL]
                jj = i % SL
                nisa.nc_matmul(dst=ps[:, j // NK, :], stationary=b[:, jj * 128:(jj + 1) * 128],
                               moving=x[:, j % NK, theirs:theirs + th])
            pr = nl.ndarray((128, G, th), dtype=fp32, buffer=nl.sbuf)
            nisa.tensor_copy(dst=pr, src=ps, engine=evac)
            oth = nl.ndarray((128, G, th), dtype=fp32, buffer=nl.sbuf)
            nisa.sendrecv(src=pr, dst=oth, send_to_rank=1 - pid, recv_from_rank=1 - pid, pipe_id=0)
            others.append(oth)
        g = t - lag
        if g >= nfull:
            ps2 = nl.ndarray((128, G, th), dtype=fp32, buffer=nl.psum)
            for j in range(tpg):
                i = g * tpg + j
                b = bufs[i // SL]
                jj = i % SL
                nisa.nc_matmul(dst=ps2[:, j // NK, :], stationary=b[:, jj * 128:(jj + 1) * 128],
                               moving=x[:, j % NK, mine:mine + th])
            pm = nl.ndarray((128, G, th), dtype=fp32, buffer=nl.sbuf)
            nisa.tensor_copy(dst=pm, src=ps2, engine=evac)
            g0 = g * G
            nisa.tensor_tensor(dst=ar.ob[:, g0:g0 + G, :], data1=pm, data2=others[g], op=nl.add)
            if ar.xw is not None:
                for kt in range(g0, g0 + G):
                    nisa.tensor_scalar(dst=ar.xw[:, kt, :], data=ar.ob[:, kt, :], op0=nl.multiply,
                                       operand0=ar.w1[:, ar.widx, kt:kt + 1], engine=nisa.engine.vector)


def gemm_gu_split(w, x, act, ring, pid, th, lag, rs=None, nsplit=ICT):
    """Gate/up GEMM of a single stream (as gemm_gu), this core's tokens first: x's columns of the other core's
    tokens arrive by the all-gather, so their matmuls run `lag` tile pairs later. Same per-column arithmetic."""
    NK = KT
    SL = ring.sl
    mine = pid * th
    theirs = (1 - pid) * th
    tpg = 2 * NK
    NG = ICT
    bufs = []
    for t in range(NG + lag):
        for half in range(3):
            g = t if half != 1 else t - lag
            if g < 0 or g >= NG:
                continue
            if half == 2:
                continue
            if g >= nsplit:
                if half == 1:
                    continue
                while len(bufs) * SL < (g + 1) * tpg:
                    s0 = len(bufs) * SL
                    bufs.append(ring.load(w[:, s0 * 128:(s0 + SL) * 128], SL))
                psf = nl.ndarray((128, 2, 2 * th), dtype=fp32, buffer=nl.psum)
                for j in range(tpg):
                    i = g * tpg + j
                    b = bufs[i // SL]
                    jj = i % SL
                    nisa.nc_matmul(dst=psf[:, j // NK, :], stationary=b[:, jj * 128:(jj + 1) * 128], moving=x[:, j % NK, :])
                gf = nl.ndarray((128, 2, 2 * th), dtype=bf16, buffer=nl.sbuf)
                nisa.tensor_copy(dst=gf, src=psf, engine=nisa.engine.scalar)
                glf = nl.ndarray((128, 2 * th), dtype=bf16, buffer=nl.sbuf)
                nisa.activation(dst=glf, op=nl.gelu_apprx_tanh, data=gf[:, 0, :])
                nisa.tensor_tensor(dst=act[:, g, :], data1=glf, data2=gf[:, 1, :], op=nl.multiply)
                continue
            c0 = mine if half == 0 else theirs
            while len(bufs) * SL < (g + 1) * tpg:
                s0 = len(bufs) * SL
                bufs.append(ring.load(w[:, s0 * 128:(s0 + SL) * 128], SL))
            ps = nl.ndarray((128, 2, th), dtype=fp32, buffer=nl.psum)
            for j in range(tpg):
                i = g * tpg + j
                b = bufs[i // SL]
                jj = i % SL
                nisa.nc_matmul(dst=ps[:, j // NK, :], stationary=b[:, jj * 128:(jj + 1) * 128],
                               moving=x[:, j % NK, c0:c0 + th])
            g2 = nl.ndarray((128, 2, th), dtype=bf16, buffer=nl.sbuf)
            if rs is not None:
                defer_finish(rs)
                nisa.tensor_tensor(dst=g2, data1=ps, data2=rs.rstd.ap(pattern=[[2 * th, 128], [0, 2], [1, th]], offset=c0),
                                   op=nl.multiply)
            else:
                nisa.tensor_copy(dst=g2, src=ps, engine=nisa.engine.scalar)
            gl_ = nl.ndarray((128, th), dtype=bf16, buffer=nl.sbuf)
            nisa.activation(dst=gl_, op=nl.gelu_apprx_tanh, data=g2[:, 0, :])
            nisa.tensor_tensor(dst=act[:, g, c0:c0 + th], data1=gl_, data2=g2[:, 1, :], op=nl.multiply)


def layer_small(lnw, qkw):
    """[(1 + w) of the 4 norms [128, 4, KT] fp32, (1 + w) of the q/k norms per q/k tile [128, 12] fp32]."""
    w1 = load_norm_w(lnw, 4)
    qk_sb = nl.ndarray((128, 2, 2), dtype=bf16, buffer=nl.sbuf)
    nisa.dma_copy(dst=qk_sb, src=qkw.ap(pattern=[[4, 128], [2, 2], [1, 2]], offset=0))
    qk1 = nl.ndarray((128, 2, 2), dtype=fp32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=qk1, data=qk_sb, op0=nl.add, operand0=1.0)
    w12 = nl.ndarray((128, 12), dtype=fp32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=bc(w12, 12, [[2, 4], [1, 2]]), src=bc(qk1, 4, [[0, 4], [1, 2]]))
    nisa.tensor_copy(dst=bc(w12, 12, [[2, 2], [1, 2]], 8), src=bc(qk1, 4, [[0, 2], [1, 2]], 2))
    return [w1, w12]


def p_norm1(S, w1, C):
    if S.defer:
        r = rmsnorm_defer(S.x, w1, 0, S.th, C.pid, C.identf, C.ones, AG_GRP, C.identm)
        S.xn = r[0]
        S.rsn1 = r[1]
        return
    S.xn = rmsnorm_ag(S.x, w1, 0, S.th, C.pid, C.identf, C.ones, AG_GRP, C.identm)


def p_kv_load(S, win, kc, vc, C):
    """The cached window of this core's KV heads for stream S (ACT queue)."""
    T = S.T
    sreg = win[0]
    nold = win[1] - T
    nbo = nold // 128
    S.kos = []
    S.vos = []
    if nold > 0:
        for gl in range(2):
            ko = nl.ndarray((128, 2, nold), dtype=bf16, buffer=nl.sbuf)
            adma(dst=ko, src=kc.ap(pattern=[[S_, 128], [128 * S_, 2], [1, nold]], offset=(2 * C.pid + gl) * HD * S_,
                                   scalar_offset=sreg, indirect_dim=2))
            vo = nl.ndarray((128, nbo, HD), dtype=bf16, buffer=nl.sbuf)
            adma(dst=vo, src=vc.ap(pattern=[[1024, 128], [128 * 1024, nbo], [1, HD]], offset=(2 * C.pid + gl) * HD,
                                   scalar_offset=sreg, indirect_dim=0))
            S.kos.append(ko)
            S.vos.append(vo)


def p_qk(S, win, w12, kc, vc, C):
    """q/k norms, RoPE, V to token-major, cache writes."""
    T = S.T
    nqt = T // 128
    cs = win[3]
    sn = win[4]
    qkv = S.qkv
    qn = nl.ndarray((128, 12, T), dtype=bf16, buffer=nl.sbuf)
    for h in range(6):
        rh = nl.ndarray((128, T), dtype=fp32, buffer=nl.sbuf)
        gram_rstd(qkv, 2 * h, 2, T, HD, C.identf, C.ones, rh, C.identm)
        for dt in range(2):
            t = 2 * h + dt
            nisa.scalar_tensor_tensor(dst=qn[:, t, :], data=qkv[:, t, :], op0=nl.multiply, operand0=w12[:, t:t + 1],
                                      op1=nl.multiply, operand1=rh)
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
    vn = nl.ndarray((128, nqt, 512), dtype=bf16, buffer=nl.sbuf)
    for tb in range(nqt):
        pst = nl.ndarray((128, 4, 128), dtype=fp32, buffer=nl.psum)
        for vt in range(4):
            nisa.nc_matmul(dst=pst[:, vt, :], stationary=qkv[:, 12 + vt, tb * 128:(tb + 1) * 128], moving=C.ident)
        nisa.tensor_copy(dst=vn[:, tb, :], src=pst.ap(pattern=[[512, 128], [1, 512]], offset=0))
    nisa.dma_copy(dst=kc.ap(pattern=[[S_, 128], [128 * S_, 4], [1, T]], offset=(2 * C.pid) * HD * S_,
                            scalar_offset=S.st.preg, indirect_dim=2), src=qr[:, 8:12, :])
    nisa.dma_copy(dst=vc.ap(pattern=[[1024, 128], [128 * 1024, nqt], [1, 512]], offset=512 * C.pid,
                            scalar_offset=S.st.preg, indirect_dim=0), src=vn)
    S.qr = qr
    S.vn = vn


def qkv_gemm_norm(w, S, w12, ring, evac, C):
    """qkv GEMM (as gemm(), G = 2) with each q/k head's norm statistics and scaling emitted right after the head's
    two output tiles are evacuated, so the norm chains run during the rest of the GEMM. Writes S.qkv, S.qn."""
    T = S.T
    NK = KT
    NT = 16
    G = 2
    x = S.xn
    dst = S.qkv
    S.qn = nl.ndarray((128, 12, T), dtype=bf16, buffer=nl.sbuf)
    S.qr = nl.ndarray((128, 12, T), dtype=bf16, buffer=nl.sbuf)
    ntiles = NT * NK
    SL = ring.sl
    ps = None
    pend = None
    for s0 in range(0, ntiles, SL):
        n = min(SL, ntiles - s0)
        wsb = ring.load(w[:, s0 * 128:(s0 + n) * 128], n)
        for j in range(n):
            i = s0 + j
            nt = i // NK
            kt = i % NK
            g0 = (nt // G) * G
            if kt == 0 and nt == g0:
                ps = nl.ndarray((128, G, T), dtype=fp32, buffer=nl.psum)
            nisa.nc_matmul(dst=ps[:, nt - g0, :], stationary=wsb[:, j * 128:(j + 1) * 128], moving=x[:, kt, :])
            if kt == NK - 1 and nt == g0 + G - 1:
                if S.rsn1 is not None and g0 >= 12:
                    evac_scaled(dst[:, g0:g0 + G, :], ps, S.rsn1, G, T)
                else:
                    nisa.tensor_copy(dst=dst[:, g0:g0 + G, :], src=ps, engine=evac)
                if pend is not None:
                    qk_norm_finish(S, pend, w12, C)
                    pend = None
                if g0 < 12:
                    h = g0 // 2
                    pend = [h, gram_rstd_a(dst, 2 * h, 2, T, HD, C.identf, C.ones)]
    if pend is not None:
        qk_norm_finish(S, pend, w12, C)


def qk_hook(acc, g0, T):
    S = acc.qk[0]
    w12 = acc.qk[1]
    C = acc.qk[2]
    if acc.pend is not None:
        qk_norm_finish(S, acc.pend, w12, C)
        acc.pend = None
    if g0 < 12:
        h = g0 // 2
        acc.pend = [h, gram_rstd_a(S.qkv, 2 * h, 2, T, HD, C.identf, C.ones)]


def qk_flush(acc):
    if acc.pend is not None:
        qk_norm_finish(acc.qk[0], acc.pend, acc.qk[1], acc.qk[2])
        acc.pend = None


def qk_norm_finish(S, pend, w12, C):
    """Broadcast of a q/k head's rstd (deferred by one GEMM group so the TensorE does not wait for the statistics)
    and the head's scaling."""
    T = S.T
    h = pend[0]
    rh = nl.ndarray((128, T), dtype=fp32, buffer=nl.sbuf)
    gram_rstd_b(pend[1], C.ones, rh)
    for dt in range(2):
        t = 2 * h + dt
        nisa.scalar_tensor_tensor(dst=S.qn[:, t, :], data=S.qkv[:, t, :], op0=nl.multiply,
                                  operand0=w12[:, t:t + 1], op1=nl.multiply, operand1=rh)
    if QK_ROPE_INGEMM:
        rope_heads(S, S.win, h, 1)


def p_qk_rest(S, win, kc, vc, C):
    """p_qk after the norms (S.qn given): RoPE of all heads, V to token-major, cache writes."""
    p_qk_pre_v(S, C)
    if not QK_ROPE_INGEMM:
        rope_heads(S, win, 0, 6)
    p_qk_post(S, kc, vc, C)


def p_qk_pre_v(S, C):
    T = S.T
    nqt = T // 128
    qkv = S.qkv
    vn = nl.ndarray((128, nqt, 512), dtype=bf16, buffer=nl.sbuf)
    for tb in range(nqt):
        pst = nl.ndarray((128, 4, 128), dtype=fp32, buffer=nl.psum)
        for vt in range(4):
            nisa.nc_matmul(dst=pst[:, vt, :], stationary=qkv[:, 12 + vt, tb * 128:(tb + 1) * 128], moving=C.ident)
        nisa.tensor_copy(dst=vn[:, tb, :], src=pst.ap(pattern=[[512, 128], [1, 512]], offset=0))
    S.vn = vn


def p_qk_pre(S, C):
    """Allocations of p_qk's outputs and the V transposes (independent of the q/k norms)."""
    T = S.T
    nqt = T // 128
    qkv = S.qkv
    S.qn = nl.ndarray((128, 12, T), dtype=bf16, buffer=nl.sbuf)
    S.qr = nl.ndarray((128, 12, T), dtype=bf16, buffer=nl.sbuf)
    vn = nl.ndarray((128, nqt, 512), dtype=bf16, buffer=nl.sbuf)
    for tb in range(nqt):
        pst = nl.ndarray((128, 4, 128), dtype=fp32, buffer=nl.psum)
        for vt in range(4):
            nisa.nc_matmul(dst=pst[:, vt, :], stationary=qkv[:, 12 + vt, tb * 128:(tb + 1) * 128], moving=C.ident)
        nisa.tensor_copy(dst=vn[:, tb, :], src=pst.ap(pattern=[[512, 128], [1, 512]], offset=0))
    S.vn = vn


def rope_heads(S, win, h0, nh):
    """RoPE of heads h0 .. h0+nh-1 of S.qn into S.qr (same instructions as p_qk, over nh heads)."""
    T = S.T
    qn = S.qn
    qr = S.qr
    cs = win[3]
    sn = win[4]
    lo = bc(qn, 12 * T, [[2 * T, nh], [1, T]], 2 * h0 * T)
    hi = bc(qn, 12 * T, [[2 * T, nh], [1, T]], 2 * h0 * T + T)
    c6 = bc(cs, T, [[0, nh], [1, T]])
    s6 = bc(sn, T, [[0, nh], [1, T]])
    ra = nl.ndarray((128, nh, T), dtype=bf16, buffer=nl.sbuf)
    rb = nl.ndarray((128, nh, T), dtype=bf16, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=ra, data1=lo, data2=c6, op=nl.multiply)
    nisa.tensor_tensor(dst=rb, data1=hi, data2=s6, op=nl.multiply)
    nisa.tensor_tensor(dst=bc(qr, 12 * T, [[2 * T, nh], [1, T]], 2 * h0 * T), data1=ra, data2=rb, op=nl.subtract)
    rc = nl.ndarray((128, nh, T), dtype=bf16, buffer=nl.sbuf)
    rd = nl.ndarray((128, nh, T), dtype=bf16, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=rc, data1=hi, data2=c6, op=nl.multiply)
    nisa.tensor_tensor(dst=rd, data1=lo, data2=s6, op=nl.multiply)
    nisa.tensor_tensor(dst=bc(qr, 12 * T, [[2 * T, nh], [1, T]], 2 * h0 * T + T), data1=rc, data2=rd, op=nl.add)


def p_qk_group(S, win, w12, gl, C):
    """q/k norms and RoPE of KV group gl: its k head (head 4 + gl) and q heads 2gl, 2gl + 1."""
    T = S.T
    qkv = S.qkv
    for h in [4 + gl, 2 * gl, 2 * gl + 1]:
        rh = nl.ndarray((128, T), dtype=fp32, buffer=nl.sbuf)
        gram_rstd(qkv, 2 * h, 2, T, HD, C.identf, C.ones, rh, C.identm)
        for dt in range(2):
            t = 2 * h + dt
            nisa.scalar_tensor_tensor(dst=S.qn[:, t, :], data=qkv[:, t, :], op0=nl.multiply,
                                      operand0=w12[:, t:t + 1], op1=nl.multiply, operand1=rh)
    rope_heads(S, win, 4 + gl, 1)
    rope_heads(S, win, 2 * gl, 2)


def p_qk_post(S, kc, vc, C):
    T = S.T
    nqt = T // 128
    nisa.dma_copy(dst=kc.ap(pattern=[[S_, 128], [128 * S_, 4], [1, T]], offset=(2 * C.pid) * HD * S_,
                            scalar_offset=S.st.preg, indirect_dim=2), src=S.qr[:, 8:12, :])
    nisa.dma_copy(dst=vc.ap(pattern=[[1024, 128], [128 * 1024, nqt], [1, 512]], offset=512 * C.pid,
                            scalar_offset=S.st.preg, indirect_dim=0), src=S.vn)


def p_qk_attn(items, w12, kc, vc, C):
    """q/k norms + RoPE and attention per KV group, so that group 0's attention runs while group 1's norm chains
    are in flight; items = [[S, win], ...]. Same arithmetic as p_qk + p_attn_multi."""
    work = []
    for it in items:
        S = it[0]
        p_qk_pre(S, C)
        S.attn = nl.ndarray((128, 8, S.T), dtype=bf16, buffer=nl.sbuf)
    for gl in range(2):
        for it in items:
            S = it[0]
            nq = S.T // 128
            for hl in [2 * gl, 2 * gl + 1]:
                for qt in range(nq):
                    work.append([S, it[1], hl, qt, gl])
    for it in items:
        p_qk_group(it[0], it[1], w12, 0, C)
    g1_at = QK_G1_AT
    prev = None
    done1 = False
    for i in range(len(work) + 1):
        if not done1 and (i == len(work) or i >= g1_at or work[i][4] == 1):
            for it in items:
                p_qk_group(it[0], it[1], w12, 1, C)
            for it in items:
                p_qk_post(it[0], kc, vc, C)
            done1 = True
        cur = None
        if i < len(work):
            w = work[i]
            cur = [w, attn_s1(w[0], w[1], w[2], w[3], C)]
        if prev is not None:
            w = prev[0]
            attn_s2(w[0], w[1], w[2], w[3], prev[1], C)
        prev = cur


def vis_blocks(S, win, qt):
    """128-key blocks of the window (old blocks 0 .. nbo-1, then the chunk's own blocks) that hold at least one key
    visible to query tile qt. A fully masked block contributes exact zeros (P = 0) to the row sum and to PV, so
    skipping it changes no bit. Old blocks are judged only when the stream's position is static (S.st.ps)."""
    T = S.T
    B = win[1]
    local = win[5]
    nbo = (B - T) // 128
    ps = S.st.ps
    res = []
    for b in range(nbo):
        if ps is None or not ATT_SKIP:
            res.append(b)
            continue
        k0 = max(0, ps + T - B) + 128 * b
        kmax = min(k0 + 127, ps - 1)          # old keys at positions >= ps are masked
        if kmax < k0:
            continue
        if local and kmax < ps + 128 * qt - 1023:   # older than every query's window
            continue
        res.append(b)
    nown = (qt + 1) if ATT_SKIP else T // 128   # causal: own blocks after the query tile are masked
    for b in range(nown):
        res.append(nbo + b)
    return res


def attn_s1(S, win, hl, qt, C):
    """Scores of q head hl, query tile qt: masked copy + row max per key chunk (VectorE), exp + row sum (ScalarE).
    Only the visible key blocks (vis_blocks) are computed, packed in order. Returns [p bf16 [128, nv*128],
    sum [128, 1], vis]."""
    T = S.T
    B = win[1]
    bold = win[2]
    nbo = (B - T) // 128
    gl = hl // 2
    qr = S.qr
    vis = vis_blocks(S, win, qt)
    old = []
    for b in vis:
        if b < nbo:
            old.append(b)
    nown = len(vis) - len(old)
    W = len(vis) * 128
    sc = nl.ndarray((128, W), dtype=fp32, buffer=nl.sbuf)
    mx = nl.ndarray((128, 1), dtype=fp32, buffer=nl.sbuf)
    # runs of consecutive old blocks, split into score chunks of at most QKC keys
    ck0 = []
    ckw = []
    ckd = []
    for i in range(len(old)):
        b = old[i]
        n = len(ck0)
        if n > 0 and ck0[n - 1] + ckw[n - 1] == b * 128 and ckw[n - 1] + 128 <= QKC:
            ckw[n - 1] = ckw[n - 1] + 128
        else:
            ck0.append(b * 128)
            ckw.append(128)
            ckd.append(i * 128)
    ci = 0
    for k in range(len(ck0)):
        c0 = ck0[k]
        w = ckw[k]
        d0 = ckd[k]
        ps = nl.ndarray((128, QKC), dtype=fp32, buffer=nl.psum)
        for dt in range(2):
            nisa.nc_matmul(dst=ps[:, 0:w], stationary=qr[:, hl * 2 + dt, qt * 128:(qt + 1) * 128],
                           moving=S.kos[gl][:, dt, c0:c0 + w], accumulate=(dt > 0))
        cmd = nisa.reduce_cmd.reset_reduce if ci == 0 else nisa.reduce_cmd.reduce
        nisa.select_reduce(dst=sc[:, d0:d0 + w], predicate=bold[qt][:, c0:c0 + w], on_true=ps[:, 0:w],
                           on_false=NEG, reduce_res=mx, reduce_cmd=cmd, reduce_op=nl.maximum)
        ci += 1
    wo_ = nown * 128
    o0 = len(old) * 128
    psc = nl.ndarray((128, T), dtype=fp32, buffer=nl.psum)
    for dt in range(2):
        nisa.nc_matmul(dst=psc[:, 0:wo_], stationary=qr[:, hl * 2 + dt, qt * 128:(qt + 1) * 128],
                       moving=qr[:, 8 + 2 * gl + dt, 0:wo_], accumulate=(dt > 0))
    cmd = nisa.reduce_cmd.reset_reduce if ci == 0 else nisa.reduce_cmd.reduce
    nisa.select_reduce(dst=sc[:, o0:o0 + wo_], predicate=S.st.pcur[qt][:, 0:wo_], on_true=psc[:, 0:wo_], on_false=NEG,
                       reduce_res=mx, reduce_cmd=cmd, reduce_op=nl.maximum)
    nmx = nl.ndarray((128, 1), dtype=fp32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=nmx, data=mx, op0=nl.multiply, operand0=-0.0625, engine=ATT_ENG_NMX)
    p = nl.ndarray((128, W), dtype=bf16, buffer=nl.sbuf)
    sm = nl.ndarray((128, 1), dtype=fp32, buffer=nl.sbuf)
    nisa.activation(dst=p, op=nl.exp, data=sc, bias=nmx, scale=0.0625, reduce_op=nl.add, reduce_res=sm,
                    reduce_cmd=nisa.reduce_cmd.reset_reduce)
    return [p, sm, vis]


def attn_s2(S, win, hl, qt, ps1, C):
    """P^T (TensorE transposes, ScalarE evacuation), PV, normalization, output back to feature-major."""
    T = S.T
    B = win[1]
    nbo = (B - T) // 128
    gl = hl // 2
    ident = C.ident
    p = ps1[0]
    sm = ps1[1]
    vis = ps1[2]
    nb = len(vis)
    pt = nl.ndarray((128, nb, 128), dtype=bf16, buffer=nl.sbuf)
    for b0 in range(0, nb, 4):
        n = min(4, nb - b0)
        pst = nl.ndarray((128, 4, 128), dtype=fp32, buffer=nl.psum)
        for b in range(n):
            kb = b0 + b
            nisa.nc_matmul(dst=pst[:, b, :], stationary=p[:, kb * 128:(kb + 1) * 128], moving=ident)
        nisa.tensor_copy(dst=pt[:, b0:b0 + n, :], src=pst[:, 0:n, :], engine=nisa.engine.scalar)
    po = nl.ndarray((128, HD), dtype=fp32, buffer=nl.psum)
    for i in range(nb):
        kb = vis[i]
        if kb < nbo:
            mv = S.vos[gl][:, kb, :]
        else:
            mv = S.vn[:, kb - nbo, gl * HD:(gl + 1) * HD]
        nisa.nc_matmul(dst=po, stationary=pt[:, i, :], moving=mv, accumulate=(i > 0))
    rs = nl.ndarray((128, 1), dtype=fp32, buffer=nl.sbuf)
    nisa.reciprocal(dst=rs, data=sm)
    o = nl.ndarray((128, HD), dtype=bf16, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=o, data=po, op0=nl.multiply, operand0=rs, engine=ATT_ENG_O)
    pso = nl.ndarray((128, 2, 128), dtype=fp32, buffer=nl.psum)
    for dt in range(2):
        nisa.nc_matmul(dst=pso[:, dt, :], stationary=o[:, dt * 128:(dt + 1) * 128], moving=ident)
    nisa.tensor_copy(dst=bc(S.attn, 8 * T, [[4 * T, 2], [1, 128]], hl * T + qt * 128), src=pso)


def p_attn_multi(items, C, depth=2):
    """Attention of several streams; items = [[S, win], ...]. Software-pipelined: the scores and softmax of work item
    i+1 are emitted before the PV of item i, so each engine has independent work queued."""
    work = []
    for it in items:
        S = it[0]
        S.attn = nl.ndarray((128, 8, S.T), dtype=bf16, buffer=nl.sbuf)
        for hl in range(4):
            for qt in range(S.T // 128):
                work.append([S, it[1], hl, qt])
    # scores/softmax of item i are emitted depth - 1 items ahead of its PV
    pend = []
    for i in range(len(work) + depth - 1):
        if i < len(work):
            w = work[i]
            pend.append([w, attn_s1(w[0], w[1], w[2], w[3], C)])
        if i >= depth - 1:
            k = i - (depth - 1)
            w = pend[k][0]
            attn_s2(w[0], w[1], w[2], w[3], pend[k][1], C)


def p_attn(S, win, C):
    p_attn_multi([[S, win]], C)


def p_post_o(S, w1, C):
    """allreduce of the o_proj partial sums, post-attention norm, residual, pre-feedforward norm."""
    th = S.th
    if RES_PERKT:
        h1 = rmsnorm_res(S.ob, S.x, w1, 1, th, C.identf, C.ones, C.identm, S.obxw)
    else:
        on = rmsnormT(S.ob, w1, 1, 4, th, C.identf, C.ones, C.identm)
        h1 = nl.ndarray((128, KT, th), dtype=bf16, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=h1, data1=S.x, data2=on, op=nl.add)
    S.h1 = h1
    if S.defer:
        r = rmsnorm_defer(h1, w1, 2, th, C.pid, C.identf, C.ones, AG_GRP, C.identm)
        S.xn2 = r[0]
        S.rsn2 = r[1]
        return
    S.xn2 = rmsnorm_ag(h1, w1, 2, th, C.pid, C.identf, C.ones, AG_GRP, C.identm)


def p_post_d(S, w1, C):
    th = S.th
    if RES_PERKT:
        S.x = rmsnorm_res(S.ob, S.h1, w1, 3, th, C.identf, C.ones, C.identm, S.obxw)
        return
    dn = rmsnormT(S.ob, w1, 3, 4, th, C.identf, C.ones, C.identm)
    h2 = nl.ndarray((128, KT, th), dtype=bf16, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=h2, data1=S.h1, data2=dn, op=nl.add)
    S.x = h2


S_ = S
LAG_QKV = 4
LAG_O = 10
LAG_GU = 10
LAG_D = 4
RS_SPLIT_B = True   # two streams: B's o / down GEMM groups from NFULL_*B on, the other core's tokens first
RS_SPLIT_B_MAXBG = 3072   # ... only up to this global bucket (device time: -1.2..-1.7% at 512..3072, +2.9% at 3584, +3.5% at 4096)
NFULL_OB = 0
LAG_OB = 10
NFULL_DB = 8
LAG_DB = 1


def pre2s_layer(A, B_, lw, kind, C, last_layer):
    """One layer for both streams, interleaved."""
    wqkv = lw[0]
    wo = lw[1]
    wgu = lw[2]
    wd = lw[3]
    kc = lw[6]
    vc = lw[7]
    T = A.T
    pid = C.pid
    sm = layer_small(lw[4], lw[5])
    w1 = sm[0]
    w12 = sm[1]
    if kind == 'G':
        if A.st.G is None:
            A.st.G = A.st.window(A.st.p_f, A.st.BG, False, A.st.cosg, A.st.sing)
            B_.st.G = B_.st.window(B_.st.p_f, B_.st.BG, False, B_.st.cosg, B_.st.sing)
        winA = A.st.G
        winB = B_.st.G
    else:
        winA = A.st.L
        winB = B_.st.L
    if A.xn is None:
        p_norm1(A, w1, C)
    p_norm1(B_, w1, C)
    A.qkv = nl.ndarray((128, 16, T), dtype=bf16, buffer=nl.sbuf)
    B_.qkv = nl.ndarray((128, 16, T), dtype=bf16, buffer=nl.sbuf)
    p_kv_load(A, winA, kc, vc, C)
    j = Job(wqkv[pid], A.xn, B_.xn, 16, KT, T, A.qkv, B_.qkv, 2, C.ring, LAG_QKV, nisa.engine.scalar, False)
    if QK_INGEMM2:
        A.win = winA
        B_.win = winB
        for S2 in [A, B_]:
            S2.qn = nl.ndarray((128, 12, T), dtype=bf16, buffer=nl.sbuf)
            S2.qr = nl.ndarray((128, 12, T), dtype=bf16, buffer=nl.sbuf)
        j.aA.qk = [A, w12, C]
        j.aB.qk = [B_, w12, C]
        j.aA.rs = A.rsn1
        j.aB.rs = B_.rsn1
        j.lead()
        qk_flush(j.aA)
        p_qk_rest(A, winA, kc, vc, C)
        j.tail()
        qk_flush(j.aB)
        p_kv_load(B_, winB, kc, vc, C)
        p_qk_rest(B_, winB, kc, vc, C)
    else:
        j.lead()
        p_qk(A, winA, w12, kc, vc, C)
        j.tail()
        p_kv_load(B_, winB, kc, vc, C)
        p_qk(B_, winB, w12, kc, vc, C)
    p_attn_multi([[A, winA], [B_, winB]], C)
    A.part = nl.ndarray((128, KT, T), dtype=fp32, buffer=nl.sbuf)
    B_.part = nl.ndarray((128, KT, T), dtype=fp32, buffer=nl.sbuf)
    A.ob = nl.ndarray((128, KT, A.th), dtype=bf16, buffer=nl.sbuf)
    B_.ob = nl.ndarray((128, KT, B_.th), dtype=bf16, buffer=nl.sbuf)
    arA = AR(A.ob, pid, A.th, w1, 1)
    arB = AR(B_.ob, pid, B_.th, w1, 1)
    A.obxw = arA.xw
    B_.obxw = arB.xw
    j = Job(wo[pid], A.attn, B_.attn, KT, 8, T, A.part, B_.part, 2, C.ring, LAG_O, nisa.engine.scalar, False,
            arA, arB)
    if RS_SPLIT_B and A.st.BG <= RS_SPLIT_B_MAXBG:
        j.split_b(NFULL_OB, LAG_OB)
    j.lead()
    p_post_o(A, w1, C)
    j.tail()
    p_post_o(B_, w1, C)
    A.act = nl.ndarray((128, ICT, T), dtype=bf16, buffer=nl.sbuf)
    B_.act = nl.ndarray((128, ICT, T), dtype=bf16, buffer=nl.sbuf)
    j = Job(wgu[pid], A.xn2, B_.xn2, 2 * ICT, KT, T, A.act, B_.act, 2, C.ring, LAG_GU, None, True)
    j.aA.rs = A.rsn2
    j.aB.rs = B_.rsn2
    j.lead()
    j.tail()
    A.part = nl.ndarray((128, KT, T), dtype=fp32, buffer=nl.sbuf)
    B_.part = nl.ndarray((128, KT, T), dtype=fp32, buffer=nl.sbuf)
    A.ob = nl.ndarray((128, KT, A.th), dtype=bf16, buffer=nl.sbuf)
    B_.ob = nl.ndarray((128, KT, B_.th), dtype=bf16, buffer=nl.sbuf)
    arA = AR(A.ob, pid, A.th, w1, 3)
    arB = AR(B_.ob, pid, B_.th, w1, 3)
    A.obxw = arA.xw
    B_.obxw = arB.xw
    j = Job(wd[pid], A.act, B_.act, KT, ICT, T, A.part, B_.part, 2, C.ring, LAG_D, nisa.engine.scalar, False,
            arA, arB)
    if RS_SPLIT_B and A.st.BG <= RS_SPLIT_B_MAXBG:
        j.split_b(NFULL_DB, LAG_DB)
    j.lead()
    p_post_d(A, w1, C)
    A.xn = None
    j.tail()
    p_post_d(B_, w1, C)


def kv_only_qkv(S_list, wq, C):
    """K/V part of the qkv GEMM (n-tiles 8..15: this core's 2 KV heads' k and v) for each stream of S_list, one
    pass over those weight slabs; per tile the same matmuls, accumulation order and evacuation as the full GEMM."""
    NK = KT
    ntiles = 8 * NK
    off = 8 * NK
    SL = C.ring.sl
    accs = []
    for S2 in S_list:
        accs.append(GAcc())
    for s0 in range(0, ntiles, SL):
        n = min(SL, ntiles - s0)
        wsb = C.ring.load(wq[:, (off + s0) * 128:(off + s0 + n) * 128], n)
        for si in range(len(S_list)):
            S2 = S_list[si]
            x = S2.xn
            for j in range(n):
                i = s0 + j
                nt = 8 + i // NK
                kt = i % NK
                g0 = (nt // 2) * 2
                acc = accs[si]
                if kt == 0 and nt == g0:
                    acc.ps = nl.ndarray((128, 2, S2.T), dtype=fp32, buffer=nl.psum)
                ps = acc.ps
                nisa.nc_matmul(dst=ps[:, nt - g0, :], stationary=wsb[:, j * 128:(j + 1) * 128], moving=x[:, kt, :])
                if kt == NK - 1 and nt == g0 + 1:
                    nisa.tensor_copy(dst=S2.qkv[:, g0:g0 + 2, :], src=ps, engine=nisa.engine.scalar)


def kv_only_rest(S, win, w12, kc, vc, C):
    """k norms + RoPE of this core's 2 KV heads, V to token-major, cache writes (same arithmetic as the full
    layer's qk_norm_finish / rope_heads / p_qk_pre_v / p_qk_post for those tiles)."""
    T = S.T
    S.win = win
    for h in [4, 5]:
        pend = [h, gram_rstd_a(S.qkv, 2 * h, 2, T, HD, C.identf, C.ones)]
        rh = nl.ndarray((128, T), dtype=fp32, buffer=nl.sbuf)
        gram_rstd_b(pend[1], C.ones, rh)
        for dt in range(2):
            t = 2 * h + dt
            nisa.scalar_tensor_tensor(dst=S.qn[:, t, :], data=S.qkv[:, t, :], op0=nl.multiply,
                                      operand0=w12[:, t:t + 1], op1=nl.multiply, operand1=rh)
        rope_heads(S, win, h, 1)
    p_qk_pre_v(S, C)
    p_qk_post(S, kc, vc, C)


def pre_layer_kv(streams, lw, C):
    """Last layer of a non-final prefill chunk: only its K/V (for the cache) are needed by later chunks; the
    hidden-state output of these tokens is never used, so attention, o_proj and the MLP are skipped. Local layer."""
    kc = lw[6]
    vc = lw[7]
    S0 = streams[0]
    T = S0.T
    sm = layer_small(lw[4], lw[5])
    w1 = sm[0]
    w12 = sm[1]
    for S2 in streams:
        S2.defer = False     # kv_only_qkv evacuates without a deferred rstd
        p_norm1(S2, w1, C)
        S2.qkv = nl.ndarray((128, 16, T), dtype=bf16, buffer=nl.sbuf)
        S2.qn = nl.ndarray((128, 12, T), dtype=bf16, buffer=nl.sbuf)
        S2.qr = nl.ndarray((128, 12, T), dtype=bf16, buffer=nl.sbuf)
    kv_only_qkv(streams, lw[0][C.pid], C)
    for S2 in streams:
        kv_only_rest(S2, S2.st.L, w12, kc, vc, C)


def embed_tiles(toks, emb, preg, t0, ident):
    """Embedding of chunk tokens t0 .. t0 + 127 as feature-major tiles [128, KT, 128] (same ops as embed_pre_body)."""
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
    return hT


def chunk_pos(BG, T2):
    """The chunk's start position when the NEFF's global bucket fixes it: the host picks the smallest bucket
    >= (c + 1) * T2 from PRE_GLOBAL_B, which is exactly (c + 1) * T2 for buckets up to 4096 (every 512 there; T2 =
    256 only for the single chunk at 0). Larger buckets serve several chunks: None."""
    if T2 == 256:
        return 0 if BG == 256 else None
    if T2 <= BG <= 4096 and BG % T2 == 0:
        return BG - T2
    return None


def pre2s_group_body(h, pos, cosl, sinl, cosg, sing, lws, kinds, BL, BG, T2, kvlast=False, toks=None, emb=None,
                     head=None):
    """Prefill layers for a chunk of T2 tokens (two streams of T2/2) at positions pos .. pos + T2 - 1.
    head = [col, tpos, pnext, lnf, embT, lmask, lids] (last chunk of a prompt): then the chunk's column col (the
    prompt's last token, at position tpos) goes through the last layer (lws[-1], decode arithmetic, as hl33) and the
    final norm, LM head and argmax, which write toks[tpos + 1]."""
    pid = nl.program_id(0)
    set_two_stream(True)
    T = T2 // 2
    ring = WRing(RING_P2S, 16)
    C = Consts(pid, ring)
    p0 = chunk_pos(BG, T2)
    stA = PStream(pos, T, BL, BG, kinds, cosl, sinl, cosg, sing, 0, True, True, p0)
    stB = PStream(pos, T, BL, BG, kinds, cosl, sinl, cosg, sing, T, True, True, p0)
    A = SState(stA, T)
    B_ = SState(stB, T)
    th = T // 2
    if emb is not None:
        A.x = embed_tiles(toks, emb, stA.preg, pid * th, C.ident)
        B_.x = embed_tiles(toks, emb, stA.preg, T + pid * th, C.ident)
    else:
        A.x = nl.ndarray((128, KT, th), dtype=bf16, buffer=nl.sbuf)
        nisa.dma_copy(dst=A.x, src=h.ap(pattern=[[T2, 128], [128 * T2, KT], [1, th]], offset=pid * th))
        B_.x = nl.ndarray((128, KT, th), dtype=bf16, buffer=nl.sbuf)
        nisa.dma_copy(dst=B_.x, src=h.ap(pattern=[[T2, 128], [128 * T2, KT], [1, th]], offset=T + pid * th))
    for li in range(len(kinds)):
        if kvlast and li == len(kinds) - 1:
            pre_layer_kv([A, B_], lws[li], C)
        else:
            pre2s_layer(A, B_, lws[li], kinds[li], C, li == len(kinds) - 1)
    out = nl.ndarray((D, T2), dtype=bf16, buffer=nl.shared_hbm)
    nisa.dma_copy(dst=out.ap(pattern=[[T2, 128], [128 * T2, KT], [1, th]], offset=pid * th), src=A.x)
    nisa.dma_copy(dst=out.ap(pattern=[[T2, 128], [128 * T2, KT], [1, th]], offset=T + pid * th), src=B_.x)
    if head is None:
        return out
    # both cores' columns of out are needed: column col may belong to either core
    nisa.core_barrier(data=out, cores=(0, 1))
    return dec_group_body(out, toks, None, head[1], head[2], cosl, sinl, None, None, [lws[len(kinds) - 1]], 'L', 0,
                          False, True, head[3], head[4], head[5], head[6], head[0], T2)


RING_P2S = 16
ATT_ENG_NMX = nisa.engine.vector   # -max/16 of the scores
ATT_ENG_O = nisa.engine.vector     # PV * 1/sum (PSUM -> bf16)
QK_INGEMM = True    # single stream: q/k norms emitted inside the qkv GEMM loop
QK_ROPE_INGEMM = True   # ... and each head's RoPE right after its scaling
RES_PERKT = True    # post norms: scaling and residual add per k-tile
QK_INGEMM2 = True   # two streams: the same, inside the shared qkv GEMM
DEFER_NORM = False   # input / pre-feedforward norms: GEMM on bf16(x*(1+w)), 1/rms applied at the evacuation
ATT_DEPTH1 = 4      # attention software-pipeline depth, single stream (two streams: 2)
QK_SPLIT = False    # single stream: q/k norms per KV group, interleaved with attention
QK_G1_AT = 0       # attention work item before which KV group 1's norms are emitted
AG_GRP = 4          # k-tiles per all-gather piece of a norm output
RS_FIRST1 = True    # single stream: o / down GEMMs compute the other core's tokens first (gemm_rs)
NFULL_O1 = 0
NFULL_D1 = 8
LAG_O1 = 10         # ... o_proj: own tokens after all 10 groups (the 10 o slabs stay in the 12-slot ring)
LAG_D1 = 1          # ... down: own tokens one group (5 slabs) behind
GU_SPLIT1 = True    # single stream: gate/up GEMM on this core's tokens first (gemm_gu_split)
LAG_GU1 = 2
NSPLIT_GU1 = 2
DEFER_NORM1 = False  # single stream: deferred input / pre-feedforward norms (not bitwise)
ATT_SKIP = True     # attention: skip 128-key blocks that are fully masked for a query tile


def pre1s_layer(A, lw, kind, C):
    """One layer for a single stream (same phases as pre2s_layer)."""
    T = A.T
    pid = C.pid
    kc = lw[6]
    vc = lw[7]
    sm = layer_small(lw[4], lw[5])
    w1 = sm[0]
    w12 = sm[1]
    if kind == 'G':
        if A.st.G is None:
            A.st.G = A.st.window(A.st.p_f, A.st.BG, False, A.st.cosg, A.st.sing)
        win = A.st.G
    else:
        win = A.st.L
    if A.xn is None:
        p_norm1(A, w1, C)
    A.qkv = nl.ndarray((128, 16, T), dtype=bf16, buffer=nl.sbuf)
    p_kv_load(A, win, kc, vc, C)
    if QK_INGEMM:
        A.win = win
        qkv_gemm_norm(lw[0][pid], A, w12, C.ring, nisa.engine.scalar, C)
        p_qk_rest(A, win, kc, vc, C)
        p_attn_multi([[A, win]], C, ATT_DEPTH1)
    elif QK_SPLIT:
        gemm(lw[0][pid], A.xn, 16, KT, T, A.qkv, 2, 16, C.ring, nisa.engine.scalar)
        p_qk_attn([[A, win]], w12, kc, vc, C)
    else:
        gemm(lw[0][pid], A.xn, 16, KT, T, A.qkv, 2, 16, C.ring, nisa.engine.scalar)
        p_qk(A, win, w12, kc, vc, C)
        p_attn_multi([[A, win]], C)
    A.part = nl.ndarray((128, KT, T), dtype=fp32, buffer=nl.sbuf)
    A.ob = nl.ndarray((128, KT, A.th), dtype=bf16, buffer=nl.sbuf)
    ar = AR(A.ob, pid, A.th, w1, 1)
    A.obxw = ar.xw
    if RS_FIRST1:
        gemm_rs(lw[1][pid], A.attn, KT, 8, 2, C.ring, nisa.engine.scalar, ar, LAG_O1, NFULL_O1, A.part)
    else:
        gemm(lw[1][pid], A.attn, KT, 8, T, A.part, 2, 16, C.ring, nisa.engine.scalar, ar)
    p_post_o(A, w1, C)
    A.act = nl.ndarray((128, ICT, T), dtype=bf16, buffer=nl.sbuf)
    if GU_SPLIT1:
        gemm_gu_split(lw[2][pid], A.xn2, A.act, C.ring, pid, A.th, LAG_GU1, A.rsn2, NSPLIT_GU1)
    else:
        gemm_gu(lw[2][pid], A.xn2, T, A.act, C.ring, A.rsn2)
    A.part = nl.ndarray((128, KT, T), dtype=fp32, buffer=nl.sbuf)
    A.ob = nl.ndarray((128, KT, A.th), dtype=bf16, buffer=nl.sbuf)
    ar = AR(A.ob, pid, A.th, w1, 3)
    A.obxw = ar.xw
    if RS_FIRST1:
        gemm_rs(lw[3][pid], A.act, KT, ICT, 2, C.ring, nisa.engine.scalar, ar, LAG_D1, NFULL_D1, A.part)
    else:
        gemm(lw[3][pid], A.act, KT, ICT, T, A.part, 2, 16, C.ring, nisa.engine.scalar, ar)
    p_post_d(A, w1, C)
    A.xn = None


def pre1s_group_body(h, pos, cosl, sinl, cosg, sing, lws, kinds, BL, BG, T, kvlast=False, toks=None, emb=None,
                     head=None):
    """Prefill layers for one chunk of T tokens (single stream). head: as in pre2s_group_body."""
    pid = nl.program_id(0)
    set_two_stream(False)
    ring = WRing(RING_PRE, 16)
    C = Consts(pid, ring)
    st = PStream(pos, T, BL, BG, kinds, cosl, sinl, cosg, sing, 0, True, True, chunk_pos(BG, T))
    A = SState(st, T)
    A.defer = DEFER_NORM1
    th = T // 2
    if emb is not None:
        A.x = embed_tiles(toks, emb, st.preg, pid * th, C.ident)
    else:
        A.x = nl.ndarray((128, KT, th), dtype=bf16, buffer=nl.sbuf)
        nisa.dma_copy(dst=A.x, src=h.ap(pattern=[[T, 128], [128 * T, KT], [1, th]], offset=pid * th))
    for li in range(len(kinds)):
        if kvlast and li == len(kinds) - 1:
            pre_layer_kv([A], lws[li], C)
        else:
            pre1s_layer(A, lws[li], kinds[li], C)
    out = nl.ndarray((D, T), dtype=bf16, buffer=nl.shared_hbm)
    nisa.dma_copy(dst=out.ap(pattern=[[T, 128], [128 * T, KT], [1, th]], offset=pid * th), src=A.x)
    if head is None:
        return out
    nisa.core_barrier(data=out, cores=(0, 1))
    return dec_group_body(out, toks, None, head[1], head[2], cosl, sinl, None, None, [lws[len(kinds) - 1]], 'L', 0,
                          False, True, head[3], head[4], head[5], head[6], head[0], T)
