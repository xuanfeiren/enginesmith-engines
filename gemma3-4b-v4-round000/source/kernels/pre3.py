"""Step-scheduled two-stream prefill. A chunk of 2T tokens is two streams (A = first T tokens, B = the next T) with
their own weight streams (one SBUF ring each). Stream B runs half a layer behind A: while A is in its
norm/attention half of layer l (latency-bound: vector/scalar chains between short TensorE bursts), B is in the MLP
half of layer l-1 (TensorE-bound). Each half-layer is a list of small steps (one weight slab, one attention work
item, one norm, ...); the two lists are merged by estimated TensorE time, so the TensorE always has independent
work queued. Per-stream arithmetic is that of pre2s.py (identical results).
"""
import nki
import nki.language as nl
import nki.isa as nisa
from gemma import *
from pre2s import *


class St3(nl.NKIObject):
    """One stream: activations (SState), its own weight ring and GEMM accumulator, current layer state."""

    def __init__(self, st, T, nbuf, pid):
        self.S = SState(st, T)
        self.T = T
        self.ring = WRing(nbuf, 16)
        self.acc = GAcc()
        self.pid = pid
        self.lw = None
        self.kind = None
        self.w1 = None
        self.w12 = None
        self.win = None
        self.work = None
        self.prev = None
        self.ar = None


def first_half(Z):
    """Steps of the norm/attention half of a layer: [code, index, est. TensorE us]."""
    T = Z.T
    steps = [['LW', 0, 0.3], ['N1', 0, 4.0]]
    for i in range(20):
        steps.append(['Q', i, 1.8])
    steps.append(['K', 0, 5.0])
    nitems = 4 * (T // 128)
    for i in range(nitems + 1):
        steps.append(['A', i, 3.7])
    for i in range(10):
        steps.append(['O', i, 1.8])
    steps.append(['P', 0, 8.0])
    return steps


def second_half(Z):
    steps = []
    for i in range(100):
        steps.append(['G', i, 1.8])
    for i in range(50):
        steps.append(['D', i, 1.8])
    steps.append(['E', 0, 4.0])
    return steps


def slab_load(Z, w, i):
    SL = Z.ring.sl
    return Z.ring.load(w[:, i * SL * 128:(i + 1) * SL * 128], SL)


def run_step(Z, step, C, lw, kind):
    S = Z.S
    T = Z.T
    pid = Z.pid
    code = step[0]
    i = step[1]
    if code == 'LW':
        Z.lw = lw
        Z.kind = kind
        sm = layer_small(lw[4], lw[5])
        Z.w1 = sm[0]
        Z.w12 = sm[1]
        if kind == 'G':
            if S.st.G is None:
                S.st.G = S.st.window(S.st.p_f, S.st.BG, False, S.st.cosg, S.st.sing)
            Z.win = S.st.G
        else:
            Z.win = S.st.L
        p_kv_load(S, Z.win, lw[6], lw[7], C)
        S.qkv = nl.ndarray((128, 16, T), dtype=bf16, buffer=nl.sbuf)
    elif code == 'N1':
        p_norm1(S, Z.w1, C)
    elif code == 'Q':
        wsb = slab_load(Z, Z.lw[0][pid], i)
        mm_slab(Z.acc, wsb, i * 16, 16, KT, 16, T, S.xn, S.qkv, 2, nisa.engine.scalar)
    elif code == 'K':
        p_qk(S, Z.win, Z.w12, Z.lw[6], Z.lw[7], C)
        S.attn = nl.ndarray((128, 8, T), dtype=bf16, buffer=nl.sbuf)
        Z.work = []
        for hl in range(4):
            for qt in range(T // 128):
                Z.work.append([hl, qt])
        Z.prev = None
    elif code == 'A':
        cur = None
        if i < len(Z.work):
            w = Z.work[i]
            cur = [w, attn_s1(S, Z.win, w[0], w[1], C)]
        if Z.prev is not None:
            w = Z.prev[0]
            attn_s2(S, Z.win, w[0], w[1], Z.prev[1], C)
        Z.prev = cur
        if i == len(Z.work):
            S.part = nl.ndarray((128, KT, T), dtype=fp32, buffer=nl.sbuf)
            S.ob = nl.ndarray((128, KT, T), dtype=bf16, buffer=nl.sbuf)
            Z.ar = AR(S.ob, pid)
    elif code == 'O':
        wsb = slab_load(Z, Z.lw[1][pid], i)
        mm_slab(Z.acc, wsb, i * 16, 16, 8, KT, T, S.attn, S.part, 2, nisa.engine.scalar, Z.ar)
    elif code == 'P':
        p_post_o(S, Z.w1, C)
        S.act = nl.ndarray((128, ICT, T), dtype=bf16, buffer=nl.sbuf)
    elif code == 'G':
        wsb = slab_load(Z, Z.lw[2][pid], i)
        gu_slab(Z.acc, wsb, i * 16, 16, T, S.xn2, S.act)
        if i == 99:
            S.part = nl.ndarray((128, KT, T), dtype=fp32, buffer=nl.sbuf)
            S.ob = nl.ndarray((128, KT, T), dtype=bf16, buffer=nl.sbuf)
            Z.ar = AR(S.ob, pid)
    elif code == 'D':
        wsb = slab_load(Z, Z.lw[3][pid], i)
        mm_slab(Z.acc, wsb, i * 16, 16, ICT, KT, T, S.act, S.part, 2, nisa.engine.scalar, Z.ar)
    elif code == 'E':
        p_post_d(S, Z.w1, C)
        S.xn = None


def merge_run(ZA, LA, lwA, kA, ZB, LB, lwB, kB, C):
    """Emit the steps of both lists, interleaved by cumulative estimated TensorE time."""
    ca = 0.0
    cb = 0.0
    for s in LA:
        ca += s[2]
    for s in LB:
        cb += s[2]
    ia = 0
    ib = 0
    da = 0.0
    db = 0.0
    for _ in range(len(LA) + len(LB)):
        takeA = False
        if ib >= len(LB):
            takeA = True
        elif ia < len(LA):
            fa = (da + LA[ia][2] * 0.5) / ca
            fb = (db + LB[ib][2] * 0.5) / cb
            takeA = fa <= fb
        if takeA:
            run_step(ZA, LA[ia], C, lwA, kA)
            da += LA[ia][2]
            ia += 1
        else:
            run_step(ZB, LB[ib], C, lwB, kB)
            db += LB[ib][2]
            ib += 1


def pre3_group_body(h, pos, cosl, sinl, cosg, sing, lws, kinds, BL, BG, T2):
    """Prefill layers for a chunk of T2 tokens (two streams of T2/2) at positions pos .. pos + T2 - 1."""
    pid = nl.program_id(0)
    T = T2 // 2
    C = Consts(pid, None)
    stA = PStream(pos, T, BL, BG, kinds, cosl, sinl, cosg, sing, 0, True, True)
    stB = PStream(pos, T, BL, BG, kinds, cosl, sinl, cosg, sing, T, True, True)
    ZA = St3(stA, T, RING_P3, pid)
    ZB = St3(stB, T, RING_P3, pid)
    ZA.S.x = nl.ndarray((128, KT, T), dtype=bf16, buffer=nl.sbuf)
    nisa.dma_copy(dst=ZA.S.x, src=h.ap(pattern=[[T2, 128], [128 * T2, KT], [1, T]], offset=0))
    ZB.S.x = nl.ndarray((128, KT, T), dtype=bf16, buffer=nl.sbuf)
    nisa.dma_copy(dst=ZB.S.x, src=h.ap(pattern=[[T2, 128], [128 * T2, KT], [1, T]], offset=T))
    nl_ = len(kinds)
    # slot 0: A first half of layer 0 alone
    merge_run(ZA, first_half(ZA), lws[0], kinds[0], ZB, [], None, None, C)
    for li in range(nl_):
        # A second half of layer li with B first half of layer li
        merge_run(ZA, second_half(ZA), lws[li], kinds[li], ZB, first_half(ZB), lws[li], kinds[li], C)
        # A first half of layer li+1 with B second half of layer li
        if li + 1 < nl_:
            merge_run(ZA, first_half(ZA), lws[li + 1], kinds[li + 1], ZB, second_half(ZB), lws[li], kinds[li], C)
        else:
            merge_run(ZA, [], None, None, ZB, second_half(ZB), lws[li], kinds[li], C)
    out = nl.ndarray((D, T2), dtype=bf16, buffer=nl.shared_hbm)
    half = KT // 2
    nisa.dma_copy(dst=out.ap(pattern=[[T2, 128], [128 * T2, half], [1, T]], offset=half * pid * 128 * T2),
                  src=ZA.S.x[:, half * pid:half * (pid + 1), :])
    nisa.dma_copy(dst=out.ap(pattern=[[T2, 128], [128 * T2, half], [1, T]], offset=half * pid * 128 * T2 + T),
                  src=ZB.S.x[:, half * pid:half * (pid + 1), :])
    return out


RING_P3 = 6
