"""Round 007 (SP): sequence-parallel residual stream for the prefill layer NEFF (prompt buckets with 512-token blocks).

The per-block all-reduces of prefill_tp (o_proj and down partials, bf16 over the 8 ranks) become reduce-scatters
over token shards plus all-gathers of the normalised activations:

  token t of block b (512 tokens) belongs to rank r = (t % 512) // 64; inside the rank, physical core pid owns the
  32 tokens [pid * 32, pid * 32 + 32) of the rank's 64. Only the owner keeps the residual h of its tokens up to date
  (in s_h, natural column layout; other columns go stale).

  norm1: each core normalises its 32 tokens -> xa[b] (the rank's 64-token shard, bf16) -> all_gather -> xg[b] =
         the whole block's x1 on every rank (QKV unchanged)
  o_proj: core pid computes hidden tiles [20 pid, 20 pid + 20) for all 512 tokens (N-split as before) into
         hb_a[b] [8 shards, 128, HT, 64] -> reduce_scatter -> ra[b] = the rank's shard, all hidden tiles
  norm2: h += ra on the own 32 tokens (written back), normalised -> xb[b] -> all_gather -> x2g[b]
  down:  -> hb_m[b] -> reduce_scatter -> rm[b]; the layer's tail adds rm to the own 32 tokens of h.

Per element the arithmetic is prefill_tp's (same bf16 partial sums, same fp32 residual adds, same norm ops); the
rank sums are those of the reduce-scatter instead of the all-reduce. The norms run on 1/16 of the tokens per core
instead of all of them on both cores, and the residual finishes move 1/16 of the bytes.
"""
import nki
import nki.language as nl
import nki.isa as nisa
import nki.collectives as ncc
from layer_decode import EPS, D
from nkilib.core.attention.attention_cte import _attention_cte

SCALE = 0.08838834764831845


def _hd(dst, src):
    nisa.dma_copy(dst=dst, src=src, dge_mode=nisa.dge_mode.hwdge, engine=nisa.engine.sync)


def _slab(W, nt, KT):
    t = nl.ndarray((128, KT * 128), dtype=W.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=t, src=W[:, nl.ds(nt * KT * 128, KT * 128)], dge_mode=nisa.dge_mode.hwdge,
                  engine=nisa.engine.sync)
    return t


def _hview(h, blk, pid, rreg, HT, S, W, TS, HC):
    """h[:, :, blk * TB + rank * TS + pid * HC : + HC] (the own tokens of this core) as a DMA access pattern;
    rreg holds 2 * rank + pid"""
    NB = S // (W * TS)
    return h.reshape((128, HT, NB, 2 * W, HC)).ap(pattern=[[HT * S, 128], [S, HT], [1, HC]],
                                                    offset=blk * W * TS, scalar_offset=rreg, indirect_dim=3)


def _norm_own(hc, n, w_sb, onesb, HT, xo):
    """RMSNorm of hc [128, HT, n] fp32 (SBUF) into xo [128, HT, n] bf16: prefill_tp._norm_blk's instructions"""
    sq = nl.ndarray((128, HT, n), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.activation(dst=sq, op=nl.square, data=hc)
    ss = nl.ndarray((128, n), dtype=nl.float32, buffer=nl.psum)
    for kt in range(HT):
        nisa.nc_matmul(dst=ss, stationary=onesb, moving=sq[:, kt, :])
    rstd = nl.ndarray((128, n), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=rstd, data=ss, op0=nl.multiply, operand0=1.0 / (HT * 128), op1=nl.add, operand1=EPS)
    nisa.tensor_scalar(dst=rstd, data=rstd, op0=nl.rsqrt, operand0=0.0, engine=nisa.engine.gpsimd)
    for kt in range(HT):
        nisa.scalar_tensor_tensor(dst=xo[:, kt, :], data=hc[:, kt, :], op0=nl.multiply,
                                  operand0=w_sb[:, kt:kt + 1], op1=nl.multiply, operand1=rstd)


SPSN = [1]   # round 008 (SPS): every SP collective split into SPSN[0] parts over hidden tiles (see _rs_part)
GCHK = [1]   # round 008 (GCH): the gathered block loaded in GCHK[0] DMAs of consecutive hidden tiles
MPK = [0]    # round 009 (MP): unpadded MLP copies wgp / wdp (25 gate/up slabs per core: 12 + 12 + one packed
             # [gate half | up half] slab; down over 12 + 12 own / other tiles + one combo tile of both cores' halves)
SOCK = [1]   # round 008 (SOC): o_proj / down outputs written to the reduce-scatter buffer in SOCK[0] chunks


def _gathered(xg, blk, HT, W, HC):
    """the all-gathered block xg[blk] [2W, 128, HT, HC] -> SBUF [128, 2W, HT * HC] in one DMA (GCH: GCHK[0] DMAs of
    consecutive hidden tiles, so the first matmuls can start on the first chunk); the matmul moving operand of hidden
    tile kt is x[:, :, kt * HC:(kt + 1) * HC] (slot q, token j = token q * HC + j of the block)"""
    Q = 2 * W
    x = nl.ndarray((128, Q, HT * HC), dtype=nl.bfloat16, buffer=nl.sbuf)
    NC = GCHK[0]
    if SPSN[0] > 1:   # SPS: part g = hidden tiles [g HP, g HP + HP) in its own tensor xg[blk][g]; NC chunks in all
        NS = SPSN[0]
        HP = HT // NS
        CP = NC // NS
        if CP < 1:
            CP = 1
        for g in range(NS):
            for c in range(CP):
                k0 = (c * HP) // CP
                k1 = ((c + 1) * HP) // CP
                _hd(x[:, :, (g * HP + k0) * HC:(g * HP + k1) * HC],
                    xg[blk][g].ap(pattern=[[HP * HC, 128], [128 * HP * HC, Q], [1, (k1 - k0) * HC]], offset=k0 * HC))
        return x
    for c in range(NC):
        k0 = (c * HT) // NC
        k1 = ((c + 1) * HT) // NC
        _hd(x[:, :, k0 * HC:k1 * HC], xg[blk].ap(pattern=[[HT * HC, 128], [128 * HT * HC, Q], [1, (k1 - k0) * HC]],
                                                    offset=k0 * HC))
    return x


def _scatter_out(po, hb, blk, pid, HH, HT, W, HC, j0=0, j1=-1):
    """po [128, 2W, HH * HC] (this core's hidden tiles, all tokens, slot-major) -> hb[blk, q, :, own tiles, :] for
    every slot q, in one DMA (own tiles [j0, j1) only: SOC)"""
    Q = 2 * W
    if j1 < 0:
        j1 = HH
    _hd(hb[blk].ap(pattern=[[HT * HC, 128], [128 * HT * HC, Q], [1, (j1 - j0) * HC]], offset=(pid * HH + j0) * HC),
        po[:, :, j0 * HC:j1 * HC])


def _sps_flush(po, hb, ro, blk, pid, HH, HT, W, HC, j, rg):
    """SPS: after own tile j: write the SOC chunk it completes into its part's buffer; when it completes part g, barrier
    + reduce-scatter of part g (hb / ro = [blk][part])"""
    NS = SPSN[0]
    Q = 2 * W
    for g in range(NS):
        j0, j1 = _rs_tiles(HH, g)
        if j0 <= j < j1:
            n = j1 - j0
            NC = SOCK[0] // NS
            if NC < 1:
                NC = 1
            for c in range(NC):
                a = j0 + (c * n) // NC
                b = j0 + ((c + 1) * n) // NC
                if j + 1 == b:
                    _hd(hb[blk][g].ap(pattern=[[(HT // NS) * HC, 128], [128 * (HT // NS) * HC, Q], [1, (b - a) * HC]],
                                      offset=(pid * n + a - j0) * HC), po[:, :, a * HC:b * HC])
            if j + 1 == j1:
                nisa.core_barrier(data=hb[blk][g], cores=(0, 1))
                ncc.reduce_scatter(srcs=[hb[blk][g]], dsts=[ro[blk][g]], op=nl.add, replica_group=rg, collective_dim=0)


def _soc_flush(po, hb, blk, pid, HH, HT, W, HC, j):
    """SOC: after own tile j's output, write the chunk it completes"""
    NC = SOCK[0]
    for c in range(NC):
        if j + 1 == ((c + 1) * HH) // NC:
            _scatter_out(po, hb, blk, pid, HH, HT, W, HC, (c * HH) // NC, j + 1)


def _rs_tiles(HH, g):
    """SPS: own tiles [j0, j1) of each core in reduce-scatter part g (part g holds, per core c, the core's own tiles
    j0 .. j1 at local hidden index c * (j1 - j0) + (j - j0))"""
    NS = SPSN[0]
    return (g * HH) // NS, ((g + 1) * HH) // NS


def _load_rs(r, rb, pid, HT, HC):
    """SPS: the own-token shard of a split reduce-scatter result (rb = [part][2, 128, HT / NS, HC]) -> r [128, HT, HC]
    in natural hidden order"""
    NS = SPSN[0]
    HH = HT // 2
    for g in range(NS):
        j0, j1 = _rs_tiles(HH, g)
        n = j1 - j0
        for c in range(2):
            _hd(r[:, c * HH + j0:c * HH + j1, :], rb[g][pid][:, c * n:(c + 1) * n, :])


def _norm_own_ag(hc, n, w_sb, onesb, HT, xo, xa_b, xg_b, pid, rg):
    """SPS: _norm_own with the bf16 output written and all-gathered in SPSN[0] parts of consecutive hidden tiles
    (part g issued as soon as its tiles are normalised); the same instructions per element"""
    NS = SPSN[0]
    sq = nl.ndarray((128, HT, n), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.activation(dst=sq, op=nl.square, data=hc)
    ss = nl.ndarray((128, n), dtype=nl.float32, buffer=nl.psum)
    for kt in range(HT):
        nisa.nc_matmul(dst=ss, stationary=onesb, moving=sq[:, kt, :])
    rstd = nl.ndarray((128, n), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=rstd, data=ss, op0=nl.multiply, operand0=1.0 / (HT * 128), op1=nl.add, operand1=EPS)
    nisa.tensor_scalar(dst=rstd, data=rstd, op0=nl.rsqrt, operand0=0.0, engine=nisa.engine.gpsimd)
    HP = HT // NS
    for g in range(NS):
        for kt in range(g * HP, (g + 1) * HP):
            nisa.scalar_tensor_tensor(dst=xo[:, kt, :], data=hc[:, kt, :], op0=nl.multiply,
                                      operand0=w_sb[:, kt:kt + 1], op1=nl.multiply, operand1=rstd)
        _hd(xa_b[g][pid], xo[:, g * HP:(g + 1) * HP, :])
        nisa.core_barrier(data=xa_b[g], cores=(0, 1))
        ncc.all_gather(srcs=[xa_b[g]], dsts=[xg_b[g]], replica_group=rg, collective_dim=0)


def _norm1(h, blk, pid, rreg, HT, S, W, TS, HC, w1, onesb, xa, xg, rg):
    hc = nl.ndarray((128, HT, HC), dtype=nl.float32, buffer=nl.sbuf)
    _hd(hc, _hview(h, blk, pid, rreg, HT, S, W, TS, HC))
    xo = nl.ndarray((128, HT, HC), dtype=nl.bfloat16, buffer=nl.sbuf)
    if SPSN[0] > 1:
        _norm_own_ag(hc, HC, w1, onesb, HT, xo, xa[blk], xg[blk], pid, rg)
        return
    _norm_own(hc, HC, w1, onesb, HT, xo)
    _hd(xa[blk][pid], xo)
    nisa.core_barrier(data=xa[blk], cores=(0, 1))
    ncc.all_gather(srcs=[xa[blk]], dsts=[xg[blk]], replica_group=rg, collective_dim=0)


def _norm2(h, blk, pid, rreg, HT, S, W, TS, HC, ra, w2, onesb, xb, x2g, rg):
    """h += ra on the own tokens (written back), normalised -> xb -> all_gather -> x2g"""
    bf = nl.bfloat16
    hc = nl.ndarray((128, HT, HC), dtype=nl.float32, buffer=nl.sbuf)
    _hd(hc, _hview(h, blk, pid, rreg, HT, S, W, TS, HC))
    r = nl.ndarray((128, HT, HC), dtype=bf, buffer=nl.sbuf)
    if SPSN[0] > 1:
        _load_rs(r, ra[blk], pid, HT, HC)
    else:
        _hd(r, ra[blk][pid])
    nisa.tensor_tensor(dst=hc, data1=hc, data2=r, op=nl.add)
    _hd(_hview(h, blk, pid, rreg, HT, S, W, TS, HC), hc)
    xo = nl.ndarray((128, HT, HC), dtype=bf, buffer=nl.sbuf)
    if SPSN[0] > 1:
        _norm_own_ag(hc, HC, w2, onesb, HT, xo, xb[blk], x2g[blk], pid, rg)
        return
    _norm_own(hc, HC, w2, onesb, HT, xo)
    _hd(xb[blk][pid], xo)
    nisa.core_barrier(data=xb[blk], cores=(0, 1))
    ncc.all_gather(srcs=[xb[blk]], dsts=[x2g[blk]], replica_group=rg, collective_dim=0)


def _qkv_post(l, nt, blk, t0, ps, TB, NQL, NKL, D, pid, onesb, qnc, knc, rot_sb, cs, sn, ident, s_q, s_k, s_v,
              kcache, vcache, VPERM, SPH, aq0, ak0, av0):
    """q / k norm + RoPE + cache writes of QKV tile nt (ps), or the V transpose + cache writes"""
    bf = nl.bfloat16
    if nt < NQL + NKL:
        sq = nl.ndarray((128, TB), dtype=bf, buffer=nl.sbuf)
        nisa.activation(dst=sq, op=nl.square, data=ps)
        ss = nl.ndarray((128, TB), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_matmul(dst=ss, stationary=onesb, moving=sq)
        rs = nl.ndarray((128, TB), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=rs, data=ss, op0=nl.multiply, operand0=1.0 / D, op1=nl.add, operand1=EPS)
        nisa.tensor_scalar(dst=rs, data=rs, op0=nl.rsqrt, operand0=0.0, engine=nisa.engine.gpsimd)
        yb = nl.ndarray((128, TB), dtype=bf, buffer=nl.sbuf)
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
        ro = nl.ndarray((128, TB), dtype=bf, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=ro, data1=xc, data2=xs, op=nl.add, engine=nisa.engine.gpsimd)
        if nt < NQL:
            _hd(s_q[pid * NQL + nt, :, nl.ds(t0, TB)], ro)
            if blk < SPH:
                _hd(aq0[pid * NQL + nt, :, nl.ds(t0, TB)], ro)
        else:
            j = nt - NQL
            _hd(s_k[pid * NKL + j, :, nl.ds(t0, TB)], ro)
            _hd(kcache[l, pid, j, :, nl.ds(t0, TB)], ro)
            if blk < SPH:
                _hd(ak0[pid * NKL + j, :, nl.ds(t0, TB)], ro)
    else:
        j = nt - NQL - NKL
        vb = nl.ndarray((128, TB), dtype=bf, buffer=nl.sbuf)
        nisa.activation(dst=vb, op=nl.copy, data=ps)
        vr = nl.ndarray((128, TB // 128, D), dtype=bf, buffer=nl.sbuf)
        for tb in range(TB // 128):
            tp = nl.ndarray((128, D), dtype=nl.float32, buffer=nl.psum)
            nisa.nc_matmul(dst=tp, stationary=vb[:, tb * 128:(tb + 1) * 128], moving=ident)
            nisa.tensor_copy(dst=vr[:, tb, :], src=tp)
        for tb in range(TB // 128):
            _hd(s_v[pid * NKL + j, nl.ds(t0 + tb * 128, 128), :], vr[:, tb, :])
            if blk < SPH:
                _hd(av0[pid * NKL + j, nl.ds(t0 + tb * 128, 128), :], vr[:, tb, :])
            if VPERM:
                VJ = vcache.shape[2] // 128
                NKD = vcache.shape[3]
                _hd(vcache[l, pid].ap(pattern=[[VJ * NKD, 128], [1, D]],
                                      offset=((t0 + tb * 128) // 128) * NKD + j * D), vr[:, tb, :])
            else:
                _hd(vcache[0, pid, nl.ds(t0 + tb * 128, 128), j * D:(j + 1) * D], vr[:, tb, :])


def _pre_gu(wgu_l, n, ITL, HT):
    """SPG (round 008): the first n gate / up slab pairs of an MLP block, issued now (ahead of the norm2 loads that
    wait on the reduce-scatter in the in-order Sync queue)"""
    out = []
    UO = ITL - 1 if MPK[0] else ITL   # MP: up slabs follow the 12 gate slabs
    for i in range(n):
        out.append((_slab(wgu_l, i, HT), _slab(wgu_l, UO + i, HT)))
    return out


def _oproj(l, blk, it, src_o, t_off, TB, NQL, HH, HT, W, HC, pid, wo, hb_a, ra, rg, SPO, done2,
           h, rreg, S, TS, w2, onesb, xb, x2g, SPG=0, wgu_l=None, ITL=0):
    """o_proj of block blk (attention output src_o[:, :, t_off:t_off + TB]), N-split over the cores, slot-major bf16
    sums -> hb_a[blk] -> reduce_scatter -> ra[blk]; with SPO, block it - SPO's norm2 + all-gather first"""
    bf = nl.bfloat16
    oc = nl.ndarray((128, 2 * NQL, TB), dtype=bf, buffer=nl.sbuf)
    for hh in range(2 * NQL):
        _hd(oc[:, hh, :], src_o[hh, :, nl.ds(t_off, TB)])
    po = nl.ndarray((128, 2 * W, HH * HC), dtype=bf, buffer=nl.sbuf)
    pre = None
    for j in range(HH):
        nt = pid * HH + j
        w0 = _slab(wo[l, 0], nt, NQL)
        w1_ = _slab(wo[l, 1], nt, NQL)
        ps = nl.ndarray((128, TB), dtype=nl.float32, buffer=nl.psum)
        for kt in range(NQL):
            nisa.nc_matmul(dst=ps, stationary=w0[:, kt * 128:(kt + 1) * 128], moving=oc[:, kt, :])
        for kt in range(NQL):
            nisa.nc_matmul(dst=ps, stationary=w1_[:, kt * 128:(kt + 1) * 128], moving=oc[:, NQL + kt, :])
        nisa.activation(dst=po[:, :, j * HC:(j + 1) * HC], op=nl.copy, data=ps.reshape((128, 2 * W, HC)))
        if SPSN[0] > 1:
            if SPO and it >= SPO and done2[it - SPO] == 0 and j + 1 == _rs_tiles(HH, 0)[1]:
                # SPS: block it - SPO's norm2 + all-gathers ahead of this block's first reduce-scatter part
                if SPG > 0:
                    pre = _pre_gu(wgu_l, SPG, ITL, HT)
                _norm2(h, it - SPO, pid, rreg, HT, S, W, TS, HC, ra, w2, onesb, xb, x2g, rg)
                done2[it - SPO] = 1
            _sps_flush(po, hb_a, ra, blk, pid, HH, HT, W, HC, j, rg)
        elif SOCK[0] > 1:
            _soc_flush(po, hb_a, blk, pid, HH, HT, W, HC, j)
    if SPSN[0] > 1:
        return pre
    if SPO and it >= SPO and done2[it - SPO] == 0:   # block it - SPO's norm2 + all-gather ahead of this block's
        # reduce-scatter in the collective queue
        if SPG > 0:
            pre = _pre_gu(wgu_l, SPG, ITL, HT)
        _norm2(h, it - SPO, pid, rreg, HT, S, W, TS, HC, ra, w2, onesb, xb, x2g, rg)
        done2[it - SPO] = 1
    if SOCK[0] <= 1:
        _scatter_out(po, hb_a, blk, pid, HH, HT, W, HC)
    nisa.core_barrier(data=hb_a[blk], cores=(0, 1))
    ncc.reduce_scatter(srcs=[hb_a[blk]], dsts=[ra[blk]], op=nl.add, replica_group=rg, collective_dim=0)
    return pre


def _down(l, blk, act, oth, TE, ITL, HH, HT, W, HC, pid, wd, hb_m, rm, rg):
    """down projection of MLP block blk (N-split: own hidden tiles over both cores' activations) -> hb_m[blk] ->
    reduce_scatter -> rm[blk]"""
    pm = nl.ndarray((128, 2 * W, HH * HC), dtype=nl.bfloat16, buffer=nl.sbuf)
    for j in range(HH):
        nt = pid * HH + j
        if MPK[0]:   # round 009 (MP): 12 own + 12 other tiles + the combo tile (oth = (other tiles, combo))
            NF = ITL - 1
            wm = _slab(wd[l, pid], nt, ITL)
            wx = nl.ndarray((128, ITL * 128), dtype=wd.dtype, buffer=nl.sbuf)   # (same tile shape as wm; slot 12 unused)
            nisa.dma_copy(dst=wx[:, 0:NF * 128], src=wd[l, 1 - pid][:, nl.ds(nt * ITL * 128, NF * 128)],
                          dge_mode=nisa.dge_mode.hwdge, engine=nisa.engine.sync)
            ps = nl.ndarray((128, TE), dtype=nl.float32, buffer=nl.psum)
            for kt in range(NF):
                nisa.nc_matmul(dst=ps, stationary=wm[:, kt * 128:(kt + 1) * 128], moving=act[:, kt, :])
            for kt in range(NF):
                nisa.nc_matmul(dst=ps, stationary=wx[:, kt * 128:(kt + 1) * 128], moving=oth[:, kt, :])
            nisa.nc_matmul(dst=ps, stationary=wm[:, NF * 128:(NF + 1) * 128], moving=oth[:, NF, :])
        else:
            wm = _slab(wd[l, pid], nt, ITL)
            wx = _slab(wd[l, 1 - pid], nt, ITL)
            ps = nl.ndarray((128, TE), dtype=nl.float32, buffer=nl.psum)
            for kt in range(ITL):
                nisa.nc_matmul(dst=ps, stationary=wm[:, kt * 128:(kt + 1) * 128], moving=act[:, kt, :])
            for kt in range(ITL):
                nisa.nc_matmul(dst=ps, stationary=wx[:, kt * 128:(kt + 1) * 128], moving=oth[:, kt, :])
        nisa.activation(dst=pm[:, :, j * HC:(j + 1) * HC], op=nl.copy, data=ps.reshape((128, 2 * W, HC)))
        if SPSN[0] > 1:
            _sps_flush(pm, hb_m, rm, blk, pid, HH, HT, W, HC, j, rg)
        elif SOCK[0] > 1:
            _soc_flush(pm, hb_m, blk, pid, HH, HT, W, HC, j)
    if SPSN[0] > 1:
        return
    if SOCK[0] <= 1:
        _scatter_out(pm, hb_m, blk, pid, HH, HT, W, HC)
    nisa.core_barrier(data=hb_m[blk], cores=(0, 1))
    ncc.reduce_scatter(srcs=[hb_m[blk]], dsts=[rm[blk]], op=nl.add, replica_group=rg, collective_dim=0)


def _spbuf(shape, dt, name):
    """one SP collective buffer, or (SPS) a list of SPSN[0] part tensors of HT / SPSN[0] hidden tiles each"""
    NS = SPSN[0]
    if NS <= 1:
        return nl.ndarray(shape, dtype=dt, buffer=nl.shared_hbm, name=name)
    out = []
    for g in range(NS):
        out.append(nl.ndarray((shape[0], shape[1], shape[2] // NS, shape[3]), dtype=dt, buffer=nl.shared_hbm,
                              name=f"{name}p{g}"))
    return out


def layer_sp(h, ln1, ln2, qn, kn, wqkv, wo, wgu, wd, rot, cosT, sinT, kcache, vcache, s_q, s_k, s_v, s_o,
             NQL, NKL, ITL, W, rg, onesb, ident, VPERM, LAG=-1, PN=0, TLG=0, SPO=0, SPA=0, LPN=1, SPD=0, SPH=0, SPHN=0, SPP=0, SPQ=0, GCH=1, SOC=1, SPS=1, SPG=0, KTP=0, KTQ=0, MP=0):
    SPSN[0] = SPS
    MPK[0] = MP
    assert not (MP and KTP)
    GCHK[0] = GCH
    SOCK[0] = SOC
    S = h.shape[2]
    HT = h.shape[1]
    TB = 512
    TE = 512
    NB = S // TB
    TS = TB // W        # 64 tokens per rank per block
    HC = TS // 2        # 32 per physical core
    HH = HT // 2
    NQKV = NQL + 2 * NKL
    pid = nl.program_id(0)
    if LAG < 0 or LAG > NB:
        LAG = NB
    rk = ncc.rank_id()
    rk_sb = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.sbuf)
    nisa.register_store(rk_sb, rk)
    rp_sb = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.sbuf)   # 2 * rank + pid: the own HC-token slot of a block
    nisa.tensor_scalar(dst=rp_sb, data=rk_sb, op0=nl.multiply, operand0=2, op1=nl.add, operand1=pid)
    rreg = nisa.register_alloc()
    nisa.register_load(rreg, rp_sb)

    bf = nl.bfloat16
    L_xa = []
    L_xg = []
    L_xb = []
    L_x2g = []
    L_hb_a = []
    L_ra = []
    L_hb_m = []
    L_rm = []
    for l_ in range(LPN):
        xa_ = []
        for b_ in range(NB):
            xa_.append(_spbuf((2, 128, HT, HC), bf, f"sp{l_}_xa{b_}"))
        L_xa.append(xa_)
        xg_ = []
        for b_ in range(NB):
            xg_.append(_spbuf((2 * W, 128, HT, HC), bf, f"sp{l_}_xg{b_}"))
        L_xg.append(xg_)
        xb_ = []
        for b_ in range(NB):
            xb_.append(_spbuf((2, 128, HT, HC), bf, f"sp{l_}_xb{b_}"))
        L_xb.append(xb_)
        x2g_ = []
        for b_ in range(NB):
            x2g_.append(_spbuf((2 * W, 128, HT, HC), bf, f"sp{l_}_x2g{b_}"))
        L_x2g.append(x2g_)
        hb_a_ = []
        for b_ in range(NB):
            hb_a_.append(_spbuf((2 * W, 128, HT, HC), bf, f"sp{l_}_hb_a{b_}"))
        L_hb_a.append(hb_a_)
        ra_ = []
        for b_ in range(NB):
            ra_.append(_spbuf((2, 128, HT, HC), bf, f"sp{l_}_ra{b_}"))
        L_ra.append(ra_)
        hb_m_ = []
        for b_ in range(NB):
            hb_m_.append(_spbuf((2 * W, 128, HT, HC), bf, f"sp{l_}_hb_m{b_}"))
        L_hb_m.append(hb_m_)
        rm_ = []
        for b_ in range(NB):
            rm_.append(_spbuf((2, 128, HT, HC), bf, f"sp{l_}_rm{b_}"))
        L_rm.append(rm_)
    act_x = nl.ndarray((2, 128, ITL, S), dtype=bf, buffer=nl.shared_hbm, name="sp_act_x")
    rot_sb = nl.ndarray((128, 128), dtype=bf, buffer=nl.sbuf)
    nisa.dma_copy(dst=rot_sb, src=rot)
    aq0 = None
    ak0 = None
    av0 = None
    ao0 = None
    if SPH:   # attention inputs / output of the first SPH blocks alone (separate tensors: nkilib on full tensors)
        SH = SPH * TB
        aq0 = nl.ndarray((2 * NQL, 128, SH), dtype=bf, buffer=nl.shared_hbm, name="sp_aq0")
        ak0 = nl.ndarray((2 * NKL, 128, SH), dtype=bf, buffer=nl.shared_hbm, name="sp_ak0")
        av0 = nl.ndarray((2 * NKL, SH, 128), dtype=bf, buffer=nl.shared_hbm, name="sp_av0")
        ao0 = nl.ndarray((2 * NQL, 128, SH), dtype=bf, buffer=nl.shared_hbm, name="sp_ao0")
    defer = []
    for l in range(LPN):
        xa = L_xa[l]
        xg = L_xg[l]
        xb = L_xb[l]
        x2g = L_x2g[l]
        hb_a = L_hb_a[l]
        ra = L_ra[l]
        hb_m = L_hb_m[l]
        rm = L_rm[l]
        w1 = nl.ndarray((128, HT), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=w1, src=ln1[l])
        w2 = nl.ndarray((128, HT), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=w2, src=ln2[l])
        qnc = nl.ndarray((128, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=qnc, src=qn[l])
        knc = nl.ndarray((128, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=knc, src=kn[l])
        wn1 = None
        if l + 1 < LPN:
            wn1 = nl.ndarray((128, HT), dtype=nl.float32, buffer=nl.sbuf)
            nisa.dma_copy(dst=wn1, src=ln1[l + 1])

        # ===== A0: norm1 of the own tokens, all-gathers launched NA blocks ahead of the QKV block that needs them
        # (GpSimd triggers the collectives in order and also runs the QKV blocks' rsqrt / RoPE adds: issuing all eight
        # up front stalled the first QKV block behind the last trigger)
        NA = SPA
        if NA <= 0 or NA > NB:
            NA = NB
        if l > 0:
            NA = NB   # every norm1 of this layer was issued in the previous layer's MLP loop
        for blk in range(NA):
            if l == 0:
                _norm1(h, blk, pid, rreg, HT, S, W, TS, HC, w1, onesb, xa, xg, rg)

        # ===== A: QKV + q/k norm + RoPE + cache writes (prefill_tp's instructions)
        dq = defer
        defer = []
        for blk in range(NB):
            t0 = blk * TB
            if blk + NA < NB:
                _norm1(h, blk + NA, pid, rreg, HT, S, W, TS, HC, w1, onesb, xa, xg, rg)
            if blk >= SPD and blk - SPD < len(dq):   # the previous layer's last finishes + this layer's norm1
                hf = _tail(h, dq[blk - SPD], pid, rreg, HT, S, W, TS, HC, L_rm[l - 1])
                _norm1_sb(hf, dq[blk - SPD], pid, HT, HC, w1, onesb, xa, xg, rg)
            x1 = _gathered(xg, blk, HT, W, HC)
            cs = nl.ndarray((128, TB), dtype=nl.float32, buffer=nl.sbuf)
            _hd(cs, cosT[:, nl.ds(t0, TB)])
            sn = nl.ndarray((128, TB), dtype=nl.float32, buffer=nl.sbuf)
            _hd(sn, sinT[:, nl.ds(t0, TB)])
            prev = []
            kq = []
            if KTQ > 0:   # round 008 (KTQ): the first KTQ QKV tiles kt-outer over the all-gather parts (same order
                # per accumulator)
                for nt in range(KTQ):
                    kq.append((_slab(wqkv[l, pid], nt, HT), nl.ndarray((128, TB), dtype=nl.float32, buffer=nl.psum)))
                NSq = SPSN[0]
                HPq = HT // NSq
                for g in range(NSq):
                    for nt in range(KTQ):
                        for kt in range(g * HPq, (g + 1) * HPq):
                            nisa.nc_matmul(dst=kq[nt][1], stationary=kq[nt][0][:, kt * 128:(kt + 1) * 128],
                                           moving=x1[:, :, kt * HC:(kt + 1) * HC])
            for nt in range(NQKV):
                if nt < KTQ:
                    ps = kq[nt][1]
                else:
                    wt = _slab(wqkv[l, pid], nt, HT)
                    ps = nl.ndarray((128, TB), dtype=nl.float32, buffer=nl.psum)
                    for kt in range(HT):
                        nisa.nc_matmul(dst=ps, stationary=wt[:, kt * 128:(kt + 1) * 128], moving=x1[:, :, kt * HC:(kt + 1) * HC])
                if SPQ:   # tile nt - 1's post-processing after tile nt's matmuls (the in-order TensorE does not
                    # wait for its norm / rsqrt / RoPE chain)
                    if len(prev) > 0:
                        _qkv_post(l, prev[0], blk, t0, prev[1], TB, NQL, NKL, D, pid, onesb, qnc, knc, rot_sb, cs, sn,
                                  ident, s_q, s_k, s_v, kcache, vcache, VPERM, SPH, aq0, ak0, av0)
                    prev = [nt, ps]
                else:
                    _qkv_post(l, nt, blk, t0, ps, TB, NQL, NKL, D, pid, onesb, qnc, knc, rot_sb, cs, sn,
                              ident, s_q, s_k, s_v, kcache, vcache, VPERM, SPH, aq0, ak0, av0)
            if len(prev) > 0:
                _qkv_post(l, prev[0], blk, t0, prev[1], TB, NQL, NKL, D, pid, onesb, qnc, knc, rot_sb, cs, sn,
                          ident, s_q, s_k, s_v, kcache, vcache, VPERM, SPH, aq0, ak0, av0)
        nisa.core_barrier(data=s_q, cores=(0, 1))
        nisa.core_barrier(data=s_k, cores=(0, 1))
        nisa.core_barrier(data=s_v, cores=(0, 1))
        done2 = []
        fin = []
        dn = []
        for blk in range(NB):
            done2.append(0)
            fin.append(0)
            dn.append(0)
        # ===== B: attention (nkilib)
        if SPH:   # the first SPH blocks' attention alone first (bitwise equal to those rows of the full call), so
            # their o_proj + reduce-scatter + norm2 + all-gather chains run during the full call
            nisa.core_barrier(data=aq0, cores=(0, 1))
            nisa.core_barrier(data=ak0, cores=(0, 1))
            nisa.core_barrier(data=av0, cores=(0, 1))
            _attention_cte(q=aq0, k=ak0, v=av0, scale=SCALE, causal_mask=True, tp_q=False, tp_k=False, tp_out=True,
                           out_o_hbm=ao0)
            nisa.core_barrier(data=ao0, cores=(0, 1))
            for b_ in range(SPH):   # (norm2 of the previous pre-emitted block ahead of each reduce-scatter)
                _oproj(l, b_, b_, ao0, b_ * TB, TB, NQL, HH, HT, W, HC, pid, wo, hb_a, ra, rg, 1, done2,
                       h, rreg, S, TS, w2, onesb, xb, x2g)
            if SPHN:   # and the last one's norm2 + all-gather before the full attention (TensorE waits for its RS)
                _norm2(h, SPH - 1, pid, rreg, HT, S, W, TS, HC, ra, w2, onesb, xb, x2g, rg)
                done2[SPH - 1] = 1
        _attention_cte(q=s_q, k=s_k, v=s_v, scale=SCALE, causal_mask=True, tp_q=False, tp_k=False, tp_out=True,
                       out_o_hbm=s_o)
        nisa.core_barrier(data=s_o, cores=(0, 1))

        # ===== C + D/E: o_proj block it, MLP block it - LAG (its reduce-scatter has had time)
        pend = []
        for it in range(NB + LAG):
            pgu = None
            if it < NB and it >= SPH:
                pgu = _oproj(l, it, it, s_o, it * TB, TB, NQL, HH, HT, W, HC, pid, wo, hb_a, ra, rg, SPO, done2,
                             h, rreg, S, TS, w2, onesb, xb, x2g, SPG, wgu[l, pid], ITL)
                if pgu is not None and it - SPO != it - LAG:
                    pgu = None   # (prefetched for another block: not used; SPG assumes SPO == LAG)
            if it >= LAG:
                blk = it - LAG
                t0 = blk * TE
                if pgu is None and SPG > 0:
                    pgu = _pre_gu(wgu[l, pid], SPG, ITL, HT)
                if done2[blk] == 0:
                    _norm2(h, blk, pid, rreg, HT, S, W, TS, HC, ra, w2, onesb, xb, x2g, rg)
                    done2[blk] = 1
                x2 = _gathered(x2g, blk, HT, W, HC)
                act = nl.ndarray((128, ITL, TE), dtype=bf, buffer=nl.sbuf)
                if KTP > 0:   # round 008 (KTP): the first KTP gate / up pairs kt-outer over the all-gather parts (each
                    # accumulator still sums kt 0 .. HT-1 in order), so TensorE works on part 0 while part 1 lands
                    kw = []
                    for i in range(KTP):
                        if pgu is not None and i < len(pgu):
                            wg = pgu[i][0]
                            wu = pgu[i][1]
                        else:
                            wg = _slab(wgu[l, pid], i, HT)
                            wu = _slab(wgu[l, pid], ITL + i, HT)
                        pg = nl.ndarray((128, TE), dtype=nl.float32, buffer=nl.psum)
                        pu = nl.ndarray((128, TE), dtype=nl.float32, buffer=nl.psum)
                        kw.append((wg, wu, pg, pu))
                    NSg = SPSN[0]
                    HPg = HT // NSg
                    for g in range(NSg):
                        for i in range(KTP):
                            for kt in range(g * HPg, (g + 1) * HPg):
                                nisa.nc_matmul(dst=kw[i][2], stationary=kw[i][0][:, kt * 128:(kt + 1) * 128],
                                               moving=x2[:, :, kt * HC:(kt + 1) * HC])
                            for kt in range(g * HPg, (g + 1) * HPg):
                                nisa.nc_matmul(dst=kw[i][3], stationary=kw[i][1][:, kt * 128:(kt + 1) * 128],
                                               moving=x2[:, :, kt * HC:(kt + 1) * HC])
                    for i in range(KTP):
                        sg = nl.ndarray((128, TE), dtype=nl.float32, buffer=nl.sbuf)
                        nisa.activation(dst=sg, op=nl.silu, data=kw[i][2])
                        nisa.tensor_tensor(dst=act[:, i, :], data1=sg, data2=kw[i][3], op=nl.multiply)
                NFU = ITL - 1 if MP else ITL   # MP: 12 full pairs here, the packed slab after the loop
                for i in range(KTP, NFU):
                    if pgu is not None and i < len(pgu):
                        wg = pgu[i][0]
                    else:
                        wg = _slab(wgu[l, pid], i, HT)
                    pg = nl.ndarray((128, TE), dtype=nl.float32, buffer=nl.psum)
                    for kt in range(HT):
                        nisa.nc_matmul(dst=pg, stationary=wg[:, kt * 128:(kt + 1) * 128], moving=x2[:, :, kt * HC:(kt + 1) * HC])
                    if pgu is not None and i < len(pgu):
                        wu = pgu[i][1]
                    else:
                        wu = _slab(wgu[l, pid], NFU + i, HT)
                    pu = nl.ndarray((128, TE), dtype=nl.float32, buffer=nl.psum)
                    for kt in range(HT):
                        nisa.nc_matmul(dst=pu, stationary=wu[:, kt * 128:(kt + 1) * 128], moving=x2[:, :, kt * HC:(kt + 1) * HC])
                    sg = nl.ndarray((128, TE), dtype=nl.float32, buffer=nl.sbuf)
                    nisa.activation(dst=sg, op=nl.silu, data=pg)
                    nisa.tensor_tensor(dst=act[:, i, :], data1=sg, data2=pu, op=nl.multiply)
                if MP:   # the packed slab: psum rows 0-63 = gate half, 64-127 = up half of this core's last 64 rows
                    wp = _slab(wgu[l, pid], 2 * NFU, HT)
                    pp = nl.ndarray((128, TE), dtype=nl.float32, buffer=nl.psum)
                    for kt in range(HT):
                        nisa.nc_matmul(dst=pp, stationary=wp[:, kt * 128:(kt + 1) * 128], moving=x2[:, :, kt * HC:(kt + 1) * HC])
                    sgh = nl.ndarray((64, TE), dtype=nl.float32, buffer=nl.sbuf)
                    nisa.activation(dst=sgh, op=nl.silu, data=pp[0:64, :])
                    nisa.tensor_tensor(dst=act[0:64, NFU, :], data1=sgh, data2=pp[64:128, :], op=nl.multiply)
                if PN and blk + 1 < NB and blk + 1 <= it:
                    if done2[blk + 1] == 0:
                        # the next MLP block's norm2 + all-gather now (its reduce-scatter was launched before)
                        _norm2(h, blk + 1, pid, rreg, HT, S, W, TS, HC, ra, w2, onesb, xb, x2g, rg)
                        done2[blk + 1] = 1
                if TLG > 0 and blk >= TLG and dn[blk - TLG] == 1:   # finish MLP block blk - TLG (its reduce-scatter
                    # has had TLG blocks; with SPP only if its down is already emitted)
                    hf = _tail(h, blk - TLG, pid, rreg, HT, S, W, TS, HC, rm)
                    fin[blk - TLG] = 1
                    if l + 1 < LPN:   # and the next layer's norm1 + all-gather of that block
                        _norm1_sb(hf, blk - TLG, pid, HT, HC, wn1, onesb, L_xa[l + 1], L_xg[l + 1], rg)
                if MP:   # 12 full tiles + the half tile (rows 0-63 of slot 12); oth = the other core's 12 tiles + in
                    # slot 12 the combo tile (rows 0-63 core 0's half, 64-127 core 1's)
                    _hd(act_x[pid, :, 0:ITL - 1, nl.ds(t0, TE)], act[:, 0:ITL - 1, :])
                    _hd(act_x[pid, 0:64, ITL - 1, nl.ds(t0, TE)], act[0:64, ITL - 1, :])
                    nisa.core_barrier(data=act_x, cores=(0, 1))
                    oth = nl.ndarray((128, ITL, TE), dtype=bf, buffer=nl.sbuf)
                    _hd(oth[:, 0:ITL - 1, :], act_x[1 - pid, :, 0:ITL - 1, nl.ds(t0, TE)])
                    _hd(oth[0:64, ITL - 1, :], act_x[0, 0:64, ITL - 1, nl.ds(t0, TE)])
                    _hd(oth[64:128, ITL - 1, :], act_x[1, 0:64, ITL - 1, nl.ds(t0, TE)])
                else:
                    _hd(act_x[pid, :, :, nl.ds(t0, TE)], act)
                    nisa.core_barrier(data=act_x, cores=(0, 1))
                    oth = nl.ndarray((128, ITL, TE), dtype=bf, buffer=nl.sbuf)
                    _hd(oth, act_x[1 - pid, :, :, nl.ds(t0, TE)])
                if SPP:   # down of the previous MLP block now: its act exchange had this block's gate/up
                    if len(pend) > 0:
                        _down(l, pend[0], pend[1], pend[2], TE, ITL, HH, HT, W, HC, pid, wd, hb_m, rm, rg)
                        dn[pend[0]] = 1
                    pend = [blk, act, oth]
                else:
                    _down(l, blk, act, oth, TE, ITL, HH, HT, W, HC, pid, wd, hb_m, rm, rg)
                    dn[blk] = 1
        if SPP and len(pend) > 0:
            _down(l, pend[0], pend[1], pend[2], TE, ITL, HH, HT, W, HC, pid, wd, hb_m, rm, rg)
            pend = []
        # ===== tail: h += rm on the own tokens (blocks not finished inside the loop)
        for blk in range(NB):
            if fin[blk] == 0:
                if l + 1 < LPN and SPD and len(defer) + SPD <= blk:   # deferred into the next layer's QKV loop (TensorE would wait here for
                    # the last reduce-scatters with the next layer's first QKV block ready)
                    defer.append(blk)
                else:
                    hf = _tail(h, blk, pid, rreg, HT, S, W, TS, HC, rm)
                    if l + 1 < LPN:
                        _norm1_sb(hf, blk, pid, HT, HC, wn1, onesb, L_xa[l + 1], L_xg[l + 1], rg)


def _tail(h, blk, pid, rreg, HT, S, W, TS, HC, rm):
    hc = nl.ndarray((128, HT, HC), dtype=nl.float32, buffer=nl.sbuf)
    _hd(hc, _hview(h, blk, pid, rreg, HT, S, W, TS, HC))
    r = nl.ndarray((128, HT, HC), dtype=nl.bfloat16, buffer=nl.sbuf)
    if SPSN[0] > 1:
        _load_rs(r, rm[blk], pid, HT, HC)
    else:
        _hd(r, rm[blk][pid])
    nisa.tensor_tensor(dst=hc, data1=hc, data2=r, op=nl.add)
    _hd(_hview(h, blk, pid, rreg, HT, S, W, TS, HC), hc)
    return hc


def _norm1_sb(hc, blk, pid, HT, HC, w1, onesb, xa, xg, rg):
    """_norm1 from the residual already in SBUF (the previous layer's finish, LPN > 1)"""
    xo = nl.ndarray((128, HT, HC), dtype=nl.bfloat16, buffer=nl.sbuf)
    if SPSN[0] > 1:
        _norm_own_ag(hc, HC, w1, onesb, HT, xo, xa[blk], xg[blk], pid, rg)
        return
    _norm_own(hc, HC, w1, onesb, HT, xo)
    _hd(xa[blk][pid], xo)
    nisa.core_barrier(data=xa[blk], cores=(0, 1))
    ncc.all_gather(srcs=[xa[blk]], dsts=[xg[blk]], replica_group=rg, collective_dim=0)


def head_sp_mask(hl, plen, rkb, NVL, W):
    """SP head: only the owner rank of token plen - 1 has its residual; hl [128, HT] fp32 of every rank is masked
    to the owner's (x * 1, others * 0) and summed over the ranks (exact: one non-zero term)"""
    HT = hl.shape[1]
    pb = nl.ndarray((128, 1), dtype=nl.int32, buffer=nl.sbuf)
    nisa.dma_copy(dst=pb, src=plen.ap(pattern=[[0, 128], [1, 1]], offset=0))
    ow = nl.ndarray((128, 1), dtype=nl.int32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=ow, data=pb, op0=nl.add, operand0=-1)
    nisa.tensor_scalar(dst=ow, data=ow, op0=nl.bitwise_and, operand0=511)
    nisa.tensor_scalar(dst=ow, data=ow, op0=nl.right_shift, operand0=6)
    owf = nl.ndarray((128, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=owf, src=ow)
    nisa.tensor_scalar(dst=owf, data=owf, op0=nl.multiply, operand0=float(2 * NVL * 128))   # = rkb of the owner
    rb = nl.ndarray((128, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=rb, src=rkb)
    m = nl.ndarray((128, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=m, data1=rb, data2=owf, op=nl.equal)
    hm = nl.ndarray((128, HT), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=hm, data=hl, op0=nl.multiply, operand0=m)
    return hm


def head_sp(hl, plen, rkb, NVL, W, rg):
    hm = head_sp_mask(hl, plen, rkb, NVL, W)
    HT = hl.shape[1]
    hs = nl.ndarray((128, HT), dtype=nl.float32, buffer=nl.sbuf)
    ncc.all_reduce(srcs=[hm], dsts=[hs], op=nl.add, replica_group=rg)
    return hs
