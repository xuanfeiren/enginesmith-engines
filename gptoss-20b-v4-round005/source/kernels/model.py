"""NKI kernels for gpt-oss-20b decode on Trainium2: TP=4 logical cores x 2 physical cores (8 "vcores").

Vcore g = 2 * rank + pid owns query heads 8g..8g+7, KV head g, and the intermediate slice
[360g, 360g + 360) of every expert. Hidden vectors live in SBUF in column layout [120, 24]
(element kt*120 + p at [p, kt]). Weight layouts are produced on the host (src/engine.cpp, tools/layout.py).
"""
import nki
import nki.language as nl
import nki.isa as nisa
import nki.collectives as ncc

H = 2880
P = 120            # partitions used for hidden-dim tiles
HT = 24            # H / P
NQ = 8             # query heads per vcore
D = 64             # head dim
E = 32             # experts
TOPK = 4
IT = 3             # intermediate tiles (of 120) per vcore per expert
GU_T = 2 * IT      # gate tiles then up tiles
GU_F = GU_T * HT * P     # free elements per partition of one expert's gate_up slice (17280)
DN_F = IT * HT * P       # free elements per partition of one expert's down slice (8640)
QKV_T = NQ + 2           # 8 q heads, k, v (64 columns each)
QKV_F = QKV_T * HT * D   # 15360
WO_F = NQ * HT * P       # 23040
EPS = 1e-5
SWIGLU_LIMIT = 7.0
ALPHA = 1.702
NEG = -1.0e30
SLIDE = 128
VOCAB = 201088
NVT = 200                # vocab tiles (of 128) per vcore; 8 * 200 * 128 = 204800 >= VOCAB
LM_G = 4                 # vocab tiles per LM-head DMA group (NVT / LM_G groups, an even number: two PSUM halves)
HT2 = 23                 # 128-row tiles of the hidden dim (23 * 128 = 2944 >= H, rows >= H are zero)
LM_F = LM_G * HT2 * 128  # free elements per partition per LM-head group (11776): 128-partition DMAs stream at ~2x
                         # the rate of 120-partition ones (measured 405 vs 190 GB/s, scratch/mb/dmab.py)
BIG = 1048576.0          # 2^20 > any token id
WDT = nl.float8_e5m2     # expert weights: the MXFP4 values are exact in e5m2 (src/engine.cpp mx_row)
GU_DMA = 3               # DMAs per expert gate/up slice in decode (session 4: 1 -> 242.5, 6 -> 245.4, 3 -> 239.1 ms / 64 tokens)
LATE_WO = False              # wo2 at its own layer's start instead of one layer ahead (slower: 319.5 -> 330.5 ms)
PERSIST_AW = False           # persistent ping-pong buffers for the attention weights (no gain: session 3, and again in session 4: 239.1 -> 239.9 ms)
PERSIST_EW = False           # persistent buffers for the 4 experts' weights (slower: 319.5 -> 331.7 ms)
AW_ENG = nisa.engine.sync  # DGE queue of the one-layer-ahead attention-weight DMAs (not behind the Sync queue's all-reduce DMAs)
CT_GU = True             # decode gate/up with TensorE column tiling (wgu8 layout [p][kt][it][up][n], src/engine.cpp)
CT_DN = True             # decode down projection activation-stationary with column tiling (same wd8 layout)
AR_SBUF = True           # attention all-reduce on SBUF tensors
AR_SPLIT = False         # one SBUF all-reduce per expert as soon as its partial is done: slower (261 -> 322 ms / 64 tokens), the collectives stall the expert DMAs
AR_SBUF_MOE = False      # MoE all-reduce on SBUF tensors: no gain (260.6 -> 259.8 ms / 64 tokens)
CT_QKV = True            # decode QKV with column tiling
CT_OPROJ = True          # decode o_proj with column tiling
QG = [(0, 3), (3, 3), (6, 2), (8, 2)]  # (first head, heads) per column tile in the column-tiled QKV
AR_GATHER = True         # all-reduces as SBUF all-gather + local sum
AR_GATHER_HEAD = False    # the LM head's two max all-reduces as all-gathers (no gain)
CT_HEAD = True           # LM head activation-stationary with TensorE column tiling (session 5)
TOP8 = True              # decode top-4 by max8 + nc_find_index8 (session 5)
SHARE_W = True           # session 5: attention sets ping-pong in R0 / R1, layer l's experts in R[l % 2] + C, so the
                         # next layer's attention weights stream during this layer's attention chain (DMA idle)
AW_EARLY = True          # SHARE_W: next layer's attention weights at the layer start (False: after the experts)
ATT_T_MIN = 1024         # session 5: full-layer decode attention with keys on partitions for buckets >= this
AW_AFTER_KV = False      # ... and after this (full) layer's K/V cache reads: decode 4096 -0.9% (session 5)
AW_DEP = True            # the layer's first expert DMA depends on the next layer's attention-weight DMAs (issue order)
EXP_ENG = (nisa.engine.sync, nisa.engine.sync)  # DGE queues of the even / odd experts' weight DMAs (sync + scalar: no gain, session 5)
AW_B = (QKV_F + WO_F // 2 + HT * E) * 2  # bytes per partition of one attention set (55296)
EXP_B = (GU_F + DN_F)                    # fp8 bytes per partition of one expert (25920)
XS = 12                  # layers per expert-weight stack in the fused decode NEFF
TAGS = ["_" + str(i) for i in range(64)]  # per-layer names of the collective buffers


def _load_col(src2d, dtype):
    """[1, H] HBM -> [P, HT] SBUF column layout."""
    t = nl.ndarray((P, HT), dtype=dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=t, src=src2d.ap(pattern=[[1, P], [P, HT]], offset=0))
    return t


def _store_col(dst2d, t):
    nisa.dma_copy(dst=dst2d.ap(pattern=[[1, P], [P, HT]], offset=0), src=t)


def _rmsnorm(x_f32, w_f32, ones_pp, pad=False):
    """x [P, HT] fp32 (SBUF), w [P, HT] fp32 -> bf16(w * (x * rsqrt(mean(x^2) + eps))) [P, HT] (pad: [128, HT] with
    rows P..127 zero, the moving / stationary operand of the 128-partition weight copies)."""
    sq = nl.ndarray((P, HT), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=sq, data1=x_f32, data2=x_f32, op=nl.multiply)
    ss = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_reduce(dst=ss, op=nl.add, data=sq, axis=1)
    tot = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.psum)
    nisa.nc_matmul(dst=tot, stationary=ones_pp, moving=ss)
    var = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=var, data=tot, op0=nl.multiply, operand0=1.0 / H, op1=nl.add, operand1=EPS)
    rstd = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=rstd, data=var, op0=nl.rsqrt, operand0=0.0, engine=nisa.engine.gpsimd)
    xs = nl.ndarray((P, HT), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=xs, data=x_f32, op0=nl.multiply, operand0=rstd)
    if pad:
        y = nl.ndarray((128, HT), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.memset(dst=y, value=0.0)
        nisa.tensor_tensor(dst=y[0:P, 0:HT], data1=xs, data2=w_f32, op=nl.multiply)
        return y
    y = nl.ndarray((P, HT), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=y, data1=xs, data2=w_f32, op=nl.multiply)
    return y


def _sel2():
    """[64, 2, 128] bf16: sel2[d, j, p] = 1 iff p == d + 64 j (the head-pair stacking of the decode o_proj)."""
    io = nl.ndarray((D, 2, 128), dtype=nl.float32, buffer=nl.sbuf)
    nisa.iota(dst=io, pattern=[[-64, 2], [1, 128]], offset=0, channel_multiplier=-1)  # p - 64 j - d
    s = nl.ndarray((D, 2, 128), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=s, data=io, op0=nl.equal, operand0=0.0)
    return s


def _ones(shape):
    t = nl.ndarray(shape, dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(dst=t, value=1.0)
    return t


def _allreduce(part, ncols, pid, rg, name, sbuf=False):
    """part [P, ncols] fp32 SBUF partial on every vcore -> full sum over 8 vcores [P, ncols] fp32 SBUF.
    sbuf: the all-reduce reads and writes SBUF directly (no HBM hops / core barrier; measured 15.8 -> 11.3 us for
    ncols 24, but 18.9 -> 20.8 us for ncols 96, scratch/mb/ar.py)."""
    other = nl.ndarray((P, ncols), dtype=nl.float32, buffer=nl.sbuf)
    nisa.sendrecv(src=part, dst=other, send_to_rank=1 - pid, recv_from_rank=1 - pid, pipe_id=0,
                  dma_engine=nisa.dma_engine.gpsimd_dma)
    both = nl.ndarray((P, ncols), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=both, data1=part, data2=other, op=nl.add)
    if AR_GATHER and rg is not None:
        # all-gather of the 4 ranks' partials (SBUF) + a fixed-order local sum (identical on every rank)
        gat = nl.ndarray((P, 4 * ncols), dtype=nl.float32, buffer=nl.sbuf)
        ncc.all_gather(srcs=[both], dsts=[gat], replica_group=rg, collective_dim=1)
        g01 = nl.ndarray((P, ncols), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=g01, data1=gat[0:P, 0:ncols], data2=gat[0:P, ncols:2 * ncols], op=nl.add)
        g23 = nl.ndarray((P, ncols), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=g23, data1=gat[0:P, 2 * ncols:3 * ncols], data2=gat[0:P, 3 * ncols:4 * ncols], op=nl.add)
        res = nl.ndarray((P, ncols), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=res, data1=g01, data2=g23, op=nl.add)
        return res
    if sbuf and rg is not None:
        res = nl.ndarray((P, ncols), dtype=nl.float32, buffer=nl.sbuf)
        ncc.all_reduce(srcs=[both], dsts=[res], op=nl.add, replica_group=rg)
        return res
    src = nl.ndarray((P, ncols), dtype=nl.float32, buffer=nl.shared_hbm, name=name + "_src")
    dst = nl.ndarray((P, ncols), dtype=nl.float32, buffer=nl.shared_hbm, name=name + "_dst")
    nisa.dma_copy(dst=src, src=both)
    if rg is None:  # single-rank profiling build
        return both
    nisa.core_barrier(data=src, cores=(0, 1))
    ncc.all_reduce(srcs=[src], dsts=[dst], op=nl.add, replica_group=rg)
    res = nl.ndarray((P, ncols), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=res, src=dst)
    return res


def _pos_ctx(pos, cs, need_sliding, need_full):
    """Per-step position state shared by all layers of a step: register, masks limits, cos/sin columns."""
    MAXPOS = cs.shape[0]
    p_sb = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.sbuf)
    nisa.dma_copy(dst=p_sb, src=pos)
    preg = nisa.register_alloc()
    nisa.register_load(preg, p_sb)
    # pos broadcast over the NQ head partitions (fp32) for the mask
    p8i = nl.ndarray((NQ, 1), dtype=nl.int32, buffer=nl.sbuf)
    nisa.dma_copy(dst=p8i, src=pos.ap(pattern=[[0, NQ], [1, 1]], offset=0))
    p8 = nl.ndarray((NQ, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=p8, src=p8i)
    if need_sliding:
        # window [s, s + 128) with s = max(pos - 127, 0); slot j valid iff j <= pos - s = min(pos, 127)
        s_sb = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=s_sb, data=p_sb, op0=nl.add, operand0=-(SLIDE - 1.0), op1=nl.maximum, operand1=0.0)
        sreg = nisa.register_alloc()
        nisa.register_load(sreg, s_sb)
        lim = nl.ndarray((NQ, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=lim, data=p8, op0=nl.minimum, operand0=SLIDE - 1.0)
    # cos / sin columns for this position
    csc = nl.ndarray((D, 2), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=csc, src=cs.ap(pattern=[[1, D], [D, 2]], offset=0, scalar_offset=preg, indirect_dim=0))
    if not need_sliding:
        sreg = preg
        lim = p8
    return p_sb, preg, sreg, lim, p8, csc


def _key_mask_T(pos, B):
    """[128, B // 128, NQ] fp32: NEG where key 128 j + p >= pos (the cached slots a full layer must not see), else 0."""
    NB = B // 128
    pbi = nl.ndarray((128, 1), dtype=nl.int32, buffer=nl.sbuf)
    nisa.dma_copy(dst=pbi, src=pos.ap(pattern=[[0, 128], [1, 1]], offset=0))
    pbf = nl.ndarray((128, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=pbf, src=pbi)
    kio = nl.ndarray((128, NB, NQ), dtype=nl.float32, buffer=nl.sbuf)
    nisa.iota(dst=kio, pattern=[[128, NB], [0, NQ]], offset=0, channel_multiplier=1)
    m = nl.ndarray((128, NB, NQ), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=m, data=kio, op0=nl.greater_equal, operand0=pbf, op1=nl.multiply, operand1=NEG)
    return m


def _bcast_heads(v_col):
    """[NQ, 1] fp32 per-head scalars -> [128, NQ] fp32 (every partition the row; exact: transpose and 1.0 x v)."""
    tp = nl.ndarray((1, NQ), dtype=nl.float32, buffer=nl.psum)
    nisa.nc_transpose(dst=tp, data=v_col)
    row = nl.ndarray((1, NQ), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=row, src=tp)
    bp = nl.ndarray((128, NQ), dtype=nl.float32, buffer=nl.psum)
    nisa.nc_matmul(dst=bp, stationary=_ones((1, 128)), moving=row)
    b = nl.ndarray((128, NQ), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=b, src=bp)
    return b


def _attn_keys_on_partitions(qk, k_sb, v_sb, mskT, scur, sink_sb, vrow, B):
    """Full-layer decode attention with the cached keys on partitions (session 5): scores^T [128, NB, NQ] (key
    128 j + p, head h) instead of [NQ, B] on 8 partitions, so each elementwise pass touches NB * NQ elements per
    partition instead of B, and the probabilities are already the PV matmul's moving operand (no transposes).
    Same rounding points as the [NQ, B] form (bf16 q.k, x 0.125 + mask, exact max, exp(s - m), bf16 p = e / sum);
    only the fp32 order of the softmax sum differs. Returns acc_o [D, NQ] fp32 PSUM (+ the current token)."""
    NB = B // 128
    sT_ps = nl.ndarray((128, NB, NQ), dtype=nl.float32, buffer=nl.psum)
    for j in range(NB):
        nisa.nc_matmul(dst=sT_ps[0:128, j, 0:NQ], stationary=k_sb[0:D, j * 128:(j + 1) * 128], moving=qk[0:D, 0:NQ])
    sTb = nl.ndarray((128, NB, NQ), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.tensor_copy(dst=sTb, src=sT_ps)
    sT = nl.ndarray((128, NB, NQ), dtype=nl.float32, buffer=nl.sbuf)
    nisa.scalar_tensor_tensor(dst=sT, data=sTb, op0=nl.multiply, operand0=0.125, op1=nl.add, operand1=mskT)
    mx1 = nl.ndarray((128, NQ), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_reduce(dst=mx1, op=nl.maximum, data=sT.permute((0, 2, 1)), axis=2)
    mxT = nl.ndarray((NQ, 128), dtype=nl.float32, buffer=nl.psum)
    nisa.nc_transpose(dst=mxT, data=mx1)
    m1 = nl.ndarray((NQ, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_reduce(dst=m1, op=nl.maximum, data=mxT, axis=1)
    m0 = nl.ndarray((NQ, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=m0, data1=m1, data2=scur, op=nl.maximum)
    m = nl.ndarray((NQ, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=m, data1=m0, data2=sink_sb, op=nl.maximum)
    negm = nl.ndarray((NQ, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=negm, data=m, op0=nl.multiply, operand0=-1.0)
    nb = _bcast_heads(negm)
    dd = nl.ndarray((128, NB, NQ), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=dd, data1=sT, data2=nb.expand_dim(1).broadcast(1, NB), op=nl.add)
    pexp = nl.ndarray((128, NB, NQ), dtype=nl.float32, buffer=nl.sbuf)
    nisa.activation(dst=pexp, op=nl.exp, data=dd)
    ps1 = nl.ndarray((128, NQ), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_reduce(dst=ps1, op=nl.add, data=pexp.permute((0, 2, 1)), axis=2)
    psp = nl.ndarray((NQ, 1), dtype=nl.float32, buffer=nl.psum)
    nisa.nc_matmul(dst=psp, stationary=ps1, moving=_ones((128, 1)))
    psum_ = nl.ndarray((NQ, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=psum_, src=psp)
    esink = nl.ndarray((NQ, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.activation(dst=esink, op=nl.exp, data=sink_sb, bias=negm, scale=1.0)
    ecur = nl.ndarray((NQ, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.activation(dst=ecur, op=nl.exp, data=scur, bias=negm, scale=1.0)
    tot0 = nl.ndarray((NQ, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=tot0, data1=psum_, data2=ecur, op=nl.add)
    tot = nl.ndarray((NQ, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=tot, data1=tot0, data2=esink, op=nl.add)
    rtot = nl.ndarray((NQ, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.reciprocal(dst=rtot, data=tot)
    rb = _bcast_heads(rtot)
    probT = nl.ndarray((128, NB, NQ), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=probT, data1=pexp, data2=rb.expand_dim(1).broadcast(1, NB), op=nl.multiply)
    pcur = nl.ndarray((NQ, 1), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=pcur, data=ecur, op0=nl.multiply, operand0=rtot)
    prow_ps = nl.ndarray((1, NQ), dtype=nl.bfloat16, buffer=nl.psum)
    nisa.nc_transpose(dst=prow_ps, data=pcur)
    prow = nl.ndarray((1, NQ), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.tensor_copy(dst=prow, src=prow_ps)
    acc_o = nl.ndarray((D, NQ), dtype=nl.float32, buffer=nl.psum)
    for j in range(NB):
        nisa.nc_matmul(dst=acc_o, stationary=v_sb[0:128, j, 0:D], moving=probT[0:128, j, 0:NQ])
    nisa.nc_matmul(dst=acc_o, stationary=vrow, moving=prow)  # + v_cur p_cur
    return acc_o


def _attn_bufs():
    """Two persistent SBUF buffer sets for the attention + router weights (layer li uses set li % 2): fixed
    allocations, so that the one-layer-ahead weight DMA waits only for layer li - 2's use of its set."""
    bufs = []
    for k in range(2 if PERSIST_AW else 0):
        bufs.append((nl.ndarray((128, QKV_F), dtype=nl.bfloat16, buffer=nl.sbuf),
                     nl.ndarray((128, WO_F // 2), dtype=nl.bfloat16, buffer=nl.sbuf),
                     nl.ndarray((P, HT * E), dtype=nl.bfloat16, buffer=nl.sbuf)))
    if SHARE_W:
        # R0 / R1 (bf16; their fp8 views hold experts 0, 1 of the layer whose attention set they held) and C (fp8,
        # experts 2, 3). Fixed addresses (address=) gave wrong tokens: the auto-allocator does not avoid them.
        regs = []
        regs.append(nl.ndarray((128, AW_B // 2), dtype=nl.bfloat16, buffer=nl.sbuf))
        regs.append(nl.ndarray((128, AW_B // 2), dtype=nl.bfloat16, buffer=nl.sbuf))
        regs.append(nl.ndarray((128, 2 * EXP_B), dtype=WDT, buffer=nl.sbuf))
        return regs, None
    ebufs = []
    for k in range(TOPK if PERSIST_EW else 0):  # expert weight buffers, one per selected expert
        ebufs.append((nl.ndarray((P, GU_F), dtype=WDT, buffer=nl.sbuf), nl.ndarray((P, DN_F), dtype=WDT, buffer=nl.sbuf)))
    return bufs, ebufs


def _load_wo(wo_sb, wo2, li, pid):
    WO2 = WO_F // 2  # wo2 [G, 2, 128, 4 * H]: head pairs (row (hh % 2) * 64 + d, column (hh // 2) * H + h)
    return nisa.dma_copy(dst=wo_sb, src=wo2.ap(pattern=[[WO2, 128], [1, WO2]], offset=(li * 2 + pid) * 128 * WO2), dge_mode=nisa.dge_mode.hwdge, engine=AW_ENG)


def _attn_w(wqkv, wo2, wr, li, pid, bufs):
    """Layer li's attention + router weights -> SBUF set li % 2 (wqkv [P, QKV_F], wo2 [128, 4 * H], wr [P, HT * E])."""
    if SHARE_W:
        r = bufs[li % 2]
        wqkv_sb = r[0:128, 0:QKV_F]
        wo_sb = r[0:128, QKV_F:QKV_F + WO_F // 2]
        wr_sb = r[0:P, QKV_F + WO_F // 2:QKV_F + WO_F // 2 + HT * E]
    elif PERSIST_AW:
        wqkv_sb, wo_sb, wr_sb = bufs[li % 2]
    else:
        wqkv_sb = nl.ndarray((128, QKV_F), dtype=nl.bfloat16, buffer=nl.sbuf)
        wo_sb = None if LATE_WO else nl.ndarray((128, WO_F // 2), dtype=nl.bfloat16, buffer=nl.sbuf)
        wr_sb = nl.ndarray((P, HT * E), dtype=nl.bfloat16, buffer=nl.sbuf)
    i1 = nisa.dma_copy(dst=wqkv_sb, src=wqkv.ap(pattern=[[QKV_F, 128], [1, QKV_F]], offset=(li * 2 + pid) * 128 * QKV_F), dge_mode=nisa.dge_mode.hwdge, engine=AW_ENG)
    if not LATE_WO:
        i2 = _load_wo(wo_sb, wo2, li, pid)
    i3 = nisa.dma_copy(dst=wr_sb, src=wr.ap(pattern=[[HT * E, P], [1, HT * E]], offset=li * P * HT * E), dge_mode=nisa.dge_mode.hwdge, engine=AW_ENG)
    if SHARE_W:
        return wqkv_sb, wo_sb, wr_sb, [i1, i2, i3]
    return wqkv_sb, wo_sb, wr_sb


def _layer(x_bf, li, ln1, wqkv, bqkv, sinks, kc, vc, kcw, vcw, wo2, bo, ln2, wr, br, wgu, bgu, wd, bd, lx,
           preg, sreg, lim_s, lim_f, csc, rmat_sb, ones_pp, sel2, pid, rg, sliding, B, tag, aw, nxt_l, bufs, ebufs,
           prev_end=None, mskT=None):
    """One decoder layer for one token. x_bf [P, HT] bf16 SBUF (the layer input) -> h2 [P, HT] bf16 SBUF.

    Layer tensors are stacked over a group of G layers (leading layer index li, a compile-time constant):
    ln1/ln2/bo [G, P, HT] fp32 (column layout); wqkv [G, 2, P, QKV_F] bf16; bqkv [G, 2, D, QKV_T] fp32;
    sinks [G, 2, NQ, 1] fp32; kc [G, 2, D, T] bf16 (K transposed); vc [G, 2, T, D] bf16 (always 4-D: the
    position-indexed DMAs name the position dimension); wo [G, 2, D, WO_F] bf16;
    wr [G, P, HT * E] bf16; br [G, 1, E] fp32; wgu [G * E, 2 * P * GU_F] bf16; bgu [G * E, 2 * P * GU_T] fp32;
    wd [G * E, 2 * P * DN_F] bf16; bd [G * E, H] fp32. wgu / wd are indexed with layer lx instead of li.
    sliding: window of the last 128 positions of a full-length cache; else static bucket of B.
    """

    # ---- weights that do not depend on the token: this layer's attention weights were issued one layer ahead
    # (aw); issue the next layer's now, ahead of this layer's expert weights in the DMA queue, so that they stream
    # while this layer's attention runs (the queue is idle then) instead of after this layer's experts
    if SHARE_W:
        wqkv_sb, wo_sb, wr_sb, _ = aw
    else:
        wqkv_sb, wo_sb, wr_sb = aw
    if LATE_WO:  # o_proj weights at the layer start (the DMA queues are idle during the attention)
        wo_sb = nl.ndarray((128, WO_F // 2), dtype=nl.bfloat16, buffer=nl.sbuf)
        _load_wo(wo_sb, wo2, li, pid)

    # ---- small constants
    ln1_sb = nl.ndarray((P, HT), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=ln1_sb, src=ln1.ap(pattern=[[HT, P], [1, HT]], offset=li * P * HT))
    ln2_sb = nl.ndarray((P, HT), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=ln2_sb, src=ln2.ap(pattern=[[HT, P], [1, HT]], offset=li * P * HT))
    bo_sb = nl.ndarray((P, HT), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=bo_sb, src=bo.ap(pattern=[[HT, P], [1, HT]], offset=li * P * HT))
    bqkv_sb = nl.ndarray((D, QKV_T), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=bqkv_sb, src=bqkv.ap(pattern=[[QKV_T, D], [1, QKV_T]], offset=(li * 2 + pid) * D * QKV_T))
    sink_sb = nl.ndarray((NQ, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=sink_sb, src=sinks.ap(pattern=[[1, NQ], [1, 1]], offset=(li * 2 + pid) * NQ))
    br_sb = nl.ndarray((1, E), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=br_sb, src=br.ap(pattern=[[E, 1], [1, E]], offset=li * E))

    if sliding:
        lim = lim_s
    else:
        lim = lim_f
    T = kc.shape[-1]

    # ---- K/V cache reads first: they do not wait for this step's cache write (written through the aliases
    # kcw / vcw of the same device tensors); the current position's slot is masked out below and the current token's
    # key and value are taken from SBUF instead
    NB = B // 128
    k_sb = nl.ndarray((D, B), dtype=nl.bfloat16, buffer=nl.sbuf)
    v_sb = nl.ndarray((128, NB, D), dtype=nl.bfloat16, buffer=nl.sbuf)
    if sliding:
        kv1 = nisa.dma_copy(dst=k_sb, src=kc.ap(pattern=[[T, D], [1, B]], offset=(li * 2 + pid) * D * T, scalar_offset=sreg, indirect_dim=3))
        kv2 = nisa.dma_copy(dst=v_sb, src=vc.ap(pattern=[[D, 128], [128 * D, NB], [1, D]], offset=(li * 2 + pid) * T * D,
                                          scalar_offset=sreg, indirect_dim=2))
    else:
        kv1 = nisa.dma_copy(dst=k_sb, src=kc.ap(pattern=[[T, D], [1, B]], offset=(li * 2 + pid) * D * T))
        kv2 = nisa.dma_copy(dst=v_sb, src=vc.ap(pattern=[[D, 128], [128 * D, NB], [1, D]], offset=(li * 2 + pid) * T * D))

    if SHARE_W and AW_EARLY:  # next layer's attention weights now (region (li + 1) % 2 is free: the previous layer's experts are done)
        aw_next = _attn_w(wqkv, wo2, wr, nxt_l, pid, bufs) if nxt_l is not None else None
        if aw_next is not None and prev_end is not None:
            # region (li + 1) % 2 is also the fp8 tensor of the previous layer's experts 0, 1 (an alias the
            # dependency tracking does not see): write it only after the previous layer's output
            for d_ in aw_next[3]:
                d_.depends_on(prev_end)
                if AW_AFTER_KV and not sliding:  # the K/V reads of a full layer are on the critical path
                    d_.depends_on(kv1)
                    d_.depends_on(kv2)

    # ---- input norm
    x = nl.ndarray((P, HT), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=x, src=x_bf)
    xn = _rmsnorm(x, ln1_sb, ones_pp, pad=True)

    # ---- QKV (weight-stationary, output [64, 10] column layout)
    # activation-stationary (wqkv [p][kt][t][n]: each kt's 640 outputs stream as the moving operand), output row
    # [1, (t, n)], exact fp32 transposes back to [n, t]
    if CT_QKV:
        # QKV with TensorE column tiling: tile c computes heads QG[c] (3, 3, 2, 2 of the 10 64-column heads; the
        # wqkv8 columns per kt are already head-contiguous), accumulating over kt in the same order (bitwise
        # identical); rows 32 c hold the results, 3 exact fp32 transposes give [64, head] columns
        xq32 = nl.ndarray((128, HT, 32), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.memset(dst=xq32, value=0.0)
        nisa.tensor_copy(dst=xq32[0:128, 0:HT, 0:1], src=xn.reshape((128, HT, 1)))
        acc_q4 = nl.ndarray((128, 512), dtype=nl.float32, buffer=nl.psum)
        for kt in range(HT):
            for c in range(4):
                h0, nh = QG[c]
                nisa.nc_matmul(dst=acc_q4[32 * c:32 * c + 32, 0:nh * D], stationary=xq32[0:128, kt, 0:32],
                               moving=wqkv_sb[0:128, kt * QKV_T * D + h0 * D:kt * QKV_T * D + (h0 + nh) * D],
                               tile_position=(0, 32 * c), tile_size=(128, 32))
        rq = nl.ndarray((128, 3 * D), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=rq[0:64, 0:3 * D], src=acc_q4[0:64, 0:3 * D])
        nisa.tensor_copy(dst=rq[64:128, 0:2 * D], src=acc_q4[64:128, 0:2 * D])
        nisa.memset(dst=rq[64:128, 2 * D:3 * D], value=0.0)
        tq = nl.ndarray((D, 3, 128), dtype=nl.float32, buffer=nl.psum)
        for j in range(3):
            nisa.nc_transpose(dst=tq[0:D, j, 0:97], data=rq[0:97, j * D:(j + 1) * D])
        tq4 = tq.reshape((D, 3, 4, 32))
        qkv = nl.ndarray((D, QKV_T), dtype=nl.bfloat16, buffer=nl.sbuf)
        for c in range(4):
            h0, nh = QG[c]
            nisa.tensor_tensor(dst=qkv[0:D, h0:h0 + nh], data1=tq4[0:D, 0:nh, c, 0], data2=bqkv_sb[0:D, h0:h0 + nh], op=nl.add)
    else:
        QH = QKV_T * D // 2
        acc_q = []
        for hh in range(2):
            acc_q.append(nl.ndarray((1, 512), dtype=nl.float32, buffer=nl.psum))
        for kt in range(HT):
            for hh in range(2):
                nisa.nc_matmul(dst=acc_q[hh][0:1, 0:QH], stationary=xn[0:128, kt:kt + 1],
                               moving=wqkv_sb[0:128, kt * QKV_T * D + hh * QH:kt * QKV_T * D + (hh + 1) * QH])
        qkv_row = nl.ndarray((1, QKV_T * D), dtype=nl.float32, buffer=nl.sbuf)
        for hh in range(2):
            nisa.tensor_copy(dst=qkv_row[0:1, hh * QH:(hh + 1) * QH], src=acc_q[hh][0:1, 0:QH])
        acc_qkv = nl.ndarray((D, QKV_T), dtype=nl.float32, buffer=nl.psum)
        for t in range(QKV_T):
            nisa.nc_transpose(dst=acc_qkv[0:D, t:t + 1], data=qkv_row[0:1, t * D:(t + 1) * D])
        qkv = nl.ndarray((D, QKV_T), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=qkv, data1=acc_qkv, data2=bqkv_sb, op=nl.add)
    # RoPE on q heads and k: x * cos + rotate_half(x) * sin
    rot = nl.ndarray((D, NQ + 1), dtype=nl.float32, buffer=nl.psum)
    nisa.nc_matmul(dst=rot, stationary=rmat_sb, moving=qkv[0:D, 0:NQ + 1])
    t1 = nl.ndarray((D, NQ + 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=t1, data=qkv[0:D, 0:NQ + 1], op0=nl.multiply, operand0=csc[0:D, 0:1])
    t2 = nl.ndarray((D, NQ + 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=t2, data=rot, op0=nl.multiply, operand0=csc[0:D, 1:2])
    qk = nl.ndarray((D, NQ + 1), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=qk, data1=t1, data2=t2, op=nl.add)

    # ---- KV cache write at pos (through the aliases; read by the next steps only)
    kslot = kcw.ap(pattern=[[T, D], [1, 1]], offset=(li * 2 + pid) * D * T, scalar_offset=preg, indirect_dim=3)
    nisa.dma_copy(dst=kslot, src=qk[0:D, NQ:NQ + 1])
    vslot = vcw.ap(pattern=[[1, D], [1, 1]], offset=(li * 2 + pid) * T * D, scalar_offset=preg, indirect_dim=2)
    nisa.dma_copy(dst=vslot, src=qkv[0:D, NQ + 1:NQ + 2])

    # ---- attention: the B cached positions before pos (slot of pos masked) + the current token from SBUF
    scur_ps = nl.ndarray((NQ, 1), dtype=nl.float32, buffer=nl.psum)
    nisa.nc_matmul(dst=scur_ps, stationary=qk[0:D, 0:NQ], moving=qk[0:D, NQ:NQ + 1])
    scur_bf = nl.ndarray((NQ, 1), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.tensor_copy(dst=scur_bf, src=scur_ps)
    scur = nl.ndarray((NQ, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=scur, data=scur_bf, op0=nl.multiply, operand0=0.125)
    vrow_ps = nl.ndarray((1, D), dtype=nl.bfloat16, buffer=nl.psum)
    nisa.nc_transpose(dst=vrow_ps, data=qkv[0:D, NQ + 1:NQ + 2])
    vrow = nl.ndarray((1, D), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.tensor_copy(dst=vrow, src=vrow_ps)

    if mskT is not None:
        acc_o = _attn_keys_on_partitions(qk, k_sb, v_sb, mskT, scur, sink_sb, vrow, B)
    else:
        s_bf = nl.ndarray((NQ, B), dtype=nl.bfloat16, buffer=nl.sbuf)
        # score chunk (one PSUM bank), the largest of 512 / 384 / 256 / 128 dividing B
        if B % 512 == 0:
            CH = 512
        elif B % 384 == 0:
            CH = 384
        elif B % 256 == 0:
            CH = 256
        else:
            CH = 128
        for c in range(B // CH):
            sp = nl.ndarray((NQ, CH), dtype=nl.float32, buffer=nl.psum)
            nisa.nc_matmul(dst=sp, stationary=qk[0:D, 0:NQ], moving=k_sb[0:D, c * CH:(c + 1) * CH])
            nisa.tensor_copy(dst=s_bf[0:NQ, c * CH:(c + 1) * CH], src=sp)
        # mask: cached slot j valid iff j < lim (lim = pos, or pos - s in the sliding window: the slot of pos itself is
        # stale here, the current token is added separately)
        iot = nl.ndarray((NQ, B), dtype=nl.float32, buffer=nl.sbuf)
        nisa.iota(dst=iot, pattern=[[1, B]], offset=0, channel_multiplier=0)
        msk = nl.ndarray((NQ, B), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=msk, data=iot, op0=nl.greater_equal, operand0=lim, op1=nl.multiply, operand1=NEG)
        s = nl.ndarray((NQ, B), dtype=nl.float32, buffer=nl.sbuf)
        # s = s_bf * 0.125 + msk
        nisa.scalar_tensor_tensor(dst=s, data=s_bf, op0=nl.multiply, operand0=0.125, op1=nl.add, operand1=msk)
        m1 = nl.ndarray((NQ, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_reduce(dst=m1, op=nl.maximum, data=s, axis=1)
        m0 = nl.ndarray((NQ, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=m0, data1=m1, data2=scur, op=nl.maximum)
        m = nl.ndarray((NQ, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=m, data1=m0, data2=sink_sb, op=nl.maximum)
        negm = nl.ndarray((NQ, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=negm, data=m, op0=nl.multiply, operand0=-1.0)
        pexp = nl.ndarray((NQ, B), dtype=nl.float32, buffer=nl.sbuf)
        psum_ = nl.ndarray((NQ, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.activation_reduce(dst=pexp, op=nl.exp, data=s, reduce_op=nl.add, reduce_res=psum_, bias=negm, scale=1.0)
        esink = nl.ndarray((NQ, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.activation(dst=esink, op=nl.exp, data=sink_sb, bias=negm, scale=1.0)
        ecur = nl.ndarray((NQ, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.activation(dst=ecur, op=nl.exp, data=scur, bias=negm, scale=1.0)
        tot0 = nl.ndarray((NQ, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=tot0, data1=psum_, data2=ecur, op=nl.add)
        tot = nl.ndarray((NQ, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=tot, data1=tot0, data2=esink, op=nl.add)
        rtot = nl.ndarray((NQ, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.reciprocal(dst=rtot, data=tot)
        prob = nl.ndarray((NQ, B), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=prob, data=pexp, op0=nl.multiply, operand0=rtot)
        pcur = nl.ndarray((NQ, 1), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=pcur, data=ecur, op0=nl.multiply, operand0=rtot)
        prow_ps = nl.ndarray((1, NQ), dtype=nl.bfloat16, buffer=nl.psum)
        nisa.nc_transpose(dst=prow_ps, data=pcur)
        prow = nl.ndarray((1, NQ), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.tensor_copy(dst=prow, src=prow_ps)
        # P^T tiles and PV: O^T [D, NQ] = sum_t V[t, :]^T P[t, :]
        acc_o = nl.ndarray((D, NQ), dtype=nl.float32, buffer=nl.psum)
        for j in range(NB):
            ptp = nl.ndarray((128, NQ), dtype=nl.bfloat16, buffer=nl.psum)
            nisa.nc_transpose(dst=ptp, data=prob[0:NQ, j * 128:(j + 1) * 128])
            pts = nl.ndarray((128, NQ), dtype=nl.bfloat16, buffer=nl.sbuf)
            nisa.tensor_copy(dst=pts, src=ptp)
            nisa.nc_matmul(dst=acc_o, stationary=v_sb[0:128, j, 0:D], moving=pts)
        nisa.nc_matmul(dst=acc_o, stationary=vrow, moving=prow)  # + v_cur p_cur
    ao = nl.ndarray((D, NQ), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.tensor_copy(dst=ao, src=acc_o)

    # ---- o_proj (K split over vcores) + all-reduce + bias + residual
    # activation-stationary: per head the attention output column is stationary and wo (row h = nt * P + n
    # contiguous per head) streams; output row [1, H], exact fp32 transposes back to the column layout
    # (chunks interleaved over OG PSUM banks: consecutive matmuls into one bank serialize, ~286 vs ~80 ns apart)
    # head pairs stacked on 128 partitions (a 64-row contraction streams at half the rate): ao2[(hh % 2) * 64 + d,
    # hh // 2] = ao[d, hh], by two exact 0/1 selection matmuls
    ao3 = ao.reshape((D, NQ // 2, 2))
    ao2_ps = nl.ndarray((128, NQ // 2), dtype=nl.float32, buffer=nl.psum)
    nisa.nc_matmul(dst=ao2_ps, stationary=sel2[0:D, 0, 0:128], moving=ao3[0:D, 0:NQ // 2, 0])
    nisa.nc_matmul(dst=ao2_ps, stationary=sel2[0:D, 1, 0:128], moving=ao3[0:D, 0:NQ // 2, 1])
    ao2 = nl.ndarray((128, NQ // 2), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.tensor_copy(dst=ao2, src=ao2_ps)
    part = nl.ndarray((P, HT), dtype=nl.float32, buffer=nl.sbuf)
    if CT_OPROJ:
        # o_proj with column tiling: tile c computes output tiles nt = 4 jj + c (as CT_DN), accumulating over the
        # head pairs in the same order (bitwise identical); 6 exact fp32 transposes give the column layout
        ao32 = nl.ndarray((128, NQ // 2, 32), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.memset(dst=ao32, value=0.0)
        nisa.tensor_copy(dst=ao32[0:128, 0:NQ // 2, 0:1], src=ao2.reshape((128, NQ // 2, 1)))
        wo5 = wo_sb.reshape((128, NQ // 2, 6, 4, P))
        acco = []
        for h in range(2):
            acco.append(nl.ndarray((128, 512), dtype=nl.float32, buffer=nl.psum))
        for hp in range(NQ // 2):
            for h in range(2):
                for c in range(4):
                    nisa.nc_matmul(dst=acco[h][32 * c:32 * c + 32, 0:3 * P], stationary=ao32[0:128, hp, 0:32],
                                   moving=wo5[0:128, hp, 3 * h:3 * h + 3, c, 0:P],
                                   tile_position=(0, 32 * c), tile_size=(128, 32))
        po4 = part.reshape((P, 2, 3, 4))
        for h in range(2):
            ro = nl.ndarray((97, 3 * P), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_copy(dst=ro, src=acco[h][0:97, 0:3 * P])
            tpo = nl.ndarray((P, 3, 128), dtype=nl.float32, buffer=nl.psum)
            for j in range(3):
                nisa.nc_transpose(dst=tpo[0:P, j, 0:97], data=ro[0:97, j * P:(j + 1) * P])
            nisa.tensor_copy(dst=po4[0:P, h, 0:3, 0:4], src=tpo.reshape((P, 3, 4, 32))[0:P, 0:3, 0:4, 0])
    else:
        OC = 480
        OG = 3
        o_row = nl.ndarray((1, H), dtype=nl.float32, buffer=nl.sbuf)
        for g in range(H // OC // OG):
            acc_os = []
            for c in range(OG):
                acc_os.append(nl.ndarray((1, 512), dtype=nl.float32, buffer=nl.psum))
            for hp in range(NQ // 2):
                for c in range(OG):
                    cc = g * OG + c
                    nisa.nc_matmul(dst=acc_os[c][0:1, 0:OC], stationary=ao2[0:128, hp:hp + 1],
                                   moving=wo_sb[0:128, hp * H + cc * OC:hp * H + (cc + 1) * OC])
            for c in range(OG):
                cc = g * OG + c
                nisa.tensor_copy(dst=o_row[0:1, cc * OC:(cc + 1) * OC], src=acc_os[c][0:1, 0:OC])
        acc_p = nl.ndarray((P, HT), dtype=nl.float32, buffer=nl.psum)
        for nt in range(HT):
            nisa.nc_transpose(dst=acc_p[0:P, nt:nt + 1], data=o_row[0:1, nt * P:(nt + 1) * P])
        nisa.tensor_copy(dst=part, src=acc_p)
    red = _allreduce(part, HT, pid, rg, "ar_attn" + tag, sbuf=AR_SBUF)
    attn = nl.ndarray((P, HT), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=attn, data1=red, data2=bo_sb, op=nl.add)
    h1_bf = nl.ndarray((P, HT), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=h1_bf, data1=x_bf, data2=attn, op=nl.add)
    h1 = nl.ndarray((P, HT), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=h1, src=h1_bf)

    # ---- post-attention norm + router
    xn2 = _rmsnorm(h1, ln2_sb, ones_pp, pad=True)
    acc_r = nl.ndarray((1, E), dtype=nl.float32, buffer=nl.psum)
    for kt in range(HT):
        nisa.nc_matmul(dst=acc_r, stationary=xn2[0:P, kt:kt + 1], moving=wr_sb[0:P, kt * E:(kt + 1) * E])
    lg_bf = nl.ndarray((1, E), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=lg_bf, data1=acc_r, data2=br_sb, op=nl.add)
    cur = nl.ndarray((1, E), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=cur, src=lg_bf)
    # top-4, ties to the lowest expert id
    io = nl.ndarray((1, E), dtype=nl.float32, buffer=nl.sbuf)
    nisa.iota(dst=io, pattern=[[1, E]], offset=0)
    rev = nl.ndarray((1, E), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=rev, data=io, op0=nl.multiply, operand0=-1.0, op1=nl.add, operand1=64.0)
    idxf = nl.ndarray((1, TOPK), dtype=nl.float32, buffer=nl.sbuf)
    vals = nl.ndarray((1, TOPK), dtype=nl.float32, buffer=nl.sbuf)
    if TOP8:
        # max8: the 8 largest values in descending order; nc_find_index8 pairs duplicate values with their positions
        # in ascending order, so ties go to the lowest expert id first (the same selection and order as below)
        v8 = nl.ndarray((1, 8), dtype=nl.float32, buffer=nl.sbuf)
        nisa.max8(dst=v8, src=cur)
        i8 = nl.ndarray((1, 8), dtype=nl.uint32, buffer=nl.sbuf)
        nisa.nc_find_index8(dst=i8, data=cur, vals=v8)
        nisa.tensor_copy(dst=vals, src=v8[0:1, 0:TOPK])
        idxi = nl.ndarray((1, TOPK), dtype=nl.int32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=idxi, src=i8[0:1, 0:TOPK])
    for i in range(0 if TOP8 else TOPK):
        mx = nl.ndarray((1, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_reduce(dst=mx, op=nl.maximum, data=cur, axis=1)
        nisa.tensor_copy(dst=vals[0:1, i:i + 1], src=mx)
        sc = nl.ndarray((1, E), dtype=nl.float32, buffer=nl.sbuf)
        nisa.scalar_tensor_tensor(dst=sc, data=cur, op0=nl.equal, operand0=mx, op1=nl.multiply, operand1=rev)
        sm = nl.ndarray((1, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_reduce(dst=sm, op=nl.maximum, data=sc, axis=1)
        nisa.tensor_scalar(dst=idxf[0:1, i:i + 1], data=sm, op0=nl.multiply, operand0=-1.0, op1=nl.add, operand1=64.0)
        hit = nl.ndarray((1, E), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=hit, data=rev, op0=nl.equal, operand0=sm, op1=nl.multiply, operand1=NEG)
        nxt = nl.ndarray((1, E), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=nxt, data1=cur, data2=hit, op=nl.add)
        cur = nxt
    if not TOP8:
        idxi = nl.ndarray((1, TOPK), dtype=nl.int32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=idxi, src=idxf)
    # routing weights: softmax over the 4 selected (bf16 out)
    negv = nl.ndarray((1, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=negv, data=vals[0:1, 0:1], op0=nl.multiply, operand0=-1.0)
    ev = nl.ndarray((1, TOPK), dtype=nl.float32, buffer=nl.sbuf)
    esum = nl.ndarray((1, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.activation_reduce(dst=ev, op=nl.exp, data=vals, reduce_op=nl.add, reduce_res=esum, bias=negv, scale=1.0)
    rs = nl.ndarray((1, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.reciprocal(dst=rs, data=esum)
    wbf = nl.ndarray((1, TOPK), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=wbf, data=ev, op0=nl.multiply, operand0=rs)
    wf = nl.ndarray((1, TOPK), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=wf, src=wbf)
    ones_1p = _ones((1, P))
    wb_ps = nl.ndarray((P, TOPK), dtype=nl.float32, buffer=nl.psum)
    nisa.nc_matmul(dst=wb_ps, stationary=ones_1p, moving=wf)
    wb = nl.ndarray((P, TOPK), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=wb, src=wb_ps)

    # ---- experts: this vcore's intermediate slice of each selected expert
    if CT_GU:  # x zero-padded to 32 stationary columns per kt (column-tiled gate/up)
        x32 = nl.ndarray((128, HT, 32), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.memset(dst=x32, value=0.0)
        nisa.tensor_copy(dst=x32[0:128, 0:HT, 0:1], src=xn2.reshape((128, HT, 1)))
    epart = nl.ndarray((P, TOPK * HT), dtype=nl.float32, buffer=nl.sbuf)
    eps = []
    ereds = []
    for i in range(TOPK if AR_SPLIT else 0):
        eps.append(nl.ndarray((P, HT), dtype=nl.float32, buffer=nl.sbuf))
    bds = []
    for i in range(TOPK):
        ereg = nisa.register_alloc()
        nisa.register_load(ereg, idxi[0:1, i:i + 1])
        if SHARE_W:
            if i < 2:
                evw = bufs[li % 2].view(WDT)
            else:
                evw = bufs[2]
            e0 = (i % 2) * EXP_B
            gu_sb = evw[0:128, e0:e0 + GU_F]
            dn_sb = evw[0:128, e0 + GU_F:e0 + EXP_B]
        elif PERSIST_EW:
            gu_sb, dn_sb = ebufs[i]
        else:
            gu_sb = nl.ndarray((128, GU_F), dtype=WDT, buffer=nl.sbuf)  # [p][kt][t][n], rows >= P zero
            dn_sb = nl.ndarray((128, DN_F), dtype=WDT, buffer=nl.sbuf)
        for c in range(GU_DMA):  # GU_DMA DMAs of HT / GU_DMA kt each (the order the GEMV consumes them)
            F = GU_F // GU_DMA
            ins_ = nisa.dma_copy(dst=gu_sb[0:128, c * F:(c + 1) * F],
                          src=wgu.ap(pattern=[[GU_F, 128], [1, F]], offset=lx * E * 2 * 128 * GU_F + pid * 128 * GU_F + c * F,
                                     scalar_offset=ereg, indirect_dim=0), dge_mode=nisa.dge_mode.hwdge, engine=EXP_ENG[i % 2])
            if SHARE_W and AW_EARLY and AW_DEP and i == 0 and c == 0 and aw_next is not None:
                for d_ in aw_next[3]:
                    ins_.depends_on(d_)
        nisa.dma_copy(dst=dn_sb, src=wd.ap(pattern=[[DN_F, 128], [1, DN_F]], offset=lx * E * 2 * 128 * DN_F + pid * 128 * DN_F,
                                           scalar_offset=ereg, indirect_dim=0), dge_mode=nisa.dge_mode.hwdge, engine=EXP_ENG[i % 2])
        bgu_sb = nl.ndarray((P, GU_T), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=bgu_sb, src=bgu.ap(pattern=[[GU_T, P], [1, GU_T]], offset=li * E * 2 * P * GU_T + pid * P * GU_T,
                                             scalar_offset=ereg, indirect_dim=0))
        bd_sb = nl.ndarray((P, HT), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=bd_sb, src=bd.ap(pattern=[[1, P], [P, HT]], offset=li * E * H, scalar_offset=ereg, indirect_dim=0))
        bds.append(bd_sb)

        if CT_GU:
            # gate/up with TensorE column tiling: column tile it (32 PE columns, x zero-padded to 32 stationary
            # columns) computes intermediate tile it's 240 outputs [gate 120 | up 120] (wgu8 per kt: [it][up][n]);
            # the three tiles run in parallel (measured 4 experts 30.8 -> ~8 us of TensorE, scratch/mb/ct2.py).
            # Same per-output accumulation order over kt as before (bitwise identical); rows 32 it of acc hold the
            # results, one exact fp32 transpose per [gate | up] half gives the column layout.
            acc_c = nl.ndarray((128, 512), dtype=nl.float32, buffer=nl.psum)
            for kt in range(HT):
                for it in range(IT):
                    nisa.nc_matmul(dst=acc_c[32 * it:32 * it + 32, 0:2 * P], stationary=x32[0:128, kt, 0:32],
                                   moving=gu_sb[0:128, kt * GU_T * P + it * 2 * P:kt * GU_T * P + (it + 1) * 2 * P],
                                   tile_position=(0, 32 * it), tile_size=(128, 32))
            r3 = nl.ndarray((65, 2 * P), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_copy(dst=r3, src=acc_c[0:65, 0:2 * P])
            tp = nl.ndarray((P, 2, 96), dtype=nl.float32, buffer=nl.psum)
            for j in range(2):
                nisa.nc_transpose(dst=tp[0:P, j, 0:65], data=r3[0:65, j * P:(j + 1) * P])
            acc_gu = tp.reshape((P, 2, IT, 32))[0:P, 0:2, 0:IT, 0]
            bgu_v = bgu_sb.reshape((P, 2, IT))
        else:
            # gate/up, activation-stationary (x column stationary, the weights stream as the moving operand: half the
            # matrix-engine time of 144 weight-stationary 120 x 120 matvecs), output row [1, (t, n)], then exact fp32
            # transposes back to the column layout [n, t]
            GH = GU_T * P // 2  # 360 outputs per PSUM bank
            acc_r = []
            for hh in range(2):
                acc_r.append(nl.ndarray((1, 512), dtype=nl.float32, buffer=nl.psum))
            for kt in range(HT):
                for hh in range(2):
                    nisa.nc_matmul(dst=acc_r[hh][0:1, 0:GH], stationary=xn2[0:128, kt:kt + 1],
                                   moving=gu_sb[0:128, kt * GU_T * P + hh * GH:kt * GU_T * P + (hh + 1) * GH])
            gu_row = nl.ndarray((1, GU_T * P), dtype=nl.float32, buffer=nl.sbuf)
            for hh in range(2):
                nisa.tensor_copy(dst=gu_row[0:1, hh * GH:(hh + 1) * GH], src=acc_r[hh][0:1, 0:GH])
            acc_gu = nl.ndarray((P, GU_T), dtype=nl.float32, buffer=nl.psum)
            for t in range(GU_T):
                nisa.nc_transpose(dst=acc_gu[0:P, t:t + 1], data=gu_row[0:1, t * P:(t + 1) * P])
            bgu_v = bgu_sb
        gu = nl.ndarray((P, GU_T), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=gu.reshape((P, 2, IT)) if CT_GU else gu, data1=acc_gu, data2=bgu_v, op=nl.add)
        gate = nl.ndarray((P, IT), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=gate, data=gu[0:P, 0:IT], op0=nl.minimum, operand0=SWIGLU_LIMIT)
        up = nl.ndarray((P, IT), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=up, data=gu[0:P, IT:GU_T], op0=nl.minimum, operand0=SWIGLU_LIMIT,
                           op1=nl.maximum, operand1=-SWIGLU_LIMIT)
        ga = nl.ndarray((P, IT), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=ga, data=gate, op0=nl.multiply, operand0=ALPHA)
        sg = nl.ndarray((P, IT), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.activation(dst=sg, op=nl.sigmoid, data=ga)
        glu = nl.ndarray((P, IT), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=glu, data1=gate, data2=sg, op=nl.multiply)
        up1 = nl.ndarray((P, IT), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=up1, data=up, op0=nl.add, operand0=1.0)
        inter = nl.ndarray((128, IT), dtype=nl.bfloat16, buffer=nl.sbuf)  # rows >= P zero (128-row down weights)
        nisa.memset(dst=inter, value=0.0)
        nisa.tensor_tensor(dst=inter[0:P, 0:IT], data1=up1, data2=glu, op=nl.multiply)

        if CT_DN:
            # down, activation-stationary with column tiling: tile c computes output tiles nt = 4 jj + c (jj 0..5, two
            # PSUM banks of 3 x 120), accumulating over it in the same order as before (bitwise identical); rows 32 c
            # hold the results, 6 exact fp32 transposes give the column layout
            i32 = nl.ndarray((128, IT, 32), dtype=nl.bfloat16, buffer=nl.sbuf)
            nisa.memset(dst=i32, value=0.0)
            nisa.tensor_copy(dst=i32[0:128, 0:IT, 0:1], src=inter.reshape((128, IT, 1)))
            dn5 = dn_sb.reshape((128, IT, 6, 4, P))
            accd = []
            for h in range(2):
                accd.append(nl.ndarray((128, 512), dtype=nl.float32, buffer=nl.psum))
            for it in range(IT):
                for h in range(2):
                    for c in range(4):
                        nisa.nc_matmul(dst=accd[h][32 * c:32 * c + 32, 0:3 * P], stationary=i32[0:128, it, 0:32],
                                       moving=dn5[0:128, it, 3 * h:3 * h + 3, c, 0:P],
                                       tile_position=(0, 32 * c), tile_size=(128, 32))
            ep4 = (eps[i] if AR_SPLIT else epart[0:P, i * HT:(i + 1) * HT]).reshape((P, 2, 3, 4))
            for h in range(2):
                rd = nl.ndarray((97, 3 * P), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_copy(dst=rd, src=accd[h][0:97, 0:3 * P])
                tpd = nl.ndarray((P, 3, 128), dtype=nl.float32, buffer=nl.psum)
                for j in range(3):
                    nisa.nc_transpose(dst=tpd[0:P, j, 0:97], data=rd[0:97, j * P:(j + 1) * P])
                nisa.tensor_copy(dst=ep4[0:P, h, 0:3, 0:4], src=tpd.reshape((P, 3, 4, 32))[0:P, 0:3, 0:4, 0])
        else:
            acc_d = nl.ndarray((P, HT), dtype=nl.float32, buffer=nl.psum)
            for nt in range(HT):
                for it in range(IT):
                    nisa.nc_matmul(dst=acc_d[0:P, nt:nt + 1], stationary=dn_sb[0:128, (it * HT + nt) * P:(it * HT + nt + 1) * P],
                                   moving=inter[0:128, it:it + 1])
            nisa.tensor_copy(dst=eps[i] if AR_SPLIT else epart[0:P, i * HT:(i + 1) * HT], src=acc_d)
        if AR_SPLIT and i < TOPK - 1:  # all-reduce each expert's partial as soon as it is complete (SBUF form)
            ereds.append(_allreduce(eps[i], HT, pid, rg, "ar_moe" + str(i) + tag, sbuf=True))

    # the next layer's attention weights: in the Sync queue after this layer's expert weights and before the
    # all-reduce's DMAs (which wait for the collective), so they stream while the down GEMVs and the all-reduce run
    if not (SHARE_W and AW_EARLY):
        aw_next = _attn_w(wqkv, wo2, wr, nxt_l, pid, bufs) if nxt_l is not None else None
    if AR_SPLIT:
        ereds.append(_allreduce(eps[TOPK - 1], HT, pid, rg, "ar_moe" + str(TOPK - 1) + tag, sbuf=True))
    else:
        ered = _allreduce(epart, TOPK * HT, pid, rg, "ar_moe" + tag, sbuf=AR_SBUF_MOE)
    moe = nl.ndarray((P, HT), dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(dst=moe, value=0.0)
    for i in range(TOPK):
        y = nl.ndarray((P, HT), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=y, data1=ereds[i] if AR_SPLIT else ered[0:P, i * HT:(i + 1) * HT], data2=bds[i], op=nl.add)
        z = nl.ndarray((P, HT), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=z, data=y, op0=nl.multiply, operand0=wb[0:P, i:i + 1])
        m2 = nl.ndarray((P, HT), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=m2, data1=moe, data2=z, op=nl.add)
        moe = m2
    moe_bf = nl.ndarray((P, HT), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.tensor_copy(dst=moe_bf, src=moe)
    h2 = nl.ndarray((P, HT), dtype=nl.bfloat16, buffer=nl.sbuf)
    end_ins = nisa.tensor_tensor(dst=h2, data1=h1_bf, data2=moe_bf, op=nl.add)
    return h2, aw_next, end_ins



@nki.jit
def decode_layer(h, pos, ln1, wqkv, bqkv, cs, rmat, sinks, kc, vc, kcw, vcw, wo2, bo, ln2, wr, br,
                 wgu, bgu, wd, bd, rg, sliding, B):
    """One decoder layer (unstacked tensors, G = 1), h [1, H] bf16 -> h_out [1, H] bf16. Kernel tests only."""
    pid = nl.program_id(0)
    h_out = nl.ndarray((1, H), dtype=nl.bfloat16, buffer=nl.shared_hbm)
    p_sb, preg, sreg, lim_s, lim_f, csc = _pos_ctx(pos, cs, sliding, not sliding)
    rmat_sb = nl.ndarray((D, D), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.dma_copy(dst=rmat_sb, src=rmat)
    ones_pp = _ones((P, P))
    sel2 = _sel2()
    x_bf = _load_col(h, nl.bfloat16)
    bufs, ebufs = _attn_bufs()
    h2, _, _ = _layer(x_bf, 0, ln1, wqkv, bqkv, sinks, kc, vc, kcw, vcw, wo2, bo, ln2, wr, br, wgu, bgu, wd, bd, 0,
                   preg, sreg, lim_s, lim_f, csc, rmat_sb, ones_pp, sel2, pid, rg, sliding, B, "",
                   _attn_w(wqkv, wo2, wr, 0, pid, bufs), None, bufs, ebufs)
    _store_col(h_out, h2)
    return h_out



@nki.jit
def embed(toks, pos, emb):
    """h [1, H] bf16 = emb[toks[0, pos]]."""
    h_out = nl.ndarray((1, H), dtype=nl.bfloat16, buffer=nl.shared_hbm)
    p_sb = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.sbuf)
    nisa.dma_copy(dst=p_sb, src=pos)
    preg = nisa.register_alloc()
    nisa.register_load(preg, p_sb)
    t_sb = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.sbuf)
    nisa.dma_copy(dst=t_sb, src=toks.ap(pattern=[[1, 1], [1, 1]], offset=0, scalar_offset=preg, indirect_dim=1))
    treg = nisa.register_alloc()
    nisa.register_load(treg, t_sb)
    row = nl.ndarray((1, H), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.dma_copy(dst=row, src=emb.ap(pattern=[[H, 1], [1, H]], offset=0, scalar_offset=treg, indirect_dim=0))
    nisa.dma_copy(dst=h_out, src=row)
    return h_out


@nki.jit
def inc(pos):
    """pos_out = pos + 1 (prefill steps that need no logits)."""
    p_sb = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.sbuf)
    nisa.dma_copy(dst=p_sb, src=pos)
    p1 = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=p1, data=p_sb, op0=nl.add, operand0=1.0)
    pos_out = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.shared_hbm)
    nisa.dma_copy(dst=pos_out, src=p1)
    return pos_out


def _transpose_col(x_col):
    """[128, 1] fp32 -> [1, 128] fp32 SBUF (TensorE transpose, exact)."""
    tp = nl.ndarray((1, 128), dtype=nl.float32, buffer=nl.psum)
    nisa.nc_transpose(dst=tp, data=x_col)
    r = nl.ndarray((1, 128), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=r, src=tp)
    return r


def _allreduce_max_scalar(v11, rg, name):
    """[1, 1] fp32 SBUF -> max over ranks [1, 1] (replicated over 8 columns in an HBM buffer)."""
    rep = nl.ndarray((1, 8), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=rep, data=_ones((1, 8)), op0=nl.multiply, operand0=v11)
    if AR_GATHER_HEAD:  # SBUF all-gather + local max (exact): no gain measured (239 -> 241 ms / 64 tokens)
        gat = nl.ndarray((1, 32), dtype=nl.float32, buffer=nl.sbuf)
        ncc.all_gather(srcs=[rep], dsts=[gat], replica_group=rg, collective_dim=1)
        res = nl.ndarray((1, 8), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_reduce(dst=res[0:1, 0:1], op=nl.maximum, data=gat, axis=1)
        return res
    src = nl.ndarray((1, 8), dtype=nl.float32, buffer=nl.shared_hbm, name=name + "_src")
    dst = nl.ndarray((1, 8), dtype=nl.float32, buffer=nl.shared_hbm, name=name + "_dst")
    nisa.dma_copy(dst=src, src=rep)
    nisa.core_barrier(data=src, cores=(0, 1))
    ncc.all_reduce(srcs=[src], dsts=[dst], op=nl.maximum, replica_group=rg)
    res = nl.ndarray((1, 8), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=res, src=dst)
    return res


@nki.jit
def head(h, pos, toks, lnf, wlm, lmbias, lmscore, rg):
    """Final norm, LM head (vocab split over the 8 vcores), greedy argmax (ties -> lowest id) over bf16 logits;
    toks[0, pos + 1] = argmax (in place); returns pos + 1.

    lnf [P, HT] fp32; wlm [2 * NVT / LM_G, 128 * LM_F] bf16 (row k % 128 of tile k // 128); lmbias / lmscore [2, 128, NVT] fp32 (0 / -3e38 for
    padding; BIG - token id)."""
    pid = nl.program_id(0)
    ones_pp = _ones((P, P))
    x_bf = _load_col(h, nl.bfloat16)
    return _head(x_bf, pos, toks, lnf, wlm, lmbias, lmscore, rg, pid, ones_pp)


def _head(x_bf, pos, toks, lnf, wlm, lmbias, lmscore, rg, pid, ones_pp):
    NG = NVT // LM_G
    assert NG % 2 == 0 and (NG // 2) * LM_G == NVT // 2
    lnf_sb = nl.ndarray((P, HT), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=lnf_sb, src=lnf)
    bias_sb = nl.ndarray((128, NVT), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=bias_sb, src=lmbias.ap(pattern=[[NVT, 128], [1, NVT]], offset=pid * 128 * NVT))
    score_sb = nl.ndarray((128, NVT), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=score_sb, src=lmscore.ap(pattern=[[NVT, 128], [1, NVT]], offset=pid * 128 * NVT))
    x = nl.ndarray((P, HT), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=x, src=x_bf)
    xn = _rmsnorm(x, lnf_sb, ones_pp)
    # hidden vector in 128-row tiles (k = kt * 128 + p; rows >= H zero) through a small HBM round trip
    xs = nl.ndarray((1, H), dtype=nl.bfloat16, buffer=nl.shared_hbm)
    _store_col(xs, xn)
    xn2 = nl.ndarray((128, HT2), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.memset(dst=xn2, value=0.0)
    nisa.dma_copy(dst=xn2[0:128, 0:HT2 - 1], src=xs.ap(pattern=[[1, 128], [128, HT2 - 1]], offset=0))
    nisa.dma_copy(dst=xn2[0:H - (HT2 - 1) * 128, HT2 - 1:HT2], src=xs.ap(pattern=[[1, H - (HT2 - 1) * 128], [128, 1]], offset=(HT2 - 1) * 128))

    lg = nl.ndarray((128, NVT), dtype=nl.float32, buffer=nl.sbuf)
    HALF = NVT // 2
    if CT_HEAD:
        # activation-stationary with TensorE column tiling: the hidden tile kt (zero-padded to 32 stationary
        # columns) is the stationary operand, column tile j computes the group's vocab tile j (128 moving columns),
        # accumulating over kt in the same order as the weight-stationary form (same per-logit fp32 sums); rows 32 j
        # of the PSUM tile hold the logits, one exact fp32 transpose per group gives the column layout [128, tile].
        assert LM_G == 4
        xp32 = nl.ndarray((128, HT2, 32), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.memset(dst=xp32, value=0.0)
        nisa.tensor_copy(dst=xp32[0:128, 0:HT2, 0:1], src=xn2.reshape((128, HT2, 1)))
        GB = 2  # groups per transposed PSUM tile
        assert NG % GB == 0
        for gb in range(NG // GB):
            tps = nl.ndarray((128, GB, 128), dtype=nl.float32, buffer=nl.psum)
            for gg in range(GB):
                grp = gb * GB + gg
                w = nl.ndarray((128, LM_F), dtype=nl.bfloat16, buffer=nl.sbuf)
                nisa.dma_copy(dst=w, src=wlm.ap(pattern=[[LM_F, 128], [1, LM_F]], offset=(pid * NG + grp) * 128 * LM_F), dge_mode=nisa.dge_mode.hwdge, engine=nisa.engine.sync)
                acc = nl.ndarray((128, 128), dtype=nl.float32, buffer=nl.psum)
                for kt in range(HT2):
                    for j in range(LM_G):
                        nisa.nc_matmul(dst=acc[32 * j:32 * j + 32, 0:128], stationary=xp32[0:128, kt, 0:32],
                                       moving=w[0:128, (j * HT2 + kt) * 128:(j * HT2 + kt + 1) * 128],
                                       tile_position=(0, 32 * j), tile_size=(128, 32))
                r = nl.ndarray((97, 128), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_copy(dst=r, src=acc[0:97, 0:128])
                nisa.nc_transpose(dst=tps[0:128, gg, 0:97], data=r)
            c0 = gb * GB * LM_G
            lb = nl.ndarray((128, GB, LM_G), dtype=nl.bfloat16, buffer=nl.sbuf)
            nisa.tensor_copy(dst=lb, src=tps.reshape((128, GB, LM_G, 32))[0:128, 0:GB, 0:LM_G, 0])
            nisa.tensor_tensor(dst=lg[0:128, c0:c0 + GB * LM_G], data1=lb.reshape((128, GB * LM_G)),
                               data2=bias_sb[0:128, c0:c0 + GB * LM_G], op=nl.add)
    for half in range(0 if CT_HEAD else 2):
        acc = nl.ndarray((128, HALF), dtype=nl.float32, buffer=nl.psum)
        for gg in range(NG // 2):
            grp = half * (NG // 2) + gg
            w = nl.ndarray((128, LM_F), dtype=nl.bfloat16, buffer=nl.sbuf)
            nisa.dma_copy(dst=w, src=wlm.ap(pattern=[[LM_F, 128], [1, LM_F]], offset=(pid * NG + grp) * 128 * LM_F), dge_mode=nisa.dge_mode.hwdge, engine=nisa.engine.sync)
            for j in range(LM_G):
                col_ = gg * LM_G + j
                for kt in range(HT2):
                    nisa.nc_matmul(dst=acc[0:128, col_:col_ + 1],
                                   stationary=w[0:128, (j * HT2 + kt) * 128:(j * HT2 + kt + 1) * 128],
                                   moving=xn2[0:128, kt:kt + 1])
        lb = nl.ndarray((128, HALF), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.tensor_copy(dst=lb, src=acc)
        nisa.tensor_tensor(dst=lg[0:128, half * HALF:(half + 1) * HALF], data1=lb,
                           data2=bias_sb[0:128, half * HALF:(half + 1) * HALF], op=nl.add)
    mp = nl.ndarray((128, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_reduce(dst=mp, op=nl.maximum, data=lg, axis=1)
    sc = nl.ndarray((128, NVT), dtype=nl.float32, buffer=nl.sbuf)
    nisa.scalar_tensor_tensor(dst=sc, data=lg, op0=nl.equal, operand0=mp, op1=nl.multiply, operand1=score_sb)
    sp = nl.ndarray((128, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_reduce(dst=sp, op=nl.maximum, data=sc, axis=1)
    mrow = _transpose_col(mp)
    srow = _transpose_col(sp)
    pr = nl.ndarray((1, 2), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_reduce(dst=pr[0:1, 0:1], op=nl.maximum, data=mrow, axis=1)
    s2 = nl.ndarray((1, 128), dtype=nl.float32, buffer=nl.sbuf)
    nisa.scalar_tensor_tensor(dst=s2, data=mrow, op0=nl.equal, operand0=pr[0:1, 0:1], op1=nl.multiply, operand1=srow)
    nisa.tensor_reduce(dst=pr[0:1, 1:2], op=nl.maximum, data=s2, axis=1)
    # combine the two physical cores
    po = nl.ndarray((1, 2), dtype=nl.float32, buffer=nl.sbuf)
    nisa.sendrecv(src=pr, dst=po, send_to_rank=1 - pid, recv_from_rank=1 - pid, pipe_id=0,
                  dma_engine=nisa.dma_engine.gpsimd_dma)
    mc = nl.ndarray((1, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=mc, data1=pr[0:1, 0:1], data2=po[0:1, 0:1], op=nl.maximum)
    sa = nl.ndarray((1, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.scalar_tensor_tensor(dst=sa, data=pr[0:1, 0:1], op0=nl.equal, operand0=mc, op1=nl.multiply, operand1=pr[0:1, 1:2])
    sb_ = nl.ndarray((1, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.scalar_tensor_tensor(dst=sb_, data=po[0:1, 0:1], op0=nl.equal, operand0=mc, op1=nl.multiply, operand1=po[0:1, 1:2])
    sc2 = nl.ndarray((1, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=sc2, data1=sa, data2=sb_, op=nl.maximum)
    # combine the ranks
    mg = _allreduce_max_scalar(mc, rg, "ar_m")
    sl = nl.ndarray((1, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.scalar_tensor_tensor(dst=sl, data=mc, op0=nl.equal, operand0=mg[0:1, 0:1], op1=nl.multiply, operand1=sc2)
    sg = _allreduce_max_scalar(sl, rg, "ar_s")
    tokf = nl.ndarray((1, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=tokf, data=sg[0:1, 0:1], op0=nl.multiply, operand0=-1.0, op1=nl.add, operand1=BIG)
    toki = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=toki, src=tokf)
    # write toks[pos + 1], return pos + 1
    p_sb = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.sbuf)
    nisa.dma_copy(dst=p_sb, src=pos)
    p1 = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=p1, data=p_sb, op0=nl.add, operand0=1.0)
    p1reg = nisa.register_alloc()
    nisa.register_load(p1reg, p1)
    nisa.dma_copy(dst=toks.ap(pattern=[[1, 1], [1, 1]], offset=0, scalar_offset=p1reg, indirect_dim=1), src=toki)
    pos_out = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.shared_hbm)
    nisa.dma_copy(dst=pos_out, src=p1)
    tok_out = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.shared_hbm)
    nisa.dma_copy(dst=tok_out, src=toki)
    return pos_out, tok_out



@nki.jit
def decode_step(h, pos, toks, emb, cs, rmat, ln1, wqkv, bqkv, sinks, kc, vc, kcw, vcw, wo2, bo, ln2, wr, br, wgu, bgu, wd,
                bd, wgu2, wd2, lnf, wlm, lmbias, lmscore, rg, G, B, first, last):
    """G consecutive decoder layers (layer types alternate sliding / full, starting with sliding) for one token at
    position pos[0, 0], all in one NEFF. first: the input is the embedding of toks[0, pos] (else h [1, H] bf16).
    last: the final norm, LM head and argmax follow (toks[0, pos + 1] written in place; returns pos + 1 and the
    token), else returns h_out [1, H] bf16. Layer tensors are stacked over the group (see _layer); the expert weights
    are two stacks (wgu/wd: layers [0, XS), wgu2/wd2: layers [XS, G)) so that no DMA offset reaches 2^31 elements."""
    pid = nl.program_id(0)
    p_sb, preg, sreg, lim_s, lim_f, csc = _pos_ctx(pos, cs, True, G > 1)
    rmat_sb = nl.ndarray((D, D), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.dma_copy(dst=rmat_sb, src=rmat)
    ones_pp = _ones((P, P))
    sel2 = _sel2()
    if first:
        t_sb = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.sbuf)
        nisa.dma_copy(dst=t_sb, src=toks.ap(pattern=[[1, 1], [1, 1]], offset=0, scalar_offset=preg, indirect_dim=1))
        treg = nisa.register_alloc()
        nisa.register_load(treg, t_sb)
        x_bf = nl.ndarray((P, HT), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.dma_copy(dst=x_bf, src=emb.ap(pattern=[[1, P], [P, HT]], offset=0, scalar_offset=treg, indirect_dim=0))
    else:
        x_bf = _load_col(h, nl.bfloat16)
    bufs, ebufs = _attn_bufs()
    mskT = _key_mask_T(pos, B) if B >= ATT_T_MIN else None
    aw = _attn_w(wqkv, wo2, wr, 0, pid, bufs)
    pend = None
    for li in range(G):
        x_bf, aw, pend = _layer(x_bf, li, ln1, wqkv, bqkv, sinks, kc, vc, kcw, vcw, wo2, bo, ln2, wr, br,
                      wgu if li < XS else wgu2, bgu, wd if li < XS else wd2, bd, li if li < XS else li - XS,
                      preg, sreg, lim_s, lim_f, csc, rmat_sb, ones_pp, sel2, pid, rg, li % 2 == 0, SLIDE if li % 2 == 0 else B, TAGS[li],
                      aw, li + 1 if li + 1 < G else None, bufs, ebufs, pend, None if li % 2 == 0 else mskT)
    if last:
        return _head(x_bf, pos, toks, lnf, wlm, lmbias, lmscore, rg, pid, ones_pp)
    h_out = nl.ndarray((1, H), dtype=nl.bfloat16, buffer=nl.shared_hbm)
    _store_col(h_out, x_bf)
    return h_out
