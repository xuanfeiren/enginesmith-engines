"""Prefill kernels: chunks of SC tokens at a run-time start position p0 (a multiple of SC), feature-major
activations hT [H, MAXPOS] (hidden feature on partitions in [P, HT] tiles, tokens along the free dimension).

Same weight tensors and vcore split as decode (kernels/model.py). The MoE is computed densely over all 32 experts
for every token, each expert's contribution scaled by its routing weight (exactly 0 for unselected experts).
"""
import nki
import nki.language as nl
import nki.isa as nisa
import nki.collectives as ncc

from nkilib_attn.core.attention.attention_cte import _attention_cte

from model import (_sel2, WDT, H, P, HT, NQ, D, E, TOPK, IT, GU_T, GU_F, DN_F, QKV_T, QKV_F, WO_F, EPS, SWIGLU_LIMIT, TAGS,
                   ALPHA, NEG, SLIDE, VOCAB, XS, _ones)

SC = 256          # tokens per chunk
NQT = SC // 128   # query tiles per chunk
CC_WARM = False   # a tiny all-reduce first in the dense-layer NEFF: slower (38.0 -> 38.5 ms at 256)
DMOE_LOOP = True   # dense one-chunk MoE: device loop over the used experts only (_dense_moe_loop)
DMOE_TM = False    # session 5: ... token-major (activations stationary, fp8 weights moving; _dmoe_expert_tm): slower, 30.1 -> 31.5 ms at 256
AR_SBUF_PF = False  # the dense layer's all-reduces on SBUF tensors: slower (38.0 -> 38.9 ms at 256)


def _p0(pos):
    """pos [1, 1] int32 -> (SBUF int32 tile, register, [128, 1] fp32 broadcast)."""
    p_sb = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.sbuf)
    nisa.dma_copy(dst=p_sb, src=pos)
    reg = nisa.register_alloc()
    nisa.register_load(reg, p_sb)
    pbi = nl.ndarray((128, 1), dtype=nl.int32, buffer=nl.sbuf)
    nisa.dma_copy(dst=pbi, src=pos.ap(pattern=[[0, 128], [1, 1]], offset=0))
    pb = nl.ndarray((128, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=pb, src=pbi)
    return p_sb, reg, pb


def _rmsnorm_fm(x, w_sb, ones_pp, n=SC):
    """x [P, HT, n] fp32 feature-major; w [P, HT] fp32 -> bf16 [P, HT, n]."""
    tot = nl.ndarray((P, n), dtype=nl.float32, buffer=nl.psum)
    for kt in range(HT):
        sq = nl.ndarray((P, n), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=sq, data1=x[0:P, kt, 0:n], data2=x[0:P, kt, 0:n], op=nl.multiply)
        nisa.nc_matmul(dst=tot, stationary=ones_pp, moving=sq)
    var = nl.ndarray((P, n), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=var, data=tot, op0=nl.multiply, operand0=1.0 / H, op1=nl.add, operand1=EPS)
    rstd = nl.ndarray((P, n), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=rstd, data=var, op0=nl.rsqrt, operand0=0.0, engine=nisa.engine.gpsimd)
    y = nl.ndarray((P, HT, n), dtype=nl.bfloat16, buffer=nl.sbuf)
    for kt in range(HT):
        xs = nl.ndarray((P, n), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=xs, data1=x[0:P, kt, 0:n], data2=rstd, op=nl.multiply)
        nisa.tensor_scalar(dst=y[0:P, kt, 0:n], data=xs, op0=nl.multiply, operand0=w_sb[0:P, kt:kt + 1])
    return y



def _cc_warm(rg, name):
    """A tiny all-reduce at the start of an execution: the first collective of an execution starts late and runs slow
    (profile: 2.9 MB all-reduce 300 us first vs 39 us second); issued first, that cost overlaps the attention."""
    if not CC_WARM:
        return
    t = nl.ndarray((1, 8), dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(dst=t, value=0.0)
    src = nl.ndarray((1, 8), dtype=nl.float32, buffer=nl.shared_hbm, name=name + "_wsrc")
    dst = nl.ndarray((1, 8), dtype=nl.float32, buffer=nl.shared_hbm, name=name + "_wdst")
    nisa.dma_copy(dst=src, src=t)
    nisa.core_barrier(data=src, cores=(0, 1))
    ncc.all_reduce(srcs=[src], dsts=[dst], op=nl.add, replica_group=rg)

def _allreduce_fm(part, pid, xch, slot, rg, name):
    """part [P, HT, SC] fp32 partial per vcore -> sum over the 8 vcores. xch: caller-provided HBM scratch
    [2 slots, 2 pids, P, HT * SC] fp32 for the exchange between the two physical cores."""
    F = HT * SC
    other = nl.ndarray((P, HT, SC), dtype=nl.float32, buffer=nl.sbuf)
    if XCH_SENDRECV:
        nisa.sendrecv(src=part, dst=other, send_to_rank=1 - pid, recv_from_rank=1 - pid, pipe_id=0,
                      dma_engine=nisa.dma_engine.dma)
    else:
        nisa.dma_copy(dst=xch.ap(pattern=[[F, P], [1, F]], offset=(slot * 2 + pid) * P * F), src=part)
        nisa.core_barrier(data=xch, cores=(0, 1))
        nisa.dma_copy(dst=other, src=xch.ap(pattern=[[F, P], [1, F]], offset=(slot * 2 + 1 - pid) * P * F))
    both = nl.ndarray((P, HT, SC), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=both, data1=part, data2=other, op=nl.add)
    if AR_SBUF_PF:
        res = nl.ndarray((P, HT, SC), dtype=nl.float32, buffer=nl.sbuf)
        ncc.all_reduce(srcs=[both.reshape((P, F))], dsts=[res.reshape((P, F))], op=nl.add, replica_group=rg)
        return res
    src = nl.ndarray((P, F), dtype=nl.float32, buffer=nl.shared_hbm, name=name + "_src")
    dst = nl.ndarray((P, F), dtype=nl.float32, buffer=nl.shared_hbm, name=name + "_dst")
    nisa.dma_copy(dst=src, src=both)
    nisa.core_barrier(data=src, cores=(0, 1))
    ncc.all_reduce(srcs=[src], dsts=[dst], op=nl.add, replica_group=rg)
    res = nl.ndarray((P, HT, SC), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=res, src=dst)
    return res


@nki.jit
def prefill_embed(toks, pos, emb, hT):
    """hT[:, p0:p0+SC] = emb[toks[0, p0:p0+SC]]^T (in place); returns a dummy [1, 1]."""
    MAXPOS = hT.shape[1]
    p_sb, preg, pb = _p0(pos)
    for j in range(NQT):
        idx = nl.ndarray((128, 1), dtype=nl.int32, buffer=nl.sbuf)
        nisa.dma_copy(dst=idx, src=toks.ap(pattern=[[1, 128], [1, 1]], offset=j * 128, scalar_offset=preg, indirect_dim=1))
        rows = nl.ndarray((128, H), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.dma_copy(dst=rows, src=emb.ap(pattern=[[H, 128], [1, H]], offset=0, vector_offset=idx, indirect_dim=0))
        ft = nl.ndarray((P, HT, 128), dtype=nl.bfloat16, buffer=nl.sbuf)
        for kt in range(HT):
            tp = nl.ndarray((P, 128), dtype=nl.bfloat16, buffer=nl.psum)
            nisa.nc_transpose(dst=tp, data=rows[0:128, kt * P:(kt + 1) * P])
            nisa.tensor_copy(dst=ft[0:P, kt, 0:128], src=tp)
        nisa.dma_copy(dst=hT.ap(pattern=[[MAXPOS, P], [P * MAXPOS, HT], [1, 128]], offset=j * 128,
                                scalar_offset=preg, indirect_dim=1), src=ft)
    dummy = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.shared_hbm)
    nisa.dma_copy(dst=dummy, src=p_sb)
    return dummy


@nki.jit
def prefill_gather(hT, pos):
    """h [1, H] = hT[:, pos]."""
    MAXPOS = hT.shape[1]
    p_sb = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.sbuf)
    nisa.dma_copy(dst=p_sb, src=pos)
    preg = nisa.register_alloc()
    nisa.register_load(preg, p_sb)
    t = nl.ndarray((P, HT), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.dma_copy(dst=t, src=hT.ap(pattern=[[MAXPOS, P], [P * MAXPOS, HT]], offset=0, scalar_offset=preg, indirect_dim=1))
    h_out = nl.ndarray((1, H), dtype=nl.bfloat16, buffer=nl.shared_hbm)
    nisa.dma_copy(dst=h_out.ap(pattern=[[1, P], [P, HT]], offset=0), src=t)
    return h_out



def _dmoe_expert(xn2, gu_sb, dn_sb, bgu_sb, wb, moe, first):
    """One expert of the dense one-chunk MoE (this vcore's intermediate slice, all SC tokens): moe (+)= down(inter),
    inter = SwiGLU(gate/up) * routing weight (the arithmetic of prefill_layer's static loop)."""
    inter = nl.ndarray((P, IT, SC), dtype=nl.bfloat16, buffer=nl.sbuf)
    for it in range(IT):
        gts = []
        for half in range(2):
            t = it + IT * half
            acc = nl.ndarray((P, SC), dtype=nl.float32, buffer=nl.psum)
            for kt in range(HT):
                nisa.nc_matmul(dst=acc, stationary=gu_sb[0:P, (kt * GU_T + t) * P:(kt * GU_T + t + 1) * P],
                               moving=xn2[0:P, kt, 0:SC])
            g_ = nl.ndarray((P, SC), dtype=nl.bfloat16, buffer=nl.sbuf)
            nisa.tensor_scalar(dst=g_, data=acc, op0=nl.add, operand0=bgu_sb[0:P, t:t + 1])
            gts.append(g_)
        gate = nl.ndarray((P, SC), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=gate, data=gts[0], op0=nl.minimum, operand0=SWIGLU_LIMIT)
        up = nl.ndarray((P, SC), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=up, data=gts[1], op0=nl.minimum, operand0=SWIGLU_LIMIT, op1=nl.maximum,
                           operand1=-SWIGLU_LIMIT)
        ga = nl.ndarray((P, SC), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=ga, data=gate, op0=nl.multiply, operand0=ALPHA)
        sg = nl.ndarray((P, SC), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.activation(dst=sg, op=nl.sigmoid, data=ga)
        glu = nl.ndarray((P, SC), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=glu, data1=gate, data2=sg, op=nl.multiply)
        up1 = nl.ndarray((P, SC), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=up1, data=up, op0=nl.add, operand0=1.0)
        itr = nl.ndarray((P, SC), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=itr, data1=up1, data2=glu, op=nl.multiply)
        nisa.tensor_tensor(dst=inter[0:P, it, 0:SC], data1=itr, data2=wb, op=nl.multiply)
    for nt in range(HT):
        acc = nl.ndarray((P, SC), dtype=nl.float32, buffer=nl.psum)
        for it in range(IT):
            nisa.nc_matmul(dst=acc, stationary=dn_sb[0:P, (it * HT + nt) * P:(it * HT + nt + 1) * P],
                           moving=inter[0:P, it, 0:SC])
        if first:
            nisa.tensor_copy(dst=moe[0:P, nt, 0:SC], src=acc)
        else:
            nisa.tensor_tensor(dst=moe[0:P, nt, 0:SC], data1=moe[0:P, nt, 0:SC], data2=acc, op=nl.add)


def _dmoe_expert_tm(xn2, gu_sb, dn_sb, bgt, wcb, moe_tm):
    """_dmoe_expert token-major: per 128-token tile j the activations xn2[:, kt, tile] are the stationary operand and
    the expert's fp8 weights stream as the moving operand (the moving rate of fp8 weights is higher than that of the
    bf16 activations); same per-element arithmetic and accumulation order (over kt, it, experts) as _dmoe_expert.
    bgt [128, GU_T * P] fp32: the gate/up bias row (t, n) broadcast; wcb [128, NQT] bf16: routing weight of token
    j * 128 + p; moe_tm[j] [128, H] fp32 (+)= the expert's output rows."""
    wc = nl.ndarray((128, NQT), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=wc, src=wcb)
    G4 = 4 * P
    for j in range(NQT):
        acc_a = nl.ndarray((128, G4), dtype=nl.float32, buffer=nl.psum)  # t = 0..3
        acc_b = nl.ndarray((128, GU_T * P - G4), dtype=nl.float32, buffer=nl.psum)  # t = 4, 5
        for kt in range(HT):
            nisa.nc_matmul(dst=acc_a, stationary=xn2[0:P, kt, j * 128:(j + 1) * 128],
                           moving=gu_sb[0:P, kt * GU_T * P:kt * GU_T * P + G4])
            nisa.nc_matmul(dst=acc_b, stationary=xn2[0:P, kt, j * 128:(j + 1) * 128],
                           moving=gu_sb[0:P, kt * GU_T * P + G4:(kt + 1) * GU_T * P])
        g_ = nl.ndarray((128, GU_T * P), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=g_[0:128, 0:G4], data1=acc_a, data2=bgt[0:128, 0:G4], op=nl.add)
        nisa.tensor_tensor(dst=g_[0:128, G4:GU_T * P], data1=acc_b, data2=bgt[0:128, G4:GU_T * P], op=nl.add)
        NI = IT * P
        gate = nl.ndarray((128, NI), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=gate, data=g_[0:128, 0:NI], op0=nl.minimum, operand0=SWIGLU_LIMIT)
        up = nl.ndarray((128, NI), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=up, data=g_[0:128, NI:2 * NI], op0=nl.minimum, operand0=SWIGLU_LIMIT,
                           op1=nl.maximum, operand1=-SWIGLU_LIMIT)
        ga = nl.ndarray((128, NI), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=ga, data=gate, op0=nl.multiply, operand0=ALPHA)
        sg = nl.ndarray((128, NI), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.activation(dst=sg, op=nl.sigmoid, data=ga)
        glu = nl.ndarray((128, NI), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=glu, data1=gate, data2=sg, op=nl.multiply)
        up1 = nl.ndarray((128, NI), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=up1, data=up, op0=nl.add, operand0=1.0)
        itr = nl.ndarray((128, NI), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=itr, data1=up1, data2=glu, op=nl.multiply)
        itw = nl.ndarray((128, NI), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=itw, data=itr, op0=nl.multiply, operand0=wc[0:128, j:j + 1])
        interT = nl.ndarray((P, IT, 128), dtype=nl.bfloat16, buffer=nl.sbuf)
        for it in range(IT):
            ip = nl.ndarray((P, 128), dtype=nl.bfloat16, buffer=nl.psum)
            nisa.nc_transpose(dst=ip, data=itw[0:128, it * P:(it + 1) * P])
            nisa.tensor_copy(dst=interT[0:P, it, 0:128], src=ip)
        accs = []
        for c in range(len(DCH)):
            accs.append(nl.ndarray((128, 512), dtype=nl.float32, buffer=nl.psum))
        for it in range(IT):
            for c in range(len(DCH)):
                c0 = DCH[c][0]
                cw = DCH[c][1]
                nisa.nc_matmul(dst=accs[c][0:128, 0:cw], stationary=interT[0:P, it, 0:128],
                               moving=dn_sb[0:P, it * H + c0:it * H + c0 + cw])
        for c in range(len(DCH)):
            c0 = DCH[c][0]
            cw = DCH[c][1]
            nisa.tensor_tensor(dst=moe_tm[j][0:128, c0:c0 + cw], data1=moe_tm[j][0:128, c0:c0 + cw],
                               data2=accs[c][0:128, 0:cw], op=nl.add)


def _dmoe_load(wgu, wd, bgu, rtw, gu_sb, dn_sb, bgu_sb, wb, ereg, rreg, pid, lx, li, bgut=None):
    """Weights, gate/up bias and routing-weight row (broadcast over P) of expert ereg (register; E = the dummy
    expert: its weight DMAs are out of range and skipped, its routing-weight row rtw[E] is zero). (The 128-partition
    copies wgu8 / wd8 instead: no change, 33.3 ms at 256: the loop is not DMA-bound.)"""
    F = HT * P
    for c in range(GU_T):
        nisa.dma_copy(dst=gu_sb[0:P, c * F:(c + 1) * F],
                      src=wgu.ap(pattern=[[GU_F, P], [1, F]], offset=lx * E * 2 * P * GU_F + pid * P * GU_F + c * F, scalar_offset=ereg,
                                 indirect_dim=0),
                      dge_mode=nisa.dge_mode.hwdge, engine=nisa.engine.sync, oob_mode=nisa.oob_mode.skip)
    nisa.dma_copy(dst=dn_sb, src=wd.ap(pattern=[[DN_F, P], [1, DN_F]], offset=lx * E * 2 * P * DN_F + pid * P * DN_F, scalar_offset=ereg,
                                       indirect_dim=0),
                  dge_mode=nisa.dge_mode.hwdge, engine=nisa.engine.sync, oob_mode=nisa.oob_mode.skip)
    if bgut is not None:  # token-major: bias row broadcast over 128 partitions, routing weight per token
        nisa.dma_copy(dst=bgu_sb, src=bgut.ap(pattern=[[0, 128], [1, GU_T * P]], offset=li * E * 2 * P * GU_T + pid * P * GU_T,
                                              scalar_offset=ereg, indirect_dim=0), oob_mode=nisa.oob_mode.skip)
        nisa.dma_copy(dst=wb, src=rtw.ap(pattern=[[1, 128], [128, NQT]], offset=0, scalar_offset=rreg, indirect_dim=0))
        return
    nisa.dma_copy(dst=bgu_sb, src=bgu.ap(pattern=[[GU_T, P], [1, GU_T]], offset=li * E * 2 * P * GU_T + pid * P * GU_T, scalar_offset=ereg,
                                         indirect_dim=0), oob_mode=nisa.oob_mode.skip)
    nisa.dma_copy(dst=wb, src=rtw.ap(pattern=[[0, P], [1, SC]], offset=0, scalar_offset=rreg, indirect_dim=0))


def _dense_moe_loop(xn2, rtT, wgu, bgu, wd, pid, li=0, lx=0, tag="", bgut=None):
    """The dense one-chunk MoE over the USED experts only (some token of the chunk selected them; the others
    contribute exact zeros): their ids in ascending order -> elist (pairs, padded with the dummy expert E), one device
    loop over the pairs; expert 2i+1's weights load while 2i computes and 2i+2's while 2i+1 computes (two buffers).
    Same arithmetic and expert order as the static loop (bitwise identical). Layer li of stacked tensors (lx: the
    layer within the expert-weight stack); the dummy's weight row is just past the end of the stack (its DMAs are
    skipped), its routing-weight row is rtw[E] (zero): elist rows are (weight row, routing row, 0, 0)."""
    DUMW = wgu.shape[0] - lx * E  # dummy weight row (relative to layer lx's first expert): one past the stack
    NPR = E // 2 + 1  # pair rows (one spare)
    # routing-weight rows in HBM (row E: zero, the dummy)
    rtw = nl.ndarray((E + 1, SC), dtype=nl.bfloat16, buffer=nl.shared_hbm, name="dm_rtw" + tag)
    nisa.dma_copy(dst=rtw.ap(pattern=[[SC, E], [1, SC]], offset=0), src=rtT)
    zr = nl.ndarray((1, SC), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.memset(dst=zr, value=0.0)
    nisa.dma_copy(dst=rtw.ap(pattern=[[SC, 1], [1, SC]], offset=E * SC), src=zr)
    # used experts: u[e] = max_t rtT[e, t] > 0; slot pos[e] = sum_{e' < e} u[e'] (exact integers in fp32)
    rmx = nl.ndarray((E, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_reduce(dst=rmx, op=nl.maximum, data=rtT, axis=1)
    u = nl.ndarray((E, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=u, data=rmx, op0=nl.greater, operand0=0.0)
    u_b = nl.ndarray((E, 1), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.tensor_copy(dst=u_b, src=u)
    ltd = nl.ndarray((E, E), dtype=nl.float32, buffer=nl.sbuf)  # column - row
    nisa.iota(dst=ltd, pattern=[[1, E]], offset=0, channel_multiplier=-1)
    lst = nl.ndarray((E, E), dtype=nl.bfloat16, buffer=nl.sbuf)  # [e', e] = 1 if e' < e
    nisa.tensor_scalar(dst=lst, data=ltd, op0=nl.greater, operand0=0.0)
    pos_p = nl.ndarray((E, 1), dtype=nl.float32, buffer=nl.psum)
    nisa.nc_matmul(dst=pos_p, stationary=lst, moving=u_b)
    onesE = nl.ndarray((E, 1), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.memset(dst=onesE, value=1.0)
    nu_p = nl.ndarray((1, 1), dtype=nl.float32, buffer=nl.psum)
    nisa.nc_matmul(dst=nu_p, stationary=onesE, moving=u_b)
    nu = nl.ndarray((1, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=nu, src=nu_p)
    # pairs = ceil(nu / 2) = #{k : 2 k < nu}
    k2 = nl.ndarray((1, NPR), dtype=nl.float32, buffer=nl.sbuf)
    nisa.iota(dst=k2, pattern=[[2, NPR]], offset=0, channel_multiplier=0)
    kl = nl.ndarray((1, NPR), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=kl, data=k2, op0=nl.less, operand0=nu)
    npf = nl.ndarray((1, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_reduce(dst=npf, op=nl.add, data=kl, axis=1)
    npi = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=npi, src=npf)
    npreg = nisa.register_alloc()
    nisa.register_load(npreg, npi)
    # elist: rows of 4 int32 (expert id, 0, 0, 0), all E first; used expert e -> row pos[e]
    elist = nl.ndarray((2 * NPR, 4), dtype=nl.int32, buffer=nl.shared_hbm, name="dm_elist" + tag)
    einit = nl.ndarray((1, 2 * NPR, 4), dtype=nl.int32, buffer=nl.sbuf)
    nisa.memset(dst=einit, value=0)
    nisa.memset(dst=einit[0:1, 0:2 * NPR, 0:1], value=DUMW)
    nisa.memset(dst=einit[0:1, 0:2 * NPR, 1:2], value=E)
    nisa.dma_copy(dst=elist.ap(pattern=[[2 * NPR * 4, 1], [1, 2 * NPR * 4]], offset=0), src=einit)
    eio = nl.ndarray((E, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.iota(dst=eio, pattern=[[0, 1]], offset=0, channel_multiplier=1)
    erow = nl.ndarray((E, 4), dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(dst=erow, value=0.0)
    nisa.tensor_copy(dst=erow[0:E, 0:1], src=eio)
    nisa.tensor_copy(dst=erow[0:E, 1:2], src=eio)
    erow_i = nl.ndarray((E, 4), dtype=nl.int32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=erow_i, src=erow)
    pd = nl.ndarray((E, 1), dtype=nl.float32, buffer=nl.sbuf)  # used ? pos : out of range
    nisa.tensor_scalar(dst=pd, data=pos_p, op0=nl.add, operand0=-OOBROW, op1=nl.multiply, operand1=u)
    nisa.tensor_scalar(dst=pd, data=pd, op0=nl.add, operand0=OOBROW)
    pd_i = nl.ndarray((E, 1), dtype=nl.int32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=pd_i, src=pd)
    nisa.dma_copy(dst=elist.ap(pattern=[[4, E], [1, 4]], offset=0, vector_offset=pd_i, indirect_dim=0), src=erow_i,
                  oob_mode=nisa.oob_mode.skip)
    elp = elist.reshape((NPR, 8))  # pair k = rows 2 k, 2 k + 1
    # two weight buffers; the first listed expert's weights before the loop
    TM = bgut is not None  # token-major experts (DMOE_TM)
    gus, dns, bgs, wbs = [], [], [], []
    for k in range(2):
        gus.append(nl.ndarray((P, GU_F), dtype=WDT, buffer=nl.sbuf))
        dns.append(nl.ndarray((P, DN_F), dtype=WDT, buffer=nl.sbuf))
        if TM:
            bgs.append(nl.ndarray((128, GU_T * P), dtype=nl.float32, buffer=nl.sbuf))
            wbs.append(nl.ndarray((128, NQT), dtype=nl.bfloat16, buffer=nl.sbuf))
        else:
            bgs.append(nl.ndarray((P, GU_T), dtype=nl.float32, buffer=nl.sbuf))
            wbs.append(nl.ndarray((P, SC), dtype=nl.bfloat16, buffer=nl.sbuf))
    h0 = nl.ndarray((1, 4), dtype=nl.int32, buffer=nl.sbuf)
    nisa.dma_copy(dst=h0, src=elist.ap(pattern=[[4, 1], [1, 4]], offset=0))
    r0 = nisa.register_alloc()
    nisa.register_load(r0, h0[0:1, 0:1])
    q0 = nisa.register_alloc()
    nisa.register_load(q0, h0[0:1, 1:2])
    _dmoe_load(wgu, wd, bgu, rtw, gus[0], dns[0], bgs[0], wbs[0], r0, q0, pid, lx, li, bgut)
    moe = nl.ndarray((P, HT, SC), dtype=nl.float32, buffer=nl.sbuf)
    moe_tm = []
    if TM:
        for j in range(NQT):
            mt = nl.ndarray((128, H), dtype=nl.float32, buffer=nl.sbuf)
            nisa.memset(dst=mt, value=0.0)
            moe_tm.append(mt)
    else:
        nisa.memset(dst=moe, value=0.0)
    for j in nl.dynamic_range(0, npreg, 1):
        hr = nl.ndarray((1, 12), dtype=nl.int32, buffer=nl.sbuf)  # pair j (2 rows) + the next pair's first row
        nisa.dma_copy(dst=hr, src=elp.ap(pattern=[[12, 1], [1, 12]], offset=0, scalar_offset=j, indirect_dim=0))
        r1 = nisa.register_alloc()
        nisa.register_load(r1, hr[0:1, 4:5])
        q1 = nisa.register_alloc()
        nisa.register_load(q1, hr[0:1, 5:6])
        r2 = nisa.register_alloc()
        nisa.register_load(r2, hr[0:1, 8:9])
        q2 = nisa.register_alloc()
        nisa.register_load(q2, hr[0:1, 9:10])
        _dmoe_load(wgu, wd, bgu, rtw, gus[1], dns[1], bgs[1], wbs[1], r1, q1, pid, lx, li, bgut)
        if TM:
            _dmoe_expert_tm(xn2, gus[0], dns[0], bgs[0], wbs[0], moe_tm)
        else:
            _dmoe_expert(xn2, gus[0], dns[0], bgs[0], wbs[0], moe, False)
        _dmoe_load(wgu, wd, bgu, rtw, gus[0], dns[0], bgs[0], wbs[0], r2, q2, pid, lx, li, bgut)
        if TM:
            _dmoe_expert_tm(xn2, gus[1], dns[1], bgs[1], wbs[1], moe_tm)
        else:
            _dmoe_expert(xn2, gus[1], dns[1], bgs[1], wbs[1], moe, False)
    if TM:  # back to feature-major (exact fp32 transposes)
        for j in range(NQT):
            for nt in range(HT):
                tp = nl.ndarray((P, 128), dtype=nl.float32, buffer=nl.psum)
                nisa.nc_transpose(dst=tp, data=moe_tm[j][0:128, nt * P:(nt + 1) * P])
                nisa.tensor_copy(dst=moe[0:P, nt, j * 128:(j + 1) * 128], src=tp)
    return moe

@nki.jit
def prefill_layer(hT, pos, ln1, wqkv, bqkv, csT, rmat, sinks, kc, vc, wo2, bo, ln2, wr, br,
                  wgu, bgu, wd, bd, bgut, xch, rg, sliding, KB, dbg=False):
    """One decoder layer for the SC tokens at positions [p0, p0 + SC) (per-layer tensors); see _dense_layer."""
    pid = nl.program_id(0)
    _cc_warm(rg, "pl")
    p_sb, preg, pb = _p0(pos)
    ones_pp = _ones((P, P))
    rmat_sb = nl.ndarray((D, D), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.dma_copy(dst=rmat_sb, src=rmat)
    r = _dense_layer(None, 0, 0, hT, p_sb, preg, pb, ones_pp, rmat_sb, ln1, wqkv, bqkv, csT, sinks, kc, vc, wo2, bo,
                     ln2, wr, br, wgu, bgu, wd, bd, bgut, xch, rg, pid, sliding, KB, "", True, dbg)
    dummy = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.shared_hbm)
    nisa.dma_copy(dst=dummy, src=p_sb)
    if dbg:
        return dummy, r[1], r[2]
    return dummy


@nki.jit
def prefill_dense(hT, pos, ln1, wqkv, bqkv, csT, rmat, sinks, kc, vc, wo2, bo, ln2, wr, br,
                  wgu, bgu, wd, bd, wgu2, wd2, bgut, xch, rg, G, KB):
    """G consecutive dense one-chunk layers (sliding / full alternating, starting with sliding) in ONE NEFF
    (session 5: one execution per layer cost a per-execution start / rank-skew delay at its first collective).
    Layer tensors stacked over the layers (expert weights: wgu / wd layers [0, XS), wgu2 / wd2 the rest); the
    hidden state stays in SBUF between layers, hT gets the last layer's output."""
    pid = nl.program_id(0)
    p_sb, preg, pb = _p0(pos)
    ones_pp = _ones((P, P))
    rmat_sb = nl.ndarray((D, D), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.dma_copy(dst=rmat_sb, src=rmat)
    x = None
    for li in range(G):
        x = _dense_layer(x, li, li if li < XS else li - XS, hT, p_sb, preg, pb, ones_pp, rmat_sb, ln1, wqkv, bqkv,
                         csT, sinks, kc, vc, wo2, bo, ln2, wr, br, wgu if li < XS else wgu2, bgu,
                         wd if li < XS else wd2, bd, bgut, xch, rg, pid, li % 2 == 0, KB, TAGS[li], li == G - 1, False)
    dummy = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.shared_hbm)
    nisa.dma_copy(dst=dummy, src=p_sb)
    return dummy


def _dense_layer(x_in, li, lx, hT, p_sb, preg, pb, ones_pp, rmat_sb, ln1, wqkv, bqkv, csT, sinks, kc, vc, wo2, bo,
                 ln2, wr, br, wgu, bgu, wd, bd, bgut, xch, rg, pid, sliding, KB, tag, store, dbg):
    """One decoder layer (layer li of stacked tensors, or a per-layer slice with li = 0) for the SC tokens at
    positions [p0, p0 + SC), p0 = pos[0, 0]. x_in: the layer input [P, HT, SC] bf16 SBUF, or None (read hT);
    store: write the output to hT. K/V of the chunk written to the caches. csT [128, MAXPOS] fp32 (cos x2 rows | sin
    x2 rows). Full layers attend to cache positions [0, KB) (KB >= p0 + SC); sliding layers to [p0 - 128, p0 + SC).
    Returns the output [P, HT, SC] bf16 SBUF."""
    MAXPOS = hT.shape[1]
    T = kc.shape[-1]
    KD = len(kc.shape) - 1  # position dimension of K^T [(G,) 2, D, T] and V [(G,) 2, T, D]
    VD = len(vc.shape) - 2
    ln1_sb = nl.ndarray((P, HT), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=ln1_sb, src=ln1.ap(pattern=[[HT, P], [1, HT]], offset=li * P * HT))
    ln2_sb = nl.ndarray((P, HT), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=ln2_sb, src=ln2.ap(pattern=[[HT, P], [1, HT]], offset=li * P * HT))
    bo_sb = nl.ndarray((P, HT), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=bo_sb, src=bo.ap(pattern=[[HT, P], [1, HT]], offset=li * P * HT))
    bqkv_sb = nl.ndarray((D, QKV_T), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=bqkv_sb, src=bqkv.ap(pattern=[[QKV_T, D], [1, QKV_T]], offset=(li * 2 + pid) * D * QKV_T))
    sinkb = nl.ndarray((128, NQ), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=sinkb, src=sinks.ap(pattern=[[0, 128], [1, NQ]], offset=(li * 2 + pid) * NQ))
    wqkv_sb = nl.ndarray((P, QKV_F), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.dma_copy(dst=wqkv_sb, src=wqkv.ap(pattern=[[QKV_F, P], [1, QKV_F]], offset=(li * 2 + pid) * P * QKV_F))
    WO2 = WO_F // 2  # wo2: head pairs on 128 partitions (row (hh % 2) * 64 + d, column (hh // 2) * H + h)
    wo_sb = nl.ndarray((128, WO2), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.dma_copy(dst=wo_sb, src=wo2.ap(pattern=[[WO2, 128], [1, WO2]], offset=(li * 2 + pid) * 128 * WO2))
    wr_sb = nl.ndarray((P, HT * E), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.dma_copy(dst=wr_sb, src=wr.ap(pattern=[[HT * E, P], [1, HT * E]], offset=li * P * HT * E))
    cs_sb = nl.ndarray((128, SC), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=cs_sb, src=csT.ap(pattern=[[MAXPOS, 128], [1, SC]], offset=0, scalar_offset=preg, indirect_dim=1))

    # ---- load chunk, norm
    if x_in is None:
        x_bf = nl.ndarray((P, HT, SC), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.dma_copy(dst=x_bf, src=hT.ap(pattern=[[MAXPOS, P], [P * MAXPOS, HT], [1, SC]], offset=0,
                                          scalar_offset=preg, indirect_dim=1))
    else:
        x_bf = x_in
    x = nl.ndarray((P, HT, SC), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=x, src=x_bf)
    xn = _rmsnorm_fm(x, ln1_sb, ones_pp)

    # ---- QKV + RoPE -> qk [D, QKV_T, SC] bf16 (q heads 0..7, k, v)
    qk = nl.ndarray((D, QKV_T, SC), dtype=nl.bfloat16, buffer=nl.sbuf)
    for t in range(QKV_T):
        acc = nl.ndarray((D, SC), dtype=nl.float32, buffer=nl.psum)
        for kt in range(HT):
            nisa.nc_matmul(dst=acc, stationary=wqkv_sb[0:P, (kt * QKV_T + t) * D:(kt * QKV_T + t + 1) * D],
                           moving=xn[0:P, kt, 0:SC])
        raw = nl.ndarray((D, SC), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=raw, data=acc, op0=nl.add, operand0=bqkv_sb[0:D, t:t + 1])
        if t == QKV_T - 1:
            nisa.tensor_copy(dst=qk[0:D, t, 0:SC], src=raw)
        else:
            rot = nl.ndarray((D, SC), dtype=nl.float32, buffer=nl.psum)
            nisa.nc_matmul(dst=rot, stationary=rmat_sb, moving=raw)
            t1 = nl.ndarray((D, SC), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_tensor(dst=t1, data1=raw, data2=cs_sb[0:D, 0:SC], op=nl.multiply)
            t2 = nl.ndarray((D, SC), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_tensor(dst=t2, data1=rot, data2=cs_sb[D:2 * D, 0:SC], op=nl.multiply)
            nisa.tensor_tensor(dst=qk[0:D, t, 0:SC], data1=t1, data2=t2, op=nl.add)

    # ---- write K^T and V of the chunk at [p0, p0 + SC)
    nisa.dma_copy(dst=kc.ap(pattern=[[T, D], [1, SC]], offset=(li * 2 + pid) * D * T, scalar_offset=preg, indirect_dim=KD),
                  src=qk[0:D, NQ, 0:SC])
    for j in range(NQT):
        vt = nl.ndarray((128, D), dtype=nl.bfloat16, buffer=nl.psum)
        nisa.nc_transpose(dst=vt, data=qk[0:D, NQ + 1, j * 128:(j + 1) * 128])
        vts = nl.ndarray((128, D), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.tensor_copy(dst=vts, src=vt)
        nisa.dma_copy(dst=vc.ap(pattern=[[D, 128], [1, D]], offset=(li * 2 + pid) * T * D + j * 128 * D, scalar_offset=preg,
                                indirect_dim=VD), src=vts)

    # ---- attention (reads the caches after the writes above)
    if sliding:
        NK = SLIDE + SC
        ks_sb = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.sbuf)  # key window start max(p0 - 128, 0)
        nisa.tensor_scalar(dst=ks_sb, data=p_sb, op0=nl.add, operand0=-float(SLIDE), op1=nl.maximum, operand1=0.0)
        kreg = nisa.register_alloc()
        nisa.register_load(kreg, ks_sb)
        ksb_i = nl.ndarray((128, 1), dtype=nl.int32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=ksb_i, data=pb, op0=nl.add, operand0=-float(SLIDE), op1=nl.maximum, operand1=0.0)
        kstart = nl.ndarray((128, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=kstart, src=ksb_i)
    else:
        NK = KB
    NKB = NK // 128
    k_sb = nl.ndarray((D, NK), dtype=nl.bfloat16, buffer=nl.sbuf)
    v_sb = nl.ndarray((128, NKB, D), dtype=nl.bfloat16, buffer=nl.sbuf)
    if sliding:
        nisa.dma_copy(dst=k_sb, src=kc.ap(pattern=[[T, D], [1, NK]], offset=(li * 2 + pid) * D * T, scalar_offset=kreg, indirect_dim=KD))
        nisa.dma_copy(dst=v_sb, src=vc.ap(pattern=[[D, 128], [128 * D, NKB], [1, D]], offset=(li * 2 + pid) * T * D,
                                          scalar_offset=kreg, indirect_dim=VD))
    else:
        nisa.dma_copy(dst=k_sb, src=kc.ap(pattern=[[T, D], [1, NK]], offset=(li * 2 + pid) * D * T))
        nisa.dma_copy(dst=v_sb, src=vc.ap(pattern=[[D, 128], [128 * D, NKB], [1, D]], offset=(li * 2 + pid) * T * D))

    ao = nl.ndarray((D, NQ, SC), dtype=nl.bfloat16, buffer=nl.sbuf)
    CH = 512 if NK % 512 == 0 else 128
    for qt in range(NQT):
        # mask[i, j]: query position qp = p0 + qt*128 + i, key position kp = kstart + j
        dif = nl.ndarray((128, NK), dtype=nl.float32, buffer=nl.sbuf)  # kp - qp
        nisa.iota(dst=dif, pattern=[[1, NK]], offset=-qt * 128, channel_multiplier=-1)
        off = nl.ndarray((128, 1), dtype=nl.float32, buffer=nl.sbuf)  # kstart - p0
        if sliding:
            nisa.tensor_tensor(dst=off, data1=kstart, data2=pb, op=nl.subtract)
        else:
            nisa.tensor_scalar(dst=off, data=pb, op0=nl.multiply, operand0=-1.0)
        d2 = nl.ndarray((128, NK), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=d2, data=dif, op0=nl.add, operand0=off)
        msk = nl.ndarray((128, NK), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=msk, data=d2, op0=nl.greater, operand0=0.0, op1=nl.multiply, operand1=NEG)
        if sliding:
            m2 = nl.ndarray((128, NK), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_scalar(dst=m2, data=d2, op0=nl.less_equal, operand0=-float(SLIDE), op1=nl.multiply, operand1=NEG)
            m3 = nl.ndarray((128, NK), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_tensor(dst=m3, data1=msk, data2=m2, op=nl.add)
            msk = m3
        for hh in range(NQ):
            s_bf = nl.ndarray((128, NK), dtype=nl.bfloat16, buffer=nl.sbuf)
            for c in range(NK // CH):
                sp = nl.ndarray((128, CH), dtype=nl.float32, buffer=nl.psum)
                nisa.nc_matmul(dst=sp, stationary=qk[0:D, hh, qt * 128:(qt + 1) * 128], moving=k_sb[0:D, c * CH:(c + 1) * CH])
                nisa.tensor_copy(dst=s_bf[0:128, c * CH:(c + 1) * CH], src=sp)
            s = nl.ndarray((128, NK), dtype=nl.float32, buffer=nl.sbuf)
            nisa.scalar_tensor_tensor(dst=s, data=s_bf, op0=nl.multiply, operand0=0.125, op1=nl.add, operand1=msk)
            m1 = nl.ndarray((128, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_reduce(dst=m1, op=nl.maximum, data=s, axis=1)
            m = nl.ndarray((128, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_tensor(dst=m, data1=m1, data2=sinkb[0:128, hh:hh + 1], op=nl.maximum)
            negm = nl.ndarray((128, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_scalar(dst=negm, data=m, op0=nl.multiply, operand0=-1.0)
            pexp = nl.ndarray((128, NK), dtype=nl.float32, buffer=nl.sbuf)
            ssum = nl.ndarray((128, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.activation_reduce(dst=pexp, op=nl.exp, data=s, reduce_op=nl.add, reduce_res=ssum, bias=negm, scale=1.0)
            es = nl.ndarray((128, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.activation(dst=es, op=nl.exp, data=sinkb[0:128, hh:hh + 1], bias=negm, scale=1.0)
            tot = nl.ndarray((128, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_tensor(dst=tot, data1=ssum, data2=es, op=nl.add)
            rt = nl.ndarray((128, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.reciprocal(dst=rt, data=tot)
            prob = nl.ndarray((128, NK), dtype=nl.bfloat16, buffer=nl.sbuf)
            nisa.tensor_scalar(dst=prob, data=pexp, op0=nl.multiply, operand0=rt)
            acc_o = nl.ndarray((D, 128), dtype=nl.float32, buffer=nl.psum)
            for kb in range(NKB):
                ptp = nl.ndarray((128, 128), dtype=nl.bfloat16, buffer=nl.psum)
                nisa.nc_transpose(dst=ptp, data=prob[0:128, kb * 128:(kb + 1) * 128])
                pts = nl.ndarray((128, 128), dtype=nl.bfloat16, buffer=nl.sbuf)
                nisa.tensor_copy(dst=pts, src=ptp)
                nisa.nc_matmul(dst=acc_o, stationary=v_sb[0:128, kb, 0:D], moving=pts)
            nisa.tensor_copy(dst=ao[0:D, hh, qt * 128:(qt + 1) * 128], src=acc_o)

    # ---- o_proj + all-reduce + bias + residual
    # head pairs stacked on 128 partitions (exact 0/1 selection matmuls), then o_proj with a 128-row contraction
    sel2 = _sel2()
    ao2 = nl.ndarray((128, NQ // 2, SC), dtype=nl.bfloat16, buffer=nl.sbuf)
    for hp in range(NQ // 2):
        ap_ = nl.ndarray((128, SC), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_matmul(dst=ap_, stationary=sel2[0:D, 0, 0:128], moving=ao[0:D, 2 * hp, 0:SC])
        nisa.nc_matmul(dst=ap_, stationary=sel2[0:D, 1, 0:128], moving=ao[0:D, 2 * hp + 1, 0:SC])
        nisa.tensor_copy(dst=ao2[0:128, hp, 0:SC], src=ap_)
    part = nl.ndarray((P, HT, SC), dtype=nl.float32, buffer=nl.sbuf)
    for nt in range(HT):
        acc = nl.ndarray((P, SC), dtype=nl.float32, buffer=nl.psum)
        for hp in range(NQ // 2):
            nisa.nc_matmul(dst=acc, stationary=wo_sb[0:128, hp * H + nt * P:hp * H + (nt + 1) * P], moving=ao2[0:128, hp, 0:SC])
        nisa.tensor_copy(dst=part[0:P, nt, 0:SC], src=acc)
    red = _allreduce_fm(part, pid, xch, 0, rg, "pf_attn" + tag)
    h1 = nl.ndarray((P, HT, SC), dtype=nl.float32, buffer=nl.sbuf)
    h1_bf = nl.ndarray((P, HT, SC), dtype=nl.bfloat16, buffer=nl.sbuf)
    for kt in range(HT):
        at = nl.ndarray((P, SC), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=at, data=red[0:P, kt, 0:SC], op0=nl.add, operand0=bo_sb[0:P, kt:kt + 1])
        nisa.tensor_tensor(dst=h1_bf[0:P, kt, 0:SC], data1=x_bf[0:P, kt, 0:SC], data2=at, op=nl.add)
    nisa.tensor_copy(dst=h1, src=h1_bf)

    # ---- norm + router + top-4 per token
    xn2 = _rmsnorm_fm(h1, ln2_sb, ones_pp)
    lgp = nl.ndarray((E, SC), dtype=nl.float32, buffer=nl.psum)
    for kt in range(HT):
        nisa.nc_matmul(dst=lgp, stationary=wr_sb[0:P, kt * E:(kt + 1) * E], moving=xn2[0:P, kt, 0:SC])
    brc = nl.ndarray((E, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=brc, src=br.ap(pattern=[[1, E], [1, 1]], offset=li * E))
    lgb = nl.ndarray((E, SC), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=lgb, data=lgp, op0=nl.add, operand0=brc)
    lgf = nl.ndarray((E, SC), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=lgf, src=lgb)
    io = nl.ndarray((128, E), dtype=nl.float32, buffer=nl.sbuf)
    nisa.iota(dst=io, pattern=[[1, E]], offset=0, channel_multiplier=0)
    rev = nl.ndarray((128, E), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=rev, data=io, op0=nl.multiply, operand0=-1.0, op1=nl.add, operand1=64.0)
    rtT = nl.ndarray((E, SC), dtype=nl.bfloat16, buffer=nl.sbuf)  # routing weights^T (0 for unselected)
    for j in range(NQT):
        tp = nl.ndarray((128, E), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_transpose(dst=tp, data=lgf[0:E, j * 128:(j + 1) * 128])
        cur = nl.ndarray((128, E), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=cur, src=tp)
        vals = nl.ndarray((128, TOPK), dtype=nl.float32, buffer=nl.sbuf)
        hits = []
        for i in range(TOPK):
            mx = nl.ndarray((128, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_reduce(dst=mx, op=nl.maximum, data=cur, axis=1)
            nisa.tensor_copy(dst=vals[0:128, i:i + 1], src=mx)
            sc = nl.ndarray((128, E), dtype=nl.float32, buffer=nl.sbuf)
            nisa.scalar_tensor_tensor(dst=sc, data=cur, op0=nl.equal, operand0=mx, op1=nl.multiply, operand1=rev)
            sm = nl.ndarray((128, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_reduce(dst=sm, op=nl.maximum, data=sc, axis=1)
            hit = nl.ndarray((128, E), dtype=nl.float32, buffer=nl.sbuf)  # one-hot of the selected expert
            nisa.tensor_scalar(dst=hit, data=rev, op0=nl.equal, operand0=sm)
            hits.append(hit)
            nxt = nl.ndarray((128, E), dtype=nl.float32, buffer=nl.sbuf)
            nisa.scalar_tensor_tensor(dst=nxt, data=hit, op0=nl.multiply, operand0=NEG, op1=nl.add, operand1=cur)
            cur = nxt
        negv = nl.ndarray((128, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=negv, data=vals[0:128, 0:1], op0=nl.multiply, operand0=-1.0)
        ev = nl.ndarray((128, TOPK), dtype=nl.float32, buffer=nl.sbuf)
        esum = nl.ndarray((128, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.activation_reduce(dst=ev, op=nl.exp, data=vals, reduce_op=nl.add, reduce_res=esum, bias=negv, scale=1.0)
        rs = nl.ndarray((128, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.reciprocal(dst=rs, data=esum)
        wbf = nl.ndarray((128, TOPK), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=wbf, data=ev, op0=nl.multiply, operand0=rs)
        wf = nl.ndarray((128, TOPK), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=wf, src=wbf)
        dense = nl.ndarray((128, E), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=dense, data=hits[0], op0=nl.multiply, operand0=wf[0:128, 0:1])
        for i in range(1, TOPK):
            d2_ = nl.ndarray((128, E), dtype=nl.float32, buffer=nl.sbuf)
            nisa.scalar_tensor_tensor(dst=d2_, data=hits[i], op0=nl.multiply, operand0=wf[0:128, i:i + 1], op1=nl.add,
                                      operand1=dense)
            dense = d2_
        tb = nl.ndarray((E, 128), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_transpose(dst=tb, data=dense)
        nisa.tensor_copy(dst=rtT[0:E, j * 128:(j + 1) * 128], src=tb)

    if DMOE_LOOP:
        moe = _dense_moe_loop(xn2, rtT, wgu, bgu, wd, pid, li, lx, tag, bgut if DMOE_TM else None)
    else:
        # ---- dense MoE over all experts (this vcore's intermediate slice), routing weight applied to the
        # intermediate activation; the bias term sum_e w_e * b_e is added after the all-reduce
        eio = nl.ndarray((E, P), dtype=nl.float32, buffer=nl.sbuf)
        nisa.iota(dst=eio, pattern=[[0, P]], offset=0, channel_multiplier=1)  # partition id
        moe = nl.ndarray((P, HT, SC), dtype=nl.float32, buffer=nl.sbuf)
        for e in range(E):
            gu_sb = nl.ndarray((P, GU_F), dtype=WDT, buffer=nl.sbuf)
            F = HT * P
            for c in range(GU_T):  # [p][kt][t][n], 6 DMAs of 4 kt
                nisa.dma_copy(dst=gu_sb[0:P, c * F:(c + 1) * F],
                              src=wgu.ap(pattern=[[GU_F, P], [1, F]], offset=(e * 2 + pid) * P * GU_F + c * F),
                              dge_mode=nisa.dge_mode.hwdge, engine=nisa.engine.sync)
            dn_sb = nl.ndarray((P, DN_F), dtype=WDT, buffer=nl.sbuf)
            nisa.dma_copy(dst=dn_sb, src=wd.ap(pattern=[[DN_F, P], [1, DN_F]], offset=(e * 2 + pid) * P * DN_F),
                          dge_mode=nisa.dge_mode.hwdge, engine=nisa.engine.sync)
            bgu_sb = nl.ndarray((P, GU_T), dtype=nl.float32, buffer=nl.sbuf)
            nisa.dma_copy(dst=bgu_sb, src=bgu.ap(pattern=[[GU_T, P], [1, GU_T]], offset=(e * 2 + pid) * P * GU_T))
            # routing weight of expert e broadcast over the P partitions: sel_e^T @ rtT
            sel = nl.ndarray((E, P), dtype=nl.bfloat16, buffer=nl.sbuf)
            nisa.tensor_scalar(dst=sel, data=eio, op0=nl.equal, operand0=float(e))
            wbp = nl.ndarray((P, SC), dtype=nl.float32, buffer=nl.psum)
            nisa.nc_matmul(dst=wbp, stationary=sel, moving=rtT)
            wb = nl.ndarray((P, SC), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_copy(dst=wb, src=wbp)
            inter = nl.ndarray((P, IT, SC), dtype=nl.bfloat16, buffer=nl.sbuf)
            for it in range(IT):
                gts = []
                for half in range(2):
                    t = it + IT * half
                    acc = nl.ndarray((P, SC), dtype=nl.float32, buffer=nl.psum)
                    for kt in range(HT):
                        nisa.nc_matmul(dst=acc, stationary=gu_sb[0:P, (kt * GU_T + t) * P:(kt * GU_T + t + 1) * P],
                                       moving=xn2[0:P, kt, 0:SC])
                    g_ = nl.ndarray((P, SC), dtype=nl.bfloat16, buffer=nl.sbuf)
                    nisa.tensor_scalar(dst=g_, data=acc, op0=nl.add, operand0=bgu_sb[0:P, t:t + 1])
                    gts.append(g_)
                gate = nl.ndarray((P, SC), dtype=nl.bfloat16, buffer=nl.sbuf)
                nisa.tensor_scalar(dst=gate, data=gts[0], op0=nl.minimum, operand0=SWIGLU_LIMIT)
                up = nl.ndarray((P, SC), dtype=nl.bfloat16, buffer=nl.sbuf)
                nisa.tensor_scalar(dst=up, data=gts[1], op0=nl.minimum, operand0=SWIGLU_LIMIT, op1=nl.maximum,
                                   operand1=-SWIGLU_LIMIT)
                ga = nl.ndarray((P, SC), dtype=nl.bfloat16, buffer=nl.sbuf)
                nisa.tensor_scalar(dst=ga, data=gate, op0=nl.multiply, operand0=ALPHA)
                sg = nl.ndarray((P, SC), dtype=nl.bfloat16, buffer=nl.sbuf)
                nisa.activation(dst=sg, op=nl.sigmoid, data=ga)
                glu = nl.ndarray((P, SC), dtype=nl.bfloat16, buffer=nl.sbuf)
                nisa.tensor_tensor(dst=glu, data1=gate, data2=sg, op=nl.multiply)
                up1 = nl.ndarray((P, SC), dtype=nl.bfloat16, buffer=nl.sbuf)
                nisa.tensor_scalar(dst=up1, data=up, op0=nl.add, operand0=1.0)
                itr = nl.ndarray((P, SC), dtype=nl.bfloat16, buffer=nl.sbuf)
                nisa.tensor_tensor(dst=itr, data1=up1, data2=glu, op=nl.multiply)
                nisa.tensor_tensor(dst=inter[0:P, it, 0:SC], data1=itr, data2=wb, op=nl.multiply)
            for nt in range(HT):
                acc = nl.ndarray((P, SC), dtype=nl.float32, buffer=nl.psum)
                for it in range(IT):
                    nisa.nc_matmul(dst=acc, stationary=dn_sb[0:P, (it * HT + nt) * P:(it * HT + nt + 1) * P],
                                   moving=inter[0:P, it, 0:SC])
                if e == 0:
                    nisa.tensor_copy(dst=moe[0:P, nt, 0:SC], src=acc)
                else:
                    nisa.tensor_tensor(dst=moe[0:P, nt, 0:SC], data1=moe[0:P, nt, 0:SC], data2=acc, op=nl.add)
    mred = _allreduce_fm(moe, pid, xch, 1, rg, "pf_moe" + tag)
    # bias term: sum_e w_e(token) * bd[e, :]  (bd [E, H] fp32 -> stationary tiles [E, P])
    bd_sb = nl.ndarray((E, H), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=bd_sb, src=bd.ap(pattern=[[H, E], [1, H]], offset=li * E * H))
    rtf = nl.ndarray((E, SC), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=rtf, src=rtT)
    h2 = nl.ndarray((P, HT, SC), dtype=nl.bfloat16, buffer=nl.sbuf)
    for nt in range(HT):
        bp = nl.ndarray((P, SC), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_matmul(dst=bp, stationary=bd_sb[0:E, nt * P:(nt + 1) * P], moving=rtf)
        mo = nl.ndarray((P, SC), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=mo, data1=mred[0:P, nt, 0:SC], data2=bp, op=nl.add)
        nisa.tensor_tensor(dst=h2[0:P, nt, 0:SC], data1=h1_bf[0:P, nt, 0:SC], data2=mo, op=nl.add)
    if store:
        nisa.dma_copy(dst=hT.ap(pattern=[[MAXPOS, P], [P * MAXPOS, HT], [1, SC]], offset=0, scalar_offset=preg,
                                indirect_dim=1), src=h2)
    if dbg:
        o = nl.ndarray((P, HT, SC), dtype=nl.bfloat16, buffer=nl.shared_hbm)
        nisa.dma_copy(dst=o, src=h2)
        ok = nl.ndarray((D, SC), dtype=nl.bfloat16, buffer=nl.shared_hbm)
        nisa.dma_copy(dst=ok, src=qk[0:D, NQ, 0:SC])
        return h2, o, ok
    return h2


def _attn_pre(hT, p0, p_sb, preg, pb, ones_pp, ln1_sb, ln2_sb, bo_sb, bqkv_sb, rmat_sb, sinkb, wqkv_sb, wo_sb, wr_sb,
                csT, kc, vc, br, xch, xtm, rti, rdn, pid, rg, sliding, KB, slot, tag):
    """prefill_attn for the chunk at p0 (p_sb [1, 1] int32, preg register, pb [128, 1] fp32 broadcast) with the
    layer's weights already in SBUF; slot / tag: exchange slot and collective buffer names of this chunk."""
    MAXPOS = hT.shape[1]
    T = kc.shape[2]
    cs_sb = nl.ndarray((128, SC), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=cs_sb, src=csT.ap(pattern=[[MAXPOS, 128], [1, SC]], offset=p0))

    # ---- load chunk, norm
    x_bf = nl.ndarray((P, HT, SC), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.dma_copy(dst=x_bf, src=hT.ap(pattern=[[MAXPOS, P], [P * MAXPOS, HT], [1, SC]], offset=p0))
    x = nl.ndarray((P, HT, SC), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=x, src=x_bf)
    xn = _rmsnorm_fm(x, ln1_sb, ones_pp)

    # ---- QKV + RoPE -> qk [D, QKV_T, SC] bf16 (q heads 0..7, k, v)
    qk = nl.ndarray((D, QKV_T, SC), dtype=nl.bfloat16, buffer=nl.sbuf)
    for t in range(QKV_T):
        acc = nl.ndarray((D, SC), dtype=nl.float32, buffer=nl.psum)
        for kt in range(HT):
            nisa.nc_matmul(dst=acc, stationary=wqkv_sb[0:P, (kt * QKV_T + t) * D:(kt * QKV_T + t + 1) * D],
                           moving=xn[0:P, kt, 0:SC])
        raw = nl.ndarray((D, SC), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=raw, data=acc, op0=nl.add, operand0=bqkv_sb[0:D, t:t + 1])
        if t == QKV_T - 1:
            nisa.tensor_copy(dst=qk[0:D, t, 0:SC], src=raw)
        else:
            rot = nl.ndarray((D, SC), dtype=nl.float32, buffer=nl.psum)
            nisa.nc_matmul(dst=rot, stationary=rmat_sb, moving=raw)
            t1 = nl.ndarray((D, SC), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_tensor(dst=t1, data1=raw, data2=cs_sb[0:D, 0:SC], op=nl.multiply)
            t2 = nl.ndarray((D, SC), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_tensor(dst=t2, data1=rot, data2=cs_sb[D:2 * D, 0:SC], op=nl.multiply)
            nisa.tensor_tensor(dst=qk[0:D, t, 0:SC], data1=t1, data2=t2, op=nl.add)

    # ---- write K^T and V of the chunk at [p0, p0 + SC)
    nisa.dma_copy(dst=kc.ap(pattern=[[T, D], [1, SC]], offset=pid * D * T + p0),
                  src=qk[0:D, NQ, 0:SC])
    for j in range(NQT):
        vt = nl.ndarray((128, D), dtype=nl.bfloat16, buffer=nl.psum)
        nisa.nc_transpose(dst=vt, data=qk[0:D, NQ + 1, j * 128:(j + 1) * 128])
        vts = nl.ndarray((128, D), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.tensor_copy(dst=vts, src=vt)
        nisa.dma_copy(dst=vc.ap(pattern=[[D, 128], [1, D]], offset=pid * T * D + (p0 + j * 128) * D), src=vts)

    # ---- attention (reads the caches after the writes above)
    if sliding:
        NK = SLIDE + SC
        ks_sb = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.sbuf)  # key window start max(p0 - 128, 0)
        nisa.tensor_scalar(dst=ks_sb, data=p_sb, op0=nl.add, operand0=-float(SLIDE), op1=nl.maximum, operand1=0.0)
        kreg = nisa.register_alloc()
        nisa.register_load(kreg, ks_sb)
        ksb_i = nl.ndarray((128, 1), dtype=nl.int32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=ksb_i, data=pb, op0=nl.add, operand0=-float(SLIDE), op1=nl.maximum, operand1=0.0)
        kstart = nl.ndarray((128, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=kstart, src=ksb_i)
    else:
        NK = KB
    NKB = NK // 128
    k_sb = nl.ndarray((D, NK), dtype=nl.bfloat16, buffer=nl.sbuf)
    v_sb = nl.ndarray((128, NKB, D), dtype=nl.bfloat16, buffer=nl.sbuf)
    if sliding:
        nisa.dma_copy(dst=k_sb, src=kc.ap(pattern=[[T, D], [1, NK]], offset=pid * D * T + max(p0 - SLIDE, 0)))
        nisa.dma_copy(dst=v_sb, src=vc.ap(pattern=[[D, 128], [128 * D, NKB], [1, D]],
                                          offset=pid * T * D + max(p0 - SLIDE, 0) * D))
    else:
        nisa.dma_copy(dst=k_sb, src=kc.ap(pattern=[[T, D], [1, NK]], offset=pid * D * T))
        nisa.dma_copy(dst=v_sb, src=vc.ap(pattern=[[D, 128], [128 * D, NKB], [1, D]], offset=pid * T * D))

    ao = nl.ndarray((D, NQ, SC), dtype=nl.bfloat16, buffer=nl.sbuf)
    CH = 512 if NK % 512 == 0 else (256 if NK % 256 == 0 else 128)
    assert NK % CH == 0
    for qt in range(NQT):
        # mask[i, j]: query position qp = p0 + qt*128 + i, key position kp = kstart + j
        dif = nl.ndarray((128, NK), dtype=nl.float32, buffer=nl.sbuf)  # kp - qp
        nisa.iota(dst=dif, pattern=[[1, NK]], offset=-qt * 128, channel_multiplier=-1)
        off = nl.ndarray((128, 1), dtype=nl.float32, buffer=nl.sbuf)  # kstart - p0
        if sliding:
            nisa.tensor_tensor(dst=off, data1=kstart, data2=pb, op=nl.subtract)
        else:
            nisa.tensor_scalar(dst=off, data=pb, op0=nl.multiply, operand0=-1.0)
        d2 = nl.ndarray((128, NK), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=d2, data=dif, op0=nl.add, operand0=off)
        msk = nl.ndarray((128, NK), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=msk, data=d2, op0=nl.greater, operand0=0.0, op1=nl.multiply, operand1=NEG)
        if sliding:
            m2 = nl.ndarray((128, NK), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_scalar(dst=m2, data=d2, op0=nl.less_equal, operand0=-float(SLIDE), op1=nl.multiply, operand1=NEG)
            m3 = nl.ndarray((128, NK), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_tensor(dst=m3, data1=msk, data2=m2, op=nl.add)
            msk = m3
        for hh in range(NQ):
            s_bf = nl.ndarray((128, NK), dtype=nl.bfloat16, buffer=nl.sbuf)
            for c in range(NK // CH):
                sp = nl.ndarray((128, CH), dtype=nl.float32, buffer=nl.psum)
                nisa.nc_matmul(dst=sp, stationary=qk[0:D, hh, qt * 128:(qt + 1) * 128], moving=k_sb[0:D, c * CH:(c + 1) * CH])
                nisa.tensor_copy(dst=s_bf[0:128, c * CH:(c + 1) * CH], src=sp)
            s = nl.ndarray((128, NK), dtype=nl.float32, buffer=nl.sbuf)
            nisa.scalar_tensor_tensor(dst=s, data=s_bf, op0=nl.multiply, operand0=0.125, op1=nl.add, operand1=msk)
            m1 = nl.ndarray((128, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_reduce(dst=m1, op=nl.maximum, data=s, axis=1)
            m = nl.ndarray((128, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_tensor(dst=m, data1=m1, data2=sinkb[0:128, hh:hh + 1], op=nl.maximum)
            negm = nl.ndarray((128, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_scalar(dst=negm, data=m, op0=nl.multiply, operand0=-1.0)
            pexp = nl.ndarray((128, NK), dtype=nl.float32, buffer=nl.sbuf)
            ssum = nl.ndarray((128, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.activation_reduce(dst=pexp, op=nl.exp, data=s, reduce_op=nl.add, reduce_res=ssum, bias=negm, scale=1.0)
            es = nl.ndarray((128, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.activation(dst=es, op=nl.exp, data=sinkb[0:128, hh:hh + 1], bias=negm, scale=1.0)
            tot = nl.ndarray((128, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_tensor(dst=tot, data1=ssum, data2=es, op=nl.add)
            rt = nl.ndarray((128, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.reciprocal(dst=rt, data=tot)
            prob = nl.ndarray((128, NK), dtype=nl.bfloat16, buffer=nl.sbuf)
            nisa.tensor_scalar(dst=prob, data=pexp, op0=nl.multiply, operand0=rt)
            acc_o = nl.ndarray((D, 128), dtype=nl.float32, buffer=nl.psum)
            for kb in range(NKB):
                ptp = nl.ndarray((128, 128), dtype=nl.bfloat16, buffer=nl.psum)
                nisa.nc_transpose(dst=ptp, data=prob[0:128, kb * 128:(kb + 1) * 128])
                pts = nl.ndarray((128, 128), dtype=nl.bfloat16, buffer=nl.sbuf)
                nisa.tensor_copy(dst=pts, src=ptp)
                nisa.nc_matmul(dst=acc_o, stationary=v_sb[0:128, kb, 0:D], moving=pts)
            nisa.tensor_copy(dst=ao[0:D, hh, qt * 128:(qt + 1) * 128], src=acc_o)

    # ---- o_proj + all-reduce + bias + residual
    part = nl.ndarray((P, HT, SC), dtype=nl.float32, buffer=nl.sbuf)
    for nt in range(HT):
        acc = nl.ndarray((P, SC), dtype=nl.float32, buffer=nl.psum)
        for hh in range(NQ):
            nisa.nc_matmul(dst=acc, stationary=wo_sb[0:D, (hh * HT + nt) * P:(hh * HT + nt + 1) * P], moving=ao[0:D, hh, 0:SC])
        nisa.tensor_copy(dst=part[0:P, nt, 0:SC], src=acc)
    red = _allreduce_fm(part, pid, xch, slot, rg, "pf_attn" + tag)
    return red, x_bf


def _attn_post(red, x_bf, hT, p0, p_sb, preg, pb, ones_pp, ln1_sb, ln2_sb, bo_sb, bqkv_sb, rmat_sb, sinkb, wqkv_sb, wo_sb, wr_sb,
                csT, kc, vc, br, xch, xtm, rti, rdn, pid, rg, sliding, KB, slot, tag):
    """prefill_attn for the chunk at p0 (p_sb [1, 1] int32, preg register, pb [128, 1] fp32 broadcast) with the
    layer's weights already in SBUF; slot / tag: exchange slot and collective buffer names of this chunk."""
    MAXPOS = hT.shape[1]
    h1 = nl.ndarray((P, HT, SC), dtype=nl.float32, buffer=nl.sbuf)
    h1_bf = nl.ndarray((P, HT, SC), dtype=nl.bfloat16, buffer=nl.sbuf)
    for kt in range(HT):
        at = nl.ndarray((P, SC), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=at, data=red[0:P, kt, 0:SC], op0=nl.add, operand0=bo_sb[0:P, kt:kt + 1])
        nisa.tensor_tensor(dst=h1_bf[0:P, kt, 0:SC], data1=x_bf[0:P, kt, 0:SC], data2=at, op=nl.add)
    nisa.tensor_copy(dst=h1, src=h1_bf)

    # ---- norm + router + top-4 per token
    xn2 = _rmsnorm_fm(h1, ln2_sb, ones_pp)
    lgp = nl.ndarray((E, SC), dtype=nl.float32, buffer=nl.psum)
    for kt in range(HT):
        nisa.nc_matmul(dst=lgp, stationary=wr_sb[0:P, kt * E:(kt + 1) * E], moving=xn2[0:P, kt, 0:SC])
    brc = nl.ndarray((E, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=brc, src=br.ap(pattern=[[1, E], [1, 1]], offset=0))
    lgb = nl.ndarray((E, SC), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=lgb, data=lgp, op0=nl.add, operand0=brc)
    lgf = nl.ndarray((E, SC), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=lgf, src=lgb)
    io = nl.ndarray((128, E), dtype=nl.float32, buffer=nl.sbuf)
    nisa.iota(dst=io, pattern=[[1, E]], offset=0, channel_multiplier=0)
    rev = nl.ndarray((128, E), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=rev, data=io, op0=nl.multiply, operand0=-1.0, op1=nl.add, operand1=64.0)
    for j in range(NQT):
        tp = nl.ndarray((128, E), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_transpose(dst=tp, data=lgf[0:E, j * 128:(j + 1) * 128])
        cur = nl.ndarray((128, E), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=cur, src=tp)
        vals = nl.ndarray((128, TOPK), dtype=nl.float32, buffer=nl.sbuf)
        r8 = nl.ndarray((128, 2 * TOPK), dtype=nl.float32, buffer=nl.sbuf)  # expert ids, routing weights
        hits = []
        for i in range(TOPK):
            mx = nl.ndarray((128, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_reduce(dst=mx, op=nl.maximum, data=cur, axis=1)
            nisa.tensor_copy(dst=vals[0:128, i:i + 1], src=mx)
            sc = nl.ndarray((128, E), dtype=nl.float32, buffer=nl.sbuf)
            nisa.scalar_tensor_tensor(dst=sc, data=cur, op0=nl.equal, operand0=mx, op1=nl.multiply, operand1=rev)
            sm = nl.ndarray((128, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_reduce(dst=sm, op=nl.maximum, data=sc, axis=1)
            nisa.tensor_scalar(dst=r8[0:128, i:i + 1], data=sm, op0=nl.multiply, operand0=-1.0, op1=nl.add,
                               operand1=64.0)
            hit = nl.ndarray((128, E), dtype=nl.float32, buffer=nl.sbuf)  # one-hot of the selected expert
            nisa.tensor_scalar(dst=hit, data=rev, op0=nl.equal, operand0=sm)
            hits.append(hit)
            nxt = nl.ndarray((128, E), dtype=nl.float32, buffer=nl.sbuf)
            nisa.scalar_tensor_tensor(dst=nxt, data=hit, op0=nl.multiply, operand0=NEG, op1=nl.add, operand1=cur)
            cur = nxt
        negv = nl.ndarray((128, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=negv, data=vals[0:128, 0:1], op0=nl.multiply, operand0=-1.0)
        ev = nl.ndarray((128, TOPK), dtype=nl.float32, buffer=nl.sbuf)
        esum = nl.ndarray((128, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.activation_reduce(dst=ev, op=nl.exp, data=vals, reduce_op=nl.add, reduce_res=esum, bias=negv, scale=1.0)
        rs = nl.ndarray((128, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.reciprocal(dst=rs, data=esum)
        wbf = nl.ndarray((128, TOPK), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=wbf, data=ev, op0=nl.multiply, operand0=rs)
        wf = nl.ndarray((128, TOPK), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=wf, src=wbf)
        dense = nl.ndarray((128, E), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=dense, data=hits[0], op0=nl.multiply, operand0=wf[0:128, 0:1])
        for i in range(1, TOPK):
            d2_ = nl.ndarray((128, E), dtype=nl.float32, buffer=nl.sbuf)
            nisa.scalar_tensor_tensor(dst=d2_, data=hits[i], op0=nl.multiply, operand0=wf[0:128, i:i + 1], op1=nl.add,
                                      operand1=dense)
            dense = d2_
        nisa.tensor_copy(dst=r8[0:128, TOPK:2 * TOPK], src=wf)
        nisa.dma_copy(dst=rti.ap(pattern=[[2 * TOPK, 128], [1, 2 * TOPK]], offset=(p0 + j * 128) * 2 * TOPK), src=r8)
        nisa.dma_copy(dst=rdn.ap(pattern=[[E, 128], [1, E]], offset=(p0 + j * 128) * E),
                      src=dense)
        # token-major copy of the normalized activations for the MoE gathers
        xr = nl.ndarray((128, H), dtype=nl.bfloat16, buffer=nl.sbuf)
        for kt in range(HT):
            xp = nl.ndarray((128, P), dtype=nl.bfloat16, buffer=nl.psum)
            nisa.nc_transpose(dst=xp, data=xn2[0:P, kt, j * 128:(j + 1) * 128])
            nisa.tensor_copy(dst=xr[0:128, kt * P:(kt + 1) * P], src=xp)
        nisa.dma_copy(dst=xtm.ap(pattern=[[H, 128], [1, H]], offset=(p0 + j * 128) * H),
                      src=xr)
    nisa.dma_copy(dst=hT.ap(pattern=[[MAXPOS, P], [P * MAXPOS, HT], [1, SC]], offset=p0), src=h1_bf)


def _fa_qkv(hT, p0, ones_pp, ln1_sb, bqkvh, rmat2, wqkv_sb, csT, kc, vc, qbuf, kbuf, vbuf, pid, S):
    """Flash-attention prefill, phase 1 for the chunk at p0: norm, QKV, RoPE; K^T / V to the caches and, with q
    scaled by 1/8 (exact), q / K^T / V to the contiguous scratch qbuf [16, D, S], kbuf [2, D, S], vbuf [2, S, D]."""
    MAXPOS = hT.shape[1]
    T = kc.shape[2]
    cs_sb = nl.ndarray((128, SC), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=cs_sb, src=csT.ap(pattern=[[MAXPOS, 128], [1, SC]], offset=p0))

    # ---- load chunk, norm
    x_bf = nl.ndarray((P, HT, SC), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.dma_copy(dst=x_bf, src=hT.ap(pattern=[[MAXPOS, P], [P * MAXPOS, HT], [1, SC]], offset=p0))
    x = nl.ndarray((P, HT, SC), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=x, src=x_bf)
    xn = _rmsnorm_fm(x, ln1_sb, ones_pp)

    # ---- QKV + RoPE, head pairs on 128 partitions: qk2[j * 64 + d, hp, s] = (q heads 0..7, k, v)[2 hp + j][d, s]
    # (128-column stationary tiles: half the TensorE instructions of one head per tile; RoPE by a block-diagonal
    # rotation, both exact rearrangements of the same arithmetic)
    NP_ = QKV_T // 2
    bq2 = nl.ndarray((128, NP_), dtype=nl.float32, buffer=nl.sbuf)
    for jj in range(2):
        nisa.dma_copy(dst=bq2[jj * D:(jj + 1) * D, 0:NP_], src=bqkvh.ap(pattern=[[QKV_T, D], [2, NP_]], offset=pid * D * QKV_T + jj))
    cs2 = nl.ndarray((128, 2, SC), dtype=nl.float32, buffer=nl.sbuf)  # [:, 0] cos, [:, 1] sin, both halves
    for jj in range(2):
        nisa.dma_copy(dst=cs2[jj * D:(jj + 1) * D, 0:2, 0:SC], src=csT.ap(pattern=[[MAXPOS, D], [D * MAXPOS, 2], [1, SC]], offset=p0))
    qk2 = nl.ndarray((128, NP_, SC), dtype=nl.bfloat16, buffer=nl.sbuf)
    for hp in range(NP_):
        acc = nl.ndarray((128, SC), dtype=nl.float32, buffer=nl.psum)
        for kt in range(HT):
            nisa.nc_matmul(dst=acc, stationary=wqkv_sb[0:P, (kt * QKV_T + 2 * hp) * D:(kt * QKV_T + 2 * hp + 2) * D],
                           moving=xn[0:P, kt, 0:SC])
        raw = nl.ndarray((128, SC), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=raw, data=acc, op0=nl.add, operand0=bq2[0:128, hp:hp + 1])
        rot = nl.ndarray((128, SC), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_matmul(dst=rot, stationary=rmat2, moving=raw)
        t1 = nl.ndarray((128, SC), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=t1, data1=raw, data2=cs2[0:128, 0, 0:SC], op=nl.multiply)
        t2 = nl.ndarray((128, SC), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=t2, data1=rot, data2=cs2[0:128, 1, 0:SC], op=nl.multiply)
        if hp == NP_ - 1:  # (k, v): RoPE on k only
            nisa.tensor_tensor(dst=qk2[0:D, hp, 0:SC], data1=t1[0:D, 0:SC], data2=t2[0:D, 0:SC], op=nl.add)
            nisa.tensor_copy(dst=qk2[D:2 * D, hp, 0:SC], src=raw[D:2 * D, 0:SC])
        else:
            nisa.tensor_tensor(dst=qk2[0:128, hp, 0:SC], data1=t1, data2=t2, op=nl.add)
    KV = NP_ - 1

    # ---- write K^T and V of the chunk at [p0, p0 + SC) (V: token-major through SBUF -> HBM DMAs of its 64 rows)
    nisa.dma_copy(dst=kc.ap(pattern=[[T, D], [1, SC]], offset=pid * D * T + p0),
                  src=qk2[0:D, KV, 0:SC])
    vfm = nl.ndarray((D, SC), dtype=nl.bfloat16, buffer=nl.sbuf)  # V feature-major at partitions 0..63
    nisa.dma_copy(dst=vfm, src=qk2[D:2 * D, KV, 0:SC])
    for j in range(NQT):
        vt = nl.ndarray((128, D), dtype=nl.bfloat16, buffer=nl.psum)
        nisa.nc_transpose(dst=vt, data=vfm[0:D, j * 128:(j + 1) * 128])
        vts = nl.ndarray((128, D), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.tensor_copy(dst=vts, src=vt)
        nisa.dma_copy(dst=vc.ap(pattern=[[D, 128], [1, D]], offset=pid * T * D + (p0 + j * 128) * D), src=vts)
        nisa.dma_copy(dst=vbuf.ap(pattern=[[D, 128], [1, D]], offset=pid * S * D + (p0 + j * 128) * D), src=vts)

    # ---- the same chunk for the whole-sequence attention (q pre-scaled: nkilib's sliding window needs scale 1)
    qs = nl.ndarray((128, NQ // 2, SC), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=qs, data=qk2[0:128, 0:NQ // 2, 0:SC], op0=nl.multiply, operand0=0.125)
    nisa.dma_copy(dst=qbuf.ap(pattern=[[S, 128], [2 * D * S, NQ // 2], [1, SC]], offset=pid * NQ * D * S + p0), src=qs)
    nisa.dma_copy(dst=kbuf.ap(pattern=[[S, D], [1, SC]], offset=pid * D * S + p0), src=qk2[0:D, KV, 0:SC])


def _fa_out(hT, p0, ones_pp, ln2_sb, bo_sb, wo_sb, wr_sb, br, xch, xtm, rti, rdn, obuf, pid, rg, slot, tag, S):
    """Flash-attention prefill, phase 3 for the chunk at p0: o_proj of the attention output (obuf [16, D, S]),
    all-reduce, residual, post-attention norm, router, MoE inputs (see prefill_attn)."""
    MAXPOS = hT.shape[1]
    x_bf = nl.ndarray((P, HT, SC), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.dma_copy(dst=x_bf, src=hT.ap(pattern=[[MAXPOS, P], [P * MAXPOS, HT], [1, SC]], offset=p0))
    ao = nl.ndarray((D, NQ, SC), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.dma_copy(dst=ao, src=obuf.ap(pattern=[[S, D], [D * S, NQ], [1, SC]], offset=pid * NQ * D * S + p0))
    # ---- o_proj + all-reduce + bias + residual
    part = nl.ndarray((P, HT, SC), dtype=nl.float32, buffer=nl.sbuf)
    for nt in range(HT):
        acc = nl.ndarray((P, SC), dtype=nl.float32, buffer=nl.psum)
        for hh in range(NQ):
            nisa.nc_matmul(dst=acc, stationary=wo_sb[0:D, (hh * HT + nt) * P:(hh * HT + nt + 1) * P], moving=ao[0:D, hh, 0:SC])
        nisa.tensor_copy(dst=part[0:P, nt, 0:SC], src=acc)
    red = _allreduce_fm(part, pid, xch, slot, rg, "pf_attn" + tag)
    h1 = nl.ndarray((P, HT, SC), dtype=nl.float32, buffer=nl.sbuf)
    h1_bf = nl.ndarray((P, HT, SC), dtype=nl.bfloat16, buffer=nl.sbuf)
    for kt in range(HT):
        at = nl.ndarray((P, SC), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=at, data=red[0:P, kt, 0:SC], op0=nl.add, operand0=bo_sb[0:P, kt:kt + 1])
        nisa.tensor_tensor(dst=h1_bf[0:P, kt, 0:SC], data1=x_bf[0:P, kt, 0:SC], data2=at, op=nl.add)
    nisa.tensor_copy(dst=h1, src=h1_bf)

    # ---- norm + router + top-4 per token
    xn2 = _rmsnorm_fm(h1, ln2_sb, ones_pp)
    lgp = nl.ndarray((E, SC), dtype=nl.float32, buffer=nl.psum)
    for kt in range(HT):
        nisa.nc_matmul(dst=lgp, stationary=wr_sb[0:P, kt * E:(kt + 1) * E], moving=xn2[0:P, kt, 0:SC])
    brc = nl.ndarray((E, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=brc, src=br.ap(pattern=[[1, E], [1, 1]], offset=0))
    lgb = nl.ndarray((E, SC), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=lgb, data=lgp, op0=nl.add, operand0=brc)
    lgf = nl.ndarray((E, SC), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=lgf, src=lgb)
    io = nl.ndarray((128, E), dtype=nl.float32, buffer=nl.sbuf)
    nisa.iota(dst=io, pattern=[[1, E]], offset=0, channel_multiplier=0)
    rev = nl.ndarray((128, E), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=rev, data=io, op0=nl.multiply, operand0=-1.0, op1=nl.add, operand1=64.0)
    for j in range(NQT):
        tp = nl.ndarray((128, E), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_transpose(dst=tp, data=lgf[0:E, j * 128:(j + 1) * 128])
        cur = nl.ndarray((128, E), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=cur, src=tp)
        vals = nl.ndarray((128, TOPK), dtype=nl.float32, buffer=nl.sbuf)
        r8 = nl.ndarray((128, 2 * TOPK), dtype=nl.float32, buffer=nl.sbuf)  # expert ids, routing weights
        hits = []
        for i in range(TOPK):
            mx = nl.ndarray((128, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_reduce(dst=mx, op=nl.maximum, data=cur, axis=1)
            nisa.tensor_copy(dst=vals[0:128, i:i + 1], src=mx)
            sc = nl.ndarray((128, E), dtype=nl.float32, buffer=nl.sbuf)
            nisa.scalar_tensor_tensor(dst=sc, data=cur, op0=nl.equal, operand0=mx, op1=nl.multiply, operand1=rev)
            sm = nl.ndarray((128, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_reduce(dst=sm, op=nl.maximum, data=sc, axis=1)
            nisa.tensor_scalar(dst=r8[0:128, i:i + 1], data=sm, op0=nl.multiply, operand0=-1.0, op1=nl.add,
                               operand1=64.0)
            hit = nl.ndarray((128, E), dtype=nl.float32, buffer=nl.sbuf)  # one-hot of the selected expert
            nisa.tensor_scalar(dst=hit, data=rev, op0=nl.equal, operand0=sm)
            hits.append(hit)
            nxt = nl.ndarray((128, E), dtype=nl.float32, buffer=nl.sbuf)
            nisa.scalar_tensor_tensor(dst=nxt, data=hit, op0=nl.multiply, operand0=NEG, op1=nl.add, operand1=cur)
            cur = nxt
        negv = nl.ndarray((128, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=negv, data=vals[0:128, 0:1], op0=nl.multiply, operand0=-1.0)
        ev = nl.ndarray((128, TOPK), dtype=nl.float32, buffer=nl.sbuf)
        esum = nl.ndarray((128, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.activation_reduce(dst=ev, op=nl.exp, data=vals, reduce_op=nl.add, reduce_res=esum, bias=negv, scale=1.0)
        rs = nl.ndarray((128, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.reciprocal(dst=rs, data=esum)
        wbf = nl.ndarray((128, TOPK), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=wbf, data=ev, op0=nl.multiply, operand0=rs)
        wf = nl.ndarray((128, TOPK), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=wf, src=wbf)
        dense = nl.ndarray((128, E), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=dense, data=hits[0], op0=nl.multiply, operand0=wf[0:128, 0:1])
        for i in range(1, TOPK):
            d2_ = nl.ndarray((128, E), dtype=nl.float32, buffer=nl.sbuf)
            nisa.scalar_tensor_tensor(dst=d2_, data=hits[i], op0=nl.multiply, operand0=wf[0:128, i:i + 1], op1=nl.add,
                                      operand1=dense)
            dense = d2_
        nisa.tensor_copy(dst=r8[0:128, TOPK:2 * TOPK], src=wf)
        nisa.dma_copy(dst=rti.ap(pattern=[[2 * TOPK, 128], [1, 2 * TOPK]], offset=(p0 + j * 128) * 2 * TOPK), src=r8)
        nisa.dma_copy(dst=rdn.ap(pattern=[[E, 128], [1, E]], offset=(p0 + j * 128) * E),
                      src=dense)
        # token-major copy of the normalized activations for the MoE gathers
        xr = nl.ndarray((128, H), dtype=nl.bfloat16, buffer=nl.sbuf)
        for kt in range(HT):
            xp = nl.ndarray((128, P), dtype=nl.bfloat16, buffer=nl.psum)
            nisa.nc_transpose(dst=xp, data=xn2[0:P, kt, j * 128:(j + 1) * 128])
            nisa.tensor_copy(dst=xr[0:128, kt * P:(kt + 1) * P], src=xp)
        nisa.dma_copy(dst=xtm.ap(pattern=[[H, 128], [1, H]], offset=(p0 + j * 128) * H),
                      src=xr)
    nisa.dma_copy(dst=hT.ap(pattern=[[MAXPOS, P], [P * MAXPOS, HT], [1, SC]], offset=p0), src=h1_bf)




def _fa_oproj(p0, wo_sb, xch, obuf, pid, c, NCHK, S):
    """Phase 3a for the chunk at p0: this core's o_proj partial (its 8 heads) [P, HT, SC] fp32 -> xch[pid, c]."""
    # head pairs on 128 partitions: obuf rows (2 hp + j) * D + d are consecutive, partition (j * 64 + d) of pair hp
    ao2 = nl.ndarray((128, NQ // 2, SC), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.dma_copy(dst=ao2, src=obuf.ap(pattern=[[S, 128], [2 * D * S, NQ // 2], [1, SC]], offset=pid * NQ * D * S + p0))
    part = nl.ndarray((P, HT, SC), dtype=nl.float32, buffer=nl.sbuf)
    for nt in range(HT):
        acc = nl.ndarray((P, SC), dtype=nl.float32, buffer=nl.psum)
        for hp in range(NQ // 2):
            nisa.nc_matmul(dst=acc, stationary=wo_sb[0:128, hp * H + nt * P:hp * H + (nt + 1) * P], moving=ao2[0:128, hp, 0:SC])
        nisa.tensor_copy(dst=part[0:P, nt, 0:SC], src=acc)
    F = HT * SC
    nisa.dma_copy(dst=xch.ap(pattern=[[F, P], [1, F]], offset=(pid * NCHK + c) * P * F), src=part)


def _fa_oproj_half(t0, wo_sb, obuf, pid, S):
    """This core's o_proj partial (its 8 heads) of the 128 tokens at t0: [P, HT, 128] fp32."""
    ao2 = nl.ndarray((128, NQ // 2, 128), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.dma_copy(dst=ao2, src=obuf.ap(pattern=[[S, 128], [2 * D * S, NQ // 2], [1, 128]], offset=pid * NQ * D * S + t0))
    part = nl.ndarray((P, HT, 128), dtype=nl.float32, buffer=nl.sbuf)
    for nt in range(HT):
        acc = nl.ndarray((P, 128), dtype=nl.float32, buffer=nl.psum)
        for hp in range(NQ // 2):
            nisa.nc_matmul(dst=acc, stationary=wo_sb[0:128, hp * H + nt * P:hp * H + (nt + 1) * P], moving=ao2[0:128, hp, 0:128])
        nisa.tensor_copy(dst=part[0:P, nt, 0:128], src=acc)
    return part


def _fa_post(hT, t0, red, ones_pp, ln2_sb, bo_sb, wr_sb, br, xtm, rti, rdn):
    """Phase 3c for the 128 tokens at t0 (this core's half of a chunk): red [P, HT, 128] fp32 = the all-reduced
    attention output; residual, post-attention norm, router, MoE inputs, hT[:, t0:t0 + 128] = h1."""
    MAXPOS = hT.shape[1]
    x_bf = nl.ndarray((P, HT, 128), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.dma_copy(dst=x_bf, src=hT.ap(pattern=[[MAXPOS, P], [P * MAXPOS, HT], [1, 128]], offset=t0))
    h1 = nl.ndarray((P, HT, 128), dtype=nl.float32, buffer=nl.sbuf)
    h1_bf = nl.ndarray((P, HT, 128), dtype=nl.bfloat16, buffer=nl.sbuf)
    for kt in range(HT):
        at = nl.ndarray((P, 128), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=at, data=red[0:P, kt, 0:128], op0=nl.add, operand0=bo_sb[0:P, kt:kt + 1])
        nisa.tensor_tensor(dst=h1_bf[0:P, kt, 0:128], data1=x_bf[0:P, kt, 0:128], data2=at, op=nl.add)
    nisa.tensor_copy(dst=h1, src=h1_bf)

    # ---- norm + router + top-4 per token
    xn2 = _rmsnorm_fm(h1, ln2_sb, ones_pp, 128)
    lgp = nl.ndarray((E, 128), dtype=nl.float32, buffer=nl.psum)
    for kt in range(HT):
        nisa.nc_matmul(dst=lgp, stationary=wr_sb[0:P, kt * E:(kt + 1) * E], moving=xn2[0:P, kt, 0:128])
    brc = nl.ndarray((E, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=brc, src=br.ap(pattern=[[1, E], [1, 1]], offset=0))
    lgb = nl.ndarray((E, 128), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=lgb, data=lgp, op0=nl.add, operand0=brc)
    lgf = nl.ndarray((E, 128), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=lgf, src=lgb)
    io = nl.ndarray((128, E), dtype=nl.float32, buffer=nl.sbuf)
    nisa.iota(dst=io, pattern=[[1, E]], offset=0, channel_multiplier=0)
    rev = nl.ndarray((128, E), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=rev, data=io, op0=nl.multiply, operand0=-1.0, op1=nl.add, operand1=64.0)
    for j in range(1):
        tp = nl.ndarray((128, E), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_transpose(dst=tp, data=lgf[0:E, j * 128:(j + 1) * 128])
        cur = nl.ndarray((128, E), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=cur, src=tp)
        vals = nl.ndarray((128, TOPK), dtype=nl.float32, buffer=nl.sbuf)
        r8 = nl.ndarray((128, 2 * TOPK), dtype=nl.float32, buffer=nl.sbuf)  # expert ids, routing weights
        hits = []
        for i in range(TOPK):
            mx = nl.ndarray((128, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_reduce(dst=mx, op=nl.maximum, data=cur, axis=1)
            nisa.tensor_copy(dst=vals[0:128, i:i + 1], src=mx)
            sc = nl.ndarray((128, E), dtype=nl.float32, buffer=nl.sbuf)
            nisa.scalar_tensor_tensor(dst=sc, data=cur, op0=nl.equal, operand0=mx, op1=nl.multiply, operand1=rev)
            sm = nl.ndarray((128, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_reduce(dst=sm, op=nl.maximum, data=sc, axis=1)
            nisa.tensor_scalar(dst=r8[0:128, i:i + 1], data=sm, op0=nl.multiply, operand0=-1.0, op1=nl.add,
                               operand1=64.0)
            hit = nl.ndarray((128, E), dtype=nl.float32, buffer=nl.sbuf)  # one-hot of the selected expert
            nisa.tensor_scalar(dst=hit, data=rev, op0=nl.equal, operand0=sm)
            hits.append(hit)
            nxt = nl.ndarray((128, E), dtype=nl.float32, buffer=nl.sbuf)
            nisa.scalar_tensor_tensor(dst=nxt, data=hit, op0=nl.multiply, operand0=NEG, op1=nl.add, operand1=cur)
            cur = nxt
        negv = nl.ndarray((128, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=negv, data=vals[0:128, 0:1], op0=nl.multiply, operand0=-1.0)
        ev = nl.ndarray((128, TOPK), dtype=nl.float32, buffer=nl.sbuf)
        esum = nl.ndarray((128, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.activation_reduce(dst=ev, op=nl.exp, data=vals, reduce_op=nl.add, reduce_res=esum, bias=negv, scale=1.0)
        rs = nl.ndarray((128, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.reciprocal(dst=rs, data=esum)
        wbf = nl.ndarray((128, TOPK), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=wbf, data=ev, op0=nl.multiply, operand0=rs)
        wf = nl.ndarray((128, TOPK), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=wf, src=wbf)
        dense = nl.ndarray((128, E), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=dense, data=hits[0], op0=nl.multiply, operand0=wf[0:128, 0:1])
        for i in range(1, TOPK):
            d2_ = nl.ndarray((128, E), dtype=nl.float32, buffer=nl.sbuf)
            nisa.scalar_tensor_tensor(dst=d2_, data=hits[i], op0=nl.multiply, operand0=wf[0:128, i:i + 1], op1=nl.add,
                                      operand1=dense)
            dense = d2_
        nisa.tensor_copy(dst=r8[0:128, TOPK:2 * TOPK], src=wf)
        nisa.dma_copy(dst=rti.ap(pattern=[[2 * TOPK, 128], [1, 2 * TOPK]], offset=t0 * 2 * TOPK), src=r8)
        nisa.dma_copy(dst=rdn.ap(pattern=[[E, 128], [1, E]], offset=t0 * E),
                      src=dense)
        # token-major copy of the normalized activations for the MoE gathers
        xr = nl.ndarray((128, H), dtype=nl.bfloat16, buffer=nl.sbuf)
        for kt in range(HT):
            xp = nl.ndarray((128, P), dtype=nl.bfloat16, buffer=nl.psum)
            nisa.nc_transpose(dst=xp, data=xn2[0:P, kt, j * 128:(j + 1) * 128])
            nisa.tensor_copy(dst=xr[0:128, kt * P:(kt + 1) * P], src=xp)
        nisa.dma_copy(dst=xtm.ap(pattern=[[H, 128], [1, H]], offset=t0 * H),
                      src=xr)
    nisa.dma_copy(dst=hT.ap(pattern=[[MAXPOS, P], [P * MAXPOS, HT], [1, 128]], offset=t0), src=h1_bf)


@nki.jit
def prefill_attn_fa(hT, ln1, wqkv, bqkv, csT, rmat, sinkf, kc, vc, wo2, bo, ln2, wr, br, xch, xtm, rti, rdn,
                    qbuf, kbuf, vbuf, obuf, rg, sliding, NCHK):
    """Attention half of a layer for the chunks 0 .. NCHK - 1 (positions [0, S), S = NCHK * SC) in one execution:
    phase 1 per chunk (QKV, RoPE, caches, scratch q/k/v), phase 2 one whole-sequence causal (sliding-window) flash
    attention with sinks (nkilib attention_cte, copied into kernels/nkilib_attn), phase 3 per chunk (o_proj,
    all-reduce, residual, norm, router; see prefill_attn). sinkf: the layer's sinks viewed as [16, 1].
    qbuf [16, D, S] / kbuf [2, D, S] / vbuf [2, S, D] / obuf [16, D, S] bf16, xch [2, MAXPOS, H] fp32: caller-provided
    scratch.
    nkilib places its SBUF buffers at fixed addresses, so no tile of phase 1 is used after phase 2 (the weights of
    phase 3 are loaded again)."""
    pid = nl.program_id(0)
    S = NCHK * SC
    ones_pp = _ones((P, P))
    ln1_sb = nl.ndarray((P, HT), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=ln1_sb, src=ln1)
    rmat2 = nl.ndarray((128, 128), dtype=nl.bfloat16, buffer=nl.sbuf)  # block-diagonal (rmat, rmat)
    nisa.memset(dst=rmat2, value=0.0)
    for jj in range(2):
        nisa.dma_copy(dst=rmat2[jj * D:(jj + 1) * D, jj * D:(jj + 1) * D], src=rmat)
    wqkv_sb = nl.ndarray((P, QKV_F), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.dma_copy(dst=wqkv_sb, src=wqkv.ap(pattern=[[QKV_F, P], [1, QKV_F]], offset=pid * P * QKV_F))
    for c in range(NCHK):
        _fa_qkv(hT, c * SC, ones_pp, ln1_sb, bqkv, rmat2, wqkv_sb, csT, kc, vc, qbuf, kbuf, vbuf, pid, S)
    _attention_cte(q=qbuf, k=kbuf, v=vbuf, scale=1.0, causal_mask=True, sink=sinkf, sliding_window=SLIDE if sliding else None,
                   tp_q=False, tp_k=False, tp_out=True, out_o_hbm=obuf)
    ones2 = _ones((P, P))
    ln2_sb = nl.ndarray((P, HT), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=ln2_sb, src=ln2)
    bo_sb = nl.ndarray((P, HT), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=bo_sb, src=bo)
    WO2 = WO_F // 2  # wo2: head pairs on 128 partitions (row (hh % 2) * 64 + d, column (hh // 2) * H + h)
    wo_sb = nl.ndarray((128, WO2), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.dma_copy(dst=wo_sb, src=wo2.ap(pattern=[[WO2, 128], [1, WO2]], offset=pid * 128 * WO2))
    wr_sb = nl.ndarray((P, HT * E), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.dma_copy(dst=wr_sb, src=wr)
    # phase 3: o_proj partials of all chunks -> xch; core pid then finishes the 128-token half pid of every chunk:
    # + the other core's partial -> src, ONE all-reduce over the 4 ranks, residual / norm / router / MoE inputs
    F = HT * SC
    src = nl.ndarray((2 * NCHK, P, HT * 128), dtype=nl.float32, buffer=nl.shared_hbm, name="pa_src")
    dst = nl.ndarray((2 * NCHK, P, HT * 128), dtype=nl.float32, buffer=nl.shared_hbm, name="pa_dst")
    if XCH_SENDRECV:
        # per chunk: o_proj of the token half the other core finishes (sent to it) and of this core's half (+ the
        # received partial of the other core) -> src; no xch round trip through HBM
        for c in range(NCHK):
            give = _fa_oproj_half(c * SC + (1 - pid) * 128, wo_sb, obuf, pid, S)
            mine = _fa_oproj_half(c * SC + pid * 128, wo_sb, obuf, pid, S)
            other = nl.ndarray((P, HT, 128), dtype=nl.float32, buffer=nl.sbuf)
            nisa.sendrecv(src=give, dst=other, send_to_rank=1 - pid, recv_from_rank=1 - pid, pipe_id=0,
                          dma_engine=nisa.dma_engine.dma)
            both = nl.ndarray((P, HT, 128), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_tensor(dst=both, data1=mine, data2=other, op=nl.add)
            nisa.dma_copy(dst=src.ap(pattern=[[HT * 128, P], [1, HT * 128]], offset=(2 * c + pid) * P * HT * 128), src=both)
    else:
        for c in range(NCHK):
            _fa_oproj(c * SC, wo_sb, xch, obuf, pid, c, NCHK, S)
        nisa.core_barrier(data=xch, cores=(0, 1))
    for c in range(NCHK if not XCH_SENDRECV else 0):
        mine = nl.ndarray((P, HT, 128), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=mine, src=xch.ap(pattern=[[F, P], [SC, HT], [1, 128]], offset=(pid * NCHK + c) * P * F + pid * 128))
        other = nl.ndarray((P, HT, 128), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=other, src=xch.ap(pattern=[[F, P], [SC, HT], [1, 128]],
                                            offset=((1 - pid) * NCHK + c) * P * F + pid * 128))
        both = nl.ndarray((P, HT, 128), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=both, data1=mine, data2=other, op=nl.add)
        nisa.dma_copy(dst=src.ap(pattern=[[HT * 128, P], [1, HT * 128]], offset=(2 * c + pid) * P * HT * 128), src=both)
    nisa.core_barrier(data=src, cores=(0, 1))
    ncc.all_reduce(srcs=[src], dsts=[dst], op=nl.add, replica_group=rg)
    for c in range(NCHK):
        red = nl.ndarray((P, HT, 128), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=red, src=dst.ap(pattern=[[HT * 128, P], [1, HT * 128]], offset=(2 * c + pid) * P * HT * 128))
        _fa_post(hT, c * SC + pid * 128, red, ones2, ln2_sb, bo_sb, wr_sb, br, xtm, rti, rdn)
    dummy = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.shared_hbm)
    nisa.dma_copy(dst=dummy, src=ln1_sb[0:1, 0:1])
    return dummy


@nki.jit
def prefill_attn_multi(hT, ln1, wqkv, bqkv, csT, rmat, sinks, kc, vc, wo, bo, ln2, wr, br, xch, xtm, rti, rdn,
                       rg, sliding, NCHK):
    """prefill_attn for the chunks 0 .. NCHK - 1 (positions [0, NCHK * SC)) in one execution: the layer's weights
    load once, every chunk's position is a constant and full layers attend to exactly [0, p0 + SC)."""
    pid = nl.program_id(0)
    ones_pp = _ones((P, P))
    ln1_sb = nl.ndarray((P, HT), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=ln1_sb, src=ln1)
    ln2_sb = nl.ndarray((P, HT), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=ln2_sb, src=ln2)
    bo_sb = nl.ndarray((P, HT), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=bo_sb, src=bo)
    bqkv_sb = nl.ndarray((D, QKV_T), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=bqkv_sb, src=bqkv.ap(pattern=[[QKV_T, D], [1, QKV_T]], offset=pid * D * QKV_T))
    rmat_sb = nl.ndarray((D, D), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.dma_copy(dst=rmat_sb, src=rmat)
    sinkb = nl.ndarray((128, NQ), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=sinkb, src=sinks.ap(pattern=[[0, 128], [1, NQ]], offset=pid * NQ))
    wqkv_sb = nl.ndarray((P, QKV_F), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.dma_copy(dst=wqkv_sb, src=wqkv.ap(pattern=[[QKV_F, P], [1, QKV_F]], offset=pid * P * QKV_F))
    wo_sb = nl.ndarray((D, WO_F), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.dma_copy(dst=wo_sb, src=wo.ap(pattern=[[WO_F, D], [1, WO_F]], offset=pid * D * WO_F))
    wr_sb = nl.ndarray((P, HT * E), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.dma_copy(dst=wr_sb, src=wr)
    for c in range(NCHK):
        p0 = c * SC
        p_sb = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.sbuf)
        nisa.memset(dst=p_sb, value=p0)
        preg = nisa.register_alloc(p0)
        pb = nl.ndarray((128, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.memset(dst=pb, value=float(p0))
        # chunk c's attention (up to its all-reduce) is issued before chunk c - 1's post-attention part, so that
        # every engine's in-order queue has independent work while the all-reduce runs
        cur = _attn_pre(hT, p0, p_sb, preg, pb, ones_pp, ln1_sb, ln2_sb, bo_sb, bqkv_sb, rmat_sb, sinkb, wqkv_sb, wo_sb,
                        wr_sb, csT, kc, vc, br, xch, xtm, rti, rdn, pid, rg, sliding, 0 if sliding else p0 + SC, c % 2,
                        TAGS[c])
        if PIPE == 0:
            _attn_post(cur[0], cur[1], hT, p0, p_sb, preg, pb, ones_pp, ln1_sb, ln2_sb, bo_sb, bqkv_sb, rmat_sb,
                       sinkb, wqkv_sb, wo_sb, wr_sb, csT, kc, vc, br, xch, xtm, rti, rdn, pid, rg, sliding, 0,
                       c % 2, TAGS[c])
        if PIPE == 1 and c > 0:
            _attn_post(prev[0], prev[1], hT, p0 - SC, p_prev, preg, pb, ones_pp, ln1_sb, ln2_sb, bo_sb, bqkv_sb, rmat_sb,
                       sinkb, wqkv_sb, wo_sb, wr_sb, csT, kc, vc, br, xch, xtm, rti, rdn, pid, rg, sliding, 0,
                       (c - 1) % 2, TAGS[c - 1])
        prev = cur
        p_prev = p_sb
    if PIPE == 1:
        _attn_post(prev[0], prev[1], hT, (NCHK - 1) * SC, p_prev, preg, pb, ones_pp, ln1_sb, ln2_sb, bo_sb, bqkv_sb, rmat_sb,
                   sinkb, wqkv_sb, wo_sb, wr_sb, csT, kc, vc, br, xch, xtm, rti, rdn, pid, rg, sliding, 0, (NCHK - 1) % 2,
                   TAGS[NCHK - 1])
    dummy = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.shared_hbm)
    nisa.dma_copy(dst=dummy, src=p_sb)
    return dummy


@nki.jit
def prefill_attn(hT, pos, ln1, wqkv, bqkv, csT, rmat, sinks, kc, vc, wo, bo, ln2, wr, br, xch, xtm, rti, rdn,
                 rg, sliding, KB):
    """Attention half of a decoder layer for the SC tokens at positions [p0, p0 + SC), p0 = pos[0, 0]: K/V of the
    chunk written to the caches, hT[:, chunk] = h1 (the residual stream after attention, in place), and for the MoE
    (prefill_moe): xtm[t] = post-attention-norm activations (token-major [MAXPOS, H] bf16), rti[t] = (top-4 expert
    ids, their routing weights) [MAXPOS, 8] fp32, rdn[t] = dense routing weights [MAXPOS, E] fp32.
    csT [128, MAXPOS] fp32 (cos x2 rows | sin x2 rows).
    Full layers attend to cache positions [0, KB) (KB >= p0 + SC); sliding layers to [p0 - 128, p0 + SC)."""
    pid = nl.program_id(0)
    MAXPOS = hT.shape[1]
    T = kc.shape[2]
    p_sb, preg, pb = _p0(pos)
    ones_pp = _ones((P, P))
    ln1_sb = nl.ndarray((P, HT), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=ln1_sb, src=ln1)
    ln2_sb = nl.ndarray((P, HT), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=ln2_sb, src=ln2)
    bo_sb = nl.ndarray((P, HT), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=bo_sb, src=bo)
    bqkv_sb = nl.ndarray((D, QKV_T), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=bqkv_sb, src=bqkv.ap(pattern=[[QKV_T, D], [1, QKV_T]], offset=pid * D * QKV_T))
    rmat_sb = nl.ndarray((D, D), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.dma_copy(dst=rmat_sb, src=rmat)
    sinkb = nl.ndarray((128, NQ), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=sinkb, src=sinks.ap(pattern=[[0, 128], [1, NQ]], offset=pid * NQ))
    wqkv_sb = nl.ndarray((P, QKV_F), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.dma_copy(dst=wqkv_sb, src=wqkv.ap(pattern=[[QKV_F, P], [1, QKV_F]], offset=pid * P * QKV_F))
    wo_sb = nl.ndarray((D, WO_F), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.dma_copy(dst=wo_sb, src=wo.ap(pattern=[[WO_F, D], [1, WO_F]], offset=pid * D * WO_F))
    wr_sb = nl.ndarray((P, HT * E), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.dma_copy(dst=wr_sb, src=wr)
    cs_sb = nl.ndarray((128, SC), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=cs_sb, src=csT.ap(pattern=[[MAXPOS, 128], [1, SC]], offset=0, scalar_offset=preg, indirect_dim=1))

    # ---- load chunk, norm
    x_bf = nl.ndarray((P, HT, SC), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.dma_copy(dst=x_bf, src=hT.ap(pattern=[[MAXPOS, P], [P * MAXPOS, HT], [1, SC]], offset=0,
                                      scalar_offset=preg, indirect_dim=1))
    x = nl.ndarray((P, HT, SC), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=x, src=x_bf)
    xn = _rmsnorm_fm(x, ln1_sb, ones_pp)

    # ---- QKV + RoPE -> qk [D, QKV_T, SC] bf16 (q heads 0..7, k, v)
    qk = nl.ndarray((D, QKV_T, SC), dtype=nl.bfloat16, buffer=nl.sbuf)
    for t in range(QKV_T):
        acc = nl.ndarray((D, SC), dtype=nl.float32, buffer=nl.psum)
        for kt in range(HT):
            nisa.nc_matmul(dst=acc, stationary=wqkv_sb[0:P, (kt * QKV_T + t) * D:(kt * QKV_T + t + 1) * D],
                           moving=xn[0:P, kt, 0:SC])
        raw = nl.ndarray((D, SC), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=raw, data=acc, op0=nl.add, operand0=bqkv_sb[0:D, t:t + 1])
        if t == QKV_T - 1:
            nisa.tensor_copy(dst=qk[0:D, t, 0:SC], src=raw)
        else:
            rot = nl.ndarray((D, SC), dtype=nl.float32, buffer=nl.psum)
            nisa.nc_matmul(dst=rot, stationary=rmat_sb, moving=raw)
            t1 = nl.ndarray((D, SC), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_tensor(dst=t1, data1=raw, data2=cs_sb[0:D, 0:SC], op=nl.multiply)
            t2 = nl.ndarray((D, SC), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_tensor(dst=t2, data1=rot, data2=cs_sb[D:2 * D, 0:SC], op=nl.multiply)
            nisa.tensor_tensor(dst=qk[0:D, t, 0:SC], data1=t1, data2=t2, op=nl.add)

    # ---- write K^T and V of the chunk at [p0, p0 + SC)
    nisa.dma_copy(dst=kc.ap(pattern=[[T, D], [1, SC]], offset=pid * D * T, scalar_offset=preg, indirect_dim=2),
                  src=qk[0:D, NQ, 0:SC])
    for j in range(NQT):
        vt = nl.ndarray((128, D), dtype=nl.bfloat16, buffer=nl.psum)
        nisa.nc_transpose(dst=vt, data=qk[0:D, NQ + 1, j * 128:(j + 1) * 128])
        vts = nl.ndarray((128, D), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.tensor_copy(dst=vts, src=vt)
        nisa.dma_copy(dst=vc.ap(pattern=[[D, 128], [1, D]], offset=pid * T * D + j * 128 * D, scalar_offset=preg,
                                indirect_dim=1), src=vts)

    # ---- attention (reads the caches after the writes above)
    if sliding:
        NK = SLIDE + SC
        ks_sb = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.sbuf)  # key window start max(p0 - 128, 0)
        nisa.tensor_scalar(dst=ks_sb, data=p_sb, op0=nl.add, operand0=-float(SLIDE), op1=nl.maximum, operand1=0.0)
        kreg = nisa.register_alloc()
        nisa.register_load(kreg, ks_sb)
        ksb_i = nl.ndarray((128, 1), dtype=nl.int32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=ksb_i, data=pb, op0=nl.add, operand0=-float(SLIDE), op1=nl.maximum, operand1=0.0)
        kstart = nl.ndarray((128, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=kstart, src=ksb_i)
    else:
        NK = KB
    NKB = NK // 128
    k_sb = nl.ndarray((D, NK), dtype=nl.bfloat16, buffer=nl.sbuf)
    v_sb = nl.ndarray((128, NKB, D), dtype=nl.bfloat16, buffer=nl.sbuf)
    if sliding:
        nisa.dma_copy(dst=k_sb, src=kc.ap(pattern=[[T, D], [1, NK]], offset=pid * D * T, scalar_offset=kreg, indirect_dim=2))
        nisa.dma_copy(dst=v_sb, src=vc.ap(pattern=[[D, 128], [128 * D, NKB], [1, D]], offset=pid * T * D,
                                          scalar_offset=kreg, indirect_dim=1))
    else:
        nisa.dma_copy(dst=k_sb, src=kc.ap(pattern=[[T, D], [1, NK]], offset=pid * D * T))
        nisa.dma_copy(dst=v_sb, src=vc.ap(pattern=[[D, 128], [128 * D, NKB], [1, D]], offset=pid * T * D))

    ao = nl.ndarray((D, NQ, SC), dtype=nl.bfloat16, buffer=nl.sbuf)
    CH = 512 if NK % 512 == 0 else 128
    for qt in range(NQT):
        # mask[i, j]: query position qp = p0 + qt*128 + i, key position kp = kstart + j
        dif = nl.ndarray((128, NK), dtype=nl.float32, buffer=nl.sbuf)  # kp - qp
        nisa.iota(dst=dif, pattern=[[1, NK]], offset=-qt * 128, channel_multiplier=-1)
        off = nl.ndarray((128, 1), dtype=nl.float32, buffer=nl.sbuf)  # kstart - p0
        if sliding:
            nisa.tensor_tensor(dst=off, data1=kstart, data2=pb, op=nl.subtract)
        else:
            nisa.tensor_scalar(dst=off, data=pb, op0=nl.multiply, operand0=-1.0)
        d2 = nl.ndarray((128, NK), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=d2, data=dif, op0=nl.add, operand0=off)
        msk = nl.ndarray((128, NK), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=msk, data=d2, op0=nl.greater, operand0=0.0, op1=nl.multiply, operand1=NEG)
        if sliding:
            m2 = nl.ndarray((128, NK), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_scalar(dst=m2, data=d2, op0=nl.less_equal, operand0=-float(SLIDE), op1=nl.multiply, operand1=NEG)
            m3 = nl.ndarray((128, NK), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_tensor(dst=m3, data1=msk, data2=m2, op=nl.add)
            msk = m3
        for hh in range(NQ):
            s_bf = nl.ndarray((128, NK), dtype=nl.bfloat16, buffer=nl.sbuf)
            for c in range(NK // CH):
                sp = nl.ndarray((128, CH), dtype=nl.float32, buffer=nl.psum)
                nisa.nc_matmul(dst=sp, stationary=qk[0:D, hh, qt * 128:(qt + 1) * 128], moving=k_sb[0:D, c * CH:(c + 1) * CH])
                nisa.tensor_copy(dst=s_bf[0:128, c * CH:(c + 1) * CH], src=sp)
            s = nl.ndarray((128, NK), dtype=nl.float32, buffer=nl.sbuf)
            nisa.scalar_tensor_tensor(dst=s, data=s_bf, op0=nl.multiply, operand0=0.125, op1=nl.add, operand1=msk)
            m1 = nl.ndarray((128, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_reduce(dst=m1, op=nl.maximum, data=s, axis=1)
            m = nl.ndarray((128, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_tensor(dst=m, data1=m1, data2=sinkb[0:128, hh:hh + 1], op=nl.maximum)
            negm = nl.ndarray((128, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_scalar(dst=negm, data=m, op0=nl.multiply, operand0=-1.0)
            pexp = nl.ndarray((128, NK), dtype=nl.float32, buffer=nl.sbuf)
            ssum = nl.ndarray((128, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.activation_reduce(dst=pexp, op=nl.exp, data=s, reduce_op=nl.add, reduce_res=ssum, bias=negm, scale=1.0)
            es = nl.ndarray((128, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.activation(dst=es, op=nl.exp, data=sinkb[0:128, hh:hh + 1], bias=negm, scale=1.0)
            tot = nl.ndarray((128, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_tensor(dst=tot, data1=ssum, data2=es, op=nl.add)
            rt = nl.ndarray((128, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.reciprocal(dst=rt, data=tot)
            prob = nl.ndarray((128, NK), dtype=nl.bfloat16, buffer=nl.sbuf)
            nisa.tensor_scalar(dst=prob, data=pexp, op0=nl.multiply, operand0=rt)
            acc_o = nl.ndarray((D, 128), dtype=nl.float32, buffer=nl.psum)
            for kb in range(NKB):
                ptp = nl.ndarray((128, 128), dtype=nl.bfloat16, buffer=nl.psum)
                nisa.nc_transpose(dst=ptp, data=prob[0:128, kb * 128:(kb + 1) * 128])
                pts = nl.ndarray((128, 128), dtype=nl.bfloat16, buffer=nl.sbuf)
                nisa.tensor_copy(dst=pts, src=ptp)
                nisa.nc_matmul(dst=acc_o, stationary=v_sb[0:128, kb, 0:D], moving=pts)
            nisa.tensor_copy(dst=ao[0:D, hh, qt * 128:(qt + 1) * 128], src=acc_o)

    # ---- o_proj + all-reduce + bias + residual
    part = nl.ndarray((P, HT, SC), dtype=nl.float32, buffer=nl.sbuf)
    for nt in range(HT):
        acc = nl.ndarray((P, SC), dtype=nl.float32, buffer=nl.psum)
        for hh in range(NQ):
            nisa.nc_matmul(dst=acc, stationary=wo_sb[0:D, (hh * HT + nt) * P:(hh * HT + nt + 1) * P], moving=ao[0:D, hh, 0:SC])
        nisa.tensor_copy(dst=part[0:P, nt, 0:SC], src=acc)
    red = _allreduce_fm(part, pid, xch, 0, rg, "pf_attn")
    h1 = nl.ndarray((P, HT, SC), dtype=nl.float32, buffer=nl.sbuf)
    h1_bf = nl.ndarray((P, HT, SC), dtype=nl.bfloat16, buffer=nl.sbuf)
    for kt in range(HT):
        at = nl.ndarray((P, SC), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=at, data=red[0:P, kt, 0:SC], op0=nl.add, operand0=bo_sb[0:P, kt:kt + 1])
        nisa.tensor_tensor(dst=h1_bf[0:P, kt, 0:SC], data1=x_bf[0:P, kt, 0:SC], data2=at, op=nl.add)
    nisa.tensor_copy(dst=h1, src=h1_bf)

    # ---- norm + router + top-4 per token
    xn2 = _rmsnorm_fm(h1, ln2_sb, ones_pp)
    lgp = nl.ndarray((E, SC), dtype=nl.float32, buffer=nl.psum)
    for kt in range(HT):
        nisa.nc_matmul(dst=lgp, stationary=wr_sb[0:P, kt * E:(kt + 1) * E], moving=xn2[0:P, kt, 0:SC])
    brc = nl.ndarray((E, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=brc, src=br.ap(pattern=[[1, E], [1, 1]], offset=0))
    lgb = nl.ndarray((E, SC), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=lgb, data=lgp, op0=nl.add, operand0=brc)
    lgf = nl.ndarray((E, SC), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=lgf, src=lgb)
    io = nl.ndarray((128, E), dtype=nl.float32, buffer=nl.sbuf)
    nisa.iota(dst=io, pattern=[[1, E]], offset=0, channel_multiplier=0)
    rev = nl.ndarray((128, E), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=rev, data=io, op0=nl.multiply, operand0=-1.0, op1=nl.add, operand1=64.0)
    for j in range(NQT):
        tp = nl.ndarray((128, E), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_transpose(dst=tp, data=lgf[0:E, j * 128:(j + 1) * 128])
        cur = nl.ndarray((128, E), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=cur, src=tp)
        vals = nl.ndarray((128, TOPK), dtype=nl.float32, buffer=nl.sbuf)
        r8 = nl.ndarray((128, 2 * TOPK), dtype=nl.float32, buffer=nl.sbuf)  # expert ids, routing weights
        hits = []
        for i in range(TOPK):
            mx = nl.ndarray((128, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_reduce(dst=mx, op=nl.maximum, data=cur, axis=1)
            nisa.tensor_copy(dst=vals[0:128, i:i + 1], src=mx)
            sc = nl.ndarray((128, E), dtype=nl.float32, buffer=nl.sbuf)
            nisa.scalar_tensor_tensor(dst=sc, data=cur, op0=nl.equal, operand0=mx, op1=nl.multiply, operand1=rev)
            sm = nl.ndarray((128, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_reduce(dst=sm, op=nl.maximum, data=sc, axis=1)
            nisa.tensor_scalar(dst=r8[0:128, i:i + 1], data=sm, op0=nl.multiply, operand0=-1.0, op1=nl.add,
                               operand1=64.0)
            hit = nl.ndarray((128, E), dtype=nl.float32, buffer=nl.sbuf)  # one-hot of the selected expert
            nisa.tensor_scalar(dst=hit, data=rev, op0=nl.equal, operand0=sm)
            hits.append(hit)
            nxt = nl.ndarray((128, E), dtype=nl.float32, buffer=nl.sbuf)
            nisa.scalar_tensor_tensor(dst=nxt, data=hit, op0=nl.multiply, operand0=NEG, op1=nl.add, operand1=cur)
            cur = nxt
        negv = nl.ndarray((128, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=negv, data=vals[0:128, 0:1], op0=nl.multiply, operand0=-1.0)
        ev = nl.ndarray((128, TOPK), dtype=nl.float32, buffer=nl.sbuf)
        esum = nl.ndarray((128, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.activation_reduce(dst=ev, op=nl.exp, data=vals, reduce_op=nl.add, reduce_res=esum, bias=negv, scale=1.0)
        rs = nl.ndarray((128, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.reciprocal(dst=rs, data=esum)
        wbf = nl.ndarray((128, TOPK), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=wbf, data=ev, op0=nl.multiply, operand0=rs)
        wf = nl.ndarray((128, TOPK), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=wf, src=wbf)
        dense = nl.ndarray((128, E), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=dense, data=hits[0], op0=nl.multiply, operand0=wf[0:128, 0:1])
        for i in range(1, TOPK):
            d2_ = nl.ndarray((128, E), dtype=nl.float32, buffer=nl.sbuf)
            nisa.scalar_tensor_tensor(dst=d2_, data=hits[i], op0=nl.multiply, operand0=wf[0:128, i:i + 1], op1=nl.add,
                                      operand1=dense)
            dense = d2_
        nisa.tensor_copy(dst=r8[0:128, TOPK:2 * TOPK], src=wf)
        nisa.dma_copy(dst=rti.ap(pattern=[[2 * TOPK, 128], [1, 2 * TOPK]], offset=j * 128 * 2 * TOPK, scalar_offset=preg,
                                 indirect_dim=0), src=r8)
        nisa.dma_copy(dst=rdn.ap(pattern=[[E, 128], [1, E]], offset=j * 128 * E, scalar_offset=preg, indirect_dim=0),
                      src=dense)
        # token-major copy of the normalized activations for the MoE gathers
        xr = nl.ndarray((128, H), dtype=nl.bfloat16, buffer=nl.sbuf)
        for kt in range(HT):
            xp = nl.ndarray((128, P), dtype=nl.bfloat16, buffer=nl.psum)
            nisa.nc_transpose(dst=xp, data=xn2[0:P, kt, j * 128:(j + 1) * 128])
            nisa.tensor_copy(dst=xr[0:128, kt * P:(kt + 1) * P], src=xp)
        nisa.dma_copy(dst=xtm.ap(pattern=[[H, 128], [1, H]], offset=j * 128 * H, scalar_offset=preg, indirect_dim=0),
                      src=xr)
    nisa.dma_copy(dst=hT.ap(pattern=[[MAXPOS, P], [P * MAXPOS, HT], [1, SC]], offset=0, scalar_offset=preg,
                            indirect_dim=1), src=h1_bf)
    dummy = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.shared_hbm)
    nisa.dma_copy(dst=dummy, src=p_sb)
    return dummy



PIPE = 1          # prefill_attn_multi: 1 = issue chunk c's attention before chunk c - 1's post-attention part
XCH_SENDRECV = True  # prefill_moe combine: exchange the cores' partials by sendrecv (else through xch in HBM)
SB = 2            # 128-row sub-blocks per iteration of prefill_moe's block loop
WENG = [nisa.engine.sync, nisa.engine.scalar]  # hardware DGE queues for the expert weights
OOBROW = 4194304.0  # an out-of-range row index: DMAs with oob_mode skip leave such rows alone
DCH = [(c * 512, min(512, H - c * 512)) for c in range((H + 511) // 512)]  # output column chunks of the down GEMM


def _moe_load_w(wgu, wd, bgut, gu_b, dn_b, bg_b, e, pid, wreg):
    """Expert e's weights (this physical core's slice) into weight buffer e % 2. wreg: None, or a register: the
    expert is e + wreg (out of range: every DMA is skipped)."""
    k = e % 2
    F = HT * P
    KC = HT // GU_T  # kt per DMA (gu_b [P, 2, HT, GU_T, P]: kt-major like the HBM layout)
    for t in range(GU_T):
        if wreg is None:
            src = wgu.ap(pattern=[[GU_F, P], [1, F]], offset=(e * 2 + pid) * P * GU_F + t * F)
        else:
            src = wgu.ap(pattern=[[GU_F, P], [1, F]], offset=(e * 2 + pid) * P * GU_F + t * F, scalar_offset=wreg,
                         indirect_dim=0)
        nisa.dma_copy(dst=gu_b[0:P, k, t * KC:(t + 1) * KC, 0:GU_T, 0:P], src=src, dge_mode=nisa.dge_mode.hwdge,
                      engine=WENG[t % 2], oob_mode=nisa.oob_mode.skip)
    for it in range(IT):
        if wreg is None:
            src = wd.ap(pattern=[[DN_F, P], [1, H]], offset=(e * 2 + pid) * P * DN_F + it * H)
        else:
            src = wd.ap(pattern=[[DN_F, P], [1, H]], offset=(e * 2 + pid) * P * DN_F + it * H, scalar_offset=wreg,
                        indirect_dim=0)
        nisa.dma_copy(dst=dn_b[0:P, k, it * H:(it + 1) * H], src=src, dge_mode=nisa.dge_mode.hwdge,
                      engine=WENG[(it + 1) % 2], oob_mode=nisa.oob_mode.skip)
    # gate/up bias row (t, n), broadcast over 128 partitions
    if wreg is None:
        src = bgut.ap(pattern=[[0, 128], [1, GU_T * P]], offset=(e * 2 + pid) * P * GU_T)
    else:
        src = bgut.ap(pattern=[[0, 128], [1, GU_T * P]], offset=(e * 2 + pid) * P * GU_T, scalar_offset=wreg,
                      indirect_dim=0)
    nisa.dma_copy(dst=bg_b[0:128, k, 0:GU_T, 0:P], src=src, oob_mode=nisa.oob_mode.skip)


def _moe_gather(rec, xtm, row0, b, rr):
    """rr = rec rows [row0 + b, + 128) (b: loop register or None); returns the listed tokens' activations [128, H]
    (token-major; rows marked out of range are skipped and stay stale)."""
    if b is None:
        nisa.dma_copy(dst=rr, src=rec.ap(pattern=[[4, 128], [1, 4]], offset=row0 * 4))
    else:
        nisa.dma_copy(dst=rr, src=rec.ap(pattern=[[4, 128], [1, 4]], offset=row0 * 4, scalar_offset=b, indirect_dim=0),
                      oob_mode=nisa.oob_mode.skip)
    ti = nl.ndarray((128, 1), dtype=nl.int32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=ti, src=rr[0:128, 0:1])
    xg = nl.ndarray((128, H), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.dma_copy(dst=xg, src=xtm.ap(pattern=[[H, 128], [1, H]], offset=0, vector_offset=ti, indirect_dim=0),
                  oob_mode=nisa.oob_mode.skip)
    return xg


def _moe_transpose(xg, xT):
    """xT [P, HT, 128] = xg^T (feature-major)."""
    for kt in range(HT):
        xp = nl.ndarray((P, 128), dtype=nl.bfloat16, buffer=nl.psum)
        nisa.nc_transpose(dst=xp, data=xg[0:128, kt * P:(kt + 1) * P])
        nisa.tensor_copy(dst=xT[0:P, kt, 0:128], src=xp)


def _moe_scatter(ybuf, yoff, yi, yt):
    """ybuf[yoff / H + yi[p]] = yt[p] (rows with an out-of-range index are skipped)."""
    nisa.dma_copy(dst=ybuf.ap(pattern=[[H, 128], [1, H]], offset=yoff, vector_offset=yi, indirect_dim=0), src=yt,
                  oob_mode=nisa.oob_mode.skip)


def _moe_stage2(rr, xT, gu_b, dn_b, bg_b, k, yt, yi):
    """One block of 128 listed tokens of the expert in weight buffer k: gate/up, SwiGLU, x routing weight, down;
    yt [128, H] bf16 = the output rows, yi = their ybuf rows (rr[:, 1])."""
    nisa.tensor_copy(dst=yi, src=rr[0:128, 1:2])
    # gate/up, token-major: g[tok, (t, n)] = sum_kt xT[:, kt, tok]^T gu[t, kt]
    acc_a = nl.ndarray((128, 4, P), dtype=nl.float32, buffer=nl.psum)  # t = 0..3
    acc_b = nl.ndarray((128, 2, P), dtype=nl.float32, buffer=nl.psum)  # t = 4, 5
    for kt in range(HT):
        nisa.nc_matmul(dst=acc_a, stationary=xT[0:P, kt, 0:128], moving=gu_b[0:P, k, kt, 0:4, 0:P])
        nisa.nc_matmul(dst=acc_b, stationary=xT[0:P, kt, 0:128], moving=gu_b[0:P, k, kt, 4:6, 0:P])
    g_ = nl.ndarray((128, GU_T, P), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=g_[0:128, 0:4, 0:P], data1=acc_a, data2=bg_b[0:128, k, 0:4, 0:P], op=nl.add)
    nisa.tensor_tensor(dst=g_[0:128, 4:6, 0:P], data1=acc_b, data2=bg_b[0:128, k, 4:6, 0:P], op=nl.add)
    gate = nl.ndarray((128, IT, P), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=gate, data=g_[0:128, 0:IT, 0:P], op0=nl.minimum, operand0=SWIGLU_LIMIT)
    up = nl.ndarray((128, IT, P), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=up, data=g_[0:128, IT:GU_T, 0:P], op0=nl.minimum, operand0=SWIGLU_LIMIT,
                       op1=nl.maximum, operand1=-SWIGLU_LIMIT)
    ga = nl.ndarray((128, IT, P), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=ga, data=gate, op0=nl.multiply, operand0=ALPHA)
    sg = nl.ndarray((128, IT, P), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.activation(dst=sg, op=nl.sigmoid, data=ga)
    glu = nl.ndarray((128, IT, P), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=glu, data1=gate, data2=sg, op=nl.multiply)
    up1 = nl.ndarray((128, IT, P), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=up1, data=up, op0=nl.add, operand0=1.0)
    itr = nl.ndarray((128, IT, P), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=itr, data1=up1, data2=glu, op=nl.multiply)
    itw = nl.ndarray((128, IT, P), dtype=nl.bfloat16, buffer=nl.sbuf)  # x routing weight (per token)
    nisa.tensor_scalar(dst=itw, data=itr, op0=nl.multiply, operand0=rr[0:128, 2:3])
    interT = nl.ndarray((P, IT, 128), dtype=nl.bfloat16, buffer=nl.sbuf)
    for it in range(IT):
        ip = nl.ndarray((P, 128), dtype=nl.bfloat16, buffer=nl.psum)
        nisa.nc_transpose(dst=ip, data=itw[0:128, it, 0:P])
        nisa.tensor_copy(dst=interT[0:P, it, 0:128], src=ip)
    # down, token-major: y[tok, n] = sum_it interT[:, it, tok]^T dn[it] (it outer: one stationary load per it)
    accs = []
    for c in range(len(DCH)):
        acc = nl.ndarray((128, 512), dtype=nl.float32, buffer=nl.psum)
        accs.append(acc)
    for it in range(IT):
        for c in range(len(DCH)):
            c0 = DCH[c][0]
            cw = DCH[c][1]
            nisa.nc_matmul(dst=accs[c][0:128, 0:cw], stationary=interT[0:P, it, 0:128],
                           moving=dn_b[0:P, k, it * H + c0:it * H + c0 + cw])
    for c in range(len(DCH)):
        c0 = DCH[c][0]
        cw = DCH[c][1]
        nisa.tensor_copy(dst=yt[0:128, c0:c0 + cw], src=accs[c][0:128, 0:cw])


def _moe_slot_sum(ybuf, pid, j, MAXPOS):
    """This core's partial MoE output of tile j: the sum of its 4 slot rows per token, [128, H] fp32."""
    yl = []
    for i in range(TOPK):
        y_ = nl.ndarray((128, H), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.dma_copy(dst=y_, src=ybuf.ap(pattern=[[H, 128], [1, H]], offset=((pid * 4 + i) * MAXPOS + j * 128) * H))
        yl.append(y_)
    y01 = nl.ndarray((128, H), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=y01, data1=yl[0], data2=yl[1], op=nl.add)
    y23 = nl.ndarray((128, H), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=y23, data1=yl[2], data2=yl[3], op=nl.add, engine=nisa.engine.gpsimd)
    ysum = nl.ndarray((128, H), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=ysum, data1=y01, data2=y23, op=nl.add)
    return ysum


@nki.jit
def prefill_moe(hT, plen, xtm, rti, rdn, wgu, bgut, wd, bd, rec, blist, ybuf, xch, rg, NT):
    """Sparse MoE of one layer for the prompt positions [0, NT * 128) (tokens >= plen[0, 0] are padding and are
    routed nowhere). Inputs from prefill_attn: hT[:, t] = h1, xtm, rti, rdn. hT[:, t] = h2 = h1 + moe (in place).

    1. route: per expert e, the list of (token t, slot row i * MAXPOS + t, routing weight) of the tokens that
       selected e as their i-th expert, in token order: rec[e * CAP + k] (CAP = MAXPOS); unused rows are marked
       out of range.
    2. experts: for every expert, a device loop over blocks of 128 listed tokens: gather the tokens' activations,
       this vcore's gate/up slice and SwiGLU (token-major: the activations are the stationary operand and the weights
       stream), x routing weight, down slice (token-major output), scatter the bf16-rounded
       partial rows to ybuf[pid][i * MAXPOS + t] (every token fills its 4 slot rows exactly once).
    3. combine: moe partial = sum of the 4 slot rows (slot order), all-reduce over the 8 vcores, + bias term
       sum_e w_e b_e, h2 = bf16(h1 + bf16(moe)).
    bgut [E, 2 * GU_T * P] fp32: gate/up bias per physical core in (t, n) order.
    rec [E * (NT + SB) * 128, 4] fp32, blist [4 * NT + 2, 4] int32, ybuf [2 * 4 * MAXPOS, H] bf16 (one half per physical core), xch [2, 2, 128, 2 * H] fp32: caller-provided scratch."""
    pid = nl.program_id(0)
    MAXPOS = hT.shape[1]
    CAP = (NT + SB) * 128  # rows per expert in rec (the loop's blocks of SB x 128 rows may end past the list)
    NTOK = NT * 128
    # ---- constants
    plb_i = nl.ndarray((128, 1), dtype=nl.int32, buffer=nl.sbuf)
    nisa.dma_copy(dst=plb_i, src=plen.ap(pattern=[[0, 128], [1, 1]], offset=0))
    plb = nl.ndarray((128, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=plb, src=plb_i)
    piota = nl.ndarray((128, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.iota(dst=piota, pattern=[[0, 1]], offset=0, channel_multiplier=1)
    io_e = nl.ndarray((128, E), dtype=nl.float32, buffer=nl.sbuf)
    nisa.iota(dst=io_e, pattern=[[1, E]], offset=0, channel_multiplier=0)
    lt_d = nl.ndarray((128, 128), dtype=nl.float32, buffer=nl.sbuf)  # column - row
    nisa.iota(dst=lt_d, pattern=[[1, 128]], offset=0, channel_multiplier=-1)
    lstrict = nl.ndarray((128, 128), dtype=nl.bfloat16, buffer=nl.sbuf)  # [t', t] = 1 if t' < t
    nisa.tensor_scalar(dst=lstrict, data=lt_d, op0=nl.greater, operand0=0.0)
    ones_b = nl.ndarray((128, 128), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.memset(dst=ones_b, value=1.0)
    ones_1p = _ones((1, P))

    # ---- expert 0's weights first: they do not depend on the routing and stream while it runs
    gu_b = nl.ndarray((P, 2, HT, GU_T, P), dtype=WDT, buffer=nl.sbuf)
    dn_b = nl.ndarray((P, 2, DN_F), dtype=WDT, buffer=nl.sbuf)
    bg_b = nl.ndarray((128, 2, GU_T, P), dtype=nl.float32, buffer=nl.sbuf)  # gate/up bias row, broadcast
    _moe_load_w(wgu, wd, bgut, gu_b, dn_b, bg_b, 0, pid, None)

    # ---- 1. route
    RI = E * CAP // 128  # rec rows per partition of the init
    inv = nl.ndarray((128, RI, 4), dtype=nl.float32, buffer=nl.sbuf)  # (OOB, OOB, 0, 0) rows
    nisa.memset(dst=inv, value=0.0)
    nisa.memset(dst=inv[0:128, 0:RI, 0:2], value=OOBROW)
    nisa.dma_copy(dst=rec.ap(pattern=[[RI * 4, 128], [1, RI * 4]], offset=0), src=inv)
    carry = nl.ndarray((128, E), dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(dst=carry, value=0.0)
    for j in range(NT):
        r8 = nl.ndarray((128, 2 * TOPK), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=r8, src=rti.ap(pattern=[[2 * TOPK, 128], [1, 2 * TOPK]], offset=j * 128 * 2 * TOPK))
        tokf = nl.ndarray((128, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=tokf, data=piota, op0=nl.add, operand0=float(j * 128))
        valid = nl.ndarray((128, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=valid, data1=tokf, data2=plb, op=nl.less)
        ohs = []
        osum = nl.ndarray((128, E), dtype=nl.float32, buffer=nl.sbuf)
        for i in range(TOPK):
            oh = nl.ndarray((128, E), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_scalar(dst=oh, data=io_e, op0=nl.equal, operand0=r8[0:128, i:i + 1], op1=nl.multiply,
                               operand1=valid)
            ohs.append(oh)
            if i == 0:
                nisa.tensor_copy(dst=osum, src=oh)
            else:
                nisa.tensor_tensor(dst=osum, data1=osum, data2=oh, op=nl.add)
        o_bf = nl.ndarray((128, E), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.tensor_copy(dst=o_bf, src=osum)
        cum_ps = nl.ndarray((128, E), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_matmul(dst=cum_ps, stationary=lstrict, moving=o_bf)
        tot_ps = nl.ndarray((128, E), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_matmul(dst=tot_ps, stationary=ones_b, moving=o_bf)
        cum = nl.ndarray((128, E), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=cum, data1=cum_ps, data2=carry, op=nl.add)
        for i in range(TOPK):
            pr = nl.ndarray((128, E), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_tensor(dst=pr, data1=ohs[i], data2=cum, op=nl.multiply)
            rank = nl.ndarray((128, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_reduce(dst=rank, op=nl.add, data=pr, axis=1)
            d1 = nl.ndarray((128, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.scalar_tensor_tensor(dst=d1, data=r8[0:128, i:i + 1], op0=nl.multiply, operand0=float(CAP),
                                      op1=nl.add, operand1=rank)
            d2 = nl.ndarray((128, 1), dtype=nl.float32, buffer=nl.sbuf)  # valid ? d1 : OOBROW
            nisa.tensor_scalar(dst=d2, data=d1, op0=nl.add, operand0=-OOBROW, op1=nl.multiply, operand1=valid)
            d3 = nl.ndarray((128, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_scalar(dst=d3, data=d2, op0=nl.add, operand0=OOBROW)
            dst_i = nl.ndarray((128, 1), dtype=nl.int32, buffer=nl.sbuf)
            nisa.tensor_copy(dst=dst_i, src=d3)
            recv = nl.ndarray((128, 4), dtype=nl.float32, buffer=nl.sbuf)
            nisa.memset(dst=recv, value=0.0)
            nisa.tensor_copy(dst=recv[0:128, 0:1], src=tokf)
            nisa.tensor_scalar(dst=recv[0:128, 1:2], data=tokf, op0=nl.add, operand0=float(i * MAXPOS))
            nisa.tensor_copy(dst=recv[0:128, 2:3], src=r8[0:128, TOPK + i:TOPK + i + 1])
            nisa.dma_copy(dst=rec.ap(pattern=[[4, 128], [1, 4]], offset=0, vector_offset=dst_i, indirect_dim=0),
                          src=recv, oob_mode=nisa.oob_mode.skip)
        nc = nl.ndarray((128, E), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=nc, data1=carry, data2=tot_ps, op=nl.add)
        carry = nc
    cnt = nl.ndarray((1, E), dtype=nl.int32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=cnt, src=carry[0:1, 0:E])
    # blocks after the first of every expert: blist[j] = (weight-load expert or E (skip), first rec row, 0, 0), in
    # expert order; nbl = their number. Experts on partitions: n [E, 1] = carry row 0 transposed.
    ncol_p = nl.ndarray((E, 1), dtype=nl.float32, buffer=nl.psum)
    nisa.nc_transpose(dst=ncol_p, data=carry[0:1, 0:E])
    ncol = nl.ndarray((E, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=ncol, src=ncol_p)
    NKB = (NT + SB - 1) // SB
    kio = nl.ndarray((E, NKB), dtype=nl.float32, buffer=nl.sbuf)  # first row 128 + SB * 128 * k of extra block k
    nisa.iota(dst=kio, pattern=[[SB * 128, NKB]], offset=128, channel_multiplier=0)
    xb = nl.ndarray((E, NKB), dtype=nl.float32, buffer=nl.sbuf)  # extra block k of expert e exists
    nisa.tensor_scalar(dst=xb, data=kio, op0=nl.less, operand0=ncol)
    nx = nl.ndarray((E, 1), dtype=nl.float32, buffer=nl.sbuf)  # extra blocks of expert e
    nisa.tensor_reduce(dst=nx, op=nl.add, data=xb, axis=1)
    nx_b = nl.ndarray((E, 1), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.tensor_copy(dst=nx_b, src=nx)
    xoff_p = nl.ndarray((E, 1), dtype=nl.float32, buffer=nl.psum)  # extra blocks of the experts before e
    nisa.nc_matmul(dst=xoff_p, stationary=lstrict[0:E, 0:E], moving=nx_b)
    xoff = nl.ndarray((E, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=xoff, src=xoff_p)
    nbl_p = nl.ndarray((1, 1), dtype=nl.float32, buffer=nl.psum)
    nisa.nc_matmul(dst=nbl_p, stationary=ones_b[0:E, 0:1], moving=nx_b)
    BMAX = 4 * NT + 2  # >= 4 * NT // SB + 32 + 1
    binv = nl.ndarray((1, BMAX, 4), dtype=nl.int32, buffer=nl.sbuf)
    nisa.memset(dst=binv, value=0)
    nisa.memset(dst=binv[0:1, 0:BMAX, 0:1], value=E)
    nisa.memset(dst=binv[0:1, 0:BMAX, 1:2], value=int(OOBROW))
    nisa.dma_copy(dst=blist.ap(pattern=[[BMAX * 4, 1], [1, BMAX * 4]], offset=0), src=binv)
    eio = nl.ndarray((E, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.iota(dst=eio, pattern=[[0, 1]], offset=0, channel_multiplier=1)
    for kk in range(NKB):
        br_ = nl.ndarray((E, 4), dtype=nl.float32, buffer=nl.sbuf)
        nisa.memset(dst=br_, value=0.0)
        if kk == 0:
            nisa.tensor_copy(dst=br_[0:E, 0:1], src=eio)
        else:
            nisa.memset(dst=br_[0:E, 0:1], value=float(E))
        nisa.tensor_scalar(dst=br_[0:E, 1:2], data=eio, op0=nl.multiply, operand0=float(CAP), op1=nl.add,
                           operand1=float(128 + SB * 128 * kk))
        br_i = nl.ndarray((E, 4), dtype=nl.int32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=br_i, src=br_)
        bd_ = nl.ndarray((E, 1), dtype=nl.float32, buffer=nl.sbuf)  # exists ? xoff + kk : out of range
        nisa.tensor_scalar(dst=bd_, data=xoff, op0=nl.add, operand0=float(kk) - OOBROW, op1=nl.multiply,
                           operand1=xb[0:E, kk:kk + 1])
        nisa.tensor_scalar(dst=bd_, data=bd_, op0=nl.add, operand0=OOBROW)
        bd_i = nl.ndarray((E, 1), dtype=nl.int32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=bd_i, src=bd_)
        nisa.dma_copy(dst=blist.ap(pattern=[[4, E], [1, 4]], offset=0, vector_offset=bd_i, indirect_dim=0), src=br_i,
                      oob_mode=nisa.oob_mode.skip)
    nbl = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=nbl, src=nbl_p)
    nbreg = nisa.register_alloc()
    nisa.register_load(nbreg, nbl)

    # ---- 2. experts (weights double-buffered: expert e + 1 loads while expert e computes; the block loop is
    # software-pipelined: the gather + transposes of block b + 1 run beside the GEMMs of block b)
    F = HT * P
    # 2a. static: the first block (128 listed tokens) of every expert; weights double-buffered (no device loop, so
    # expert e + 1's weights stream while expert e computes); expert 0's weights were issued before the routing
    for e in range(E):
        if e + 1 < E:
            _moe_load_w(wgu, wd, bgut, gu_b, dn_b, bg_b, e + 1, pid, None)
        rr0 = nl.ndarray((128, 4), dtype=nl.float32, buffer=nl.sbuf)
        xg0 = _moe_gather(rec, xtm, e * CAP, None, rr0)
        xT0 = nl.ndarray((P, HT, 128), dtype=nl.bfloat16, buffer=nl.sbuf)
        _moe_transpose(xg0, xT0)
        yt0 = nl.ndarray((128, H), dtype=nl.bfloat16, buffer=nl.sbuf)
        yi0 = nl.ndarray((128, 1), dtype=nl.int32, buffer=nl.sbuf)
        _moe_stage2(rr0, xT0, gu_b, dn_b, bg_b, e % 2, yt0, yi0)
        _moe_scatter(ybuf, pid * 4 * MAXPOS * H, yi0, yt0)
    # 2b. device loop over the remaining blocks (blist from step 1; an expert's weights load on its first listed
    # block), software-pipelined: the gather + transposes of block j + 1 beside the GEMMs of block j, scatter one late
    hr0 = nl.ndarray((1, 4), dtype=nl.int32, buffer=nl.sbuf)
    nisa.dma_copy(dst=hr0, src=blist.ap(pattern=[[4, 1], [1, 4]], offset=0))
    rw0 = nisa.register_alloc()
    nisa.register_load(rw0, hr0[0:1, 1:2])
    rrs, xTs, yts, yis = [], [], [], []
    for sb in range(SB):
        rr_ = nl.ndarray((128, 4), dtype=nl.float32, buffer=nl.sbuf)
        nisa.memset(dst=rr_, value=OOBROW)
        xT_ = nl.ndarray((P, HT, 128), dtype=nl.bfloat16, buffer=nl.sbuf)
        _moe_transpose(_moe_gather(rec, xtm, sb * 128, rw0, rr_), xT_)
        yt_ = nl.ndarray((128, H), dtype=nl.bfloat16, buffer=nl.sbuf)
        yi_ = nl.ndarray((128, 1), dtype=nl.int32, buffer=nl.sbuf)
        nisa.memset(dst=yi_, value=-1)
        rrs.append(rr_)
        xTs.append(xT_)
        yts.append(yt_)
        yis.append(yi_)
    for j in nl.dynamic_range(0, nbreg, 1):
        hr = nl.ndarray((1, 8), dtype=nl.int32, buffer=nl.sbuf)  # block j, block j + 1
        nisa.dma_copy(dst=hr, src=blist.ap(pattern=[[8, 1], [1, 8]], offset=0, scalar_offset=j, indirect_dim=0))
        wreg = nisa.register_alloc()
        nisa.register_load(wreg, hr[0:1, 0:1])
        rnreg = nisa.register_alloc()
        nisa.register_load(rnreg, hr[0:1, 5:6])
        _moe_load_w(wgu, wd, bgut, gu_b, dn_b, bg_b, 0, pid, wreg)
        xgn, rrn = [], []
        for sb in range(SB):
            _moe_scatter(ybuf, pid * 4 * MAXPOS * H, yis[sb], yts[sb])
            rr_n = nl.ndarray((128, 4), dtype=nl.float32, buffer=nl.sbuf)
            nisa.memset(dst=rr_n, value=OOBROW)
            xgn.append(_moe_gather(rec, xtm, sb * 128, rnreg, rr_n))
            rrn.append(rr_n)
        for sb in range(SB):
            _moe_stage2(rrs[sb], xTs[sb], gu_b, dn_b, bg_b, 0, yts[sb], yis[sb])
        for sb in range(SB):
            _moe_transpose(xgn[sb], xTs[sb])
            nisa.tensor_copy(dst=rrs[sb], src=rrn[sb])
    for sb in range(SB):
        _moe_scatter(ybuf, pid * 4 * MAXPOS * H, yis[sb], yts[sb])

    # ---- 3. combine. Each physical core sums its 4 slot rows per token (its partial) -> xch[pid]; core pid then
    # finishes the tiles j with j % 2 == pid: + the other core's partial -> src; one all-reduce over the 4 ranks;
    # + bias term, h2 = h1 + moe -> hT.
    bd_sb = nl.ndarray((E, H), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=bd_sb, src=bd)
    src = nl.ndarray((NTOK, H), dtype=nl.float32, buffer=nl.shared_hbm, name="pm_src")
    dst = nl.ndarray((NTOK, H), dtype=nl.float32, buffer=nl.shared_hbm, name="pm_dst")
    if XCH_SENDRECV:
        # tile pairs: this core sums its slot rows of tiles 2 jh and 2 jh + 1, sends its partial of the tile the other
        # core finishes and receives the other core's partial of its own tile (no xch round trip through HBM)
        for jh in range(NT // 2):
            # core pid finishes tile 2 jh + pid and sends its partial of tile 2 jh + 1 - pid
            give = _moe_slot_sum(ybuf, pid, 2 * jh + 1 - pid, MAXPOS)
            mine = _moe_slot_sum(ybuf, pid, 2 * jh + pid, MAXPOS)
            other = nl.ndarray((128, H), dtype=nl.float32, buffer=nl.sbuf)
            nisa.sendrecv(src=give, dst=other, send_to_rank=1 - pid, recv_from_rank=1 - pid, pipe_id=0,
                          dma_engine=nisa.dma_engine.dma)
            both = nl.ndarray((128, H), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_tensor(dst=both, data1=mine, data2=other, op=nl.add)
            nisa.dma_copy(dst=src.ap(pattern=[[H, 128], [1, H]], offset=(2 * jh + pid) * 128 * H), src=both)
    else:
        for j in range(NT):
            ysum = _moe_slot_sum(ybuf, pid, j, MAXPOS)
            nisa.dma_copy(dst=xch.ap(pattern=[[H, 128], [1, H]], offset=(pid * NTOK + j * 128) * H), src=ysum)
        nisa.core_barrier(data=xch, cores=(0, 1))
        for jh in range(NT // 2):
            j = 2 * jh + pid
            mine = nl.ndarray((128, H), dtype=nl.float32, buffer=nl.sbuf)
            nisa.dma_copy(dst=mine, src=xch.ap(pattern=[[H, 128], [1, H]], offset=(pid * NTOK + j * 128) * H))
            other = nl.ndarray((128, H), dtype=nl.float32, buffer=nl.sbuf)
            nisa.dma_copy(dst=other, src=xch.ap(pattern=[[H, 128], [1, H]], offset=((1 - pid) * NTOK + j * 128) * H))
            both = nl.ndarray((128, H), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_tensor(dst=both, data1=mine, data2=other, op=nl.add)
            nisa.dma_copy(dst=src.ap(pattern=[[H, 128], [1, H]], offset=j * 128 * H), src=both)
    nisa.core_barrier(data=src, cores=(0, 1))
    ncc.all_reduce(srcs=[src], dsts=[dst], op=nl.add, replica_group=rg)
    for jh in range(NT // 2):
        j = 2 * jh + pid
        red = nl.ndarray((128, H), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=red, src=dst.ap(pattern=[[H, 128], [1, H]], offset=j * 128 * H))
        rd = nl.ndarray((128, E), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=rd, src=rdn.ap(pattern=[[E, 128], [1, E]], offset=j * 128 * E))
        rtp = nl.ndarray((E, 128), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_transpose(dst=rtp, data=rd)
        rtf = nl.ndarray((E, 128), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=rtf, src=rtp)
        h1 = nl.ndarray((P, HT, 128), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.dma_copy(dst=h1, src=hT.ap(pattern=[[MAXPOS, P], [P * MAXPOS, HT], [1, 128]], offset=j * 128))
        # token-major: mo = bf16(red + sum_e w_e b_e); then feature-major (bf16 transposes), h2 = h1 + mo
        mo = nl.ndarray((128, H), dtype=nl.bfloat16, buffer=nl.sbuf)
        for c in range(len(DCH)):
            c0 = DCH[c][0]
            cw = DCH[c][1]
            bp = nl.ndarray((128, 512), dtype=nl.float32, buffer=nl.psum)
            nisa.nc_matmul(dst=bp[0:128, 0:cw], stationary=rtf, moving=bd_sb[0:E, c0:c0 + cw])
            nisa.tensor_tensor(dst=mo[0:128, c0:c0 + cw], data1=red[0:128, c0:c0 + cw], data2=bp[0:128, 0:cw], op=nl.add)
        h2 = nl.ndarray((P, HT, 128), dtype=nl.bfloat16, buffer=nl.sbuf)
        for nt in range(HT):
            mp = nl.ndarray((P, 128), dtype=nl.bfloat16, buffer=nl.psum)
            nisa.nc_transpose(dst=mp, data=mo[0:128, nt * P:(nt + 1) * P])
            nisa.tensor_tensor(dst=h2[0:P, nt, 0:128], data1=h1[0:P, nt, 0:128], data2=mp, op=nl.add)
        nisa.dma_copy(dst=hT.ap(pattern=[[MAXPOS, P], [P * MAXPOS, HT], [1, 128]], offset=j * 128), src=h2)
    dummy = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.shared_hbm)
    nisa.dma_copy(dst=dummy, src=plb_i[0:1, 0:1])
    return dummy
