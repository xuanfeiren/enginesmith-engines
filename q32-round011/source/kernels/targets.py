"""The NEFFs of the Qwen3-32B engine (tensor parallel 8) and how each is traced: zero arrays of the per-rank shapes
(only shapes matter for a NEFF). The shapes here are the contract with the host (src/engine.cpp)."""
import numpy as np

# Qwen3-32B
L, HID, HT, NQ, NKV, D, INTER, VOCAB = 64, 5120, 40, 64, 8, 128, 25600, 151936
W = 8                       # tensor-parallel ranks (logical NeuronCores)
C = 2 * W                   # virtual cores (2 physical cores per rank)
# 4 q heads per virtual core; 8 kv heads < 16 virtual cores: kv head = rank, replicated on both physical cores;
# 200 intermediate tiles zero-padded to 208 (13 per virtual core)
NQL, NKL, ITL = NQ // C, max(1, NKV // C), -(-(INTER // 128) // C)      # 4, 1, 13
VT = -(-VOCAB // 128)       # 1187
NVL = -(-VT // C)           # 297
TMAX = 4608                 # positions in the KV cache (longest request 4,096 + 65)

STEP_BUCKETS = [512, 1152, 2048, 4224, 4608]   # timed decode: 256+65 / 1024+65 / 4096+65 positions; 4608 = the reference's length
STEP_SL = 2          # weight slabs of 2 output tiles (decode 64 tokens at b512: 1427.7 -> 1420.3 ms, tokens identical)
# round 003: gate/up slabs prefetched before the o_proj all-reduce (was STEP_PF = 3): with 13 (all of gate/up; the
# compiler reuses buffers as SBUF fills) the weight stream keeps going through attention, o_proj and the all-reduce.
# decode 64 tokens b512: 951 (3) / 935 (5) / 923 (7) / 917 (9) / 915 (11) / 914 ms (13); b4224 979 -> 939 (with CCQ)
STEP_PFG = 13
# round 003: the write of each all-reduce input buffer on GpSimd's software DGE, off the Sync engine's in-order weight
# queue (it waits on the layer's dependency chain): b512 914 -> 894 ms (also moving the read-back: 920; the K/V cache
# writes too: 894; read-back + write + K/V: 907; write + read-back on the Scalar engine's hwdge queue: 928)
STEP_CCQ = 1
# round 004: the kv head of a virtual core is the rank's (replicated on both physical cores): core 0 projects k, core
# 1 projects v, exchanged by sendrecv (1.3 MB less weight stream per layer and core)
STEP_KVS = 0   # measured: decode b512 893.5 -> 900.5 ms, p4096 prefill 441 -> 473 ms (tokens identical): off
# round 004: all-reduce read-back with static DMA descriptors (off the Sync weight queue): b512 893.4 -> 889.1 ms,
# b4224 912.4 -> 907.5; + the next layer's first 10 o_proj slabs prefetched before the down all-reduce: 885.5 / 906.6
STEP_RBQ = 2
STEP_PFO = 10
# round 011 (MPD 4 + MPL 3): decode MLP over exactly 1600 intermediate rows per virtual core (no zero-padded tiles):
# gate/up = the prefill MP copy wgp (24 own tiles + one packed [gate half | up half] tile, its slab first), down = wdh
# (src/engine.cpp layout_d_combo): 12 own k-tiles for every output tile + one combo k-tile (rows 0-63 this core's 64
# half rows, 64-127 the other core's, whose act half arrives by sendrecv) for output tiles [20 pid, 20 pid + 20).
# 3.8% fewer MLP bytes; decode 64 tokens b512 885.3 -> 870.6 ms (NOTES round 011). Not bitwise (down sum order).
STEP_MPD = dict(MPD=4, MPL=3)
# with MPD the gate/up prefetch depth before the o_proj all-reduce is re-tuned per bucket (decode 64 tokens):
# b512 PFG 13 / 11 / 10 / 9: 870.4 / 864.7 / 871.9 / 872.0; b1152 13 / 11: 862.0 / 863.7; b4224 13 / 11 / 9 / 7:
# 896.7 / 885.0 / 882.7 / 887.1 ms. b2048 / b4608 (not timed) take the neighbours' values
STEP_PFG_MPD = {512: 11, 1152: 13, 2048: 11, 4224: 9, 4608: 9}
STEP_PF = 3          # slabs of the next GEMV issued before each all-reduce (QKV has only 3 slabs of 2 tiles at 32B TP=8; 14B used 6)

# V tiles per DMA of the decode step's V-cache read (permuted layout, step_tp.VPERM). Round 002 (decode 64 tokens):
# b1152 941 (1) / 951 (4) / 955 ms (all); b4224 1035 (1) / 979 (4) / 984 ms (all). Round 003, with the deeper gate/up
# prefetch: b512 894 (1) / 896 (4 = all); b1152 911 (1) / 897 (3) / 895 ms (9 = all); b4224 933 (4) / 913 ms (33 = all)
STEP_VCH = {512: 1, 1152: 9, 2048: 16, 4224: 33, 4608: 36}

STEP_SPEC = {
    "pos": ((1, 1), "i32"), "plen": ((1, 1), "i32"), "prompt": ((TMAX, 1), "i32"), "toks": ((TMAX + 1, 1), "i32"),
    "embed": ((VOCAB, HID), "bf16"),
    "ln1": ((L, 128, HT), "f32"), "ln2": ((L, 128, HT), "f32"), "qn": ((L, 128, 1), "f32"), "kn": ((L, 128, 1), "f32"),
    "wqkv": ((L, 2, 128, (NQL + 2 * NKL) * HT * 128), "bf16"), "wo": ((L, 2, 128, HT * NQL * 128), "bf16"),
    "wgu": ((L, 2, 128, 2 * ITL * HT * 128), "bf16"), "wd": ((L, 2, 128, HT * ITL * 128), "bf16"),
    # round 011 (MPD): the decode step's unpadded MLP weights: wgp = the prefill MP gate/up copy (src/engine.cpp
    # layout_gu_mp), wdh = a decode-only down copy in the HALF layout (layout_d_half)
    "wgp": ((L, 2, 128, (2 * (ITL - 1) + 1) * HT * 128), "bf16"), "wdh": ((L, 2, 128, HT * ITL * 128), "bf16"),
    "normf": ((128, HT), "f32"), "wlm": ((2, 128, NVL * HT * 128), "bf16"), "rot": ((128, 128), "bf16"),
    "cos": ((TMAX, 64), "f32"), "sin": ((TMAX, 64), "f32"),
    "kcache": ((L, 2, NKL, 128, TMAX), "bf16"), "vcache": ((L, 2, TMAX, NKL * 128), "bf16"), "rkb": ((128, 1), "f32"),
}
STEP_ORDER = ["pos", "plen", "prompt", "toks", "embed", "ln1", "ln2", "qn", "kn", "wqkv", "wo", "wgu", "wd", "wgp", "wdh", "normf",
              "wlm", "rot", "cos", "sin", "kcache", "vcache", "rkb"]

PREFILL_BUCKETS = [256, 512, 1024, 2048, 4096]
PF_OVL = 0          # prefill layer: 1 / 2 = combine each block right after its partial / software-pipelined (no gain)
# prefill layer (round 002, see NOTES): bf16 HBM-buffer all-reduce per 512-token block (CCHBM, CCBF; the round-001 fp32
# 128-token SBUF messages ran at ~8 GB/s across the two devices), o_proj / down N-split over the two physical cores
# written straight into the all-reduce buffer (NSP), MLP token blocks of 512 (TEB). p4096 1031 -> 441 ms.
PF_CC = dict(CCBF=1, CCN=512, CCHBM=1, NSP=1, TEB=512)
# round 004: FN = attention all-reduce finish fused with norm2 per MLP block (no h write / barrier / read round trip);
# PMF = each MLP block's all-reduce finished one block later instead of all of them in the layer's tail
# measured slower (p4096 441 -> 468 / 473 ms with / without PMF, p1024 116.6 -> 121): off
PF_CC.update(FN=0, PMF=0)
# round 004 (HALF): intermediate rows split exactly 1600 per virtual core (12 tiles + a 64-row half tile) instead
# of 13 tiles with 8 zero tiles at the end (weights re-laid out by src/engine.cpp layout_gu_half / layout_d_half)
HALF = 0   # measured: decode b512 893.5 -> 894-897 ms (HBM read -3%, layer period unchanged), p1024 116.7 -> 120.9: off
PF_CC.update(HALF=HALF)
import os as _os, json as _json
STEP_X = {}
if _os.environ.get("STEP_OVERRIDE"):   # experiments only: '{"PQP": 1}' merged into every step target
    STEP_X = _json.loads(_os.environ["STEP_OVERRIDE"])
if _os.environ.get("PF_CC_OVERRIDE"):   # experiments only: e.g. '{"CCBF": 0, "CCN": 128}' (part of the target, so of the cache key)
    PF_CC = _json.loads(_os.environ["PF_CC_OVERRIDE"])
STAGES = {1: "emb", 2: "layer", 3: "head"}


# round 004: o_proj block b interleaved with MLP block b - ILV in one loop (instead of all o_proj blocks, then all MLP
# blocks): p4096 441.4 -> 437.1 (ILV 2) / 430.7 (3) / 438.9 ms (4), 448 (1); p2048 224.8 -> 229.0 (2) / 224.8 (3)
PF_ILV = {4096: 3}
# round 004: LPN layers per execution of the prefill layer NEFF (the layer body unrolled LPN times), with the first
# PFG gate/up slab pairs prefetched before the attention all-reduce and (PFQ) the next layer's QKV slabs before the
# MLP all-reduce, so the weight stream runs through the all-reduces and across layers. {S: (LPN, PFG, PFQ)}
PF_GROUP = {256: (16, 0, 1)}   # p256: 41.1 -> 37.9 ms with PF_BUCKET[256] (LPN 8: 38.7-42, 32: 38.0, 64: 40.2; 16 with
# PFG 2: 39.0); p1024 LPN 8 / 16: 119-121 vs 116.6, p4096 LPN 4: 463.8 vs 430.3 (not used there)
# per-bucket prefill switches merged over PF_CC (round 004): p256 FN 41.1 -> 40.4, KVS -> 40.6, both -> 40.2 ms;
# at S=512 both were slower (66.7 -> 68.5)
PF_BUCKET = {256: dict(FN=1, KVS=1)}
# round 005 (PN): the next MLP block's attention finish + norm2 issued during this block (after its gate/up), so the
# in-order TensorE does not wait at each block transition for the norm's sum-of-squares matmul. PN=2: the norm's
# TensorE part interleaved with the down loop. p4096 430.4 -> 416.4 (PN 1) / 423.4 ms (PN 2); p1024 116.9 -> 116.0 /
# 113.7; p2048 225.4 -> 221.4 / 220.6; tokens identical
PF_BUCKET[1024] = dict(PN=2)
# FN0 (round 005): the first MLP block's finish fused with norm2 (FN for block 0 only): p4096 416.6 -> 413.6,
# p2048 220.8 -> 217.0, p1024 113.8 -> 115.9 ms (not used there); tokens identical
PF_BUCKET[2048] = dict(PN=2, FN0=1)
PF_BUCKET[4096] = dict(PN=1, FN0=1)
# round 006: HS (S=256) = the residual kept in SBUF for the 16 layers of an execution (no h HBM round trips, no h
# core barriers), HSC / NCH = the norms' residual add / h load + square + sum-of-squares matmuls pipelined over
# chunks of hidden tiles. p256 38.1 -> 36.3 (HS) -> 33.7 ms (+ HSC 8; 4: 34.0, 10: 33.6, 20: 33.8); p1024 113.7 -> 110.4
# (NCH 8; 4: 110.9, 10: 110.3, 20: 110.4); p4096 413.3 -> 394.5 (NCH 20; 5: 402, 8: 398.5, 10: 400.5); tokens identical
PF_BUCKET[256].update(HS=1, HSC=8)
PF_BUCKET[512] = dict(NCH=8)
PF_BUCKET[1024].update(NCH=8)
PF_BUCKET[2048].update(NCH=8)
PF_BUCKET[4096].update(NCH=20)
# round 007 (SP, prefill_sp.py): sequence-parallel residual. The block all-reduces become reduce-scatters over token
# shards (64 tokens per rank, 32 per physical core; bitwise equal to the all-reduce's sums) + all-gathers of the
# normalised activations; the norms and residual finishes run on the own 32 tokens only, LPN layers per execution
# with the next layer's norm1 + all-gather issued during this layer's MLP blocks. Tokens identical.
# p4096 394.7 -> 375.8 ms, p1024 110.2 -> 101.0 ms (NOTES round 007 for the switches)
# later in round 007: LPN 16 at 1024 (8: 101.05, 32: 100.2 ms), LPN 8 at 4096 (4: 374.4, 16: 371.6 ms but a 12-min
# compile); SPP (down of MLP block b - 1 after block b's gate/up: 4096 376.0 -> 374.4, 1024 101.0 -> 105.6) and SPQ
# (QKV tile post-processing one tile behind its matmuls: 4096 372.9 -> 370.0, 1024 100.5 -> 101.4) at 4096 only
PF_GROUP[1024] = (16, 0, 0)
PF_GROUP[4096] = (8, 0, 0)
PF_BUCKET[1024] = dict(SP=1, SPL=1, SPO=1, SPN=1, SPA=1, SPD=1, SPT=1)
PF_BUCKET[4096] = dict(SP=1, SPL=2, SPO=2, SPN=1, SPA=2, SPD=1, SPT=2, SPP=1, SPQ=1)
# round 008: the SP collective chains trimmed, bitwise identical (prefill_sp.py): GCH = the all-gathered block loaded in
# 4 DMAs (the first matmuls start on the first chunk), SOC = o_proj / down outputs written to the reduce-scatter
# buffer in 4 chunks during the loop (not one 2.6 MB write after it), SPS = every reduce-scatter / all-gather split in
# 2 parts over hidden tiles (the first part's collective runs while the second is computed; consumers start on the
# first part), SPG = the first 3 gate / up slab pairs of each MLP block issued before its norm2 loads (which wait on the
# reduce-scatter in the in-order Sync queue). p1024 100.4 -> 97.7 ms, p4096 370.3 -> 353.9 ms (NOTES round 008)
PF_BUCKET[1024].update(GCH=4, SOC=4, SPS=2, SPG=3)
PF_BUCKET[4096].update(GCH=4, SOC=4, SPS=2, SPG=3)
# round 009 (MP): the SP buckets read unpadded prefill copies of the MLP weights (wgp / wdp, src/engine.cpp
# layout_gu_mp / layout_d_mp): per core 12 gate + 12 up slabs + one packed [gate half | up half] slab (25 instead of 26
# with the zero tile), down over 12 own + 12 other tiles + one combo tile of both cores' 64-row halves (25 instead of
# 26). 3.8% fewer MLP matmuls and weight bytes. p1024 97.9 -> 94.9 ms, p4096 353.7 -> 346.0 ms (SPG 2: SPG 3 no longer
# fits SBUF; SPG 1: 409 ms). Tokens identical on the test prompts (not bitwise: the down sums run in another order).
# p256 (prefill_tp path, MP there too): 33.7 -> 33.7 ms and one stream diverged at token 20: off
PF_BUCKET[1024].update(MP=1)
PF_BUCKET[4096].update(MP=1, SPG=2)
if _os.environ.get("PF_BUCKET_OVERRIDE"):   # experiments only: '{"256": {"HS": 1}}' merged into PF_BUCKET
    for _k, _v in _json.loads(_os.environ["PF_BUCKET_OVERRIDE"]).items():
        PF_BUCKET.setdefault(int(_k), {}).update(_v)
if _os.environ.get("PF_GROUP_OVERRIDE"):   # experiments only: '{"256": [8, 4, 1]}'
    PF_GROUP = {int(k): tuple(v) for k, v in _json.loads(_os.environ["PF_GROUP_OVERRIDE"]).items()}


def prefill_spec(S, st=2):
    """per-layer tensors have a leading layer dim of LPN: the host binds them to slices of the stacked tensors"""
    sp = dict(STEP_SPEC)
    sp.pop("wdh", None)   # decode-only (round 011 MPD)
    lpn = PF_GROUP.get(S, (1, 0, 0))[0] if st == 2 else 1
    for n in ["ln1", "ln2", "qn", "kn", "wqkv", "wo", "wgu", "wd", "kcache", "vcache"]:
        sp[n] = ((lpn,) + sp[n][0][1:], sp[n][1])
    sp.update({"ids": ((S, 1), "i32"), "cosT": ((128, TMAX), "f32"), "sinT": ((128, TMAX), "f32"),
               "s_h": ((128, HT, S), "f32"), "s_q": ((2 * NQL, 128, S), "bf16"), "s_k": ((2 * NKL, 128, S), "bf16"),
               "s_v": ((2 * NKL, S, 128), "bf16"), "s_o": ((2 * NQL, 128, S), "bf16"),
               "s_p": ((2, 128, HT, S), "f32"),
               # round 009 (MP): unpadded prefill MLP copies (src/engine.cpp layout_gu_mp / layout_d_mp)
               "wgp": ((lpn, 2, 128, (2 * (ITL - 1) + 1) * HT * 128), "bf16"), "wdp": ((lpn, 2, 128, HT * ITL * 128), "bf16")})
    return sp


PREFILL_ORDER = ["ids", "plen", "toks", "embed", "ln1", "ln2", "qn", "kn", "wqkv", "wo", "wgu", "wd", "wgp", "wdp", "normf", "wlm",
                 "rot", "cosT", "sinT", "kcache", "vcache", "rkb", "s_h", "s_q", "s_k", "s_v", "s_o", "s_p"]

STEP_DEPS = ["step_tp.py", "layer_decode.py", "step_decode.py"]
TARGETS = [dict(name=f"step_b{tb}", kind="step", TB=tb, VCH=STEP_VCH[tb], SL=STEP_SL, PF=STEP_PF, PFG=STEP_PFG, CCQ=STEP_CCQ, KVS=STEP_KVS, RBQ=STEP_RBQ, PFO=STEP_PFO, cost=10 + tb // 512, deps=STEP_DEPS,
                spec=sorted((k, list(v[0]), v[1]) for k, v in STEP_SPEC.items())) for tb in STEP_BUCKETS]
TARGETS += [dict(name=f"pf{S}_{STAGES[st]}", kind="prefill", S=S, STAGE=st, cost=(S // 256 if st == 2 else 1),
                 deps=STEP_DEPS + ["prefill_tp.py", "prefill_sp.py"],
                 spec=sorted((k, list(v[0]), v[1]) for k, v in prefill_spec(S, st).items()))
            for S in PREFILL_BUCKETS for st in (1, 2, 3)]
for t in TARGETS:
    if t["kind"] == "step":
        t["HALF"] = HALF
        t.update(STEP_MPD)
        t["PFG"] = STEP_PFG_MPD[t["TB"]]
        t.update(STEP_X)
    if t["kind"] == "prefill" and t["STAGE"] == 2:
        t["OVL"] = PF_OVL
        t["KVS"] = STEP_KVS
        t["ILV"] = PF_ILV.get(t["S"], 0)
        g = PF_GROUP.get(t["S"], (1, 0, 0))
        t["LPN"], t["PFG"], t["PFQ"] = g
        t.update(PF_CC)
        t.update(PF_BUCKET.get(t["S"], {}))
    if t["kind"] == "prefill" and t["STAGE"] == 3 and PF_BUCKET.get(t["S"], {}).get("SP", 0):
        t["SP"] = 1   # round 007: the head of an SP bucket takes token plen - 1 from its owner rank


def _zeros(spec, order):
    import ml_dtypes
    DT = {"bf16": ml_dtypes.bfloat16, "f32": np.float32, "i32": np.int32}
    return [np.zeros(spec[n][0], DT[spec[n][1]]) for n in order]


def run(t):
    from nki.collectives import ReplicaGroup
    if t["kind"] == "step":
        import step_tp
        per = [_zeros(STEP_SPEC, STEP_ORDER) for _ in range(W)]
        args = [[per[r][i] for r in range(W)] for i in range(len(STEP_ORDER))]
        step_tp.step_tp[2](*args, rg=ReplicaGroup([list(range(W))]), TB=t["TB"], SL=t.get("SL", 1), hw=t.get("hw", 2),
                           NQL=NQL, NKL=NKL, ITL=ITL, W=W, PF=t.get("PF", 4), HBMCC=t.get("HBMCC", 1), VCH=t.get("VCH", 1),
                           PFG=t.get("PFG", -1), CCQ=t.get("CCQ", 0), KVS=t.get("KVS", 0),
                           PQP=t.get("PQP", 0), PQA=t.get("PQA", 0), HALF=t.get("HALF", 0),
                           RBQ=t.get("RBQ", 0), KVQ=t.get("KVQ", 0), PFO=t.get("PFO", 0),
                           SLO=t.get("SLO", 0), DSP=t.get("DSP", 1),
                           CCQM=t.get("CCQM", -1), RBQM=t.get("RBQM", -1), RING=t.get("RING", 0), RCP=t.get("RCP", 0), BCP=t.get("BCP", 0),
                           MPD=t.get("MPD", 0), MPL=t.get("MPL", 0), DN3=t.get("DN3", 0), MPX=t.get("MPX", 0), MPC=t.get("MPC", 0))
    elif t["kind"] == "prefill":
        import prefill_tp
        sp = prefill_spec(t["S"], t["STAGE"])
        per = [_zeros(sp, PREFILL_ORDER) for _ in range(W)]
        for r in range(W):
            per[r][PREFILL_ORDER.index("plen")][:] = 1   # the head reads h at plen - 1: keep the build run in bounds
        args = [[per[r][i] for r in range(W)] for i in range(len(PREFILL_ORDER))]
        prefill_tp.prefill_tp[2](*args, STAGE=t["STAGE"], NQL=NQL, NKL=NKL, ITL=ITL, W=W,
                                 rg=ReplicaGroup([list(range(W))]), OVL=t.get("OVL", 0),
                                 TEB=t.get("TEB", 256), CCN=t.get("CCN", 128), CCBF=t.get("CCBF", 0),
                                 CCHBM=t.get("CCHBM", 0), NSP=t.get("NSP", 0), SB=t.get("SB", 0),
                                 KVS=t.get("KVS", 0), LPN=t.get("LPN", 1), PFG=t.get("PFG", 0), PFQ=t.get("PFQ", 0),
                                 FN=t.get("FN", 0), PMF=t.get("PMF", 0), HALF=t.get("HALF", 0),
                                 SQL=t.get("SQL", 0), ILV=t.get("ILV", 0),
                                 QW=t.get("QW", 0), QNS=t.get("QNS", 0), AQL=t.get("AQL", 0),
                                 NSP2=t.get("NSP2", 0), PN=t.get("PN", 0), PNA=t.get("PNA", 0), XN=t.get("XN", 0), FN0=t.get("FN0", 0),
                                 HS=t.get("HS", 0), HSC=t.get("HSC", 1), NCH=t.get("NCH", 1),
                                 NCQL=t.get("NCQL", 0), NCH2=t.get("NCH2", 0),
                                 SP=t.get("SP", 0), SPL=t.get("SPL", -1), SPN=t.get("SPN", 0), SPT=t.get("SPT", 0), SPO=t.get("SPO", 0), SPA=t.get("SPA", 0), SPD=t.get("SPD", 0), SPH=t.get("SPH", 0), SPHN=t.get("SPHN", 0), SPP=t.get("SPP", 0), SPQ=t.get("SPQ", 0), GCH=t.get("GCH", 1), SOC=t.get("SOC", 1), SPS=t.get("SPS", 1), SPG=t.get("SPG", 0), KTP=t.get("KTP", 0), KTQ=t.get("KTQ", 0), MP=t.get("MP", 0))
    else:
        raise ValueError(t["kind"])
