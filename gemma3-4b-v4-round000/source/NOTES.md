# Gemma-3-4B-it engine on Trainium2 (one logical NeuronCore, LNC=2)

## Round 011 (2026-11-09): rounds 007, 009 and 010 stacked onto round 008

Rounds 007, 009 and 010 each measured real, bitwise-identical gains in ab.py but were rejected on evaluation
noise (the CPU reference's times vary by up to 50% between rounds). None of them was in the round-008 engine. This
round re-applies all three on top of round 008 (git history: branch r011, cherry-picks of 09782ca / c11be50, the
round-010 diff 8567c25..8f73bcc, the round-007 diff 142fbb3..d52c1bd with conflicts resolved):
- round 009: single-stream `gemm_rs` / `gemm_gu_split` (other core's tokens first), two-stream `split_b` for
  global buckets <= 3072;
- round 010: 4-step decode NEFFs (`dgA4x{B}`), host runs 4 / 2 / 1-step NEFFs;
- round 007: attention skips fully masked 128-key blocks (`ATT_SKIP`), hl33 folded into the last prefill chunk
  (`...KH` NEFFs, `FOLD`), two-stream reduce-scatter add on the GpSimd (`ENG.rs_add`), only used NEFFs built.
  Conflict resolution: round 007's decode local window `BLd` = min(1024, B) is round 008's `LB` (same value at
  every bucket, kept `LB`); round 008's post-norm multiply in `ar_piece` follows the GpSimd add.

Verified: same.py round-008 engine (core 22) vs candidate (core 23): 64/64 identical (40 standard prompts x 3
request shapes + 24 random prompts at chunk / bucket boundaries 128 .. 8120 tokens, session lab/extra.jsonl).
Measured (ab.py, 8 rounds each, tokens identical), speedup over round 008:
| order | prefill 256 / 1024 / 4096 | decode 256 / 1024 / 4096 |
|---|---|---|
| r008 core 22, candidate core 23 | 1.0513 / 1.0241 / 1.0171 | 1.0030 / 1.0033 / 1.0043 |
| r008 core 23, candidate core 22 | 1.0464 / 1.0227 / 1.0160 | 1.0002 / 0.9995 / 1.0007 |
Absolute (candidate): prefill 15.1 / 52.6 / 213.4 ms, decode (64) 766-767 / 768-770 / 782-784 ms.
Expected score effect ~ +1.6% (prefill mean ~ +2.9%, decode ~ +0.2%); p256 prefill ~ +4.9% (> the 4% case rule).
dev-eval 001 (final engine, 2 rounds): PASSED, score 16.49 (prefill 25.57, decode 10.64), mismatching steps
1 / 1 / 0 per round; candidate prefill 15.0-15.1 / 52.5 / 213.2 ms, decode 766 / 768 / 782 ms (the CPU reference's
p4096 prefill ran 7.4 s in one round vs 12.0 s in the other, so the score again mostly reflects the reference).
Build: clean build of all used NEFFs ~25 min (48 jobs, another build running alongside).

Tried and rejected this round:
- Two-stream `rs_split_groups` add on the GpSimd (`ENG.rs_add`, as round 007 did for `ar_piece`) together with
  `RS_SPLIT_B_MAXBG` 3072 -> 4096 (to see whether round 009's slowdown at BG >= 3584 was the VectorE queueing):
  ab.py vs this round's engine, 8 rounds x 2 core orders: prefill 1024 1.0007 / 1.0000, 4096 0.9940 / 0.9925;
  decode within +-0.25%. Not the cause; not kept.

## Round 008 (2026-11-08): 2-step decode NEFFs for every bucket, post norms off the critical path, short local windows

All three changes bitwise identical to the round-005 engine: same.py 40/40 (lengths 1..4097, 3 request shapes)
after the first, 52/52 after the second and after the third (the 40 + 12 random prompts of 1535 .. 8120 tokens, so
decode runs in every global bucket 512 .. 8192). Long-output check (200-token prompt + 7993 tokens to the
8192-position limit, then a refused continuation; 3001-token prompt split / chained vs single): identical to the
round-005 engine.

Changes:
- **2-step decode NEFFs for all global buckets** (`dgA2x{B}`, `DEC_MULTI_MAXB` / host `DEC2_MAXB` 8192), with the
  persistent SBUF buffer for the global K/V window only up to B = 1536 in the multi-step body
  (`DEC_MULTI_PERSIST_MAX`). Round 005 found dgA2x4608 15% slower than two dgA4608 steps; the cause was that buffer
  (72 KB per partition at B = 4608). Device time (neuron-explorer, random inputs): dgA4608 13.12 ms per step,
  dgA2x4608 with the persistent buffer 30.34 ms per 2 steps, without it 25.48 ms (12.74 per step, -2.9%);
  dgA2x2048 with or without the buffer: 25.48 ms (same). The 4096 case decodes at bucket 4608 (positions
  4096 .. 4160), so it now runs 32 two-step NEFFs instead of 64 single ones.
- **Post norms (post-attention, post-feedforward): x * (1 + w) per reduce-scatter piece** (`NORM_SPLIT`,
  `ar_piece`): as each o_proj / down_proj reduce-scatter piece lands, the VectorE computes ob * (1 + w) in fp32
  for its two k-tiles (during the GEMM); after the statistics only `* rstd` and the residual add remain, batched
  5 k-tiles per instruction (`RES_GRP`) instead of 20 fused `scalar_tensor_tensor` + 20 adds. Bitwise: the fused
  instruction computes (x * w) * rstd in fp32 with one final rounding; tensor_scalar (fp32 out) followed by
  tensor_tensor gives the same bits (device check on random data, also equal to numpy). Microbenchmark
  ([128, 20 x 128] per core): 20 stt 5.68 us vs 4 batched multiplies 3.32 us; 20 bf16 adds 3.12 us vs 1 add 1.93 us.
  Device time: pgA256_256K 13.34 -> 13.13 ms (-1.6%), p2s512A1280_4096K 27.94 -> 27.88 ms (-0.2%: the two-stream
  NEFFs already hide most chain time).
- **Decode at global bucket 512: local windows of 512 keys** (`DEC_LOCAL_SMALL`, `LB` in dec_group_body /
  dec_multi_body): every position below 512 fits in a 512-key window starting at 0, so the 29 local layers read
  half the K/V (round 005 read 1024 keys, the upper half masked). dgA2x512 25.10 -> 24.94 ms, dgA512 12.58 -> 12.53.

Measured (ab.py, 8 rounds, round-005 engine on core 22, candidate on core 23; round 005 measured core 23 ~0.5%
slower in prefill in an A/A run, so prefill gains are slightly understated):
| step | prefill speedup 256 / 1024 / 4096 | decode speedup 256 / 1024 / 4096 |
|---|---|---|
| post norms | 1.0160 / 1.0097 / 1.0043 | 1.0016 / 1.0012 / 1.0047 |
| + 2-step decode at every bucket | 1.0133 / 1.0095 / 1.0046 | 1.0022 / 1.0019 / **1.0222** |
| + 512-key local windows at bucket 512 (final) | 1.0145 / 1.0097 / 1.0045 | **1.0083** / 1.0015 / **1.0216** |
Absolute (final candidate): prefill 15.8 / 53.8 / 216.9 ms, decode (64) 767.4 / 769.7 / 784.4 ms.
Build time: 15 min (10 before; 6 more 2-step decode NEFFs).

Profile of the 256-token prefill NEFF (pgA256_256K, round-005 engine, neuron-explorer json; tools were session
scratch): 13.34 ms, TensorE active 87%, ScalarE 21%, VectorE 23%, 6.5 GB read (487 GB/s). Per layer (~397 us):
qkv GEMM + q/k norms 36 us, attention 18 us, o GEMM 17.5 us, **o chain 31 us idle** (reduce-scatter tail ~12 us +
norms ~19 us), gate/up 178 us, down 87 us, **down chain 28 us idle** (tail ~9 us + norms ~18 us). Reduce-scatter
pieces are one 128 KB DMA descriptor each on DMA engine 15 (queue qSyncSpillReload0), which the weight stream also
uses (10-31 GB/s per piece under load, 190 GB/s when the weight stream is idle); after the GEMM they land on a ~2.3 us
cadence (per-sendrecv cross-core handshake, pipe 0 serializes them), so 10 pieces outlast the 17.5 us o GEMM.
The hl33 NEFF (layer 33 for the last token + LM head) is DMA-bound: 2.45 ms, 1.54 GB at 627 GB/s.

Tried and rejected this round (device time, neuron-explorer):
- **Two 128-token streams for prompts <= 256** (pre2s with T2 = 256; gram_rstd generalized to 64-token blocks):
  pgA 13.34 -> 15.64 ms (+17%). A microbenchmark shows a 128-column matmul costs half a 256-column one (57 vs 116 ns
  per instruction, distinct stationaries; the round-002 note "128 columns: 98.5 ns" does not hold), so the PE rate
  is not the problem: the chains of the two streams are not hidden (the ring limits the lag to < 16 slabs, ~14 us of
  128-column work) and their count doubles (Gram -> Gram idle 1.35 ms vs 0.71 ms).
- x * (1 + w) on the ScalarE for all four norms (activation copy with per-partition scale, fp32 out; bitwise equal):
  pgA 13.34 -> 13.51 ms, p2s 27.94 -> 28.18 ms: ScalarE needs ~0.6 us per [128, 128] tile (ACT active 27% -> 39%),
  which puts it on the critical path. The same on the VectorE for the input / pre-feedforward norms would sit in
  front of the statistics (in-order engine), so only the post norms use it.
- **o_proj split by output features** (single stream): attention output all-gathered head by head during
  attention (bf16, 512 KB per core), each core computing its 10 output n-tiles from both cores' heads in two PSUM
  accumulators (bitwise the same partial sums and fp32 add), 64 KB bf16 pieces exchanged for the token split (vs
  1.3 MB fp32 reduce-scatter); needs a second o_proj layout on the host (wo2). pgA 13.125 -> 13.138 ms (no gain):
  the exchange pieces still land 5-10 us after the GEMM (same ~2.3 us per-sendrecv cadence), and the GEMM waits
  ~4 us for the last attention head.
- Fence that holds the next weight-ring load until the last reduce-scatter piece has landed (a VectorE write into
  the next ring slot, WAW): pgA -0.15% (noise level): the burst does not compete much with the tail; the per-piece
  handshake does.
- DMA bandwidth microbenchmark with 16 KB-per-partition slabs: 166 GB/s per core on the Sync HWDGE queue, 177 with
  DGE none, no better split over the Sync and ACT queues; worse than the engine's own 4 KB-descriptor weight stream
  (~320 GB/s per core in decode), so not pursued.

Next steps (estimates): the cross-core exchange is the remaining lever for prefill (~15% of the 256-token NEFF is
TensorE idle in the two chains per layer, half of it reduce-scatter tail); nothing found that shortens the
per-sendrecv handshake. Small items: fold hl33 into the final chunk NEFF (one boundary, ~0.25% p256); 4-step
decode NEFFs (~0.2% decode, 16 boundaries per 64 tokens).


## Round 005 (2026-11-08): fewer NEFF boundaries, layer 33 only where it is needed

All changes bitwise identical to the round-003 engine: same.py (40 prompts x 3 shapes, lengths 1..4097) 40/40
after every step, the last one on the final engine.

Changes:
- **Layer 33 computes only K/V in prefill** (`pre_layer_kv` in pre2s.py: input norm, the k/v n-tiles 8..15 of the
  qkv GEMM, k norms, RoPE, V transpose, cache writes; same instructions per tile as the full layer). The hidden
  state after layer 33 is only needed for the prompt's last token. That token goes through layer 33 in `hl33_{256,512}`
  = `dec_group_body` (decode arithmetic, T=1) over the layer-33 weights reading column `col` of the prefill hidden
  state, + final norm + LM head + argmax (replaces head_pre/head_pre512; same DMA cost as the head plus one
  DMA-bound decode layer). Tokens identical on all same.py prompts (the decode-arithmetic layer 33 for the first
  token did not change any token there).
- **Prefill: one NEFF per 512-token chunk** (`p2s512A{BL}_{BG}K`, `pgA256_256K`: embedding + 34 layers, every
  global bucket) instead of embed + 3 (or 6) group NEFFs. The round-002 note that a 12-layer NEFF was 4.7% slower
  than two 6-layer ones at BG 4096 did not hold for the 34-layer NEFF (+0.7% at 4096 vs the 6-group path).
- **Decode: one NEFF per step** (`dgA{B}`: embedding, 34 layers, LM head, argmax) instead of 3, and **two steps per
  NEFF** (`dgA2x{B}`, `dec_multi_body`; step 2 embeds the token step 1 chose from SBUF, both cores hold it) while the
  global bucket is <= 1536 (`DEC2_MAXB`).

Measured (ab.py, 6 rounds, base = round-003 engine on core 22, candidate on core 23; A/A of the round-003 engine
22 vs 23: prefill 0.9925 / 0.9962 / 0.9995, decode 1.0014 / 1.0012 / 1.0029, i.e. core 23 ~0.5% slower in prefill):
| step | prefill speedup 256 / 1024 / 4096 | decode speedup 256 / 1024 / 4096 |
|---|---|---|
| layer 33 KV-only, non-final chunks | 0.999 / 1.013 / 1.024 | 1.001 / 1.001 / 1.003 |
| + decode 1 NEFF per step | 0.993 / 1.010 / 1.023 | 1.016 / 1.019 / 1.010 |
| + hl33 (all chunks KV-only) | 1.005 / 1.020 / 1.025 | 1.015 / 1.019 / 1.010 |
| + 34-layer chunk NEFF (BG <= 2048) | 1.009 / 1.029 / 1.031 (+0.7% at 4096 for all buckets) | same |
| + embedding inside the chunk NEFF | 1.018 / 1.032 / 1.041 | 1.016 / 1.020 / 1.010 |
| 2-step decode (same binary, vs 1-step) | - | 1.005 / 1.005 / (1-step there) |
Absolute (candidate): prefill 16.0 / 54.3 / 217.6 ms, decode (64) ~772 / 770 / 797 ms.
dev-eval 001 (final engine, 2 rounds): PASSED, score 15.23 (prefill 22.24, decode 10.43; the CPU reference ran
faster than in round 003's evaluation), mismatching steps 1/1/0 per round. Candidate times there: prefill
16.0 / 54.2 / 217.5 ms, decode 772 / 770 / 797 ms (round-003 evaluation: 16.2 / 55.9 / 226.2, 786 / 787 / 803).
A NEFF boundary costs ~40 us in prefill and ~90 us in decode (measured from these steps).

Tried and rejected:
- 2-step decode NEFFs at larger global buckets: 0% at 2048 / 3072, **-15%** at 4608 (942 vs 801 ms per 64 tokens;
  not investigated, probably SBUF / the persistent global K/V buffer in the doubled program).
- fp8 LM-head screening with exact rescoring: not built, ruled out by the skill (any low-precision weight copy).

Next steps (estimates): fold hl33 into the final chunk NEFF (one boundary, ~0.25% p256; needs the last token's
column exchanged between the cores); find why 2-step decode is slow at B 4608 (+0.5% decode at 4096).


## Round 003 (2026-11-08): prefill chains (all bitwise identical to round 002)

Profiling the prefill group NEFFs (neuron-explorer json: per-core TensorE timeline, chains split by pc order; tools
were session scratch, not kept) showed ~28% of every prefill layer was TensorE idle between the GEMMs (256-token
group pgW256_256: 428.7 us per layer, 305 us of GEMM): the norm / all-reduce / q-k-norm chains were serial, bound by
VectorE + ScalarE latency, and ran redundantly on both cores.

Changes (each verified bitwise against the round-002 engine with same.py; final: 40/40 prompts x 3 shapes, plus
5000/6500/8100-token prompts 3/3):
- **Token split of the residual stream across the two cores** (`AR(..., th)` -> reduce-scatter in `ar_piece`,
  `rmsnorm_ag`): each core keeps the residual, norms and adds of its own 128 tokens of each 256-token stream; the
  o/down fp32 partial sums are reduce-scattered over tokens (1.3 MB instead of a 2.6 MB all-reduce, same fp32 add)
  and norm outputs that feed a GEMM are all-gathered (bf16, pieces of AG_GRP = 4 k-tiles). NEFF input/output
  columns are per core accordingly. pgW256_256 5186 -> 5035 us, p2s512M1280_4096 5488 -> 5283 us.
- Norm statistics chain (`gram_rstd(..., identm)`): sqrt(ss/n + eps) as one activation with scale/bias, diag(1/r)
  on the VectorE (`DR_ENGINE`). Attention: -max/16 and PV * 1/sum on the VectorE (`ATT_ENG_*`).
- q/k norms inside the qkv GEMM (`qkv_gemm_norm` single stream, `qk_hook` on the two-stream job): each head's Gram +
  statistics (`gram_rstd_a`) right after its two tiles are evacuated, the broadcast (`gram_rstd_b`), scaling and
  RoPE one GEMM group later, so the TensorE never waits for a chain.
- Post norms: scaling and residual add per k-tile (`rmsnorm_res`), so the next Gram follows tile by tile.
- Single-stream attention software pipeline depth 4 (`ATT_DEPTH1`; two streams keep 2: depth 3 was +0.6% there).
- Device time: pgW256_256 5186 -> 4880 us (-5.9%), p2s512M1280_4096 5488 -> 5178 us (-5.7%).

Measured (ab.py, 8 rounds, cores 22/23, vs the round-002 engine; tokens identical):
| case | prefill ms r002 -> r003 | speedup | decode ms r002 -> r003 |
|---|---|---|---|
| 256 | 17.2 -> 16.2 | +6.35% | 787.7 -> 785.7 (+0.25%) |
| 1024 | 59.2 -> 55.9 | +5.96% | 789.4 -> 787.2 (+0.28%) |
| 4096 | 239.7 -> 226.4 | +5.89% | 806.3 -> 803.1 (+0.40%) |
dev-eval 002 (final engine): PASSED, score 17.37 (prefill 29.29, decode 10.31), mismatching steps 1/1/0
(dev-eval 001, before the last two changes: 16.92). The CPU reference's times vary by up to 50% between runs.
Reference judge (own script, 24 random prompts 1..4096 tokens x 65): 12 / 1560 mismatching steps, worst stream 2.

Where the time is now (device profile): single stream 395-400 us per layer = 309 us GEMM + ~90 us of chains, of
which ~15 us per chain is the reduce-scatter tail after the o and down GEMMs. A sendrecv piece completes ~13 us after
its group is evacuated (GpSimd handshake waits for the previous piece's DMA, which shares the DMA engines with the
weight stream); hidden during the GEMM, exposed at its end. Two streams: GEMMs 78% of TensorE time, attention 13%,
Gram norms 8.5%; TensorE ~91% busy. Decode unchanged (at the HBM streaming floor).

Tried and rejected this round (device time of pgW256_256 unless noted; [baseline of the moment]):
- sendrecv pieces over 2 / 4 pipe_ids (all-reduce and later reduce-scatter, also all-gather): 4814-4828 [4808],
  5180 [5186]; all-reduce pieces of 4 n-tiles: 5234 [5180]; reduce-scatter pieces of 4 n-tiles (o or down): 4884 /
  4892 [4880]; half of each piece on the GpSimd SB-to-SB DMA (`gpsimd_dma`): 6026; whole reduce-scatter pieces on
  `gpsimd_dma` (1 KB/partition): 5746 [4873].
- Exchange through shared HBM + `core_barrier` instead of sendrecv: 5695 (Sync queue) / 5503 (default queue) [4808].
- Emitting each all-reduce piece 2 / 5 slab loads later (Sync-queue order): 4885 / 4864 [4873]; weight ring 6 / 8 /
  16 slots: 4911 / 4884 / 4895 [4880].
- select_reduce (max, off-diagonal masked) for the Gram diagonal: 5092-5109 [5037]; rstd elementwise on the
  broadcast [128,128] tile (sqrt from PSUM + reciprocal): 5031 [4984]; diag(1/r) on GpSimd: 5186 [5014].
- q/k norms per KV group interleaved with attention (`QK_SPLIT`, 3 placements): 5014-5086 [4984].
- Two-stream lags (LAG_D 8, LAG_QKV 8, LAG_O 13, LAG_GU 13): 5243-5306 [5247] (p2s M). Evacuation engine of the
  two-stream qkv / o / down jobs on VectorE or compiler's choice: 5181-5222 [5178] (p2s M).
- AG_GRP 2 / 5 / 10: 4889 / 4885 / 4949 [4887].
- **Deferred norms** (`DEFER_NORM`, kept in the source, off): the input and pre-feedforward norms feed the GEMM
  with bf16(x * (1 + w)) and apply 1/rms at the evacuation (v tiles, gate/up; q/k need none, their norm is scale
  invariant), statistics broadcast lazily (`DeferRS`). pgW 4873 -> 4808 (-1.35%), p2s M 5178 -> 5137 (-0.8%), but
  not bitwise: judge 14 / 1560 mismatches (worst stream 5, margins up to 2.4) vs 12 / 1560 (worst 2) for the bitwise
  engine. Not worth ~0.5% of score.
- Findings: head_pre512 (prefill LM head) runs at the DMA floor (2.19 ms for 1.35 GB); `select_reduce` supports only
  max; `tensor_scalar` has no divide; NKI source cannot read attributes of list elements in expressions
  (`it[0].T`: assign to a name first).

Next steps (estimates): skipping the fully masked 128-key block per query tile in local layers with a 1280 window
(~0.5% on 4096 prompts); fewer decode NEFFs per token (3 -> 1: ~0.5% decode); a lower-latency cross-core exchange
would be worth ~7% of prefill (the reduce-scatter tails), none found in the public API.

## Structure (round 002; round 003 changes above)

- `kernels/gemma.py`: NKI kernels (public `nki.language` / `nki.isa` only).
  - **Decode**: `dec_group_body` runs several layers of one decode step in one NEFF: `dgF{B}` = embedding + layers
    0-11, `dgM{B}` = layers 12-23, `dgT{B}` = layers 24-33 + final norm + LM head + argmax.
    B = global-attention bucket (512 .. 8192, chosen by the host from the position). 3 executions per token.
    - hidden state stays in SBUF ([128, KT, 1]) across the layers of a group;
    - every weight goes through one SBUF ring (`WRing`, 6 slots x 2 MB per core) on the Sync engine's hardware DGE
      queue (Sync runs nothing else), so the stream prefetches across GEMMs and layers;
    - the K/V window of each layer is loaded at the top of the layer (Sync queue, persistent double buffers) and
      *excludes the current position*: the current token's score is a separate q.k matmul and its V row is added
      as a one-row 128-block in the PV accumulation. So nothing on the Sync queue waits for compute;
    - T=1 RMSNorm: sum of squares as x^T x per k-tile on the TensorE (bf16 products exact in fp32), stats on one
      partition, broadcast to all partitions by a K=1 fp32 matmul with a ones row (exact); q/k norms the same per head;
      RoPE, q/k norms and GELU*up as a few batched instructions with access-pattern views (`bc`).
  - **Prefill** (`kernels/pre2s.py`): prompts <= 256 tokens: `pre1s_group_body` (`pgW256_256` for layers 0-11 and
    12-23, `pgT256_256` for 24-33), one 256-token chunk. Longer prompts: 512-token chunks, `pre2s_group_body`
    (`p2s512W{BL}_{BG}` + `p2s512T{BL}_{BG}` while the global bucket BG <= 2048; `p2s512M{BL}_{BG}` x 5 +
    `p2s512T4_1280` beyond, because the 12-layer NEFF was 4.7% slower than two 6-layer ones at BG = 4096): the chunk is
    two streams of 256 tokens (A = first half, B = second) that share one pass over the weight slabs (B's matmuls of
    slab s follow A's of slab s+lag), and the program interleaves the streams phase by phase so one stream's
    vector/scalar work overlaps the other's TensorE work. Global buckets every 512 positions up to 4096.
    - `pre_layer` / the pre2s phases: Gram-matrix RMSNorm statistics (`gram_rstd`: diag of X^T X accumulated over
      k-tiles on the TensorE, stats per token on [128, 1], broadcast by a ones-matmul of diag(rstd)), fused
      `(x * w) * rstd` per k-tile, q/k norms + RoPE batched over heads, attention keys = cached window
      [start, start + B - T) (ACT-queue DMA, loaded once per KV head, no dependency on this chunk) + the chunk's own
      K/V from SBUF; scores masked with `select_reduce` (mask + row max in one VectorE pass), exp on ScalarE,
      P^T by TensorE transposes (evacuated by ScalarE), PV, gate/up GEMM fused with GELU*up per tile pair; attention
      software-pipelined (scores/softmax of work item i+1 emitted before the PV of item i), score tiles of 256 keys;
      the o_proj / down all-reduce runs piece by piece (per PSUM group of 2 n-tiles) while the GEMM continues.
  - `embed_pre*`, `head_pre*`: embedding gather (x 50.5 bf16) and final norm + LM head + argmax for the prompt's last
    token.
- `kernels/variants.py`: all NEFF variants (names, shapes, call expressions); `kernels/build_neffs.py`: NKI's documented
  build route, parallel, cached by source hash.
- `host/engine.cpp`: protocol, safetensors mmap, weight re-layout, upload, tensor sets per (chunk, group) and per
  decode parity, all executions of a request queued with `nrta_execute_schedule`, one wait.

## Numerics
- Same formulas as round 001 (HF bf16 roundings: RoPE, P in bf16, logits bf16, residuals bf16, norms in fp32 with
  one rounding). Round-002 changes reorder fp32 sums only: norm sums of squares (Gram / x^T x on the PE instead of
  fp32 ones-matmuls), the current decode token's attention term added last, norm scaling `(x*w)*rstd` (fp32) instead
  of `(x*rstd)*w`. Measured effect: same mismatch count against the reference (below).

## Verification (round 002)
- `lab/scripts/judge.py` (session lab, not in the engine): 23 random prompts (1 .. 4096 tokens) x 65 tokens through
  the engine, then the reference's `judge` (teacher forcing): round-001 engine 9 / 1495 mismatching steps,
  round-002 engine 11 / 1495 (all rank 2-5, margins <= 0.44); same prompts.
- same.py (40 prompts x 3 request shapes): every candidate self-consistent; the final two-stream prefill is bitwise
  identical to the single-stream group prefill (40/40).
- `enginesmith dev-eval` (1 round): PASSED, score 13.25 (prefill 18.03, decode 9.74), 2 mismatching steps.

## Measurements (dev-eval 003, one round; ms; the CPU reference's times vary a lot between runs)
| case | prefill ref / engine | decode (64) ref / engine |
|---|---|---|
| 256 | 325 / 17.3 | 7290 / 786 |
| 1024 | 1101 / 59.4 | 8713 / 788 |
| 4096 | 10014 / 239.5 | 9314 / 804 |
score 16.09 (prefill 24.4, decode 10.6), PASSED, 2 mismatching steps. dev-eval 002 (earlier this round): 13.25.
Round 001 engine (ab.py, same machine): prefill 22.4 / 84.3 / 352.8, decode 1027 / 1032 / 1046.

## What each change bought (A/B or device profile; profile = neuron-explorer median of 4 executions)
| change | effect |
|---|---|
| decode: 6 group NEFFs, Sync-queue weight ring, K/V window without the current position, batched T=1 ops | decode 1027 -> 861 ms (64 tokens) |
| decode: weight ring 2 MB slots (was 512 KB) | dgM1536 2200 -> 2105 us, dgT 3693 -> 3472 us (profile) |
| decode: persistent K/V window buffers (compiler had reused their space, ordering the next layer's load behind the MLP) | part of the next row |
| decode: T=1 norms from x^T x (20 fp32 ones-matmuls took ~8.6 us per norm) | dgM1536 2105 -> 2031 us; decode 861 -> 846 ms |
| prefill layer v2 (ring, batched RoPE/q-k norms, cached window + SBUF chunk K/V, fused GELU) | 587 -> 546 us per layer (L256) |
| prefill: Gram-matrix norm statistics | 546 -> 498 us per layer |
| prefill: 6 layer-group NEFFs per chunk | prefill 256/1024/4096: 22.4/84.3/352.8 -> 18.6/69.2/280.1 ms |
| prefill: two-stream 512-token chunks | 1024: 69.2 -> 63.4 ms; 4096: 280 -> 254 ms |
| norm weights partition-major on the host (the [4, D] layout made 10k two-byte DMA descriptors per layer) | dgM1536 2037 -> 2013 us, pgM256 2828 -> 2783 us |
| decode V cache row written from the token-major copy (was 512 two-byte descriptors per layer on GpSimd's queue) | dgM1536 2013 -> 1897 us; decode 846 -> 795 ms |
| decode: 3 NEFFs per step instead of 6 | decode 796 -> 786 ms |
| prefill: all-reduce in pieces during the o/down GEMMs | p2s512 group 5398 -> 5256 us, pgM256 2783 -> 2623 us |
| prefill: pipelined attention, 256-key score tiles | p2s512M1280_2048 5503 -> 5399 us |
| prefill: 3 group NEFFs per chunk (BG <= 2048), exact global buckets | prefill +0.6 / +0.6 / +0.7 % (256/1024/4096) |
| host: spin on a non-blocking stdin poll for 10 s before blocking in read() | prefill 256: +1.9%, 1024: +0.4% |
| host: prompt tokens + last-token column + position in one device tensor (slices), one write per prefill | prefill 256: +0.5% |
| total (ab.py vs the round-001 engine) | prefill 22.4/84.3/352.8 -> 17.2/59.3/239.8 ms, decode 1027 -> 786 ms |

## Floors (measured)
- DMA streaming ceiling of one logical core (both physical cores streaming, lab/bw): ~605 GB/s per NEFF incl. its
  ~15 us preamble, ~640 GB/s net. Decode reads 7.76 GB per token -> ~12.1 ms floor; the engine runs 12.28 ms/token
  (decode groups at 620-628 GB/s). Decode is at the floor; only fixed per-NEFF costs remain.
- TensorE: a 128x128 bf16 matmul with 256 moving columns takes 113.5 ns (128 columns: 98.5 ns; 512 columns: 327 ns,
  so keep 256). Prefill GEMMs: 2880 tiles per layer per core per 256 tokens -> 327 us per 256-token layer; prefill
  layers run at ~435 us (256 prompt) and ~437 us per 256-token stream (512-chunks, TensorE 90% busy). Remaining:
  attention (~8% of TensorE time), Gram norms (~6%), idle (~10%).

## Tried and rejected (round 002)
- Step-scheduled two streams with a half-layer offset and separate weight streams (kernels/pre3.py, not built):
  5655 vs 5249 us per 6-layer group: the MLP at 256 tokens already consumes ~290 GB/s of weights per core, so a
  second weight stream makes it DMA-bound.
- Larger lags between the two prefill streams, splitting each shared GEMM into lead/tail around the other stream's
  phase: no gain (within 0.6%) or worse (lags 6/12/12: +6%).
- Alternating P^T evacuations between ScalarE and VectorE: neutral to -0.2%.
- Batched q/k-norm Gram statistics for all 6 heads in one PSUM tile ([128, 6, 128] + broadcast matmuls of 4 diag
  blocks): two-stream group 10420 -> 11272 us (+8%, PSUM pressure); single stream unchanged. Reverted.
- ScalarE cannot run `tensor_tensor` (only vector/gpsimd); GpSimd elementwise is ~5x slower than VectorE
  ([128,256] multiply: DVE 0.34 us, ScalarE tensor_scalar 0.40 us, GpSimd 2.08 us); bf16-only VectorE ops run at
  ~1.6 elements/ns per partition, ops with fp32 operands at ~0.65.
- First execution of a NEFF: no measurable extra cost (second request of each length equally fast).
- Decode: one local K/V window buffer + one more ring slot: no change. Decode without the cross-core sendrecv
  (timing experiment): no faster, so the core syncs do not limit decode.
- Weight DMAs on the Sync HWDGE queue *without* a ring (per-GEMM allocation): no gain (compiler did not prefetch).
- One ring tensor `[128, nbuf, SL*128]` indexed per slot: the compiler tracked it as one buffer (each DMA waited for
  the previous slab's matmuls); separate `nl.ndarray` per slot works.
- K/V window loads on the Sync queue after the cache write (in program order): blocked the weight stream behind
  attention compute. On the ACT queue: dynamic-address DMAs cost ACT time, attention got slower.
- Decode cache writes on the ACT HWDGE queue instead of GpSimd: no change. Decode allreduce with the default DMA
  engine instead of `gpsimd_dma`: no change (2067 vs 2031 us).
- Prefill: kt-reduction of squares by `tensor_reduce` over a strided view (8.7 us per norm) - slower than PE.
- Two-stream prefill for the 256-token prompt (two 128-token streams): 2961 vs 2834 us per 6-layer group (worse).
- DMA ceiling measured with a pure streaming kernel (lab/bw): ~375 GB/s per physical core for any slot size
  (512 KB - 4 MB), HWDGE or DGE none; SWDGE alternating slower. Decode groups reach ~74-82% of it.

## Pitfalls found
- `nc_matmul(..., accumulate=False)` into one column of a PSUM tile whose other columns hold finished results
  corrupts them (decode q/k norm: garbage tokens). Let the compiler infer the flag (accumulate omitted).
- Kernel source is parsed: classes must derive `nl.NKIObject`; no tuple unpacking / `f(*args)`.
- Profiles: `hbm_read_bytes` and per-engine numbers are for the NEFF; the instruction trace (`sg01`) interleaves both
  physical cores; per-core DMA from the `dma` records by subgraph.

## Next steps (estimates)
- Decode is at the streaming floor (~1-2% left in per-NEFF fixed costs).
- Prefill is TensorE-bound (90% busy): what remains is non-GEMM TensorE work (Gram norm matmuls + fp32 broadcasts,
  attention transposes) and ~10% idle in the norm/attention phases. Ideas: fewer fp32 broadcast matmuls (batch
  the q/k-norm ones), skip statically fully-masked 128-key blocks in local attention (~1%).
- Verification extras run this round: 200-token prompt + 7993 tokens to the 8192 limit (returns all tokens, the
  next continuation is refused), 8000-token prompt + 192 tokens, split/chain consistency at 5000 and 8000 tokens.

## Round 001 notes (summary)
First engine: per-layer NEFFs (36 per decode step), feature-major layouts, weight tiles `[core][p][(nt*KT+kt)*128+j]`,
core split (q/k/v by heads, o/down along K with fp32 sendrecv allreduce, gate/up along N, LM head along vocab),
KV cache `kc[4,256,8192]` (K transposed) / `vc[8192,1024]`, RoPE tables from torch at build time (bit-identical to
HF), argmax ties to the lowest id. Long-output check: 200-token prompt + 7993 tokens to the 8192-position limit works.
