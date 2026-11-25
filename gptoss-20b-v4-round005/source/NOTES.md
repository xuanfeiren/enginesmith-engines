# gpt-oss-20b on Trainium2: engine notes

Sessions 001 (first engine), 002 (fused decode, sparse prefill MoE, whole-sequence flash attention in prefill),
003 (fp8 e5m2 expert weights, activation-stationary decode GEMVs, KV reads off the critical path; see "Session 003"),
004 (TensorE column tiling for every decode GEMV, all-reduces as SBUF all-gathers; see "Session 004") and 005
(fewer NEFF executions in prefill, shared SBUF weight regions and keys-on-partitions attention in decode; see
"Session 005"). Numbers are labelled "measured" with where, or "estimate".

## Session 005 (read first)

Result: dev-eval 002 on the engine as left (clean build of these sources, 9.0 min, 3 rounds): PASSED, mismatching
steps 0,0,0 / 3,3,3 / 0,0,0, score 0.0604 (session 004: 0.0575, +5.0%; dev-eval 001 of this session, 1 round:
0.0605). Prefill 30.6 / 70.2 / 208.2 ms, decode (64 tokens) 225.6 / 227.6 / 234.5 ms at 256 / 1024 / 4096
(session 004: 33.0 / 70.2 / 207.9 and 238.9 / 246.8 / 258.5). same.py of the final clean build vs the dev-eval-001
binary: 40/40. Only item 5 is not bitwise identical to session 004 (prompts > ~1000 tokens: 1152+ buckets).

Measuring setup used throughout: candidate on cores 4-7 vs a saved binary on cores 0-3 (`ab.py`, `same.py` of the
skill scripts; wrapper scripts set NEURON_RT_VISIBLE_CORES). Cores 4-7 run prefill ~1-2% slower than cores 0-3 with
the SAME binary (measured both ways), so prefill ratios below are slightly pessimistic for the candidate.
Profiles: `tools/dev/prof.sh` (neuron-explorer, zero inputs; zero routing = experts 0-3 everywhere).

### What changed (in order; "same.py 40/40" = every token of 40 cases x 3 request shapes identical to the parent)
1. **LM head activation-stationary with TensorE column tiling** (`CT_HEAD`, kernels/model.py `_head`): the hidden
   tile (zero-padded to 32 stationary columns) is stationary, the 4 vocab tiles of a weight group stream as moving
   operands in the 4 PE column groups, one exact fp32 transpose per group gives the [128, tile] logits layout.
   4600 weight-stationary 1-column matmuls -> 1150 rounds of 4 parallel 128-column ones. The head is now
   DMA-bound: ~440 us at ~340 GB/s (145 MB per physical core) + ~60 us tail (was ~620 us, PE ~= DMA).
2. **Decode top-4 by `max8` + `nc_find_index8`** (`TOP8`): `max8` gives the 8 largest descending, `nc_find_index8`
   pairs duplicate values with ascending positions = ties to the lowest expert id, the same selection and order as
   the 4 rounds of reduce / compare (22 serial single-partition ops). Router + top-4 ~13 -> ~7 us per layer.
   Items 1 + 2: same.py vs session 004 40/40; decode 238.9 / 246.8 / 258.4 -> 233.4 / 237.3 / 254.2 ms (ab.py).
3. **Shared SBUF weight regions** (`SHARE_W`): attention sets ping-pong in R0 / R1 (bf16 [128, 27648] each), and
   layer l's experts 0, 1 live in the fp8 `.view` of R[l % 2] (its attention weights are dead once the router has
   run), experts 2, 3 in C. That frees R[(l + 1) % 2] during layer l's attention chain, so the next layer's
   attention weights (7.1 MB) are issued at the layer START and stream while the DMA queue is otherwise idle
   (profile: before, 19.5 MB streamed in one ~56 us block after the router). Needed: `instr.depends_on()` of those
   DMAs on the previous layer's output (`prev_end`; the compiler otherwise placed them after the experts) and the
   first expert DMA on them. same.py 40/40; profile step384 125.9 -> 122.0 us per layer; decode 233.9 / 237.4 /
   254.5 -> 225.5 / 231.6 / 254.1 ms (ab.py). No gain at 4096 (see 5).
   - FAILED variant, worth knowing: the regions at fixed SBUF addresses (`nl.ndarray(..., address=...)`) with
     separately declared fp8 tensors at the same address gave deterministic WRONG tokens (from token 12 of a 256
     prompt), at the top of SBUF and at address 0 alike: the auto-allocator does not avoid explicit addresses (and
     the compiler keeps internal constants such as the transpose identity at the very top, ~229 KB). Use `.view`.
4. **All 24 dense one-chunk prefill layers in ONE NEFF** (`prefill_dense`, NEFF `pdense`; kernels/prefill.py
   `_dense_layer` is the shared body, per-layer offsets into the stacked tensors, the hidden state stays in SBUF
   between layers). The dummy expert of the used-expert loop now has a weight row one past the stack (DMAs
   skipped) and a separate routing row E (zero): elist rows are (weight row, routing row, 0, 0). Per-layer
   executions paid a start / rank-skew delay at each first collective (profile: first 2.9 MB all-reduce of an
   execution 149-174 us, later ones 33-41 us). same.py 40/40; prefill 256: 32.5 -> 30.6 ms (ab.py).
5. **Full-layer decode attention with the keys on partitions** (`ATT_T_MIN = 1024`, `_attn_keys_on_partitions`):
   scores^T [128, NB, NQ] (key 128 j + p) by NB small matmuls (K block stationary, q moving), the mask
   precomputed once per step (`_key_mask_T`), exact max (transpose + reduce), per-head scalars broadcast over
   partitions by exact ones-matmuls, exp, probabilities bf16 and directly the PV moving operand (no transposes).
   Each elementwise pass touches NB * 8 elements per partition instead of B on 8 partitions (B = 4224: 4.4 us per
   pass before). Same rounding points; only the fp32 order of the softmax sum differs (NOT bitwise: ab.py token
   mismatches after 12..36 tokens in 3 of 6 requests). Profile step4224 4115 -> 3908 us per step; decode 1024 /
   4096: 231.8 / 254.0 -> 227.6 / 234.7 ms (ab.py). dev-eval 001 (items 1-5): PASSED, mismatching steps 0 / 3 / 0,
   score 0.0605.
### Measured and rejected this session
- **Multi-chunk prefill with fewer executions** (attention half + sparse MoE in one NEFF; code not kept):
  - Two layers per NEFF (`prefill_pair`: the old kernels' bodies on per-layer VIEWS `t[i]` of 2-layer slices of the
    stacks, `nrt_tensor_allocate_slice` of 2 layers; HBM integer indexing drops the dimension and `.ap(offset=)`
    composes with the view's base): WRONG tokens for 511..4096-token prompts (first token already). With only the
    first layer of each pair computed it matched the per-layer NEFFs exactly, so the second layer in the same NEFF
    goes wrong (two nkilib attention calls in one NEFF? its SBUF buffers sit at fixed addresses); core barriers
    on every shared scratch tensor between the halves did not help. `ppair18` also exceeds SBUF.
  - One layer per NEFF (`prefill_lay`, 24 instead of 48 executions): tokens identical, but prefill 1024 70.6 ->
    68.5 ms and 4096 208.3 -> 215.1 ms (ab.py): no net gain. It also failed `hello` reproducibly
    (NRT_EXEC_UNIT_UNRECOVERABLE in the 4400-token warmup prompt, only after the whole warmup sequence; passes
    with ENGINE_SYNC_EACH or other orders). Not analysed further.
- Dense one-chunk MoE token-major (`DMOE_TM`, `_dmoe_expert_tm`: activations stationary, fp8 weights moving,
  fp32 [128, 2880] accumulators, transposes back once per layer): bitwise identical (ab.py) but slower, prefill 256
  30.1 -> 31.5 ms. fp8 moving matmuls stream ~2.4 columns/ns marginal but cost ~160 ns fixed each (480 columns
  366 ns, 240 columns 266 ns); bf16-moving 256 columns 273 ns.
- Expert weight DMAs alternated over the Sync and Scalar queues: 123.8 -> ~123 us per layer (none), even with all
  4 experts' DMAs issued before any expert compute (so the Activation engine reaches them on time): the expert
  stream stays at ~240 GB/s aggregate. This is close to the HBM share of a physical core (2.9 TB/s per chip / 8 =
  ~360 GB/s with all 8 streaming), so the decode expert phase is near its bandwidth floor.
- GU_DMA = 1 with SHARE_W: 123.8 -> 126.6 us per layer.
- Next layer's attention DMAs also after this (full) layer's K/V reads (`AW_AFTER_KV`): decode 4096 -0.9%.
- The `no_reorder()` block around the early attention-weight DMAs: neuronx-cc segfault (exit 70).

### Decode layer now (profile step384, zero routing; relative to the router's max8)
experts 0-56 us (DMA ~240 GB/s, TensorE busy ~42 of them) -> MoE all-gather + combine ~16 -> residual, norm, QKV,
attention, o_proj ~22 -> attention all-gather ~18 -> norm, router, top-4 ~7. Period ~122 us; LM head ~0.5 ms.
Per layer and core 19.5 MB of weights = ~54 us at the HBM share: the rest (~65 us) is the latency chain.

### Next steps (estimates)
- Decode: the chain is collectives (~34 us per layer for two tiny all-gathers + sendrecv) and ~30 dependent small
  ops. Speculative expert-weight prefetch (the previous token's experts of the same layer, skip the DMA via an
  out-of-range offset on a hit) could hide part of the 50 us expert stream: needs the hit rate first.
- Prefill 256 (30 ms): MoE loop ~17 used experts x ~35-40 us per layer (TensorE, 273 ns per 256-column matmul)
  and two 2.9 MB all-reduces per layer at ~100 us of DMA each (120-partition transfers, sendrecv + HBM copies).
- Prefill 4096 / 1024: unchanged kernels (pmoe combine 1.4 ms per layer at 4096, pattfa all-reduce 0.5 ms).
- tools/test_prefill.py and tools/test_layer.py are stale (old argument lists); same.py against a saved binary was
  the check used.

## Session 004 (read first)

Result: dev-eval 004 on the engine as left (clean build of these sources, 3 rounds): PASSED, mismatching steps
0,0,0 / 1,1,1 / 0,0,0, score 0.0574 (session 003: 0.0504, +13.9%). Prefill 33.0 / 70.2 / 207.9 ms, decode (64
tokens) 238.9 / 246.8 / 258.5 ms at 256 / 1024 / 4096 (session 003: 37.3 / 70.2 / 208.5 and 299.2 / 305.5 / 319.9).
(dev-eval 002 / 003 of this session are void: 002 was disturbed by a build on the same cores, 003 was cut off.)

Decode 64 tokens after a 256-token prompt (tools/dev/drive.py, cores 4-7): 300.0 -> 239.1 ms (-20%). Prefill of
one-chunk prompts (<= 256 tokens): 38.0 -> ~33.3 ms (the dense layer's MoE loops over the used experts only; item 5).
Every other prefill NEFF is the same arithmetic as session 003. All changes except items 2 and 3 are bitwise
identical to their parent (same.py 40/40 after each).

### What changed (measured with drive.py, 64-token decode after 256 tokens; parent = session 003: 300.0 ms)
1. **TensorE column tiling for the decode GEMVs** (`CT_GU`, `CT_DN`, `CT_QKV`, `CT_OPROJ` in kernels/model.py). An
   activation-stationary GEMV uses 1 of the 128 PE columns. `nc_matmul(..., tile_position=(0, 32 c),
   tile_size=(128, 32))` runs up to 4 matmuls with <= 32-column stationaries in parallel on the 4 column groups
   (the trick of AWS nkilib's moe_tkg kernels; no code copied). Each column tile owns DIFFERENT output columns and
   accumulates over the contraction in the same order as before, so every output is bitwise identical (same.py
   40/40 after each step). The stationary is x zero-padded to 32 columns (x in column 0): all 32 PSUM rows of a tile
   get written (zeros), so the PSUM -> SBUF copy and the exact fp32 transposes (rows 32 c -> columns 32 c) never read
   uninitialised data (a 1-column stationary + reading the other rows failed to compile / would risk NaN).
   - gate/up: 3 tiles = the 3 intermediate tiles it, moving [gate 120 | up 120] (wgu8 per kt now [it][up][n],
     src/engine.cpp `g8`; prefill keeps its own `wgu` layout), then 2 transposes of [65, 120]. Micro-benchmark
     (tools/dev/mb/ct.py, ct2.py, 4 experts' gate/up from SBUF): 30.8 us -> ~8 us of TensorE; a split of the
     contraction over the tiles (kt % 4) needs a cross-tile fp32 sum and was only 2x (15.8 us).
   - down: activation-stationary now, tile c = output tiles nt = 4 jj + c (moving AP strided over nt), 6
     transposes of [97, 120]; replaces 72 weight-stationary matmuls per expert.
   - QKV: tiles = heads (3, 3, 2, 2 of the 10 64-column heads, `QG`), 3 transposes of [97, 64].
   - o_proj: as down (tile c = nt = 4 jj + c, accumulation over the 4 head pairs), 6 transposes.
   gate/up + down: 300.0 -> 278.0 ms; QKV + o_proj: 260.6 -> 248.1 ms.
2. **Attention all-reduce on SBUF tensors** (`AR_SBUF`: `ncc.all_reduce` accepts SBUF src/dst; no HBM hop, no
   core barrier): 278.0 -> 260.6 ms. Not bitwise (the collective's fp32 summation order differs): checked with
   random data (tools/dev/mb/ar.py: max rel. error vs float64 7.0e-8 for both forms) and dev-eval 001 (below).
3. **All-reduces as SBUF all-gather + local sum** (`AR_GATHER`): `ncc.all_gather` of the 4 ranks' partials into
   [P, 4 n] and a fixed-order sum (r0 + r1) + (r2 + r3), identical on every rank. Micro-benchmark: 11.3 us (SBUF
   all_reduce) -> ~2.5-9 us (noisy) per [120, 24] reduction. Both decode all-reduces (attention and MoE): 248.1 ->
   239.1 ms. Max rel. error vs float64 6.2e-8.
5. **Dense one-chunk MoE over the used experts only** (`DMOE_LOOP`, prefill.py `_dense_moe_loop`): per 256-token
   chunk only ~17 of 32 experts are selected by any token (nonce prompts); the others contribute exact zeros. On
   device: u[e] = max_t rtT[e, t] > 0, slot = exclusive prefix sum (strict-triangular matmul), scatter of the ids to
   an HBM list (ascending, padded with the dummy expert E whose weight DMAs are out of range = skipped and whose
   routing-weight row is zero), ONE device loop over pairs (`nl.dynamic_range`, count = #{k : 2k < n_used}); in
   pair i expert 2i+1's weights (register offsets) load while 2i computes and 2i+2's while 2i+1 computes. Same
   arithmetic and expert order: bitwise identical (same.py 40/40). 256 tokens: 38.0 -> 33.3 ms (12 layers 19.8 ->
   19.0, 24 layers 38.0 -> 33.4: the early layers' routing is less skewed). Using the 128-partition weight copies
   in this loop: no change (the loop is not DMA-bound).
4. dev-eval 001 (changes 1 (gate/up, down) + 2, 1 round): PASSED, mismatching steps 2 / 0 / 0, score 0.0539
   (decode 259.0 / 264.3 / 285.6 ms, prefill unchanged).

Decode layer now (neuron-explorer, dprof4_1152, steady state): period ~132 us = MoE all-gather ~11 us -> combine,
norm, QKV, attention, o_proj ~27 us -> attention all-gather ~8-19 us -> router + top-4 ~13 us -> expert weight
stream ~55-60 us (12.45 MB experts + 7.1 MB next-layer attention weights at ~325 GB/s, one Sync queue over 16 DMA
engines, with a dip to ~130 GB/s while the next layer's wo2/wqkv DMA waits on TensorE progress = SBUF reuse).
LM head inside the step ~0.6 ms (bench op `head`: 620 us per execution; weight-stationary, PE ~= DMA time).

### Measured and rejected this session
- One SBUF all-reduce per expert as soon as its partial is done (`AR_SPLIT`): 261 -> 322 ms (collectives in flight
  stall the expert weight DMAs).
- MoE all-reduce on SBUF tensors (`AR_SBUF_MOE`, before AR_GATHER): 260.6 -> 259.8 (noise). Micro-benchmark: for
  [120, 96] the SBUF all-reduce is slower than the HBM one (20.8 vs 18.9 us).
- The LM head's two max all-reduces as all-gathers (`AR_GATHER_HEAD`): 239.1 -> 241.0 ms.
- Persistent ping-pong SBUF buffers for the attention weights (`PERSIST_AW`) again: 239.1 -> 239.9 ms.
- GU_DMA 1 (one gate/up DMA per expert): 242.5 ms; GU_DMA 6: 245.4 ms (3: 239.1).
- Prefill dense layer (256 tokens): a tiny all-reduce first in the NEFF (`CC_WARM`, prefill.py): 38.0 -> 38.5 ms;
  its all-reduces on SBUF tensors (`AR_SBUF_PF`): 38.9 ms. The SBUF all-gather of [120, 3072] fp32 fails at run
  time (NRT_INVALID). The HBM all-reduce of [120, 3072] alone: ~55 us (micro-benchmark).

### Prefill facts measured this session (for the next one)
- Eval prompts are Lily nonce tokens (enginesmith workload.prompt), the same family as the tools' nonce prompts, so
  the routing below is representative. Real routing is very skewed: after 12 layers, 4096 tokens: top experts get
  4059 / 3018 / 2593 / 2482 / 2091 tokens, 146 blocks of 128 rows; per 256-token chunk only ~17 of 32 experts and
  ~31 of 64 (expert, 128-token tile) pairs have any token.
- 256 tokens: 1.5 ms per dense layer (ENGINE_PF_LAYERS 12 vs 24), fixed part ~2.0 ms (0 layers: embed 79 us,
  gather 94 us, head 620 us per queued execution, the rest per-request launch / completion latency). In the dense
  layer the FIRST all-reduce (attention, 2.9 MB) takes ~300 us (91 us start delay + 211 us) while the same-size MoE
  one takes ~37 us; not DMA contention (no other DMA runs); not fixed by a dummy collective first.
  Dense MoE 0.88 ms/layer is TensorE-bound over all 32 experts x 256 tokens; skipping the unused (expert, tile)
  pairs (about half, see above) is the clearest prefill-256 lever (needs device-side control flow).
- 4096 tokens: pattfa_f16 3.46 ms (phase 1 QKV/RoPE 0.74 ms, VectorE-bound: _rmsnorm_fm's 4 fp32 passes per chunk;
  nkilib attention 1.30 ms; o_proj 0.34; one 23.6 MB all-reduce ~0.5 ms with nothing overlapping; post 0.55).
  pmoe32 with real routing 5.36 ms (route 0.4, static first blocks 0.6, block loop 2.95 = ~26 us per 128-row block,
  TensorE-bound near its moving-column rate, combine 1.4). pattn 0.82 + pmoe8 2.02 ms per layer at 1024 (bench).

### Next steps (estimates)
- Prefill 256 is the most score-sensitive case (1 ms = ~0.45% score): its first all-reduce per dense layer (~300 us
  vs ~37 us for the same size later; cause unknown), ~2 ms fixed per request (head 0.62 ms + launch latency), the
  dense loop still ~0.6 ms/layer (VectorE SwiGLU chain between the two unrolled experts; try 3 per iteration or
  skipping empty 128-token tiles).
- 4096 / 1024: pmoe combine 1.4 ms/layer, pattfa all-reduce 0.5 ms with nothing overlapping, phase 1 VectorE-bound
  (_rmsnorm_fm: drop the bf16 -> fp32 cast pass, bf16 inputs are exact).
- Decode: ~55-60 us/layer of weight streaming at ~325 GB/s (stall while next-layer attention weights wait on
  TensorE); full-layer attention softmax on 8 partitions (~5 us); LM head ~0.6 ms per token (PE ~= DMA).

### Tools added (tools/dev)
- `mb/`: micro-benchmarks run through NKI's run-once route under the GPU lock (`run.sh` 1 rank, `runc.sh` 4 ranks)
  and profiled with neuron-explorer (`prof1.sh`, `profc.sh`): `ct.py` / `ct2.py` column-tiled GEMVs, `ar.py`
  collectives (modes h = HBM all-reduce, s = SBUF all-reduce, g = SBUF all-gather + sum, n = no sendrecv,
  x = sendrecv only). Note: a 4-rank NEFF's first collective shows a large start delay in these profiles.
- `crit.py NTFF A_US B_US`: instruction runs of one core in a time window (critical-path reading).
- `rep.py`: the same prompt repeatedly (determinism check).
- The bench op also takes `pembed` and `pgather`; ENGINE_TIMING=1 prints run_steps' time.
- tools/test_layer.py (decode_layer, G = 1) still uses the 120-partition `wgu` layout and is NOT updated for the
  column-tiled decode (wgu8 order [it][up][n]); same.py against a saved build was the check used instead.

## Session 003

Result: dev-eval 005 on the engine as left (full rebuild of these sources, 3 rounds): PASSED, mismatching steps
2,2,2 / 0,0,0 / 0,0,0, score 0.0504 (session 002: 0.0372, +35%). Prefill 37.3 / 70.2 / 208.5 ms, decode (64
tokens) 299.2 / 305.5 / 319.9 ms at 256 / 1024 / 4096 (session 002: 40.1 / 91.5 / 254.6 and 460 / 469 / 490).
Biggest levers, in order: fp8 e5m2 expert weights (exact), the activation-stationary decode GEMVs, 128-partition
DMAs (the LM head), KV reads before the write, o_proj with 128-row contraction, the sendrecv core exchange in prefill.
Development tooling of this session: scratch scripts outside engine/ (not committed): a neuron-explorer wrapper for
any build_neffs variant (`ES_GROUP=4 prof.sh dprof4_1024 POS`), micro-benchmarks for TensorE shapes and DMA patterns
(the DMA one is described in item 16). tools/dev/test_l0.py and tools/test_layer.py / test_head.py were updated
for the new layouts and are the numerics checks used throughout.

### What changed (all measured on cores 8-11 with tools/dev/drive.py, 64-token decode after a 256-token prompt
unless noted; parent engine = session 002: 460.8 ms)
1. **Expert weights as fp8 e5m2, exactly.** This checkpoint's MXFP4 block scales are 2^-12 .. 2^9 (all 24 layers,
   checked), so every dequantized value fp4 * 2^s = 2^b * {1, 1.5} with b in [-13, 11] lies in e5m2's *normal* range
   with one of its two mantissa bits: the e5m2 byte is the exact value (src/engine.cpp `f2e5m2` dies otherwise).
   TensorE takes mixed operands (fp8 stationary or moving with a bf16 other operand, fp32 accumulation), so every
   kernel keeps its structure and only the SBUF tile dtype changes (`WDT` in kernels/model.py). Expert bytes per
   token halve (decode 24.9 -> 12.45 MB per physical core per layer). same.py vs the session-002 engine: 40/40 cases
   identical (every token). Decode 460.8 -> 381.1 ms, prefill 40.7/92.8/258.7 -> 39.7/80.3/245.2 ms (ab.py, 4 rounds).
2. **pmoe4 removed** (prompts of 257..512 tokens use pmoe8): with fp8 weights, pmoe4 computed some tokens of its
   block loop's first sub-block wrong (layer-0 check tools/dev/test_l0.py at 300 tokens: rel. error up to 0.34 on
   tokens 132..252, varying run to run). The session-002 engine has the same latent race: in one same.py run its own
   three request shapes disagreed on lily-p257 and rand-p300 (it was not caught before because it is rare with bf16
   weights). pmoe8 at 300 tokens is bitwise identical to the session-002 result. A ybuf core_barrier before the
   combine and 16-bit-container DMAs for the loop's weights did not fix pmoe4 (not kept). Cause not found.
3. **Decode gate/up activation-stationary**: the x column is the stationary operand and each kt's 720 outputs
   stream as the moving operand (wgu layout now kt-major `[p][kt][t][n]`, prefill kernels re-indexed), output row
   [1, 720] -> 6 exact fp32 `nc_transpose`s back to the column layout. Profile: a weight-stationary 120x120 matvec
   costs ~100 ns of TensorE (144 per expert gate/up = 14.4 us), the activation-stationary form 7.3 us
   (micro-benchmark scratch/mb). 381 -> 359 ms. Tokens identical (same.py 38/40; the 2 "bad" cases were the
   session-002 engine's own pmoe4 inconsistency).
4. GU_DMA = 3 DMAs per expert gate/up slice (5760 B per partition each) instead of 6: 359 -> 355.7. One DMA per
   expert: 386.8 (the GEMV waits for the whole expert).
5. **QKV and o_proj activation-stationary** (wqkv layout kt-major `[p][kt][t][n]`; o_proj streams wo per head):
   QKV 15 -> 11 us per layer. o_proj stayed ~20 us: a 64-row contraction (one head) streams at half the rate.
6. **KV cache read before the write**: the step reads K/V (positions < pos) at the start of the layer and writes the
   new K/V through aliases `kcw`/`vcw` bound to the same device tensors (so the compiler does not order the reads
   after the write); the slot of pos is masked (j < lim) and the current token's score / value come from SBUF (one
   extra [NQ, 1] score matmul, two small transposes, one K=1 PV matmul). Not bitwise: the softmax sum adds the
   current term separately (fp32 order). Decode-layer check against the numpy reference (tools/test_layer.py,
   layer 1 pos 600, layer 0 pos 300, layer 3 pos 2000): error figures identical to 4 digits. 359 -> 340 ms
   (4096: 387.5 -> 365.4). same.py vs session 002: 26/40 identical, the others diverge after 17..61 tokens
   (near-ties). dev-eval 001: PASSED, 0/0/0 mismatching steps, score 0.0452.
7. **o_proj with 128-row contraction**: head pairs stacked on 128 partitions (decode-only weight `wo2`
   [2][128][4 * H], row (hh % 2) * 64 + d); the attention output [64, 8] -> [128, 4] by two exact 0/1 selection
   matmuls (`_sel2`). o_proj 20 -> 8 us per layer; 340 -> 321 ms; layer error identical.
8. Next layer's attention weights issued after this layer's expert DMAs and before the MoE all-reduce (`_attn_w`,
   `aw`): 321 -> 319.5 (small: the compiler still schedules the wqkv DMA after the all-reduce's DMAs, waiting on a
   VectorE semaphore = SBUF reuse).

9. dev-eval 002 (changes 1-8, 3 rounds): PASSED, 0/0/0 mismatching steps in every round, score 0.0467
   (prefill 39.1 / 77.8 / 238.5 ms, decode 316.9 / 326.5 / 336.6 ms).
10. **Prefill o_proj with 128-row contraction** (head pairs, `wo2`; flash-attention path loads the pairs straight
    from obuf, the dense one-chunk layer stacks them with `_sel2` matmuls): drive.py prefill 1024 77.8 -> 75.6,
    4096 238.5 -> 225 ms (together with 11). Layer-0 check (test_l0 at 256 / 600 / 2048): identical figures.
11. **Physical-core exchange by sendrecv** instead of the xch HBM round trip (`XCH_SENDRECV`): prefill_moe combine
    (tile pairs: send the partial of the tile the other core finishes), prefill_attn_fa phase 3 (o_proj per
    128-token half, the other core's half sent), `_allreduce_fm` (dense one-chunk layer). sendrecv on
    `dma_engine.gpsimd_dma` is limited to 1 KB per partition; `dma_engine.dma` takes the 11.5-24 KB rows.
    1024 75.6 -> 73.6, 4096 225 -> 219, 256 39.6 -> 38.1 ms; layer checks identical.
12. **Prefill QKV with head pairs** (`_fa_qkv`: 128-column stationary tiles, block-diagonal RoPE rotation, pair
    layouts of bias / cos / sin, V moved to partitions 0..63 by one SBUF DMA and transposed once for both the cache
    and vbuf): 1024 73.6 -> 72.2, 4096 219 -> 214.7 ms. Identical layer checks.
13. **Decode attention buckets 384 / 1152 / 4224** (prompt + 128 above the timed lengths; FULL_BUCKETS now
    384/512/1152/2048/4224/4608, score chunk = largest of 512/384/256/128 dividing B): decode after a 1024-token
    prompt 328.2 -> 319.4 ms (it ran in the 2048 bucket), 256: 320 -> 318.7, 4096: no change. Warmup prompts
    include 400 and 4400 so that every bucket runs at load. Long-output check (tools/dev/longout.py): every token,
    up to cache slot 4607 (its "past the end" line is off by one: that request still ends at slot 4607).
    same.py vs session 002: 22/40 identical, the rest diverge after 0..57 tokens (two at the first token: lily-p15,
    lily-p33), all three request shapes self-consistent.

14. dev-eval 003 (changes 1-13, 3 rounds): PASSED, mismatching steps 2,2,2 / 0,0,0 / 0,0,0, score 0.0492
    (prefill 37.5 / 70.4 / 208.0 ms, decode 315.3 / 317.5 / 337.4 ms). A clean build from a fresh copy of the
    sources (8.0 min, two 4-core build groups) gives the same tokens in all 40 same.py cases.
15. prefill_moe: expert 0's weight DMAs issued before the routing phase (they do not depend on it): 4096
    214.7 -> 211.8 ms, 1024 unchanged; layer checks identical.
16. **DMA partition count matters most** (scratch/mb/dmab.py, one core, 134 MB streamed into SBUF, hwdge Sync):
    exactly 128 partitions 402-410 GB/s (16 KB, 32 KB or 48 KB per partition alike); 120 partitions 190 GB/s; 64 /
    96 / 112 partitions 201 / 209 / 217 GB/s; 127 partitions 27 GB/s. Sync vs Sync+Scalar queues vs default DGE: no
    difference. Every weight of this engine was laid out on 120 partitions (H = 24 x 120).
    - LM head on 128 partitions (hidden dim in 23 tiles of 128, rows >= 2880 zero; the normalized vector moved
      into that layout by a small HBM round trip): decode 318.7 -> 303.1 ms (256), 339 -> 324 (4096).
      tools/test_head.py: argmax identical to the numpy reference for 3 seeds.
    - Decode copies of wgu / wd / wqkv padded to 128 partitions (rows 120..127 zero, the activation operands get
      zero rows: `_rmsnorm(pad=True)`, `inter` [128, 3]; same arithmetic, 6.7% more bytes): 303.1 -> 300.9 ms only.
      The expert weights now stream at ~330 GB/s, but the expert phase is TensorE-bound (~70 us per layer: 4 x 48
      activation-stationary gate/up matmuls ~208 ns each + 4 x 72 weight-stationary down matmuls ~100 ns; the
      profiler reports TensorE power throttling, avg util limit ~0.55). Per-layer period in the full 24-layer
      step (neuron-explorer step384): ~160-165 us.

17. dev-eval 004 (changes 1-16, 3 rounds): PASSED, mismatching steps 2,2,2 / 0,0,0 / 0,0,0, score 0.0504
    (prefill 37.5 / 70.2 / 208.5 ms, decode 299.3 / 305.8 / 320.1 ms).

### Measured and rejected this session
- prefill_moe on the 128-partition expert copies (xT / interT padded with zero rows): 4096 211.8 -> 220.5 ms
  (slower), 1024 72.2 -> 71.8 (noise). Layer checks identical. Not kept (the static pass is not the only limit;
  not analysed further).
- Next layer's attention weights issued at the start of the layer (before routing): 321 -> 343 ms.
- Same on the Scalar engine's DGE queue: 327 ms. Persistent ping-pong buffers for them: no change (319.5).
- Persistent SBUF buffers for the 4 experts' weights: 331.7 ms (the 3rd/4th expert's DMAs wait on TensorE progress
  because the allocator reuses an earlier expert's buffer; fixing that this way costs elsewhere).
- wo2 loaded at its own layer's start: 330.5 ms.
- o_proj chunks interleaved over 3 PSUM banks (with the 64-row form): no change.
- MXFP4 decoded on chip (4-bit storage): not tried; decode expert phase is DMA- and TensorE-bound at about the same
  time, 4-bit would need >= 2 vector ops per weight (estimate ~75 us per layer of VectorE), not worth it.

### Where decode time goes (neuron-explorer, dprof4_1024, steady-state layer ~167 us, before change 16)
QKV 11 -> RoPE/scores/softmax/PV 6 -> o_proj 8 -> attention all-reduce ~10 + DMA hops 4 -> norm/router/top-4 10 ->
expert DMA start ~8 -> experts ~80 (DMA-bound, 12.45 MB at 155..225 GB/s: the 3rd/4th expert's DMAs wait for SBUF
reuse) -> MoE all-reduce + combine ~18. LM head ~0.77 ms per token (148 MB per physical core at ~200 GB/s; 15 DMA
engines at ~13.4 GB/s each, also with 24 KB descriptors). The DMA engine rate looks like the ceiling here.

### Where prefill time goes (pmoe32 = MoE of one layer at 4096 tokens, 3.52 ms)
route 0.41 ms (128 SW-DGE scatters of rec rows on GpSimd) / static first-block pass 0.6 ms (DMA-bound, 100 MB of fp8
weights at 175 GB/s) / heavy-block loop 0.7 ms (TensorE) / combine 1.75 ms (ybuf slot rows 94 MB + xch 47+47 MB per
core, all-reduce ~0.3 ms, bias term + transposes 0.5 ms). Dense 256-token layer (player_s, 1.47 ms): attention part
0.43 ms of which ~0.18 ms is the all-reduce of 2.9 MB fp32, dense MoE 0.88 ms (TensorE), MoE all-reduce 0.135 ms.
The sparse path at 257 tokens takes 54 ms (vs 40 dense at 256).

### Ideas not done (estimates)
- Prefill attention weights (wqkv / wo2 into prefill_attn_fa) on 128 partitions; prefill_moe on 128 partitions was
  slower (see rejected), the reason is not known.
- Decode expert phase is TensorE-bound now: fewer / wider matmuls (down projection activation-stationary, 18
  instead of 72 matmuls per expert) are the next lever there.
- Decode top-4 with max8 / nc_find_index8 instead of 4 rounds of reduce/compare (~4 us per layer); check ties.
- Decode down projection activation-stationary (only the last expert's tail is on the critical path, ~3 us/layer).

## Layout of the engine

- `kernels/model.py`: decode. `decode_step` = embedding + all 24 layers + final norm, LM head, greedy argmax in ONE
  NEFF per decode step (one variant per full-attention bucket 512/1024/2048/4608). `_layer` is one layer for one
  token (column layout [120, 24]; session 003: QKV, o_proj and expert gate/up activation-stationary, K/V read
  before the write, next layer's attention weights prefetched); `head` (the LM head alone) is still used after prefill. `decode_layer` (one
  layer) is kept for `tools/test_layer.py`. Session 005: `_head` column-tiled (`CT_HEAD`), top-4 by max8
  (`TOP8`), weight regions R0 / R1 / C (`_attn_bufs`, `SHARE_W`) with the next layer's attention weights issued at
  the layer start, full layers of buckets >= 1024 use `_attn_keys_on_partitions` (mask from `_key_mask_T`, once
  per step).
- `kernels/prefill.py`: prefill.
  - One-chunk prompts (<= 256 tokens): `prefill_dense` (NEFF `pdense`, session 005) runs all 24 dense layers in
    one execution; `_dense_layer` is the layer body (also behind `prefill_layer`, the per-layer NEFFs
    `player_s` / `player_f512` kept for ENGINE_PF_LAYERS / ENGINE_NO_PDENSE debugging). Its MoE is a device loop
    over the experts some token selected (`_dense_moe_loop`, session 004).
  - Longer prompts run layer-major: per layer one `prefill_attn_fa` execution (attention for chunks 0..NCHK-1,
    variants NCHK = 2/4/8/12/16/18, sliding and full) and one `prefill_moe` execution (sparse MoE over the whole
    prompt, variants NT = 8/16/24/32/36 tiles of 128 tokens; NT = 4 removed in session 003).
  - `prefill_attn_fa`: phase 1 per 256-token chunk (norm, QKV, RoPE, K/V caches, q/k/v scratch), phase 2 ONE
    whole-sequence flash attention (`kernels/nkilib_attn`: AWS nkilib `attention_cte`, Apache-2.0, causal, sinks,
    sliding window 128 for sliding layers; q pre-scaled by 1/8), phase 3 o_proj partials of all chunks -> `xch`,
    then each physical core finishes the 128-token half `pid` of every chunk (other core's partial, ONE all-reduce
    for the prompt, residual, norm, router, MoE inputs). `prefill_attn_multi` (per-chunk attention, exact causal key
    ranges) and `prefill_attn` (one chunk) remain in the source but are no longer built.
  - `prefill_embed` / `prefill_gather` as before.
- `tools/build_neffs.py`: compiles every NEFF variant (NKI's documented run-once route), cached in
  `target/neffcache` keyed by the kernel sources. Dev-only knobs: `ES_KEEP=prefix,...` (reuse NEFFs already in the
  output dir), `ES_GROUP` (decode layers per NEFF for `dprofG_B` profiling variants), `ES_DUMP=dir ES_POS=p`
  (write a variant's zero inputs as .npy + print the neuron-explorer ifmap instead of compiling).
- `src/engine.cpp`: loader (MXFP4 -> bf16, exact), layouts, NEFF loading, plans, protocol. Layer tensors are now
  one device tensor per name stacked over the 24 layers (`LAYER_BYTES`), with per-layer slices
  (`nrt_tensor_allocate_slice`) for the per-layer prefill NEFFs. The expert stacks `wgu`/`wd` are split in two
  (layers 0-11, 12-23) because one 24-layer stack exceeds 2^31 elements and DMA offsets past that are out of range.
  Dev-only ops: `bench` (time n queued executions of one NEFF: `step<i>`, `head`, `pmoe<i>`, `pattn_s<i>`,
  `pattn_f<i>` with `layer`), `read` (global tensors and `L<l>_<name>` layer tensors; NaN-safe). Env:
  `ENGINE_NO_WARMUP`, `ENGINE_PF_LAYERS` (debug).
- `tools/dev/`: the session's measurement and debugging tools (profiling with neuron-explorer, in-engine layer
  checks against the numpy reference, bench op driver); see its README.
- Load-time warmup: every NEFF is executed once during `hello` (prefills of 256, 512, 600, 1024, 1500, 2048, 3072,
  4096, 4600 tokens + 2 decode steps each). Without it the first request that touches a NEFF pays a large
  first-execution cost (measured in dev-eval 001: 256-token prefill 92.6 ms instead of 40 ms).

## Parallel split (unchanged)

TP = 4 logical cores x 2 physical = 8 vcores (g = 2*rank + pid): q heads 8g..8g+7, KV head g, expert
intermediate slice 360g..360g+359, LM head vocab slice. Decode: two all-reduces per layer (sendrecv between the
physical cores + `all_reduce` over the 4 ranks).

## Prefill MoE (`prefill_moe`), how it works

1. route: one-hot of the 4 selected experts per token, exclusive prefix sum over tokens by matmul with a strict
   triangular matrix (exact integers in fp32) -> each (token, slot) pair's row in a per-expert list `rec[e*CAP + k]`
   = (token, slot row i*MAXPOS + t, routing weight), written by DMA scatter (`vector_offset` on the destination,
   `oob_mode.skip` for padding tokens). Unused rows are pre-marked out of range (one contiguous DMA).
2. experts: (a) a STATIC pass over the first 128-row block of every expert, weights double-buffered (no device loop
   -> the next expert's weights stream during this expert's compute); (b) ONE device loop (`nl.dynamic_range`) over
   a list `blist` of all remaining blocks (built on device), 2 x 128 rows per iteration, software-pipelined
   (gather + transposes of block j+1 beside the GEMMs of block j, scatter one iteration late); an expert's weights
   load (dynamic offset) only on its first listed block (others get an out-of-range offset and are skipped).
   Per block: gather the tokens' normalized activations (`xtm`, token-major, written by the attention NEFF),
   24 PE transposes, gate/up token-major (activations stationary, weights streaming, 48 matmuls), bias, SwiGLU,
   x routing weight, 3 transposes, down token-major (18 matmuls), scatter bf16 rows to `ybuf[pid][slot][t]`.
3. combine: per token sum of the 4 slot rows (fp32), core pid finishes tiles j % 2 == pid (+ other core's partial),
   ONE all-reduce over the whole prompt, + bias term (token-major matmul R @ bd), bf16 transposes, residual -> hT.

Numerics vs session 001: slot-order sum instead of expert-id order, and each vcore's partial expert output rounded
to bf16 before the sum (the reference rounds each expert's output to bf16 too). Layer-0 check against the numpy
reference (tools/layout.py, 512 tokens): median rel. error 0.0029 vs 0.0027 for the dense session-001 path.

## Session 002 results

### Verification
- Fused decode: `same.py` (40 cases, every token) identical to the session-001 engine (logs/same-001.json).
- Prefill layer 0 in the engine vs the numpy reference (/tmp tool test_l0.py: ENGINE_PF_LAYERS=1, `read hT`):
  identical error figures for the dense, sparse and multi-chunk versions (after the bugs below were fixed).
- `enginesmith dev-eval` 001 (sparse MoE, before warmup and multi-chunk attention): PASSED, mismatching steps
  0 / 2 / 0, score 0.0273.
- dev-eval 002 (+ warmup, multi-chunk attention): PASSED, 0 / 2 / 0, score 0.0357 (prefill 40.2 / 96.1 / 308.7 ms,
  decode 460 / 468 / 489 ms).
- dev-eval 003 (+ flash attention for full layers with >= 8 chunks; clean build from scratch in 8.0 min, 27 NEFFs, two
  4-core build groups): PASSED, 0 / 2 / 0, score 0.0363 (prefill 40.1 / 96.1 / 283.3 ms).
- Flash attention everywhere + split phase 3: layer-0 error (512 tokens) median 0.0043 (was 0.0029), layer-1 error
  (2048 tokens, tools test_l01) median 0.0117 / p99 0.083 (was 0.0101 / 0.055): nkilib keeps the scores in fp32 where
  the reference (and the older kernel) round q.k to bf16. ab.py: 1024 96.8 -> 93.9, 2048 161.9 -> 151.9, 4096
  284.6 -> 259.4 ms.
- dev-eval 004 (flash attention for all multi-chunk layers, split phase 3; the engine as left by session 002):
  PASSED, 2 rounds, mismatching steps 0,0 / 0,0 / 0,0, score 0.0372 (prefill 39.8-40.3 / 91.3-91.6 / 254.3-255.1 ms,
  decode 460.6-461.0 / 468.5-468.8 / 489.9-490.0 ms; session 001: 0.0238).
- Long output: 200-token prompt + 4000 tokens, + 407, + 1 (cache slot 4607), 4500 + 100: every token returned.

### Measured (harness dev-eval 004 for the final engine; session 001 from its dev-eval, ms)
| case | prefill 001 -> 002 | decode (64 tokens) 001 -> 002 |
|---|---|---|
| 256 | 40.4 -> 40.1 | 692 -> 461 |
| 1024 | 156 -> 91.5 | 701 -> 469 |
| 4096 | 684 -> 255 | 693 -> 490 |

Steps of the session (prefill ms at 256 / 1024 / 4096 unless noted):
- Fused decode step (24 layers + head in one NEFF): decode 669 -> 461 ms / 64 tokens (1.45x), tokens identical.
- Sparse MoE v1 (per-expert device loops): 77 / 138 / 424 (256 regressed: per-loop barriers).
- + double-buffered weights, x-stationary gate/up, pipelined loop, contiguous bias/rec DMAs, static first block,
  flat heavy-block list, split combine: MoE kernel at 1024 tokens 3.9 -> 2.6 ms; 52.7 / 112.6 / 395.7.
- + bf16 slot rows, dense path for one chunk: 40.2 / 108.2 / 371.7.
- + multi-chunk attention NEFF (exact causal key ranges, weights once per layer): 40.3 / 96.8 / 309.3.
- + whole-sequence flash attention (nkilib) for full layers with >= 8 chunks: 4096 309 -> 287 (dev-eval 003: 283).
- + flash attention for every multi-chunk layer, phase 3 split by token halves over the two physical cores with
  one all-reduce per layer: 40 / 91.5 / 255 (dev-eval 004).

### Where prefill time goes now (measured, neuron-explorer / bench op, 4096 tokens)
- MoE kernel ~6.5 ms per layer (real routing: very skewed, the top expert gets ~4058 of 4096 tokens, top 5 ~85% of
  pairs): route ~0.45, static first blocks 1.25 (weight-DMA bound), heavy-block loop ~3.3 (PE ~65% busy, ~26 us per
  128 rows), combine ~1.4.
- Attention per layer (before flash attention): 16 chunks in one NEFF; per chunk ~300 us, latency-bound (16
  per-head chains). After: bench op not re-run; neuron-explorer single executions 3.8 ms (sliding) / 4.3 ms (full)
  per layer at 4096 tokens, of which nkilib attention 0.73 / 1.34 ms.

### Bugs found (worth remembering)
- Stacked 24-layer expert tensors exceed 2^31 elements: DMA offsets past that are out of range (OOB at
  run time only from layer 17 on). Split the stacks.
- 4-D stacked KV caches: the position-indexed DMAs must name the position dimension in `indirect_dim` (3 for K^T
  [G,2,D,T], 2 for V [G,2,T,D]); with indirect_dim of the old 3-D layout the zero-input build passed (pos 0) and the
  engine failed with NRT_EXEC_OOB.
- `ybuf` is per logical core: the two physical cores must write disjoint halves (first version overwrote each
  other's partials; layer error 0.28).
- Score chunking `CH = 512 if NK % 512 == 0 else 256` silently skipped the last 128 keys for NK = 384 (stale SBUF
  -> NaN rows that varied run to run). Use the largest of 512/256/128 dividing NK, assert it.
- A strided 4-byte-element broadcast DMA (gate/up bias row from [n, t] order) took ~100 us per expert; a host-side
  transposed copy (`bgut`) made it one contiguous row.

### Tried and rejected (measured)
- Decode: prefetching the next layer's attention weights during this layer's attention: 461 -> 487 ms / 64 tokens
  (slower; the queue already overlaps the next QKV load with the combine phase, and the early DMA competes with
  this layer's weights).
- Prefill MoE: per-expert dynamic loops: each loop entry and iteration is an all-engine barrier that also waits for
  outstanding DMAs, so a weight prefetch cannot overlap a device loop (issuing it inside iteration 0 only stretched
  that iteration: 25 -> 48 us). 4 x 128 rows per loop iteration instead of 2: 7.27 -> 7.42 ms (more padding).
- Splitting expert weight DMAs into 36 smaller ones: no gain by itself (the slow part was the bias DMA).
- Pipelining chunk c's attention ahead of chunk c-1's post-attention part: 1.45 -> 1.456 ms (no gain; chunks are
  latency-bound internally).
- Decode, all measured with ab.py (64-token decode, 256 / 4096-token prompts), session-002 engine as A:
  - expert + LM-head weight DMAs alternated over the Sync and Scalar hardware DGE queues: 461 -> 463, 490 -> 494.
  - x-stationary QKV and o_proj (activation column stationary, weights as 512-wide moving operand; 432 -> 96
    matmuls per layer; bitwise identical, same.py 40/40): 461 -> 461, 490 -> 488 (no gain: not on the critical path).
  - next layer's attention weights issued from the Scalar engine's DGE queue right after this layer's softmax:
    461 -> 500 (slower).
  - the same on the Sync queue, gated behind a 1-element DMA of this layer's first norm: 461 -> 465 (no gain).
  Profile of the full 24-layer step (neuron-explorer, steady-state layers): ~267 us per layer = ~80 us attention
  chain (DMA idle: KV write -> read, attention, o_proj, all-reduce, router) + ~150 us streaming 31.6 MB at
  ~210 GB/s (the 4 experts' 25 MB + the next layer's 6.6 MB attention weights) + ~35 us combine / all-reduce.
- Prefill attention phase 1 in 512-token blocks instead of 256: neuron-explorer 3.82 -> 3.62 ms per sliding layer,
  end-to-end 91.6 -> 91.8 / 254.8 -> 255.1 ms (no gain; not kept).

- Prefill attention: standalone nkilib attention_cte at 4096 tokens (16 q / 2 kv heads per logical core, d 64,
  neuron-explorer): 0.73 ms sliding, 1.34 ms full; the per-chunk work around it (QKV, o_proj, all-reduce, router)
  was ~280 us per chunk and is now the larger part.

## Next steps after session 002 (estimates; see "Session 003" for what was done)
1. Prefill MoE (~60% of prefill at 1024 and 4096): the heavy-block loop (PE ~65% busy; fewer PE instructions per
   block, e.g. a DMA transpose for the gathered activations instead of 24 PE transposes), the static first-block
   pass is bound by streaming all 32 experts' weights (199 MB per physical core per layer, ~1 ms), combine ~1.2 ms
   at 4096 tokens.
2. Decode (~7.2 ms/token, see the profile above): the 80 us attention chain per layer is latency (several small
   dependent DMAs on the GpSimd queue: KV-slot write -> cache read); reading the cache before the write and adding
   the current token from SBUF would shorten it (changes rounding order). MXFP4 on device would cut expert bytes
   3.8x but needs an on-chip decode cheaper than the bytes saved.
3. Prefill attention phase 3: router / top-4 runs as ~20 small vector ops per 128 tokens.
