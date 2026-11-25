# DeepSeek-V4.1-Flash engine on trn2.48xlarge — notes

## Round 008: per-token work split over the physical cores, shared expert in the attention half, host waits

Base: round 007 (commit 531f2d5).  Every change is bitwise: the same instructions on the same values, only which
physical core runs them (LNC=2), which NEFF, or host-side scheduling.  Verified by token streams
(`tools/lab/run.py --set std`: 21 prompts, lengths 1 .. 4097, split requests + 64 decode tokens) against the
round-007 binary: identical for every build below; the final build also on 8100 + 64, 200 + 1500 and 4100 + 2000
tokens (`/tmp`-style driver as in round 006: identical; 8100 prefill 4390 -> 3771 ms, decode 47.5 -> 40.5 and
50.7 -> 42.6 ms/token on the two long decodes).

Summary (lab, final build vs round 007, one process each): prefill 256 / 1024 / 4096 = 229 / 553 / 1881 ms (round
007: 250 / 593 / 2081), decode ~2.55-2.72 s per 64 tokens (round 007: 2.98-3.21).  dev-eval 004 (final): score
74.45 (round 007's dev-eval on the same prompts 68.06); table at the end of this section.  A clean-tree build
(no target/ cache) was token-identical on the 21 prompts (NEFF bytes differ between fresh compiles).

1. **Per-token work of the 4-token attention half runs on one physical core, not both** (`lib.tok_mine`,
   `lib.exchange_tok`).  Profiles showed that under LNC=2 every non-GEMV step of `layer_impl_m` (indexer, sparse
   attention core, hc_pre / norms / act_quant / RoPE, hc_post / ffn_norm) ran redundantly on both physical cores:
   in `layer4_i1_1024` the indexer (~235 us) and the attention core (~215 us) ran with the DMA idle, twice.
   Now core 0 owns tokens 0-1 and core 1 tokens 2-3 for those steps; the results the GEMVs (split by output tiles,
   unchanged) need on both cores are exchanged with `nisa.sendrecv` (x / xn / qr / kv rows, xf before the
   partition-batched router; `out_proj_m` swaps the attention outputs' head halves, since wo_a groups 0-3 / 4-7 live
   on cores 0 / 1).  Indexer selections and residual rows are written by the owner only.
   Device time (neuron-explorer, zero inputs): layer4_c2_1024 808 -> 681 us, layer4_i1_1024 1107 -> 818,
   layer4_w 739 -> 618 (first step, indexer + attention core only: 728 / 886 / 693).
2. **The group's shared expert runs at the end of the attention half** (`k_layer.shared_m`, helper core) instead
   of in the home core's `moeu4_S` (which now takes its bf16 output `s2in`, carried in the M4 transfer: new slot
   M4_S2).  Same instructions on the same bf16 ffn inputs.  The helpers had 0.7-0.9 ms slack per group after (1),
   the home MoE had become the bottleneck: moeu4_16 1402 -> 1292 us, layer4_* +~120 us.
3. **moeu: per-token prologue / epilogue on the owner core** (act_quant of x, hc_post + store): moeu4_16
   1292 -> 1272 us.
4. **Idle pipeline-stage threads wait with umonitor / umwait instead of pause loops** (`idle_wait` in
   `Queue::pop`; build flag -mwaitpkg; lab `DSV4_SPIN_PAUSE=1` restores the pause loops, `DSV4_UMW_X=1` /
   `DSV4_UMW_P=1` also use it for the expert workers' queues / the expert-phase `pending` waits).  In decode ~40
   stage threads wait at any moment; their pause loops slowed the threads inside runtime calls (micro-benchmark
   `tools/lab/mb/htspin.cpp`: a schedule call 18.6 us alone, 30.4 us with 89 spinning threads on the socket).
   Host-only, tokens identical.  First version (umwait in all three waits): decode 46.7-50.3 -> 42.8-44.7 ms/token
   (xin write 28 -> 15 us, attention schedule call 70 -> 54 us), **but the first prefill requests after idle got
   slower and recovered over successive requests** (p256 of 4 successive requests: 282 / 263 / 249 / 231 ms vs
   250 / 234 / 232 / 235 with pause loops; dev-eval 002 p256 279.9 ms vs 240.0) -- presumably the cores that sat in
   C0.1 run slower at first (not resolved).  Per wait (4 requests p256 / p1024 prefill, decode ms per 64 tokens,
   one process each):

   | umwait in | p256 | decode |
   |---|---|---|
   | none (round 007) | 250 234 233 231 | 2967-3131 |
   | workers only | 307 244 242 263 | 2960-3128 |
   | pending only | 261 240 238 234 | 2928-3129 |
   | workers + pending | 280 239 241 254 | 2958-3118 |
   | stage queues only (kept) | 258 234 231 235; 254 232 232 245; 243 239 239 236 | 2549-2846 |
   | stage queues + pending | 272 236 233 230; 262 249 232 233 | 2551-2764 |
   | all three | 275 255 258 233; 284 270 245 243 | 2539-2735 |

   With the self warm-up (6) in place, the pending waits were measured again (2 x 4 requests each): stage queues
   + pending decode 2626 vs 2675 ms (-1.8 %), p256 233.5 vs 233.6, p1024 553.8 vs 555.4: **kept (final: stage
   queues + pending)**; + workers p256 244.8 (first 258 / 243), p1024 560.9: not kept.
   Also tried and not kept: umwait C0.2 (same as C0.1); a pause-spin window before umwait (0.5 / 2 / 6 ms: the
   first prefill still 271-280 ms); pause loops during prompt requests and umwait only during continuation
   requests (first p256 310 / 278 ms: worse); tpause in the completion polls (`Call::wait`) and umwait for the
   sw / cw counters (no gain).
5. **Decode expert placement: a routed expert goes to the home core only when two would share a core**
   (`home1_min`, lab `DSV4_HOME1=1` = always as before, `DSV4_HOME1_MIN=1` = also for a shared device).  After (4)
   the home expert (queued behind the shared expert) had become the last to finish (331 us after the push; remote
   ones 178-281).  Decode ms per 64 tokens, 4 runs each from alternating processes (256 / 1024 / 4096): always
   home (round 007 rule) 2768 / 2797 / 2881; never 2723 / 2755 / 2752 (-1.6 / -1.5 / -4.5 %); only on a shared core
   (kept) -1.5 / -2.1 / -0.6 % vs never; also on a shared device +0.2 / +0.9 / +1.3 %.  Tokens identical (the same
   expert NEFF, the combine sums in expert order).  (Measured with umwait in all three waits.)
6. **Self warm-up at load** (`self_warm`, lab `DSV4_NO_SELFWARM=1`): two real requests (300 tokens + 8, 1100 + 2)
   before `hello` answers, then a reset.  The first prompt requests of a process were slower than later ones in
   every build (round 007: p256 of 4 successive requests 242-250 / 234 / 233 / 231 ms; with (4) 252-300 / 232-241 /
   ...), and the harness times the request right after its one warm-up.  Running every NEFF call once at load
   (moeu S x buffer, layer4 variants, decode variants: 0.1 s) did not remove it (first p256 253 / 272 vs 300 / 259);
   the real requests do: first p256 231 / 230 vs 252 / 252 without (4 successive requests, same build otherwise).
7. **Decode: the Engram rows / kv launch of a step after the step's job is queued** (`run_tokens`; the Engram stage
   of layer 1 waits for `ekv`): the 2 x 24 table rows + scales a token gathers from the mmap'd Engram tables fault
   in ~96 pages, which the main thread did before layer 0 could start.  Decode -3 % (2539-2693 vs 2569-2823 ms per
   64 tokens, together with (6) and (8)).
8. **Prompt Engram rows gathered ahead** (`prefetch_engram`: 8 threads at the start of a prompt request fill a
   pre-touched buffer; the Engram stages copy from it, lab `DSV4_NO_EPREFETCH=1`): the same bytes; ~25k page faults
   per 256-token prompt move out of the Engram stages.  No measurable prefill change at 256 / 1024 (kept: the
   Engram stages were close to the period, 1.7 ms per group, in round 007's p1024 profile).
9. malloc: per-job host buffers kept in the heap (`M_MMAP_THRESHOLD` 32 MB, no trim; lab `DSV4_MALLOC_DEFAULT=1`):
   no measurable change (the page faults were the Engram gathers, see (8)).
10. **Stage threads never block** (with umwait the stage queue wait has no budget; lab `DSV4_STAGE_SLEEP=1`):
   the MoE-stage threads sleep through every decode request (their 83 ms spin budget expires), and the first
   groups of the next prompt paid a futex wake-up at each hand-off.  Stage trace of a prompt's first group, per
   layer (us): attention-out -> MoE-stage arrival 392 -> 3, MoE out -> next layer 283 -> 48; the first group leaves
   layer 39 after 117 ms (153 before, p1024 trace of the (1)+(2) build).  End to end only ~1 % (p256 230 / 231 vs
   233 / 232 / 230 ms): the rest of a 256-token prefill is the throughput phase (~1.63 ms per group), where helper
   pairs (2 x ~0.8 ms) and the heavy home MoEs are both ~fully busy.

Ideas left (round 009+):
- The prefill pipeline is now balanced at ~1.65 ms per 4-token group: helper pairs (2 x ~0.8 ms device), the home
  MoE of the layers with the most distinct experts (S ~20: moeu ~1.6 ms) and the Engram stages are all close.
  More needs less work per token everywhere at once: 8-token groups (GEMV tiles shared by 8 tokens: the attention
  half's streaming is ~45 % of its device time; distinct experts per 4 tokens 16.2 -> 12.9) need a cheaper moeu8
  VectorE path (the block-scale products of padded token columns; e.g. GpSimd taking part of the multiply) and the
  host pipeline generalised to M = 8.  A first group's per-layer latency still has ~450 us of host work (A stage
  arrive -> launch 263 us: step4 write + schedule call; MoE arrive -> launch 189 us).
- Decode (~40.5 ms/token): per layer ~575 us attention device + ~47 us model-switch penalty (the layer NEFF always
  follows other models on its core) + expert phase ~300 us (6 remote expert1, completions ~23 us apart) + host.
  The runtime costs (schedule ~50 us, completion stagger) are the bulk of what is left; the decode NEFF's
  non-streaming parts are latency chains (engines mostly idle), so splitting them over the two physical cores
  like the prefill parts would gain little (estimated <= 25 us per layer for the attention core).

Lab timings (one process per column, `tools/lab/run.py --set std`, prefill ms / decode ms per 64 tokens):

| prompt | round 007 | (1) | (1)+(2) | (1)-(5), umwait in all waits | final (1)-(9) |
|---|---|---|---|---|---|
| 256 (first request) | 250 / 2975 | 251 / 2954 | 247 / 2989 | 292 / 2740 | 230 / 2662 |
| 1024 | 593 / 3073 | 594 / 3161 | 563 / 3092 | 582 / 2746 | 553 / 2716 |
| 2048 | 1068 / 3212 | 1060 / 3264 | 991 / 3170 | 991 / 2670 | 981 / 2713 |
| 4096 | 2081 / 3150 | 1970 / 3238 | 1909 / 3190 | 1917 / 2816 | 1884 / 2821 |

Tokens of all builds identical on the 21 prompts.

Where the prefill time goes now (stage trace, p1024): period ~1.7 ms per group; the home MoE of the layers whose
groups route to the most distinct experts is the bottleneck: mean distinct experts per 4-token group by layer
(1024 + 4096 prompts) ranges from 10.7 (layer 18) to 21.3 (19) and 20.9 (0); layers 0, 15, 19, 23, 25, 27, 29,
31, 34 are at 19.3-21.5 (moeu ~1.6-1.9 ms).  Helpers have 0.2-0.9 ms slack.

Measured and rejected / findings this round:
- Runtime: an execution after >=150 us of idle costs ~+33 us (139 vs 172 us for expert1; threshold between 100
  and 150 us); polling the runtime or a tiny NEFF 40-100 us before does not avoid it.  A model switch on a core
  costs more: layer_c2_1024_p after shared1 631 vs 584 us; after another load of the same NEFF 616 (lab
  `tools/lab/mb/switch.cpp`, `idle.cpp`).  NEURON_RT_DBG_EARLY_POSTING / EARLY_COMPLETION / ENABLE_BRANCH_PREFETCH,
  NEURON_RT_INSTR_FETCH_ON_H2D, NEURON_RT_DISABLE_PACK_DESCRIPTORS: no effect on the switch cost.
- In the engine the decode schedule calls take 2-3x their isolated time (sched layer 70 vs 21 us); not the number
  of loaded models (`mb/nmodels.cpp`: 22.5 us with 1 or 150 models on the core) but CPU contention (item 4).
- Prefetching 8 wo_a tiles of this core's first group before the 4-token attention core: c2 802 -> 800, i1 943 ->
  955, w 736 -> 750 us (no gain, removed).
- Router GEMV with x stationary and the router weights moving ([M, 384] rows, then transposes to the partition
  layout): layer_c2_1024_p 566 -> 573 us, layer4_c2_1024 802 -> 811 (slower, removed).
- `nrt_execute` from 6 threads instead of queued executions: the calls serialize (~40 us apart), 6 experts done at
  237-435 us instead of 203-266 (`mb/conc.cpp` mode 5).
- Helper table pairing distant layers (3+19, 4+21, ... one ratio-2 and one ratio-1 layer per core, `DSV4_ATTMAP`;
  aimed at the fill: the first group of a prompt needs 3.55 ms per layer, 153 ms through 40 layers, because a
  group's layer-(l+1) attention queues behind the next group's layer-l attention on the shared helper): p256 -1.3 %,
  p1024 -0.5 %, p4096 0, decode +1 % (noise level; 2 x 2 processes).  Not adopted.
- 8-token MoE groups would cut the distinct experts per 4 tokens from 16.2 to 12.9 (routes of 1024 + 4096
  prompts), but round 007 measured moeu8 VectorE-bound (no faster per token) and the helpers would need 8-token
  attention as well; not attempted.
- The 4 single tail tokens of a prompt of length 4k are pipelined behind the last group: lengths 4k-3 / 4k / 4k+1
  have the same prefill time within noise (253/256/257: 273/252/243, 241/245/249 ms).

Dev-evals this round (1 round each, public prompts; round 007's own dev-eval on these prompts: 68.06, prefill
118.26 / decode 39.17):

| dev-eval | build | score | prefill mean | decode mean | prefill ms 256 / 1024 / 4096 | decode ms per 64 tokens | mismatches |
|---|---|---|---|---|---|---|---|
| 001 | (1)-(2) | 69.67 | 122.10 | 39.76 | 240.0 / 560.5 / 1908.8 | 2980 / 3036 / 3199 | 5 / 2 / 12 |
| 002 | (1)-(5), umwait in all waits | 71.03 | 115.19 | 43.80 | 279.9 / 596.5 / 1899.1 | 2705 / 2756 / 2822 | 5 / 2 / 12 |
| 003 | (1)-(9) without the pending umwait and (10) | 73.36 | 122.88 | 43.79 | 236.8 / 557.1 / 1895.5 | 2684 / 2705 / 2779 | 5 / 2 / 12 |
| 004 | **final** (1)-(10) | **74.45** | 123.02 | 45.06 | 234.4 / 554.7 / 1884.3 | 2564 / 2673 / 2818 | 5 / 2 / 12 |

The mismatch counts are those of rounds 002-007 on these prompts (the arithmetic is unchanged; p4096 is 1 below its
cap, a property of that prompt).

## Round 007: round 006's engine + ascending expert order + deduplicated prefill MoE + helper remap

Base: round 006's code (commit 57c7ca3, rejected only for one high-mismatch p1024 stream of its formal evaluation;
its arithmetic is round 005's: verified here again, tokens identical to a round-005 binary at 5 / 256 / 1027 /
4096 + 64).  Changes, each verified by token streams (lab `tools/drive.py`, `tools/lab/run.py --set std`: 21
prompts, lengths 1 .. 4097, split requests + 64 decode tokens):

1. **Routed experts summed in ascending expert id order** (host `sort6`, decode spread / moe / moe4 paths).  The
   official `MoE.forward` loops over the experts, so each token's fp32 `y` receives its 6 expert outputs in
   ascending id order; rounds 001-006 summed in router rank order.  A summation-order change (not bitwise to
   round 006): on 12 lab prompts (6 x 256, 4 x 1024, 2 x 4096 + 64) 11 streams identical to the rank order, one
   differs from step 6; reference judge of that pair: 7 (ascending) vs 6 (rank) mismatching steps, i.e. neutral.
   Kept because (2) needs it.  `DSV4_RANK_ORDER=1` (lab) restores the old order.
2. **Deduplicated 4-token MoE** (`kernels/k_moeu.py`, NEFFs `moeu4_S`, S = 9..22, 24): a 4-token group routes 24
   (token, expert) pairs to only 16.2 distinct experts on average (measured, lab routes of 1024 + 4096 prompts:
   histogram 9..24, mode 16).  Each distinct expert is a slot (ascending id) applied to all 4 tokens at once
   (moving operand [128, 16]: the tokens' block-masked x), routing weight 0 for tokens not routed there (exact
   zeros added).  Per token the instructions and rounding of moe4 (same block sums, scale products, 160 / 72-element
   reductions, swiglu, act_quant per token row; permuted copies only).  Tokens identical to moe4 with sorted
   experts on all 21 + 4 lab prompts.  Device time (neuron-explorer, zero inputs): moe4 1842 us, moeu4_16
   1403 us (first version with per-token instruction loops: 1661 us; the M-batched access patterns fixed it:
   VectorE bound).  `DSV4_NO_MOEU=1` runs moe4.
3. **Helper remap**: layers 0 / 1 (window-only) moved from the Engram cores 40 / 41 to 63 / 62, layers 38 / 39 from
   56 / 57 (shared with sources 2 / 8) to 40 / 41.  Found with the stage trace (`DSV4_TRACE`, /tmp/head.py-style
   breakdown arrival -> launch -> done -> out per stage): the pipeline head (layer 0's attention) waited 1.7 ms for
   a 0.74 ms NEFF behind the Engram-14 stage on core 41, and cores 56 / 57 (source + normal layer) were saturated.
   Period per 4-token group 2.05 -> ~1.9 ms; now saturated: cores 62 / 63 (index source + window layer).
   Lab override `DSV4_ATTMAP=l:core,...`.

Lab timings (one process each, prefill ms at 5 / 256 / 1027 / 4096, decode ms per 64 tokens ~2950-3200 for all):
round 005 binary 135 / 318 / 849 / 2996; round 006 159 / 265 / 620 / 2232; + ascending 165 / 267 / 608 / 2226;
+ moeu 148 / 244 / 612 / 2208 (std set: faster at every length, e.g. 2048: 1085 vs 1137); + remap 152 / 254 /
595 / 2080.

**dev-eval 001 (this round's final engine, 1 round, public prompts): PASSED, score 68.06** (prefill mean 118.26,
decode 39.17; round 005's best 55.80).  Prefill 250.8 / 592.3 / 2085.0 ms, decode 2962 / 3037 / 3157 ms per 64
tokens at 256 / 1024 / 4096; speedups 173.0 / 143.9 / 66.4 (prefill), 43.4 / 37.7 / 36.7 (decode).  Mismatching
steps 5 / 2 / 12 (the same counts as rounds 002-006 on these prompts; p4096 is 1 below its cap, a property of that
prompt); controls worst 8 (p5).

Where the prefill time goes now: period ~1.9 ms per 4-token group, set by helper cores 62 / 63 (index source 32 /
36 + window layer 1 / 0); every other helper (normal-layer pairs ~2 x 1.0 ms, sources ~1.2-1.5 ms) is close, the
home cores (moeu ~1.4 ms + ~0.15 ms host) have 0.4-0.6 ms spare.  Fill (40 x per-layer latency ~2.6 ms) is ~40 %
of p256.  Decode unchanged (~46-49 ms/token; see round 006).

Measured and rejected:
- Other placements of the window layers 0 / 1: on 56 / 57 (with s2 sources 2 / 8): 247 / 591 / 2115 ms; on 58 / 59
  (with 14 / 20): 4096 2311 ms; kept 63 / 62 (254 / 595 / 2080).
- **Home share of the attention halves** (`init_home_share`, lab `DSV4_SHARE=1`): 1 of 3 groups' attention of
  one layer per helper pair (2 of 5 for 38 / 39, 1 of 3 for 0 / 1) on the home core, which moeu left idle
  0.5-1.0 ms per group; window ring copied between the sites on each switch (25 us per copy).  Tokens identical,
  but slower: 269 / 656 / 2220 ms (256 / 1027 / 4096) vs 244 / 612 / 2208; only layers 0 / 1 shared: 255 / 693 /
  2373.  Mixing two pipeline stages in one core's in-order queue adds their latencies.
- **8-token groups**: `layer8_*` (layer_impl_m(8)) device time: c2_2560 1265 us (vs 845 for 4 tokens), w 1035
  (741), but s2_4096 2405 (1360), s1_8192 4018 (1937), i1_8192 2325 (1364): the source / indexer layers scale
  linearly, so their helper cores would become the bottleneck.  moeu8_26 (8 tokens, 26 slots, all tokens per
  slot): 2825 us vs 2 x 1842, VectorE 93 % busy (the block-scale products of the padded token columns).  Not
  integrated.
- Runtime serialization hypothesis (the period stayed ~2.05 ms through device changes): rejected by
  micro-benchmark (`tools/lab/conc.cpp`): 64 threads x 64 cores running a tiny NEFF back to back 96 us per
  iteration (1 thread 64 us); moeu4_16 schedule call 34 / 38 / 46 us with 1 / 16 / 40 concurrent.  In the engine
  the schedule calls average 89-97 us (CPU contention of ~150 threads), but the period was set by core placement.

## Round 006: prefill attention on helper cores

**dev-eval 002 (this round's final engine): PASSED, score 68.64** (prefill mean 115.94, decode 40.64).  Prefill
253.7 / 597.5 / 2212.2 ms, decode 2997 / 3108 / 3204 ms per 64 tokens at 256 / 1024 / 4096; mismatching steps
5 / 2 / 12 (unchanged streams).  Also checked against the round-005 binary: an 8100-token prompt + 64 tokens
(prefill 6012 -> 4667 ms) and a 200-token prompt + 3000 decode tokens: identical tokens, no errors.

**dev-eval 001 (commit 'lab: DSV4_TRACE ...', the helper version without item 3): PASSED, score 65.92** (prefill
mean 111.37, decode 39.01; round 005's best 55.80 / 82.89 / 37.56).  Prefill 258.3 / 616.4 / 2293.5 ms, decode
2993 / 3023 / 3145 ms per 64 tokens at 256 / 1024 / 4096; mismatching steps 5 / 2 / 12 (the same streams as rounds
002-005: the arithmetic is unchanged).

**1. Prefill: every layer's 4-token attention half runs on a helper core; the home core runs only the MoE.**
Before, a layer core did layer4 (~0.85 ms device) + moe4 (~1.84 ms) per 4-token group, so the pipeline stage was
~2.7 ms while 14 cores (42-55) were idle in prefill and the 8 twins only helped 8 of 40 layers.  Now (`Att4`,
`att_core`): cores 40..63 hold copies of the dense attention weights of one or two layers each (table in
`att_core`; Engram cores 40 / 41 also take layers 1 / 0), their own window ring, compressed-row caches and step
tensors.  Pipeline stages per layer: A_l (helper: layer4 NEFF, read its outputs) -> M_l (home: write x | h2 | rw,
bind the 24 experts, moe4, double-buffered as before).  Same NEFFs, same inputs: bitwise the same.
- The host writes every compressed row / index key to both copies (home for decode, helper for prefill groups).
- Before the first single-token step after groups (the prompt's tail tokens, then decode run on the home core as
  before) the helper's window ring is copied to the home core (`sync_att`, 128 KB per layer).  Stale ring slots and
  cache rows of earlier requests are never valid (the attention list marks only this request's positions).
- Single-token jobs skip the MoE stage when no group is in it (`m_inflight`), so decode has no extra hop.
- Twins are off (their 115 GB now hold more decode replicas: 9604 second replicas instead of 6806).
- Lab (one process, nonce prompts, 64 decode tokens), base = round 005 binary:

  | prompt | base prefill ms | this | tokens |
  |---|---|---|---|
  | 256 | 318 | 272 | identical |
  | 1024 | 835 | 617 | identical |
  | 4096 | 3020 | 2220 | identical |
  | 5 / 258 / 1027 / 4099 | 140 / 310 / 837 / 2995 | 159 / 268 / 623 / 2232 | identical |

- Stage trace (`DSV4_TRACE`, `tools/tra.py` in the lab): every stage now completes a group every ~1.75-1.95 ms;
  the home moe4 executions are queued back to back (the next launch happens before the previous completes), i.e.
  the home MoE (~1.8 ms device per group) is the bottleneck, and the total device work per group (~40 x 1.8 MoE +
  ~40 x 0.9 attention + Engram) over 64 cores is ~1.75 ms: the load is balanced; more needs faster kernels.
- Tried: queueing a helper's next group before reading the previous group's outputs (double-buffered helper step
  tensors): slower (1024: 686 vs 617 ms, 4096: 2299 vs 2220), the outputs reach the MoE stage later.

**2. Decode: the next layer's packed input is written by the workers; one routed expert on the home core.**
The expert workers write their outputs straight into the next layer's `pk_in` (y6 slot j) and the layer thread
writes h2 | fmix | s2 there once the shared expert is done, so the next stage only launches its NEFF (no 112 KB
write on the critical path).  One routed expert per step is queued behind the shared expert on the home core (the
worst-placed one: sharing a core or device with another), so 5 run remotely.  Lab decode ms/token (3 lengths, two
processes each): base 50.3, this 49.3 (pk_in writes alone 49.6); tokens identical.

**3. One host transfer per group and layer between helper and home** (`M4_*` layout: x | h2 | pad | rw | ridx |
cpack in one helper tensor `apack[b]`, x | h2 | pad | rw in one home tensor `min4[b]`; the NEFF outputs / inputs are
slices): 1 read + 1 write instead of 5 + 3.  Tokens identical; prefill neutral within noise (256: 277 vs 282 / 271,
1024: 604 vs 610 / 627, 4096: 2222 vs 2218 / 2226 ms).  Kept (fewer runtime calls).

Ideas left (round 007): 8-token attention only for the normal layers with moeu8 (needs a cheaper VectorE path
for the padded token columns, e.g. host-gathered x columns, C = 4 per slot: ~2.35 ms per 8 tokens estimated) and a
placement that keeps sources alone; a split MoE for the fill phase.

Ideas left: the home moe4 (~1.8 ms device per 4-token group, latency-bound: TensorE ~68%, sync-engine DMA issue
~64%, periodic 1-2.5 us TensorE stalls without a semaphore wait every ~7 us that coincide with GpSimd
`qSyncSpillReload` handshakes -- probably instruction-stream refills of the ~110k fully unrolled TensorE
instructions; device loops cannot index matmul operands, see the backend notes) is now the prefill bottleneck, and
the total device work per group is balanced over the 64 cores; the fill (~40 x 3 ms per layer) dominates p256.

Measured and rejected this round (all bitwise-neutral ideas):
- Decode attention NEFF (`layer_c2_2560_p`, 573 us device; streaming phases run at 600-680 GB/s, ~200 us are
  gaps): wq_a / wkv prefetched at the start and 12 instead of 8 wo_a tiles prefetched under the attention core:
  585 us (worse); `layer_c1_8192_p` 655 -> 646 us.  The start gap shrank but the attention-core gap grew.
- moe4 program order (pair i-1's W2 issued before pair i's swiglu / act_quant chain): 1840 -> 1845 us.  The
  profile shows the TensorE ~68% and the VectorE ~50% busy, the weight DMA at ~270 GB/s per physical core (the
  bf16 attention streams reach ~340): latency-bound, cause not found.
- 4-token attention: 8 wo_a tiles prefetched under the attention cores: 846 -> 842 us.
- Packing the attention NEFF's ~36 inputs: a micro-benchmark (`nrta_execute_schedule` of NEFFs with 1..31 tensors,
  idle runtime) costs 22 us + 0.27 us per tensor, so packing would save ~8 us per layer.  Not done.
- Runtime behaviour measured with micro-benchmarks (lab `mb/conc.cpp`): 6 executions started together on 6 cores
  complete staggered by 25-40 us each (also for 4-us NEFFs; free-running threads do not show it), and an
  execution after ~1 ms of idle costs ~45-75 us more than back to back (not our thread's C-state: spinning
  instead of sleeping, pinning, warm-up NEFFs of another model did not help).  These are the decode expert phase's
  overheads (~360 us for 101 us of device time).

## Round 005: round 003's engine resubmitted, with the cause of its gate failure

**The engine is round 003's code** (commit a8db5ab; rounds 003 and 004 were rejected, round 002 was the base), plus
lab-only diagnostics.  Round 003 was rejected for one step in one stream, not for a defect of its own:

- The formal round-003 evaluation failed at p256 round 1: step 59 of the stream, reference token 223, ours ranked
  8657 (10.8 nats).  Replaying the harness's exact sequence (`lab/exact.py`, seed of rounds/003/evaluation: warm-up
  prompt, then the round's p256 split request) **round 002's binary gives the same 65 tokens as round 003's**,
  deterministic over repeated processes.  So the accepted round-002 engine fails that prompt as well; the
  difference between rounds 002 and 003 was the prompt draw.
- What happens at that step (position 314, input `#` after an EOS / BOS pair in the stream): per-layer dumps of our
  engine (DSV4_DUMP: records 2000+l = ffn input | routes | weights, 3000+l = residual after the attention half,
  4000+l / 5000+l = Engram stage in / out) against the reference's (forward hooks on the official modules):
  layer 0 exact, Engram exact, layers 1-3 drift 0.019 / 0.038 / 0.048 relative (the reference itself at 32 vs 48
  threads: 0.013 / 0.025 / 0.032), then a route flip at layer 3 and our state goes to a different regime from
  layer 4 on (relative error 0.3-0.9 to layer 39; the residual norm grows like a normal position while the
  reference's stays small).  The reference at 32 threads also flips routes in 12 layers there, recovers, and its
  logits agree with 48 threads to 0.04.  Elsewhere in this stream our logits differ from the reference by 1-5, as
  do 32 vs 48 threads of the reference (up to 7.8).  Conclusion: rounding-level chaos at a bistable position,
  no implementation defect found; any engine whose summation order differs from the reference's can hit it.
- So the risk of a rank > 100 step is a property of the shared arithmetic (rounds 002 = 003 = this one), not of
  round 003's changes.

Verification this round: token streams of this build vs round 002's binary on 21 prompts (random tokens, lengths
1 .. 4097 incl. 2048/2049/4097, split requests, + 64 decode): identical; round 003's build: identical on 42.
Timings (lab, one process): prefill 256 / 1024 / 4096 = 307-315 / 835 / 2992-3009 ms, decode 3.0-3.3 s per 64
tokens (round 002: 400 / 1080 / 3900 ms, 3.5-3.7 s).

**dev-eval 001 (this round's final engine, 1 round, public prompts): PASSED, score 56.61** (prefill mean 85.79,
decode 37.36; round 002's best 46.35).  Prefill 309.6 / 850.0 / 3010.6 ms, decode 3241 / 3135 / 3260 ms per 64
tokens at 256 / 1024 / 4096; mismatching steps 5 / 2 / 12 (the same streams as rounds 002 / 003 on these prompts).

Measured and rejected this round:
- **fp8 storage of the dense attention weights** (would halve the ~253 MB/layer bf16 weight stream of the
  attention half, ~-270 us of ~580 us device per decode layer).  Not exact: trn2's fp8 is legacy e4m3 (max 240).
  With a per-32x32-block factor 1/2 folded into the block scale, values >= 256 fit but odd values of the lowest two
  binades (exp field 0/1, odd mantissa) do not; with factor 1 the reverse.  12-13% of all 32x32 blocks of
  wq_a / wq_b / wkv / wo_a / wo_b (layers 0, 4, 20) contain both (values >= 256: 0.3-1.8%; such tiny values
  1.4e-4), and 87% of 128x128 TensorE tiles contain such a block, so a correction pass would cost about as much as
  it saves.  Script: lab fp8stats (counts from the safetensors bytes).
- **Prefill attention halves on helper cores** (so a layer core runs only moe4: 1.84 ms per group instead of
  layer4 ~0.9 + moe4 1.84): would need ~26 helper cores (8 source layers alone at 1.0-1.6 ms, 32 others in pairs,
  2 Engram stages at ~1.24 ms busy) and only 24 are free; with any layer left at 2.7 ms there is no gain.  Not done.
- **Expert-phase overhead (decode)**: micro-benchmark (expert1 calls, 300 iterations, cores on different devices):
  1 call 142-167 us wall (101 us device); n concurrent calls from n threads 231 (n=2), 278 (4), 289 (6),
  301 us (8) per iteration -- the schedule call stays ~25 us, the *completion wait* grows 123 -> 188 us; one
  thread scheduling all n is worse (schedule calls serialize: 28 us each).  Two calls queued on one core: 242 us.
  So ~290 us is the runtime floor for 6 experts at once; the decode expert phase measures ~360-400 us.

Decode time per layer now (DSV4_PROFILE, 4096 + 64): 1180 us = step/inputs 30 + schedule 67 + attention-half
wait 614 + expert phase ~400 (6 concurrent expert1 ~236 us wall each, shared1 on the home core ~190 us after the
attention); prefill: 2.7 ms per 4-token group per layer core (layer4 0.92-1.0 ms wall, sources 1.0-1.6; moe4
1.87 ms), the stage gap is device-bound.

Ideas left (estimates): decode attention split over two cores by heads (32 + 32 heads, wo_a groups 0-3 / 4-7,
wo_b split-K with an fp32 partial sum and a small post NEFF): ~-150 us per layer (~-10% decode), changes the
wo_b summation order (not bitwise).  The rest of the round-003 list (M=2 prefill groups for short prompts,
Engram gate fused into layers 1 / 14, packed attention inputs, head over 10 cores) is 1-6% each.

Lab tools of this round (copied to tools/lab/; paths inside are this session's): exact.py (replays an evaluation's
exact request sequence from its seed), ref_topk / ref_layers (reference top-k logits and per-layer hook dumps,
teacher-forced on our tokens), cmp_logits / cmp_layers.  Engine diagnostics: DSV4_DUMP_LOGITS now also works on
the decode path with DSV4_NO_HSPLIT=1 (logits are stored tiled: id v at [v % 128][v / 128] of a [128, 1010]
array).

## Round 003: what changed, measured

Every change keeps round 002's arithmetic: the same instructions on the same values, only reordered, batched over
partitions, moved between NEFFs, or with weights laid out differently, so the token streams are those of round 002
(and its correctness results carry over).  Verified bitwise after every step with a residual dump
(`DSV4_DUMP=file`: every layer's output residual of every prefill group, last prompt token and decode step; 15,512
records for a 300- and an 1100-token prompt + 16 decode tokens each, `tools/dcmp.py`) against the round-002 binary
built with the same dump patch: identical every time; and token streams on 20 edge lengths (1 .. 4097, decodes
crossing the indexer's 512 / 1024 thresholds) identical.  Lab driver: `tools/drive.py` (one engine process, split
requests, nonce prompts, 1 warm-up), `tools/cmp.py`, `tools/profneff.sh` (device time of one NEFF), `tools/dmatl.py`
(DMA / engine timeline of a profile).

Lab timings (prefill ms at 256 / 1024 / 4096, decode ms/token; one run each, noise about +-2%):

| step | prefill 256 / 1024 / 4096 | decode | device time (us) |
|---|---|---|---|
| round 002 | 400 / 1080 / 3877 | 55.8 / 55.9 / 56.5 | layer_c2_2560_p 735, layer4_c2_2560 1499, moe4 2063 |
| 1-2. mixes reordered / batched | 350 / 941 / 3357 | 50.4 / 52.6 / 53.1 | 612, 988 |
| 3. expert scales, buffer ring, pipelining | 341 / 932 / 3197 | 50.0 / 52.5 / 52.2 | moe4 1847, expert1 101 |
| 4. engram4 | 329 / 884 / 3181 | | engram stage 3.1 -> 1.2 ms / group |
| 5. router layout, ffn mixes in moe4 | 323 / 855 / 3137 | 49.4 / 49.4 / 49.4 | 579, 885 |
| 6. steps ahead | 315 / 859 / 3092 | 48.3-48.9 / 49.6 / 49.8-50.1 | |
| 7. two tensors per expert | 312 / 844 / 3056 | 48.1 / 48.8 / 48.8 | moe4 1835 |
| 8. 4-token Sinkhorn under wo_a | **310 / 841 / 3004** | **48.3 / 47.7 / 50.1** | layer4_c2_2560 842 |

1. **The hyper-connection mixes off the critical path.**  The model uses each sublayer's mixes in the *next*
   sublayer (model.py Block.forward: the ffn's hc_pre uses attn_pre; ffn_pre goes to the next layer).  The decode
   attention-half NEFF computed the ffn mixes (fp32 projection + sigmoid + 20-iteration Sinkhorn, ~62 us of tiny
   serial instructions) between hc_post and the router, and the attention mixes before the q projection.  Now the
   ffn mixes run in `shared1` (queued behind the attention half on the same core, in parallel with the routed
   experts) and in the M=1 `moe` NEFF; the attention mixes are issued just before the wq_b GEMV (their Sinkhorn
   overlaps the weight stream).  layer_c2_2560_p 735 -> 612 us.
2. **4-token prefill: the per-token serial parts batched over partitions** (`kernels/k_batch.py`): the attention
   and ffn mixes and the router of the 4 tokens run as [4, ...] tiles (token m on partition m) instead of 4 times
   in a row: same element-wise instructions, per-partition reductions, matmuls whose output elements are the same
   dot products.  layer4_c2_2560 1499 -> 988 us.
3. **Expert GEMV (fp4 experts): descriptors, buffers, overlap.**  moe4 was DMA-bound at ~480 GB/s with the DMA
   engines ~60% busy.  (a) The E8M0 scales were loaded per n-tile as 128 rows of 144-320 bytes, as many descriptors
   as the weights: now laid out per partition at load (`weights.h expert_scales`: s[p][nt][b]) and loaded in one
   DMA per matrix on the sync queue.  (b) Every GEMV call allocated its own tile buffers, which the allocator put at
   the same SBUF addresses, so a matrix's first weight DMAs waited for the previous matrix's last matmuls (~15 us
   per expert with both DMA and TensorE idle): one ring of 6 weight + 3 scale buffers is now shared by all GEMVs of
   a kernel (`k_moe.make_rings`).  (c) Software pipelining: pair i's W1/W3 are issued before pair i-1's W2
   (`expert_up` / `expert_down`), so the TensorE works while the swiglu / act_quant of i-1 run.  moe4 2063 ->
   1847 us, moe 700 -> 633, expert1 111 -> 101.  The expert stream now runs at ~290 GB/s per physical core
   (16 DMA engines x ~25 GB/s per descriptor stream); with 4 logical cores of a device streaming at once in prefill
   this is close to the device's HBM bandwidth per core.  A 12-buffer ring: no change.
4. **engram4**: the Engram stages (layers 1 and 14, own cores) ran 4 single-token Engram NEFFs per prefill group and
   had become the pipeline's slowest stage (busy 3.1 ms per group; everything else waited on them, found with the
   `stage gap` / `busy engram` profile counters).  `k_misc.engram4`: the token-only kv GEMV of the 4 tokens with each
   weight tile loaded once (lib.gemv_cm; per token the arithmetic of gemv_c) and the 4 gates.  1.2 ms per group.
5. **Router scores on the partitions** (`k_moe.gate_scores`): the gate GEMV as weight-stationary [128 experts, 3]
   tiles instead of a [1, 384] row, so the long softplus / sqrt chains (~60 instructions) cover 3 values per
   partition instead of 384 on one; the rows come back through bit-exact TensorE transposes for max8 /
   find_index8.  Same dot products (fp32 PSUM, kt order).  -33 us per decode layer, also in the 4-token router.
   And the 4-token ffn mixes moved from layer4 into moe4, their Sinkhorn iterations interleaved with the expert
   pairs (hidden under the expert stream): layer4 988 -> 885 us, moe4 unchanged.
6. **Decode step tensors written ahead** by a step-writer thread: at the start of a token for every layer whose mask
   does not depend on this token's indexer selection, and by an index source right after reading its selection for
   the layers reusing it.  The packed (`_p`) variants read their pre-mix (the previous layer's ffn pre) from pk_in
   instead of the step tensor (the same 16 bytes).  ~-0.5-1 ms/token (on the cores' expert workers the writes
   delayed expert executions: no gain; writing the consumers' steps after the expert phase instead: worse).
7. **Two tensors per routed expert** (`we` = w1 | w3 | w2 fp8, `se` = s1 | s3 | s2 bf16, views in the kernel)
   instead of six: the schedule call costs ~0.6 us per bound tensor (micro-benchmark: 2 vs 40 inputs, 63 vs 93 us),
   and moe4 had 157.  Per prefill group and layer: binding 35 -> 11 us, moe4 schedule call 92 -> 71 us.
8. **The 4-token attention mixes' Sinkhorn under the wo_a stream** (`out_proj_m(mixst=...)`): 885 -> 842 us.

**dev-eval 001 (this round's final engine, 1 round): PASSED, score 55.74** (prefill mean 84.67, decode 36.70; round
002's best: 46.35 / 64.29 / 33.42).  Prefill 306.5 / 832.6 / 2992.4 ms, decode 3304 / 3182 / 3153 ms per 64 tokens
at 256 / 1024 / 4096; speedups 139.3 / 97.0 / 44.9 (prefill) and 38.7 / 35.3 / 36.1 (decode).  Mismatching steps
5 / 2 / 12; controls worst 8 (p5, p65, p128), 88 of 1300 judged steps in all (allowed 130).  These are round 002's
streams (bitwise the same arithmetic): p4096's 12 of 13 allowed is a property of that prompt, as in round 002's
dev-eval 001 (also 12); the formal round-002 evaluation had 7 / 3 there.

Where the time goes now (decode, per layer ~1.2 ms): attention half 579 us device + ~70 us schedule call + ~50 us
model-switch / completion; expert phase ~400 us (6 x expert1 101 us device, each ~250 us wall: xin write ~26 us,
schedule ~45 us, the max of 6); host transfers ~70 us.  The runtime's per-execution costs (schedule call 30-70 us,
+13..46 us when a core switches between models, measured with lab/xb/xs) are now the largest overhead.  Prefill:
~2.76 ms per 4-token group (normal layers: layer4 ~0.85-0.9 ms + moe4 1.84 ms, device-bound), plus ~120 ms of
pipeline fill (one group through ~50 stages), so at 256 tokens the fill is ~40% of the time.

Measured and rejected this round:
- **Tensor parallel attention with in-NEFF collectives** (all_gather over the 4 cores of a device): a collective
  NEFF has ~180 us latency per execution one at a time (77 us when many are queued) vs ~55 us for a plain NEFF; and
  a NEFF with collectives must be loaded and executed by every rank of the global communicator (a group that does
  not execute hangs the others; `ctx_device_id` must equal the global id), so 16 independent per-device groups are
  not possible with one NEFF.
- **Split prefill MoE** (helper cores 42-55 holding ~half the experts of 2-3 layers each; a group's pairs split
  between home `xslots{S}_sh` and helper `xslots{S}` NEFFs, `comb4` summing in rank order): bitwise identical, but
  prefill 1010 / 3475 ms vs 935 / 3381 without (the helpers, 2-3 layers x ~12 pairs x 90 us, became the
  bottleneck; extra NEFF round trips), and helper copies as first replicas made decode 3-5 ms/token slower.
- Keep-awake threads (CPUs sleep in C6, 190 us exit latency): 192 SCHED_IDLE spinners pinned per CPU made decode
  82-90 ms/token.  Tight completion polling (DSV4_POLL=0): no change.  shared1 launched after the attention half
  (DSV4_SHARED_LATE): +2-3 ms/token.
- Host-memory tensors as NEFF inputs/outputs: `NRT_TENSOR_PLACEMENT_HOST` allocation fails (NRT_RESOURCE).
- Row-tiled matmuls for the block-scaled GEMV (4 x [32 k, 128 n] stationary tiles instead of the x-mask trick):
  bitwise identical, 2.4x slower (4x the instructions).  fp8 double-row mode needs fp8 on both operands (x is bf16;
  the e4m3fn activation values above 240 do not fit the legacy fp8); fp4 matmuls are NeuronCore-v4 only.
- Second expert replicas: no measurable decode effect (DSV4_ONE_REPLICA 52.6 / 51.5 vs 52.3 / 52.3 ms/token).

Ideas not done (estimates): smaller prefill groups for short prompts (M=2: ~-6% at 256, worse at 1024+; needs
layer2 / moe2 variants and generalising the 4-token host code); fusing the decode Engram gate into layers 1 / 14
(~0.5 ms/token); packing the attention NEFF's ~36 inputs (~15 us per layer from the schedule call); the LM head over
10 cores instead of 5 (~0.25 ms/token); fp8 storage of the dense weights where exact per block (would halve the
attention weight stream, ~-8 ms/token, but changes the GEMV summation order: a numerics change, not done).

Lab switches added: DSV4_DUMP, DSV4_NO_ENGRAM4, DSV4_NO_STEPS_AHEAD, DSV4_SHARED_LATE, DSV4_POLL; profile counters
`stage gap <stage>`, `busy engram`, `sched ...`, `bind4`.

## Round 002: what changed, measured

All changes keep the arithmetic of round 001: token streams were compared bitwise against the round-001 binary
(lab driver, nonce prompts, 256 / 1024 / 4096 + 64 decode tokens, and a 6000-token prompt + 2150 decode tokens)
and were identical at every step.  The engine now uses all 64 visible logical cores (the runtime gets the caller's
`NEURON_RT_VISIBLE_CORES`; `DSV4_MAX_CORES` limits it in the lab).

Decode ms/token (lab, one process, nonce prompts; noise about +-3%): 72.5 (round 001) -> 66.8 (spread experts)
-> 65.7 (pre-bound expert calls) -> 60-62 (combine on the next core) -> 56-58 (head split, Engram kv ahead,
second replicas) -> 54-57 (shared rows async, device-aware placement, warm-up).  Prefill 256 / 1024 / 4096:
436 / 1198 / 4611 ms -> 420 / 1088 / 3888 ms (overlap of the next group's attention with the MoE, then twins).
dev-eval 001 (one round, before the twins and the async rows): PASSED, score 47.49 (prefill mean 65.1, decode 34.6;
mismatching steps 5 / 2 / 12; the reference's prefill times there were inflated by a NEFF build running at the same
time, so the score is optimistic).
dev-eval 002 (the final engine of this round, clean): PASSED, **score 47.31** (prefill mean 65.34, decode 34.26);
prefill 397 / 1084 / 3911 ms, decode 3578 / 3480 / 3601 ms per 64 tokens at 256 / 1024 / 4096; mismatching steps
5 / 2 / 12 (identical streams to round 001's arithmetic, so the same counts; p4096 is 1 below the per-stream cap).

1. **Routed experts spread over the cores (decode).**  At load, a second copy of every layer's 384 routed experts
   (fp4 -> legacy fp8 tiled, scales) is written to the other cores in proportion to their free HBM (smooth weighted
   round robin; never the layer's own core), plus second replicas where memory remains (~6800 with the twins).
   A decode step (one job in the pipeline) runs its 6 experts as `expert1` NEFFs (k_moe.expert1: one expert,
   output moe_impl's bf16 `y2b`) on up to 6 cores at once through per-core worker threads, while the home core runs
   `shared1` (queued right behind the attention NEFF).  Placement per step: least loaded of replica 1 / replica 2 /
   home (home counts 1.3 for the shared expert), +0.5 per execution already on the same device.
   Measured: expert1 111 us device, 145-150 us wall alone, 200-260 us wall when 6-7 run at once (the runtime's
   schedule call alone is 20-40 us and slows with concurrency; not the HBM: same with cores on different devices).
   The expert phase costs ~430-550 us per layer instead of the 671 us moe NEFF.
2. **Every expert call pre-bound** (one tensor set per replica and per home expert, ~37k sets).  Rebinding an input
   tensor costs ~14 us per tensor per execution (micro-benchmark); switching between pre-bound sets is free.
   All pre-bound calls run once at load (0.2 s) because the first use of a set is slower.
3. **The combine runs on the next core.**  `combine` (sum of the 6 y2b in moe_impl's order, + shared, bf16,
   hc_post) is fused in front of the next layer's attention half (`layer_*_p` variants, k_layer.attn_in_core with
   `pk`); the host moves one pack h2 | fmix | y6 | s2 (112 KB, `PK_*`).  Layers 1 and 14 (after an Engram stage) and
   the Engram stages run the separate `combine` NEFF queued before their kernel; the head parts likewise.
   The fused prologue measured the same as the queued separate combine (layer_c1_512 705 us vs _p 711 us device).
4. **LM head over 5 cores** (39..43, `head_part`: 25,856 ids each, max + lowest id; `head_final` on core 39 picks
   the largest, ties to the lowest id).  Same GEMV and accumulation order as `head`, so identical logits.
   2.13 ms -> ~0.9-1.0 ms per token.
5. **Engram split (decode):** `engram_kv` (token-only GEMV, 315 MB) is launched on the Engram cores when the step
   starts; the Engram stage runs only `engram_gate` (combine queued before it).  656 us -> ~270 us per stage.
6. **Prefill: the next group's attention half is queued while the current group's MoE runs** (double-buffered
   layer4 / moe4 outputs, `run_layer4_a/b/c`); hides the host work between NEFFs: 1198 -> 1159 ms, 4611 -> 4475 ms.
7. **Prefill twins:** the 8 kv / index source layers (2, 8, 14, 20, 24, 28, 32, 36: attention halves 1.85-2.17 ms
   for 4 tokens, vs 1.4-1.6 elsewhere) have a twin core (63..56) with a full copy of their experts; their moe4 runs
   on the twin as a pipeline stage of its own.  1186 -> 1088 ms, 4516 -> 3888 ms; the bottleneck is now the
   ordinary c1/c2 layers (attention 1.48-1.58 + moe4 2.04 ms).
8. **Shared compressed rows** (kv sources -> the layers reusing them; layer 20 has 19 consumers) are written by the
   consumer cores' worker threads, batched per group; a consumer waits for `job->cw[l]` before its kernel.

Measured and rejected (or neutral) this round:
- Weight prefetch in the attention half (`lib.prefetch`, wq_b / wo_a / wo_b tiles issued early, router weights
  under wo_b): device time 698 -> 705 us, no change (the DMA gap at 60-120 us stays).  Left in (harmless).
- Expert weight DMAs split over the sync + scalar queues (by halves: expert1 118 us, moe4 2356 us; alternating
  tiles: 112 / 2123 us) or 2-3 n-tiles per DMA (113 / 2041, 115 / 2084) vs 111 / 2043: no gain.  The block-scaled
  GEMV is bound by TensorE stationary loads (one 128x128 fp8 tile per 4-column matmul).
- Shorter spin before blocking in the host queues (20000 or 2000 pauses): decode 57 -> 67-69 ms (wake-ups cost).
- `NEURON_RT_ONE_THREAD_PER_CORE`, `NEURON_RT_DISABLE_EXECUTION_BARRIER`, `NEURON_RT_IO_RING_CACHE_SIZE`,
  `NEURON_RT_LOW_LATENCY_TASKS_CPU_AFFINITY`: no effect on concurrent execution cost.
- Binding a tensor allocated on another core (same device or not): rejected by the runtime (NRT_INVALID).
- Host transfers are cheap: 10 KB write 5.7 us / read 5.1 us, 40 KB 8.6 / 7.6 us.

Ideas not done (estimates): fp32-order-changing GEMV forms (x stationary, weights moving with pow2 factors folded
into x) could cut expert time but change rounding (p4096 already had 12 of 13 allowed mismatches in dev-eval 001);
large-M expert-deduplicated prefill (each expert read once per 128+ tokens) is the big prefill lever (~2x at 4096)
but needs new MoE and chunked-attention kernels.

Lab switches: DSV4_NO_SPREAD, DSV4_NO_HSPLIT, DSV4_NO_FUSECOMB, DSV4_NO_OVERLAP4, DSV4_NO_TWIN, DSV4_SYNC_ROWS,
DSV4_ONE_REPLICA, DSV4_NO_DEVAWARE, DSV4_NO_WARM, DSV4_SPIN_X / DSV4_SPIN_STAGE, DSV4_PROFILE.


## Layout of the engine

- `kernels/` NKI kernels (build time only), `kernels/specs.py` = every NEFF and its argument shapes,
  `kernels/build_neffs.py` compiles them through NKI's documented route (one process per kernel, one NeuronCore
  each, cached under `target/neff-cache` by source hash).
- `src/engine.cpp` host (libnrt only), `src/weights.h` checkpoint mmap + load-time re-layouts, `src/util.h` JSON.
- `data/engram_map.bin`, `data/engram_consts.h`: Engram hashing constants generated once by
  `tools/gen_engram.py` from the official `inference/engram.py` + the tokenizer (compressed token map, hash
  multipliers, primes, offsets).  Committed so the build needs no tokenizer.
- `build.sh`: NEFFs -> `target/neff`, `.incbin` them (and the token map) into `target/engine`.  A clean build
  (empty cache, 8 build cores) takes ~8 min; `NEFF_BUILD_CORES=0,...,31` ~3 min.  The cache key hashes *all* kernel
  sources, so any kernel edit rebuilds every NEFF.  Never edit kernels/ while a build runs.
- `kernels/k_batch.py`: the M-token (partition-batched) mixes and router; `kernels/k_moe4.py` is generated text
  (see round 003); `tools/`: lab driver, comparisons, profiling helpers (need the session's skill scripts).

## Placement and data flow

- Layer l (0..39) runs on logical NeuronCore l; all of its weights live there (routed experts ~14.4 GB, dense
  ~0.4 GB, caches).  The LM head sits on core 39; the Engram stages (layers 1, 14) run on cores 40, 41.  The engine claims only
  the first 42 of the visible cores.
  Loading takes ~2 min.
- Every token, prompt tokens included, is one decode-style step through the 40 layers.  **The host runs one thread
  per layer core and pipelines prompt tokens**: core l works on token t while core l-1 works on token t+1, so a
  prefill costs about one layer step per token (~2.4 ms) instead of 40.
- Per layer and token (`Engine::run_layer`): one host write of the small inputs (`step` tensor: positions,
  pre-mix, RoPE rows, compressed-attention mask; the kernel inputs are slices of it; in decode written ahead by the
  step-writer thread), the residual write, `[engram]`, the fused attention-half NEFF `layer_*` (kernels/k_layer.py:
  attn_in, compressor of kv sources, indexer of index sources when > 512 compressed positions, sparse attention,
  wo_a, wo_b, hc_post, hc_pre(attn_pre), ffn_norm, router; the ffn mixes are computed by shared1 / moe / moe4), a
  read of the router output, the experts (decode: expert1 on 6 cores + shared1 at home; prompt groups: moe4 with
  the 24 experts bound as 2 tensors each), one read of residual + pre-mix.  Executions are queued
  (`nrta_execute_schedule` + polling: ~150 us less than `nrt_execute`).
- Routed expert storage: per expert `we` = w1 | w3 | w2 (fp8 tiles, 3 x 11.8 MB) and `se` = s1 | s3 | s2 (bf16 scales
  per partition, 3 x 737 KB), contiguous per expert on the home core, the replicas and the twins.
- Prompt tokens go in groups of 4 (`layer4_*` NEFFs, `k_layer.layer_impl_m`, `k_moe4`), positions aligned to
  multiples of 4, the last prompt token alone; per-token arithmetic is that of M=1 (logits bit-identical).
- NEFF variants by layer type and compressed bucket T (ratio 2: 512, 1024, 2048, 2560, 4096; ratio 1: 512, 1024,
  1536, 2048, 4096, 4608, 8192): `layer_w` (window only), `layer_c2_T` /
  `layer_c1_T` (consumers), `layer_s2store_T` / `layer_s2emit_T` (ratio-2 kv sources by position parity),
  `layer_s1_T` (layer 20), `layer_i1_T` (24, 28, 32, 36).  Generated by `kernels/gen_layer_variants.py`.
- Shared state: the compressed KV row and index key a kv source adds are patched into the kernel's SBUF copies
  (so the kernel never reads back its own HBM writes), output, and written by the host into this layer's caches and
  into those of the layers reusing them.  The indexer selection (in-kernel mask) is output and turned by the host
  into the mask of the layers reusing it.  With <= 512 compressed positions the selection is all of them.
- **No DMA in any kernel has a run-time offset** (see pitfalls): RoPE rows, masks and positions come from the
  host-written step tensor; expert weights are bound by the host; cache rows are patched in SBUF and written by
  the host; the window ring is updated in-kernel with a predicated copy and written back whole.
- The host computes only scheduling data: positions, the Engram n-gram hash ids (integer hashing of token ids, as
  `NgramHashState`), the gathered Engram table rows and embedding rows (raw checkpoint bytes), masks from the
  device's selection.  RoPE cos/sin tables are built at load (model constants, fp32 like torch).

## Numerics (what matches what)

- Dense fp8 weights are decoded to bf16 at load (exact, fp8 * 2^k); GEMVs are bf16 x bf16 with fp32 PSUM
  accumulation -> products are exact, only the summation order differs from the reference.
- trn2's TensorE fp8 is the legacy e4m3 (max 240): values 256..448 of e4m3fn decode to inf/NaN on the DMA and
  vector casts too (measured).  So e4m3fn is never fed to the hardware as fp8.
- Routed experts: fp4 e2m1 nibbles -> legacy fp8 bytes holding value * 2^-6 (exact for all 16 codes, subnormal
  0.5 included; verified 0 error in a matmul).  Expert GEMV keeps the official per-32-K-block structure: each
  [128 k, 128 n] fp8 weight tile is the stationary operand, the moving operand is x masked to one 32-block per
  column ([128, 4]), the fp32 block sums are multiplied by the block scale (bf16 2^(e-127+6)) and added.
- act_quant (fp8 e4m3fn RNE with pow2 scales) and fp4_act_quant are emulated exactly in fp32 with the 2^23
  magic-number rounding (`lib.round_e4m3fn`, `lib.round_e2m1`); divisions by non-pow2 scales use a refined
  reciprocal (<= 1 ulp from true division).
- sparse_attn is emulated exactly as the reference orders it (`k_attn_out.attn_core_list`): the host builds the
  reference's key list (window entries in the reference's order -- prompt order for prompt tokens, ring order in
  decode -- then the selected compressed keys ascending; up to 640 entries) as list positions (`wrank`, `lvalid`,
  `crank`; for index layers the kernel derives `crank` itself from its own top-512 selection, 128 + rank).
  One-hot TensorE matmuls gather the keys into list order, and the online softmax runs over 64-entry blocks:
  running max, P rounded to bf16 per block, fp32 rescale, fp32 PV.  This cut mismatches on long prompts
  (p2048: 14 -> 1, p2049: 14 -> 4).
- exp / sigmoid / log / softplus / sqrt use range-reduced fp32 versions (`lib.*_precise`) instead of the
  activation-table approximations (dev build; +1-2 ms decode).
- Indexer top-512: bf16 scores rounded as the official code (bf16 dot, relu, * bf16 weight, fp32 sum -> bf16),
  exact radix-order threshold by binary search over the 16-bit keys, ties by lowest index; matched the reference's
  selection 512/512 on layers 2, 20, 24, 36 (golden data).
- Per-kernel checks against golden data from the CPU reference (lab, `golden/g1100`: 1100-token prompt + 3
  decode steps, per-layer inputs/outputs/caches): attn_in bit-exact (q: 1-2 ulp diffs in 1/32768), compressor key
  rows exact (ckv 1 element in 512 differs), indexer selection exact, MoE layer output 2e-5 relative, Engram
  bit-exact, head logits 1.8e-7 relative.

## Pitfalls found (keep)

- **A DMA whose offset comes from SBUF (scalar_offset / vector_offset) can read the offset before the instruction
  producing it has written it, even when the producer is another DMA from a kernel input.**  Seen as wrong
  expert weights (first tiles of the first expert), and as NRT_EXEC_OOB crashes at random layers (dev-eval 003).
  All run-time offsets were removed from the kernels.
- The async version briefly wrote a kv source's new compressed row into its own cache after launching the kernels
  that read it (nondeterministic logits); caught by running the same request twice and comparing logits.

- PSUM tiles must not straddle 2 KiB banks: `lib.ps` pads every PSUM tile to a full bank.
- Engines cannot start an instruction at partition 3 (alignment); RoPE on row 3 of a [4,128] tile is done on the
  whole tile with an identity rotation on the other rows.
- `nl.divide` is not a tensor_tensor op on trn2; inner functions are not allowed in kernels.
- Kernels compiled with `kernel(...)` (no [2]) are LNC=1 NEFFs and run on the logical core (one physical core).

## Measurements

| version | decode ms/token | prefill 1100 tok | notes |
|---|---|---|---|
| v1 sequential, LNC1, sync | 135 | 142 s | one step per token through 40 cores |
| v2 + host pipeline | 135 | 5.9 s | identical tokens |
| v3 no dynamic DMA, LNC2 | 105 | 4.6 s | 4097-token control: 23.5 s |
| v4 async queue, step tensor, hwdge weights | 78 | 2.6 s | 4097: 15.4 s |
| v5 fused attention NEFF | 74.6 | 2.6 s | logits identical to v4 |
| + Engram on own cores, M=4 prompt groups | ~72 | - | 4096 prompt: 7.0 -> 4.6 s |
| v6 exact sparse_attn list emulation | 70-74 | - | 256 / 1024 / 4096 prompt: 0.38 / 1.12 / 4.31 s |
| + precise exp/log/sqrt functions | 75-76 | - | 4096: 4.68 s |

Fidelity (mismatching steps of 65 greedy steps vs the reference judge; p65, p257, p2048, p2049 Lily controls):
reference vs itself with another thread count 5, 1, 5, 4; v5 7, 8, 14, 14; v6 10, 7, 1, 4; precise functions
8, 4, 6, 4
(p63 6, p128 8, p255 7, p4097 1; p5 11 -> 8, p64 11 -> 4; p4096's first token, wrong by 1.18 nats in v6, is
right).  The model is chaotic: these counts move by several between runs of semantically equal arithmetic,
so they are measured on several prompts, not one.
Dev-evals: 003 crash (dynamic-offset DMA race), 004 failed (p2048/p2049 14 mismatches, 8.05% total),
005 (v6) passed: score 39.76 (prefill speedup 60.4, decode 26.2; prefill 417 / 1184 / 4406 ms at 256 / 1024 /
4096, decode ~70-74 ms/token), controls worst 11 mismatches (p5, p64), 68 of 1105 steps overall.
The precise functions cost ~5% speed and were kept for the margin on the per-stream gate.
Finer compressed buckets (1536, 2560, 4608) gave no measurable prefill gain.

Device time per NEFF (neuron-explorer, LNC2): attn_in 0.29 ms (106 MB), attn_out 0.42 ms (163 MB),
moe 0.70 -> 0.61 ms with weight DMAs on hardware DGE (sync queue) and scales on the scalar queue (297 MB,
~490 GB/s), head 2.1 ms (1.32 GB, 630 GB/s), hc_mixes alone 105 -> 77 us after dropping the Newton step of the
reciprocal (VectorE reciprocal is correctly rounded: 0 ulp on 65536 samples).
`nrt_execute` costs ~150-350 us more wall time than the device time; queued execution with polling ~50-100 us.
The profiler reports engine throttling to ~50% utilisation limit during these kernels (not understood).

Logits vs the CPU reference (golden 1100-token prefill): relative L2 0.125-0.137, max abs 0.67-0.95 — the size
of the reference's own variation under a change of fp32 summation order (0.13 / 2.0, REFERENCE-VALIDATION.md).
A judge of a 12-token continuation: 3 low-margin mismatches (0.42-0.76 nats, ranks 3-9).
