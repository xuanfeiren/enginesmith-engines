# Qwen3-32B engine for Trainium2 (tensor parallel 8): notes

## Round 011 (2026-11-08): unpadded decode MLP (MPD 4: 1600 intermediate rows per core, combo k-tile)

Starting point: round 009 (= round 010's unchanged engine). The NEFF cache of the harness build dir
(`runs/.../build/target/neffcache`) matched the source, so the first build took 3 s. Measured: whole requests,
cores 8-15, engine alone, the same random prompts through each engine (`tests/tools/r11/sweep.py LENS N ENGINE...`,
`SEED=7`); variants: `tests/tools/r11/var.sh NAME CORES step_b512` (work dirs `/tmp/q32w11/NAME`, `STEP_OVERRIDE`
JSON, other NEFFs from this build). Decode profiles: `tests/tools/r11/pstep.sh NEFF OUTDIR` (neuron-explorer on
cores 8-15, json reduced to a pickle with `red.py`); analysis scripts in `tests/tools/r11/` (`cc.py` all-reduce
times / layer period, `phase.py` DMA rate per core around the down all-reduce, `gap.py` / `rings.py` DMA bytes per
1 us / per DMA engine, `sync.py` / `sbmap.py` the Sync engine's DMA instructions with SBUF destination offsets
(`dst_addr_imm`), `ahead.py` bytes streamed between the down GEMV end and the next matmul). The chain experiments'
code is in `experiments/round011-chain-switches.patch` (against the round-009 kernels; includes round 010's switches).

### Diagnosis of the decode step (b512 profile, 215.4 us per layer)

- The weight stream stops ~15 us per layer (core 1; ~10 us core 0) around the **down** all-reduce, from when the
  queued next-layer prefetch (QKV 3 slabs + 10 o_proj slabs) has drained until the static read-back of the
  all-reduce, triggered from the in-order Sync queue, fires (+13 us). Only ~27 KB/partition stream in the 29 us
  from the down GEMV end to the next layer's first matmul (78 KB would be needed at full rate).
- The Sync engine keeps ~16 hwdge DMA instructions in flight (one per qSyncDynamicHW ring semaphore); each
  DMA_DIRECT2D costs 0.7-0.85 us of Sync time regardless of size; the 2 KB/partition o_proj slabs fill the window
  with little data.
- The compiler's SB allocator (log: "SB spills = 192 + 64 tensors, remats", 20 KB each) rematerialises 4 prefetch
  slabs per layer: SBUF (224 KB/partition, 16 KB DGE scratch) is the limit on prefetch depth.
- All-reduce latency: 20 KB fp32 12.2 us when the DMA engines are idle, 18 us during weight streaming (o_proj
  all-reduce), 40 KB 18 us, 15 KB 20 us (with traffic), 5 KB 11.4-12.3 us: a ~11 us fixed cost; running the weight
  stream during the all-reduce slows it by 2-6 us.
- **The MLP phase is bandwidth-bound**: timing-only builds (wrong values) without 1 of the 13 intermediate tiles per
  core: gate/up 24 tiles + down 12 k-tiles 845.4 ms, gate/up only 859.8, down only 841.3 vs 885.7 ms (64 tokens,
  b512). Every MB per core and layer of MLP weights costs ~10 ms per 64 tokens.

### Adopted: MPD 4 + MPL 3 (`kernels/step_tp.py`, `targets.STEP_MPD`, host `layout_d_combo`)

Each virtual core owns exactly 1600 intermediate rows (12 full tiles + a 64-row half; round 009's MP split) instead
of 13 padded tiles (vcore 15 had 8 zero tiles, all cores streamed 13):
- gate/up from the prefill MP copy `wgp` (now also a step input: the same device tensor): 24 own tiles + the packed
  tile [gate half | up half]; the packed tile's slab is issued first (MPL 3); act half = silu(psum rows 0-63) x
  rows 64-127.
- down from a new decode-only copy `wdh` (2.2 GB per rank, `layout_d_combo`): 12 own k-tiles for all 40 output tiles,
  then one **combo** k-tile (rows 0-63 this core's 64 half rows, 64-127 the other core's) for output tiles
  [20 pid, 20 pid + 20): the cores exchange their act halves by one 64-value sendrecv; every matmul is a full
  128-row one and the two cores stay balanced (12.5 tiles each).
- 3.8% fewer MLP bytes (25 + 12.5 instead of 26 + 13 tile units). Not bitwise: the down sums run in another order.

| variant (decode 64 tokens, b512, ms; baseline 885.3-885.7 in the same runs) | ms | |
|---|---|---|
| MPD 1: HALF-style down (64-row half tiles, one 64-partition DMA + 40 K=64 matmuls), packed slab last | 905.4 | the gate/up prefetch depth fell (5 slabs before the o_proj read-back instead of ~9) |
| MPD 1 + packed tile in a 3-tile last slab (MPL 1) | 885.4 | |
| MPD 2: down zero-padded to 13 k-tiles (half rows + zeros); packed slab separate / in the last slab | 889.5 / 880.9 | |
| MPD 4 (combo), packed in the last slab / **first (MPL 3)** | 878.3 / **870.6-870.8** | 4 of 5 streams identical, one diverges at token 20 |
| MPD 4 + combo sums in their own PSUM tile (MPC 2) | 870.5 | no gain, not used |
| MPD 7: combo gate/up tiles too (wgx, 2 tiles: no packed tile, no sendrecv; 26 + 12.5 units) / slab first | 878.5 / 894.7 | duplicates the half rows: fewer bytes saved |

**Gate/up prefetch depth re-tuned with MPD** (`targets.STEP_PFG_MPD`, decode 64 tokens): b512 PFG 13 / 11 / 10 / 9:
870.4 / **864.7** / 871.9 / 872.0; b1152 13 / 11: **862.0** / 863.7; b4224 13 / 11 / 9 / 7: 896.7 / 885.0 / **882.7** /
887.1 ms; b2048 / b4608 (not timed) take 11 / 9. MPL 1 instead of 3 at b4224: 893.9 (not used).

**Final engine vs round 009** (in-tree build: all NEFFs rebuilt, 15 min 19 s, then the step NEFFs again 7 min),
20 prompts of 1, 2, 100, 255-257, 511-513, 1000, 1023-1025, 2047-2049, 4095-4097, 4150 tokens + 64 generated
(`SEED=7 sweep.py`, round-009 = the harness build dir's engine): decode b512 885.3 -> **866.2**, b1152 888.4 ->
**862.2**, b4224 907.1 -> **882.6** ms (-2.2 / -2.9 / -2.7%); prefill unchanged (33.6 / 94.4 / 345.6 ms; 4150:
1110.9 -> 1090.6). Tokens: 15 of 18 streams identical, 513 / 1025 / 4096 diverge at generated token 23 / 55 / 36
(down sums in another order). Long output: 200-token prompt + 4,000 + 407 tokens all returned (position 4,608),
then a reset request answered.

dev-eval 001 (2 rounds, all lengths, MPD with PFG 13 everywhere): **PASSED, score 1.6450** (prefill 1.9152, decode
1.4128; round 009's formal 1.6328 / 1.9189 / 1.3892); p256 2.006 / 1.362, p1024 2.319 / 1.499, p4096 1.510 / 1.382
(candidate decode ms 870.3 / 861.9 / 896.7); mismatching steps [0, 0], [0, 0], [1, 1].

**dev-eval 002 (2 rounds, all lengths, the final engine): PASSED, score 1.6508** (prefill 1.9085, decode 1.4279);
p256 1.987 / 1.374, p1024 2.315 / 1.499, p4096 1.511 / 1.414; candidate ms per round: prefill 33.6 / 34.0, 94.5 / 94.4,
345.5 / 345.7, decode 866.1 / 866.0, 862.1 / 862.3, 884.7 / 882.5; mismatching steps [0, 0], [0, 0], [1, 1] (the
prefill speedups are lower than round 009's formal ones only because the reference prefilled p256 faster: 67.6 / 66.9 ms).

Clean build (copy without target/, empty NEFF cache, cores 0-15): 15 min 35 s; tokens and times identical to the
in-tree build (`SEED=11`: 1, 256, 1024, 4096 + 16).

What changed in the engine: `kernels/step_tp.py` (MPD switch: `_mpd_slabs` / `_mpd_load` / `_mpd_mm`, the MPD 4
branch of `_layer_tp`; new step inputs `wgp`, `wdh`; experiment switches MPD 1 / 2 / 5 / 7, MPL, MPC, DN3, MPX off),
`kernels/targets.py` (`STEP_MPD`, `STEP_PFG_MPD`, `wgp` / `wdh` in `STEP_SPEC`), `src/engine.cpp` (`layout_d_combo`
for `wdh`, layer tensors that no step NEFF uses are allocated by the prefill setup; experiment layouts
`layout_d_mpz` / `layout_gx` behind `ENGINE_X_*` env vars), `kernels/build_neffs.py` (`KEEP_CC_LOG=1` keeps the
compiler log next to the cached NEFF). Memory: +2.2 GB per rank (`wdh`).

### Measured and rejected this round (b512 decode 64 tokens unless noted; baseline 885.3-885.7)

| change | result |
|---|---|
| next layer's o_proj prefetch all 20 slabs (PFO 20) / + 2 or 4 next-layer gate/up slabs (PFN, round 010 code) | profiles unchanged: the extra DMAs wait for SBUF (TensorE semaphores) and issue after the read-back |
| down slabs of 6 tiles (20 KB, SLD 6) + PFN 2 / 4 | profile 214.5 us/layer vs 215.4; 885.3-885.5 ms (=) |
| down all-reduce read-back on the Scalar hwdge queue (RBQM 3) | stream runs through the all-reduce, but the all-reduce takes 14.4 vs 12.2 us and the stream stops later (SBUF): 217.7 us/layer (892 ms in round 010) |
| H2: no two-core sendrecv, each core writes its fp32 partial into a 2-slot buffer, one 40 KB all-reduce | 938.8 (down) / 964.8 (both) |
| BFAR: both per-layer all-reduces in bf16 | 914.4 (tokens identical on 3 x 65) |
| all-reduce DMA priority (`ncc.all_reduce(priority=)`) | does not compile: NeuronCore-v4+ only |
| DSA: down all-reduce split in 30 + 10 (or 20 + 20) hidden tiles, the first part during the rest of the down GEMV (read-back Sync / Scalar / GpSimd) | 944.8 / 924.2; 947.9 / 948.3 (bitwise: tokens identical) |
| OB: next layer's o_proj prefetched in 1 / 2 DMAs (one 40 KB/partition tile) | 902.5 / 897.6 (the down all-reduce slows to 15.7 us under traffic) |
| DN3: down slabs as 3-D tiles by one access-pattern DMA (same bytes) | 885.5 (=) |
| PFG 12 (control for the timing builds) | 889.5 |

### Next (estimates)

- The packed gate/up tile still costs ~15 ms against the timing-only 24 + 12 tile build (844.7): a cheaper home for
  the 64 half rows would be worth up to ~1.5%.
- QKV: the rank's kv head is projected by both cores (1.3 MB per core and layer, ~13 ms per 64 tokens at the MLP's
  marginal rate); KVS (round 004) put a sendrecv on the chain and lost then.
- Prefill unchanged this round.

## Round 010 (2026-11-07, rejected candidate; its notes kept): no change adopted (engine = round 009); HAM throttle found, ten levers measured and rejected

The engine source is unchanged from round 009 (same NEFF cache keys, NEFFs byte-identical to the round-009 clean
build). Every experiment's code is in `experiments/round010-switches.patch` (all switches default off; apply with
`git apply` to reproduce). Tools: `tests/tools/r10/` (`var.sh NAME CORES TARGET...` builds a variant in
`/tmp/q32w10/NAME` with `PF_BUCKET_OVERRIDE` / `STEP_OVERRIDE`, `tbuild.py` for timing-only env switches under a dummy
key, `pf.py LENS N ENGINE...` prefill-only medians, `tl.py` per-engine busy / DMA / HAM timeline from a profile json,
`ham.py` / `ham2.py` throttle windows vs TensorE activity, `stall.py` decode DMA stalls around the all-reduces,
`lk.sh` = the GPU lock: `/tmp/engine-gpu-0.lock` must be opened read-only, `flock FILE` fails on the sticky /tmp).
Measured: whole requests, cores 8-15, engine alone, the same random prompts through each engine.

Baseline this round: prefill p256 33.7-33.9, p1024 94.4-95.0, p4096 346.0 ms; decode 64 tokens 885.5 / 888.6 / 907.2 ms.
Decode cost model (64 tokens = 13.825 ms per step + ~1 ms per request; N = 1 / 2 / 8 / 32 / 64 / 128 tokens:
14.6 / 28.8 / 111.6 / 443 / 885.9 / 1770.7 ms).

**Finding: the TensorE activity monitor (HAM) throttles TensorE to 50% after idle phases.** The profile json has a
`ham` list: windows (multiples of 3.41 us) with util limit k/n = 8/8 or 4/8. p256 layer NEFF: 50% of the time at
4/8; 857 of 4,496 us of TensorE busy time ran under the limit. p1024 layer NEFF: 3.06 of 16.4 ms of TensorE busy under
the limit, mostly inside the nkilib attention (one 126 us window at TensorE busy 0.81 per layer) and the QKV phase.
The limit drops when TensorE goes idle (the all-reduces) and is lifted only after a few windows of dense matmuls.
No documented knob (libnrt only has the PE_HAM register block). Keep-alive matmuls during the all-reduces did not help:

| experiment (p256 unless noted) | result | verdict |
|---|---|---|
| keep-alive dummy matmuls after each all-reduce, dead (no consumer) | eliminated by the compiler (TensorE time unchanged) | - |
| keep-alive, live (N=128, accumulated, sunk to an HBM dummy) 200 / 400 per all-reduce | 34.01 / 34.06 vs 33.63 ms | rejected |
| NOATT re-measured (nkilib attention skipped, timing only) | 34.35 vs 33.8 ms (slower, as in round 008) | attention is not the p256 critical path |
| NOATT + phase-A writes on the Scalar queue | 34.30 | - |
| `QPD`: QKV tile post-processing one tile behind (prefill_tp HS path; bitwise) | 33.81 vs 33.74 | no gain |
| `DR`: deferred rstd (x = bf16(h*w), rstd applied to v / gate / up outputs; q/k norms cancel it; not bitwise, 1 of 8 tokens differed) | 33.53 vs 33.74-33.87 (-0.6%) | too small for a numerics change |
| QPD + DR | 33.64 | - |
| `WQS` 1 / 2: phase-A q/k/v/cache writes + s_o reads off the Sync queue (Scalar hwdge / GpSimd swdge) | 33.67 / 34.05 vs 33.67 | no gain |
| p1024 / p4096 `SKV` (SP path: core 0 projects k, core 1 v, each writes both copies; bitwise, tokens identical) | 96.0 / 368.8 vs 94.5 / 346.0 ms | rejected (much slower) |
| decode `RBQM` 7 / 3: down all-reduce read-back on GpSimd swdge / Scalar hwdge | 892.0 / 892.2 vs 885.5 | rejected |
| decode `PFN` 1 / 2: next layer's first gate/up slabs prefetched with the QKV / o_proj prefetch | 885.3-885.5 (=) | no gain |
| decode `PFE` 4 / 10 / 16 (+PFN 2): next-layer prefetch issued inside the down GEMV after PFE tiles | 1020 / 974 / 955 / 974 | rejected (down slabs starve) |
| decode PFG 10 / 11 / 11 + PFO 20 / PFG 8 + PFN 3 | 894.7 / 888.8 / 889.4 / 895.4 | rejected (13 stays) |

Decode profile (b512 step, `stall.py`): weight stream ~332 GB/s per physical core while streaming, 1,529 of 14,195
us below 150 GB/s; the stall is at the **down** all-reduce (~15-20 us per layer, none at the o_proj one). There the
Sync queue first waits ~10 us on semaphores (slab buffers freed only at the end of the down GEMV), then issues the
next DMAs at ~45 GB/s while the all-reduce runs, then the read-back. Step start ~15 us and tail (argmax all-reduces)
~75 us without DMA. Per-request fixed prefill cost (ENGINE_DBG_PF_LAYERS=0): 1.28 ms at p256 (engine-side ~1.0 ms:
~0.3 ms host writes / thread wake, ~0.6 ms emb + head NEFFs).

**dev-eval 001 (1 round, all lengths, the unchanged engine): PASSED, score 1.6528** (prefill 1.921, decode 1.422);
p256 2.014 / 1.336, p1024 2.325 / 1.470, p4096 1.514 / 1.465 (candidate ms 33.5 / 885.4, 94.2 / 888.4, 345.5 / 907.0;
the higher decode speedups are only reference noise: 1183-1329 ms); mismatching steps [0], [0], [1]. dev-eval opens
`/tmp/engine-gpu-0.lock` for append, which /tmp's protected_regular refuses for this user: run it as
`tests/tools/r10/lk.sh env GPU_LOCK_HELD=1 enginesmith dev-eval ...` (the wrapper holds the lock via a read-only fd).

### Next (estimates)

- The HAM throttle is the largest untapped item in prefill (p1024: ~19% of TensorE time at half rate, much of it in
  the nkilib attention). A hand-written SBUF-resident attention with dense TensorE bursts might stay unthrottled;
  untried, large.
- Decode: the down all-reduce stall (~1 ms per step) is set by the compiler's slab-buffer reuse; every queue /
  prefetch-order lever reachable from NKI has now been measured neutral or worse.
- Folding the head NEFF into the last layer execution and the emb into the first (~0.3-0.5 ms per prefill,
  ~1% at p256); untried.

## Round 009 (2026-11-07): round 008's SP chain trims taken over + unpadded prefill MLP (MP)

Starting point: the round-007 commit plus the source of round 008 (rejected only on the margin: formal score +0.35%
with decode -1.14% from reference noise; its prefill changes are bitwise identical to round 007 and measured
p1024 100.3 -> 97.7, p4096 370 -> 353.7 ms). Round 008's NEFF cache (`/tmp/q32w8/clean/target/neffcache`) was reused
for the in-tree build. Its notes section follows below unchanged.

Measured: whole requests, cores 8-15, engine alone, the same random prompts through each engine
(`tests/tools/r9/sweep.py LENS N ENGINE...`, `SEED=7`); variants: `tests/tools/r9/var.sh NAME CORES TARGET...`
(work dirs `/tmp/q32w9/NAME`, other NEFFs from the round-008 build) with `PF_BUCKET_OVERRIDE`.

**Diagnosis: TensorE rate, single-core probes** (`probes/p9_mm.py`, profiled with `neuron-explorer capture`, 3840
matmuls of K = M = 128): N = 512 / 256 / 128 moving columns all take the same time per column (833-861 us TensorE
active, 217 ns per N = 512 matmul = 2.36 columns / ns at the profile's 2.4 GHz PE clock: the LDWEIGHTS of the next
stationary overlap the current matmul; reusing one stationary for two moving blocks gains nothing). Two M = 64 (column
tiling, `tile_position` / `tile_size`) or two K = 64 (row tiling) matmuls run in parallel in the time of one full
matmul; without tiling they cost two. Elementwise ops between PSUM partitions 64-127 and SBUF partitions 0-63
compile and are exact (`probes/p9_part.py`). So the prefill MLP's cost is its matmul count, and the 200 intermediate
tiles zero-padded to 208 (13 per core) cost 4% of it.

**MP (adopted for the 1024 / 4096 buckets).** Each virtual core v owns intermediate rows [1600 v, 1600 v + 1600)
(12 full tiles + a 64-row half; no zero tiles). New prefill-only weight copies (`wgp`, `wdp`, 6.4 GB more per rank,
host layout `layout_gu_mp` / `layout_d_mp` in `src/engine.cpp`, +2 s load):
- gate / up: 12 gate + 12 up slabs + one packed slab whose 128 output columns are [gate half | up half], so its PSUM
  rows 0-63 are the gate half and rows 64-127 the up half: act half = silu(rows 0-63) * rows 64-127 (25 slabs of 40
  matmuls instead of 26);
- down (N-split, as before): 12 own + 12 other-core tiles + one combo K tile whose rows 0-63 are core 0's act half and
  rows 64-127 core 1's (both read from the act exchange buffer), weights in slot 12 of both cores' `wdp` slices
  (25 matmuls per output tile instead of 26);
- `MP` switch in `prefill_sp.py` (SP path) and `prefill_tp.py` (non-SP path, used only in an experiment).
Not bitwise: the down sums run in another order (12 + 12 + combo instead of 13 + 13).

| variant | prefill ms | tokens vs round 008 |
|---|---|---|
| p1024 round 008 / **MP** | 97.9 / **94.9** | 6 prompts x 33 identical |
| p4096 round 008 / MP (SPG 3: does not fit SBUF) / **MP + SPG 2** / MP + SPG 1 | 353.7 / - / **346.0** / 409.3 | identical (2 x 17) |
| p256 (prefill_tp HS path) round 008 / MP | 33.7 / 33.7 | one of 6 streams diverged at token 20: **not used** |
| silu of the gate half in place in PSUM (to save the SBUF tile) | does not compile ("Both data1 and data2 cannot be in PSUM") | - |

Final engine (in-tree build, prefill NEFFs rebuilt: 6 min 55 s) vs round 008 on 20 prompts of 1, 2, 100, 255-257,
511-513, 1000, 1023-1025, 2047-2049, 4095-4097, 4150 tokens + 64 generated (`SEED=7`): prefill 1024 bucket 97.8 ->
94.4-95.0, 4096 bucket 353.6 -> 346.3 (4097: 368.4 -> 361.2, 4150: 1119.0 -> 1111.4) ms; 256 / 512 / 2048 buckets and
decode unchanged. Tokens: 16 of 20 streams identical; 1000 / 1023 / 1025 / 4095 diverge at generated token 11 / 5 /
14 / 4 (random-token prompts have near-tied logits; the down-sum order changed). Against the reference on the public
prompts: dev-eval below.

**dev-eval 002 (1 round, all lengths): PASSED, score 1.6216** (prefill 1.919, decode 1.370; round 007's formal result:
1.6082 / 1.8355 / 1.4091); p256 2.022 / 1.329, p1024 2.310 / 1.428, p4096 1.514 / 1.355 (candidate ms 33.4 / 885.7,
94.5 / 888.5, 347.0 / 907.0; the decode speedups are lower only because the reference decoded faster in this round:
1177 / 1269 / 1229 ms); mismatching steps [0], [0], [1].

**dev-eval 003 (2 rounds, all lengths, the final engine): PASSED, score 1.6426** (prefill 1.927, decode 1.400);
p256 2.044 / 1.357, p1024 2.326 / 1.458, p4096 1.506 / 1.387; candidate ms per round: prefill 33.6 / 33.6, 94.3 / 94.4,
347.4 / 346.7, decode 885.3 / 885.7, 888.3 / 888.7, 907.0 / 907.0; mismatching steps [0, 0], [0, 0], [1, 1].
Clean build (copy without target/, empty NEFF cache, cores 0-15): 16 min 1 s; tokens and times identical to the
in-tree build (`SEED=11`: 1, 256, 1000, 1024, 2049, 4096, 4100 + 16).

Thermal note (also in round 008): back-to-back 4,096-token prefills in one process read 354 / 398 / 434 ms; compare
first requests only.

### Next (estimates)

- MP for the 512 / 2048 buckets (not timed; they keep the padded path).
- p256: neither MP nor the round-008 levers move it; per layer ~210 us MLP (TensorE and the weight DMA both near
  their rates), ~120 us attention phase, two exposed 57 us all-reduces.
- The prefill MLP is now 25 matmuls per kt / output tile; the remaining TensorE time at 4096 is ~3.9 ms of matmuls per
  4.9 ms layer (gaps ~11%, mostly the collective chains).

## Round 008 (2026-11-07): SP collective chains trimmed (GCH / SOC / SPS / SPG), runtime CC knobs and decode levers measured

Measured: whole requests, cores 8-15, engine alone, the same random prompts through each engine (`tests/tools/r8/sweep.py
LENS N ENGINE...` = `tests/dec.py` that survives failing variants; `pf.py LENS ENGINE...` = prefill only, 3 prompts per
length). Variants: `tests/tools/r8/var.sh NAME CORES TARGET...` copies the source to `/tmp/q32w8/NAME`, takes the other
NEFFs from the round-007 build (`/tmp/q32w8/base_target`) and rebuilds only the named targets with the
`PF_BUCKET_OVERRIDE` / `STEP_OVERRIDE` JSON in the environment. Timing-only builds (`timing_build.py WORKDIR PF_NOATT|PF_NOAR S`)
use a dummy cache key. Profiles: `tests/tools/prof.sh` + `neuron-explorer view` + `/tmp/q32w7/red.py` (copied to r8), then
`gapl.py PKL FROM_US TO_US` (TensorE gaps > 3 us with the instruction that ended them, interleaved with the collectives).
Back-to-back 4,096-token prefills heat the device: the 2nd / 3rd request of `pf.py 4096` read 377-430 ms for every
engine; compare first requests (or one 4,096 request per engine) only.

Where the time goes (timing-only builds, wrong tokens):

| build | p256 | p1024 | p4096 ms |
|---|---|---|---|
| round-007 engine | 33.7 | 100.4 | 370.3 |
| no attention (nkilib call skipped; the SP path needed its own switch) | 34.5 | 95.0 | 349.0 |
| no collectives (S=256: `PF_NOAR`; SP: reduce-scatters / all-gathers replaced by a local copy) | 25.9 | 75.0 | 319 |

So attention is ~5% at 1024 / 4096 (a 2x faster attention would not reach the 4% case rule) and hidden at 256, while
the collectives cost 23% / 25% / 14%. Profile of the p1024 layer (1.39 ms in the profiler, TensorE gaps ~390 us / layer):
(1) layer boundary: last block's down reduce-scatter -> finish + norm1 -> all-gather -> QKV of block 1 (~120 us exposed,
QKV(0) is the only work); (2) first MLP block: reduce-scatter(o_proj 0) -> norm2 -> all-gather -> gate/up (~150 us, only
o_proj(1) to overlap). Each chain also waited for one 5.2 MB gathered-activation load after the all-gather (~15-35 us) and
the reduce-scatter for one 2.6 MB slot-major write after the last output tile. Collective cost is not a fixed latency:
probe `p8_coal.py`: reduce-scatter of a 5.2 MB block 42-71 us, all-gather (5.2 MB out) 38-49 us, queued back to back.

**Mechanisms adopted (all bitwise identical: the same data, the same matmuls in the same accumulation order)** --
switches in `prefill_sp.py`, set in `targets.PF_BUCKET[1024 / 4096]`:
- `GCH=4`: the gathered block loaded in 4 DMAs of consecutive hidden tiles (`_gathered`), so the first matmuls start
  on the first chunk.
- `SOC=4`: o_proj / down outputs written to the reduce-scatter buffer in 4 chunks inside the tile loop
  (`_soc_flush`), not one 2.6 MB write after it.
- `SPS=2`: every SP collective split in 2 parts over hidden tiles (`_spbuf` part tensors). Reduce-scatter part g holds
  each core's own tiles [10 g, 10 g + 10) (local index 10 c + j), so part 0 is complete halfway through o_proj / down
  and its reduce-scatter runs while part 1 is computed (`_sps_flush`); consumers reassemble the shard (`_load_rs`). The
  norms write and all-gather their output in 2 parts of consecutive hidden tiles (`_norm_own_ag`), so QKV / gate-up start
  on part 0. Probe `p8_half.py`: a reduce-scatter of the two hidden-tile halves is bitwise equal to the whole block's on
  all 8 ranks (random bf16).
- `SPG=3`: the first 3 gate / up slab pairs of MLP block b issued before block b's norm2 (`_pre_gu`, inside `_oproj`
  right before the SPO norm2): the norm2's reduce-scatter-result loads block the in-order Sync queue, and the gate/up
  slabs behind them were the last ~50 us of the first-MLP-block gap.

| p1024 variant (prefill ms, 3 prompts) | | p4096 (first request) | |
|---|---|---|---|
| round 007 | 100.4 | round 007 | 370.3 |
| GCH 4 / SOC 4 / GCH 4 + SOC 4 | 99.6 / 99.0 / 98.4 | GCH 4 + SOC 4 | 371.5 (=) |
| GCH 4 + SOC 10 / 20; GCH 2 + SOC 10; GCH 5 + SOC 5; GCH 8 + SOC 2 | 98.5 / 99.2 / 98.3 / 99.0 / 99.7 | | |
| **+ SPS 2** | 98.2 | | |
| **+ SPG 3** (SPG 1 / 5: 98.5 / 99.5; 7: does not fit SBUF) | **97.7** | **GCH 4 + SOC 4 + SPS 2 + SPG 3** | **353.9** |
| + KTP 2 + KTQ 3 / KTP 3 (first gate-up pairs / QKV tiles kt-outer over the all-gather parts) | 97.9 / 97.6 (=, off) | SOC 8 / SPG 4 / SPG 2 | 355.0 / 353.3 / 353.6 |
| | | SPS 4 | does not fit SBUF |

Profile after (p1024, SPG 3): TensorE gaps 266 us / layer (was 390); what is left is mostly the collectives
themselves (second reduce-scatter half 39-47 us + first all-gather half 34 us per chain) and ~10-20 us of norm work.

Final engine (in-tree build of the final source, all prefill NEFFs rebuilt: 7 min 6 s): **tokens identical to round 007**
on 20 prompts of 1, 2, 100, 255-257, 511-513, 1000, 1023-1025, 2047-2049, 4095-4097, 4150 tokens + 64 generated
(`SEED=7 sweep.py`); prefill 1024 bucket 100.3 -> 97.6-99.0, 4096 bucket 370 -> 353.1-353.7 (4097: 385.4 -> 368.8, 4150:
1135.7 -> 1118.7) ms; 256 / 512 / 2048 buckets and decode unchanged (they use unchanged code paths).

### Rejected this round (numbers; switches kept off where the code stayed)

| change | result | |
|---|---|---|
| runtime collective algorithm `NEURON_RT_CC_ALG_TYPES` (engine env): `INTRA_FULL_MESH_ALG` | prefill 43.9 / 124.7 / 462.9, decode 934.5 / 936.9 / 956.5 ms (vs 34.0 / 100.3 / 370.4, 885 / 888 / 907) | slower |
| ring, kangaring, bw-opt / latency-opt / grouped mesh, hier | `NRT_INVALID` at load; single-step mesh: runtime assertion; `^INTRA_RDH_ALG`: `NRT_EXEC_OOB` | - |
| `NEURON_RT_DBG_RDH_DOUBLE_BUF=1` | `NRT_EXEC_OOB` | - |
| `NEURON_RT_DBG_MESH_DOUBLE_BUF=1`, `NEURON_RT_CC_CHANNEL_BUFFER_N` 2 / 8, `NEURON_RT_CC_CHUNK_SIZE` 256 KB / 4 MB, `NEURON_RT_DBG_EARLY_POSTING=1`, `NEURON_RT_DBG_CC_DMA_PACKET_SIZE=65536` | all within +-0.3% (374.2 at 4096 for MESH_DOUBLE_BUF) | no effect |
| a second CC stream: the same 8 ranks in a permuted replica group (`p8_streams.py`) | runs on stream 0, serialized (8 x 2.6 MB all-reduces 528 us either way); semantics: RS gives rank r the shard at its group position, AG slot j = rank PERM[j], AR bitwise equal. NKI exposes `stream_id` only for collective permutes | no lever |
| p256 gate/up prefetch during the attention all-reduce `PF_GROUP[256]` = (16, 2, 1) / (16, 4, 1); QKV prefetch off / + o_proj (PFQ 0 / 2) | 34.1 / 36.2; 34.5 / 35.9 vs 33.4 ms (concurrent weight DMA slows the all-reduce) | rejected |
| p256 `ARS`: the o_proj / down all-reduces in 2 / 4 parts over hidden tiles, part 0 computed first (`prefill_tp._ars_nt`) | down 2 parts 33.8, both 2 parts 34.9, down 4 parts 36.2 vs 33.4 ms; 4 parts changed a token (a smaller all-reduce message is **not** always bitwise equal) | rejected, not in the tree (`/tmp/q32w8/arssrc`) |
| p1024 `SPX`: block 0's attention (separate nkilib call, SPH) + o_proj between QKV(0) and QKV(1) (2-stage pipeline over the 2 blocks) | 103.5 (+ SPHN 102.9) vs 100.2 ms, tokens identical; profile: every nkilib call stalls ~35-58 us internally (gaps ending at `attention_cte.py:3881`) | rejected, not in the tree |
| nkilib context-parallel offset for block 1's queries alone | requires `scale=1.0` (pre-scaled q: other rounding) and disables causal compute skipping | not tried |
| decode: down-projection slabs of 4 / 6 tiles (13 / 20 KB descriptors instead of 6.6 KB, `SLD`) | 899.3 / 885.2 vs 885.3 ms (b512) | rejected |
| decode: the next layer's QKV (+ o_proj) prefetch issued only once the chain reaches the down all-reduce (`DLY` 1 / 2: a 1-element write into each prefetch tile from the two-core sum) | 927.4 / 929.5 vs 886 ms | rejected |
| decode: `HALF` (1,600 intermediate rows per core, no zero tiles), re-measured with today's step | 888.4 vs 885.6 ms (decode-only path, tokens identical on 3 x 64); profile: 8.21 vs 8.46 GB in 14.24 vs 14.19 ms, the same 215 us layer period | rejected |

Decode profile facts (b512 step, `prof_step`): DMA active 98.7%, 8.46 GB in 14.19 ms. The two physical cores of a rank
share ~650 GB/s (5-us buckets: each core 300-345 GB/s while both stream; one core reaches 414-446 GB/s while the other
is in its all-reduce dip); the pair averages ~600 GB/s (93%). Per DMA ring one transfer in flight, ~44 ns + size / 28.8
B/ns. The layer period (215 us) did not move with 3% fewer bytes (HALF) nor with larger descriptors (SLD): the step is
bound by the per-layer chain around the two all-reduces, not by bytes.

Fixed per-request overhead at p256 (emb + head + host, `ENGINE_DBG_PF_LAYERS=0`): 1.4 ms; one 16-layer execution 8.0 ms.

### Next (estimates)

- p1024 / p4096: the remaining chain time is the collectives themselves (~40 us per half); splitting further runs out
  of SBUF (SPS 4) or adds per-op cost. An overlap of the first MLP chain needs TensorE work independent of the attention
  output; another nkilib call costs ~50 us.
- p256: both all-reduces stay exposed (~120 us / 500 us layer); prefetch and split variants all lose.
- decode: HBM / chain-bound; no lever found this round.

Tools and probes added: `tests/tools/r8/*` (sweep, pf, gapl, pairtl, dmaconc, ratehist, mk / var / link / timing_build,
ccview / cclist, runp), `probes/p8_streams.py`, `p8_coal.py`, `p8_half.py`.

## Round 007 (2026-11-07): sequence-parallel residual in the prefill (SP, `kernels/prefill_sp.py`), 1024 / 4096 buckets

Measured: whole requests, cores 8-15, `SEED=7 tests/dec.py` (same random prompts through each engine), engine alone;
experiments link only the changed NEFFs (`tests/tools/exp7.sh`, work dir `/tmp/q32w7`, base = `/tmp/q32w7/base` =
the round-006 commit rebuilt). Profiles: `tests/tools/r7/prof.sh` (neuron-explorer rank 0 -> reduced pickle; scripts in `tests/tools/r7`) +
`tl2.py` (per-engine busy per time bucket, CC ops), `gaps.py` (TensorE idle gaps by the instruction that ended them).

**Mechanism.** The two per-block bf16 all-reduces of every prefill layer (o_proj and down partials) become
reduce-scatters over token shards plus all-gathers of the normalised activations:
- token t of a 512-token block belongs to rank (t % 512) // 64, inside the rank to physical core ((t % 64) // 32);
  only the owner keeps the residual of its 32 tokens current (in `s_h`, natural layout, own columns addressed through a
  register = 2 * `ncc.rank_id()` + pid; other columns go stale);
- norm1 / norm2 run on the own 32 tokens only (1/16 of the VectorE work and h bytes), their bf16 outputs are
  all-gathered (`[16 slots, 128, HT, 32]` per block, one tensor per block and layer: tensor-level dependencies),
  the matmuls read the gathered block through a 2-free-dim moving AP (slot, token) = natural token order;
- o_proj / down write their bf16 sums in slot-major layout (one DMA per block) and reduce-scatter them; the residual
  adds (h + ra, + rm) touch only the own 32 tokens;
- LPN layers per execution: the next layer's norm1 + all-gather of block b is issued right after the MLP finish of
  block b (`SPT` blocks later), so the next layer's QKV starts without waiting; the last blocks' finishes are deferred
  into the next layer's QKV loop (`SPD`) so the in-order TensorE does not wait on the last reduce-scatter;
- the head (`head_sp`): every rank masks its (possibly stale) column plen-1 to the owner's (rkb == owner's rkb) and an
  fp32 all-reduce of [128, 40] sums one non-zero term (exact).
Probe (`probes/p7_rs_ag.py`, run with `tests/tools/r7/runp.py`): the reduce-scatter result equals the all-reduce result bitwise on all 8 ranks
(random bf16 [128, 40, 512]); per 5.2 MB block: all-reduce 99 us, reduce-scatter 60.5 us, all-gather 52 us (queued
NEFF executions, 8 vs 1 ops). So every element is computed exactly as before: **tokens identical**.

| step (p4096 unless noted) | prefill ms |
|---|---|
| round 006 engine | 394.7 (p1024 110.2) |
| SP first version: all o_proj blocks, then MLP; 8 per-shard DMAs with 64-128 B runs | 479.6 |
| + MLP lag 3 (`SPL`) / + norm2 of the next MLP block after this block's gate/up (`SPN`) | 473.4 / 420 |
| + slot-major layouts (one 2.5 KB-run DMA per gather / scatter), lag 1, finishes 2 blocks late (`SPT=2`) | 394.0 |
| + one HBM tensor per block (no false CC dependencies), norm2 ahead of the next reduce-scatter (`SPO`) | 391.6 |
| + norm1 all-gathers 2 blocks ahead of QKV instead of all up front (`SPA=2`; GpSimd triggers CCs in order) | 391.2 |
| SPL / SPO = 2 / 2 (2 / 1: 411.9, 3 / 1: 416.1, 3 / 2: 412.8) | 390.5 |
| **LPN 2 / 4 / 8 layers per execution** | 382.0 / **377.5** / 378.2 |
| + deferred last finishes (`SPD=1`) | **375.8** |
| SPT 1 (with SPD 1 / 2) | 376.7 / 376.8 |
| p1024: SP, LPN 4 / 8 (SPL 1, SPO 1, SPN 1, SPA 1) | 107.7 / 107.3 (SPL 2: 107.8; no SPO, SPA 2: 109.4) |
| p1024: + SPD 1 (no SPT) | 108.4 |
| **p1024: + SPT 1 + SPD 1** (the next layer's norm1 of block 0 during MLP block 1) | **101.0** (SPT 1 alone: 101.8) |

Later steps (same method; the adopted switches are in `targets.PF_GROUP` / `PF_BUCKET`):

| change | result | verdict |
|---|---|---|
| **`SPP` (4096): the down projection of MLP block b - 1 issued after block b's gate/up** (its activation exchange through HBM had a whole gate/up to land; finishes only after the down they need is emitted) | p4096 376.0 -> **374.4**; p1024 101.0 -> 105.6 | 4096 only |
| **LPN 16 at 1024** / 32 | 101.05 -> **100.45** / 100.2 | 16 (32 builds longer) |
| **LPN 8 at 4096 (with SPP)** / 16 | 374.4 -> **372.5** / 371.6 (16: 12-min compile of that NEFF) | 8 |
| **`SPQ` (4096): QKV tile nt's q/k norm + RoPE + cache writes emitted after tile nt + 1's matmuls** (in-order TensorE no longer waits for the norm / rsqrt / RoPE chain of each tile) | p4096 372.9 -> **370.0**; p1024 100.5 -> 101.4 | 4096 only |
| `SPH`: the first 1 / 2 blocks' attention as a separate nkilib call (`probes/p7_att_split.py`: the first 512 rows alone are bitwise equal to the full call's), their o_proj + reduce-scatter + norm2 chains during the full call | p4096 381.3 (SPH 1) / 384.3 (1 + its norm2 ahead) / 388.8 (2) / 390.8 (2 + norm2) vs 376; p1024 106.6 / 106.0 vs 101.2 | rejected (the extra call costs more than the gap it hides) |
| nkilib prefix caching (block 1 with block 0 as prior) | does not compile: "Prefix Caching ... only support scale=1.0" (would need pre-scaled q: different rounding) | - |

Final engine vs round 006 (in-tree build of the final source: 7 min 44 s): **tokens identical** on 20 prompts (`SEED=7`, list
below) and 16 more (`SEED=11`: 1, 3, 64, 130, 256, 300, 512, 700, 1023, 1024, 1100, 2048, 3000, 4096, 4100, 4200 + 64);
prefill 1024 bucket 110.3 -> **100.4**, 4096 394.9 -> **370.0** ms, 4100 / 4200 tokens (4096 prefill + decode steps)
452.1 -> 428.1 / 1867 -> 1845 ms; others and decode unchanged. Long output: 200-token prompt + 4,000 + 408 tokens all
returned, one more 1-token request answered.

**dev-eval 002 (2 rounds, all lengths, the final engine): PASSED, score 1.6018** (prefill 1.834, decode 1.399); p256
2.011 / 1.356, p1024 2.177 / 1.445, p4096 1.410 / 1.397; candidate ms per round: prefill 33.8 / 33.7, 100.6 / 100.3,
370.4 / 369.8, decode 885.6 / 885.7, 888.2 / 888.7, 909.4 / 907.3; 0 mismatching steps.

First candidate (LPN 8 / 4, no SPP / SPQ; in-tree build, all prefill NEFFs rebuilt: 6 min 37 s): **tokens identical** on 20 prompts of
1, 2, 100, 255-257, 511-513, 1000, 1023-1025, 2047-2049, 4095-4097, 4150 tokens + 64 generated (`SEED=7`); prefill
1024 bucket 110.3 -> 101.1, 4096 394.7 -> 375.6 ms (others unchanged), decode unchanged.

**dev-eval 001 (1 round, all lengths): PASSED, score 1.6003** (prefill 1.828, decode 1.401); p256 2.023 / 1.346,
p1024 2.171 / 1.469, p4096 1.390 / 1.391 (candidate ms 33.8 / 885.4, 101.0 / 888.4, 375.8 / 907.2); 0 mismatching steps.

Profile of the adopted p4096 NEFF (4 layers, 20.94 ms = 5.23 ms / layer, round 006: 5.64): phase A ~700 us (QKV),
attention ~700 us, MLP blocks ~470 us; TensorE idle gaps per layer: ~180 us at the first MLP block (o_proj(0) ->
reduce-scatter -> norm2 -> all-gather chain), 25-60 us at each later MLP block start, ~50 us at the layer boundary.
p1024 (8 layers, k8a before SPT/SPD): 1.48 ms / layer with ~165 us at the layer boundary (fixed by SPT 1 + SPD 1) and
~160 us at the first MLP block.

Clean build (copy without target/, empty NEFF cache, cores 0-15): **15 min 12 s** (pf4096_layer with LPN 8 compiles in
~6 min); tokens and times identical to the session build (`SEED=11`, 1 / 256 / 1024 / 2049 / 4096 / 4100 + 16).

Profile of the final p4096 NEFF (8 layers, 41.37 ms = 5.17 ms / layer, round 006: 5.64): the largest TensorE gap left
is ~150 us at the first MLP block of every layer. It is the collective queue, not the schedule: after the attention,
reduce-scatter(0) -> reduce-scatter(1) -> all-gather(0) run back to back (~52 + 66 + 53 us, CC ops are serial) while
TensorE has only ~25 us of o_proj per block to fill. Tried and worse with the final code: SPO 1 (norm2(0) ahead of
reduce-scatter(1)) 377.2, SPO 1 + SPL 3 381.3 vs 370.0 ms (the norm's sum-of-squares matmul then stalls the in-order
TensorE on the reduce-scatter). Other gaps: 20-75 us at each later MLP block start, ~50 us at layer boundaries.

### Next (estimates)

- p4096 / p1024 first MLP block (~150 us / layer, 3% at 4096, ~10% at 1024): needs TensorE work that does not depend on
  the attention output, or a cheaper first-block chain; a separate first-block attention call (SPH) cost more than it hid.
- The norms' cross-partition sum of squares is a TensorE matmul (bitwise reference path); a VectorE/GpSimd reduction
  would decouple TensorE from the collective chain but changes the summation order (not bitwise).
- Decode untouched this round (see round 006: local optimum of the DMA-queue levers).

Not done: SP for the 512 / 2048 buckets (not timed; they keep the round-006 path), S=256 (HS path: its two exposed
all-reduces would become reduce-scatter + norm + all-gather chains of about the same length).

## Round 006 (2026-11-06/07): prefill residual kept in SBUF at S=256 (HS), norm chains pipelined over hidden-tile chunks (HSC / NCH)

Measured: whole requests, cores 8-15, `SEED=7 tests/dec.py` (same random prompts through each engine), engine alone;
experiments link only the changed NEFFs with the round-005 build's other NEFFs (`tests/tools/exp.sh`, now `/tmp/q32w6`,
base = `/tmp/q32w5/clean`, the round-005 clean build). Stable to +-0.3 ms; occasional one-off outliers (130-200 ms for a
34 ms prefill) appeared with every engine while the machine load was ~30-56 (other runs building), never reproduced.

Diagnosis (p256 layer NEFF profile, 16 layers, 510 us per layer in the profiler): each of the two all-reduces (58 us,
2.6 MB) is followed by ~40 us where only one engine works at a time before TensorE restarts: the result load (DMA) ->
one big VectorE add of all 40 hidden tiles -> one big Scalar square -> the 40 sum-of-squares matmuls -> rstd -> the
norm multiplies. The same serial load -> square -> matmul chain starts every norm of the larger buckets (`_norm_blk`,
`_norm_pre`, `_finish_norm`).

| change | result | verdict |
|---|---|---|
| **`HS` (S=256): the residual h [128, 40, 256] fp32 (40 KB / partition) lives in SBUF for the 16 layers of an execution** (both cores hold all of it and apply the same adds; each writes its half to HBM after the last layer): no h HBM reads in norm1 / finish+norm2, no h write / core barriers per layer | p256 38.1 -> **36.3 ms**; tokens identical | adopted |
| HS with 32 / 64 layers per execution | 36.2-36.5 / ~38 ms | 16 kept |
| **`HSC` (with HS): the residual add, the square and the sum-of-squares matmuls in chunks of hidden tiles** (same per-element ops, the matmuls in the same kt order, so bitwise identical) | p256 36.3 -> 34.0 (4 chunks) / **33.7 (8)** / 33.6 (10) / 33.8 (20) | 8 adopted |
| **`NCH`: the same chunking for the h load + square + sum of squares of `_norm_blk`, `_norm_pre`, `_finish_norm` (S >= 512)** | p1024 113.7 -> 110.9 (4) / **110.4 (8)** / 110.3 (10) / 110.4 (20); p4096 413.3 -> 402 (5) / 398.5 (8) / 400.5 (10) / **394.5 (20)** | 8 for 512 / 1024 / 2048, 20 for 4096 |

Also measured with HS / NCH in place (all rejected, switches kept off):

| variant | result |
|---|---|
| p4096 / p1024 `PNA=1` again (next block's norm1 interleaved with this block's QKV tiles) | 395.5 vs 394.7 / 110.8 vs 110.4 |
| chunked h loads issued from the Scalar hwdge queue (`NCQL` 1 = all, 2 = odd chunks; Sync is 58-89% busy issuing DMAs in phase A) | p4096 399.4 / 397.7, p1024 114.3 / 112 |
| MLP-side norms (PN / FN0 / LATE norm2) with their own chunk count `NCH2` 1 / 2 / 4 / 40 (phase A kept) | p4096 410 / 404.7 / 403.2 / 402.7 vs 394.4 (the MLP-side chunking alone is worth 16 ms) |
| phase A chunks 8 / 40 with MLP-side 20 at 4096; 20 / 4 with 8 at 1024 | 393.9 / 398.8 (noise level); 110.8 / 111 |
| p4096 `PMF` lag 1 / 2 with the finishes on the Scalar queue (`AQL=4`; PMF + FN0 block 0 fixed to use `_hbm_finish2`) | 400.3 / 409.5 (tokens identical) |
| p1024 `FN0=1` (now with the chunked `_finish_norm`) / + `PN=1` | 110.1 (=) / 112.4 |

Profiles after this round (random inputs, rank 0): p4096 layer 5.64 ms (TensorE 81% active): phase A ~800 us with
the norm and the QKV of consecutive blocks still alternating (VectorE ~55%, Scalar ~27%, TensorE ~30%, Sync 58-89%
busy issuing ~62 DMAs per block, DMA ~200 GB/s: no single engine saturated), attention ~700 us (nkilib: VectorE and
Scalar saturated), MLP 8 x ~420 us, tail ~400 us of DMA-bound finishes (105 MB per core). p1024 layer 1.55 ms: ~200 us
idle from the end of o_proj to the first MLP block (attention all-reduce 95 us + finish + norm2), ~150 us tail.
p256 (HS) 510 us per layer: the two all-reduces (58 us each) are the only idle time left.

Final engine vs round 005 (full build in this tree from an empty NEFF cache: 12 min 25 s; rebuilt after the last
source edits, 3 min 23 s): **tokens identical** on 20 prompts of 1, 2, 100, 255-257, 511-513, 1000, 1023-1025,
2047-2049, 4095-4097, 4150 tokens + 64 generated (`SEED=7 tests/dec.py`) and on 16 more (`SEED=11`: 1, 3, 64, 130,
256, 300, 512, 700, 1023, 1024, 1100, 2048, 3000, 4096, 4100, 4200 + 64; 4100 / 4200 run past the largest prefill
bucket); prefill 256 bucket 38.1 -> 33.6, 512 67.4 -> 66.0, 1024 113.9 -> 110.1, 2048 216.9 -> 204.4, 4096 413.3 ->
394.5 ms; decode unchanged (885.2 / 888.2 / 906.7 ms at b512 / b1152 / b4224).

**dev-eval 001 (1 round, all lengths): PASSED, score 1.5675** (prefill 1.763, decode 1.394; round 005's formal
result 1.5038 / prefill 1.625); p256 2.078 / 1.362, p1024 1.988 / 1.443, p4096 1.327 / 1.378 (candidate ms 33.6 /
885.3, 109.9 / 888.3, 393.9 / 906.9); 0 mismatching steps.

**dev-eval 002 (2 rounds, all lengths, the final engine): PASSED, score 1.5535** (prefill 1.737, decode 1.389); p256
1.994 / 1.343, p1024 1.987 / 1.453, p4096 1.324 / 1.374; candidate ms per round: prefill 33.6 / 33.7, 110.2 / 110.3,
394.5 / 394.1, decode 885.2 / 885.6, 888.0 / 888.7, 906.8 / 907.4; lower than dev-eval 001 only because the reference
prefilled p256 faster in these rounds (67.0-67.2 vs 69.8 ms). 0 mismatching steps. Long output: 200-token prompt +
4,000 + 408 tokens all returned, and one more 1-token request is answered (step at position 4,607); the round-005
engine does exactly the same (same token).

What changed in the engine: `kernels/prefill_tp.py` (`HS`: `_norm_sb`, `_sb_add`, the residual tile `hs`; `HSC` /
`NCH`: `HSCH`, `NCHK`, `_load_sq`, chunked branches of `_norm_blk`, `_norm_pre`, `_finish_norm`; experiment
switches `NCQL`, `NCH2` off; `_hbm_finish2` DMAs through `_hqf` (= `_hd` unless `AQL & 4`)), `kernels/targets.py`
(`PF_BUCKET`: 256 HS + HSC 8, 512 / 1024 / 2048 NCH 8, 4096 NCH 20; `PF_BUCKET_OVERRIDE` env for experiments).
Decode NEFFs: built from the edited `step_tp.py` / `layer_decode.py` with every new switch off (same instruction
stream; tokens and times identical to round 005).

### Next (estimates)

- p4096 phase A (~800 us / layer): consecutive blocks' norm and QKV still alternate although no engine is saturated;
  the Sync queue issues ~62 DMAs per block (40 of them the chunked h loads). Fewer, larger h-load DMAs with the square
  still chunked (DMA chunk != compute chunk) is untried.
- p4096 tail (~400 us / layer, DMA-bound finishes): keeping the finish results for the next layer's norm1 (one
  multi-layer NEFF) would remove one h read + write per block; LPN > 1 at 4096 lost in round 004 (463.8 vs 430.3).
- p1024: the attention all-reduce of block 0 (95 us) plus the finish / norm2 before the first MLP block.
- HS for S=512 (h = 80 KB / partition) is untried (not a timed length).

### Decode experiments (all rejected; switches kept in `step_tp.py`, off)

Profile of the b512 step (`neuron-explorer`, rank 0, both physical cores; new analysis scripts in `/tmp/q32w6/a*.py`
list every Sync / GpSimd / Vector instruction with the semaphores it waits on): layer period 217 us, the weight stream
stops for ~10 us at the down all-reduce and runs at 150-280 GB/s for ~14 us after it. Causes found:
1. **The static all-reduce read-back (round 004 `RBQ=2`) is triggered from the Sync engine's in-order stream**
   (`qSyncSpillReload0`): it waits for the all-reduce, so no weight DMA is issued from ~9 us before the all-reduce
   until it ends (seen on both all-reduces; at the o_proj all-reduce the backlog of 20 KB gate/up slabs hides it).
2. The all-reduce input write (`hb`, GpSimd software DGE) takes ~6.5 us from issue to completion: its transfers start
   ~4.3 us after the issue, spread over 1.7 us across the 16 DMA rings that are busy with weight descriptors.
3. TensorE signals the end of a slab only where a compute engine reads PSUM (end of a GEMV), so a DMA reusing a slab
   buffer (WAR) waits for the next such signal: the next layer's QKV prefetch waited 13 us for the down GEMV.

| variant (decode 64 tokens at b512, base 885.2-885.5 ms) | ms | |
|---|---|---|
| hb write on the Scalar hwdge queue (`CCQ=3`): the write still takes 6.8 us | 909.6-910.2 | rejected |
| hb write / read-back as static DMAs triggered by GpSimd or Scalar (`CCQ`/`RBQ` 4, 5) | do not compile ("engine param can only be set when dge_mode=hwdge") | - |
| read-back on the Scalar hwdge queue (`RBQ=3`) | 906.2 | rejected |
| weight DMAs split into 2 descriptors per partition (`DSP=2`, 10 KB elements): hb write latency unchanged (7 us) | 885.2 | no gain |
| o_proj slabs of 10 tiles (`SLO=10`, `PFO=2`) | 890.4 | rejected |
| down all-reduce only: hb write hwdge Sync / Scalar / static (`CCQM` 6 / 3 / 2) | 892.6 / 899.6 / 885.6 | rejected |
| **ring of N slab buffers for every weight GEMV of the step** (`RING=N`: 20 KB slabs, QKV 2 tiles, o_proj 20, gate/up 2, down 6, LM head 2; loads issued N-1 slabs ahead across GEMV / layer boundaries; bitwise identical) | N=5: 918.5, N=7: 906.3 | rejected |
| ring + each slab's PSUM columns copied to SBUF right after its matmuls (`RCP=1`, forces a TensorE signal per slab: confirmed in the profile) | N=6 / 7: 908.3 / 908.4 | rejected: the down all-reduce dip stays (~8 us) and new ones appear |
| per-slab PSUM copies with the compiler's own slab allocation (`BCP` 1 = down, 7 = all) | 886.6-889.0 / 888.5 | rejected |
| norm rsqrt on VectorE | does not compile (rsqrt is GpSimd-only) | - |

Conclusion: the decode step is at a local optimum for every DMA-queue / buffer-allocation lever reachable from NKI;
what is left (~20 us per layer) is the all-reduce chain itself (hb write latency, the collective, the in-order Sync
engine blocked on the read-back trigger). Not tried: a decomposition with fewer all-reduces; bf16 hb (numerics).

## Round 005 (2026-11-06): prefill norm2 issued one MLP block early (PN), first-block fused finish (FN0)

Measured: whole requests, cores 8-15, `SEED=7 tests/dec.py` (same random prompts through each engine), engine alone;
experiments link only the changed NEFFs with the session build's other NEFFs (`tests/tools/exp.sh`, now `/tmp/q32w5`).
Baseline (round-004 engine rebuilt): prefill p1024 116.6-117.0, p2048 225.1-225.4, p4096 430.0-431.7 ms.

Diagnosis (`neuron-explorer` json of the S=4096 layer NEFF, rank 0; new: the gap analysis keys each TensorE gap by the
source line and the semaphore of the instruction that ended it): TensorE is ~91% busy inside MLP blocks but drops to
10-40% for ~120 us at each of the 7 block transitions, waiting on `_norm_blk`'s sum-of-squares matmul (line 82),
which waits on the Scalar engine's square of h, which waits on the attention all-reduce finish (h write, core
barrier, h read on the in-order Sync queue). TensorE is in order, so the norm2 of block b+1 placed at the block
boundary stalls it. Phase A (norm1 + QKV, ~800 us) is VectorE-bound: the norm's fused multiply is 610 us of it,
duplicated on both cores. Attention (~650 us) is Scalar-bound (nkilib exp). The layer ends with ~450 us waiting on
the last MLP all-reduce (275 us) and the finishes.

| change | result | verdict |
|---|---|---|
| **`PN=1`: block b+1's attention finish + barrier + norm2 issued right after block b's gate/up** (computed into a second x2 tile, used by block b+1) | p4096 430.4 -> **416.4**, p2048 225.4 -> 221.4, p1024 116.9 -> 116.0; tokens identical | adopted for 4096 |
| **`PN=2`: as PN=1, but the norm split into load + square (`_norm_pre`) and the TensorE / VectorE part (`_norm_post`), the posts interleaved at 1/3 and 2/3 of the down loop** | p1024 116.9 -> **113.7**, p2048 -> **220.6**, p4096 423.4 (worse than PN 1) | adopted for 1024, 2048 |
| **`FN0`: the first MLP block's attention finish fused with norm2 in SBUF (FN for block 0 only; the h update of block 0 done at the end with `_hbm_finish2`)** | p4096 416.6 -> **413.6**, p2048 220.8 -> **217.0**, p1024 113.8 -> 115.9 | adopted for 4096, 2048 |
| `PMF` = lag k (MLP all-reduce of block b - k finished during block b instead of all in the tail) | without PN: lag 2 430.5, lag 3 433.3 vs 430.2; with PN: lag 1 424-430 / 1024 113.5 (=), lag 2 432.4 vs 416.2 | rejected |
| `PNA`: phase A norm1 of block b+1 split and interleaved with block b's QKV tiles | p4096 417.0 vs 416.3, p1024 (with PN 2) 116.9 vs 113.7 | rejected |
| `XN`: each core normalises only its half of every 512-token block (half the h reads, squares and VectorE multiplies), exchange through an HBM buffer `pf_x1x` + core barrier | p4096 416.6 -> 438.8, p1024 113.8 -> 120.4 | rejected |
| `XN=2`: XN pipelined one block ahead inside the QKV loop | p4096 413.3 -> 421, p1024 113.8 -> 114.3 | rejected |
| `PN=3`: the next block's finish + norm2 in the middle of this block's gate/up instead of after it | p4096 413.6 -> 431.4, p1024 114.4 -> 119.7 | rejected, removed |
| p4096 `ILV` 2 / 4 re-tuned with PN + FN0 | 420-426 / 418.6 vs 413.5 (ILV 3) | ILV 3 kept |
| decode: the gate/up prefetch slabs (PFG 13) on the Scalar engine's hwdge queue (`GQS` 1), or alternating Sync / Scalar (`GQS` 2), so two queues stream weights during the attention chain | b512 885.3 -> 923.4 / 915.5 | rejected, reverted |
| decode: K/V cache writes through write-only aliases `kcw` / `vcw` bound to the same tensors (the next layer's cache loads, and the weight DMAs behind them on the Sync queue, waited on the GpSimd cache-write DMA ~0.6 ms per step) | b512 884.7 -> 890.2, b1152 887.9 -> 889.2, b4224 906.4 -> 906.3; tokens identical | rejected, reverted |

Final engine vs the round-004 engine (rebuilt from its commit, same NEFF cache): **tokens identical** on 20 random
prompts of 1, 2, 100, 255-257, 511-513, 1000, 1023-1025, 2047-2049, 4095-4097, 4150 tokens + 64 generated (`SEED=7
tests/dec.py`); prefill 256 bucket 38.0 -> 38.2 (unchanged), 512 67.0 -> 67.4, 1024 116.7 -> 113.7, 2048 225.1 -> 216.8,
4096 430.6 -> 413.2 ms; decode unchanged (885.5 / 888.4 / 907.2 ms at b512 / b1152 / b4224).

**dev-eval 001 (1 round, all lengths): PASSED, score 1.5123** (prefill 1.633, decode 1.4005; round 004's formal
result: 1.4927, prefill 1.5947); p256 1.782 / 1.369, p1024 1.933 / 1.468, p4096 1.264 / 1.367 (candidate ms 38.0 /
886.0, 113.2 / 888.8, 413.0 / 907.3); 0 mismatching steps.

**dev-eval 002 (2 rounds, all lengths, the final engine): PASSED, score 1.4973** (prefill 1.628, decode 1.377); p256
1.778 / 1.330, p1024 1.922 / 1.438, p4096 1.263 / 1.365; candidate ms per round: prefill 38.0 / 37.9, 113.6 / 113.7,
413.2 / 413.2, decode 885.8 / 885.8, 889.1 / 888.8, 907.8 / 907.7 (decode identical to round 004's formal numbers);
the decode speedups are lower than dev-eval 001 only because the reference decoded faster in these rounds
(1173-1283 ms vs 1186-1426 in round 004's formal run). 0 mismatching steps. Clean build (copy without target/, empty
NEFF cache, cores 0-15): 12 min 20 s; NEFF bytes differ from the session build (compilation is not byte-reproducible)
but tokens and times are identical (`SEED=11`, 1 / 256 / 1024 / 2049 / 4096 / 4100 + 64). Long output: 200-token
prompt + 4,000 + 408 tokens (to position 4,608) all returned.

What changed in the engine: `kernels/prefill_tp.py` (`PN`, `_norm_pre` / `_norm_post`, `FN0`; experiment switches
`PMF` as a lag, `PNA`, `XN` off), `kernels/targets.py` (`PF_BUCKET` 1024: PN 2; 2048: PN 2 + FN0; 4096: PN 1 + FN0).
Decode NEFFs are byte-for-byte the round-004 ones (same cache keys).

### Next (estimates)

- p4096 layer (~6.0 ms in the profiler): phase A ~800 us VectorE-bound (norm multiply 610 us, both cores), attention
  ~650 us Scalar-bound (nkilib exp at ~0.7 ns / element / lane), ~450 us tail on the last MLP all-reduce + finishes,
  remaining MLP transition gaps ~20-40 us each. Splitting the last block's MLP all-reduce in two (hidden-tile
  quarters) could hide ~100 us / layer (not tried).
- p1024: ~200 us idle per layer waiting for the first attention all-reduce + finish + norm2, ~200 us tail.
- Decode: unchanged; see the profile notes above (SBUF-capacity-bound prefetch around the two all-reduces).

Decode profile (b512 step, 14.21 ms, 8.46 GB, DMA active 98.8%): the stream stops ~10 us at the o_proj all-reduce and
~20 us at the down all-reduce per layer (~220 us period). Gaps in the Sync queue's issue: 4.3 ms per step waiting on
TensorE (slab buffers freed only when consumed: SBUF-capacity-bound prefetch), 2.7 ms on its own DMA completions,
0.6 ms on GpSimd (the cache writes, see kcw above). No cheap lever found.

## Round 004 (2026-11-06): p256 layer groups, p4096 o_proj/MLP interleave, decode all-reduce read-back off the weight queue

Measured: whole requests, cores 8-15, `tests/dec.py` (same random prompts through each engine), engine alone.

Baseline = the round-003 engine rebuilt from its commit: decode 64 tokens 893.4 (b512) / 893.5 (b1152) / 912.4 ms
(b4224), prefill 41.0-41.1 / 116.6 / 441.2 ms (p256 / p1024 / p4096). Experiments link only the changed NEFFs with
the baseline's other NEFFs (`tests/tools/exp.sh`).

| change | result | verdict |
|---|---|---|
| decode: read-back of each all-reduce result with static descriptors (`dge_mode.none`, `STEP RBQ=2`), off the Sync weight queue | b512 893.4 -> 889.1, b1152 -> 889.5, b4224 912.4 -> 907.5; tokens identical | adopted |
| + next layer's first 10 o_proj slabs prefetched before the down all-reduce (`PFO=10`) | b512 885.2-885.5, b4224 906.6-906.7 (PFO 5: 888.2 / 907.0; 15: 886.1 / 912.8-915.7; 20: 885.9 / 911.9) | adopted |
| decode: K/V projection split over the two physical cores (`KVS`: core 0 projects k, core 1 v, sendrecv; 1.3 MB/layer less weight stream) | b512 893.5 -> 900.5, b4224 912.5 -> 922.6; prefill (same split) p256 41.1 -> 40.5, p4096 441 -> 473; tokens identical | rejected |
| decode: intermediate dim exactly 1600 rows per core (12 tiles + a 64-row half tile) instead of 13 tiles with 8 zero tiles at the end (`HALF`; host re-layout `layout_gu_half` / `layout_d_half`) | HBM read per step -3% (8.46 -> 8.21 GB), but b512 894-897, b4224 916.5, p1024 prefill 116.7 -> 120.9; streams diverge at tokens 10-64 (down-proj partial sums regrouped) | rejected (code kept, off) |
| decode: next layer's QKV prefetch into tiles allocated once per step (`PQP` 1 / 2) | 895.3 / 893.6 (no change); pinning them with `address=` fails to compile (SBUF full: "Spilling is disabled") | rejected |
| decode: all-reduce input write with static descriptors (`CCQ=2`) | b512 894.7 (vs 893.4) | rejected |
| decode: K/V cache bucket loads with static descriptors (`KVQ=2`) | b512 894.5-900, b4224 931 | rejected |
| decode: SBUF all-reduce of the two-core sum instead of the HBM one (`HBMCC=2`) | b512 1024 ms | rejected |
| **p256: 16 layers per layer-NEFF execution, next layer's QKV slabs prefetched before the MLP all-reduce (`PF_GROUP[256] = (16, 0, 1)`), with `FN` + `KVS` for this bucket (`PF_BUCKET`)** | p256 41.1 -> **37.9 ms** (FN 40.4, KVS 40.6, FN+KVS 40.2 without grouping; LPN 8: 38.7-42, 32: 38.0, 64: 40.2, 16 + PFG 2: 39.0); tokens identical | adopted |
| prefill grouping for other buckets | p1024 LPN 8 / 16 (PFQ): 119-121 vs 116.6; p4096 LPN 4: 463.8 vs 430.3; S=512 FN+KVS: 66.7 -> 68.5 | not used |
| **p4096: o_proj block b interleaved with MLP block b - 3 in one loop (`PF_ILV[4096] = 3`)** instead of all o_proj blocks then all MLP blocks | p4096 441.4 -> **430.7** (ILV 1: 448, 2: 437, 4: 439); p2048 ILV 2: 229 vs 224.8; tokens identical | adopted (4096 only) |
| prefill: first try of LPN grouping with gate/up prefetch (`PFG` 4) | S=256 LPN 8: 40.4 vs 40.6 ms; the prefetch DMAs wait on PE semaphores (SBUF reuse) | superseded |
| prefill: attention all-reduce finish fused with norm2 (`FN`) and MLP finishes pipelined one block later (`PMF`) | p1024 116.6 -> 120.8, p4096 441 -> 468 (FN+PMF) / 473 (FN); PMF alone p4096 453; first FN version had a cross-core race (wrong tokens: core 0 re-read h after core 1 had updated its half) | FN only for S=256 |
| prefill: QKV slabs loaded once per layer and kept for all token blocks (`QW`); norm sub-chunks of 128 (`QNS`) | p4096 444.9 (QW) / 435.4 (QW+QNS) / 432.7 (QNS) vs 430.6; p1024 119-120.6 | rejected |
| prefill: non-weight DMAs (finishes, all-reduce inputs; level 2 also norms, caches, activation exchange) with static descriptors (`SQL` 1 / 2) | p4096 441 / 613; p1024 117 / 185 | rejected |
| prefill: all-reduce input writes / activation exchange / finishes on the Scalar engine's hwdge queue (`AQL` 3 / 7) | p4096 436.5 / 437.2 vs 430.2 | rejected |
| prefill: half of the norm's fused multiply moved to Scalar (h*w, fp32) + GpSimd (*rstd) (`NSP2` 2 / 3) | p4096 472 / 444, p1024 127 / 123 (GpSimd is slow) | rejected |
| prefill: attention of the first half as a separate exact sub-problem, its o_proj + all-reduce overlapping the full attention (`SA`) | wrong tokens (nkilib on sliced HBM views) and slower (122.5 vs 116.7) | removed |

Candidate 1 (all adopted rows; full clean build 13 min): **tokens identical to the round-003 engine** on 18 random
prompts of 1, 2, 100, 255-257, 511-513, 1000, 1024, 1025, 2047, 2049, 4095-4097, 4150 tokens + 64 generated
(`SEED=7 tests/dec.py`); prefill p256 41.0 -> 38.3, p1024 116.6 -> 116.9, p4096 441.4 -> 430.7 ms; decode 64 tokens
b512 893.5 -> 886.2, b1152 893.4 -> 889.4, b4224 912.5 -> 907.9 ms.
**dev-eval 001 (1 round, all lengths): PASSED, score 1.4913** (prefill 1.591, decode 1.398); p256 1.768 / 1.351,
p1024 1.876 / 1.461, p4096 1.214 / 1.383 (cand ms 38.3 / 885.4, 116.7 / 888.5, 430.4 / 907.3); 0 mismatching steps.

Final engine (clean `rm -rf target && bash build.sh` in this tree: 12 min 53 s, cores 0-15): tokens identical to the
round-003 engine on 16 more prompts (`SEED=11`: 1, 3, 64, 130, 256, 300, 512, 700, 1023, 1024, 1100, 2048, 3000,
4096, 4100, 4200 tokens + 64; 4100 / 4200 run past the largest prefill bucket through decode steps); prefill
37.9-38.2 / 116.8 / 429.9-430.6 ms, decode 885.3 / 888.3 / 906.8 ms (b512 / b1152 / b4224).

**dev-eval 002 (2 rounds, all lengths, the final engine): PASSED, score 1.4834** (prefill 1.565, decode 1.406);
p256 1.679 / 1.349, p1024 1.879 / 1.470, p4096 1.215 / 1.402; candidate ms per round: prefill 43.3 / 37.9 (round 1 a
one-off outlier, as in round 003's dev-eval), 116.8 / 116.6, 430.1 / 430.2; decode 885.5 / 889.8, 888.5 / 888.1,
907.2 / 907.0; 0 mismatching steps in every timed stream.

What changed in the engine (all switches in `kernels/targets.py`; everything else is off by default and documented below):
- `step_tp.py`: `RBQ=2` (all-reduce read-back `dma_copy(..., dge_mode=none)`), `PFO=10` (`_prefetch` of the next
  layer's o_proj slabs next to the QKV prefetch, passed on as `(qkv, o)` and used by `_gemv_pf`).
- `prefill_tp.py`: STAGE 2 body is now a loop over `LPN` layers (`PF_GROUP`: 16 for S=256; the host reads the
  layers per execution from the size of the `wqkv` input, `PNL` in `src/engine.cpp`, and runs L / LPN executions); the
  o_proj blocks (NSP + LATE) moved into one loop with the MLP blocks (`it` loop, MLP block `it - LAG`; `ILV` sets LAG,
  default LAG = number of blocks = the old order); `PFQ` prefetches the next layer's QKV slabs; `FN` (S=256 only) adds
  each all-reduce result to all of h in SBUF and normalises from there (`_finish_norm`), h is written once at the end
  of the layer (`_hbm_finish2` adds both results); `KVS` (S=256 only): core 0 projects k, core 1 v, each writes both
  cores' copies of s_k / s_v / the caches.
- Unused switches kept for experiments: `HALF` (needs `src/model.h HALF = true` too), `PQP` / `PQA` (PQA does not
  compile), `KVQ`, `CCQ=2`, `HBMCC=2`, `SQL`, `AQL`, `NSP2`, `QW`, `QNS`, `PMF`, `PFG` (prefill).

Profiles (`neuron-explorer`, rank 0 of 8, `tests/tools/tl.py` buckets the json timeline per engine):
- decode b512 step 14.29 ms, per layer ~217 us = attention segment 43.6 + o_proj all-reduce 18.8 + MLP segment 141.8 +
  down all-reduce 12.2 us. The DMA streams at ~330 GB/s per physical core (peak ~358) except ~20 us of dips per layer
  around the two all-reduces. Fewer weight bytes (HALF: -3%) did not shorten the layer: the MLP segment stayed 142 us.
- prefill S=256 layer 588 us: QKV 60, attention 50, o_proj 30, attention all-reduce 57 (all engines idle), finish +
  norm2 57, MLP 190 (PE saturated), MLP all-reduce 57 (idle), finish 25. Throttled to a 50% utilisation limit ~50% of the time.
- prefill S=4096 layer 6.32 ms: phase A 850 us (PE ~70%, DMA ~200 GB/s), attention 650, o_proj 200, ~600 us of
  partial PE while the attention all-reduces drain, MLP blocks with 60-100 us PE gaps at each block transition, 450 us
  DMA-bound tail of MLP finishes.
- NKI's scheduler reorders independent instructions (`nl.no_reorder` exists): program order alone does not decide
  DMA issue order, and SBUF reuse decides when a prefetch can start.

Also rejected (decode, b512, with RBQ + PFO): gate/up prefetch 11 slabs 889.3 (vs 885.5); QKV prefetch 2 slabs 888.4;
weight slabs alternating Sync / Scalar hwdge queues (`hw=1`) 919.4; PFO 5 / 15 at b1152: 889.5 / 888.9 (vs 888.4).
p256 with the next layer's o_proj slabs prefetched too (`PFQ=2`): 38.2-38.5 vs 37.8.

How to run the experiments: `tests/tools/exp.sh NAME TARGET...` copies the engine source, takes every
NEFF from a reference build (`BASE`), rebuilds only the named targets (`BUILD_CORES`, with `STEP_OVERRIDE` /
`PF_CC_OVERRIDE` / `PF_GROUP_OVERRIDE` JSON in the environment) and links an engine. Profiles: `neuron-explorer capture
--collectives-worker-count=8 -r 8 -i 0 --num-exec 3` on cores 8-15 (step NEFFs need `-m` with a per-rank input
list: pos=300, plen=1, random weights), `view --output-format json` writes `ntff.json` (1.4 GB for a step); per-engine
busy time and DMA bytes per time bucket from `instruction` / `dma` of subgraph `sg01`, collectives from `cc_ops`.

### Next (estimates)

- Decode is chain-bound, not byte-bound: ~217 us per layer, ~20 us of it with the weight stream stopped at the two
  all-reduces (SBUF full of prefetched slabs, PE waiting on the chain). Fewer bytes (HALF) did not help; only moving
  chain DMAs off the in-order weight queue did (RBQ). Candidates: the all-reduce input write (swdge, 5-8 us) on a
  queue that does not wait behind weights; a 2-step decode NEFF (removes one step start + tail per 2 tokens, ~0.5%).
- p1024 (2 blocks) is unchanged: ~175 us per layer idle waiting for the first attention all-reduce + finish + norm2,
  ~100 us at the MLP block transition, ~150 us tail. FN / PMF / grouping / ILV all lost there.
- p4096: phase A ~850 us per layer (PE ~70%, DVE busy), attention ~650 us (nkilib), MLP block transitions ~60-100 us
  each (the activation-exchange write and the all-reduce input writes wait behind weight slabs on the Sync queue; moving
  them to the Scalar queue lost), ~400 us tail of MLP finishes.


## Round 003 (2026-11-06): decode weight stream kept busy through the per-layer dependency chain

Only the decode step NEFF changed (`kernels/step_tp.py`, switches `STEP_PFG`, `STEP_CCQ` in `targets.py`). Measured:
whole requests generate(P, 1) + generate((), 64), cores 8-15, engine alone, `tests/dec.py` (same random
prompts for every engine, one process per engine; times stable to +-0.5 ms over 3-4 prompts). Experiments built
only the changed NEFF in a copy of the engine and linked it with the other NEFFs of the session build.

Diagnosis (`neuron-explorer` json timeline of the b512 step, rank 0, 1 us DMA buckets): the weight stream of a physical
core runs at ~330-360 GB/s, but dropped to ~0 for 6-10 us around every all-reduce (128 per step). Two causes:
1. **SBUF ran dry of queued weights**: only 3 gate/up slabs were prefetched before the o_proj all-reduce, so the Sync
   queue had nothing to fetch during QKV -> attention -> o_proj -> all-reduce -> norm.
2. **The Sync engine's in-order DMA queue waited on the chain**: the all-reduce buffer write (hb <- s2) sits on the
   same hwdge queue as the weights, so every later weight DMA waited until the layer's partial was ready.

| decode 64 tokens (ms), b512 (p256) | | tokens |
|---|---|---|
| round 002 (gate/up prefetch 3 slabs) | 951.4 | |
| `STEP_PFG` = 5 / 7 / 9 / 11 / **13** (all of gate/up; the compiler reuses buffers as SBUF fills) | 935 / 923 / 917 / 915 / **914** | identical |
| + hb write, read-back and K/V cache writes on GpSimd software DGE | 907 | identical |
| + hb write and read-back on the Scalar engine's hwdge queue instead | 928 (rejected) | identical |
| + read-back only on software DGE | 920 (rejected) | |
| **+ hb write only on software DGE (`STEP_CCQ=1`, adopted)**; with K/V writes too: 894 as well | **894** | identical |
| SL=1 (PF 6, PFG 26) / SL=3 (PF 2, PFG 8) / SL=4 (PF 1, PFG 6), with CCQ | 974 / 935 / 962 (rejected) | identical |
| PFG=9 with the adopted CCQ | 900 (13 kept) | |

V-cache read in one DMA per layer (`STEP_VCH`, all tiles) now helps the larger buckets (it did not in round 002):
b1152 911 (1 tile per DMA) / 897 (3) / **895** (9 = all); b4224 933 (4) / **913** (33 = all); b512 894 (1) / 896 (4 = all,
kept 1). b2048 / b4608 set to all as well (16 / 36).

Adopted engine vs the round-002 engine, same random prompts (`SEED=7` and `SEED=11`, 17 prompts of 1 .. 4,200
tokens + 64 generated, crossing buckets 1152 / 4224): **all token streams identical**; decode 64 tokens p256 951 ->
893, p1024 941 -> 893, p4096 979 -> 912 ms; prefill unchanged.

b4224 (p4096): 979 -> 939 ms with PFG=13 + the first CCQ variant. Profile of the PFG=13 step: 14.81 ms device time
(was 15.51), DMA active 96.4% (was 92.5%).

dev-eval 001 of this round (1 round, PFG=13 + the first CCQ variant, 907 ms at b512): **PASSED, score 1.4439**
(prefill 1.540, decode 1.354); p256 1.636 / 1.309, p1024 1.880 / 1.414, p4096 1.187 / 1.341; 0 mismatching steps.

dev-eval 002 of this round (2 rounds, the adopted engine): **PASSED, score 1.4497** (prefill 1.518, decode 1.385);
p256 1.584 / 1.341, p1024 1.876 / 1.438, p4096 1.176 / 1.378 (candidate decode ms 893.6 / 893.7 / 912.8; candidate
prefill had two one-off outliers in round 2: p256 44.7, p4096 448.2 vs 41.1 / 440.1 in round 1); 0 mismatching steps.

Clean build (copy without target/, empty NEFF cache, cores 0-15): 11 min 24 s; tokens and times identical to the
session build (256 / 1024 / 4096 + 64). New tool: `tests/dec.py LENS N ENGINE...` (same prompts through several
engines one after another, prints prefill / decode ms and the first divergence from the first engine).

### Rejected in round 003

- Prefill QKV blocks of 1,024 tokens (each QKV weight slab shared by two 512-token sub-blocks; o_proj / MLP blocks
  unchanged): p4096 441 -> 458, p1024 116.7 -> 128.6, p2048 226 -> 244 ms (tokens identical). Like the 1,024-token
  MLP blocks below, larger blocks lose more to SBUF pressure than they save in weight re-reads.
- Profile of the S=4096 layer NEFF on all 8 ranks (random inputs): every rank 6.4-6.5 ms, TensorE 71-75% active,
  throttle limit ~0.81 on average. Rank 0 timeline (sg01): phase A (norm + QKV) ~800 us at ~200 GB/s DMA (DMA-bound:
  h is read in fp32 by both cores), attention ~700 us, o_proj ~200 us, then ~1 ms with PE mostly idle while the 8
  serialized attention all-reduces (5.2 MB bf16 each, 95-376 us, the first one longest) gate the MLP blocks, MLP at
  full PE, ~0.4 ms tail for the last MLP all-reduce + finishes. MLP all-reduce durations grow 144 -> 332 us over
  the layer (they overlap compute and wait for the slowest rank).

- Prefill MLP blocks of 1,024 tokens sharing each weight slab between two 512-token matmul sub-blocks (half the MLP
  weight re-reads; attention all-reduces of the two 512 blocks finished before each MLP block): p4096 441 -> 482 ms,
  p1024 116.6 -> 124.6, p2048 224.6 -> 232.1 (tokens identical). Profile of the adopted S=4096 layer NEFF (random
  inputs): 6.49 ms, TensorE active 70%, throttled to a 50% utilisation limit 36% of the time, DMA active 87%.
- All-reduce straight into SBUF from the HBM buffer: does not compile ("All src & dst tensors must have the same
  buffer type").
- More decode prefetch (b512, decode 64 tokens, adopted engine 893.5 ms): all 20 down slabs before the o_proj
  all-reduce (`PFD`): 921; 4 down slabs + the next layer's 20 o_proj slabs: 907; next layer's o_proj slabs only
  (`PFO=20`): 890 at b512 but 914.4 vs 912.6 at b4224 (not adopted). Read-back of the all-reduce on the Scalar
  engine's hwdge queue (with PFO): 906. Timeline of the adopted b512 step: 14.42 ms device time, DMA active 97.7%;
  the remaining dips are at the end of each o_proj all-reduce (the Sync queue waits to read the result back, ~3 us)
  and during each down all-reduce (SBUF full of next-layer slabs until the norm, ~8 us).
- Two layers per prefill layer-NEFF execution (`PF_LPN=2`: per-layer tensors sliced 2 layers at a time, the layer
  body looped inside the NEFF, `src/engine.cpp` executes L / 2 layer NEFFs): tokens identical, but p256 41.0 -> 40.4,
  p1024 116.6 -> 117.7, p2000 224.8 -> 217.7, p4096 440.7 -> 444.5 ms: mixed, not adopted. The in-order engines do
  not overlap one layer's last all-reduce with the next layer's weight stream on their own.
- Profile of the S=256 layer NEFF (random inputs, rank 0): 747 us, of which ~200 us waiting on the attention
  all-reduce and ~60 us on the MLP one (one 256-token block: nothing to overlap). The first all-reduce of a profiled
  NEFF is long (202 us at S=256, 376 at 4096 vs ~60 / ~100 for the next ones): probably rank launch skew under the
  profiler; in the engine the layer costs ~0.6 ms.

### Next (estimates)

- Decode b512 step is 14.0 ms per token in the engine vs ~12.3 ms of pure weight streaming at the ~345 GB/s per
  physical core seen in the timeline: ~1 ms is the pauses below, the rest the LM head / argmax tail and step start.
- Prefill: the attention all-reduces of the 8 blocks at S=4096 run back to back after o_proj and gate the first MLP
  blocks (~1 ms per layer of poorly used PE); at S=256 / 1024 the two per-layer all-reduces are exposed. Ideas: block
  the layer by token blocks (attention of block b needs only K/V of blocks <= b) so all-reduces overlap compute, or
  hoist the next layer's first QKV slabs before the last all-reduce in a multi-layer NEFF.

### What is left in decode (from the timeline)

- After each all-reduce the stream still pauses for ~5-8 us while SBUF is full of prefetched gate/up (or next-QKV)
  slabs that wait for the norm of the reduced h: about 1 ms per step.
- The two physical cores of a rank drift: the sendrecv before an all-reduce waited up to 15.7 us for the other core.
- Software-DGE DMAs take ~5-8 us from issue to completion for 20 KB (the chain's hb write), hwdge ~1-2 us.


## Round 002 (2026-11-06): prefill all-reduce over HBM buffers, o_proj / down N-split over the two physical cores

All changes are in the prefill layer NEFF (`kernels/prefill_tp.py`, switches in `targets.PF_CC`); decode is unchanged.
Measured whole-request prefill generate(P, 1), cores 8-15, engine alone, `tests/compare.py` (same random prompts):

| variant (cumulative) | p256 | p1024 | p4096 | tokens vs previous row |
|---|---|---|---|---|
| round 001: fp32 SBUF all-reduce, 128-token messages | 75 | 261 | 1031 | |
| HBM-buffer all-reduce (`CCHBM=1`), fp32, 512-token messages (10.5 MB) | 51 | 165 | 641 | 6/6 identical |
| same, 2048-token messages | 51 | 164 | 647 | not adopted |
| **bf16** HBM all-reduce (`CCBF=1`), 512 tokens | 45 | 141 | 547 | 6/6 identical; dev-eval 001 score 1.2857 |
| + per-block launch, finish after the loop (`OVL=3`, K-split kept) | 45 | 142 | 550 | not adopted |
| + MLP blocks of 512 (`TEB=512`, K-split kept) | 45 | 141 | 547 | not adopted |
| **N-split** o_proj / down over the two physical cores (`NSP=1`) | 41 | 129 | 506 | 7/8 (one 1024 stream diverges at token 13) |
| + `TEB=512` (now it helps: gate/up weights re-read half as often) | 41 | 122 | 480 | 6/6 identical |
| **+ finish block b's attention all-reduce just before MLP block b** (`NSP_LATE`) -- adopted | **41** | **116** | **441** | 6/6 identical |
| timing only: no all-reduce at all (`PF_NOAR=1`), K-split bf16 / adopted N-split | 41 / 33 | 118 / 104 | 458 / ~414 | wrong tokens |

Mechanisms:
- **The round-001 bottleneck was the SBUF all-reduce across the two devices** (FINDINGS 1: 8.4 GB/s at 1 MB, vs 62
  GB/s for 16 MB HBM-buffer all-reduces). `_hbm_launch` / `_hbm_finish`: internal `nl.shared_hbm` buffers
  `pf_hb_*` / `pf_hr_*` [S/blk, 128, HT, blk] (one slot per token block, never reused within a NEFF), core_barrier,
  `ncc.all_reduce(hb[blk] -> hr[blk])` (runs once per logical core), both cores read their half back.
- **bf16 all-reduce** (the reference all-reduces bf16 too). Over SBUF bf16 gave NaN / divergence in round 001; over
  HBM it is clean (dev-eval controls 1 and 4 mismatching steps of 221 allowed).
- **N-split (`NSP`)**: core pid computes hidden tiles [20 pid, 20 pid + 20) of o_proj over all 8 heads of the rank
  (it reads the other core's K rows from `wo[l, 1 - pid]`: same tensor, no layout change) and of down over all 26
  intermediate tiles (the SwiGLU activations of the other core come through an internal HBM buffer `pf_act_x`). The
  rank sum is written in bf16 straight into the all-reduce buffer: the fp32 partial exchange through `s_p` (84 MB
  written + 84 MB read per core per combine at S=4096) is gone. `s_p` is still an (unused) input.
- Profile of the S=4096 layer NEFF (`neuron-explorer capture --collectives-worker-count=8 -r 8`, rank 0):
  K-split bf16: 8.46 ms, 2.81 GB HBM read, TensorE 58%, DMA 91%, cc 1.96 ms; N-split + TEB 512: 7.09 ms, 1.69 GB,
  TensorE 64%, DMA 83%, cc 2.39 ms (before NSP_LATE). Throttle limit ~0.75 on average.

dev-eval 002 (1 round, all lengths, the adopted engine): **PASSED, score 1.3995** (prefill 1.541, decode 1.271);
p256 1.647 / 1.252, p1024 1.872 / 1.366, p4096 1.186 / 1.201 (prefill ms ref / cand 67.6 / 41.0, 218.6 / 116.8,
522.3 / 440.3); 0 mismatching steps in the timed streams; controls 4 / 221 allowed (p3, p127, p128; rank 2).

### Decode: permuted V-cache rows (`step_tp.VPERM`, `targets.STEP_VCH`)

Profile (rank 0, `neuron-explorer`, 8 workers): step b512 15.51 ms, 8.46 GB read, MBU 0.76, DMA active 92.5%, 130 cc
ops 1.79 ms; b4224 16.86 ms, 8.71 GB: +0.24 GB costs +1.35 ms. The V read was 33 DMAs of 128 x 256 B rows per layer.
Now row `(t % 128) * (TMAX / 128) + t // 128` of `vcache` holds token t (same tensor shape; values unchanged): a
partition's tokens are contiguous, read in chunks of `STEP_VCH[bucket]` tiles. The decode step computes the write row
from pos on the device (int shift / and, fp32 multiply-add, register); the prefill writes the same layout.

| V tiles per DMA | decode 64 tokens p256 (b512) | p1024 (b1152) | p4096 (b4224) |
|---|---|---|---|
| round 001 layout (1 tile per DMA) | 952 | 941 | 1035-1036 |
| permuted, VCH=1 (same access shape as before) | 951 | 941 | 1035 |
| permuted, VCH=4 | 955 | 951 | 979 |
| permuted, all in one DMA | 950 | 955 | 984-985 |
| **adopted: 1 for b512 / b1152, 4 for b2048 / b4224 / b4608** | 952 | 941 | **979** |

Tokens identical to the previous engine on prompts 1, 130, 256, 1024, 2049, 4096, 4100 (64 tokens) and through the
decode-only path (`ENGINE_NO_PREFILL=1`, 300 tokens + 16).

### Rejected in round 002 (numbers above / here)

- 2048-token HBM all-reduce messages instead of 512: 647 vs 641 ms at 4096 (fp32).
- K-split + per-block all-reduce launch (`OVL=3`): 550 vs 547; K-split + `TEB=512`: 547 (TEB only helps with NSP).
- 128-token blocks for the 256 / 512 prompt buckets (`SB=512`, so a block's all-reduce could overlap the next
  block): p256 41 -> 51-52 ms, 400 tokens 67 -> 90 ms. The 128-wide matmuls cost more than the overlap saves.

### dev-eval 003 (2 rounds, all lengths, the final round-002 engine): **PASSED, score 1.4162** (prefill 1.531, decode 1.310)

| case | prefill speedup | decode speedup | prefill ms ref / cand | decode ms ref / cand |
|---|---|---|---|---|
| p256 | 1.612 | 1.271 | 67.2-67.5 / 41.1-42.4 | 1186-1232 / 951.5 |
| p1024 | 1.879 | 1.383 | 218-220 / 116.6 | 1287-1316 / 941.1 |
| p4096 | 1.186 | 1.278 | 522.5 / 440.6 | 1248-1254 / 978.9 |

0 mismatching steps in the timed streams; controls 4 mismatching steps / 221 allowed (p3, p127, p128; rank 2).
Clean build (copy without target/, empty NEFF cache, cores 0-15): 11 min 33 s; tokens identical to the session build
(256 / 1024 / 4096 + 64).

### Next (estimates)

- Prefill at 4096 is now 441 ms vs ~414 without any all-reduce (timing-only build): the MLP's last block
  all-reduce per layer is still exposed (~0.15 ms x 64), and TensorE is ~64% active, DMA ~83%, throttled. Ideas:
  share the norm between the two cores (each normalises half the tokens; today both read all of h), larger QKV /
  MLP blocks with two 512-wide PSUM tiles per weight slab (fewer weight re-reads).
- Decode base step 14.9 ms vs a 12.9 ms floor at 658 GB/s: 128 all-reduces x ~14 us = 1.8 ms of cc time; MBU 0.76.
  Needs a timeline (`--output-format json`) to see whether the weight stream stalls around each all-reduce.

Notes for experiments: run the GPU lock as `GPU_LOCK_FILE=<session dir>/.gpu.lock`; `/tmp/q32work/noar_build.py`-style
NOAR builds need a dummy target key (the env switch is not in the cache key) -- never leave such NEFFs in the cache
under a real key.


## Round 001 of the q32 run (2026-11-06): the seed ported from Qwen3-14B TP=2 to Qwen3-32B TP=8

Same architecture as the seed (everything below "Seed history" still describes the mechanisms); what changed:

| piece | change |
|---|---|
| `kernels/targets.py`, `src/model.h` | 32B shapes: L=64, NQ=64, NKV=8, INTER=25600; W=8 ranks = 16 virtual cores; NQL=4 q heads, NKL=1 kv head (**kv head = rank, replicated on both physical cores**, each keeps its own cache copy), ITL=13 (**200 intermediate tiles zero-padded to 208**: gate/up rows and down columns zero, a layout change, SiLU(0)*0=0), NVL=75 vocab tiles per virtual core |
| `src/engine.cpp` loader | kv head index `v * NKV / C`; `ws_blocks` writes zero tiles for null rows / negative k tiles (the padding) |
| `STEP_PF` | 6 -> 3 (QKV is only 3 slabs of 2 tiles now) |
| `PF_CC` (prefill all-reduce) | **fp32, 128-token messages** (the 14B round's first version). bf16 x 256 (the 14B choice) gives all-NaN logits at prompt buckets >= 1024 over 8 ranks / two devices (head argmax returns BIGC = 1048576); bf16 x 128 is finite but diverged from the decode-step path at generated token 2 on a 1,024-token prompt. `PF_CC_OVERRIDE='{"CCBF":1,"CCN":128}'` at build time selects a variant (part of the cache key; experiments only) |
| `build_neffs.py` | build pools default to the run's 16 cores as two 8-core jobs (`ENGINESMITH_NEURON_CORE`..+7, +8..+15); clean build from an empty cache 9 min 44 s (step NEFFs 199-217 s each, prefill layer 17-74 s) |
| tests | `smoke.py` / `compare.py` point at Qwen3-32B and cores 8-15; `tests/probe.py ENGINE[:ENV=V] CORES LENS N` = one generate(P, N) per random prompt (same rng seed: compare runs only with the same LENS list) |

Experiments ran on cores 8-15 (the run owns 0-15; the harness gives 0-7 to the engine and 8-15 to the judge).
`/tmp/engine-gpu-0.lock` belongs to another user and cannot be opened: run dev-eval with
`GPU_LOCK_FILE=<session dir>/.gpu.lock`.

### Verified (round 001)

- Real text ("The capital of France is Paris. The capital of Germany is"): " Berlin. The capital of Italy is Rome. ...
  Lisbon. The"; prefill+decode split = one request; 1-token prompt quirk kept.
- Prefill path vs decode-step path (`ENGINE_NO_PREFILL=1`), random prompts 1, 2, 256, 400, 512, 513, 700, 1024 (4-6
  tokens): identical. 2,049 and 4,096: identical for 3 tokens, then both pick 785 vs the decode path's 104544 after
  the same `151668, 271` context (near-tie between the two numerics paths); 4,100: identical for 6.
- Long output: 200-token prompt + 4,000 tokens + 408 more (position 4,608, the compiled limit) all returned; one
  more request is refused with an error.
- **dev-eval 002 (1 round, all lengths): PASSED**, 0 mismatching steps in all timed streams; controls 11 / 1,300
  mismatching steps (allowed 130), worst control p2049 7 / 13 allowed (rank <= 8).

| case | prefill ref / cand ms | prefill speedup | decode 64 ref / cand ms | decode speedup |
|---|---|---|---|---|
| p256 | 67.0 / 73.9 | 0.906 | 1206.5 / 952.0 | 1.267 |
| p1024 | 218.8 / 262.4 | 0.834 | 1261.1 / 941.3 | 1.340 |
| p4096 | 520.5 / 1030.7 | 0.505 | 1247.9 / 1035.8 | 1.205 |

**score 0.9595** (prefill 0.725, decode 1.270). Decode 14.9 ms/token at b512 (lab 2 measured 15.37 for the step).
Hello: not measured precisely (well under the harness's limits in dev-eval).

### Next (estimates, not measured)

- Prefill is the weak half, and at 4,096 it is 2x the reference: the fp32 128-token SBUF all-reduce across two
  devices dominates (14B: fp32 x 128 cost ~55 ms more than bf16 x 256 at 2 ranks; here 64 layers x 8 ranks). Try
  an HBM-buffer all-reduce for the prefill combine (the decode step's `_xchg_add_tp_hbm`, which works across the
  two devices), larger fp32 messages, or reduce-scatter + all-gather; check every variant against the decode-step
  path at 1024 / 2049 / 4096 *and* in dev-eval (bf16 variants changed tokens).
- Decode b4224 / b4608 (p4096 decode 16.2 ms/token vs 14.9 at b512): attention over the KV bucket.
- Decode is DMA-bound per core (8 GB/core-step); the per-layer all-reduce count (2 x 64) costs ~1-2 ms.

## Seed history: Qwen3-14B TP=2 (the seed's round 001; mechanisms unchanged, numbers are 14B)

## What it is (round 001)

One native executable (`target/engine`, links libnrt only) that speaks the EngineSmith protocol and runs Qwen3-14B
bf16 tensor parallel over the first two visible logical NeuronCores (LNC=2 each: 4 "virtual cores" = 2 ranks x 2
physical cores). All NEFFs are embedded (`.incbin`, `target/gen/neffs.S`), built at build time through NKI's
documented route (`kernels/build_neffs.py`: `NKI_ARTIFACTS_DIR` + `@nki.jit` called with per-rank numpy lists).

| piece | file | what |
|---|---|---|
| decode step | `kernels/step_tp.py` (lab 2, unchanged) | one SPMD NEFF per step: embedding, 40 layers, LM head, argmax (ties to lowest id), `toks[pos+1]`, `pos+1`; in-NEFF `nki.collectives.all_reduce` (HBM buffers) of the o_proj / down partials; `PF=6 HBMCC=1`, `SL=2` (weight slabs of 2 output tiles). Attention buckets 512 / 1152 / 2048 / 4224 / 4608 (`step_b*`), chosen per step by the host (`pos <= TB`) |
| prefill | `kernels/prefill_tp.py` (from lab 1 `prefill_g.py`) | layered: `pf<S>_emb`, `pf<S>_layer` (executed 40 times on per-layer tensor slices), `pf<S>_head` (final norm + LM head + TP argmax, writes `toks[plen]`). Prompt buckets S = 256 / 512 / 1024 / 2048 / 4096. TP change: the o_proj / down partials are summed over the two physical cores (scratch `s_p` + barrier) and then SBUF-all-reduced over the ranks in 256-token sub-chunks as bf16 (`_combine_tp`, `PF_CC` in targets.py) |
| host | `src/engine.cpp`, `src/json.h`, `src/model.h` | safetensors mmap + per-rank re-layout (values unchanged, 48 threads, 2.7 s for 28 GB), NEFF load (`nrt_load_collectives` per rank), shared device tensors bound by name, per-layer slices (`nrt_tensor_allocate_slice`), one scheduling thread per rank, whole request queued as one chain (prefill NEFFs + decode steps), tokens read once per request |
| targets / shapes | `kernels/targets.py` | the NEFF list and per-rank shapes (the contract with `src/model.h`) |

Request flow: the prompt is written to `prompt` (the prefill's `ids` is a slice of it), `plen` and `pos` written;
then one queued chain: prefill of the smallest bucket >= plen (42 executions) + the decode steps; `toks` read once.
A prompt longer than 4,096 tokens: the first 4,096 by prefill (with `plen` temporarily 4,096), the rest by decode
steps. **Reference quirk** (NxD Inference): a one-token prompt `[t]` is answered as the prediction after `[t, 0]`;
the engine prefills `[t, 0]`, then rewrites `toks[1] = g`, `plen = 1` and continues at position 1, as the
reference does. Padding positions of a prefill bucket write K/V garbage beyond plen; decode steps overwrite each
position before it is read and mask everything at or after pos.

Runtime environment set by the engine itself: `NEURON_RT_VISIBLE_CORES` = the first two of the given list (the
harness lists four; the last two are the judge's), `NEURON_RT_ROOT_COMM_ID=localhost:<port from core and pid>`,
`NEURON_RT_DISABLE_EXECUTION_BARRIER=1`. libnrt prints to fd 1: the engine dups the original stdout for replies
and points fd 1 at stderr.

Build: `bash build.sh` -> `kernels/build_neffs.py target/neff` (cache `target/neffcache/<name>-<key>.neff`, key =
the kernel sources the target depends on + the target's shapes + neuronx-cc / nki versions; two parallel 2-core
jobs on `BUILD_CORES`, default the run's cores ENGINESMITH_NEURON_CORE..+3) -> g++. From an empty cache: step NEFFs
~320-340 s each (5), prefill layer NEFFs 24 / 35 / 41 / 75 / 144 s (S = 256 .. 4096), emb / head 11-49 s; a clean
clone (`git clone` + `bash build.sh`, empty cache, cores 56-59) built in 18 min 6 s. Cached: 3 s. Never put a third job on cores outside
the run's four: at formal-build time 60-63 may belong to someone else.

## Measured (round 001, cores 56-57, engine alone, `tests/compare.py` / `tests/smoke.py`, wall time per request)

| | 256 | 512 | 1,024 | 2,048 | 4,096 |
|---|---|---|---|---|---|
| prefill bucket, whole request generate(P, 1) | 46 ms | 87 ms | 165 ms | 328 ms | 654 ms |
| decode, ms/token (generate((), 16..64)) | 22.4 (b512) | | 22.8 (b1536-2048) | | 23.7 (b4608) |

Correctness: prefill path vs token-by-token decode-step path (`ENGINE_NO_PREFILL=1`), random prompts of 2, 3, 17,
100, 129, 200, 255, 256, 300, 513, 1,025, 2,049, 4,097 tokens + 16-24 generated tokens: identical streams.
Real-text prompt "The capital of France is Paris. The capital of Germany is": " Berlin. The capital of Italy is
Rome. ... Lisbon. The" (as HF fp32 in lab 2).

Profile of one prefill layer at S = 4,096 built with W=1 (rank 0's shard, no collectives; `neuron-explorer`):
13.0 ms, TensorE active 86%, DMA active 97%, HBM read 5.7 GB (weights re-read per token block), throttling limit
0.5 part of the time. The TP layer costs ~16 ms (654 ms / 40): ~3 ms per layer is the cross-rank all-reduce of the
two [128, 5120, 4096] fp32 partials (SBUF, 128-token messages).

## dev-eval (public prompts)

| run | p256 prefill / decode | p1024 | p4096 | score |
|---|---|---|---|---|
| dev-eval 002 (2 rounds), fp32 prefill all-reduce | 2.943 / 1.282 | 2.230 / 1.334 | 1.518 / 1.285 | **1.6725** (prefill 2.15, decode 1.30), 0 mismatches |
| dev-eval 003, bf16 prefill all-reduce (outliers, below) | 0.600 / 1.280 | 1.450 / 1.351 | 1.659 / 1.298 | 1.2161, 0 mismatches in timed cases; controls p2047 / p2048 / p2049 1 mismatch each (margins 0.125-0.25) |
| dev-eval 004, bf16 all-reduce + reply written before the stderr log | 3.100 / 1.287 | 2.410 / 1.350 | 1.655 / 1.296 | **1.7409** (prefill 2.31, decode 1.31), 0 mismatches |

Reference (NxD Inference, TP 2) per request: prefill 137 / 368 / 993 ms, decode 64 tokens 1,835 / 1,939 / 1,937 ms
(28.7 / 30.3 / 30.3 ms per token). dev-eval 001 (1 round) had candidate prefill outliers (p256 522 ms, p4096 995 ms
by the harness while the engine's own timer said 46 / 653 ms); not reproduced by dev-eval 002 or by the harness's
own client (`enginesmith.protocol.Engine`) run directly. dev-eval 003 had them again (p256 197 / 276 ms for 44 ms
of engine time, p1024 301 / 220 for 152): the engine wrote its stderr log line *before* the reply, and on this shared
machine (other runs building, load ~35) a small write to the scratch-dir file can block; the time fell between the
engine's timer and the reply. Now the reply goes out first with one write(2), the log after it (it records the
previous log's own duration). dev-eval 004: no outliers.

## Prefill all-reduce experiments (S = 4,096, whole request, cores 60-61, engine alone)

| variant | prefill ms |
|---|---|
| fp32, 128-token SBUF messages ([128, 20 x 128] fp32 = 1.3 MB), combine after all blocks (round 001 first version) | 654 |
| no all-reduce at all (`PF_NOAR=1`, wrong tokens: timing only) | 558 |
| fp32, combine each block right after its partial (`OVL=1`) | 658 |
| fp32, software-pipelined: launch block b's all-reduce, consume it after block b+1 (`OVL=2`) | 671 |
| fp32, MLP token blocks of 512 instead of 256 (`TEB=512`: half the MLP weight re-reads) | 657 |
| fp32, 256-token messages (2.6 MB) | fails at run time: `enc_post_operation: Assertion alg == ENC_ALG_MESH` |
| **bf16 messages, 128 tokens** | 619 |
| **bf16 messages, 256 tokens (1.3 MB) -- adopted (`PF_CC`)** | **599** (256: 44, 1,024: 153) |

The cross-rank all-reduce does not overlap with the GEMMs in any arrangement tried; bytes and call count are what
cost. bf16 partials: the per-rank sum of the two physical cores' fp32 partials is rounded to bf16, summed over the
ranks in bf16 and added to the fp32 residual (NxD Inference all-reduces bf16 activations too). Not bitwise equal to
the fp32 version: on 9 random prompts vs the decode-step path, 8 identical, one diverged at generated token 6.

## Decode experiments (64 tokens after a 256-token prompt, bucket 512, cores 60-61; whole request)

| step NEFF variant | ms per 64 tokens |
|---|---|
| SL=1, PF=4, weight DMAs on hwdge / sync (round 001 first version) | 1427.7 |
| hw=1 (weight DMAs alternating sync / scalar hwdge queues) | 1438.4 (slower) |
| **SL=2 (2 output tiles per slab: 20 KB contiguous per partition), PF=4 -- adopted** | **1420.3** (tokens identical on 10 prompts x 65 tokens) |
| SL=4, PF=2 | 1465.1 (slower) |
| SL=2, PF=8 | does not compile (QKV has 7 slabs) |
| SL=2, PF=2 | 1420.0 |
| **SL=2, PF=6 -- adopted** | **1416.5** (vs 1420.2 for PF=4 in the same session; tokens identical on 6 prompts x 65 tokens) |

Buckets: the timed decode of p1024 runs positions 1024..1088, of p4096 4096..4160; buckets 1152 and 4224 (instead
of 2048 and 4608) took decode 64 tokens from 1453 -> 1424 ms (p1024) and 1513 -> 1478 ms (p4096). 3072 dropped
(only control prompts reach it; they use 4224). Device profile of the b512 step (`neuron-explorer capture` with
`--collectives-worker-count=2 -r 2 -m multi.txt`, all 21 inputs given as .npy, `pos` = 300): 23.2 ms (sync
execution), 14.04 GB read from HBM (606 GB/s), MBU 0.85, DMA active 99.6%, collectives 0.93 ms in 82 ops (mostly
hidden behind the weight stream). Decode is DMA-throughput bound.

dev-eval 005 (2 rounds, SL=2 + buckets 1152 / 4224, b3072 still present): p256 3.115 / 1.289, p1024 2.419 / 1.376,
p4096 1.658 / 1.321, **score 1.7555** (prefill 2.32, decode 1.33), 0 mismatches in the timed cases.

dev-eval 006 (**5 rounds**, the formal shape; SL=2 PF=4, buckets 512 / 1152 / 2048 / 4224 / 4608): p256 3.095 /
1.272, p1024 2.408 / 1.351, p4096 1.656 / 1.297, **score 1.7376** (prefill 2.311, decode 1.306), 0 mismatching steps
in all 15 timed streams. Candidate times are stable to 0.3% over rounds; the reference's decode varies 1,810-1,985 ms.

dev-eval 007 (2 rounds, final round-001 engine: SL=2 PF=6): p256 3.111 / 1.296, p1024 2.415 / 1.366, p4096
1.657 / 1.306, **score 1.7504** (prefill 2.318, decode 1.322), 0 mismatching steps. Hello ~37-47 s.

## Prefill profile (S = 4,096, one layer NEFF)

- `neuron-explorer` profiles collective NEFFs on 2 ranks (`--collectives-worker-count=2 -r 2 -i 0`); random inputs
  are fine for the layer NEFF. TP layer (bf16 all-reduce): 13.45 ms device time, 32 cc ops 1.65 ms, TensorE active
  83%, DMA active 97%, HBM read 5.7 GB, throttled to a 50% utilization limit for 25% of the time (same with zero
  inputs: 13.44 ms). With `TEB=512` (MLP blocks of 512 tokens): 13.21 ms, 3.6 GB read, but end to end unchanged
  (600 vs 599 ms), so it was not adopted.
- In the engine each layer costs 14.8-15.0 ms whether 1 or 40 layers run (ENGINE_DBG_PF_LAYERS), whether executions
  are queued 1, 2, 3 or all at a time (ENGINE_DBG_DEPTH), whether every layer uses one tensor set or its own
  (ENGINE_DBG_PF_SAMESET), with or without NEURON_RT_DISABLE_EXECUTION_BARRIER. The ~1.5 ms per layer between the
  profiler's total_time and the engine is **the weight values**: binding the layer's weights to freshly allocated
  (uninitialised, presumably zero) tensors took S=4,096 from 599 to 537 ms (`ENGINE_DBG_PF_FRESH=1`, all layers on
  one set); copying layer 0's real weights into those fresh tensors (`ENGINE_DBG_PF_FRESHCOPY=1`) brought it back to
  599 ms. Same HBM index for both (`nrt_tensor_get_device_allocation_info`). So real data draws more power and the
  device throttles harder than with the profiler's inputs: the prefill is power-limited (wgu alone: 550 ms).
- W=1 shard (no collectives), TEB=512: 12.53 ms; without attention (PF_NOATT=1, timing only) 11.06 ms: nkilib
  attention is 1.47 ms per layer (43 GFLOP per core: ~29 TFLOP/s); the GEMMs run at ~86% of TensorE bf16 peak.
  Prefill is compute- (and power-) bound; what is left is attention efficiency and the ~41 ms of all-reduce.

## Debug knobs (environment of the engine process; timing experiments only, never set by the harness)

`ENGINE_NO_PREFILL=1` (prompt through decode steps, the fp32 path), `ENGINE_DBG_PF_LAYERS=n` (run only n prefill
layers), `ENGINE_DBG_PF_SAMESET=1`, `ENGINE_DBG_PF_FRESH=1|name:name`, `ENGINE_DBG_PF_FRESHCOPY=1`,
`ENGINE_DBG_DEPTH=n` (at most n executions in flight), `ENGINE_DBG_KEEP_BARRIER=1`. Kernel env at build time:
`PF_NOAR=1`, `PF_NOATT=1` (wrong results, timing only; they are not part of the cache key: never build the engine
with them set). Test tools: `tests/seq.py` (the harness's per-round sequence), `tests/compare.py` (two engines token
for token), `tests/smoke.py`; `CORES=60-63` runs them on the experiment cores.

## Ideas not done yet (estimates)

- prefill: the unexplained 1.5 ms per layer (engine vs profiler, ~60 ms at 4,096); attention (1.47 ms per layer).
- decode: DMA-bound at ~606 GB/s per logical core (profile); the measured best GEMV rate was 658-674 GB/s.
- hello takes ~47 s (5 + 15 NEFF loads, 26 s of it nrt_load of the step NEFFs): not scored.
