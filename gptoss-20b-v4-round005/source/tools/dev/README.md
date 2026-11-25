# Development tools (session 002; not part of the build)

Engines are run through small wrapper scripts that set `NEURON_RT_VISIBLE_CORES` to the experiment cores
(e.g. `#!/bin/bash` / `export NEURON_RT_VISIBLE_CORES=16-19` / `exec .../target/engine "$@"`), because the skill's
scripts start engines with a clean environment. Scratch goes to /tmp/es-dev.

- `one.sh VARIANT`: compile + run one NEFF variant on cores 16-19 (CORES=...) into /tmp/es-dev/art_VARIANT.
- `prof.sh VARIANT NEFF POS`: neuron-explorer capture of a (collective, 4-rank) NEFF with zero inputs from
  `build_neffs.py` (ES_DUMP), pos = POS; writes ntff.json and a 20 us-bin engine/DMA timeline (`tl.py`).
- `prof_moe.sh NT`: the same for `pmoe{NT}` with synthetic or real (RTI=file.npy PLEN=n) routing.
- `dump_rti.py ENGINE PLEN LAYERS OUT`: real routing table after LAYERS layers (`read` op).
- `test_l0.py ENGINE S` / `test_l01.py ENGINE S`: prefill layer 0 (1) in the engine (ENGINE_PF_LAYERS) vs the numpy
  reference in `tools/layout.py`; `test_l0b.py`: routing tables and slot buffers of the MoE.
- `bench.py ENGINE PLEN what:layer ...`: the engine's `bench` op (queued executions of one NEFF) after a prefill.
- `drive.py`, `longout.py`: request timing and the long-output check.
