#!/usr/bin/env bash
# Build: NKI kernels -> NEFFs (build time only: Python, NKI, neuronx-cc), embedded into one native executable that
# links only libnrt and the C/C++ runtime. Output: target/engine.
set -euo pipefail
cd "$(dirname "$0")"
V=${NEURON_VENV:-/opt/aws_neuronx_venv_pytorch_inference_vllm_0_24_0_1_1_0}
NRT=${NRT:-/opt/aws/neuron}
mkdir -p target/neff
# NKI's documented route runs each kernel once: give it this run's engine cores (free while building).
C0=${ENGINESMITH_NEURON_CORE:-0}
NC=${ENGINESMITH_NEURON_CORES:-4}
export NEURON_RT_VISIBLE_CORES=${BUILD_CORES:-$C0-$((C0 + NC - 1))}
export NEURON_RT_ROOT_COMM_ID=localhost:$((61000 + C0))
export NEURON_PLATFORM_TARGET_OVERRIDE=trn2
PATH=$V/bin:$PATH python tools/build_neffs.py target/neff
# embed the NEFFs
{
  echo '.section .rodata'
  for f in target/neff/*.neff; do
    n=$(basename "$f" .neff)
    echo ".global neff_${n}_start"; echo ".global neff_${n}_end"; echo ".balign 64"
    echo "neff_${n}_start:"; echo ".incbin \"$(realpath "$f")\""; echo "neff_${n}_end:"
  done
  echo '.section .note.GNU-stack,"",@progbits'
} > target/neffs.S
g++ -O2 -std=c++17 -pthread -I"$NRT/include" src/engine.cpp target/neffs.S -o target/engine \
    -L"$NRT/lib" -lnrt -Wl,-rpath,"$NRT/lib"
echo "built target/engine"
