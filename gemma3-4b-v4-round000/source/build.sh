#!/usr/bin/env bash
# Builds target/engine: NKI kernels -> NEFFs (Python/NKI/neuronx-cc, build time only), RoPE tables (torch, build
# time only), then the native host linked against libnrt with every NEFF and table embedded (.incbin).
set -euo pipefail
cd "$(dirname "$0")"
V=${NEURON_VENV:-/opt/aws_neuronx_venv_pytorch_inference_vllm_0_24_0_1_1_0}
NRT=${NRT_HOME:-/opt/aws/neuron}
export PATH=$V/bin:$PATH
export NEURON_PLATFORM_TARGET_OVERRIDE=trn2
# nki runs each kernel once after compiling it; the build does not need that run, so it gets no NeuronCore
# (the run fails after kernel.neff is written and the build ignores it). Never touch other runs' cores.
export NEURON_RT_VISIBLE_CORES=${BUILD_NEURON_CORES:-4095}
mkdir -p target
JOBS=${BUILD_JOBS:-32}   # 22 of the NEFFs are 34-layer prefill programs (~10 min each to compile)
"$V/bin/python" kernels/build_neffs.py target/neff --jobs "$JOBS"
if [ ! -s target/rope_tables.bin ] || [ kernels/gen_tables.py -nt target/rope_tables.bin ]; then
  "$V/bin/python" kernels/gen_tables.py target/rope_tables.bin.tmp
  mv target/rope_tables.bin.tmp target/rope_tables.bin
fi
# embedded blobs
NAMES=$(cd target/neff && ls *.neff | sed 's/\.neff$//')
{
  echo '    .section .rodata'
  for n in $NAMES rope_tables; do
    if [ "$n" = rope_tables ]; then f="$PWD/target/rope_tables.bin"; else f="$PWD/target/neff/$n.neff"; fi
    echo "    .global blob_${n}_start, blob_${n}_end"
    echo "    .balign 64"
    echo "blob_${n}_start:"
    echo "    .incbin \"$f\""
    echo "blob_${n}_end:"
  done
  echo '    .section .note.GNU-stack,"",@progbits'
} > target/blobs.S
{
  for n in $NAMES rope_tables; do echo "extern \"C\" const unsigned char blob_${n}_start[], blob_${n}_end[];"; done
  echo "static const Blob kBlobs[] = {"
  for n in $NAMES rope_tables; do echo "  {\"$n\", blob_${n}_start, blob_${n}_end},"; done
  echo "};"
} > target/blobs.inc
g++ -O2 -std=c++17 -I"$NRT/include" -Itarget -Ihost host/engine.cpp target/blobs.S -o target/engine.tmp \
  -L"$NRT/lib" -lnrt -Wl,-rpath,"$NRT/lib" -lpthread
mv target/engine.tmp target/engine
echo "built target/engine"
