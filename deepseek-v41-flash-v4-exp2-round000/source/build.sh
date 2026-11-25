#!/usr/bin/env bash
# Build: compile the NKI kernels to NEFFs (Python/NKI/neuronx-cc, build time only; cached under target/), embed
# the NEFFs and the Engram token map in the executable, link against libnrt only.
set -euo pipefail
cd "$(dirname "$0")"
V=${NEURON_VENV:-/opt/aws_neuronx_venv_pytorch_inference_vllm_0_24_0_1_1_0}
NRT=${NRT_HOME:-/opt/aws/neuron}
export PATH=$V/bin:$PATH
export NEURON_PLATFORM_TARGET_OVERRIDE=trn2
# NKI's documented route runs each kernel once: give the builds free NeuronCores of this run
BASE=${ENGINESMITH_NEURON_CORE:-0}
if [ -z "${NEFF_BUILD_CORES:-}" ]; then
  NEFF_BUILD_CORES=$(seq -s, $BASE $((BASE + 7)))
fi
export NEFF_BUILD_CORES
mkdir -p target/neff target/neff-cache target/obj
rm -f target/neff/*.neff
python kernels/build_neffs.py target/neff target/neff-cache
# embed blobs
ASM=target/obj/blobs.S
: > $ASM
echo '.section .rodata' >> $ASM
for f in target/neff/*.neff data/engram_map.bin; do
  n=$(basename "$f"); n=${n%.neff}; n=${n%.bin}
  echo ".global _blob_${n}_start" >> $ASM
  echo ".global _blob_${n}_end" >> $ASM
  echo ".balign 64" >> $ASM
  echo "_blob_${n}_start:" >> $ASM
  echo ".incbin \"$(pwd)/$f\"" >> $ASM
  echo "_blob_${n}_end:" >> $ASM
done
echo '.section .note.GNU-stack,"",@progbits' >> $ASM
TAB=target/obj/blobs_table.h
: > $TAB
for f in target/neff/*.neff data/engram_map.bin; do
  n=$(basename "$f"); n=${n%.neff}; n=${n%.bin}
  echo "extern \"C\" const unsigned char _blob_${n}_start[]; extern \"C\" const unsigned char _blob_${n}_end[];" >> $TAB
done
echo "static const struct { const char* name; const unsigned char* s; const unsigned char* e; } BLOBS[] = {" >> $TAB
for f in target/neff/*.neff data/engram_map.bin; do
  n=$(basename "$f"); n=${n%.neff}; n=${n%.bin}
  echo "  {\"${n}\", _blob_${n}_start, _blob_${n}_end}," >> $TAB
done
echo "};" >> $TAB
g++ -O2 -std=c++17 -mwaitpkg -Wall -Wno-unused-function -I$NRT/include -Itarget/obj src/engine.cpp $ASM -o target/engine \
  -L$NRT/lib -lnrt -lpthread -Wl,-rpath,$NRT/lib
echo "built target/engine"
