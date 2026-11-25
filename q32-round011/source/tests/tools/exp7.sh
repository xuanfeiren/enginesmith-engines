#!/bin/bash
# exp.sh NAME [TARGET...]: copy the engine source to /tmp/q32w7/NAME, take every NEFF from the session build
# (BASE/target/neff, default /tmp/q32w7/base), rebuild only TARGETs (BUILD_CORES, default 0-7), link target/engine there.
set -e
E=$(cd "$(dirname "$0")/../.." && pwd)   # the engine checkout this script lives in
N=$1; shift
D=/tmp/q32w7/$N
mkdir -p $D
rsync -a --exclude target --exclude .git $E/ $D/
mkdir -p $D/target/neff $D/target/gen
cp ${BASE:-/tmp/q32w7/base}/target/neff/*.neff $D/target/neff/
mkdir -p $D/target/neffcache
cd $D
V=/opt/aws_neuronx_venv_pytorch_inference_vllm_0_24_0_1_1_0
export PATH=$V/bin:$PATH NEURON_PLATFORM_TARGET_OVERRIDE=trn2
if [ $# -gt 0 ]; then BUILD_CORES=${BUILD_CORES:-0-7} python kernels/build_neffs.py target/neff "$@"; fi
NRT=/opt/aws/neuron
NEFFS=$(python -c "import sys; sys.path.insert(0, 'kernels'); import targets; print(' '.join(t['name'] for t in targets.TARGETS))")
{
  echo '.section .rodata'
  for n in $NEFFS; do
    echo ".global neff_${n}_start"; echo ".global neff_${n}_end"; echo ".balign 64"
    echo "neff_${n}_start:"; echo ".incbin \"target/neff/${n}.neff\""; echo "neff_${n}_end:"
  done
  echo '.section .note.GNU-stack,"",@progbits'
} > target/gen/neffs.S
{
  echo 'struct NeffBlob { const char* name; const char* start; const char* end; };'
  for n in $NEFFS; do echo "extern \"C\" const char neff_${n}_start[], neff_${n}_end[];"; done
  echo 'extern const NeffBlob NEFFS[] = {'
  for n in $NEFFS; do echo "  {\"${n}\", neff_${n}_start, neff_${n}_end},"; done
  echo '};'
  echo "extern const int NEFF_COUNT = sizeof(NEFFS) / sizeof(NEFFS[0]);"
} > target/gen/neff_table.cpp
g++ -O2 -std=c++17 -I$NRT/include -Isrc src/engine.cpp target/gen/neff_table.cpp target/gen/neffs.S -o target/engine \
  -L$NRT/lib -lnrt -lpthread -Wl,-rpath,$NRT/lib
echo "built $D/target/engine"
