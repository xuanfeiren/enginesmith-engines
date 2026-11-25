#!/bin/bash
# var.sh NAME CORES TARGETS... (env overrides passed through; SRC= other source dir): build only TARGETS + link
set -e
N=$1; C=$2; shift 2
T=/home/ubuntu/es-wave2/tasks/q32/runs/q32-20261106-022117/work/011/engine/tests/tools
bash $T/r11/mk.sh $N ${SRC:-}
D=/tmp/q32w11/$N; cd $D
V=/opt/aws_neuronx_venv_pytorch_inference_vllm_0_24_0_1_1_0; export PATH=$V/bin:$PATH NEURON_PLATFORM_TARGET_OVERRIDE=trn2
BUILD_CORES=$C python kernels/build_neffs.py target/neff "$@" > build.log 2>&1 || { tail -30 build.log; exit 1; }
bash $T/r10/link.sh $D > /dev/null
echo "built $N: $(grep -E 'built in|cached' build.log | tr '\n' ' ')"
