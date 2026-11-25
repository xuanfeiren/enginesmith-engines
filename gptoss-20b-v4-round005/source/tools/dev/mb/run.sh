#!/bin/bash
# run.sh SCRIPT TAG CORE args... : compile+run under the GPU lock, artifacts in art_TAG
s=$1; t=$2; c=$3; shift 3
V=/opt/aws_neuronx_venv_pytorch_inference_vllm_0_24_0_1_1_0; export PATH=$V/bin:$PATH; GL=$(cd $(dirname $0)/../../../../brief/skill/scripts && pwd)/gpu_lock.py
rm -rf art_$t; mkdir -p art_$t
NKI_ARTIFACTS_DIR=$PWD/art_$t NEURON_RT_VISIBLE_CORES=$c NEURON_PLATFORM_TARGET_OVERRIDE=trn2 timeout 900 python3 $GL -- $V/bin/python $s "$@" > log_$t.txt 2>&1
grep -E "error|err|Error" log_$t.txt | head -5
