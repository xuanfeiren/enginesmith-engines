#!/bin/bash
# runc.sh SCRIPT TAG CORES PORT args... : 4-rank compile+run
s=$1; t=$2; c=$3; port=$4; shift 4
V=/opt/aws_neuronx_venv_pytorch_inference_vllm_0_24_0_1_1_0; export PATH=$V/bin:$PATH
rm -rf art_$t; mkdir -p art_$t
NKI_ARTIFACTS_DIR=$PWD/art_$t NEURON_RT_VISIBLE_CORES=$c NEURON_RT_ROOT_COMM_ID=localhost:$port NEURON_PLATFORM_TARGET_OVERRIDE=trn2 timeout 900 $V/bin/python $s "$@" > log_$t.txt 2>&1
grep -E "error|Error|^out" log_$t.txt | head -5
