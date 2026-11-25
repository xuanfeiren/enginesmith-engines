#!/bin/bash
# usage: one.sh VARIANT  -> compiles one NEFF variant on cores 8-11 into /tmp/es-dev/art_VARIANT
n=$1; rm -rf /tmp/es-dev/art_$n; mkdir -p /tmp/es-dev/art_$n
cd /tmp/es-dev
( time env PATH=/opt/aws_neuronx_venv_pytorch_inference_vllm_0_24_0_1_1_0/bin:$PATH NKI_ARTIFACTS_DIR=/tmp/es-dev/art_$n NEURON_RT_VISIBLE_CORES=${CORES:-16-19} NEURON_RT_ROOT_COMM_ID=localhost:${PORT:-61016} NEURON_PLATFORM_TARGET_OVERRIDE=trn2 /opt/aws_neuronx_venv_pytorch_inference_vllm_0_24_0_1_1_0/bin/python ${ENG:-$(cd $(dirname $0)/../.. && pwd)}/tools/build_neffs.py --one $n ) > /tmp/es-dev/$n.log 2>&1
tail -3 /tmp/es-dev/$n.log; grep -m3 "error" /tmp/es-dev/$n.log; ls -la /tmp/es-dev/art_$n/*.neff 2>/dev/null
