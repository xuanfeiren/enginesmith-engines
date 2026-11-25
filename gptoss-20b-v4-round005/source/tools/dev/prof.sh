#!/bin/bash
# usage: prof.sh VARIANT NEFF POS -> /tmp/es-dev/prof/VARIANT/{ntff.json, timeline}
v=$1; neff=$2; pos=$3; d=/tmp/es-dev/prof/$v; rm -rf $d; mkdir -p $d/in; cd $d
PY=/opt/aws_neuronx_venv_pytorch_inference_vllm_0_24_0_1_1_0/bin/python
ES_DUMP=$d/in ES_POS=$pos NKI_ARTIFACTS_DIR=/tmp/x $PY ${ENG:-$(cd $(dirname $0)/../.. && pwd)}/tools/build_neffs.py --one $v | grep IFMAP | sed 's/^IFMAP //' > ifmap.txt
for i in 1 2 3 4; do cat ifmap.txt; done > multi.txt
cp $neff k.neff
NEURON_RT_VISIBLE_CORES=${CORES:-16-19} timeout 600 /opt/aws/neuron/bin/neuron-explorer capture -n k.neff -s p.ntff --num-exec 3 -r 4 --collectives-worker-count 4 -i 0 -m multi.txt > cap.log 2>&1
NEURON_RT_VISIBLE_CORES=${CORES:-16-19} timeout 600 /opt/aws/neuron/bin/neuron-explorer view -n k.neff -s p_rank_0_exec_3.ntff --output-format json > view.log 2>&1
python3 /tmp/es-dev/tools/tl.py ntff.json ${BIN:-20000} sg01 > timeline.txt
python3 -c "import json;s=json.load(open('ntff.json'))['summary'][0];print('total_us', s['total_exec_time']*1e6, 'pe_active', s['tensor_engine_active_time']*1e6 if 'tensor_engine_active_time' in s else '', 'dma_active', s.get('dma_active_time',0)*1e6)"
