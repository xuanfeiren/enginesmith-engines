#!/bin/bash
# usage: prof_moe.sh NT  (profile pmoeNT with random routing over NT*128 tokens)
NT=$1; v=pmoe$NT; d=/tmp/es-dev/prof/$v; rm -rf $d; mkdir -p $d/in; cd $d
PY=/opt/aws_neuronx_venv_pytorch_inference_vllm_0_24_0_1_1_0/bin/python
ES_DUMP=$d/in NKI_ARTIFACTS_DIR=/tmp/x $PY $(cd $(dirname $0)/../.. && pwd)/tools/build_neffs.py --one $v | grep IFMAP | sed 's/^IFMAP //' > ifmap.txt
if [ -n "$RTI" ]; then $PY -c "import numpy as np; np.save('in/rti.npy', np.load('$RTI')); np.save('in/plen.npy', np.array([[${PLEN:-0}]], np.int32))"; else $PY -c "
import numpy as np
rng = np.random.default_rng(0); T = $NT * 128
r = np.load('in/rti.npy'); 
for t in range(T):
    r[t, :4] = rng.choice(32, 4, replace=False); r[t, 4:] = [0.4, 0.3, 0.2, 0.1]
np.save('in/rti.npy', r); np.save('in/plen.npy', np.array([[T]], np.int32))
"; fi
for i in 1 2 3 4; do cat ifmap.txt; done > multi.txt
cp $(cd $(dirname $0)/../.. && pwd)/target/neff/$v.neff k.neff
NEURON_RT_VISIBLE_CORES=16-19 timeout 900 /opt/aws/neuron/bin/neuron-explorer capture -n k.neff -s p.ntff --num-exec 2 -r 4 --collectives-worker-count 4 -i 0 -m multi.txt > cap.log 2>&1
NEURON_RT_VISIBLE_CORES=16-19 timeout 900 /opt/aws/neuron/bin/neuron-explorer view -n k.neff -s p_rank_0_exec_2.ntff --output-format json > view.log 2>&1
python3 /tmp/es-dev/tools/tl.py ntff.json ${BIN:-100000} sg01 > timeline.txt
python3 -c "import json;s=json.load(open('ntff.json'))['summary'][0];print('total_us', s['total_exec_time']*1e6, 'pe_active', s.get('tensor_engine_active_time',0)*1e6, 'dma_active', s.get('dma_active_time',0)*1e6)"
