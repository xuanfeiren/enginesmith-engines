#!/bin/bash
# profc.sh TAG CORES : 4-rank neuron-explorer profile of art_TAG/kernel.neff, zero inputs
t=$1; c=${2:-4-7}; d=$PWD/prof_$t; rm -rf $d; mkdir -p $d; cd $d
NEURON_RT_VISIBLE_CORES=$c timeout 600 /opt/aws/neuron/bin/neuron-explorer capture -n ../art_$t/kernel.neff -s p.ntff --num-exec 3 -r 4 --collectives-worker-count 4 -i 0 > cap.log 2>&1
f=$(ls p_rank_0_exec_3.ntff 2>/dev/null || ls *.ntff | tail -1)
NEURON_RT_VISIBLE_CORES=$c timeout 600 /opt/aws/neuron/bin/neuron-explorer view -n ../art_$t/kernel.neff -s $f --output-format json > view.log 2>&1
python3 -c "
import json;d=json.load(open('ntff.json'));s=d['summary'][0]
print('$t', 'total_us %.1f'%(s['total_exec_time']*1e6))"
