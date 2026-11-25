#!/bin/bash
# prof1.sh TAG CORE : neuron-explorer profile of art_TAG/kernel.neff (zero inputs) -> total exec us + engine busy
GL=$(cd $(dirname $0)/../../../../brief/skill/scripts && pwd)/gpu_lock.py
t=$1; c=${2:-4}; d=$PWD/prof_$t; rm -rf $d; mkdir -p $d; cd $d
NEURON_RT_VISIBLE_CORES=$c timeout 600 python3 $GL -- /opt/aws/neuron/bin/neuron-explorer capture -n ../art_$t/kernel.neff -s p.ntff --num-exec 3 > cap.log 2>&1
f=$(ls p*exec_3*.ntff p_exec_3.ntff 2>/dev/null | head -1); [ -z "$f" ] && f=$(ls *.ntff | tail -1)
timeout 600 /opt/aws/neuron/bin/neuron-explorer view -n ../art_$t/kernel.neff -s $f --output-format json > view.log 2>&1
python3 -c "
import json;d=json.load(open('ntff.json'));s=d['summary'][0]
print('$t', 'total_us %.1f'%(s['total_exec_time']*1e6), {k:round(v*1e6,1) for k,v in s.items() if k.endswith('active_time') and isinstance(v,float)})"
