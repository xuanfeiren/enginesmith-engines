#!/bin/bash
# usage: profneff.sh NEFF TAG [core]  -> prints total_time; leaves TAG.json (per-instruction trace)
set -e
export PATH=/opt/aws/neuron/bin:$PATH
n=$1; t=$2; c=${3:-8}
d=$(mktemp -d /tmp/prof-XXXX); cp $n $d/k.neff; cd $d
NEURON_RT_VISIBLE_CORES=$c timeout 900 neuron-explorer capture -n k.neff -s p.ntff --num-exec 3 > cap.log 2>&1
timeout 900 neuron-explorer view -n k.neff -s p_exec_3.ntff --output-format summary-text > sum.txt 2>&1
echo "$t total_time $(grep -E '^\s+total_time ' sum.txt | awk '{print $2*1e6}') us  dma_active $(grep -E '^\s+dma_active_time_percent' sum.txt | awk '{print $2}')"
if [ -n "$JSON" ]; then timeout 900 neuron-explorer view -n k.neff -s p_exec_3.ntff --output-format json > /dev/null 2>&1; mv ntff.json $OLDPWD/$t.json; fi
cd /; rm -rf $d
