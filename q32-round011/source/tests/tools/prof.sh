#!/bin/bash
# prof.sh NEFF OUTDIR CORES [multi.txt]: capture rank 0 of an 8-rank collective NEFF, print summary
set -e
N=$1; O=$2; C=$3; M=$4
mkdir -p $O; cd $O
EX=/opt/aws/neuron/bin/neuron-explorer
export NEURON_RT_VISIBLE_CORES=$C NEURON_RT_ROOT_COMM_ID=localhost:$((62000 + RANDOM % 1000)) NEURON_RT_DISABLE_EXECUTION_BARRIER=1
if [ -n "$M" ]; then
  $EX capture -n $N -s p.ntff --collectives-worker-count=8 -r 8 -i 0 --num-exec 3 -m $M > cap.log 2>&1
else
  $EX capture -n $N -s p.ntff --collectives-worker-count=8 -r 8 -i 0 --num-exec 3 > cap.log 2>&1
fi
ls
F=$(ls -t *exec_3*.ntff 2>/dev/null | head -1); [ -z "$F" ] && F=$(ls -t *.ntff | head -1)
$EX view -n $N -s $F --output-format summary-text > sum.txt 2>&1
grep -E "total_time|hbm_read_bytes|dma_active_time_percent|tensor_engine_active_time_percent|mbu_est|cc_op_active_time |throttle" sum.txt | head -20
