#!/bin/bash
# pstep.sh NEFF OUTDIR: profile a b512 step NEFF on cores 8-15 (under the lock), json -> p.pkl, print summary + phase + stall
set -e
N=$1; O=$2; mkdir -p $O
T=/home/ubuntu/es-wave2/tasks/q32/runs/q32-20261106-022117/work/011/engine/tests/tools
$T/r10/lk.sh bash $T/prof.sh $N $O 8-15 ${MULTI:-/tmp/q32w10/profstep/in/multi.txt} | grep total_time
cd $O; F=$(ls -t *exec_3*.ntff | head -1)
/opt/aws/neuron/bin/neuron-explorer view -n $N -s $F --output-format json > /dev/null 2>&1
python /tmp/q32w11/an/red.py ntff.json p.pkl; rm -f ntff.json p_rank_0.ntff p_rank_0_exec_2.ntff
python /tmp/q32w11/an/cc.py p.pkl | tail -3
