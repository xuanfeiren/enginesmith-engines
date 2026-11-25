#!/bin/bash
# prof.sh NEFF NAME : profile rank 0 on cores 8-15, reduce to /tmp/q32w7/prof_NAME/red.pkl
N=$1; O=/tmp/q32w7/prof_$2
timeout 900 bash /home/ubuntu/es-wave2/tasks/q32/runs/q32-20261106-022117/work/007/engine/tests/tools/prof.sh $N $O 8-15 2>&1 | grep total_time
cd $O && timeout 900 /opt/aws/neuron/bin/neuron-explorer view -n $N -s p_rank_0_exec_3.ntff --output-format json > view.log 2>&1
timeout 1200 python3 /tmp/q32w7/red.py ntff.json red.pkl > /dev/null 2>&1; rm -f ntff.json *.ntff
