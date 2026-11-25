#!/bin/bash
# ccview.sh NEFF OUTDIR: profile rank 0 (cores 8-15), list cc ops (operation, algorithm, stream, start us, duration us)
N=$1; O=$2
timeout 600 bash /home/ubuntu/es-wave2/tasks/q32/runs/q32-20261106-022117/work/008/engine/tests/tools/prof.sh $N $O 8-15 2>&1 | grep -E " total_time|cc_op_active_time "
cd $O
F=$(ls -t *exec_3*.ntff | head -1)
timeout 600 /opt/aws/neuron/bin/neuron-explorer view -n $N -s $F --output-format json > view.log 2>&1
python3 /tmp/q32w8/tools/cclist.py ntff.json
rm -f ntff.json
