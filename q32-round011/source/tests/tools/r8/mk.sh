#!/bin/bash
# mk.sh NAME [SRC]: work dir /tmp/q32w8/NAME = copy of SRC (default the engine checkout) with the base NEFFs
set -e
E=${2:-/home/ubuntu/es-wave2/tasks/q32/runs/q32-20261106-022117/work/008/engine}
D=/tmp/q32w8/$1; mkdir -p $D
rsync -a --exclude target --exclude .git $E/ $D/
mkdir -p $D/target/neff $D/target/gen $D/target/neffcache
cp /tmp/q32w8/base_target/neff/*.neff $D/target/neff/
