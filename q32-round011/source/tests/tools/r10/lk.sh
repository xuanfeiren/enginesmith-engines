#!/bin/bash
# lk.sh CMD...: run CMD under the session GPU lock (fd open read-only: /tmp is sticky)
exec 9</tmp/engine-gpu-0.lock
flock 9
"$@"
