#!/bin/sh
Xvfb :1 -screen 0 1024x768x24 -ac +extension GLX +render -noreset &> xvfb.log &
export DISPLAY=:1
set -e
wandb login $WANDB
git pull

# enter shared memory
cp -r /workspace/cleanrl/cleanrl /dev/shm/cleanrl
cd /dev/shm/cleanrl
exec "$@"