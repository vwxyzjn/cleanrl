#!/bin/sh
Xvfb :1 -screen 0 1024x768x24 -ac +extension GLX +render -noreset &> xvfb.log &
export DISPLAY=:1
set -e
wandb login $WANDB
bash -c "echo vm.overcommit_memory=1 >> /etc/sysctl.conf" && sysctl -p
git pull
exec "$@"