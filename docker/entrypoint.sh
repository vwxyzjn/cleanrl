#!/bin/bash
Xvfb :1 -screen 0 1024x768x24 -ac +extension GLX +render -noreset &> xvfb.log &
export DISPLAY=:1
set -e
# bash -c "echo vm.overcommit_memory=1 >> /etc/sysctl.conf" && sysctl -p
exec "$@"
