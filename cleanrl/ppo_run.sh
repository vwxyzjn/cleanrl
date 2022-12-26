#!/bin/bash
hostname
xvfb-run -a -s "-screen 0 1400x900x24" python ppo_continuous_action_8M.py --cuda False --track --wandb-entity $1 --total-timesteps $2 --env-id $3 --seed $4

