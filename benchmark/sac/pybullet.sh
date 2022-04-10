#!/bin/bash

for env_id in "HalfCheetahBulletEnv-v0" "Walker2DBulletEnv-v0" "HopperBulletEnv-v0"; do
    for seed in 1 2 3; do
        poetry run python cleanrl/sac_continuous_action.py --track --capture-video --wandb-project-name cleanrl --wandb-entity openrlbenchmark --env-id $env_id --seed $seed
    done
done
