#!/bin/bash

for env_id in "HalfCheetah-v2" "Walker2d-v2" "Hopper-v2"; do
    for seed in 1 2 3; do
        poetry run python cleanrl/sac_continuous_action.py --track --capture-video --wandb-project-name cleanrl --wandb-entity openrlbenchmark --env-id $env_id --seed $seed
    done
done
