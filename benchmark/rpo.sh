#!/bin/bash

export WANDB_ENTITY=openrlbenchmark

# poetry install
# OMP_NUM_THREADS=1 xvfb-run -a poetry run python -m cleanrl_utils.benchmark \
#     --env-ids Pendulum-v1 BipedalWalker-v3\
#     --command "poetry run python cleanrl/rpo_continuous_action.py --cuda False --track --capture-video" \
#     --num-seeds 10 \
#     --workers 20

# OMP_NUM_THREADS=1 xvfb-run -a poetry run python -m cleanrl_utils.benchmark \
#     --env-ids BipedalWalker-v3\
#     --command "poetry run python cleanrl/rpo_continuous_action.py --cuda False --track --capture-video" \
#     --num-seeds 3 \
#     --workers 1

# dm_control
OMP_NUM_THREADS=1 xvfb-run -a poetry run python -m cleanrl_utils.benchmark \
    --env-ids dm_control/hopper-hop-v0\
    --command "poetry run python cleanrl/rpo_continuous_action.py --cuda False --track --capture-video" \
    --num-seeds 3 \
    --workers 1

# OMP_NUM_THREADS=1 xvfb-run -a poetry run python -m cleanrl_utils.benchmark \
#     --env-ids dm_control/acrobot-swingup-v0 dm_control/acrobot-swingup_sparse-v0 dm_control/ball_in_cup-catch-v0 dm_control/cartpole-balance-v0 dm_control/cartpole-balance_sparse-v0 dm_control/cartpole-swingup-v0 dm_control/cartpole-swingup_sparse-v0 dm_control/cartpole-two_poles-v0 dm_control/cartpole-three_poles-v0 dm_control/cheetah-run-v0 dm_control/dog-stand-v0 dm_control/dog-walk-v0 dm_control/dog-trot-v0 dm_control/dog-run-v0 dm_control/dog-fetch-v0 dm_control/finger-spin-v0 dm_control/finger-turn_easy-v0 dm_control/finger-turn_hard-v0 dm_control/fish-upright-v0 dm_control/fish-swim-v0 dm_control/hopper-stand-v0 dm_control/hopper-hop-v0 dm_control/humanoid-stand-v0 dm_control/humanoid-walk-v0 dm_control/humanoid-run-v0 dm_control/humanoid-run_pure_state-v0 dm_control/humanoid_CMU-stand-v0 dm_control/humanoid_CMU-run-v0 dm_control/lqr-lqr_2_1-v0 dm_control/lqr-lqr_6_2-v0 dm_control/manipulator-bring_ball-v0 dm_control/manipulator-bring_peg-v0 dm_control/manipulator-insert_ball-v0 dm_control/manipulator-insert_peg-v0 dm_control/pendulum-swingup-v0 dm_control/point_mass-easy-v0 dm_control/point_mass-hard-v0 dm_control/quadruped-walk-v0 dm_control/quadruped-run-v0 dm_control/quadruped-escape-v0 dm_control/quadruped-fetch-v0 dm_control/reacher-easy-v0 dm_control/reacher-hard-v0 dm_control/stacker-stack_2-v0 dm_control/stacker-stack_4-v0 dm_control/swimmer-swimmer6-v0 dm_control/swimmer-swimmer15-v0 dm_control/walker-stand-v0 dm_control/walker-walk-v0 dm_control/walker-run-v0 \
#     --command "poetry run python cleanrl/gymnasium_support/ppo_continuous_action.py --cuda False --track" \
#     --num-seeds 3 \
#     --workers 9