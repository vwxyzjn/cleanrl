#!/bin/bash
NUM_CORES=$(nproc --all)
export MKL_NUM_THREADS=$NUM_CORES OMP_NUM_THREADS=$NUM_CORES

###############################################
# region: Atari Pong                          #
# Dreamer with discrete latents
export CUDA_VISIBLE_DEVICES=7
(sleep 1s && python dreamer_atari.py \
    --track --capture-video \
    --env-id "BreakoutNoFrameskip-v4" \
    --exp-name "dreamer" \
    --seed 1 \
) >& /dev/null &

export CUDA_VISIBLE_DEVICES=6
(sleep 1s && python dreamer_atari.py \
    --track --capture-video \
    --env-id "BreakoutNoFrameskip-v4" \
    --exp-name "dreamer" \
    --seed 2 \
) >& /dev/null &


# Dreamer with discrete latents
export CUDA_VISIBLE_DEVICES=5
(sleep 1s && python dreamer_atari.py \
    --track --capture-video \
    --env-id "BreakoutNoFrameskip-v4" \
    --exp-name "dreamer__numenvs_4" \
    --num-envs 4 \
    --seed 1 \
) >& /dev/null &

export CUDA_VISIBLE_DEVICES=4
(sleep 1s && python dreamer_atari.py \
    --track --capture-video \
    --env-id "BreakoutNoFrameskip-v4" \
    --exp-name "dreamer__numenvs_4" \
    --num-envs 4 \
    --seed 2 \
) >& /dev/null &


# Dreamer with continuous latents
export CUDA_VISIBLE_DEVICES=3
(sleep 1s && python dreamer_atari.py \
    --track --capture-video \
    --env-id "BreakoutNoFrameskip-v4" \
    --exp-name "dreamer_cont_latents" \
    --rssm-discrete 0 \
    --seed 1 \
) >& /dev/null &

export CUDA_VISIBLE_DEVICES=2
(sleep 1s && python dreamer_atari.py \
    --track --capture-video \
    --env-id "BreakoutNoFrameskip-v4" \
    --exp-name "dreamer_cont_latents" \
    --rssm-discrete 0 \
    --seed 2 \
) >& /dev/null &

# endregion: Atari Pong                       #
###############################################