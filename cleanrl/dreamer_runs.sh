#!/bin/bash
NUM_CORES=$(nproc --all)
export MKL_NUM_THREADS=$NUM_CORES OMP_NUM_THREADS=$NUM_CORES

###############################################
# region: Atari Breakout                      #

# # Dreamer Disc Baseline, B=50, T=50, train-every=16
# # TODO: Requeue experiment with fixed buffer size computation
# export CUDA_VISIBLE_DEVICES=7
# for seed in 1 2; do
#     (sleep 1s && python dreamer_atari.py \
#         --track --capture-video \
#         --env-id "BreakoutNoFrameskip-v4" \
#         --batch-size 50 --batch-length 50 \
#         --train-every 16 \
#         --exp-name "dreamer_B_50_T_50_trnev_16" \
#         --seed $seed \
#     ) >& /dev/null &
# done

# # Dreamer Disc Baseline, B=50, T=50, train-every=16
# # Larger buffer size of 2 millions
# export CUDA_VISIBLE_DEVICES=7
# for seed in 1 2; do
#     (sleep 1s && python dreamer_atari.py \
#         --track --capture-video \
#         --env-id "BreakoutNoFrameskip-v4" \
#         --batch-size 50 --batch-length 50 \
#         --train-every 16 \
#         --buffer-size 2000000 \
#         --exp-name "dreamer_B_50_T_50_trnev_16_BUF_2e6" \
#         --seed $seed \
#     ) >& /dev/null &
# done

# # Dreamer Disc Baseline, B=32, T=50, train-every=16
# export CUDA_VISIBLE_DEVICES=6
# for seed in 1 2; do
#     (sleep 1s && python dreamer_atari.py \
#         --track --capture-video \
#         --env-id "BreakoutNoFrameskip-v4" \
#         --batch-size 32 --batch-length 50 \
#         --train-every 16 \
#         --exp-name "dreamer_B_32_T_50_trnev_16" \
#         --seed $seed \
#     ) >& /dev/null &
# done

# # Dreamer Disc Baseline, B=16, T=50, train-every=16
# export CUDA_VISIBLE_DEVICES=5
# for seed in 1 2; do
#     (sleep 1s && python dreamer_atari.py \
#         --track --capture-video \
#         --env-id "BreakoutNoFrameskip-v4" \
#         --batch-size 16 --batch-length 50 \
#         --train-every 16 \
#         --exp-name "dreamer_B_16_T_50_trnev_16" \
#         --seed $seed \
#     ) >& /dev/null &
# done

# # Dreamer Disc Baseline, B=50, T=50, train-every=8
# export CUDA_VISIBLE_DEVICES=4
# for seed in 1 2; do
#     (sleep 1s && python dreamer_atari.py \
#         --track --capture-video \
#         --env-id "BreakoutNoFrameskip-v4" \
#         --batch-size 50 --batch-length 50 \
#         --train-every 8 \
#         --exp-name "dreamer_B_50_T_50_trnev_8" \
#         --seed $seed \
#     ) >& /dev/null &
# done

# # Dreamer Disc Baseline, B=50, T=50, train-every=4
# export CUDA_VISIBLE_DEVICES=3
# for seed in 1 2; do
#     (sleep 1s && python dreamer_atari.py \
#         --track --capture-video \
#         --env-id "BreakoutNoFrameskip-v4" \
#         --batch-size 50 --batch-length 50 \
#         --train-every 4 \
#         --exp-name "dreamer_B_50_T_50_trnev_4" \
#         --seed $seed \
#     ) >& /dev/null &
# done

# # Dreamer Disc, B=50, T=50, train-every=16, using Torch Distributions's Bernoulli dist
# export CUDA_VISIBLE_DEVICES=2
# for seed in 1 2; do
#     (sleep 1s && python dreamer_atari_thdbern.py \
#         --track --capture-video \
#         --env-id "BreakoutNoFrameskip-v4" \
#         --batch-size 50 --batch-length 50 \
#         --train-every 16 \
#         --exp-name "dreamer_thdbern_B_50_T_50_trnev_16" \
#         --seed $seed \
#     ) >& /dev/null &
# done

# endregion: Atari Breakout                   #
###############################################

###############################################
# region: Atari Pong                           #
# export CUDA_VISIBLE_DEVICES=6
# for seed in 1 2; do
# for seed in 3 4; do
#     (sleep 1s && python dreamer_atari.py \
#         --track --capture-video \
#         --env-id "PongNoFrameskip-v4" \
#         --batch-size 50 --batch-length 50 \
#         --train-every 16 \
#         --exp-name "dreamer_B_50_T_50_trnev_16" \
#         --seed $seed \
#     ) >& /dev/null &
# done

# export CUDA_VISIBLE_DEVICES=5
# for seed in 1 2; do
#     (sleep 1s && python dreamer_atari_thdbern.py \
#         --track --capture-video \
#         --env-id "PongNoFrameskip-v4" \
#         --batch-size 50 --batch-length 50 \
#         --train-every 16 \
#         --exp-name "dreamer_thdbern_B_50_T_50_trnev_16" \
#         --seed $seed \
#     ) >& /dev/null &
# done

# Dreamer Disc Baseline, B=50, T=50, train-every=4
# export CUDA_VISIBLE_DEVICES=0
# for seed in 1 2; do
#     (sleep 1s && python dreamer_atari.py \
#         --track --capture-video \
#         --env-id "PongNoFrameskip-v4" \
#         --batch-size 50 --batch-length 50 \
#         --train-every 16 \
#         --buffer-size 2000000 \
#         --exp-name "dreamer_B_50_T_50_trnev_16_BUF_2e6" \
#         --seed $seed \
#     ) >& /dev/null &
# done

# endregion: Atari Pong                       #
###############################################

###############################################
# region: Atari Freeway                       #
# export CUDA_VISIBLE_DEVICES=0
# for seed in 1 2; do
#     (sleep 1s && python dreamer_atari.py \
#         --track --capture-video \
#         --env-id "FreewayNoFrameskip-v4" \
#         --batch-size 50 --batch-length 50 \
#         --train-every 16 \
#         --exp-name "dreamer_B_50_T_50_trnev_16" \
#         --seed $seed \
#     ) >& /dev/null &
# done

# Dreamer Disc Baseline, B=50, T=50, train-every=8
# export CUDA_VISIBLE_DEVICES=2
# for seed in 1 2; do
#     (sleep 1s && python dreamer_atari.py \
#         --track --capture-video \
#         --env-id "FreewayNoFrameskip-v4" \
#         --batch-size 50 --batch-length 50 \
#         --train-every 16 \
#         --buffer-size 2000000 \
#         --exp-name "dreamer_B_50_T_50_trnev_16_BUF_2e6" \
#         --seed $seed \
#     ) >& /dev/null &
# done

# # Dreamer Disc Baseline, B=50, T=50, train-every=4
# export CUDA_VISIBLE_DEVICES=1
# for seed in 1 2; do
#     (sleep 1s && python dreamer_atari.py \
#         --track --capture-video \
#         --env-id "FreewayNoFrameskip-v4" \
#         --batch-size 50 --batch-length 50 \
#         --train-every 4 \
#         --buffer-size 2000000 \
#         --exp-name "dreamer_B_50_T_50_trnev_4_BUF_2e6" \
#         --seed $seed \
#     ) >& /dev/null &
# done

# endregion: Atari Freeway                    #
###############################################