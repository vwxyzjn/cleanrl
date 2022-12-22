#!/bin/bash
NUM_CORES=$(nproc --all)
export MKL_NUM_THREADS=$NUM_CORES OMP_NUM_THREADS=$NUM_CORES

###############################################
# region: Atari Breakout                      #

    # region: Baseline experiments
    # ## Seed 1
    # export CUDA_VISIBLE_DEVICES=7
    # (sleep 1s && python dreamer_atari.py \
    #     --track --capture-video \
    #     --env-id "BreakoutNoFrameskip-v4" \
    #     --total-timesteps 10000000 \
    #     --batch-size 50 --batch-length 50 \
    #     --train-every 16 \
    #     --exp-name "dreamer_B_50_T_50_trnev_16" \
    #     --seed 1 \
    # ) >& /dev/null &
    # ## Seed 2
    # export CUDA_VISIBLE_DEVICES=6
    # (sleep 1s && python dreamer_atari.py \
    #     --track --capture-video \
    #     --env-id "BreakoutNoFrameskip-v4" \
    #     --total-timesteps 10000000 \
    #     --batch-size 50 --batch-length 50 \
    #     --train-every 16 \
    #     --exp-name "dreamer_B_50_T_50_trnev_16" \
    #     --seed 2 \
    # ) >& /dev/null &
    # endregion: Baseline experiments

# endregion: Atari Breakout                   #
###############################################

###############################################
# region: Atari Pong                          #

    # region: Baseline experiments
    # ## Seed 1
    # export CUDA_VISIBLE_DEVICES=7
    # (sleep 1s && python dreamer_atari.py \
    #     --track --capture-video \
    #     --env-id "PongNoFrameskip-v4" \
    #     --total-timesteps 3000000 \
    #     --batch-size 50 --batch-length 50 \
    #     --train-every 16 \
    #     --exp-name "dreamer_B_50_T_50_trnev_16" \
    #     --seed 1 \
    # ) >& /dev/null &
    # ## Seed 2
    # export CUDA_VISIBLE_DEVICES=6
    # (sleep 1s && python dreamer_atari.py \
    #     --track --capture-video \
    #     --env-id "PongNoFrameskip-v4" \
    #     --total-timesteps 3000000 \
    #     --batch-size 50 --batch-length 50 \
    #     --train-every 16 \
    #     --exp-name "dreamer_B_50_T_50_trnev_16" \
    #     --seed 2 \
    # ) >& /dev/null &
    # endregion: Baseline experiments

    # region: Investigating trade off of the number of parallel envs
    # ## num-envs = 4
    # ## Seed 1
    # export CUDA_VISIBLE_DEVICES=5
    # (sleep 1s && python dreamer_atari.py \
    #     --track --capture-video \
    #     --env-id "PongNoFrameskip-v4" \
    #     --total-timesteps 3000000 \
    #     --batch-size 50 --batch-length 50 \
    #     --train-every 16 \
    #     --num-envs 4 \
    #     --exp-name "dreamer_B_50_T_50_trnev_16_nenvs_4" \
    #     --seed 1 \
    # ) >& /dev/null &
    # ## Seed 2
    # export CUDA_VISIBLE_DEVICES=4
    # (sleep 1s && python dreamer_atari.py \
    #     --track --capture-video \
    #     --env-id "PongNoFrameskip-v4" \
    #     --total-timesteps 3000000 \
    #     --batch-size 50 --batch-length 50 \
    #     --train-every 16 \
    #     --num-envs 4 \
    #     --exp-name "dreamer_B_50_T_50_trnev_16_nenvs_4" \
    #     --seed 2 \
    # ) >& /dev/null &

    # ## num-envs = 8
    # ## Seed 1
    # export CUDA_VISIBLE_DEVICES=3
    # (sleep 1s && python dreamer_atari.py \
    #     --track --capture-video \
    #     --env-id "PongNoFrameskip-v4" \
    #     --total-timesteps 3000000 \
    #     --batch-size 50 --batch-length 50 \
    #     --train-every 16 \
    #     --num-envs 8 \
    #     --exp-name "dreamer_B_50_T_50_trnev_16_nenvs_8" \
    #     --seed 1 \
    # ) >& /dev/null &
    # ## Seed 2
    # export CUDA_VISIBLE_DEVICES=2
    # (sleep 1s && python dreamer_atari.py \
    #     --track --capture-video \
    #     --env-id "PongNoFrameskip-v4" \
    #     --total-timesteps 3000000 \
    #     --batch-size 50 --batch-length 50 \
    #     --train-every 16 \
    #     --num-envs 8 \
    #     --exp-name "dreamer_B_50_T_50_trnev_16_nenvs_8" \
    #     --seed 2 \
    # ) >& /dev/null &
    # endregion: Baseline experiments

    # region: Investigating batch length trade off on walltime and perf
    # # Batch length of 32
    # ## Seed 1
    # export CUDA_VISIBLE_DEVICES=5
    # (sleep 1s && python dreamer_atari.py \
    #     --track --capture-video \
    #     --env-id "PongNoFrameskip-v4" \
    #     --total-timesteps 3000000 \
    #     --batch-size 50 --batch-length 32 \
    #     --train-every 16 \
    #     --exp-name "dreamer_B_50_T_32_trnev_16" \
    #     --seed 1 \
    # ) >& /dev/null &
    # ## Seed 2
    # export CUDA_VISIBLE_DEVICES=4
    # (sleep 1s && python dreamer_atari.py \
    #     --track --capture-video \
    #     --env-id "PongNoFrameskip-v4" \
    #     --total-timesteps 3000000 \
    #     --batch-size 50 --batch-length 32 \
    #     --train-every 16 \
    #     --exp-name "dreamer_B_50_T_32_trnev_16" \
    #     --seed 2 \
    # ) >& /dev/null &

    # # Batch length of 20
    ## Seed 1
    # export CUDA_VISIBLE_DEVICES=3
    # (sleep 1s && python dreamer_atari.py \
    #     --track --capture-video \
    #     --env-id "PongNoFrameskip-v4" \
    #     --total-timesteps 3000000 \
    #     --batch-size 50 --batch-length 20 \
    #     --train-every 16 \
    #     --exp-name "dreamer_B_50_T_20_trnev_16" \
    #     --seed 1 \
    # ) >& /dev/null &
    # ## Seed 2
    # export CUDA_VISIBLE_DEVICES=2
    # (sleep 1s && python dreamer_atari.py \
    #     --track --capture-video \
    #     --env-id "PongNoFrameskip-v4" \
    #     --total-timesteps 3000000 \
    #     --batch-size 50 --batch-length 20 \
    #     --train-every 16 \
    #     --exp-name "dreamer_B_50_T_20_trnev_16" \
    #     --seed 2 \
    # ) >& /dev/null &
    # endregion: Investigating batch length trade off on walltime and perf

    # region: Investigating batch size trade off on walltime and perf
    # # Batch size 32
    # ## Seed 1
    # export CUDA_VISIBLE_DEVICES=1
    # (sleep 1s && python dreamer_atari.py \
    #     --track --capture-video \
    #     --env-id "PongNoFrameskip-v4" \
    #     --total-timesteps 3000000 \
    #     --batch-size 32 --batch-length 50 \
    #     --train-every 16 \
    #     --exp-name "dreamer_B_32_T_50_trnev_16" \
    #     --seed 1 \
    # ) >& /dev/null &
    # ## Seed 2
    # export CUDA_VISIBLE_DEVICES=0
    # (sleep 1s && python dreamer_atari.py \
    #     --track --capture-video \
    #     --env-id "PongNoFrameskip-v4" \
    #     --total-timesteps 3000000 \
    #     --batch-size 32 --batch-length 50 \
    #     --train-every 16 \
    #     --exp-name "dreamer_B_32_T_50_trnev_16" \
    #     --seed 2 \
    # ) >& /dev/null &
    # # endregion: Investigating batch size trade off on walltime and perf

# endregion: Atari Pong                       #
###############################################

###############################################
# region: Atari Freeway                       #

# endregion: Atari Freeway                    #
###############################################