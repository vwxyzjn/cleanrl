poetry install
OMP_NUM_THREADS=1 xvfb-run -a poetry run python -m cleanrl_utils.benchmark \
    --env-ids CartPole-v1 Acrobot-v1 MountainCar-v0 \
    --command "poetry run python cleanrl/pqn.py --no_cuda --track" \
    --num-seeds 3 \
    --workers 9 \
    --slurm-gpus-per-task 1 \
    --slurm-ntasks 1 \
    --slurm-total-cpus 10 \
    --slurm-template-path benchmark/cleanrl_1gpu.slurm_template

poetry install -E envpool
poetry run python -m cleanrl_utils.benchmark \
    --env-ids Breakout-v5 SpaceInvaders-v5 BeamRider-v5 Pong-v5 MsPacman-v5 \
    --command "poetry run python cleanrl/pqn_atari_envpool.py --track" \
    --num-seeds 3 \
    --workers 9 \
    --slurm-gpus-per-task 1 \
    --slurm-ntasks 1 \
    --slurm-total-cpus 10 \
    --slurm-template-path benchmark/cleanrl_1gpu.slurm_template

poetry install -E envpool
poetry run python -m cleanrl_utils.benchmark \
    --env-ids Breakout-v5 SpaceInvaders-v5 BeamRider-v5 Pong-v5 MsPacman-v5 \
    --command "poetry run python cleanrl/pqn_atari_envpool_lstm.py --track" \
    --num-seeds 3 \
    --workers 9 \
    --slurm-gpus-per-task 1 \
    --slurm-ntasks 1 \
    --slurm-total-cpus 10 \
    --slurm-template-path benchmark/cleanrl_1gpu.slurm_template
