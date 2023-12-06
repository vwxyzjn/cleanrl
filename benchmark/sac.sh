poetry install -E mujoco
poetry run python -m cleanrl_utils.benchmark \
    --env-ids HalfCheetah-v4 Walker2d-v4 Hopper-v4 InvertedPendulum-v4 Humanoid-v4 Pusher-v4 \
    --command "poetry run python cleanrl/sac_continuous_action.py --track" \
    --num-seeds 3 \
    --workers 18 \
    --slurm-gpus-per-task 1 \
    --slurm-ntasks 1 \
    --slurm-total-cpus 10 \
    --slurm-template-path benchmark/cleanrl_1gpu.slurm_template
