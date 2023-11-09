poetry install -E "mujoco"
python -m cleanrl_utils.benchmark \
    --env-ids HalfCheetah-v4 Walker2d-v4 Hopper-v4 InvertedPendulum-v4 Humanoid-v4 Pusher-v4 \
    --command "poetry run python cleanrl/td3_continuous_action.py --track" \
    --num-seeds 3 \
    --workers 18 \
    --slurm-gpus-per-task 1 \
    --slurm-ntasks 1 \
    --slurm-total-cpus 10 \
    --slurm-template-path benchmark/cleanrl_1gpu.slurm_template

poetry install -E "mujoco jax"
poetry run pip install --upgrade "jax[cuda11_cudnn82]==0.4.8" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
poetry run python -m cleanrl_utils.benchmark \
    --env-ids HalfCheetah-v4 Walker2d-v4 Hopper-v4 InvertedPendulum-v4 Humanoid-v4 Pusher-v4 \
    --command "poetry run python cleanrl/td3_continuous_action_jax.py --track" \
    --num-seeds 3 \
    --workers 18 \
    --slurm-gpus-per-task 1 \
    --slurm-ntasks 1 \
    --slurm-total-cpus 10 \
    --slurm-template-path benchmark/cleanrl_1gpu.slurm_template
