# export WANDB_ENTITY=openrlbenchmark

poetry install -E envpool
xvfb-run -a python -m cleanrl_utils.benchmark \
    --env-ids MontezumaRevenge-v5 \
    --command "poetry run python cleanrl/ppo_rnd_envpool.py --track" \
    --num-seeds 1 \
    --workers 1