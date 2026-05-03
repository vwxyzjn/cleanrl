# export WANDB_ENTITY=openrlbenchmark

uv pip install ".[envpool]"
xvfb-run -a python -m cleanrl_utils.benchmark \
    --env-ids MontezumaRevenge-v5 \
    --command "uv run python cleanrl/ppo_curiosity_critic_envpool.py --track" \
    --num-seeds 1 \
    --workers 1
