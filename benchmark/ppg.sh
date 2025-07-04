# export WANDB_ENTITY=openrlbenchmark

uv pip install ".[procgen]"
xvfb-run -a uv run python -m cleanrl_utils.benchmark \
    --env-ids starpilot bossfight bigfish \
    --command "uv run python cleanrl/ppg_procgen.py --track --capture_video" \
    --num-seeds 3 \
    --workers 1
