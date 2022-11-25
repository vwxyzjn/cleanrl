# export WANDB_ENTITY=openrlbenchmark

poetry install --with procgen
xvfb-run -a poetry run python -m cleanrl_utils.benchmark \
    --env-ids starpilot bossfight bigfish \
    --command "poetry run python cleanrl/ppg_procgen.py --track --capture-video" \
    --num-seeds 3 \
    --workers 1
