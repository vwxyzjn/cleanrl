# export WANDB_ENTITY=openrlbenchmark

poetry install --with envpool
xvfb-run -a python -m cleanrl_utils.benchmark \
    --env-ids MontezumaRevenge-v5 \
    --command "poetry run python cleanrl/ppo_rnd_envpool.py --track --num-iterations-obs-norm-init 2 --total-timesteps 20000000" \
    --num-seeds 1 \
    --workers 1

xvfb-run -a python -m cleanrl_utils.benchmark \
    --env-ids MontezumaRevenge-v5 \
    --command "poetry run python cleanrl/ppo_rnd_envpool_faster_3.py --track --num-iterations-obs-norm-init 2 --total-timesteps 20000000" \
    --num-seeds 1 \
    --workers 1