# export WANDB_ENTITY=openrlbenchmark

poetry install -E envpool
xvfb-run -a python -m cleanrl_utils.benchmark \
    --env-ids MontezumaRevenge-v5 \
    --command "poetry run python cleanrl/ppo_rnd_envpool.py --total-timesteps 2000000000 --ent-coef 0.001 --num-envs 128 --gamma 0.999 --int-gamma 0.99 --learning-rate 0.0001 --track --capture-video" \
    --num-seeds 1 \
    --workers 1