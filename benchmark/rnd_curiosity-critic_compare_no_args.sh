# export WANDB_ENTITY=openrlbenchmark

xvfb-run -a python -m cleanrl_utils.benchmark --env-ids MontezumaRevenge-v5 --command "uv run python cleanrl/ppo_curiosity_critic_envpool.py --track" --num-seeds 1 --start-seed 1 --workers 1
xvfb-run -a python -m cleanrl_utils.benchmark --env-ids MontezumaRevenge-v5 --command "uv run python cleanrl/ppo_rnd_envpool.py --track" --num-seeds 1 --start-seed 1 --workers 1
xvfb-run -a python -m cleanrl_utils.benchmark --env-ids MontezumaRevenge-v5 --command "uv run python cleanrl/ppo_curiosity_critic_envpool.py --track" --num-seeds 1 --start-seed 2 --workers 1
xvfb-run -a python -m cleanrl_utils.benchmark --env-ids MontezumaRevenge-v5 --command "uv run python cleanrl/ppo_rnd_envpool.py --track" --num-seeds 1 --start-seed 2 --workers 1
xvfb-run -a python -m cleanrl_utils.benchmark --env-ids MontezumaRevenge-v5 --command "uv run python cleanrl/ppo_curiosity_critic_envpool.py --track" --num-seeds 1 --start-seed 3 --workers 1
xvfb-run -a python -m cleanrl_utils.benchmark --env-ids MontezumaRevenge-v5 --command "uv run python cleanrl/ppo_rnd_envpool.py --track" --num-seeds 1 --start-seed 3 --workers 1
