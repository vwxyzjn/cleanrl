# export WANDB_ENTITY=openrlbenchmark

# comparison with PPO-DNA paper results on "Atari-5" envs
poetry install -E envpool
OMP_NUM_THREADS=1 xvfb-run -a python -m cleanrl_utils.benchmark \
    --env-ids BattleZone-v5 DoubleDunk-v5 NameThisGame-v5 Phoenix-v5 Qbert-v5 \
    --command "poetry run python cleanrl/ppo_dna_atari_envpool.py --anneal-lr False --total-timesteps 50000000 --track" \
    --num-seeds 3 \
    --workers 1

# comparison with CleanRL ppo_atari_envpool.py
poetry install -E envpool
OMP_NUM_THREADS=1 xvfb-run -a python -m cleanrl_utils.benchmark \
    --env-ids Pong-v5 BeamRider-v5 Breakout-v5 Tennis-v5 \
    --command "poetry run python cleanrl/ppo_dna_atari_envpool.py --track" \
    --num-seeds 3 \
    --workers 1
