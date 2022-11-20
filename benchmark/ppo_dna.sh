# export WANDB_ENTITY=openrlbenchmark

# comparison with PPO-DNA paper results on "Atari-5" envs
poetry install -E envpool
xvfb-run -a python -m cleanrl_utils.benchmark \
    --env-ids BattleZone-v5 DoubleDunk-v5 NameThisGame-v5 Phoenix-v5 Qbert-v5 \
    --command "poetry run python cleanrl/ppo_dna_atari_envpool.py --anneal-lr False --track" \
    --num-seeds 3 \
    --workers 1
