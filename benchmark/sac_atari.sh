poetry install -E atari
OMP_NUM_THREADS=1 python -m cleanrl_utils.benchmark \
    --env-ids PongNoFrameskip-v4 BreakoutNoFrameskip-v4 BeamRiderNoFrameskip-v4 \
    --command "poetry run python cleanrl/sac_atari.py --track" \
    --num-seeds 3 \
    --workers 2
