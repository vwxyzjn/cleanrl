poetry install
OMP_NUM_THREADS=1 xvfb-run -a poetry run python -m cleanrl_utils.benchmark \
    --env-ids CartPole-v1 Acrobot-v1 MountainCar-v0 \
    --command "poetry run python cleanrl/dqn_munchausen.py --no_cuda --track --capture_video" \
    --num-seeds 3 \
    --workers 9