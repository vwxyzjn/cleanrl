poetry install
OMP_NUM_THREADS=1 xvfb-run -a poetry run python -m cleanrl_utils.benchmark \
    --env-ids CartPole-v1 Acrobot-v1 MountainCar-v0 \
    --command "poetry run python cleanrl/c51.py --cuda False --track --capture-video" \
    --num-seeds 3 \
    --workers 9

poetry install --with atari
OMP_NUM_THREADS=1 xvfb-run -a poetry run python -m cleanrl_utils.benchmark \
    --env-ids PongNoFrameskip-v4 BeamRiderNoFrameskip-v4 BreakoutNoFrameskip-v4 \
    --command "poetry run python cleanrl/c51_atari.py --track --capture-video" \
    --num-seeds 3 \
    --workers 1

poetry install -E "jax"
poetry run pip install --upgrade "jax[cuda]==0.3.14" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
xvfb-run -a python -m cleanrl_utils.benchmark \
    --env-ids CartPole-v1 Acrobot-v1 MountainCar-v0 \
    --command "poetry run python cleanrl/c51_jax.py --track --capture-video" \
    --num-seeds 3 \
    --workers 1

poetry install -E "atari jax"
poetry run pip install --upgrade "jax[cuda]==0.3.14" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
xvfb-run -a python -m cleanrl_utils.benchmark \
    --env-ids PongNoFrameskip-v4 BeamRiderNoFrameskip-v4 BreakoutNoFrameskip-v4 \
    --command "poetry run python cleanrl/c51_atari_jax.py --track --capture-video" \
    --num-seeds 3 \
    --workers 1
