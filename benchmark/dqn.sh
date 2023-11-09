poetry install
OMP_NUM_THREADS=1 xvfb-run -a poetry run python -m cleanrl_utils.benchmark \
    --env-ids CartPole-v1 Acrobot-v1 MountainCar-v0 \
    --command "poetry run python cleanrl/dqn.py --no_cuda --track --capture_video" \
    --num-seeds 3 \
    --workers 9

poetry install -E atari
OMP_NUM_THREADS=1 xvfb-run -a poetry run python -m cleanrl_utils.benchmark \
    --env-ids PongNoFrameskip-v4 BeamRiderNoFrameskip-v4 BreakoutNoFrameskip-v4 \
    --command "poetry run python cleanrl/dqn_atari.py --track --capture_video" \
    --num-seeds 3 \
    --workers 1

poetry install -E jax
poetry run pip install --upgrade "jax[cuda11_cudnn82]==0.4.8" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
xvfb-run -a python -m cleanrl_utils.benchmark \
    --env-ids CartPole-v1 Acrobot-v1 MountainCar-v0 \
    --command "poetry run python cleanrl/dqn_jax.py --track --capture_video" \
    --num-seeds 3 \
    --workers 1

poetry install -E "atari jax"
poetry run pip install --upgrade "jax[cuda11_cudnn82]==0.4.8" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
xvfb-run -a python -m cleanrl_utils.benchmark \
    --env-ids PongNoFrameskip-v4 BeamRiderNoFrameskip-v4 BreakoutNoFrameskip-v4 \
    --command "poetry run python cleanrl/dqn_atari_jax.py --track --capture_video" \
    --num-seeds 3 \
    --workers 1
