poetry install -E atari
OMP_NUM_THREADS=1 xvfb-run -a poetry run python -m cleanrl_utils.benchmark \
    --env-ids PongNoFrameskip-v4 BeamRiderNoFrameskip-v4 BreakoutNoFrameskip-v4 \
    --command "poetry run python cleanrl/qdagger_dqn_atari_impalacnn.py --track --capture_video" \
    --num-seeds 3 \
    --workers 1


poetry install -E "atari jax"
poetry run pip install --upgrade "jax[cuda11_cudnn82]==0.4.8" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
xvfb-run -a poetry run python -m cleanrl_utils.benchmark \
    --env-ids PongNoFrameskip-v4 BeamRiderNoFrameskip-v4 BreakoutNoFrameskip-v4 \
    --command "poetry run python cleanrl/qdagger_dqn_atari_jax_impalacnn.py --track --capture_video" \
    --num-seeds 3 \
    --workers 1
