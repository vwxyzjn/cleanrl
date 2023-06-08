poetry install -E atari
OMP_NUM_THREADS=1 xvfb-run -a poetry run python -m cleanrl_utils.benchmark \
    --env-ids BreakoutNoFrameskip-v4 \
    --command "poetry run python cleanrl/qdagger_dqn_atari_impalacnn.py --teacher-policy-hf-repo cleanrl/BreakoutNoFrameskip-v4-dqn_atari-seed1 --track --capture-video" \
    --num-seeds 3 \
    --workers 1
OMP_NUM_THREADS=1 xvfb-run -a poetry run python -m cleanrl_utils.benchmark \
    --env-ids PongNoFrameskip-v4 \
    --command "poetry run python cleanrl/qdagger_dqn_atari_impalacnn.py --teacher-policy-hf-repo cleanrl/PongNoFrameskip-v4-dqn_atari-seed1 --track --capture-video" \
    --num-seeds 3 \
    --workers 1
OMP_NUM_THREADS=1 xvfb-run -a poetry run python -m cleanrl_utils.benchmark \
    --env-ids BeamRiderNoFrameskip-v4 \
    --command "poetry run python cleanrl/qdagger_dqn_atari_impalacnn.py --teacher-policy-hf-repo cleanrl/BeamRiderNoFrameskip-v4-dqn_atari-seed1 --track --capture-video" \
    --num-seeds 3 \
    --workers 1


poetry install -E "atari jax"
poetry run pip install --upgrade "jax[cuda]==0.3.17" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
OMP_NUM_THREADS=1 xvfb-run -a poetry run python -m cleanrl_utils.benchmark \
    --env-ids BreakoutNoFrameskip-v4 \
    --command "poetry run python cleanrl/qdagger_dqn_atari_jax_impalacnn.py --teacher-policy-hf-repo cleanrl/BreakoutNoFrameskip-v4-dqn_atari-seed1 --track --capture-video" \
    --num-seeds 3 \
    --workers 1
OMP_NUM_THREADS=1 xvfb-run -a poetry run python -m cleanrl_utils.benchmark \
    --env-ids PongNoFrameskip-v4 \
    --command "poetry run python cleanrl/qdagger_dqn_atari_jax_impalacnn.py --teacher-policy-hf-repo cleanrl/PongNoFrameskip-v4-dqn_atari-seed1 --track --capture-video" \
    --num-seeds 3 \
    --workers 1
OMP_NUM_THREADS=1 xvfb-run -a poetry run python -m cleanrl_utils.benchmark \
    --env-ids BeamRiderNoFrameskip-v4 \
    --command "poetry run python cleanrl/qdagger_dqn_atari_jax_impalacnn.py --teacher-policy-hf-repo cleanrl/BeamRiderNoFrameskip-v4-dqn_atari-seed1 --track --capture-video" \
    --num-seeds 3 \
    --workers 1

