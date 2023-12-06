poetry run python cleanrl/dqn_jax.py --env-id CartPole-v1 --save-model --upload-model --hf-entity cleanrl
poetry run python cleanrl/dqn_atari_jax.py --env-id SeaquestNoFrameskip-v4  --save-model --upload-model --hf-entity cleanrl

xvfb-run -a poetry run python -m cleanrl_utils.benchmark \
    --env-ids CartPole-v1 Acrobot-v1 MountainCar-v0 \
    --command "poetry run python cleanrl/dqn.py --no_cuda --track --capture_video --save-model --upload-model --hf-entity cleanrl" \
    --num-seeds 1 \
    --workers 1

CUDA_VISIBLE_DEVICES="-1" xvfb-run -a poetry run python -m cleanrl_utils.benchmark \
    --env-ids CartPole-v1 Acrobot-v1 MountainCar-v0 \
    --command "poetry run python cleanrl/dqn_jax.py --track --capture_video --save-model --upload-model --hf-entity cleanrl" \
    --num-seeds 1 \
    --workers 1

xvfb-run -a python -m cleanrl_utils.benchmark \
    --env-ids PongNoFrameskip-v4 BeamRiderNoFrameskip-v4 BreakoutNoFrameskip-v4 \
    --command "poetry run python cleanrl/dqn_atari_jax.py --track --capture_video --save-model --upload-model --hf-entity cleanrl" \
    --num-seeds 1 \
    --workers 1

xvfb-run -a python -m cleanrl_utils.benchmark \
    --env-ids PongNoFrameskip-v4 BeamRiderNoFrameskip-v4 BreakoutNoFrameskip-v4 \
    --command "poetry run python cleanrl/dqn_atari.py --track --capture_video --save-model --upload-model --hf-entity cleanrl" \
    --num-seeds 1 \
    --workers 1

python -m cleanrl_utils.benchmark \
    --env-ids Pong-v5 BeamRider-v5 Breakout-v5 \
    --command "poetry run python cleanrl/ppo_atari_envpool_xla_jax_scan.py --track --save-model --upload-model --hf-entity cleanrl" \
    --num-seeds 1 \
    --workers 1

CUDA_VISIBLE_DEVICES="1" taskset --cpu-list 16,17,18,19,20,21,22,23 python -m cleanrl_utils.benchmark \
    --env-ids Breakout-v5 \
    --command "poetry run python cleanrl/ppo_atari_envpool_xla_jax_scan.py --track --save-model --upload-model --hf-entity cleanrl" \
    --num-seeds 1 \
    --workers 1
