# export WANDB_ENTITY=openrlbenchmark

poetry install
OMP_NUM_THREADS=1 xvfb-run -a poetry run python -m cleanrl_utils.benchmark \
    --env-ids CartPole-v1 Acrobot-v1 MountainCar-v0 \
    --command "poetry run python cleanrl/ppo.py --cuda False --track --capture-video" \
    --num-seeds 3 \
    --workers 9

poetry install -E atari
OMP_NUM_THREADS=1 xvfb-run -a poetry run python -m cleanrl_utils.benchmark \
    --env-ids PongNoFrameskip-v4 BeamRiderNoFrameskip-v4 BreakoutNoFrameskip-v4 \
    --command "poetry run python cleanrl/ppo_atari.py --track --capture-video" \
    --num-seeds 3 \
    --workers 3

poetry install -E atari
OMP_NUM_THREADS=1 xvfb-run -a poetry run python -m cleanrl_utils.benchmark \
    --env-ids PongNoFrameskip-v4 BeamRiderNoFrameskip-v4 BreakoutNoFrameskip-v4 \
    --command "poetry run python cleanrl/ppo_atari_lstm.py --track --capture-video" \
    --num-seeds 3 \
    --workers 3

poetry install -E envpool
xvfb-run -a poetry run python -m cleanrl_utils.benchmark \
    --env-ids Pong-v5 BeamRider-v5 Breakout-v5 \
    --command "poetry run python cleanrl/ppo_atari_envpool.py --track --capture-video" \
    --num-seeds 3 \
    --workers 1

poetry install -E "mujoco pybullet"
poetry run python -c "import mujoco_py"
OMP_NUM_THREADS=1 xvfb-run -a poetry run python -m cleanrl_utils.benchmark \
    --env-ids HalfCheetah-v2 Walker2d-v2 Hopper-v2 \
    --command "poetry run python cleanrl/ppo_continuous_action.py --cuda False --track --capture-video" \
    --num-seeds 3 \
    --workers 9

poetry install -E procgen
xvfb-run -a poetry run python -m cleanrl_utils.benchmark \
    --env-ids starpilot bossfight bigfish \
    --command "poetry run python cleanrl/ppo_procgen.py --track --capture-video" \
    --num-seeds 3 \
    --workers 1

poetry install -E atari
xvfb-run -a poetry run python -m cleanrl_utils.benchmark \
    --env-ids PongNoFrameskip-v4 BeamRiderNoFrameskip-v4 BreakoutNoFrameskip-v4 \
    --command "poetry run torchrun --standalone --nnodes=1 --nproc_per_node=2 cleanrl/ppo_atari_multigpu.py --track --capture-video" \
    --num-seeds 3 \
    --workers 1

poetry install -E "pettingzoo atari"
poetry run AutoROM --accept-license
xvfb-run -a poetry run python -m cleanrl_utils.benchmark \
    --env-ids pong_v3 surround_v2 tennis_v3  \
    --command "poetry run python cleanrl/ppo_pettingzoo_ma_atari.py --track --capture-video" \
    --num-seeds 3 \
    --workers 3

# IMPORTANT: see specific Isaac Gym installation at
# https://docs.cleanrl.dev/rl-algorithms/ppo/#usage_8
poetry install -E "isaacgym"
xvfb-run -a poetry run python -m cleanrl_utils.benchmark \
    --env-ids Cartpole Ant Humanoid BallBalance Anymal  \
    --command "poetry run python cleanrl/ppo_continuous_action_isaacgym/ppo_continuous_action_isaacgym.py --track --capture-video" \
    --num-seeds 3 \
    --workers 1
xvfb-run -a poetry run python -m cleanrl_utils.benchmark \
    --env-ids AllegroHand ShadowHand \
    --command "poetry run python cleanrl/ppo_continuous_action_isaacgym/ppo_continuous_action_isaacgym.py --track --capture-video --num-envs 8192 --num-steps 8 --update-epochs 5 --num-minibatches 4 --reward-scaler 0.01 --total-timesteps 600000000 --record-video-step-frequency 3660" \
    --num-seeds 3 \
    --workers 1
