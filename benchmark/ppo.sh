# export WANDB_ENTITY=openrlbenchmark

poetry install
OMP_NUM_THREADS=1 xvfb-run -a python -m cleanrl_utils.benchmark \
    --env-ids CartPole-v1 Acrobot-v1 MountainCar-v0 \
    --command "poetry run python cleanrl/ppo.py --cuda False --track --capture-video" \
    --num-seeds 3 \
    --workers 9

poetry install -E atari
OMP_NUM_THREADS=1 xvfb-run -a python -m cleanrl_utils.benchmark \
    --env-ids PongNoFrameskip-v4 BeamRiderNoFrameskip-v4 BreakoutNoFrameskip-v4 \
    --command "poetry run python cleanrl/ppo_atari.py --track --capture-video" \
    --num-seeds 3 \
    --workers 3

poetry install -E atari
OMP_NUM_THREADS=1 xvfb-run -a python -m cleanrl_utils.benchmark \
    --env-ids PongNoFrameskip-v4 BeamRiderNoFrameskip-v4 BreakoutNoFrameskip-v4 \
    --command "poetry run python cleanrl/ppo_atari_lstm.py --track --capture-video" \
    --num-seeds 3 \
    --workers 3

poetry install -E envpool
xvfb-run -a python -m cleanrl_utils.benchmark \
    --env-ids Pong-v5 BeamRider-v5 Breakout-v5 \
    --command "poetry run python cleanrl/ppo_atari_envpool.py --track --capture-video" \
    --num-seeds 3 \
    --workers 1

poetry install -E "mujoco pybullet"
python -c "import mujoco_py"
OMP_NUM_THREADS=1 xvfb-run -a python -m cleanrl_utils.benchmark \
    --env-ids HalfCheetah-v2 Walker2d-v2 Hopper-v2 \
    --command "poetry run python cleanrl/ppo_continuous_action.py --cuda False --track --capture-video" \
    --num-seeds 3 \
    --workers 9

poetry install -E procgen
xvfb-run -a python -m cleanrl_utils.benchmark \
    --env-ids starpilot bossfight bigfish \
    --command "poetry run python cleanrl/ppo_procgen.py --track --capture-video" \
    --num-seeds 3 \
    --workers 1