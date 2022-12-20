# export WANDB_ENTITY=openrlbenchmark

poetry install
OMP_NUM_THREADS=1 xvfb-run -a poetry run python -m cleanrl_utils.benchmark \
    --env-ids CartPole-v1 Acrobot-v1 MountainCar-v0 \
    --command "poetry run python cleanrl/ppo.py --cuda False --track --capture-video" \
    --num-seeds 3 \
    --workers 9

poetry install --with atari
OMP_NUM_THREADS=1 xvfb-run -a poetry run python -m cleanrl_utils.benchmark \
    --env-ids PongNoFrameskip-v4 BeamRiderNoFrameskip-v4 BreakoutNoFrameskip-v4 \
    --command "poetry run python cleanrl/ppo_atari.py --track --capture-video" \
    --num-seeds 3 \
    --workers 3

poetry install --with atari
OMP_NUM_THREADS=1 xvfb-run -a poetry run python -m cleanrl_utils.benchmark \
    --env-ids PongNoFrameskip-v4 BeamRiderNoFrameskip-v4 BreakoutNoFrameskip-v4 \
    --command "poetry run python cleanrl/ppo_atari_lstm.py --track --capture-video" \
    --num-seeds 3 \
    --workers 3

poetry install --with envpool
xvfb-run -a poetry run python -m cleanrl_utils.benchmark \
    --env-ids Pong-v5 BeamRider-v5 Breakout-v5 \
    --command "poetry run python cleanrl/ppo_atari_envpool.py --track --capture-video" \
    --num-seeds 3 \
    --workers 1

poetry install --with mujoco_py,mujoco
poetry run python -c "import mujoco_py"
OMP_NUM_THREADS=1 xvfb-run -a poetry run python -m cleanrl_utils.benchmark \
    --env-ids HalfCheetah-v2 Walker2d-v2 Hopper-v2 InvertedPendulum-v2 Humanoid-v2 Pusher-v2 \
    --command "poetry run python cleanrl/ppo_continuous_action.py --cuda False --track --capture-video" \
    --num-seeds 3 \
    --workers 6

poetry install --with procgen
xvfb-run -a poetry run python -m cleanrl_utils.benchmark \
    --env-ids starpilot bossfight bigfish \
    --command "poetry run python cleanrl/ppo_procgen.py --track --capture-video" \
    --num-seeds 3 \
    --workers 1

poetry install --with atari
xvfb-run -a poetry run python -m cleanrl_utils.benchmark \
    --env-ids PongNoFrameskip-v4 BeamRiderNoFrameskip-v4 BreakoutNoFrameskip-v4 \
    --command "poetry run torchrun --standalone --nnodes=1 --nproc_per_node=2 cleanrl/ppo_atari_multigpu.py --track --capture-video" \
    --num-seeds 3 \
    --workers 1

poetry install --with pettingzoo,atari
poetry run AutoROM --accept-license
xvfb-run -a poetry run python -m cleanrl_utils.benchmark \
    --env-ids pong_v3 surround_v2 tennis_v3  \
    --command "poetry run python cleanrl/ppo_pettingzoo_ma_atari.py --track --capture-video" \
    --num-seeds 3 \
    --workers 3

# IMPORTANT: see specific Isaac Gym installation at
# https://docs.cleanrl.dev/rl-algorithms/ppo/#usage_8
poetry install --with isaacgym
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


poetry install --with envpool,jax
poetry run python -m cleanrl_utils.benchmark \
    --env-ids Alien-v5 Amidar-v5 Assault-v5 Asterix-v5 Asteroids-v5 Atlantis-v5 BankHeist-v5 BattleZone-v5 BeamRider-v5 Berzerk-v5 Bowling-v5 Boxing-v5 Breakout-v5 Centipede-v5 ChopperCommand-v5 CrazyClimber-v5 Defender-v5 DemonAttack-v5 \
    --command "poetry run python ppo_atari_envpool_xla_jax.py --track --wandb-project-name envpool-atari --wandb-entity openrlbenchmark" \
    --num-seeds 3 \
    --workers 1
poetry run python -m cleanrl_utils.benchmark \
    --env-ids DoubleDunk-v5 Enduro-v5 FishingDerby-v5 Freeway-v5 Frostbite-v5 Gopher-v5 Gravitar-v5 Hero-v5 IceHockey-v5 Jamesbond-v5 Kangaroo-v5 Krull-v5 KungFuMaster-v5 MontezumaRevenge-v5 MsPacman-v5 NameThisGame-v5 Phoenix-v5 Pitfall-v5 Pong-v5 \
    --command "poetry run python ppo_atari_envpool_xla_jax.py --track --wandb-project-name envpool-atari --wandb-entity openrlbenchmark" \
    --num-seeds 3 \
    --workers 1
poetry run python -m cleanrl_utils.benchmark \
    --env-ids PrivateEye-v5 Qbert-v5 Riverraid-v5 RoadRunner-v5 Robotank-v5 Seaquest-v5 Skiing-v5 Solaris-v5 SpaceInvaders-v5 StarGunner-v5 Surround-v5 Tennis-v5 TimePilot-v5 Tutankham-v5 UpNDown-v5 Venture-v5 VideoPinball-v5 WizardOfWor-v5 YarsRevenge-v5 Zaxxon-v5 \
    --command "poetry run python ppo_atari_envpool_xla_jax.py --track --wandb-project-name envpool-atari --wandb-entity openrlbenchmark" \
    --num-seeds 3 \
    --workers 1

# gymnasium support
poetry install --with mujoco
OMP_NUM_THREADS=1 xvfb-run -a python -m cleanrl_utils.benchmark \
    --env-ids HalfCheetah-v4 Walker2d-v4 Hopper-v4 InvertedPendulum-v4 Humanoid-v4 Pusher-v4 \
    --command "poetry run python cleanrl/gymnasium_support/ppo_continuous_action.py --cuda False --track" \
    --num-seeds 3 \
    --workers 1

poetry install --with dm_control,mujoco
OMP_NUM_THREADS=1 xvfb-run -a poetry run python -m cleanrl_utils.benchmark \
    --env-ids dm_control/acrobot-swingup-v0 dm_control/acrobot-swingup_sparse-v0 dm_control/ball_in_cup-catch-v0 dm_control/cartpole-balance-v0 dm_control/cartpole-balance_sparse-v0 dm_control/cartpole-swingup-v0 dm_control/cartpole-swingup_sparse-v0 dm_control/cartpole-two_poles-v0 dm_control/cartpole-three_poles-v0 dm_control/cheetah-run-v0 dm_control/dog-stand-v0 dm_control/dog-walk-v0 dm_control/dog-trot-v0 dm_control/dog-run-v0 dm_control/dog-fetch-v0 dm_control/finger-spin-v0 dm_control/finger-turn_easy-v0 dm_control/finger-turn_hard-v0 dm_control/fish-upright-v0 dm_control/fish-swim-v0 dm_control/hopper-stand-v0 dm_control/hopper-hop-v0 dm_control/humanoid-stand-v0 dm_control/humanoid-walk-v0 dm_control/humanoid-run-v0 dm_control/humanoid-run_pure_state-v0 dm_control/humanoid_CMU-stand-v0 dm_control/humanoid_CMU-run-v0 dm_control/lqr-lqr_2_1-v0 dm_control/lqr-lqr_6_2-v0 dm_control/manipulator-bring_ball-v0 dm_control/manipulator-bring_peg-v0 dm_control/manipulator-insert_ball-v0 dm_control/manipulator-insert_peg-v0 dm_control/pendulum-swingup-v0 dm_control/point_mass-easy-v0 dm_control/point_mass-hard-v0 dm_control/quadruped-walk-v0 dm_control/quadruped-run-v0 dm_control/quadruped-escape-v0 dm_control/quadruped-fetch-v0 dm_control/reacher-easy-v0 dm_control/reacher-hard-v0 dm_control/stacker-stack_2-v0 dm_control/stacker-stack_4-v0 dm_control/swimmer-swimmer6-v0 dm_control/swimmer-swimmer15-v0 dm_control/walker-stand-v0 dm_control/walker-walk-v0 dm_control/walker-run-v0 \
    --command "poetry run python cleanrl/gymnasium_support/ppo_continuous_action.py --cuda False --track" \
    --num-seeds 3 \
    --workers 9
 