python generate_exp.py --exp-script scripts/ppo_mujoco.sh \
    --algo ppo_continuous_action.py \
    --total-timesteps 2000000 \
    --gym-ids Reacher-v2 Pusher-v2 Thrower-v2 Striker-v2 InvertedPendulum-v2 HalfCheetah-v2 Hopper-v2 Swimmer-v2 Walker2d-v2 Ant-v2 Humanoid-v2 \
    --wandb-project-name cleanrl.benchmark \
    --wandb-entity cleanrl \
    --cuda False

python generate_exp.py --exp-script scripts/ppo_pybullet.sh \
    --algo ppo_continuous_action.py \
    --total-timesteps 2000000 \
    --gym-ids MinitaurBulletEnv-v0 MinitaurBulletDuckEnv-v0 InvertedPendulumBulletEnv-v0 InvertedDoublePendulumBulletEnv-v0 Walker2DBulletEnv-v0 HalfCheetahBulletEnv-v0 AntBulletEnv-v0 HopperBulletEnv-v0 HumanoidBulletEnv-v0 BipedalWalker-v3 LunarLanderContinuous-v2 Pendulum-v0 MountainCarContinuous-v0 \
    --wandb-project-name cleanrl.benchmark \
    --wandb-entity cleanrl \
    --cuda False

python generate_exp.py --exp-script scripts/ppo_atari.sh \
    --algo ppo_atari_visual.py \
    --total-timesteps 10000000 \
    --gym-ids BeamRiderNoFrameskip-v4 QbertNoFrameskip-v4 SpaceInvadersNoFrameskip-v4 PongNoFrameskip-v4 BreakoutNoFrameskip-v4 \
    --wandb-project-name cleanrl.benchmark \
    --wandb-entity cleanrl \
    --cuda 

python generate_exp.py --exp-script scripts/ppo_other.sh \
    --algo ppo.py \
    --total-timesteps 2000000 \
    --gym-ids CartPole-v1 Acrobot-v1 MountainCar-v0 LunarLander-v2 \
    --wandb-project-name cleanrl.benchmark \
    --wandb-entity cleanrl \
    --cuda False

python generate_exp.py --exp-script scripts/td3_mujoco.sh \
    --algo td3_continuous_action.py \
    --total-timesteps 2000000 \
    --gym-ids Reacher-v2 Pusher-v2 Thrower-v2 Striker-v2 InvertedPendulum-v2 HalfCheetah-v2 Hopper-v2 Swimmer-v2 Walker2d-v2 Ant-v2 Humanoid-v2 \
    --wandb-project-name cleanrl.benchmark \
    --wandb-entity cleanrl \
    --cuda
    
python generate_exp.py --exp-script scripts/td3_pybullet.sh \
    --algo td3_continuous_action.py \
    --total-timesteps 2000000 \
    --gym-ids MinitaurBulletEnv-v0 MinitaurBulletDuckEnv-v0 InvertedPendulumBulletEnv-v0 InvertedDoublePendulumBulletEnv-v0 Walker2DBulletEnv-v0 HalfCheetahBulletEnv-v0 AntBulletEnv-v0 HopperBulletEnv-v0 HumanoidBulletEnv-v0 BipedalWalker-v3 LunarLanderContinuous-v2 Pendulum-v0 MountainCarContinuous-v0 \
    --wandb-project-name cleanrl.benchmark \
    --wandb-entity cleanrl \
    --cuda

python generate_exp.py --exp-script scripts/ddpg_mujoco.sh \
    --algo ddpg_continuous_action.py \
    --total-timesteps 2000000 \
    --gym-ids Reacher-v2 Pusher-v2 Thrower-v2 Striker-v2 InvertedPendulum-v2 HalfCheetah-v2 Hopper-v2 Swimmer-v2 Walker2d-v2 Ant-v2 Humanoid-v2 \
    --wandb-project-name cleanrl.benchmark \
    --wandb-entity cleanrl \
    --cuda
    
python generate_exp.py --exp-script scripts/ddpg_pybullet.sh \
    --algo ddpg_continuous_action.py \
    --total-timesteps 2000000 \
    --gym-ids MinitaurBulletEnv-v0 MinitaurBulletDuckEnv-v0 InvertedPendulumBulletEnv-v0 InvertedDoublePendulumBulletEnv-v0 Walker2DBulletEnv-v0 HalfCheetahBulletEnv-v0 AntBulletEnv-v0 HopperBulletEnv-v0 HumanoidBulletEnv-v0 BipedalWalker-v3 LunarLanderContinuous-v2 Pendulum-v0 MountainCarContinuous-v0 \
    --wandb-project-name cleanrl.benchmark \
    --wandb-entity cleanrl \
    --cuda

python generate_exp.py --exp-script scripts/sac_mujoco.sh \
    --algo sac_continuous_action.py \
    --total-timesteps 2000000 \
    --gym-ids Reacher-v2 Pusher-v2 Thrower-v2 Striker-v2 InvertedPendulum-v2 HalfCheetah-v2 Hopper-v2 Swimmer-v2 Walker2d-v2 Ant-v2 Humanoid-v2 \
    --wandb-project-name cleanrl.benchmark \
    --wandb-entity cleanrl \
    --cuda
    
python generate_exp.py --exp-script scripts/sac_pybullet.sh \
    --algo sac_continuous_action.py \
    --total-timesteps 2000000 \
    --gym-ids MinitaurBulletEnv-v0 MinitaurBulletDuckEnv-v0 InvertedPendulumBulletEnv-v0 InvertedDoublePendulumBulletEnv-v0 Walker2DBulletEnv-v0 HalfCheetahBulletEnv-v0 AntBulletEnv-v0 HopperBulletEnv-v0 HumanoidBulletEnv-v0 BipedalWalker-v3 LunarLanderContinuous-v2 Pendulum-v0 MountainCarContinuous-v0 \
    --wandb-project-name cleanrl.benchmark \
    --wandb-entity cleanrl \
    --cuda

python generate_exp.py --exp-script scripts/dqn_atari.sh \
    --algo dqn_atari_visual.py \
    --total-timesteps 10000000 \
    --gym-ids BeamRiderNoFrameskip-v4 QbertNoFrameskip-v4 SpaceInvadersNoFrameskip-v4 PongNoFrameskip-v4 BreakoutNoFrameskip-v4 \
    --wandb-project-name cleanrl.benchmark \
    --wandb-entity cleanrl \
    --cuda 

python generate_exp.py --exp-script scripts/dqn_other.sh \
    --algo dqn.py \
    --total-timesteps 2000000 \
    --gym-ids CartPole-v1 Acrobot-v1 MountainCar-v0 LunarLander-v2 \
    --wandb-project-name cleanrl.benchmark \
    --wandb-entity cleanrl \
    --cuda

python generate_exp.py --exp-script scripts/c51_atari.sh \
    --algo c51_atari_visual.py \
    --total-timesteps 10000000 \
    --gym-ids BeamRiderNoFrameskip-v4 QbertNoFrameskip-v4 SpaceInvadersNoFrameskip-v4 PongNoFrameskip-v4 BreakoutNoFrameskip-v4 \
    --wandb-project-name cleanrl.benchmark \
    --wandb-entity cleanrl \
    --cuda 

python generate_exp.py --exp-script scripts/c51_other.sh \
    --algo c51.py \
    --total-timesteps 2000000 \
    --gym-ids CartPole-v1 Acrobot-v1 MountainCar-v0 LunarLander-v2 \
    --wandb-project-name cleanrl.benchmark \
    --wandb-entity cleanrl \
    --cuda