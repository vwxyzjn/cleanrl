# https://unix.stackexchange.com/questions/103920/parallelize-a-bash-for-loop
# evaluate the algorithm in parallel
# apt-get install python-opengl
NUM_CORES=2
export MKL_NUM_THREADS=$NUM_CORES OMP_NUM_THREADS=$NUM_CORES

################################################
#
#
#    Advantage Actor Critic (A2C)
#
#
################################################
##################################
# obs: Box, ac: Discrete 
##################################
# CartPole-v0
# Considered solved when reward > 195.0
for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python a2c.py \
    --gym-id CartPole-v0 \
    --total-timesteps 30000 \
    --wandb-project-name cleanrl.benchmark \
    --wandb-entity cleanrl \
    --prod-mode True \
    --capture-video True \
    --seed $seed
    ) >& /dev/null &
done
# LunarLander-v2
# TODO: add docs on the goal rewards
for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python a2c.py \
    --gym-id LunarLander-v2 \
    --total-timesteps 1000000 \
    --wandb-project-name cleanrl.benchmark \
    --wandb-entity cleanrl \
    --prod-mode True \
    --capture-video True \
    --seed $seed
    ) >& /dev/null &
done
for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python a2c.py \
    --gym-id MountainCar-v0 \
    --total-timesteps 200000 \
    --wandb-project-name cleanrl.benchmark \
    --wandb-entity cleanrl \
    --prod-mode True \
    --capture-video True \
    --seed $seed
    ) >& /dev/null &
done
##################################
# obs: Box, ac: Box 
##################################
# MountainCarContinuous-v0
# TODO: add docs on the goal rewards
for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python a2c_continuous_action.py \
    --gym-id MountainCarContinuous-v0 \
    --total-timesteps 200000 \
    --wandb-project-name cleanrl.benchmark \
    --wandb-entity cleanrl \
    --prod-mode True \
    --capture-video True \
    --seed $seed
    ) >& /dev/null &
done
# BipedalWalker-v2
# TODO: add docs on the goal rewards
for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python a2c_continuous_action.py \
    --gym-id BipedalWalker-v2 \
    --total-timesteps 2000000 \
    --episode-length 1600 \
    --wandb-project-name cleanrl.benchmark \
    --wandb-entity cleanrl \
    --prod-mode True \
    --capture-video True \
    --seed $seed
    ) >& /dev/null &
done
# Pendulum-v0
# TODO: add docs on the goal rewards
for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python a2c_continuous_action.py \
    --gym-id Pendulum-v0 \
    --total-timesteps 2000000 \
    --episode-length 200 \
    --wandb-project-name cleanrl.benchmark \
    --wandb-entity cleanrl \
    --prod-mode True \
    --capture-video True \
    --seed $seed
    ) >& /dev/null &
done
wait
# HopperBulletEnv-v0
# TODO: add docs on the goal rewards
for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python a2c_continuous_action.py \
    --gym-id HopperBulletEnv-v0 \
    --total-timesteps 2000000 \
    --episode-length 1000 \
    --wandb-project-name cleanrl.benchmark \
    --wandb-entity cleanrl \
    --prod-mode True \
    --capture-video True \
    --seed $seed
    ) >& /dev/null &
done
# InvertedPendulumBulletEnv-v0
# TODO: add docs on the goal rewards
for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python a2c_continuous_action.py \
    --gym-id InvertedPendulumBulletEnv-v0 \
    --total-timesteps 2000000 \
    --episode-length 1000 \
    --wandb-project-name cleanrl.benchmark \
    --wandb-entity cleanrl \
    --prod-mode True \
    --capture-video True \
    --seed $seed
    ) >& /dev/null &
done
# Walker2DBulletEnv-v0
# TODO: add docs on the goal rewards
for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python a2c_continuous_action.py \
    --gym-id Walker2DBulletEnv-v0 \
    --total-timesteps 2000000 \
    --episode-length 1000 \
    --wandb-project-name cleanrl.benchmark \
    --wandb-entity cleanrl \
    --prod-mode True \
    --capture-video True \
    --seed $seed
    ) >& /dev/null &
done
# HumanoidBulletEnv-v0
# TODO: add docs on the goal rewards
for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python a2c_continuous_action.py \
    --gym-id HumanoidBulletEnv-v0 \
    --total-timesteps 2000000 \
    --episode-length 1000 \
    --wandb-project-name cleanrl.benchmark \
    --wandb-entity cleanrl \
    --prod-mode True \
    --capture-video True \
    --seed $seed
    ) >& /dev/null &
done
wait
# HalfCheetahBulletEnv-v0
# TODO: add docs on the goal rewards
for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python a2c_continuous_action.py \
    --gym-id HalfCheetahBulletEnv-v0 \
    --total-timesteps 2000000 \
    --episode-length 1000 \
    --wandb-project-name cleanrl.benchmark \
    --wandb-entity cleanrl \
    --prod-mode True \
    --capture-video True \
    --seed $seed
    ) >& /dev/null &
done
# ReacherBulletEnv-v0
# TODO: add docs on the goal rewards
for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python a2c_continuous_action.py \
    --gym-id ReacherBulletEnv-v0 \
    --total-timesteps 2000000 \
    --episode-length 150 \
    --wandb-project-name cleanrl.benchmark \
    --wandb-entity cleanrl \
    --prod-mode True \
    --capture-video True \
    --seed $seed
    ) >& /dev/null &
done
##################################
# obs: Discrete, ac: Discrete
##################################
# Taxi-v3
# Considered solved when reward > 6.0
# Note: heavy exploratory problem
for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python a2c.py \
    --gym-id Taxi-v3 \
    --total-timesteps 60000 \
    --wandb-project-name cleanrl.benchmark \
    --wandb-entity cleanrl \
    --prod-mode True \
    --capture-video True \
    --seed $seed
    ) >& /dev/null &
done
wait



################################################
#
#
#    Proximal Policy Gradient (PPO)
#
#
################################################
for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo.py \
    --gym-id CartPole-v0 \
    --total-timesteps 30000 \
    --wandb-project-name cleanrl.benchmark \
    --wandb-entity cleanrl \
    --prod-mode True \
    --capture-video True \
    --seed $seed
    ) >& /dev/null &
done
for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo.py \
    --gym-id LunarLander-v2 \
    --total-timesteps 1000000 \
    --wandb-project-name cleanrl.benchmark \
    --wandb-entity cleanrl \
    --prod-mode True \
    --capture-video True \
    --seed $seed
    ) >& /dev/null &
done
for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo.py \
    --gym-id MountainCar-v0 \
    --total-timesteps 200000 \
    --wandb-project-name cleanrl.benchmark \
    --wandb-entity cleanrl \
    --prod-mode True \
    --capture-video True \
    --seed $seed
    ) >& /dev/null &
done
for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo_continuous_action.py \
    --gym-id MountainCarContinuous-v0 \
    --total-timesteps 200000 \
    --wandb-project-name cleanrl.benchmark \
    --wandb-entity cleanrl \
    --prod-mode True \
    --capture-video True \
    --seed $seed
    ) >& /dev/null &
done
for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo_continuous_action.py \
    --gym-id BipedalWalker-v2 \
    --total-timesteps 2000000 \
    --episode-length 1600 \
    --wandb-project-name cleanrl.benchmark \
    --wandb-entity cleanrl \
    --prod-mode True \
    --capture-video True \
    --seed $seed
    ) >& /dev/null &
done
for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo_continuous_action.py \
    --gym-id Pendulum-v0 \
    --total-timesteps 2000000 \
    --episode-length 200 \
    --wandb-project-name cleanrl.benchmark \
    --wandb-entity cleanrl \
    --prod-mode True \
    --capture-video True \
    --seed $seed
    ) >& /dev/null &
done
wait
for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo_continuous_action.py \
    --gym-id HopperBulletEnv-v0 \
    --total-timesteps 2000000 \
    --episode-length 1000 \
    --wandb-project-name cleanrl.benchmark \
    --wandb-entity cleanrl \
    --prod-mode True \
    --capture-video True \
    --seed $seed
    ) >& /dev/null &
done
for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo_continuous_action.py \
    --gym-id InvertedPendulumBulletEnv-v0 \
    --total-timesteps 2000000 \
    --episode-length 1000 \
    --wandb-project-name cleanrl.benchmark \
    --wandb-entity cleanrl \
    --prod-mode True \
    --capture-video True \
    --seed $seed
    ) >& /dev/null &
done
for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo_continuous_action.py \
    --gym-id Walker2DBulletEnv-v0 \
    --total-timesteps 2000000 \
    --episode-length 1000 \
    --wandb-project-name cleanrl.benchmark \
    --wandb-entity cleanrl \
    --prod-mode True \
    --capture-video True \
    --seed $seed
    ) >& /dev/null &
done
for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo_continuous_action.py \
    --gym-id HumanoidBulletEnv-v0 \
    --total-timesteps 2000000 \
    --episode-length 1000 \
    --wandb-project-name cleanrl.benchmark \
    --wandb-entity cleanrl \
    --prod-mode True \
    --capture-video True \
    --seed $seed
    ) >& /dev/null &
done
wait
for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo_continuous_action.py \
    --gym-id HalfCheetahBulletEnv-v0 \
    --total-timesteps 2000000 \
    --episode-length 1000 \
    --wandb-project-name cleanrl.benchmark \
    --wandb-entity cleanrl \
    --prod-mode True \
    --capture-video True \
    --seed $seed
    ) >& /dev/null &
done
for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo_continuous_action.py \
    --gym-id ReacherBulletEnv-v0 \
    --total-timesteps 2000000 \
    --episode-length 150 \
    --wandb-project-name cleanrl.benchmark \
    --wandb-entity cleanrl \
    --prod-mode True \
    --capture-video True \
    --seed $seed
    ) >& /dev/null &
done
for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo_continuous_action.py \
    --gym-id Taxi-v3 \
    --total-timesteps 60000 \
    --wandb-project-name cleanrl.benchmark \
    --wandb-entity cleanrl \
    --prod-mode True \
    --capture-video True \
    --seed $seed
    ) >& /dev/null &
done
for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo.py \
    --gym-id Taxi-v3 \
    --total-timesteps 60000 \
    --wandb-project-name cleanrl.benchmark \
    --wandb-entity cleanrl \
    --prod-mode True \
    --capture-video True \
    --seed $seed
    ) >& /dev/null &
done
wait



################################################
#
#
#    Soft Actor Critic (SAC)
#
#
################################################
for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python sac.py \
    --gym-id CartPole-v0 \
    --total-timesteps 30000 \
    --wandb-project-name cleanrl.benchmark \
    --wandb-entity cleanrl \
    --prod-mode True \
    --capture-video True \
    --seed $seed
    ) >& /dev/null &
done
for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python sac.py \
    --gym-id LunarLander-v2 \
    --total-timesteps 1000000 \
    --wandb-project-name cleanrl.benchmark \
    --wandb-entity cleanrl \
    --prod-mode True \
    --capture-video True \
    --seed $seed
    ) >& /dev/null &
done
for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python sac.py \
    --gym-id MountainCar-v0 \
    --total-timesteps 200000 \
    --wandb-project-name cleanrl.benchmark \
    --wandb-entity cleanrl \
    --prod-mode True \
    --capture-video True \
    --seed $seed
    ) >& /dev/null &
done
for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python sac_continuous_action.py \
    --gym-id MountainCarContinuous-v0 \
    --total-timesteps 200000 \
    --wandb-project-name cleanrl.benchmark \
    --wandb-entity cleanrl \
    --prod-mode True \
    --capture-video True \
    --seed $seed
    ) >& /dev/null &
done
for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python sac_continuous_action.py \
    --gym-id BipedalWalker-v2 \
    --total-timesteps 2000000 \
    --episode-length 1600 \
    --wandb-project-name cleanrl.benchmark \
    --wandb-entity cleanrl \
    --prod-mode True \
    --capture-video True \
    --seed $seed
    ) >& /dev/null &
done
for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python sac_continuous_action.py \
    --gym-id Pendulum-v0 \
    --total-timesteps 2000000 \
    --episode-length 200 \
    --wandb-project-name cleanrl.benchmark \
    --wandb-entity cleanrl \
    --prod-mode True \
    --capture-video True \
    --seed $seed
    ) >& /dev/null &
done
wait
for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python sac_continuous_action.py \
    --gym-id HopperBulletEnv-v0 \
    --total-timesteps 2000000 \
    --episode-length 1000 \
    --wandb-project-name cleanrl.benchmark \
    --wandb-entity cleanrl \
    --prod-mode True \
    --capture-video True \
    --seed $seed
    ) >& /dev/null &
done
for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python sac_continuous_action.py \
    --gym-id InvertedPendulumBulletEnv-v0 \
    --total-timesteps 2000000 \
    --episode-length 1000 \
    --wandb-project-name cleanrl.benchmark \
    --wandb-entity cleanrl \
    --prod-mode True \
    --capture-video True \
    --seed $seed
    ) >& /dev/null &
done
wait
for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python sac_continuous_action.py \
    --gym-id Walker2DBulletEnv-v0 \
    --total-timesteps 2000000 \
    --episode-length 1000 \
    --wandb-project-name cleanrl.benchmark \
    --wandb-entity cleanrl \
    --prod-mode True \
    --capture-video True \
    --seed $seed
    ) >& /dev/null &
done
for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python sac_continuous_action.py \
    --gym-id HumanoidBulletEnv-v0 \
    --total-timesteps 2000000 \
    --episode-length 1000 \
    --wandb-project-name cleanrl.benchmark \
    --wandb-entity cleanrl \
    --prod-mode True \
    --capture-video True \
    --seed $seed
    ) >& /dev/null &
done
wait
for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python sac_continuous_action.py \
    --gym-id HalfCheetahBulletEnv-v0 \
    --total-timesteps 2000000 \
    --episode-length 1000 \
    --wandb-project-name cleanrl.benchmark \
    --wandb-entity cleanrl \
    --prod-mode True \
    --capture-video True \
    --seed $seed
    ) >& /dev/null &
done
for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python sac_continuous_action.py \
    --gym-id ReacherBulletEnv-v0 \
    --total-timesteps 2000000 \
    --episode-length 150 \
    --wandb-project-name cleanrl.benchmark \
    --wandb-entity cleanrl \
    --prod-mode True \
    --capture-video True \
    --seed $seed
    ) >& /dev/null &
done
wait
for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python sac.py \
    --gym-id Taxi-v3 \
    --total-timesteps 60000 \
    --wandb-project-name cleanrl.benchmark \
    --wandb-entity cleanrl \
    --prod-mode True \
    --capture-video True \
    --seed $seed
    ) >& /dev/null &
done
wait



################################################
#
#
#    Deep Q-Learning (DQN)
#
#
################################################
for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python dqn.py \
    --gym-id CartPole-v0 \
    --total-timesteps 30000 \
    --wandb-project-name cleanrl.benchmark \
    --wandb-entity cleanrl \
    --prod-mode True \
    --capture-video True \
    --seed $seed
    ) >& /dev/null &
done
for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python dqn.py \
    --gym-id LunarLander-v2 \
    --total-timesteps 1000000 \
    --wandb-project-name cleanrl.benchmark \
    --wandb-entity cleanrl \
    --prod-mode True \
    --capture-video True \
    --seed $seed
    ) >& /dev/null &
done
for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python dqn.py \
    --gym-id MountainCar-v0 \
    --total-timesteps 200000 \
    --wandb-project-name cleanrl.benchmark \
    --wandb-entity cleanrl \
    --prod-mode True \
    --capture-video True \
    --seed $seed
    ) >& /dev/null &
done
for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python dqn.py \
    --gym-id Taxi-v3 \
    --total-timesteps 60000 \
    --wandb-project-name cleanrl.benchmark \
    --wandb-entity cleanrl \
    --prod-mode True \
    --capture-video True \
    --seed $seed
    ) >& /dev/null &
done
wait