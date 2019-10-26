# https://unix.stackexchange.com/questions/103920/parallelize-a-bash-for-loop
# evaluate the algorithm in parallel
NUM_CORES=2
export MKL_NUM_THREADS=$NUM_CORES OMP_NUM_THREADS=$NUM_CORES

##################################
# obs: Box, ac: Discrete 
##################################
# CartPole-v0
# Considered solved when reward > 195.0
for seed in {1..2}
do
    (sleep 0.3 && nohup python a2c.py \
    --seed $seed \
    --gym-id CartPole-v0 \
    --total-timesteps 30000 \
    --wandb-project-name cleanrl.benchmark \
    --prod-mode True
    ) >& /dev/null &
done
for seed in {1..2}
do
    (sleep 0.3 && nohup python ppo.py \
    --seed $seed \
    --gym-id CartPole-v0 \
    --total-timesteps 30000 \
    --wandb-project-name cleanrl.benchmark \
    --prod-mode True
    ) >& /dev/null &
done
for seed in {1..2}
do
    (sleep 0.3 && nohup python dqn.py \
    --seed $seed \
    --gym-id CartPole-v0 \
    --total-timesteps 30000 \
    --wandb-project-name cleanrl.benchmark \
    --prod-mode True
    ) >& /dev/null &
done
wait

# LunarLander-v2
# TODO: add docs on the goal rewards
for seed in {1..2}
do
    (sleep 0.3 && nohup python a2c.py \
    --seed $seed \
    --gym-id LunarLander-v2 \
    --total-timesteps 1000000 \
    --wandb-project-name cleanrl.benchmark \
    --prod-mode True
    ) >& /dev/null &
done
for seed in {1..2}
do
    (sleep 0.3 && nohup python ppo.py \
    --seed $seed \
    --gym-id LunarLander-v2 \
    --total-timesteps 1000000 \
    --wandb-project-name cleanrl.benchmark \
    --prod-mode True
    ) >& /dev/null &
done
for seed in {1..2}
do
    (sleep 0.3 && nohup python dqn.py \
    --seed $seed \
    --gym-id LunarLander-v2 \
    --total-timesteps 1000000 \
    --wandb-project-name cleanrl.benchmark \
    --prod-mode True
    ) >& /dev/null &
done
wait

##################################
# obs: Box, ac: Box 
##################################
# BipedalWalker-v2
# TODO: add docs on the goal rewards
for seed in {1..2}
do
    (sleep 0.3 && nohup python a2c_continuous_action.py \
    --seed $seed \
    --gym-id BipedalWalker-v2 \
    --total-timesteps 2000000 \
    --episode-length 1600 \
    --wandb-project-name cleanrl.benchmark \
    --prod-mode True
    ) >& /dev/null &
done
wait

# Pendulum-v0
# TODO: add docs on the goal rewards
for seed in {1..2}
do
    (sleep 0.3 && nohup python a2c_continuous_action.py \
    --seed $seed \
    --gym-id Pendulum-v0 \
    --total-timesteps 2000000 \
    --episode-length 200 \
    --wandb-project-name cleanrl.benchmark \
    --prod-mode True
    ) >& /dev/null &
done
wait

# HopperBulletEnv-v0
# TODO: add docs on the goal rewards
for seed in {1..2}
do
    (sleep 0.3 && nohup python a2c_continuous_action.py \
    --seed $seed \
    --gym-id HopperBulletEnv-v0 \
    --total-timesteps 2000000 \
    --episode-length 1000 \
    --wandb-project-name cleanrl.benchmark \
    --prod-mode True
    ) >& /dev/null &
done
wait

# InvertedPendulumBulletEnv-v0
# TODO: add docs on the goal rewards
for seed in {1..2}
do
    (sleep 0.3 && nohup python a2c_continuous_action.py \
    --seed $seed \
    --gym-id InvertedPendulumBulletEnv-v0 \
    --total-timesteps 2000000 \
    --episode-length 1000 \
    --wandb-project-name cleanrl.benchmark \
    --prod-mode True
    ) >& /dev/null &
done
wait

# Walker2DBulletEnv-v0
# TODO: add docs on the goal rewards
for seed in {1..2}
do
    (sleep 0.3 && nohup python a2c_continuous_action.py \
    --seed $seed \
    --gym-id Walker2DBulletEnv-v0 \
    --total-timesteps 2000000 \
    --episode-length 1000 \
    --wandb-project-name cleanrl.benchmark \
    --prod-mode True
    ) >& /dev/null &
done
wait

# HumanoidBulletEnv-v0
# TODO: add docs on the goal rewards
for seed in {1..2}
do
    (sleep 0.3 && nohup python a2c_continuous_action.py \
    --seed $seed \
    --gym-id HumanoidBulletEnv-v0 \
    --total-timesteps 2000000 \
    --episode-length 1000 \
    --wandb-project-name cleanrl.benchmark \
    --prod-mode True
    ) >& /dev/null &
done
wait

# HalfCheetahBulletEnv-v0
# TODO: add docs on the goal rewards
for seed in {1..2}
do
    (sleep 0.3 && nohup python a2c_continuous_action.py \
    --seed $seed \
    --gym-id HalfCheetahBulletEnv-v0 \
    --total-timesteps 2000000 \
    --episode-length 1000 \
    --wandb-project-name cleanrl.benchmark \
    --prod-mode True
    ) >& /dev/null &
done
wait

# ReacherBulletEnv-v0
# TODO: add docs on the goal rewards
for seed in {1..2}
do
    (sleep 0.3 && nohup python a2c_continuous_action.py \
    --seed $seed \
    --gym-id ReacherBulletEnv-v0 \
    --total-timesteps 2000000 \
    --episode-length 150 \
    --wandb-project-name cleanrl.benchmark \
    --prod-mode True
    ) >& /dev/null &
done
wait

##################################
# obs: Discrete, ac: Discrete
##################################
# Taxi-v2
# Considered solved when reward > 6.0
# Note: heavy exploratory problem
for seed in {1..2}
do
    (sleep 0.3 && nohup python a2c.py \
    --seed $seed \
    --gym-id Taxi-v2 \
    --total-timesteps 60000 \
    --wandb-project-name cleanrl.benchmark \
    --prod-mode True
    ) >& /dev/null &
done
for seed in {1..2}
do
    (sleep 0.3 && nohup python ppo.py \
    --seed $seed \
    --gym-id Taxi-v2 \
    --total-timesteps 60000 \
    --wandb-project-name cleanrl.benchmark \
    --prod-mode True
    ) >& /dev/null &
done
for seed in {1..2}
do
    (sleep 0.3 && nohup python dqn.py \
    --seed $seed \
    --gym-id Taxi-v2 \
    --total-timesteps 60000 \
    --wandb-project-name cleanrl.benchmark \
    --prod-mode True
    ) >& /dev/null &
done
wait
