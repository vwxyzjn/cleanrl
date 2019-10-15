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
# Considered solved when reward > 200 points
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
# Considered solved when reward > 200 points
for seed in {1..2}
do
    (sleep 0.3 && nohup python a2c.py \
    --seed $seed \
    --gym-id BipedalWalker-v2 \
    --total-timesteps 4000000 \
    --wandb-project-name cleanrl.benchmark \
    --prod-mode True
    ) >& /dev/null &
done
for seed in {1..2}
do
    (sleep 0.3 && nohup python ppo.py \
    --seed $seed \
    --gym-id BipedalWalker-v2 \
    --total-timesteps 4000000 \
    --wandb-project-name cleanrl.benchmark \
    --prod-mode True
    ) >& /dev/null &
done
for seed in {1..2}
do
    (sleep 0.3 && nohup python dqn.py \
    --seed $seed \
    --gym-id BipedalWalker-v2 \
    --total-timesteps 4000000 \
    --wandb-project-name cleanrl.benchmark \
    --prod-mode True
    ) >& /dev/null &
done
wait

# Pendulum-v0
# Considered solved when reward > 200 points
for seed in {1..2}
do
    (sleep 0.3 && nohup python a2c.py \
    --seed $seed \
    --gym-id Pendulum-v0 \
    --total-timesteps 4000000 \
    --wandb-project-name cleanrl.benchmark \
    --prod-mode True
    ) >& /dev/null &
done
for seed in {1..2}
do
    (sleep 0.3 && nohup python ppo.py \
    --seed $seed \
    --gym-id Pendulum-v0 \
    --total-timesteps 4000000 \
    --wandb-project-name cleanrl.benchmark \
    --prod-mode True
    ) >& /dev/null &
done
for seed in {1..2}
do
    (sleep 0.3 && nohup python dqn.py \
    --seed $seed \
    --gym-id Pendulum-v0 \
    --total-timesteps 4000000 \
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

# # Breakout-v0
# # Pixels input
# # Considered solved when reward > 200 points
# for seed in {1..2}
# do
#     (sleep 0.3 && nohup python a2c.py \
#     --seed $seed \
#     --gym-id Taxi-v2 \
#     --total-timesteps 1000000 \
# --wandb-project-name cleanrl.benchmark \#     
# --prod-mode True
#     ) >& /dev/null &
# done
# for seed in {1..2}
# do
#     (sleep 0.3 && nohup python ppo.py \
#     --seed $seed \
#     --gym-id Taxi-v2 \
#     --total-timesteps 1000000 \
# --wandb-project-name cleanrl.benchmark \#     
# --prod-mode True
#     ) >& /dev/null &
# done
# for seed in {1..2}
# do
#     (sleep 0.3 && nohup python dqn.py \
#     --seed $seed \
#     --gym-id Taxi-v2 \
#     --total-timesteps 1000000 \
# --wandb-project-name cleanrl.benchmark \#     
# --prod-mode True
#     ) >& /dev/null &
# done
# wait
