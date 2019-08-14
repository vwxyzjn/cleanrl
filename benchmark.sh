# https://unix.stackexchange.com/questions/103920/parallelize-a-bash-for-loop
# evaluate the algorithm in parallel
NUM_CORES=2
export MKL_NUM_THREADS=$NUM_CORES OMP_NUM_THREADS=$NUM_CORES

# CartPole-v0
# Considered solved when reward > 195.0
for seed in {1..5}
do
    (sleep 0.3 && nohup python a2c.py \
    --seed $seed \
    --gym-id CartPole-v0 \
    --total-timesteps 50000 \
    --prod-mode True
    ) >& /dev/null &
done
for seed in {1..5}
do
    (sleep 0.3 && nohup python ppo.py \
    --seed $seed \
    --gym-id CartPole-v0 \
    --total-timesteps 50000 \
    --prod-mode True
    ) >& /dev/null &
done
for seed in {1..5}
do
    (sleep 0.3 && nohup python dqn.py \
    --seed $seed \
    --gym-id CartPole-v0 \
    --total-timesteps 50000 \
    --prod-mode True
    ) >& /dev/null &
done


# Taxi-v2
# Considered solved when reward > 6.0
# Note: heavy exploratory problem
for seed in {1..5}
do
    (sleep 0.3 && nohup python ppo.py \
    --seed $seed \
    --gym-id Taxi-v2 \
    --total-timesteps 50000 \
    --prod-mode True
    ) >& /dev/null &
done

# MountainCar-v0
# Considered solved when reward > -180.0
# Note: heavy exploratory problem
for seed in {1..5}
do
    (sleep 0.3 && nohup python ppo.py \
    --seed $seed \
    --gym-id MountainCar-v0 \
    --total-timesteps 50000 \
    --prod-mode True
    ) >& /dev/null &
done
