NUM_CORES=2
export MKL_NUM_THREADS=$NUM_CORES OMP_NUM_THREADS=$NUM_CORES

for seed in {1..2}
do
    (sleep 0.3 && nohup python sac_pranz24.py \
    --seed $seed \
    --gym-id BipedalWalker-v2 \
    --total-timesteps 2000000 \
    --episode-length 1600 \
    --wandb-project-name cleanrl.benchmark \
    --wandb-entity cleanrl \
    --prod-mode
    ) >& /dev/null &
done
for seed in {1..2}
do
    (sleep 0.3 && nohup python sac_pranz24.py \
    --seed $seed \
    --gym-id Pendulum-v0 \
    --total-timesteps 2000000 \
    --episode-length 200 \
    --wandb-project-name cleanrl.benchmark \
    --wandb-entity cleanrl \
    --prod-mode
    ) >& /dev/null &
done
for seed in {1..2}
do
    (sleep 0.3 && nohup python sac_pranz24.py \
    --seed $seed \
    --gym-id HopperBulletEnv-v0 \
    --total-timesteps 2000000 \
    --episode-length 1000 \
    --wandb-project-name cleanrl.benchmark \
    --wandb-entity cleanrl \
    --prod-mode
    ) >& /dev/null &
done
for seed in {1..2}
do
    (sleep 0.3 && nohup python sac_pranz24.py \
    --seed $seed \
    --gym-id InvertedPendulumBulletEnv-v0 \
    --total-timesteps 2000000 \
    --episode-length 1000 \
    --wandb-project-name cleanrl.benchmark \
    --wandb-entity cleanrl \
    --prod-mode
    ) >& /dev/null &
done
for seed in {1..2}
do
    (sleep 0.3 && nohup python sac_pranz24.py \
    --seed $seed \
    --gym-id Walker2DBulletEnv-v0 \
    --total-timesteps 2000000 \
    --episode-length 1000 \
    --wandb-project-name cleanrl.benchmark \
    --wandb-entity cleanrl \
    --prod-mode
    ) >& /dev/null &
done
for seed in {1..2}
do
    (sleep 0.3 && nohup python sac_pranz24.py \
    --seed $seed \
    --gym-id HumanoidBulletEnv-v0 \
    --total-timesteps 2000000 \
    --episode-length 1000 \
    --wandb-project-name cleanrl.benchmark \
    --wandb-entity cleanrl \
    --prod-mode
    ) >& /dev/null &
done
for seed in {1..2}
do
    (sleep 0.3 && nohup python sac_pranz24.py \
    --seed $seed \
    --gym-id HalfCheetahBulletEnv-v0 \
    --total-timesteps 2000000 \
    --episode-length 1000 \
    --wandb-project-name cleanrl.benchmark \
    --wandb-entity cleanrl \
    --prod-mode
    ) >& /dev/null &
done
for seed in {1..2}
do
    (sleep 0.3 && nohup python sac_pranz24.py \
    --seed $seed \
    --gym-id ReacherBulletEnv-v0 \
    --total-timesteps 2000000 \
    --episode-length 150 \
    --wandb-project-name cleanrl.benchmark \
    --wandb-entity cleanrl \
    --prod-mode
    ) >& /dev/null &
done