NUM_CORES=2
export MKL_NUM_THREADS=$NUM_CORES OMP_NUM_THREADS=$NUM_CORES
for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo_continuous_action.py \
    --gym-id HopperBulletEnv-v0 \
    --total-timesteps 2000000 \
    --gae \
    --wandb-project-name cleanrl.ppo.gae \
    --wandb-entity cleanrl \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done
for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo_continuous_action.py \
    --gym-id Walker2DBulletEnv-v0 \
    --total-timesteps 2000000 \
    --gae \
    --wandb-project-name cleanrl.ppo.gae \
    --wandb-entity cleanrl \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done
for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo_continuous_action.py \
    --gym-id HumanoidBulletEnv-v0 \
    --total-timesteps 2000000 \
    --gae \
    --wandb-project-name cleanrl.ppo.gae \
    --wandb-entity cleanrl \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo_continuous_action.py \
    --gym-id HopperBulletEnv-v0 \
    --total-timesteps 2000000 \
    --episode-length 1000 \
    --wandb-project-name cleanrl.ppo.gae \
    --wandb-entity cleanrl \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done
for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo_continuous_action.py \
    --gym-id Walker2DBulletEnv-v0 \
    --total-timesteps 2000000 \
    --episode-length 1000 \
    --wandb-project-name cleanrl.ppo.gae \
    --wandb-entity cleanrl \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done
for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo_continuous_action.py \
    --gym-id HumanoidBulletEnv-v0 \
    --total-timesteps 2000000 \
    --episode-length 1000 \
    --wandb-project-name cleanrl.ppo.gae \
    --wandb-entity cleanrl \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done