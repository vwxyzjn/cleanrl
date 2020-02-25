# This script studies if we shoulds reset the reward filter and observation filter

NUM_CORES=2
export MKL_NUM_THREADS=$NUM_CORES OMP_NUM_THREADS=$NUM_CORES

# norm returns, reset
for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --gym-id HopperBulletEnv-v0 \
    --total-timesteps 2000000 \
    --kl \
    --norm-returns \
    --wandb-project-name cleanrl.ppo.kl.norm.reset \
    --wandb-entity cleanrl \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done
for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --gym-id HumanoidBulletEnv-v0 \
    --total-timesteps 2000000 \
    --kl \
    --norm-returns \
    --wandb-project-name cleanrl.ppo.kl.norm.reset \
    --wandb-entity cleanrl \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --gym-id HopperBulletEnv-v0 \
    --total-timesteps 2000000 \
    --kl \
    --norm-returns \
    --no-reward-reset \
    --wandb-project-name cleanrl.ppo.kl.norm.reset \
    --wandb-entity cleanrl \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done
for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --gym-id HumanoidBulletEnv-v0 \
    --total-timesteps 2000000 \
    --kl \
    --norm-returns \
    --no-reward-reset \
    --wandb-project-name cleanrl.ppo.kl.norm.reset \
    --wandb-entity cleanrl \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

# norm rewards, reset
for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --gym-id HopperBulletEnv-v0 \
    --total-timesteps 2000000 \
    --kl \
    --norm-rewards \
    --wandb-project-name cleanrl.ppo.kl.norm.reset \
    --wandb-entity cleanrl \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done
for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --gym-id HumanoidBulletEnv-v0 \
    --total-timesteps 2000000 \
    --kl \
    --norm-rewards \
    --wandb-project-name cleanrl.ppo.kl.norm.reset \
    --wandb-entity cleanrl \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done
for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --gym-id HopperBulletEnv-v0 \
    --total-timesteps 2000000 \
    --kl \
    --norm-rewards \
    --no-reward-reset \
    --wandb-project-name cleanrl.ppo.kl.norm.reset \
    --wandb-entity cleanrl \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done
for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --gym-id HumanoidBulletEnv-v0 \
    --total-timesteps 2000000 \
    --kl \
    --norm-rewards \
    --no-reward-reset \
    --wandb-project-name cleanrl.ppo.kl.norm.reset \
    --wandb-entity cleanrl \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done


# norm obs, reset
for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --gym-id HopperBulletEnv-v0 \
    --total-timesteps 2000000 \
    --kl \
    --norm-obs \
    --wandb-project-name cleanrl.ppo.kl.norm.reset \
    --wandb-entity cleanrl \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done
for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --gym-id HumanoidBulletEnv-v0 \
    --total-timesteps 2000000 \
    --kl \
    --norm-obs \
    --wandb-project-name cleanrl.ppo.kl.norm.reset \
    --wandb-entity cleanrl \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done
for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --gym-id HopperBulletEnv-v0 \
    --total-timesteps 2000000 \
    --kl \
    --norm-obs \
    --no-obs-reset \
    --wandb-project-name cleanrl.ppo.kl.norm.reset \
    --wandb-entity cleanrl \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done
for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --gym-id HumanoidBulletEnv-v0 \
    --total-timesteps 2000000 \
    --kl \
    --norm-obs \
    --no-obs-reset \
    --wandb-project-name cleanrl.ppo.kl.norm.reset \
    --wandb-entity cleanrl \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

