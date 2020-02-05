for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo_continuous_gae.py \
    --gym-id HopperBulletEnv-v0 \
    --total-timesteps 2000000 \
    --episode-length 1000 \
    --wandb-project-name cleanrl \
    --wandb-entity cleanrl \
    --prod-mode True \
    --capture-video True \
    --seed $seed
    ) >& /dev/null &
done
for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo_reward_norm.py \
    --gym-id HopperBulletEnv-v0 \
    --total-timesteps 2000000 \
    --episode-length 1000 \
    --wandb-project-name cleanrl \
    --wandb-entity cleanrl \
    --prod-mode True \
    --capture-video True \
    --seed $seed
    ) >& /dev/null &
done
for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo_adv_norm.py \
    --gym-id HopperBulletEnv-v0 \
    --total-timesteps 2000000 \
    --episode-length 1000 \
    --wandb-project-name cleanrl \
    --wandb-entity cleanrl \
    --prod-mode True \
    --capture-video True \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo_return_norm_reset.py \
    --gym-id HopperBulletEnv-v0 \
    --total-timesteps 2000000 \
    --return-filter-reset False \
    --running-state-reset False \
    --wandb-project-name cleanrl \
    --wandb-entity cleanrl \
    --prod-mode True \
    --capture-video True \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo_return_norm_reset.py \
    --gym-id HopperBulletEnv-v0 \
    --total-timesteps 2000000 \
    --return-filter-reset True \
    --running-state-reset False \
    --wandb-project-name cleanrl \
    --wandb-entity cleanrl \
    --prod-mode True \
    --capture-video True \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo_return_norm_reset.py \
    --gym-id HopperBulletEnv-v0 \
    --total-timesteps 2000000 \
    --return-filter-reset False \
    --running-state-reset True \
    --wandb-project-name cleanrl \
    --wandb-entity cleanrl \
    --prod-mode True \
    --capture-video True \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo_return_norm_reset.py \
    --gym-id HopperBulletEnv-v0 \
    --total-timesteps 2000000 \
    --return-filter-reset True \
    --running-state-reset True \
    --wandb-project-name cleanrl \
    --wandb-entity cleanrl \
    --prod-mode True \
    --capture-video True \
    --seed $seed
    ) >& /dev/null &
done