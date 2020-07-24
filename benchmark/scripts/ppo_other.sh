
for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo.py \
    --gym-id CartPole-v1 \
    --total-timesteps 100000 \
    --wandb-project-name cleanrl.benchmark \
    --wandb-entity cleanrl \
    --prod-mode \
    --cuda False \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo.py \
    --gym-id Acrobot-v1 \
    --total-timesteps 100000 \
    --wandb-project-name cleanrl.benchmark \
    --wandb-entity cleanrl \
    --prod-mode \
    --cuda False \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo.py \
    --gym-id MountainCar-v0 \
    --total-timesteps 100000 \
    --wandb-project-name cleanrl.benchmark \
    --wandb-entity cleanrl \
    --prod-mode \
    --cuda False \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo.py \
    --gym-id LunarLander-v2 \
    --total-timesteps 100000 \
    --wandb-project-name cleanrl.benchmark \
    --wandb-entity cleanrl \
    --prod-mode \
    --cuda False \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done
