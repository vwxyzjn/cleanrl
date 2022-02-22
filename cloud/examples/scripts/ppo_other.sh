
for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo.py \
    --env-id CartPole-v1 \
    --total-timesteps 2000000 \
    --wandb-project-name cleanrl.benchmark \
    --track \
    --wandb-entity cleanrl --cuda False \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo.py \
    --env-id Acrobot-v1 \
    --total-timesteps 2000000 \
    --wandb-project-name cleanrl.benchmark \
    --track \
    --wandb-entity cleanrl --cuda False \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo.py \
    --env-id MountainCar-v0 \
    --total-timesteps 2000000 \
    --wandb-project-name cleanrl.benchmark \
    --track \
    --wandb-entity cleanrl --cuda False \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo.py \
    --env-id LunarLander-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name cleanrl.benchmark \
    --track \
    --wandb-entity cleanrl --cuda False \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done
