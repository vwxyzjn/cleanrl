
for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo_atari_visual.py \
    --env-id BeamRiderNoFrameskip-v4 \
    --total-timesteps 10000000 \
    --wandb-project-name cleanrl.benchmark \
    --track \
    --wandb-entity cleanrl --cuda True \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo_atari_visual.py \
    --env-id QbertNoFrameskip-v4 \
    --total-timesteps 10000000 \
    --wandb-project-name cleanrl.benchmark \
    --track \
    --wandb-entity cleanrl --cuda True \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo_atari_visual.py \
    --env-id SpaceInvadersNoFrameskip-v4 \
    --total-timesteps 10000000 \
    --wandb-project-name cleanrl.benchmark \
    --track \
    --wandb-entity cleanrl --cuda True \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo_atari_visual.py \
    --env-id PongNoFrameskip-v4 \
    --total-timesteps 10000000 \
    --wandb-project-name cleanrl.benchmark \
    --track \
    --wandb-entity cleanrl --cuda True \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo_atari_visual.py \
    --env-id BreakoutNoFrameskip-v4 \
    --total-timesteps 10000000 \
    --wandb-project-name cleanrl.benchmark \
    --track \
    --wandb-entity cleanrl --cuda True \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done
