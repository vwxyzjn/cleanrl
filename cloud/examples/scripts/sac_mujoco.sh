
for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python sac_continuous_action.py \
    --env-id Reacher-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name cleanrl.benchmark \
    --track \
    --wandb-entity cleanrl --cuda True \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python sac_continuous_action.py \
    --env-id Pusher-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name cleanrl.benchmark \
    --track \
    --wandb-entity cleanrl --cuda True \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python sac_continuous_action.py \
    --env-id Thrower-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name cleanrl.benchmark \
    --track \
    --wandb-entity cleanrl --cuda True \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python sac_continuous_action.py \
    --env-id Striker-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name cleanrl.benchmark \
    --track \
    --wandb-entity cleanrl --cuda True \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python sac_continuous_action.py \
    --env-id InvertedPendulum-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name cleanrl.benchmark \
    --track \
    --wandb-entity cleanrl --cuda True \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python sac_continuous_action.py \
    --env-id HalfCheetah-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name cleanrl.benchmark \
    --track \
    --wandb-entity cleanrl --cuda True \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python sac_continuous_action.py \
    --env-id Hopper-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name cleanrl.benchmark \
    --track \
    --wandb-entity cleanrl --cuda True \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python sac_continuous_action.py \
    --env-id Swimmer-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name cleanrl.benchmark \
    --track \
    --wandb-entity cleanrl --cuda True \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python sac_continuous_action.py \
    --env-id Walker2d-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name cleanrl.benchmark \
    --track \
    --wandb-entity cleanrl --cuda True \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python sac_continuous_action.py \
    --env-id Ant-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name cleanrl.benchmark \
    --track \
    --wandb-entity cleanrl --cuda True \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python sac_continuous_action.py \
    --env-id Humanoid-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name cleanrl.benchmark \
    --track \
    --wandb-entity cleanrl --cuda True \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done
