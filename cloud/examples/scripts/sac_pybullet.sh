
for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python sac_continuous_action.py \
    --env-id MinitaurBulletEnv-v0 \
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
    --env-id MinitaurBulletDuckEnv-v0 \
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
    --env-id InvertedPendulumBulletEnv-v0 \
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
    --env-id InvertedDoublePendulumBulletEnv-v0 \
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
    --env-id Walker2DBulletEnv-v0 \
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
    --env-id HalfCheetahBulletEnv-v0 \
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
    --env-id AntBulletEnv-v0 \
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
    --env-id HopperBulletEnv-v0 \
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
    --env-id HumanoidBulletEnv-v0 \
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
    --env-id BipedalWalker-v3 \
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
    --env-id LunarLanderContinuous-v2 \
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
    --env-id Pendulum-v0 \
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
    --env-id MountainCarContinuous-v0 \
    --total-timesteps 2000000 \
    --wandb-project-name cleanrl.benchmark \
    --track \
    --wandb-entity cleanrl --cuda True \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done
