for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --gym-id HumanoidFlagrunBulletEnv-v0 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo-kl \
    --wandb-entity cleanrl \
    --prod-mode \
    --norm-returns \
    --no-cuda \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --gym-id HumanoidFlagrunBulletEnv-v0 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo-kl \
    --wandb-entity cleanrl \
    --prod-mode \
    --norm-adv \
    --no-cuda \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --gym-id HumanoidFlagrunBulletEnv-v0 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo-kl \
    --wandb-entity cleanrl \
    --prod-mode \
    --norm-obs \
    --no-cuda \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --gym-id HumanoidFlagrunBulletEnv-v0 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo-kl \
    --wandb-entity cleanrl \
    --prod-mode \
    --clip-vloss \
    --no-cuda \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --gym-id HumanoidFlagrunBulletEnv-v0 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo-kl \
    --wandb-entity cleanrl \
    --prod-mode \
    --anneal-lr \
    --no-cuda \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --gym-id HumanoidFlagrunBulletEnv-v0 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo-kl \
    --wandb-entity cleanrl \
    --prod-mode \
    --gae \
    --no-cuda \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --gym-id HumanoidFlagrunBulletEnv-v0 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo-kl \
    --wandb-entity cleanrl \
    --prod-mode \
    --norm-returns --norm-adv --norm-obs --clip-vloss --anneal-lr --gae \
    --no-cuda \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --gym-id HumanoidFlagrunBulletEnv-v0 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo-kl \
    --wandb-entity cleanrl \
    --prod-mode \
    --norm-returns --norm-adv --kl --norm-obs --clip-vloss --anneal-lr \
    --no-cuda \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --gym-id HumanoidFlagrunBulletEnv-v0 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo-kl \
    --wandb-entity cleanrl \
    --prod-mode \
    --norm-returns --norm-adv --kl --clip-vloss --anneal-lr --gae \
    --no-cuda \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --gym-id HumanoidFlagrunBulletEnv-v0 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo-kl \
    --wandb-entity cleanrl \
    --prod-mode \
    --norm-adv --kl --norm-obs --clip-vloss --anneal-lr --gae \
    --no-cuda \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --gym-id HumanoidFlagrunBulletEnv-v0 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo-kl \
    --wandb-entity cleanrl \
    --prod-mode \
    --norm-returns --kl --norm-obs --clip-vloss --anneal-lr --gae \
    --no-cuda \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --gym-id HumanoidFlagrunBulletEnv-v0 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo-kl \
    --wandb-entity cleanrl \
    --prod-mode \
    --norm-returns --norm-adv --kl --norm-obs --clip-vloss --gae \
    --no-cuda \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --gym-id HumanoidFlagrunBulletEnv-v0 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo-kl \
    --wandb-entity cleanrl \
    --prod-mode \
    --norm-returns --norm-adv --kl --norm-obs --anneal-lr --gae \
    --no-cuda \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --gym-id HumanoidFlagrunBulletEnv-v0 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo-kl \
    --wandb-entity cleanrl \
    --prod-mode \
    --kl --norm-returns \
    --no-cuda \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --gym-id HumanoidFlagrunBulletEnv-v0 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo-kl \
    --wandb-entity cleanrl \
    --prod-mode \
    --kl --norm-adv \
    --no-cuda \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --gym-id HumanoidFlagrunBulletEnv-v0 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo-kl \
    --wandb-entity cleanrl \
    --prod-mode \
    --kl --norm-obs \
    --no-cuda \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --gym-id HumanoidFlagrunBulletEnv-v0 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo-kl \
    --wandb-entity cleanrl \
    --prod-mode \
    --kl --clip-vloss \
    --no-cuda \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --gym-id HumanoidFlagrunBulletEnv-v0 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo-kl \
    --wandb-entity cleanrl \
    --prod-mode \
    --kl --anneal-lr \
    --no-cuda \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --gym-id HumanoidFlagrunBulletEnv-v0 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo-kl \
    --wandb-entity cleanrl \
    --prod-mode \
    --kl --gae \
    --no-cuda \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --gym-id HumanoidFlagrunBulletEnv-v0 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo-kl \
    --wandb-entity cleanrl \
    --prod-mode \
    --norm-adv --norm-returns --kl --norm-obs --clip-vloss --anneal-lr --gae \
    --no-cuda \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done
