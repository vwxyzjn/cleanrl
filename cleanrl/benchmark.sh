# https://unix.stackexchange.com/questions/103920/parallelize-a-bash-for-loop
# evaluate the algorithm in parallel
# apt-get install python-opengl
NUM_CORES=2
export MKL_NUM_THREADS=$NUM_CORES OMP_NUM_THREADS=$NUM_CORES

xvfb-run -s "-screen 0 1400x900x24" python a2c.py \
    --prod-mode True \
    --seed 1\
    --total-timesteps 2000 \
    --episode-length 100 \
    --wandb-project-name gym-microrts \
    --wandb-entity cleanrl \
    --gym-id Taxi-v3
##################################
# obs: Box, ac: Discrete 
##################################
# CartPole-v0
# Considered solved when reward > 195.0
for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -s "-screen 0 1400x900x24" python a2c.py \
    --gym-id CartPole-v0 \
    --total-timesteps 30000 \
    --wandb-project-name gym-microrts \
    --wandb-entity cleanrl \
    --prod-mode True \
    --seed $seed
    ) >& /dev/null &
done
for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -s "-screen 0 1400x900x24" python ppo.py \
    --gym-id CartPole-v0 \
    --total-timesteps 30000 \
    --wandb-project-name gym-microrts \
    --wandb-entity cleanrl \
    --prod-mode True \
    --seed $seed
    ) >& /dev/null &
done
for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -s "-screen 0 1400x900x24" python dqn.py \
    --gym-id CartPole-v0 \
    --total-timesteps 30000 \
    --wandb-project-name gym-microrts \
    --wandb-entity cleanrl \
    --prod-mode True \
    --seed $seed
    ) >& /dev/null &
done
wait

# LunarLander-v2
# TODO: add docs on the goal rewards
for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -s "-screen 0 1400x900x24" python a2c.py \
    --gym-id LunarLander-v2 \
    --total-timesteps 1000000 \
    --wandb-project-name gym-microrts \
    --wandb-entity cleanrl \
    --prod-mode True \
    --seed $seed
    ) >& /dev/null &
done
for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -s "-screen 0 1400x900x24" python ppo.py \
    --gym-id LunarLander-v2 \
    --total-timesteps 1000000 \
    --wandb-project-name gym-microrts \
    --wandb-entity cleanrl \
    --prod-mode True \
    --seed $seed
    ) >& /dev/null &
done
for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -s "-screen 0 1400x900x24" python dqn.py \
    --gym-id LunarLander-v2 \
    --total-timesteps 1000000 \
    --wandb-project-name gym-microrts \
    --wandb-entity cleanrl \
    --prod-mode True \
    --seed $seed
    ) >& /dev/null &
done
wait

##################################
# obs: Box, ac: Box 
##################################
# BipedalWalker-v2
# TODO: add docs on the goal rewards
for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -s "-screen 0 1400x900x24" python a2c_continuous_action.py \
    --gym-id BipedalWalker-v2 \
    --total-timesteps 2000000 \
    --episode-length 1600 \
    --wandb-project-name gym-microrts \
    --wandb-entity cleanrl \
    --prod-mode True \
    --seed $seed
    ) >& /dev/null &
done
wait

# Pendulum-v0
# TODO: add docs on the goal rewards
for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -s "-screen 0 1400x900x24" python a2c_continuous_action.py \
    --gym-id Pendulum-v0 \
    --total-timesteps 2000000 \
    --episode-length 200 \
    --wandb-project-name gym-microrts \
    --wandb-entity cleanrl \
    --prod-mode True \
    --seed $seed
    ) >& /dev/null &
done
wait

# HopperBulletEnv-v0
# TODO: add docs on the goal rewards
for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -s "-screen 0 1400x900x24" python a2c_continuous_action.py \
    --gym-id HopperBulletEnv-v0 \
    --total-timesteps 2000000 \
    --episode-length 1000 \
    --wandb-project-name gym-microrts \
    --wandb-entity cleanrl \
    --prod-mode True \
    --seed $seed
    ) >& /dev/null &
done
wait

# InvertedPendulumBulletEnv-v0
# TODO: add docs on the goal rewards
for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -s "-screen 0 1400x900x24" python a2c_continuous_action.py \
    --gym-id InvertedPendulumBulletEnv-v0 \
    --total-timesteps 2000000 \
    --episode-length 1000 \
    --wandb-project-name gym-microrts \
    --wandb-entity cleanrl \
    --prod-mode True \
    --seed $seed
    ) >& /dev/null &
done
wait

# Walker2DBulletEnv-v0
# TODO: add docs on the goal rewards
for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -s "-screen 0 1400x900x24" python a2c_continuous_action.py \
    --gym-id Walker2DBulletEnv-v0 \
    --total-timesteps 2000000 \
    --episode-length 1000 \
    --wandb-project-name gym-microrts \
    --wandb-entity cleanrl \
    --prod-mode True \
    --seed $seed
    ) >& /dev/null &
done
wait

# HumanoidBulletEnv-v0
# TODO: add docs on the goal rewards
for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -s "-screen 0 1400x900x24" python a2c_continuous_action.py \
    --gym-id HumanoidBulletEnv-v0 \
    --total-timesteps 2000000 \
    --episode-length 1000 \
    --wandb-project-name gym-microrts \
    --wandb-entity cleanrl \
    --prod-mode True \
    --seed $seed
    ) >& /dev/null &
done
wait

# HalfCheetahBulletEnv-v0
# TODO: add docs on the goal rewards
for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -s "-screen 0 1400x900x24" python a2c_continuous_action.py \
    --gym-id HalfCheetahBulletEnv-v0 \
    --total-timesteps 2000000 \
    --episode-length 1000 \
    --wandb-project-name gym-microrts \
    --wandb-entity cleanrl \
    --prod-mode True \
    --seed $seed
    ) >& /dev/null &
done
wait

# ReacherBulletEnv-v0
# TODO: add docs on the goal rewards
for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -s "-screen 0 1400x900x24" python a2c_continuous_action.py \
    --gym-id ReacherBulletEnv-v0 \
    --total-timesteps 2000000 \
    --episode-length 150 \
    --wandb-project-name gym-microrts \
    --wandb-entity cleanrl \
    --prod-mode True \
    --seed $seed
    ) >& /dev/null &
done
wait

##################################
# obs: Discrete, ac: Discrete
##################################
# Taxi-v2
# Considered solved when reward > 6.0
# Note: heavy exploratory problem
for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -s "-screen 0 1400x900x24" python a2c.py \
    --gym-id Taxi-v2 \
    --total-timesteps 60000 \
    --wandb-project-name gym-microrts \
    --wandb-entity cleanrl \
    --prod-mode True \
    --seed $seed
    ) >& /dev/null &
done
for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -s "-screen 0 1400x900x24" python ppo.py \
    --gym-id Taxi-v2 \
    --total-timesteps 60000 \
    --wandb-project-name gym-microrts \
    --wandb-entity cleanrl \
    --prod-mode True \
    --seed $seed
    ) >& /dev/null &
done
for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -s "-screen 0 1400x900x24" python dqn.py \
    --gym-id Taxi-v2 \
    --total-timesteps 60000 \
    --wandb-project-name gym-microrts \
    --wandb-entity cleanrl \
    --prod-mode True \
    --seed $seed
    ) >& /dev/null &
done
wait
