# CartPole-v1
CUDA_VISIBLE_DEVICES="-1" WANDB_PROJECT=openai-baselines-benchmark WANDB_ENTITY=cleanrl OPENAI_LOGDIR=$PWD/runs OPENAI_LOG_FORMAT=tensorboard poetry run python -m baselines.run_separate_networks \
    --alg=ppo2 \
    --num_timesteps=500000 \
    --num_env 4 \
    --env=CartPole-v1 \
    --network mlp \
    --value_network='copy' \
    --track \
    --seed 1

CUDA_VISIBLE_DEVICES="-1" WANDB_PROJECT=openai-baselines-benchmark WANDB_ENTITY=cleanrl OPENAI_LOGDIR=$PWD/runs OPENAI_LOG_FORMAT=tensorboard poetry run python -m baselines.run_separate_networks \
    --alg=ppo2 \
    --num_timesteps=500000 \
    --env=CartPole-v1 \
    --network mlp \
    --value_network='copy' \
    --track \
    --seed 2

CUDA_VISIBLE_DEVICES="-1" WANDB_PROJECT=openai-baselines-benchmark WANDB_ENTITY=cleanrl OPENAI_LOGDIR=$PWD/runs OPENAI_LOG_FORMAT=tensorboard poetry run python -m baselines.run_separate_networks \
    --alg=ppo2 \
    --num_timesteps=500000 \
    --env=CartPole-v1 \
    --network mlp \
    --value_network='copy' \
    --track \
    --seed 3

# Acrobot-v1
CUDA_VISIBLE_DEVICES="-1" WANDB_PROJECT=openai-baselines-benchmark WANDB_ENTITY=cleanrl OPENAI_LOGDIR=$PWD/runs OPENAI_LOG_FORMAT=tensorboard poetry run python -m baselines.run_separate_networks \
    --alg=ppo2 \
    --num_timesteps=500000 \
    --env=Acrobot-v1 \
    --network mlp \
    --value_network='copy' \
    --track \
    --seed 1

CUDA_VISIBLE_DEVICES="-1" WANDB_PROJECT=openai-baselines-benchmark WANDB_ENTITY=cleanrl OPENAI_LOGDIR=$PWD/runs OPENAI_LOG_FORMAT=tensorboard poetry run python -m baselines.run_separate_networks \
    --alg=ppo2 \
    --num_timesteps=500000 \
    --env=Acrobot-v1 \
    --network mlp \
    --value_network='copy' \
    --track \
    --seed 2

CUDA_VISIBLE_DEVICES="-1" WANDB_PROJECT=openai-baselines-benchmark WANDB_ENTITY=cleanrl OPENAI_LOGDIR=$PWD/runs OPENAI_LOG_FORMAT=tensorboard poetry run python -m baselines.run_separate_networks \
    --alg=ppo2 \
    --num_timesteps=500000 \
    --env=Acrobot-v1 \
    --network mlp \
    --value_network='copy' \
    --track \
    --seed 3

# MountainCar-v0
CUDA_VISIBLE_DEVICES="-1" WANDB_PROJECT=openai-baselines-benchmark WANDB_ENTITY=cleanrl OPENAI_LOGDIR=$PWD/runs OPENAI_LOG_FORMAT=tensorboard poetry run python -m baselines.run_separate_networks \
    --alg=ppo2 \
    --num_timesteps=500000 \
    --env=MountainCar-v0 \
    --network mlp \
    --value_network='copy' \
    --track \
    --seed 1

CUDA_VISIBLE_DEVICES="-1" WANDB_PROJECT=openai-baselines-benchmark WANDB_ENTITY=cleanrl OPENAI_LOGDIR=$PWD/runs OPENAI_LOG_FORMAT=tensorboard poetry run python -m baselines.run_separate_networks \
    --alg=ppo2 \
    --num_timesteps=500000 \
    --env=MountainCar-v0 \
    --network mlp \
    --value_network='copy' \
    --track \
    --seed 2

CUDA_VISIBLE_DEVICES="-1" WANDB_PROJECT=openai-baselines-benchmark WANDB_ENTITY=cleanrl OPENAI_LOGDIR=$PWD/runs OPENAI_LOG_FORMAT=tensorboard poetry run python -m baselines.run_separate_networks \
    --alg=ppo2 \
    --num_timesteps=500000 \
    --env=MountainCar-v0 \
    --network mlp \
    --value_network='copy' \
    --track \
    --seed 3