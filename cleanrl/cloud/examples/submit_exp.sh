python -m cleanrl.submit_exp --exp-script offline_dqn_cql_atari_visual.sh \
    --algo offline_dqn_cql_atari_visual.py \
    --total-timesteps 10000000 \
    --env-ids BeamRiderNoFrameskip-v4 QbertNoFrameskip-v4 SpaceInvadersNoFrameskip-v4 PongNoFrameskip-v4 BreakoutNoFrameskip-v4 \
    --wandb-project-name cleanrl.benchmark \
    --other-args "--wandb-entity cleanrl --cuda True" \
    --job-queue cleanrl_gpu_large_memory \
    --job-definition cleanrl \
    --num-seed 2 \
    --num-vcpu 16 \
    --num-gpu 1 \
    --num-memory 63000 \
    --num-hours 48.0 \
    --submit-aws $SUBMIT_AWS

python ppg_procgen_impala_cnn.py --env-id starpilot --capture_video --track --wandb-entity cleanrl --wandb-project cleanrl.benchmark --seed 1

python -m cleanrl.utils.submit_exp --exp-script ppo.sh \
    --algo ppo.py \
    --total-timesteps 100000 \
    --env-ids CartPole-v0 \
    --wandb-project-name cleanrl \
    --other-args "--wandb-entity cleanrl --cuda True" \
    --job-queue gpu \
    --job-definition cleanrl \
    --num-seed 1 \
    --num-vcpu 1 \
    --num-gpu 1 \
    --num-memory 13000 \
    --num-hours 48.0 \
    --submit-aws $SUBMIT_AWS

python -m cleanrl.utils.submit_exp --exp-script ppo.sh \
    --algo ppo.py \
    --total-timesteps 100000 \
    --env-ids CartPole-v0 \
    --wandb-project-name cleanrl \
    --other-args "--wandb-entity cleanrl --cuda True" \
    --job-queue cpu \
    --job-definition cleanrl \
    --num-seed 1 \
    --num-vcpu 1 \
    --num-memory 2000 \
    --num-hours 48.0 \
    --submit-aws $SUBMIT_AWS


python -m cleanrl.utils.submit_exp --exp-script ppo.sh \
    --algo ppo.py \
    --other-args "--env-id CartPole-v0 --wandb-project-name cleanrl --total-timesteps 100000 --wandb-entity cleanrl --cuda True" \
    --job-queue cpu \
    --job-definition cleanrl \
    --num-seed 1 \
    --num-vcpu 1 \
    --num-memory 2000 \
    --num-hours 48.0 \