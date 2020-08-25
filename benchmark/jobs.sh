SUBMIT_AWS=False

python jobs.py --exp-script scripts/ppo_pybullet.sh \
    --job-queue cleanrl \
    --job-definition cleanrl \
    --num-seed 2 \
    --num-vcpu 2 \
    --num-memory 3000 \
    --num-hours 48.0 \
    --submit-aws $SUBMIT_AWS

python jobs.py --exp-script scripts/ppo_atari.sh \
    --job-queue cleanrl_gpu \
    --job-definition cleanrl \
    --num-seed 2 \
    --num-vcpu 4 \
    --num-gpu 1 \
    --num-memory 14000 \
    --num-hours 48.0 \
    --submit-aws $SUBMIT_AWS

python jobs.py --exp-script scripts/ppo_other.sh \
    --job-queue cleanrl \
    --job-definition cleanrl \
    --num-seed 2 \
    --num-vcpu 1 \
    --num-memory 3000 \
    --num-hours 16.0 \
    --submit-aws $SUBMIT_AWS

python jobs.py --exp-script scripts/td3_pybullet.sh \
    --job-queue cleanrl_gpu \
    --job-definition cleanrl \
    --num-seed 2 \
    --num-vcpu 2 \
    --num-memory 14000 \
    --num-gpu 1 \
    --num-hours 48.0 \
    --submit-aws $SUBMIT_AWS

python jobs.py --exp-script scripts/ddpg_pybullet.sh \
    --job-queue cleanrl_gpu \
    --job-definition cleanrl \
    --num-seed 2 \
    --num-vcpu 2 \
    --num-memory 14000 \
    --num-gpu 1 \
    --num-hours 48.0 \
    --submit-aws $SUBMIT_AWS

python jobs.py --exp-script scripts/sac_pybullet.sh \
    --job-queue cleanrl_gpu \
    --job-definition cleanrl \
    --num-seed 2 \
    --num-vcpu 2 \
    --num-memory 14000 \
    --num-gpu 1 \
    --num-hours 48.0 \
    --submit-aws $SUBMIT_AWS

python jobs.py --exp-script scripts/dqn_atari.sh \
    --job-queue cleanrl_gpu_large_memory \
    --job-definition cleanrl \
    --num-seed 2 \
    --num-vcpu 2 \
    --num-gpu 1 \
    --num-memory 63000 \
    --num-hours 48.0 \
    --submit-aws $SUBMIT_AWS

python jobs.py --exp-script scripts/dqn_other.sh \
    --job-queue cleanrl_gpu \
    --job-definition cleanrl \
    --num-seed 2 \
    --num-vcpu 2 \
    --num-gpu 1 \
    --num-memory 3000 \
    --num-hours 16.0 \
    --submit-aws $SUBMIT_AWS

python jobs.py --exp-script scripts/c51_atari.sh \
    --job-queue cleanrl_gpu_large_memory \
    --job-definition cleanrl \
    --num-seed 2 \
    --num-vcpu 2 \
    --num-gpu 1 \
    --num-memory 63000 \
    --num-hours 48.0 \
    --submit-aws $SUBMIT_AWS

python jobs.py --exp-script scripts/c51_other.sh \
    --job-queue cleanrl_gpu \
    --job-definition cleanrl \
    --num-seed 2 \
    --num-vcpu 2 \
    --num-gpu 1 \
    --num-memory 3000 \
    --num-hours 16.0 \
    --submit-aws $SUBMIT_AWS

python jobs.py --exp-script scripts/apex_dqn_atari.sh \
    --job-queue cleanrl_gpu_large_memory \
    --job-definition cleanrl \
    --num-seed 2 \
    --num-vcpu 16 \
    --num-gpu 1 \
    --num-memory 63000 \
    --num-hours 48.0 \
    --submit-aws $SUBMIT_AWS