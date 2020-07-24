SUBMIT_AWS=False

python jobs.py --exp-script scripts/td3_mujoco.sh \
    --job-queue cleanrl \
    --job-definition cleanrl \
    --num-seed 2 \
    --num-vcpu 2 \
    --num-memory 14000 \
    --num-gpu 1 \
    --num-hours 48.0 \
    --submit-aws $SUBMIT_AWS

python jobs.py --exp-script scripts/td3_pybullet.sh \
    --job-queue cleanrl \
    --job-definition cleanrl \
    --num-seed 2 \
    --num-vcpu 2 \
    --num-memory 14000 \
    --num-gpu 1 \
    --num-hours 48.0 \
    --submit-aws $SUBMIT_AWS

python jobs.py --exp-script scripts/ddpg_mujoco.sh \
    --job-queue cleanrl \
    --job-definition cleanrl \
    --num-seed 2 \
    --num-vcpu 2 \
    --num-memory 14000 \
    --num-gpu 1 \
    --num-hours 48.0 \
    --submit-aws $SUBMIT_AWS

python jobs.py --exp-script scripts/ddpg_pybullet.sh \
    --job-queue cleanrl \
    --job-definition cleanrl \
    --num-seed 2 \
    --num-vcpu 2 \
    --num-memory 14000 \
    --num-gpu 1 \
    --num-hours 48.0 \
    --submit-aws $SUBMIT_AWS

python jobs.py --exp-script scripts/ppo_other.sh \
    --job-queue cleanrl \
    --job-definition cleanrl \
    --num-seed 2 \
    --num-vcpu 1 \
    --num-memory 1500 \
    --num-hours 16.0 \
    --submit-aws $SUBMIT_AWS
