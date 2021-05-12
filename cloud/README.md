# Cloud Integration

We package our code into a docker container and use AWS Batch to
run thousands of experiments concurrently. To make the infrastructure
easy to manage and reproducible, we use Terraform to spin up services.

# Get Started

```bash
wandb login
git clone https://github.com/vwxyzjn/cleanrl
cd cleanrl/cloud
terraform init
terraform apply
pip install awscli
python -m awscli authenticate

# submit a job using AWS's compute-optimized spot instances 
python -m cleanrl.utils.submit_exp --exp-script ppo.sh \
    --algo ppo.py \
    --other-args "--gym-id CartPole-v0 --wandb-project-name cleanrl --total-timesteps 100000 --cuda True" \
    --job-queue cpu_spot \
    --job-definition cleanrl \
    --num-seed 1 \
    --num-vcpu 1 \
    --num-memory 2000 \
    --num-hours 48.0

# submit a job using AWS's accelerated-computing spot instances 
python -m cleanrl.utils.submit_exp --exp-script ppo_atari_visual.sh \
    --algo ppo_atari_visual.py \
    --other-args "--gym-id BreakoutNoFrameskip-v4 --wandb-project-name cleanrl --cuda True" \
    --job-queue gpu_spot \
    --job-definition cleanrl \
    --num-seed 1 \
    --num-vcpu 1 \
    --num-gpu 1 \
    --num-memory 4000 \
    --num-hours 48.0

# submit a job using AWS's compute-optimized on-demand instances 
python -m cleanrl.utils.submit_exp --exp-script ppo.sh \
    --algo ppo.py \
    --other-args "--gym-id CartPole-v0 --wandb-project-name cleanrl --total-timesteps 100000 --cuda True" \
    --job-queue cpu_on_demand \
    --job-definition cleanrl \
    --num-seed 1 \
    --num-vcpu 1 \
    --num-memory 2000 \
    --num-hours 48.0

# submit a job using AWS's accelerated-computing on-demand instances 
python -m cleanrl.utils.submit_exp --exp-script ppo_atari_visual.sh \
    --algo ppo_atari_visual.py \
    --other-args "--gym-id BreakoutNoFrameskip-v4 --wandb-project-name cleanrl --cuda True" \
    --job-queue gpu_on_demand \
    --job-definition cleanrl \
    --num-seed 1 \
    --num-vcpu 1 \
    --num-gpu 1 \
    --num-memory 4000 \
    --num-hours 48.0
```