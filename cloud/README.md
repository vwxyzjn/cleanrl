# Cloud Integration

We package our code into a docker container and use AWS Batch to
run thousands of experiments concurrently. To make the infrastructure
easy to manage and reproducible, we use Terraform to spin up services.

# Get Started

```bash
wandb login
pip install cleanrl --upgrade
git clone https://github.com/vwxyzjn/cleanrl
cd cleanrl/cloud
terraform init
terraform apply
pip install awscli
python -m awscli authenticate

# dry run to inspect the generated docker command
python -m cleanrl.utils.submit_exp --algo ppo.py \
    --other-args "--gym-id CartPole-v0 --wandb-project-name cleanrl --total-timesteps 100000 --cuda True" \
    --job-queue cpu_spot \
    --job-definition cleanrl \
    --num-seed 1 \
    --num-vcpu 1 \
    --num-memory 2000 \
    --num-hours 48.0
```
The generated docker command should look like
```
docker run -d --cpuset-cpus="0" -e WANDB=xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx -e WANDB_RESUME=allow -e WANDB_RUN_ID=2avt9i7l vwxyzjn/cleanrl:latest /bin/bash -c "python ppo.py --gym-id CartPole-v0 --wandb-project-name cleanrl --total-timesteps 100000 --cuda True --seed 1"
```

Submit a job using AWS's compute-optimized spot instances 
```
python -m cleanrl.utils.submit_exp --algo ppo.py \
    --other-args "--gym-id CartPole-v0 --wandb-project-name cleanrl --total-timesteps 100000 --cuda True" \
    --job-queue cpu_spot \
    --job-definition cleanrl \
    --num-seed 1 \
    --num-vcpu 1 \
    --num-memory 2000 \
    --num-hours 48.0 \
    --submit-aws
```

Submit a job using AWS's accelerated-computing spot instances 
```
python -m cleanrl.utils.submit_exp --algo ppo_atari_visual.py \
    --other-args "--gym-id BreakoutNoFrameskip-v4 --wandb-project-name cleanrl --cuda True" \
    --job-queue gpu_spot \
    --job-definition cleanrl \
    --num-seed 1 \
    --num-vcpu 1 \
    --num-gpu 1 \
    --num-memory 4000 \
    --num-hours 48.0 \
    --submit-aws
```

Submit a job using AWS's compute-optimized on-demand instances 
```
python -m cleanrl.utils.submit_exp --algo ppo.py \
    --other-args "--gym-id CartPole-v0 --wandb-project-name cleanrl --total-timesteps 100000 --cuda True" \
    --job-queue cpu_on_demand \
    --job-definition cleanrl \
    --num-seed 1 \
    --num-vcpu 1 \
    --num-memory 2000 \
    --num-hours 48.0 \
    --submit-aws
```

Submit a job using AWS's accelerated-computing on-demand instances 
```
python -m cleanrl.utils.submit_exp --algo ppo_atari_visual.py \
    --other-args "--gym-id BreakoutNoFrameskip-v4 --wandb-project-name cleanrl --cuda True" \
    --job-queue gpu_on_demand \
    --job-definition cleanrl \
    --num-seed 1 \
    --num-vcpu 1 \
    --num-gpu 1 \
    --num-memory 4000 \
    --num-hours 48.0 \
    --submit-aws
```