# Cloud Integration

We package our code into a docker container and use AWS Batch to
run thousands of experiments concurrently. To make the infrastructure
easy to manage and reproducible, we use Terraform to spin up services.

# Get Started

```bash
# install dependencies
# if you using zsh, this needs to be pip install cleanrl\[cloud\] --upgrade
pip install cleanrl[cloud] --upgrade #
curl -OL https://releases.hashicorp.com/terraform/0.15.3/terraform_0.15.3_linux_amd64.zip
unzip terraform_0.15.3_linux_amd64.zip && rm terraform_0.15.3_linux_amd64.zip
mv terraform /usr/local/bin/
git clone https://github.com/vwxyzjn/cleanrl
cd cleanrl/cloud

# setup: these command probably need to be run separately
wandb login
python -m awscli configure
terraform init
export AWS_DEFAULT_REGION=$(aws configure get region --profile default)
terraform apply
```

Dry run to inspect the generated docker command
```
python -m cleanrl_utils.submit_exp \
    --docker-tag vwxyzjn/cleanrl:latest \
    --command "poetry run python cleanrl/ppo.py --gym-id CartPole-v1 --total-timesteps 100000 --track --capture-video" \
    --num-seed 1 \
    --num-vcpu 1 \
    --num-memory 2000 \
    --num-hours 48.0
```

The generated docker command should look like
```
docker run -d --cpuset-cpus="0" -e WANDB_API_KEY=xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx vwxyzjn/cleanrl:latest /bin/bash -c "poetry run python cleanrl/ppo.py --gym-id CartPole-v1 --total-timesteps 100000 --track --capture-video --seed 1"
```

Submit a job using AWS's compute-optimized spot instances 
```
python -m cleanrl_utils.submit_exp \
    --docker-tag vwxyzjn/cleanrl:latest \
    --command "poetry run python cleanrl/ppo.py --gym-id CartPole-v1 --total-timesteps 100000 --track --capture-video" \
    --job-queue c6g-medium-spot \
    --num-seed 1 \
    --num-vcpu 1 \
    --num-memory 2000 \
    --num-hours 48.0 \
    --provider aws
```

Submit a job using AWS's accelerated-computing spot instances 
```
python -m cleanrl_utils.submit_exp \
    --docker-tag vwxyzjn/cleanrl:latest \
    --command "poetry run python cleanrl/ppo_atari.py --gym-id BreakoutNoFrameskip-v4 --track --capture-video" \
    --job-queue g4dn-xlarge-spot \
    --num-seed 1 \
    --num-vcpu 1 \
    --num-gpu 1 \
    --num-memory 4000 \
    --num-hours 48.0 \
    --provider aws
```

Submit a job using AWS's compute-optimized on-demand instances 
```
python -m cleanrl_utils.submit_exp \
    --docker-tag vwxyzjn/cleanrl:latest \
    --command "poetry run python cleanrl/ppo.py --gym-id CartPole-v1 --total-timesteps 100000 --track --capture-video" \
    --job-queue c6g-medium \
    --num-seed 1 \
    --num-vcpu 1 \
    --num-memory 2000 \
    --num-hours 48.0 \
    --provider aws
```

Submit a job using AWS's accelerated-computing on-demand instances 
```
python -m cleanrl_utils.submit_exp \
    --docker-tag vwxyzjn/cleanrl:latest \
    --command "poetry run python cleanrl/ppo_atari.py --gym-id BreakoutNoFrameskip-v4 --track --capture-video" \
    --job-queue g4dn-xlarge \
    --num-seed 1 \
    --num-vcpu 1 \
    --num-gpu 1 \
    --num-memory 4000 \
    --num-hours 48.0 \
    --provider aws
```

Uninstalling/Deleting the infrastructure that has been set up.
```
export AWS_DEFAULT_REGION=$(aws configure get region --profile default)
terraform destroy
```
