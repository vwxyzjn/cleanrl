# Submit Experiments

### Inspection

Dry run to inspect the generated docker command
```
poetry run python -m cleanrl_utils.submit_exp \
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

### Run on AWS

Submit a job using AWS's compute-optimized spot instances 
```
poetry run python -m cleanrl_utils.submit_exp \
    --docker-tag vwxyzjn/cleanrl:latest \
    --command "poetry run python cleanrl/ppo.py --gym-id CartPole-v1 --total-timesteps 100000 --track --capture-video" \
    --job-queue c5a-large-spot \
    --num-seed 1 \
    --num-vcpu 1 \
    --num-memory 2000 \
    --num-hours 48.0 \
    --provider aws
```

Submit a job using AWS's accelerated-computing spot instances 
```
poetry run python -m cleanrl_utils.submit_exp \
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
poetry run python -m cleanrl_utils.submit_exp \
    --docker-tag vwxyzjn/cleanrl:latest \
    --command "poetry run python cleanrl/ppo.py --gym-id CartPole-v1 --total-timesteps 100000 --track --capture-video" \
    --job-queue c5a-large \
    --num-seed 1 \
    --num-vcpu 1 \
    --num-memory 2000 \
    --num-hours 48.0 \
    --provider aws
```

Submit a job using AWS's accelerated-computing on-demand instances 
```
poetry run python -m cleanrl_utils.submit_exp \
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

<script id="asciicast-445050" src="https://asciinema.org/a/445050.js" async></script>

Then you should see:

![aws_batch1.png](aws_batch1.png)
![aws_batch2.png](aws_batch2.png)

![wandb.png](wandb.png)