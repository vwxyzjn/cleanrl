# Setup checkpoints and resume training


A common question we get asked is how to set up model checkpoints to 
continue training. In this document we take this 
[PPO example](https://github.com/vwxyzjn/gym-microrts/blob/master/experiments/ppo_gridnet.py) 
and explain it.

The first step is to periodically save models. By default we save the
model to `wandb` to help scale this approach.

```python linenums="1" hl_lines="3 4 6 9-14"
num_updates = args.total_timesteps // args.batch_size

CHECKPOINT_FREQUENCY = 50
starting_update = 1

for update in range(starting_update, num_updates + 1):
    # ... do rollouts and train models

    if args.track:
        # make sure to tune `CHECKPOINT_FREQUENCY` 
        # so models are not saved too frequently
        if update % CHECKPOINT_FREQUENCY == 0:
            torch.save(agent.state_dict(), f"{wandb.run.dir}/agent.pt")
            wandb.save(f"{wandb.run.dir}/agent.pt", policy="now")
```



```python linenums="1" hl_lines="6-16"
num_updates = args.total_timesteps // args.batch_size

CHECKPOINT_FREQUENCY = 50
starting_update = 1

if args.track and wandb.run.resumed:
    starting_update = run.summary.get("charts/update") + 1
    global_step = starting_update * args.batch_size
    api = wandb.Api()
    run = api.run(f"{run.entity}/{run.project}/{run.id}")
    model = run.file("agent.pt")
    model.download(f"models/{experiment_name}/")
    agent.load_state_dict(torch.load(
        f"models/{experiment_name}/agent.pt", map_location=device))
    agent.eval()
    print(f"resumed at update {starting_update}")

for update in range(starting_update, num_updates + 1):
    # ... do rollouts and train models

    if args.track:
        # make sure to tune `CHECKPOINT_FREQUENCY` 
        # so models are not saved too frequently
        if update % CHECKPOINT_FREQUENCY == 0:
            torch.save(agent.state_dict(), f"{wandb.run.dir}/agent.pt")
            wandb.save(f"{wandb.run.dir}/agent.pt", policy="now")
```