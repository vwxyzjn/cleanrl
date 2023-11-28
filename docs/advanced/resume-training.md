# Resume Training


A common question we get asked is how to set up model checkpoints to continue training. In this document, we take this [PPO example](https://github.com/vwxyzjn/gym-microrts/blob/master/experiments/ppo_gridnet.py) to explain that question.

## Save model checkpoints

The first step is to save models periodically. By default, we save the model to `wandb`.

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

Then we could run the following to train our agents

```
python ppo_gridnet.py --prod-mode --capture_video
```

If the training was terminated early, we can still see the last updated model `agent.pt` in W&B like in this URL [https://wandb.ai/costa-huang/cleanRL/runs/21421tda/files](https://wandb.ai/costa-huang/cleanRL/runs/21421tda/files) or as follows

<iframe src="https://wandb.ai/costa-huang/cleanRL/runs/21421tda/files" style="width:100%; height:500px" title="CleanRL CartPole-v1 Example"></iframe>


## Resume training

The second step is to automatically download the `agent.pt` from the URL above and resume training as follows:


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

To resume training, note the ID of the experiment is `21421tda` as in the URL [https://wandb.ai/costa-huang/cleanRL/runs/21421tda](https://wandb.ai/costa-huang/cleanRL/runs/21421tda), so we need to pass in the ID via environment variable to trigger the resume mode of W&B:

```
WANDB_RUN_ID=21421tda WANDB_RESUME=must python ppo_gridnet.py --prod-mode --capture_video
``` 