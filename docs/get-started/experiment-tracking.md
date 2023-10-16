# Experiment tracking

To use experiment tracking with wandb, run with the `--track` flag, which will also
upload the videos recorded by the `--capture_video` flag.
```bash
poetry shell
wandb login # only required for the first time
python cleanrl/ppo.py --track --capture_video
```


<script id="asciicast-443626" src="https://asciinema.org/a/443626.js" async></script>

The console will output the url for the tracked experiment like the following

```bash
wandb:  View project at https://wandb.ai/costa-huang/cleanRL                            
wandb:  View run at https://wandb.ai/costa-huang/cleanRL/runs/10dwbgeh
```

When you open the URL, it's going to look like the following page:

<iframe src="https://wandb.ai/costa-huang/cleanRL/runs/10dwbgeh" style="width:100%; height:1000px" title="CleanRL CartPole-v1 Example"></iframe>

