

To use experiment tracking with wandb, run with the `--track` flag, which will
upload the videos recorded by the `--capture-video` flag.
```bash
poetry shell
wandb login # only required for the first time
python cleanrl/ppo.py --track --capture-video
```


<script id="asciicast-443626" src="https://asciinema.org/a/443626.js" async></script>
