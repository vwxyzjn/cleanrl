# Basic Installation

Prerequisites:

* Python 3.8+
* [Poetry](https://python-poetry.org)

Simply run the following command for a quick start

```bash
git clone https://github.com/vwxyzjn/cleanrl.git && cd cleanrl
poetry install
poetry run python cleanrl/ppo.py \
    --seed 1 \
    --gym-id CartPole-v0 \
    --total-timesteps 50000 \
    --capture-video

# open another temrminal and enter `cd cleanrl/cleanrl`
tensorboard --logdir runs
# check out the videos of the agent's gameplay in the `videos` folder
```


<script id="asciicast-443622" src="https://asciinema.org/a/443622.js" async></script>
