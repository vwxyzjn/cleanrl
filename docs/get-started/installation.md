# Installation

## Prerequisites:

* Python 3.8+
* [Poetry](https://python-poetry.org)

Simply run the following command for a quick start

```bash
git clone https://github.com/vwxyzjn/cleanrl.git && cd cleanrl
poetry install
```

<script id="asciicast-443647" src="https://asciinema.org/a/443647.js" async></script>

## Optional Dependencies

CleanRL makes it easy to install optional dependencies for common RL environments
and various development utilities. These optional dependencies are defined at
[`pyproject.toml`](https://github.com/vwxyzjn/cleanrl/blob/502f0f3abd805799d98b2d89a2564b6470b3dad0/pyproject.toml#L38-L44) as shown below:


```toml
atari = ["ale-py", "AutoROM", "stable-baselines3"]
pybullet = ["pybullet"]
procgen = ["procgen", "stable-baselines3"]
pettingzoo = ["pettingzoo", "stable-baselines3", "pygame", "pymunk"]
plot = ["pandas", "seaborn"]
cloud = ["boto3", "awscli"]
docs = ["mkdocs-material"]
spyder = ["spyder"]
```

You can install them using the following command

```
poetry install -E atari
poetry install -E pybullet
poetry install -E procgen
poetry install -E pettingzoo
poetry install -E plot
poetry install -E cloud
poetry install -E docs
poetry install -E spyder
```
