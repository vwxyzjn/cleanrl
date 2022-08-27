# Installation

## Prerequisites

* >=3.7.1,<3.10 (not yet 3.10)
* [Poetry](https://python-poetry.org)

Simply run the following command for a quick start

```bash
git clone https://github.com/vwxyzjn/cleanrl.git && cd cleanrl
poetry install
```

<script id="asciicast-443647" src="https://asciinema.org/a/443647.js" async></script>


!!! note "Working with PyPI mirrors"

    Users in some countries (e.g., China) can usually speed up package installation via faster PyPI mirrors. If this helps you, try appending the following lines to the [pyproject.toml](https://github.com/vwxyzjn/cleanrl/blob/master/pyproject.toml) at the root of this repository and run `poetry install`

    ```toml
    [[tool.poetry.source]]
    name = "douban"
    url = "https://pypi.doubanio.com/simple/"
    default = true
    ```

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

```bash
poetry install -E atari
poetry install -E pybullet
poetry install -E mujoco
poetry install -E procgen
poetry install -E envpool
poetry install -E pettingzoo
```

## Install via `pip`

While we recommend using `poetry` to manage environments and dependencies, the traditional `requirements.txt` are available:

```bash
pip install -r requirements/requirements.txt
pip install -r requirements/requirements-atari.txt
pip install -r requirements/requirements-pybullet.txt
pip install -r requirements/requirements-mujoco.txt
pip install -r requirements/requirements-procgen.txt
pip install -r requirements/requirements-envpool.txt
pip install -r requirements/requirements-pettingzoo.txt
```
