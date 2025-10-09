# Installation

## Prerequisites

* Python >=3.7.1,<3.11
* [uv 0.7.19+](https://docs.astral.sh/uv/)

Simply run the following command for a quick start

```bash
git clone https://github.com/vwxyzjn/cleanrl.git && cd cleanrl
uv pip install .
```

<script id="asciicast-443647" src="https://asciinema.org/a/443647.js" async></script>


!!! note "Working with different CUDA versions for `torch`"

    By default, the `torch` wheel is built with CUDA 10.2. If you are using newer NVIDIA GPUs (e.g., 3060 TI), you may need to specifically install CUDA 11.3 wheels by overriding the `torch` dependency with `pip`:

    ```bash
    uv pip install "torch==1.12.1" --upgrade --extra-index-url https://download.pytorch.org/whl/cu113
    ```


## Install via `pip`

While we recommend using `uv` to manage environments and dependencies, the traditional `requirements.txt` are available:

```bash
# core dependencies
pip install -r requirements/requirements.txt

# optional dependencies
pip install -r requirements/requirements-atari.txt
pip install -r requirements/requirements-mujoco.txt
pip install -r requirements/requirements-mujoco_py.txt
pip install -r requirements/requirements-procgen.txt
pip install -r requirements/requirements-envpool.txt
pip install -r requirements/requirements-pettingzoo.txt
pip install -r requirements/requirements-jax.txt
pip install -r requirements/requirements-docs.txt
pip install -r requirements/requirements-cloud.txt
```


## Optional Dependencies

CleanRL makes it easy to install optional dependencies for common RL environments
and various development utilities. These optional dependencies are defined at the
[`pyproject.toml`](https://github.com/vwxyzjn/cleanrl/blob/6afb51624a6fd51775b8351dd25099bd778cb1b1/pyproject.toml#L22-L37) as optional dependencies

You can install them using the following command

```bash
uv pip install ".[atari]"
uv pip install ".[mujoco]"
uv pip install ".[dm_control]"
uv pip install ".[procgen]"
uv pip install ".[envpool]"
uv pip install ".[pettingzoo]"
uv pip install ".[jax]"
uv pip install ".[optuna]"
uv pip install ".[docs]"
uv pip install ".[cloud]"
```
