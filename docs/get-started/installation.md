# Installation

## Prerequisites

* Python >=3.7.1,<3.11
* [Poetry 1.2.1+](https://python-poetry.org)

Simply run the following command for a quick start

```bash
git clone https://github.com/vwxyzjn/cleanrl.git && cd cleanrl
poetry install
```

<script id="asciicast-443647" src="https://asciinema.org/a/443647.js" async></script>


!!! warning "`poetry install` hangs / stucks"

    Since 1.2+ `poetry` added some keyring authentication mechanisms that may cause `poetry install` hang or stuck. See [:material-github: python-poetry/poetry#1917](https://github.com/python-poetry/poetry/issues/1917). To fix this issue, try

    ```bash
    export PYTHON_KEYRING_BACKEND=keyring.backends.null.Keyring
    poetry install
    ```


!!! note "Working with different CUDA versions for `torch`"

    By default, the `torch` wheel is built with CUDA 10.2. If you are using newer NVIDIA GPUs (e.g., 3060 TI), you may need to specifically install CUDA 11.3 wheels by overriding the `torch` dependency with `pip`:

    ```bash
    poetry run pip install "torch==1.12.1" --upgrade --extra-index-url https://download.pytorch.org/whl/cu113
    ```


!!! note "Working with PyPI mirrors"

    Users in some countries (e.g., China) can usually speed up package installation via faster PyPI mirrors. If this helps you, try appending the following lines to the [pyproject.toml](https://github.com/vwxyzjn/cleanrl/blob/master/pyproject.toml) at the root of this repository and run `poetry install`

    ```toml
    [[tool.poetry.source]]
    name = "douban"
    url = "https://pypi.doubanio.com/simple/"
    default = true
    ```


## Install via `pip`

While we recommend using `poetry` to manage environments and dependencies, the traditional `requirements.txt` are available:

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
[`pyproject.toml`](https://github.com/vwxyzjn/cleanrl/blob/6afb51624a6fd51775b8351dd25099bd778cb1b1/pyproject.toml#L22-L37) as [poetry dependency groups](https://python-poetry.org/docs/master/managing-dependencies/#dependency-groups):


```toml
[tool.poetry.group.atari]
optional = true
[tool.poetry.group.atari.dependencies]
ale-py = "0.7.4"
AutoROM = {extras = ["accept-rom-license"], version = "^0.4.2"}
opencv-python = "^4.6.0.66"

[tool.poetry.group.procgen]
optional = true
[tool.poetry.group.procgen.dependencies]
procgen = "^0.10.7"
```

You can install them using the following command

```bash
poetry install -E atari
poetry install -E mujoco
poetry install -E mujoco_py
poetry install -E dm_control
poetry install -E procgen
poetry install -E envpool
poetry install -E pettingzoo
poetry install -E jax
poetry install -E optuna
poetry install -E docs
poetry install -E cloud
```
