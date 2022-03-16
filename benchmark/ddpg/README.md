# Deep Deterministic Policy Gradient Benchmark

This repository contains instructions to reproduce our DDPG experiments.

## Install CleanRL

Prerequisites:
* Python 3.8+
* [Poetry](https://python-poetry.org)

Install dependencies:

```bash
git clone https://github.com/vwxyzjn/cleanrl.git && cd cleanrl
git checkout v0.6.0 # pinned master version
poetry install
```

## Reproduce CleanRL's DDPG Benchmark

Follow the commNote that you may need to overwrite the `--wandb-entity cleanrl` to your own W&B entity.

```bash
# reproduce the classic control experiments
bash cleanrl/mujoco.sh
```

