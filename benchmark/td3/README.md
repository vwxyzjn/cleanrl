# Twin Delayed Deep Deterministic Policy Gradient (TD3) Benchmark

This repository contains instructions to reproduce our TD3 experiments.

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

## Reproduce CleanRL's TD3 Benchmark

Execute the command below. Note that you may need to overwrite the `--wandb-entity cleanrl` to your own W&B entity.

```bash
# reproduce the MuJoCo experiments
bash mujoco.sh
```

