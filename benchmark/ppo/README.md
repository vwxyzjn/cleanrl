# Proximal Policy Optimization Benchmark

This repository contains instructions to reproduce our PPO experiments done with CleanRL and `openai/baselines`.

## Install CleanRL

Prerequisites:
* Python 3.8+
* [Poetry](https://python-poetry.org)

Install dependencies:

```bash
git clone https://github.com/vwxyzjn/cleanrl.git && cd cleanrl
poetry install
```

## Reproduce CleanRL's PPO Benchmark

Follow the scripts at the `cleanrl` sub-folder. Note that you may need to overwrite the `--wandb-entity cleanrl` to your own W&B entity.

```bash
# reproduce the classic control experiments
bash cleanrl/classic_control.sh
```

## Install `openai/baselines`

Follow the instructions at our fork https://github.com/vwxyzjn/baselines to install.

## Reproduce CleanRL's PPO Benchmark

Follow the scripts at the `baselines` sub-folder. Note that you may need to overwrite the `WANDB_ENTITY=cleanrl` to your own W&B entity.

```bash
# reproduce the classic control experiments
bash cleanrl/classic_control.sh
```
