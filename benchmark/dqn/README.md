# Deep Q-Learning Benchmark

This repository contains instructions to reproduce our DQN experiments.

Prerequisites:
* Python 3.8+
* [Poetry](https://python-poetry.org)
* [GitHub CLI](https://cli.github.com/)


## Reproducing CleanRL's DQN Benchmark

### Classic Control

```bash
git clone https://github.com/vwxyzjn/cleanrl.git && cd cleanrl
gh pr checkout 157
poetry install
bash benchmark/dqn/classic_control.sh
```

Note that you may need to overwrite the `--wandb-entity cleanrl` to your own W&B entity, in case you have not obtained access to the `cleanrl/openbenchmark` project.


### Atari games

```bash
git clone https://github.com/vwxyzjn/cleanrl.git && cd cleanrl
gh pr checkout 124
poetry install
poetry install -E atari
bash benchmark/dqn/atari.sh
```

Note that you may need to overwrite the `--wandb-entity cleanrl` to your own W&B entity, in case you have not obtained access to the `cleanrl/openbenchmark` project.
