# Soft Actor-Critic Benchmark

This repository contains instructions to reproduce our SAC experiments.

## Install CleanRL

Prerequisites:
* Python 3.8+
* [Poetry](https://python-poetry.org)

Install dependencies:

```bash
git clone https://github.com/vwxyzjn/cleanrl.git && cd cleanrl
git checkout v0.6.0 # pinned master version
poetry install
poetry install -E pybullet
poetry install -E mujoco
```

## Reproducing CleanRL's SAC Benchmark

### Pybullet
Running the following scripts will sequentially execute the `sac_continuous_action.py` over three seeds, for the `HalfCheetahBulletEnv-v0`, `Walker2DBulletEnv-v0`, and `HopperBulletEnv-v0` environments.

```bash
# reproduce the classic control experiments
bash benchmark/sac/pybullet.sh
```

### Free Mujoco
Running the following scripts will sequentially execute the `sac_continuous_action.py` over three seeds, for the `HalfCheetah-v2`, `Walker2d-v2`, and `Hopper-v2` environments.

```bash
# reproduce the classic control experiments
bash benchmark/sac/mujoco.sh
```

### Additional comment
Note that you may need to overwrite the `--wandb-entity cleanrl` to your own W&B entity, in case you have not obtained access to the `cleanrl/openbenchmark` project.
