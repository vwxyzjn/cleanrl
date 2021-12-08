# CleanRL (Clean Implementation of RL Algorithms)

<img src="
https://img.shields.io/github/license/vwxyzjn/cleanrl">
[![tests](https://github.com/vwxyzjn/cleanrl/actions/workflows/tests.yaml/badge.svg)](https://github.com/vwxyzjn/cleanrl/actions/workflows/tests.yaml)
[![ci](https://github.com/vwxyzjn/cleanrl/actions/workflows/docs.yaml/badge.svg)](https://github.com/vwxyzjn/cleanrl/actions/workflows/docs.yaml)
[<img src="https://img.shields.io/discord/767863440248143916?label=discord">](https://discord.gg/D6RCjA6sVT)
[<img src="https://badge.fury.io/py/cleanrl.svg">](
https://pypi.org/project/cleanrl/)
[<img src="https://img.shields.io/youtube/channel/views/UCDdC6BIFRI0jvcwuhi3aI6w?style=social">](https://www.youtube.com/channel/UCDdC6BIFRI0jvcwuhi3aI6w/videos)


CleanRL is a Deep Reinforcement Learning library that provides high-quality single-file implementation with research-friendly features. The implementation is clean and simple, yet we can scale it to run thousands of experiments using AWS Batch. The highlight features of CleanRL are:



* 📜 Single-file implementation
   * *Every detail about an algorithm is put into the algorithm's own file.* It is therefore easier to fully understand an algorithm and do research with.
* 📊 Benchmarked Implementation (7+ algorithms and 34+ games at https://benchmark.cleanrl.dev)
* 📈 Tensorboard Logging
* 🪛 Local Reproducibility via Seeding
* 🎮 Videos of Gameplay Capturing
* 🧫 Experiment Management with [Weights and Biases](https://wandb.ai/site)
* 💸 Cloud Integration with docker and AWS 

You can read more about CleanRL in our [technical paper]((https://arxiv.org/abs/2111.08819)) and [documentation](https://docs.cleanrl.dev/).

Good luck have fun :rocket:

## Get started

Prerequisites:
* Python 3.8+
* [Poetry](https://python-poetry.org)

To run experiments locally, give the following a try:

```bash
git clone https://github.com/vwxyzjn/cleanrl.git && cd cleanrl
poetry install

# alternatively, you could use `poetry shell` and do
# `python run cleanrl/ppo.py`
poetry run python cleanrl/ppo.py \
    --seed 1 \
    --gym-id CartPole-v0 \
    --total-timesteps 50000

# open another temrminal and enter `cd cleanrl/cleanrl`
tensorboard --logdir runs
```

To use experiment tracking with wandb, run
```bash
wandb login # only required for the first time
poetry run python cleanrl/ppo.py \
    --seed 1 \
    --gym-id CartPole-v0 \
    --total-timesteps 50000 \
    --track \
    --wandb-project-name cleanrltest
```

To run training scripts in other games:
```
poetry shell

# classic control
python cleanrl/dqn.py --gym-id CartPole-v1
python cleanrl/ppo.py --gym-id CartPole-v1
python cleanrl/c51.py --gym-id CartPole-v1

# atari
poetry install -E atari
python cleanrl/dqn_atari.py --gym-id BreakoutNoFrameskip-v4
python cleanrl/c51_atari.py --gym-id BreakoutNoFrameskip-v4
python cleanrl/ppo_atari.py --gym-id BreakoutNoFrameskip-v4
python cleanrl/apex_dqn_atari.py --gym-id BreakoutNoFrameskip-v4

# pybullet
poetry install -E pybullet
python cleanrl/td3_continuous_action.py --gym-id MinitaurBulletDuckEnv-v0
python cleanrl/ddpg_continuous_action.py --gym-id MinitaurBulletDuckEnv-v0
python cleanrl/sac_continuous_action.py --gym-id MinitaurBulletDuckEnv-v0

# procgen
poetry install -E procgen
python cleanrl/ppo_procgen.py --gym-id starpilot
python cleanrl/ppo_procgen_impala_cnn.py --gym-id starpilot
python cleanrl/ppg_procgen.py --gym-id starpilot
python cleanrl/ppg_procgen_impala_cnn.py --gym-id starpilot
```

You may also use a prebuilt development environment hosted in Gitpod:

[![Open in Gitpod](https://gitpod.io/button/open-in-gitpod.svg)](https://gitpod.io/#https://github.com/vwxyzjn/cleanrl/tree/gitpod)

## Algorithms Implemented
- [x] Deep Q-Learning (DQN)
- [x] Categorical DQN (C51)
- [x] Proximal Policy Gradient (PPO) 
- [x] Soft Actor Critic (SAC)
- [x] Deep Deterministic Policy Gradient (DDPG)
- [x] Twin Delayed Deep Deterministic Policy Gradient (TD3)
- [x] Apex Deep Q-Learning (Apex-DQN)

## Open RL Benchmark

CleanRL has a sub project called Open RL Benchmark (https://benchmark.cleanrl.dev/), where we have tracked thousands of experiments across domains. The benchmark is interactive, and researchers can easily query information such as GPU utilization and videos of an agent's gameplay that are normally hard to acquire in other RL benchmarks. Here are some screenshots.

![](docs/static/o2.png)
![](docs/static/o3.png)
![](docs/static/o1.png)


## Support and get involved

We have a [Discord Community](https://discord.gg/D6RCjA6sVT) for support. Feel free to ask questions. Posting in [Github Issues](https://github.com/vwxyzjn/cleanrl/issues) and PRs are also welcome. Also our past video recordings are available at [YouTube](https://www.youtube.com/watch?v=dm4HdGujpPs&list=PLQpKd36nzSuMynZLU2soIpNSMeXMplnKP&index=2)

## Citing CleanRL

If you use CleanRL in your work, please cite our technical [paper](https://arxiv.org/abs/2111.08819):

```bibtex
@article{huang2021cleanrl,
    title={CleanRL: High-quality Single-file Implementations of Deep Reinforcement Learning Algorithms}, 
    author={Shengyi Huang and Rousslan Fernand Julien Dossa and Chang Ye and Jeff Braga},
    year={2021},
    eprint={2111.08819},
    archivePrefix={arXiv},
    primaryClass={cs.LG}
}
```