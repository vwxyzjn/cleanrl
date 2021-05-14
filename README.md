# CleanRL (Clean Implementation of RL Algorithms)

[<img src="https://img.shields.io/badge/discord-cleanrl-green?label=Discord&logo=discord&logoColor=ffffff&labelColor=7289DA&color=2c2f33">](https://discord.gg/D6RCjA6sVT)
[![Meeting Recordings : cleanrl](https://img.shields.io/badge/meeting%20recordings-cleanrl-green?logo=youtube&logoColor=ffffff&labelColor=FF0000&color=282828&style=flat?label=healthinesses)](https://www.youtube.com/watch?v=dm4HdGujpPs&list=PLQpKd36nzSuMynZLU2soIpNSMeXMplnKP&index=2)
[<img src="https://github.com/vwxyzjn/cleanrl/workflows/build/badge.svg">](
https://github.com/vwxyzjn/cleanrl/actions)
[<img src="https://badge.fury.io/py/cleanrl.svg">](
https://pypi.org/project/cleanrl/)





CleanRL is a Deep Reinforcement Learning library that provides high-quality single-file implementation with research-friendly features. The implementation is clean and simple, yet we can scale it to run thousands of experiments using AWS Batch. The highlight features of CleanRL are:

<!-- At the same time, CleanRL tries to supply many research-friendly features such as cloud experiment management, support for continuous and discrete observation and action spaces, video recording of the game play, etc. These features will be very helpful for doing research, especially the video recording feature that *allows you to visually inspect the agents' behavior at various stages of the training*. -->



* 📜 Single-file implementation
   * *Every detail about an algorithm is put into the algorithm's own file.* It is therefore easier to fully understand an algorithm and do research with.
* 📊 Benchmarked Implementation (7+ algorithms and 34+ games at https://benchmark.cleanrl.dev)
* 📈 Tensorboard Logging
* 🪛 Local Reproducibility via Seeding
* 🎮 Videos of Gameplay Capturing
* 🧫 Experiment Management with [Weights and Biases](https://wandb.ai/site)
* 💸 Cloud Integration with docker and AWS 

Good luck have fun :rocket:

## Algorithms Implemented
- [x] Deep Q-Learning (DQN)
    * [dqn.py](https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/dqn.py)
        * For discrete action space.
    * [dqn_atari.py](https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/dqn_atari.py)
        * For playing Atari games. It uses convolutional layers and common atari-based pre-processing techniques.
    * [dqn_atari_visual.py](https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/dqn_atari_visual.py)
        * Adds q-values visulization for `dqn_atari.py`.
- [x] Categorical DQN (C51)
    * [c51.py](https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/c51.py)
        * For discrete action space.
    * [c51_atari.py](https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/c51_atari.py)
        * For playing Atari games. It uses convolutional layers and common atari-based pre-processing techniques.
    * [c51_atari_visual.py](https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/c51_atari_visual.py)
        * Adds return and q-values visulization for `dqn_atari.py`.
- [x] Proximal Policy Gradient (PPO) 
    * All of the PPO implementations below are augmented with some code-level optimizations. See https://costa.sh/blog-the-32-implementation-details-of-ppo.html for more details
    * [ppo.py](https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/ppo.py)
        * For discrete action space.
    * [ppo_continuous_action.py](https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/ppo_continuous_action.py)
        * For continuous action space. Also implemented Mujoco-specific code-level optimizations
    * [ppo_atari.py](https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/ppo_atari.py)
        * For playing Atari games. It uses convolutional layers and common atari-based pre-processing techniques.
    * [ppo_atari_visual.py](https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/ppo_atari_visual.py)
        * Adds action probability visulization for `ppo_atari.py`.
    * [experiments/ppo_self_play.py](https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/experiments/ppo_self_play.py)
        * Implements a self-play agent for https://github.com/hardmaru/slimevolleygym
    * [experiments/ppo_microrts.py](https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/experiments/ppo_microrts.py)
        * Implements invalid action masking and handling of `MultiDiscrete` action space for https://github.com/vwxyzjn/gym-microrts
    * [experiments/ppo_simple.py](https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/experiments/ppo_simple.py)
        * (Not recommended for using) Naive implementation for discrete action space. I keep it here for educational purposes because I feel this is what most people would implement if they had just read the paper, usually unaware of the amount of implementation details that come with the well-tuned PPO implmentation.
    * [experiments/ppo_simple_continuous_action.py](https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/experiments/ppo_simple_continuous_action.py)
        * (Not recommended for using) Naive implementation for continuous action space. 
- [x] Soft Actor Critic (SAC)
    * [sac_continuous_action.py](https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/sac_continuous_action.py)
        * For continuous action space.
- [x] Deep Deterministic Policy Gradient (DDPG)
    * [ddpg_continuous_action.py](https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/ddpg_continuous_action.py)
        * For continuous action space.
- [x] Twin Delayed Deep Deterministic Policy Gradient (TD3)
    * [td3_continuous_action.py](https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/td3_continuous_action.py)
        * For continuous action space.
- [x] Apex Deep Q-Learning (Apex-DQN)
    * [apex_dqn_atari_visual.py](https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/apex_dqn_atari_visual.py)
        * For playing Atari games. It uses convolutional layers and common atari-based pre-processing techniques.

## Get started

To run experiments locally, give the following a try:

```bash
git clone https://github.com/vwxyzjn/cleanrl.git && cd cleanrl
# we strongly recommend to use `venv` to manage dependencies
# this will make the experiment much more reproducible!
python -m venv venv
source venv/bin/activate
pip install cleanrl
python cleanrl/ppo.py \
    --seed 1 \
    --gym-id CartPole-v0 \
    --total-timesteps 50000 \
# open another temrminal and enter `cd cleanrl/cleanrl`
tensorboard --logdir runs
```

To use wandb integration, sign up an account at https://wandb.com and copy the API key. Then run

```bash
source venv/bin/activate
wandb login # only required for the first time
python cleanrl/ppo.py \
    --seed 1 \
    --gym-id CartPole-v0 \
    --total-timesteps 50000 \
    --prod-mode \
    --wandb-project-name cleanrltest
```


# Run RL experiments in various game environments

The following instructions assume linux environements. We first install the dependencies:

```bash
# install atari, pybullet, procgen, box2d, pettingzoo
source venv/bin/activate
pip install cleanrl[all]
# if you are using zsh, you will need to do 
# `pip install cleanrl\[all\]`


# install mujoco
curl -OL https://www.roboti.us/download/mujoco200_linux.zip
unzip mujoco200_linux.zip -d ~/mujoco/
mv ~/mujoco/mujoco200_linux ~/mujoco/mujoco200
unzip mujoco200_linux.zip -d ~/mujoco/
rm mujoco200_linux.zip
pip install gym[mujoco]
```

Now we can run the experiments:
```
source venv/bin/activate
cd cleanrl
# atari
python dqn_atari_visual.py --gym-id BeamRiderNoFrameskip-v4 
# pybullet
python td3_continuous_action.py --gym-id MinitaurBulletDuckEnv-v0
# procgen
python ppo_procgen_fast.py --gym-id starpilot
# box2d
python experiments/ppo_car_racing.py
# pettingzoo
python ppo_pettingzoo.py
```


## Open RL Benchmark

[Open RL Benchmark](https://wandb.ai/cleanrl/cleanrl.benchmark/reports/Open-RL-Benchmark-0-5-0---Vmlldzo0MDcxOA) by [CleanRL](https://github.com/vwxyzjn/cleanrl) is a comprehensive, interactive and reproducible benchmark of deep Reinforcement Learning (RL) algorithms. It uses Weights and Biases to keep track of the experiment data of popular deep RL algorithms (e.g. DQN, PPO, DDPG, TD3) in a variety of games (e.g. Atari, Mujoco, PyBullet, Procgen, Griddly, MicroRTS). The experiment data includes:

- reproducibility info:
    - [source code](https://app.wandb.ai/cleanrl/cleanrl.benchmark/runs/2jrqfugg/code?workspace=user-costa-huang) and [requirements.txt](https://app.wandb.ai/cleanrl/cleanrl.benchmark/runs/2jrqfugg/files/requirements.txt)
    - [](https://app.wandb.ai/cleanrl/cleanrl.benchmark/runs/2jrqfugg/code?workspace=user-costa-huang)﻿[hyper-parameters](https://app.wandb.ai/cleanrl/cleanrl.benchmark/runs/2jrqfugg/overview?workspace=user-costa-huang) and [the exact command to reproduce results](https://app.wandb.ai/cleanrl/cleanrl.benchmark/runs/2jrqfugg/overview?workspace=user-costa-huang)
- metrics:
    - [training metrics and videos of the agents playing the game](https://app.wandb.ai/cleanrl/cleanrl.benchmark/runs/2jrqfugg?workspace=user-costa-huang)
    - [system metrics](https://app.wandb.ai/cleanrl/cleanrl.benchmark/runs/2jrqfugg/system?workspace=user-costa-huang) and [logs](https://app.wandb.ai/cleanrl/cleanrl.benchmark/runs/2jrqfugg/logs?workspace=user-costa-huang)

[Open RL Benchmark](https://wandb.ai/cleanrl/cleanrl.benchmark/reports/Open-RL-Benchmark-0-5-0---Vmlldzo0MDcxOA) has over 1000+ experiments including runs from other projects, which is overwhelming to present in a single report. Instead, we present the results in separate reports. Please click on the links below to access them.

- [Atari results](https://wandb.ai/cleanrl/cleanrl.benchmark/reports/Atari--VmlldzoxMTExNTI)
- [Mujoco results](https://wandb.ai/cleanrl/cleanrl.benchmark/reports/Mujoco--VmlldzoxODE0NjE)
- [PyBullet results](https://wandb.ai/cleanrl/cleanrl.benchmark/reports/PyBullet-and-Other-Continuous-Action-Tasks--VmlldzoxODE0NzY)
- [Procgen results](https://wandb.ai/cleanrl/cleanrl.benchmark/reports/Procgen-New--Vmlldzo0NDUyMTg)
- [Griddly results](https://wandb.ai/griddly/griddly-paper-generalize?workspace=user-costa-huang)
- [Gym-μRTS results](https://wandb.ai/vwxyzjn/gym-microrts-paper/reports/Gym-RTS-Toward-Affordable-Deep-Reinforcement-Learning-Research-in-Real-time-Strategy-Games--Vmlldzo2MDIzMTg)
- [Slimevolleygym results](https://wandb.ai/cleanrl/cleanrl.benchmark/reports/Slimevolleygym--Vmlldzo0ODA1MjA)
- [PySC2 results](https://wandb.ai/cleanrl/cleanrl.benchmark/reports/Gym-pysc2-Benchmark--VmlldzoyNTEyMTc)
- [CarRacing-v0](https://wandb.ai/cleanrl/cleanrl.benchmark/reports/CarRacing-v0--VmlldzoyNDUwMzU)
- [Montezuma Revenge results](https://wandb.ai/cleanrl/cleanrl.benchmark/reports/Montezuma-Revenge--Vmlldzo1MDYxNTk)


We hope it could bring a new level of transparency, openness, and reproducibility. Our plan is to 
benchmark as many algorithms and games as possible. If you are interested, please join us and contribute
more algorithms and games. To get started, check out our [contribution guide](https://github.com/vwxyzjn/cleanrl/blob/master/CONTRIBUTING.md) and our [roadmap for the Open RL Benchmark](https://github.com/vwxyzjn/cleanrl/projects/1)


## Cloud integration

Check out the documentation [here](https://github.com/vwxyzjn/cleanrl/tree/master/cloud)


## Support and get involved

We have a [Discord Community](https://discord.gg/D6RCjA6sVT) for support. Feel free to ask questions. Posting in [Github Issues](https://github.com/vwxyzjn/cleanrl/issues) and PRs are also welcome. Also our past video recordings are available at [YouTube](https://www.youtube.com/watch?v=dm4HdGujpPs&list=PLQpKd36nzSuMynZLU2soIpNSMeXMplnKP&index=2)

<!-- In addition, we also have a monthly development cycle to implement new RL algorithms. Feel free to participate or ask questions there, too. You can sign up for our mailing list at our [Google Groups](https://groups.google.com/forum/#!forum/rlimplementation/join) to receive event RVSP which contains the Hangout video call address every week.  -->

## Contribution

We have a short contribution guide here https://github.com/vwxyzjn/cleanrl/blob/master/CONTRIBUTING.md. Consider adding new algorithms 
or test new games on the Open RL Benchmark (https://benchmark.cleanrl.dev)

Big thanks to all the contributors of CleanRL!


## Citing our project

Please consider using the following Bibtex entry:

```
@misc{cleanrl,
  author = {Shengyi Huang and Rousslan Dossa and Chang Ye},
  title = {CleanRL: High-quality Single-file Implementation of Deep Reinforcement Learning algorithms},
  year = {2020},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/vwxyzjn/cleanrl/}},
}
```

## References

I have been heavily inspired by the many repos and blog posts. Below contains a incomplete list of them.

* http://inoryy.com/post/tensorflow2-deep-reinforcement-learning/
* https://github.com/seungeunrho/minimalRL
* https://github.com/Shmuma/Deep-Reinforcement-Learning-Hands-On
* https://github.com/hill-a/stable-baselines

The following ones helped me a lot with the continuous action space handling:

* https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail
* https://github.com/zhangchuheng123/Reinforcement-Implementation/blob/master/code/ppo.py
