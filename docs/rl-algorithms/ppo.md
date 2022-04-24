# Proximal Policy Gradient (PPO)


## Overview

PPO is one of the most popular DRL algorithms. It runs reasonably fast by leveraging vector (parallel) environments and naturally works well with different action spaces, therefore supporting a variety of games. It also has good sample efficiency compared to algorithms such as DQN.


Original paper: 

* [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347)

Reference resources:

* [Implementation Matters in Deep Policy Gradients: A Case Study on PPO and TRPO](https://arxiv.org/abs/2005.12729)
* [What Matters In On-Policy Reinforcement Learning? A Large-Scale Empirical Study](https://arxiv.org/abs/2006.05990)
* ⭐ [The 37 Implementation Details of Proximal Policy Optimization](https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/)

All our PPO implementations below are augmented with the same code-level optimizations presented in `openai/baselines`'s [PPO](https://github.com/openai/baselines/tree/master/baselines/ppo2). To achieve this, see how we matched the implementation details in our blog post [The 37 Implementation Details of Proximal Policy Optimization](https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/).

## Implemented Variants


| Variants Implemented      | Description |
| ----------- | ----------- |
| :material-github: [`ppo.py`](https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/ppo.py), :material-file-document: [docs](/rl-algorithms/ppo/#ppopy) | For classic control tasks like `CartPole-v1`. |
| :material-github: [`ppo_atari.py`](https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/ppo_atari.py), :material-file-document: [docs](/rl-algorithms/ppo/#ppo_ataripy) |  For playing Atari games. It uses convolutional layers and common atari-based pre-processing techniques. |
| :material-github: [`ppo_continuous_action.py`](https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/ppo_continuous_action.py), :material-file-document: [docs](/rl-algorithms/ppo/#ppo_continuous_actionpy) | For continuous action space. Also implemented Mujoco-specific code-level optimizations |


Below are our single-file implementations of PPO:

## `ppo.py`

The [ppo.py](https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/ppo.py) has the following features:

* Works with the `Box` observation space of low-level features
* Works with the `Discrete` action space
* Works with envs like `CartPole-v1`

### Usage

```bash
poetry install
python cleanrl/ppo.py --help
python cleanrl/ppo.py --env-id CartPole-v1
```

### Implementation details

[ppo.py](https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/ppo.py) is based on the "13 core implementation details" in [The 37 Implementation Details of Proximal Policy Optimization](https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/), which are as follows:

1. Vectorized architecture (:material-github: [common/cmd_util.py#L22](https://github.com/openai/baselines/blob/ea25b9e8b234e6ee1bca43083f8f3cf974143998/baselines/common/cmd_util.py#L22))
1. Orthogonal Initialization of Weights and Constant Initialization of biases (:material-github: [a2c/utils.py#L58)](https://github.com/openai/baselines/blob/ea25b9e8b234e6ee1bca43083f8f3cf974143998/baselines/a2c/utils.py#L58))
1. The Adam Optimizer's Epsilon Parameter (:material-github: [ppo2/model.py#L100](https://github.com/openai/baselines/blob/ea25b9e8b234e6ee1bca43083f8f3cf974143998/baselines/ppo2/model.py#L100))
1. Adam Learning Rate Annealing (:material-github: [ppo2/ppo2.py#L133-L135](https://github.com/openai/baselines/blob/ea25b9e8b234e6ee1bca43083f8f3cf974143998/baselines/ppo2/ppo2.py#L133-L135))
1. Generalized Advantage Estimation (:material-github: [ppo2/runner.py#L56-L65](https://github.com/openai/baselines/blob/ea25b9e8b234e6ee1bca43083f8f3cf974143998/baselines/ppo2/runner.py#L56-L65))
1. Mini-batch Updates (:material-github: [ppo2/ppo2.py#L157-L166](https://github.com/openai/baselines/blob/ea25b9e8b234e6ee1bca43083f8f3cf974143998/baselines/ppo2/ppo2.py#L157-L166))
1. Normalization of Advantages (:material-github: [ppo2/model.py#L139](https://github.com/openai/baselines/blob/ea25b9e8b234e6ee1bca43083f8f3cf974143998/baselines/ppo2/model.py#L139))
1. Clipped surrogate objective (:material-github: [ppo2/model.py#L81-L86](https://github.com/openai/baselines/blob/ea25b9e8b234e6ee1bca43083f8f3cf974143998/baselines/ppo2/model.py#L81-L86))
1. Value Function Loss Clipping (:material-github: [ppo2/model.py#L68-L75](https://github.com/openai/baselines/blob/ea25b9e8b234e6ee1bca43083f8f3cf974143998/baselines/ppo2/model.py#L68-L75))
1. Overall Loss and Entropy Bonus (:material-github: [ppo2/model.py#L91](https://github.com/openai/baselines/blob/ea25b9e8b234e6ee1bca43083f8f3cf974143998/baselines/ppo2/model.py#L91))
1. Global Gradient Clipping (:material-github: [ppo2/model.py#L102-L108](https://github.com/openai/baselines/blob/ea25b9e8b234e6ee1bca43083f8f3cf974143998/baselines/ppo2/model.py#L102-L108))
1. Debug variables (:material-github: [ppo2/model.py#L115-L116](https://github.com/openai/baselines/blob/ea25b9e8b234e6ee1bca43083f8f3cf974143998/baselines/ppo2/model.py#L115-L116))
1. Separate MLP networks for policy and value functions (:material-github: [common/policies.py#L156-L160](https://github.com/openai/baselines/blob/ea25b9e8b234e6ee1bca43083f8f3cf974143998/baselines/common/policies.py#L156-L160), [baselines/common/models.py#L75-L103](https://github.com/openai/baselines/blob/ea25b9e8b234e6ee1bca43083f8f3cf974143998/baselines/common/models.py#L75-L103))

### Experiment results

PR :material-github: [vwxyzjn/cleanrl#120](https://github.com/vwxyzjn/cleanrl/pull/120) tracks our effort to conduct experiments, and the reprodudction instructions can be found at :material-github: [vwxyzjn/cleanrl/benchmark/ppo](https://github.com/vwxyzjn/cleanrl/tree/master/benchmark/ppo).

Below are the average episodic returns for `ppo.py`. To ensure the quality of the implementation, we compared the results against `openai/baselies`' PPO.

| Environment      | `ppo.py` | `openai/baselies`' PPO (Huang et al., 2022)[^1]
| ----------- | ----------- | ----------- |
| CartPole-v1      | 492.40 ± 13.05     |497.54 ± 4.02  |
| Acrobot-v1   | -89.93 ± 6.34     |  -81.82 ± 5.58 |
| MountainCar-v0   | -200.00 ± 0.00         | -200.00 ± 0.00 |


Learning curves:

<div class="grid-container">
<img src="../ppo/CartPole-v1.png">

<img src="../ppo/Acrobot-v1.png">

<img src="../ppo/MountainCar-v0.png">
</div>


Tracked experiments and game play videos:

<iframe src="https://wandb.ai/openrlbenchmark/openrlbenchmark/reports/Classic-Control-CleanRL-s-PPO--VmlldzoxODU5MDY1" style="width:100%; height:500px" title="Classic-Control-CleanRL-s-PPO"></iframe>

### Video tutorial

If you'd like to learn `ppo.py` in-depth, consider checking out the following video tutorial:


<div style="text-align: center;"><iframe width="560" height="315" src="https://www.youtube.com/embed/MEt6rrxH8W4" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe></div>


## `ppo_atari.py`

The [ppo_atari.py](https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/ppo_atari.py) has the following features:

* For playing Atari games. It uses convolutional layers and common atari-based pre-processing techniques.
* Works with the Atari's pixel `Box` observation space of shape `(210, 160, 3)`
* Works with the `Discrete` action space


### Usage

```bash
poetry install -E atari
python cleanrl/ppo_atari.py --help
python cleanrl/ppo_atari.py --env-id BreakoutNoFrameskip-v4
```

### Implementation details

[ppo_atari.py](https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/ppo_atari.py) is based on the "9 core implementation details" in [The 37 Implementation Details of Proximal Policy Optimization](https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/), which are as follows:

1. The Use of `NoopResetEnv` (:material-github: [common/atari_wrappers.py#L12](https://github.com/openai/baselines/blob/ea25b9e8b234e6ee1bca43083f8f3cf974143998/baselines/common/atari_wrappers.py#L12)) 
1. The Use of `MaxAndSkipEnv` (:material-github: [common/atari_wrappers.py#L97](https://github.com/openai/baselines/blob/ea25b9e8b234e6ee1bca43083f8f3cf974143998/baselines/common/atari_wrappers.py#L97)) 
1. The Use of `EpisodicLifeEnv` (:material-github: [common/atari_wrappers.py#L61](https://github.com/openai/baselines/blob/ea25b9e8b234e6ee1bca43083f8f3cf974143998/baselines/common/atari_wrappers.py#L61)) 
1. The Use of `FireResetEnv` (:material-github: [common/atari_wrappers.py#L41](https://github.com/openai/baselines/blob/ea25b9e8b234e6ee1bca43083f8f3cf974143998/baselines/common/atari_wrappers.py#L41)) 
1. The Use of `WarpFrame` (Image transformation) [common/atari_wrappers.py#L134](https://github.com/openai/baselines/blob/ea25b9e8b234e6ee1bca43083f8f3cf974143998/baselines/common/atari_wrappers.py#L134) 
1. The Use of `ClipRewardEnv` (:material-github: [common/atari_wrappers.py#L125](https://github.com/openai/baselines/blob/ea25b9e8b234e6ee1bca43083f8f3cf974143998/baselines/common/atari_wrappers.py#L125)) 
1. The Use of `FrameStack` (:material-github: [common/atari_wrappers.py#L188](https://github.com/openai/baselines/blob/ea25b9e8b234e6ee1bca43083f8f3cf974143998/baselines/common/atari_wrappers.py#L188)) 
1. Shared Nature-CNN network for the policy and value functions (:material-github: [common/policies.py#L157](https://github.com/openai/baselines/blob/ea25b9e8b234e6ee1bca43083f8f3cf974143998/baselines/common/policies.py#L157), [common/models.py#L15-L26](https://github.com/openai/baselines/blob/ea25b9e8b234e6ee1bca43083f8f3cf974143998/baselines/common/models.py#L15-L26))
1. Scaling the Images to Range [0, 1] (:material-github: [common/models.py#L19](https://github.com/openai/baselines/blob/9b68103b737ac46bc201dfb3121cfa5df2127e53/baselines/common/models.py#L19))

### Experiment results

PR :material-github: [vwxyzjn/cleanrl#120](https://github.com/vwxyzjn/cleanrl/pull/120) tracks our effort to conduct experiments, and the reprodudction instructions can be found at :material-github: [vwxyzjn/cleanrl/benchmark/ppo](https://github.com/vwxyzjn/cleanrl/tree/master/benchmark/ppo).

Below are the average episodic returns for `ppo_atari.py`. To ensure the quality of the implementation, we compared the results against `openai/baselies`' PPO.

| Environment      | `ppo_atari.py` | `openai/baselies`' PPO
| ----------- | ----------- | ----------- |
| BreakoutNoFrameskip-v4      | 416.31 ± 43.92     | 406.57 ± 31.554  |
| PongNoFrameskip-v4   | 20.59 ± 0.35    |  20.512 ± 0.50 |
| BeamRiderNoFrameskip-v4   | 2445.38 ± 528.91         | 2642.97 ± 670.37 |


Learning curves:

<div class="grid-container">
<img src="../ppo/BreakoutNoFrameskip-v4.png">

<img src="../ppo/PongNoFrameskip-v4.png">

<img src="../ppo/BeamRiderNoFrameskip-v4.png">
</div>


Tracked experiments and game play videos:

<iframe src="https://wandb.ai/openrlbenchmark/openrlbenchmark/reports/Atari-CleanRL-s-PPO--VmlldzoxNjk3NjYy" style="width:100%; height:500px" title="Atari-CleanRL-s-PPO"></iframe>

### Video tutorial

If you'd like to learn `ppo.py` in-depth, consider checking out the following video tutorial:


<div style="text-align: center;"><iframe width="560" height="315" src="https://www.youtube.com/embed/05RMTj-2K_Y" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe></div>


## `ppo_continuous_action.py`

The [ppo_continuous_action.py](https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/ppo_continuous_action.py) has the following features:

* For continuous action space. Also implemented Mujoco-specific code-level optimizations
* Works with the `Box` observation space of low-level features
* Works with the `Box` (continuous) action space
* Includes the 8 implementation details for  as shown in the following video tutorial (need fixing)
  [![PPO3](ppo/ppo-3-title.png)](https://youtu.be/05RMTj-2K_Y)


[^1]: Huang, Shengyi; Dossa, Rousslan Fernand Julien; Raffin, Antonin; Kanervisto, Anssi; Wang, Weixun (2022). The 37 Implementation Details of Proximal Policy Optimization. ICLR 2022 Blog Track https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/