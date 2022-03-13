# Deep Deterministic Policy Gradient (DDPG)


## Overview

DDPG is one of the most popular DRL algorithms. It runs reasonably fast by leveraging vector (parallel) environments and naturally works well with different action spaces, therefore supporting a variety of games. It also has good sample efficiency compared to algorithms such as DQN.


Original paper: 

* [Continuous control with deep reinforcement learning](https://arxiv.org/abs/1509.02971)

Reference resources:

* :material-github: [sfujim/TD3](https://github.com/sfujim/TD3)

## Implemented Variants


| Variants Implemented      | Description |
| ----------- | ----------- |
| :material-github: [`ddpg_continuous_action.py`](https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/ddpg_continuous_action.py), :material-file-document: [docs](/rl-algorithms/ddpg/#ddpg_continuous_actionpy) | For continuous action space. Also implemented Mujoco-specific code-level optimizations |


Below are our single-file implementations of PPO:

## `ddpg_continuous_action.py`

The [ppo.py](https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/ppo.py) has the following features:

* For continuous action space. Also implemented Mujoco-specific code-level optimizations
* Works with the `Box` observation space of low-level features
* Works with the `Box` (continuous) action space

### Usage

```bash
poetry install
poetry install -E pybullet
python cleanrl/ddpg_continuous_action.py --help
python cleanrl/ddpg_continuous_action.py --env-id HopperBulletEnv-v0
poetry install -E mujoco # only works in Linux
python cleanrl/ddpg_continuous_action.py --env-id Hopper-v3
```

### Implementation details

[ddpg_continuous_action.py](https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/ddpg_continuous_action.py) includes the 11 core implementation details:
<!-- 
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
1. Global Gradient Clipping (:material-github: [ppo2/model.py#L102-L108](https://github.com/openai/baselines/blob/ea25b9e8b234e6ee1bca43083f8f3cf974143998/baselines/ppo2/model.py#L102-L108)) -->


### Experiment results

PR :material-github: [vwxyzjn/cleanrl#120](https://github.com/vwxyzjn/cleanrl/pull/120) tracks our effort to conduct experiments, and the reprodudction instructions can be found at :material-github: [vwxyzjn/cleanrl/benchmark/ppo](https://github.com/vwxyzjn/cleanrl/tree/master/benchmark/ppo).

Below are the average episodic returns for `ppo.py`. To ensure the quality of the implementation, we compared the results against `openai/baselies`' PPO.

| Environment      | `ppo.py` | `openai/baselies`' PPO
| ----------- | ----------- | ----------- |
| CartPole-v1      | 488.75 ± 18.40      |497.54 ± 4.02  |
| Acrobot-v1   | -82.48 ± 5.93     |  -81.82 ± 5.58 |
| MountainCar-v0   | -200.00 ± 0.00         | -200.00 ± 0.00 |


Learning curves:
<!-- 
<div class="grid-container">
<img src="../ppo/CartPole-v1.png">

<img src="../ppo/Acrobot-v1.png">

<img src="../ppo/MountainCar-v0.png">
</div> -->


Tracked experiments and game play videos:

<!-- <iframe src="https://wandb.ai/cleanrl/benchmark/reports/ppo-py-v1-Classic-Control---VmlldzoxNTk2NjE4" style="width:100%; height:500px" title="CleanRL CartPole-v1 Example"></iframe> -->
