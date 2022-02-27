# Proximal Policy Gradient (PPO)


## Overview

PPO is one of the most popular DRL algorithms. It runs reasonably fast by leveraging vector (parallel) environments and naturally works well with different action spaces, therefore supporting a variety of games. It also has good sample efficiency compared to algorithms such as DQN.


Original paper: 

* [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347)

Reference resources:

* [Implementation Matters in Deep Policy Gradients: A Case Study on PPO and TRPO](https://arxiv.org/abs/2005.12729)
* [What Matters In On-Policy Reinforcement Learning? A Large-Scale Empirical Study](https://arxiv.org/abs/2006.05990)

All our PPO implementations below are augmented with the same code-level optimizations presented in `openai/baselines`'s [PPO](https://github.com/openai/baselines/tree/master/baselines/ppo2). See [The 32 Implementation Details of Proximal Policy Optimization (PPO) Algorithm](https://costa.sh/blog-the-32-implementation-details-of-ppo.html) for more details.

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

[ppo.py](https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/ppo.py) includes the 11 core implementation details:

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


### Experiment results

PR :material-github: [vwxyzjn/cleanrl#120](https://github.com/vwxyzjn/cleanrl/pull/120) tracks our effort to conduct experiments, and the reprodudction instructions can be found at :material-github: [vwxyzjn/cleanrl/benchmark/ppo](https://github.com/vwxyzjn/cleanrl/tree/master/benchmark/ppo).

Below are the average episodic returns for `ppo.py`. To ensure the quality of the implementation, we compared the results against `openai/baselies`' PPO.

| Environment      | `ppo.py` | `openai/baselies`' PPO
| ----------- | ----------- | ----------- |
| CartPole-v1      | 488.75 ± 18.40      |497.54 ± 4.02  |
| Acrobot-v1   | -82.48 ± 5.93     |  -81.82 ± 5.58 |
| MountainCar-v0   | -200.00 ± 0.00         | -200.00 ± 0.00 |


Learning curves:

<div class="grid-container">
<img src="../ppo/CartPole-v1.png">

<img src="../ppo/Acrobot-v1.png">

<img src="../ppo/MountainCar-v0.png">
</div>


Tracked experiments and game play videos:

<iframe src="https://wandb.ai/cleanrl/benchmark/reports/ppo-py-v1-Classic-Control---VmlldzoxNTk2NjE4" style="width:100%; height:500px" title="CleanRL CartPole-v1 Example"></iframe>

### Video tutorial

If you'd like to learn `ppo.py` in-depth, consider checking out the following video tutorial:
[![PPO1](ppo/ppo-1-title.png)](https://youtu.be/MEt6rrxH8W4)


## `ppo_atari.py`

The [ppo_atari.py](https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/ppo_atari.py) has the following features:

* For playing Atari games. It uses convolutional layers and common atari-based pre-processing techniques.
* Works with the Atari's pixel `Box` observation space of shape `(210, 160, 3)`
* Works with the `Discerete` action space
* Includes the 9 Atari-specific implementation details as shown in the following video tutorial
  [![PPO2](ppo/ppo-2-title.png)](https://youtu.be/05RMTj-2K_Y)

## `ppo_continuous_action.py`

The [ppo_continuous_action.py](https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/ppo_continuous_action.py) has the following features:

* For continuous action space. Also implemented Mujoco-specific code-level optimizations
* Works with the `Box` observation space of low-level features
* Works with the `Box` (continuous) action space
* Includes the 8 implementation details for  as shown in the following video tutorial (need fixing)
  [![PPO3](ppo/ppo-3-title.png)](https://youtu.be/05RMTj-2K_Y)
