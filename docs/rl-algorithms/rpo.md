# Robust Policy Optimization (RPO)

## Overview

RPO leverages a method of perturbing the distribution representing actions. The goal is to encourage high-entropy actions and provide a better representation of the action space. The method consists of a simple modification on top of the objective of the PPO algorithm. Assuming PPO represents action with a parameterized distribution with mean and standard deviation (e.g., Gaussian). In the RPO algorithm, the mean of the action distribution is perturbed using a random number drawn from a Uniform distribution.

Original paper: 

* [Robust Policy Optimization in Deep Reinforcement Learning](https://arxiv.org/abs/2212.07536)

## Implemented Variants


| Variants Implemented      | Description |
| ----------- | ----------- |
| :material-github: [`rpo_continuous_action.py`](https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/rpo_continuous_action.py), :material-file-document: [docs](/rl-algorithms/rpo/#rpo_continuous_actionpy) | For classic control tasks like Gym `Pendulum-v1`, dm_control, Pybullet. |

Below are our single-file implementations of RPO:

## `rpo_continuous_action.py`

`rpo_continuous_action.py` works with Gym (Gymnasium), dm_control, Pybullet environments with continuous action and vector observations.

The [rpo_continuous_action.py](https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/rpo_continuous_action.py) has the following features (similar to [ppo_continuous_action.py](https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/ppo_continuous_action.py)):

* For continuous action space. Also implemented Mujoco-specific code-level optimizations
* Works with the `Box` observation space of low-level features
* Works with the `Box` (continuous) action space
* adding experimental support for [Gymnasium](https://gymnasium.farama.org/)
* 🧪 support `dm_control` environments via [Shimmy](https://github.com/Farama-Foundation/Shimmy)

### Usage

```bash
# mujoco v4 environments
poetry install --with mujoco
python cleanrl/rpo_continuous_action.py --help
python cleanrl/rpo_continuous_action.py --env-id Hopper-v2
# dm_control v4 environments
poetry install --with mujoco,dm_control
python cleanrl/rpo_continuous_action.py --env-id dm_control/cartpole-balance-v0
# backwards compatibility with mujoco v2 environments
poetry install --with mujoco_py,mujoco
python cleanrl/rpo_continuous_action.py --env-id Hopper-v2
```

### Explanation of the logged metrics

See [related docs](/rl-algorithms/ppo/#explanation-of-the-logged-metrics) for `ppo.py`.

### Implementation details
Similar to PPO [ppo_continuous_action.py](https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/ppo_continuous_action.py) with a change in `get_action_and_value` method in `Agent` class.

[TODO: Add code snippets, and comparison with PPO]

### Experiment results

**DeepMind Control**

Results on all dm_control environments. The PPO and RPO run for 8M timesteps, and results are computed over 10 random seeds.

**Table:**

|                                       | ppo_continuous_action_8M ({'tag': ['v1.0.0-13-gcbd83f6']})   | rpo_continuous_action ({'tag': ['pr-331']})   |
|:--------------------------------------|:-------------------------------------------------------------|:----------------------------------------------|
| dm_control/acrobot-swingup-v0         | 25.12 ± 7.77                                                 | 41.62 ± 5.26                                  |
| dm_control/acrobot-swingup_sparse-v0  | 1.71 ± 0.74                                                  | 3.25 ± 0.91                                   |
| dm_control/ball_in_cup-catch-v0       | 937.71 ± 8.68                                                | 941.16 ± 9.67                                 |
| dm_control/cartpole-balance-v0        | 791.93 ± 13.67                                               | 797.17 ± 9.63                                 |
| dm_control/cartpole-balance_sparse-v0 | 990.23 ± 7.34                                                | 989.03 ± 5.20                                 |
| dm_control/cartpole-swingup-v0        | 583.60 ± 13.82                                               | 615.99 ± 11.85                                |
| dm_control/cartpole-swingup_sparse-v0 | 240.45 ± 299.08                                              | 526.18 ± 190.39                               |
| dm_control/cartpole-two_poles-v0      | 217.71 ± 5.15                                                | 219.31 ± 6.76                                 |
| dm_control/cartpole-three_poles-v0    | 159.93 ± 2.12                                                | 160.10 ± 2.36                                 |
| dm_control/cheetah-run-v0             | 473.61 ± 107.87                                              | 562.38 ± 59.67                                |
| dm_control/dog-stand-v0               | 332.40 ± 24.13                                               | 504.45 ± 135.58                               |
| dm_control/dog-walk-v0                | 124.68 ± 23.05                                               | 166.50 ± 45.85                                |
| dm_control/dog-trot-v0                | 80.08 ± 13.04                                                | 115.28 ± 30.15                                |
| dm_control/dog-run-v0                 | 67.93 ± 6.84                                                 | 104.17 ± 24.32                                |
| dm_control/dog-fetch-v0               | 28.73 ± 4.70                                                 | 44.43 ± 7.21                                  |
| dm_control/finger-spin-v0             | 628.08 ± 253.43                                              | 849.49 ± 23.43                                |
| dm_control/finger-turn_easy-v0        | 248.87 ± 80.39                                               | 447.46 ± 129.80                               |
| dm_control/finger-turn_hard-v0        | 82.28 ± 34.94                                                | 238.67 ± 134.88                               |
| dm_control/fish-upright-v0            | 541.82 ± 64.54                                               | 801.81 ± 35.79                                |
| dm_control/fish-swim-v0               | 84.81 ± 6.90                                                 | 139.70 ± 40.24                                |
| dm_control/hopper-stand-v0            | 3.11 ± 1.63                                                  | 408.24 ± 203.25                               |
| dm_control/hopper-hop-v0              | 5.84 ± 17.28                                                 | 63.69 ± 87.18                                 |
| dm_control/humanoid-stand-v0          | 21.13 ± 30.14                                                | 140.61 ± 58.36                                |
| dm_control/humanoid-walk-v0           | 8.32 ± 17.00                                                 | 77.51 ± 50.81                                 |
| dm_control/humanoid-run-v0            | 5.49 ± 9.20                                                  | 24.16 ± 19.90                                 |
| dm_control/humanoid-run_pure_state-v0 | 1.10 ± 0.13                                                  | 3.32 ± 2.50                                   |
| dm_control/humanoid_CMU-stand-v0      | 4.65 ± 0.30                                                  | 4.34 ± 0.27                                   |
| dm_control/humanoid_CMU-run-v0        | 0.86 ± 0.08                                                  | 0.85 ± 0.04                                   |
| dm_control/manipulator-bring_ball-v0  | 0.41 ± 0.22                                                  | 0.55 ± 0.37                                   |
| dm_control/manipulator-bring_peg-v0   | 0.57 ± 0.24                                                  | 2.64 ± 2.19                                   |
| dm_control/manipulator-insert_ball-v0 | 41.53 ± 15.58                                                | 39.24 ± 17.19                                 |
| dm_control/manipulator-insert_peg-v0  | 49.89 ± 14.26                                                | 53.23 ± 16.02                                 |
| dm_control/pendulum-swingup-v0        | 464.82 ± 379.09                                              | 776.21 ± 20.15                                |
| dm_control/point_mass-easy-v0         | 530.17 ± 262.14                                              | 652.18 ± 21.93                                |
| dm_control/point_mass-hard-v0         | 132.29 ± 69.71                                               | 184.29 ± 24.17                                |
| dm_control/quadruped-walk-v0          | 239.11 ± 85.20                                               | 600.71 ± 213.40                               |
| dm_control/quadruped-run-v0           | 165.70 ± 43.21                                               | 375.78 ± 119.95                               |
| dm_control/quadruped-escape-v0        | 23.82 ± 10.82                                                | 68.19 ± 30.69                                 |
| dm_control/quadruped-fetch-v0         | 183.03 ± 37.50                                               | 227.47 ± 21.67                                |
| dm_control/reacher-easy-v0            | 769.10 ± 48.35                                               | 740.93 ± 51.58                                |
| dm_control/reacher-hard-v0            | 636.89 ± 77.68                                               | 584.33 ± 43.76                                |
| dm_control/stacker-stack_2-v0         | 61.73 ± 17.34                                                | 64.32 ± 7.15                                  |
| dm_control/stacker-stack_4-v0         | 71.68 ± 29.03                                                | 53.78 ± 20.09                                 |
| dm_control/swimmer-swimmer6-v0        | 152.82 ± 30.77                                               | 166.89 ± 14.49                                |
| dm_control/swimmer-swimmer15-v0       | 148.59 ± 22.63                                               | 153.45 ± 17.12                                |
| dm_control/walker-stand-v0            | 453.73 ± 217.72                                              | 736.72 ± 145.16                               |
| dm_control/walker-walk-v0             | 308.91 ± 87.85                                               | 785.61 ± 133.29                               |
| dm_control/walker-run-v0              | 123.98 ± 91.34                                               | 384.08 ± 123.89                               |

**Learning curves:**
![](../rpo/dm_control_all_ppo_rpo_8M.png)

**Gym (Gymnasium):**

Results on two continuous gym environments. The PPO and RPO run for 8M timesteps, and results are computed over 10 random seeds.

**Table:**

|                  | ppo_continuous_action_8M ({'tag': ['v1.0.0-13-gcbd83f6']})   | rpo_continuous_action ({'tag': ['pr-331']})   |
|:-----------------|:-------------------------------------------------------------|:----------------------------------------------|
| Pendulum-v1      | -1145.00 ± 125.97                                            | -145.82 ± 7.13                                |
| BipedalWalker-v3 | 172.74 ± 99.97                                               | 231.28 ± 22.39                                |

**Learning curves:**
![](../rpo/gym.png)
