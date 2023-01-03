# Robust Policy Optimization (RPO)

## Overview

RPO leverages a method of perturbing the distribution representing actions. The goal is to encourage high-entropy actions and provide a better representation of the action space. The method consists of a simple modification on top of the objective of the PPO algorithm. Assuming PPO represents action with a parameterized distribution with mean and standard deviation (e.g., Gaussian). In the RPO algorithm, the mean of the action distribution is perturbed using a random number drawn from a Uniform distribution.

Original paper: 

* [Robust Policy Optimization in Deep Reinforcement Learning](https://arxiv.org/abs/2212.07536)

## Implemented Variants


| Variants Implemented      | Description |
| ----------- | ----------- |
| :material-github: [`rpo_continuous_action.py`](https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/rpo_continuous_action.py), :material-file-document: [docs](/rl-algorithms/rpo/#rpo_continuous_actionpy) | For classic control tasks like Gym `Pendulum-v1`, and dm_control. |

Below are our single-file implementations of RPO:

## `rpo_continuous_action.py`

`rpo_continuous_action.py` works with Gym (Gymnasium), dm_control, Mujoco environments with continuous action and vector observations.

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
# BipedalWalker-v3 experiment (hack)
poetry install
poetry run pip install box2d-py==2.3.5
python cleanrl/rpo_continuous_action.py --env-id BipedalWalker-v3
```


### Explanation of the logged metrics

See [related docs](/rl-algorithms/ppo/#explanation-of-the-logged-metrics) for `ppo.py`.

### Implementation details
[rpo_continuous_action.py](https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/rpo_continuous_action.py) has the same implementation details as `ppo_continuous_action.py` (see related [docs](/rl-algorithms/ppo/#ppo_continuous_actionpy)) but with a few lines of code differences.

```python hl_lines="30-34"
class Agent(nn.Module):
    def __init__(self, envs):
        super().__init__()
        self.critic = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, np.prod(envs.single_action_space.shape)), std=0.01),
        )
        self.actor_logstd = nn.Parameter(torch.zeros(1, np.prod(envs.single_action_space.shape)))

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        action_mean = self.actor_mean(x)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        if action is None:
            action = probs.sample()
        else: # new to RPO
            # sample again to add stochasticity, for the policy update
            z = torch.FloatTensor(action_mean.shape).uniform_(-self.rpo_alpha, self.rpo_alpha)
            action_mean = action_mean + z
            probs = Normal(action_mean, action_std)
        
```

### Experiment results

To run benchmark experiments, see  [benchmark/rpo.sh](https://github.com/vwxyzjn/cleanrl/blob/master/benchmark/rpo.sh). Specifically, execute the following command:
<script src="https://emgithub.com/embed.js?target=https%3A%2F%2Fgithub.com%2Fvwxyzjn%2Fcleanrl%2Fblob%2Fmaster%2Fbenchmark%2Frpo.sh%23L1-L6&style=github&showBorder=on&showLineNumbers=on&showFileMeta=on&showCopy=on"></script>

???+ note "Result tables, learning curves"

    === "dm_control"

        Results on all dm_control environments. The PPO and RPO run for 8M timesteps, and results are computed over 10 random seeds.

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

        Learning curves:
        ![](../rpo/dm_control_all_ppo_rpo_8M.png)
    
    
    === "MuJoCo v4"

        |                     | ppo_continuous_action_8M ({'tag': ['v1.0.0-13-gcbd83f6']})   | rpo_continuous_action ({'tag': ['pr-331']})   |
        |:--------------------|:-------------------------------------------------------------|:----------------------------------------------|
        | HumanoidStandup-v4  | 109325.87 ± 16161.71                                         | 150972.11 ± 6926.19                           |
        | Humanoid-v4         | 583.17 ± 27.88                                               | 799.44 ± 170.85                               |
        | InvertedPendulum-v4 | 888.83 ± 34.66                                               | 879.81 ± 35.52                                |
        | Walker2d-v4         | 2872.92 ± 690.53                                             | 3665.48 ± 278.61                              |

        Learning curves:
        ![](../rpo/mujoco_v4_part1.png)

        
        The following environments require tuning of `alpha` (Algorithm 1, line 13, paper: https://arxiv.org/pdf/2212.07536.pdf). As described in the paper, this variable should be tuned for environments tested. A larger value means more randomness, whereas a smaller value indicates less randomness. Some mujoco environments require a smaller `alpha=0.01` value to achieve a reasonable performance compared to `alpha=0.5` for the rest of the environments. This version (`alpha=0.01`) of runs is indicated as `rpo_continuous_action_alpha_0_01` in the table and learning curves.

        |                           | ppo_continuous_action_8M ({'tag': ['v1.0.0-13-gcbd83f6']})   | rpo_continuous_action_alpha_0_01 ({'tag': ['pr-331']})   |
        |:--------------------------|:-------------------------------------------------------------|:---------------------------------------------------------|
        | Ant-v4                    | 1824.17 ± 905.78                                             | 2702.91 ± 683.53                                         |
        | HalfCheetah-v4            | 2637.19 ± 1068.49                                            | 2716.51 ± 1314.93                                        |
        | Hopper-v4                 | 2741.42 ± 269.11                                             | 2334.22 ± 441.89                                         |
        | InvertedDoublePendulum-v4 | 5626.22 ± 289.23                                             | 5409.03 ± 318.68                                         |
        | Reacher-v4                | -4.65 ± 0.96                                                 | -3.93 ± 0.19                                             |
        | Swimmer-v4                | 124.88 ± 22.24                                               | 129.97 ± 12.02                                           |
        | Pusher-v4                 | -30.35 ± 6.43                                                | -31.48 ± 9.83                                            |

        Learning curves:
        ![](../rpo/mujoco_v4_part2.png)

        <!-- Tracked experiments: -->


        Results with `rpo_alpha=0.5` (not tuned) on the tuned environments:

        |                           | ppo_continuous_action_8M ({'tag': ['v1.0.0-13-gcbd83f6']})   | rpo_continuous_action ({'tag': ['pr-331']})   |
        |:--------------------------|:-------------------------------------------------------------|:----------------------------------------------|
        | Ant-v4                    | 1774.42 ± 819.08                                             | -7.99 ± 2.47                                  |
        | HalfCheetah-v4            | 2667.34 ± 1109.99                                            | 2163.57 ± 790.16                              |
        | Hopper-v4                 | 2761.77 ± 286.88                                             | 1557.18 ± 206.74                              |
        | InvertedDoublePendulum-v4 | 5644.00 ± 353.46                                             | 296.97 ± 15.95                                |
        | Reacher-v4                | -4.67 ± 0.88                                                 | -66.35 ± 0.66                                 |
        | Swimmer-v4                | 124.52 ± 22.10                                               | 117.82 ± 10.07                                |
        | Pusher-v4                 | -30.62 ± 6.80                                                | -276.32 ± 26.99                               |

        Learning curves:
        ![](../rpo/mujoco_v4_part2_0_5.png)


        Tracked experiments:

        <iframe loading="lazy" src="https://wandb.ai/openrlbenchmark/cleanrl/reports/RPO-on-Mujoco_v4-Part-1--VmlldzozMjU3ODIz" style="height:500px;width:100%">
        <iframe loading="lazy" src="https://wandb.ai/openrlbenchmark/cleanrl/reports/RPO-on-Mujoco_v4-Part-2--VmlldzozMjU3OTM4" style="height:500px;width:100%">       
        <iframe loading="lazy" src="https://wandb.ai/openrlbenchmark/cleanrl/reports/RPO-alpha-0-5-on-Mujoco_v4-Part-2--VmlldzozMjU4MTM1" style="height:500px;width:100%">

 
    
    === "MuJoCo v2"

        |                     | ppo_continuous_action_8M ({'tag': ['v1.0.0-13-gcbd83f6']})   | rpo_continuous_action ({'tag': ['pr-331']})   |
        |:--------------------|:-------------------------------------------------------------|:----------------------------------------------|
        | HumanoidStandup-v2  | 109118.07 ± 19422.20                                         | 156848.90 ± 11414.50                          |
        | Humanoid-v2         | 588.22 ± 43.80                                               | 717.37 ± 97.18                                |
        | InvertedPendulum-v2 | 867.64 ± 19.97                                               | 866.60 ± 27.06                                |
        | Walker2d-v2         | 3220.99 ± 923.84                                             | 4150.51 ± 348.03                              |

        Learning curves:
        ![](../rpo/mujoco_v2_part1.png)

        Tracked experiments:

        <iframe loading="lazy" src="https://wandb.ai/openrlbenchmark/cleanrl/reports/RPO-on-Mujoco_v2-Part-1--VmlldzozMjU4Mjc5" style="border:none;height:1024px;width:100%">

        The following environments require tuning of `alpha` (Algorithm 1, line 13, paper: https://arxiv.org/pdf/2212.07536.pdf). As described in the paper, this variable should be tuned for environments tested. A larger value means more randomness, whereas a smaller value indicates less randomness. Some mujoco environments require a smaller `alpha=0.01` value to achieve a reasonable performance compared to `alpha=0.5` for the rest of the environments. This version (`alpha=0.01`) of runs is indicated as `rpo_continuous_action_alpha_0_01` in the table and learning curves.

        |                           | ppo_continuous_action_8M ({'tag': ['v1.0.0-13-gcbd83f6']})   | rpo_continuous_action_alpha_0_01 ({'tag': ['pr-331']})   |
        |:--------------------------|:-------------------------------------------------------------|:---------------------------------------------------------|
        | Ant-v2                    | 2412.35 ± 949.44                                             | 3084.95 ± 759.51                                         |
        | HalfCheetah-v2            | 2717.27 ± 1269.85                                            | 2707.91 ± 1215.21                                        |
        | Hopper-v2                 | 2387.39 ± 645.41                                             | 2272.78 ± 588.66                                         |
        | InvertedDoublePendulum-v2 | 5630.91 ± 377.93                                             | 5661.29 ± 316.04                                         |
        | Reacher-v2                | -4.61 ± 0.53                                                 | -4.24 ± 0.25                                             |
        | Swimmer-v2                | 132.07 ± 9.92                                                | 141.37 ± 8.70                                            |
        | Pusher-v2                 | -33.93 ± 8.55                                                | -26.22 ± 2.52                                            |
                
        
        Learning curves:
        ![](../rpo/mujoco_v2_part2.png)

        Tracked experiments:

        <iframe loading="lazy" src="https://wandb.ai/openrlbenchmark/cleanrl/reports/RPO-on-Mujoco_v2-Part-2--VmlldzozMjU4MzI1" style="border:none;height:1024px;width:100%">

        Results with `rpo_alpha=0.5` (not tuned) on the tuned environments:

        |                           | ppo_continuous_action_8M ({'tag': ['v1.0.0-13-gcbd83f6']})   | rpo_continuous_action ({'tag': ['pr-331']})   |
        |:--------------------------|:-------------------------------------------------------------|:----------------------------------------------|
        | Ant-v2                    | 2495.65 ± 991.65                                             | -7.81 ± 3.57                                  |
        | HalfCheetah-v2            | 2722.03 ± 1231.28                                            | 2605.06 ± 1183.30                             |
        | Hopper-v2                 | 2356.83 ± 650.91                                             | 1609.79 ± 164.16                              |
        | InvertedDoublePendulum-v2 | 5675.31 ± 244.34                                             | 274.78 ± 16.40                                |
        | Reacher-v2                | -4.67 ± 0.48                                                 | -66.55 ± 0.20                                 |
        | Swimmer-v2                | 131.53 ± 9.94                                                | 114.34 ± 3.95                                 |
        | Pusher-v2                 | -33.46 ± 8.41                                                | -275.09 ± 15.65                               |

        Learning curves:
        ![](../rpo/mujoco_v2_part2_0_5.png)

        Tracked experiments:

        <iframe loading="lazy" src="https://wandb.ai/openrlbenchmark/cleanrl/reports/RPO-alpha-0-5-on-Mujoco_v2-Part-2--VmlldzozMjU4MzQ0" style="border:none;height:1024px;width:100%">
        

    
    === "Gym(Gymnasium)"

        Results on two continuous gym environments. The PPO and RPO run for 8M timesteps, and results are computed over 10 random seeds.


        |                  | ppo_continuous_action_8M ({'tag': ['v1.0.0-13-gcbd83f6']})   | rpo_continuous_action ({'tag': ['pr-331']})   |
        |:-----------------|:-------------------------------------------------------------|:----------------------------------------------|
        | Pendulum-v1      | -1141.98 ± 135.55                                            | -151.08 ± 3.73                                |
        | BipedalWalker-v3 | 172.12 ± 96.05                                               | 227.11 ± 18.23                                |

        Learning curves:
        ![](../rpo/gym.png)

        Tracked experiments:

        <iframe loading="lazy" src="https://wandb.ai/openrlbenchmark/cleanrl/reports/RPO-on-Gym-Gymnasium---VmlldzozMjU3NzUy" style="border:none;height:1024px;width:100%">


??? Failure case of `rpo_alpha=0.5`
    Overall, we observed that `rpo_alpha=0.5` is strictly equal to or better than the default PPO in 89% of environments tested (all 48/48 dm_control, 2/2 Gym, 7/11 mujoco_v4, 7/11 mujoco_v2. Total 64 out of 72 environments tested). 

    Here are the failure cases:
    `Mujoco v4: Ant-v4 InvertedDoublePendulum-v4 Reacher-v4 Pusher-v4`

    |                           | ppo_continuous_action_8M ({'tag': ['v1.0.0-13-gcbd83f6']})   | rpo_continuous_action ({'tag': ['pr-331']})   |
    |:--------------------------|:-------------------------------------------------------------|:----------------------------------------------|
    | Ant-v4                    | 1831.63 ± 867.71                                             | -10.43 ± 8.16                                 |
    | InvertedDoublePendulum-v4 | 5490.71 ± 261.50                                             | 303.36 ± 13.39                                |
    | Reacher-v4                | -4.58 ± 0.73                                                 | -66.62 ± 0.56                                 |
    | Pusher-v4                 | -30.63 ± 6.42                                                | -276.11 ± 26.52                               |

    Tracked experiments:

    <iframe loading="lazy" src="https://wandb.ai/openrlbenchmark/cleanrl/reports/RPO-alpha-0-5-s-failure-cases-on-Mujoco_v4--VmlldzozMjU4MTYy" style="border:none;height:1024px;width:100%">
    
    Learning curves:
    ![](../rpo/mujoco_v4_failure_0_5.png)


    `Mujoco v2: Ant-v2 InvertedDoublePendulum-v2 Reacher-v2 Pusher-v2`

    |                           | ppo_continuous_action_8M ({'tag': ['v1.0.0-13-gcbd83f6']})   | rpo_continuous_action ({'tag': ['pr-331']})   |
    |:--------------------------|:-------------------------------------------------------------|:----------------------------------------------|
    | Ant-v2                    | 2493.50 ± 993.24                                             | -7.26 ± 2.28                                  |
    | InvertedDoublePendulum-v2 | 5568.37 ± 401.65                                             | 278.94 ± 15.34                                |
    | Reacher-v2                | -4.62 ± 0.47                                                 | -66.61 ± 0.23                                 |
    | Pusher-v2                 | -33.51 ± 8.47                                                | -276.01 ± 15.93                               |


    Learning curves:
    ![](../rpo/mujoco_v2_failure_0_5.png)

    Tracked experiments:

    <iframe loading="lazy" src="https://wandb.ai/openrlbenchmark/cleanrl/reports/RPO-alpha-0-5-s-failure-cases-on-Mujoco_v2--VmlldzozMjU4MjQ1" style="border:none;height:1024px;width:100%">


    However, with tuning of `rpo_alpha` (`rpo_alpha=0.01` on failed cases) helps RPO to overcome the failure and RPO perform strictly equal to or better than the default PPO in all (100%) of tested environments.

