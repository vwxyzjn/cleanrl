# Deep Deterministic Policy Gradient (DDPG)


## Overview

DDPG is a popular DRL algorithm for continuous control. It runs reasonably fast by leveraging vector (parallel) environments and naturally works well with different action spaces, therefore supporting a variety of games. It also has good sample efficiency compared to algorithms such as DQN.


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

The [ddpg.py](https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/ddpg.py) has the following features:

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

Our [`ddpg_continuous_action.py`](https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/ddpg_continuous_action.py) is based on the [`OurDDPG.py`](https://github.com/sfujim/TD3/blob/master/OurDDPG.py) from :material-github: [sfujim/TD3](https://github.com/sfujim/TD3), which presents the the following implementation difference from (Lillicrap et al., 2016)[^1]:

1. [`ddpg_continuous_action.py`](https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/ddpg_continuous_action.py) uses a gaussian exploration noise $\mathcal{N}(0, 0.1)$, while (Lillicrap et al., 2016)[^1] uses Ornstein-Uhlenbeck process with $\theta=0.15$ and $\sigma=0.2$.

1. [`ddpg_continuous_action.py`](https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/ddpg_continuous_action.py) runs the experiments using the `openai/gym` MuJoCo environments, while (Lillicrap et al., 2016)[^1] uses their proprietary MuJoCo environments.

1. [`ddpg_continuous_action.py`](https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/ddpg_continuous_action.py) uses the following architecture:
    ```python
    class QNetwork(nn.Module):
        def __init__(self, env):
            super(QNetwork, self).__init__()
            self.fc1 = nn.Linear(np.array(env.single_observation_space.shape).prod() + np.prod(env.single_action_space.shape), 256)
            self.fc2 = nn.Linear(256, 256)
            self.fc3 = nn.Linear(256, 1)

        def forward(self, x, a):
            x = torch.cat([x, a], 1)
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = self.fc3(x)
            return x


    class Actor(nn.Module):
        def __init__(self, env):
            super(Actor, self).__init__()
            self.fc1 = nn.Linear(np.array(env.single_observation_space.shape).prod(), 256)
            self.fc2 = nn.Linear(256, 256)
            self.fc_mu = nn.Linear(256, np.prod(env.single_action_space.shape))

        def forward(self, x):
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            return torch.tanh(self.fc_mu(x))
    ```
    while (Lillicrap et al., 2016, see Appendix 7 EXPERIMENT DETAILS)[^1] uses the following architecture (difference highlighted):

    ```python hl_lines="4-6 9-11 19-21"
    class QNetwork(nn.Module):
        def __init__(self, env):
            super(QNetwork, self).__init__()
            self.fc1 = nn.Linear(np.array(env.single_observation_space.shape).prod(), 400)
            self.fc2 = nn.Linear(400 + np.prod(env.single_action_space.shape), 300)
            self.fc3 = nn.Linear(300, 1)

        def forward(self, x, a):
            x = F.relu(self.fc1(x))
            x = torch.cat([x, a], 1)
            x = F.relu(self.fc2(x))
            x = self.fc3(x)
            return x


    class Actor(nn.Module):
        def __init__(self, env):
            super(Actor, self).__init__()
            self.fc1 = nn.Linear(np.array(env.single_observation_space.shape).prod(), 400)
            self.fc2 = nn.Linear(400, 300)
            self.fc_mu = nn.Linear(300, np.prod(env.single_action_space.shape))

        def forward(self, x):
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            return torch.tanh(self.fc_mu(x))
    ```

1. [`ddpg_continuous_action.py`](https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/ddpg_continuous_action.py) uses the following learning rates:

    ```python
    q_optimizer = optim.Adam(list(qf1.parameters()), lr=3e-4)
    actor_optimizer = optim.Adam(list(actor.parameters()), lr=3e-4)
    ```
    while (Lillicrap et al., 2016, see Appendix 7 EXPERIMENT DETAILS)[^1] uses the following learning rates:

    ```python
    q_optimizer = optim.Adam(list(qf1.parameters()), lr=1e-4)
    actor_optimizer = optim.Adam(list(actor.parameters()), lr=1e-3)
    ```

1. [`ddpg_continuous_action.py`](https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/ddpg_continuous_action.py) uses `--batch-size=256 --tau=0.005`, while (Lillicrap et al., 2016, see Appendix 7 EXPERIMENT DETAILS)[^1] uses `--batch-size=64 --tau=0.001`

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

PR :material-github: [vwxyzjn/cleanrl#137](https://github.com/vwxyzjn/cleanrl/pull/137) tracks our effort to conduct experiments, and the reprodudction instructions can be found at :material-github: [vwxyzjn/cleanrl/benchmark/ddpg](https://github.com/vwxyzjn/cleanrl/tree/master/benchmark/ddpg).

Below are the average episodic returns for [`ddpg_continuous_action.py`](https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/ddpg_continuous_action.py) (3 random seeds). To ensure the quality of the implementation, we compared the results against (Fujimoto et al., 2018)[^2].

| Environment      | [`ddpg_continuous_action.py`](https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/ddpg_continuous_action.py) | [`OurDDPG.py`](https://github.com/sfujim/TD3/blob/master/OurDDPG.py) (Fujimoto et al., 2018, Table 1)[^2]  | [`DDPG.py`](https://github.com/sfujim/TD3/blob/master/DDPG.py) using settings from (Lillicrap et al., 2016)[^1] in (Fujimoto et al., 2018, Table 1)[^2]    |
| ----------- | ----------- | ----------- | ----------- |
| HalfCheetah      | 9260.485 ± 643.088      |8577.29  | 3305.60|
| Walker2d   | 1728.72 ± 758.33     |  3098.11 | 1843.85 |
| Hopper   | 1404.44 ± 544.78         |  1860.02 | 2020.46 |



???+ info

    Note that [`ddpg_continuous_action.py`](https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/ddpg_continuous_action.py) uses gym MuJoCo v2 environments while [`OurDDPG.py`](https://github.com/sfujim/TD3/blob/master/OurDDPG.py) (Fujimoto et al., 2018)[^2] uses the gym MuJoCo v1 environments. According to the :material-github: [openai/gym#834](https://github.com/openai/gym/pull/834), gym MuJoCo v2 environments should be equivalent to the gym MuJoCo v1 environments.

    Also note the performance of our `ddpg_continuous_action.py` seems to perform worse than the reference implementation on Walker2d and Hopper. This is likely due to :material-github: [openai/gym#938](https://github.com/openai/baselines/issues/938). We would have a hard time reproducing gym MuJoCo v1 environments because they have been long deprecated.

Learning curves:

<div class="grid-container">
<img src="../ddpg/HalfCheetah-v2.png">

<img src="../ddpg/Walker2d-v2.png">

<img src="../ddpg/Hopper-v2.png">
</div>


Tracked experiments and game play videos:

<iframe src="https://wandb.ai/openrlbenchmark/openrlbenchmark/reports/MuJoCo-CleanRL-s-DDPG--VmlldzoxNjkyMjc1" style="width:100%; height:500px" title="MuJoCo: CleanRL's DDPG"></iframe>


[^1]:Lillicrap, T.P., Hunt, J.J., Pritzel, A., Heess, N.M., Erez, T., Tassa, Y., Silver, D., & Wierstra, D. (2016). Continuous control with deep reinforcement learning. CoRR, abs/1509.02971. https://arxiv.org/abs/1509.02971

[^2]:Fujimoto, S., Hoof, H.V., & Meger, D. (2018). Addressing Function Approximation Error in Actor-Critic Methods. ArXiv, abs/1802.09477. https://arxiv.org/abs/1802.09477
