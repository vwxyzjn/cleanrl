# Soft Actor-Critic (SAC)

## Overivew

The Soft Actor-Critic (SAC) algorithm extends the DDPG algorithms by 1) using a stochastic policy, which in theory would to express multi-modal optimal policies.
This also enables the use of 2) *entropy regularization* based on the stochsatic policy's entropy. It serves as a built-in, state-dependent exploration heuristic for the agent, instead of relying on non-correlated noise processes as in DDPG **[TODO: link]**, or TD3 **[TODO: link]**
Additionally, it incorporates the 3) usage of two *Soft Q-network* to reduce the over-estimation bias issue in Q-network based methods.

Original papers:
The SAC algorithm introduction, and later and updates and improvements can be chronologically traced through the following publications:

* [Reinforcement Learning with Deep Energy-Based Policies](https://arxiv.org/abs/1702.08165)
* [Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor](https://arxiv.org/abs/1801.01290)
* [Composable Deep Reinforcement Learning for Robotic Manipulation](https://arxiv.org/abs/1803.06773)

* [Soft Actor-Critic Algorithms and Applications](https://arxiv.org/abs/1812.05905)
* [Soft Actor-Critic for Discrete Action Settings](https://arxiv.org/abs/1910.07207)

Reference resources:

* :material-github: [haarnoja/sac](https://github.com/haarnoja/sac)
* :material-github: [openai/spinningup](https://github.com/openai/spinningup/tree/master/spinup/algos/pytorch/sac)
* :material-github: [ikostrikov/pytorch-a2c-ppo-acktr-gail](https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail)
* :material-github: [denisyarats/pytorch_sac](https://github.com/denisyarats/pytorch_sac)
* :material-github: [DLR-RM/stable-baselines3](https://github.com/DLR-RM/stable-baselines3/tree/master/stable_baselines3/sac)
* :material-github: [haarnoja/softqlearning](https://github.com/haarnoja/softqlearning)
* :material-github: [rail-berkeley/softlearning](https://github.com/rail-berkeley/softlearning)

| Variants Implemented      | Description |
| ----------- | ----------- |
| :material-github: [`sac_continuous_actions.py`](https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/sac_continuous_action.py), :material-file-document: [docs](/rl-algorithms/sac/#sac_continuous_action_py) | For continuous action space |

Below is our single-file implementations of SAC:

## `sac_continuous_action.py`

The [sac_continuous_action.py](https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/sac_continuous_action.py) has the following features:

* For continuous action space.
* Works with the `Box` observation space of low-level features.
* Works with the `Box` (continuous) action space.
* Numerically stable stochastic policy based on :material-github: [openai/spinningup](https://github.com/openai/spinningup/tree/master/spinup/algos/pytorch/sac), :material-github: [ikostrikov/pytorch-a2c-ppo-acktr-gail](https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail) implementations.
* Supports automatic entropy coefficient $\alpha$ tuning, enabled by default.

### Usage

```bash
poetry install

# Pybullet
poetry install -E pybullet

## Default
python cleanrl/sac_continuous_action.py --env-id HopperBulletEnv-v0

## Automatic entropy coef. tuning
python cleanrl/sac_continuous_action.py --env-id HopperBulletEnv-v0 --autotune
```

### Explanation of the logged metrics

Running python cleanrl/ddpg_continuous_action.py will automatically record various metrics such as actor or value losses in Tensorboard. Below is the documentation for these metrics:

* `charts/episodic_return`: episodic return of the game

* `charts/SPS`: number of steps per second

* `losses/qf1_loss`, `losses/qf2_loss`: for each Soft Q-value network $Q_{\theta_i}$, $i \in \{1,2\}$, this metric holds the mean squared error (MSE) between the soft Q-value estimate $Q_{\theta_i}(s_{t}, a_t)$ and the *entropy regularized* Bellman update target estimated as $r_t + \gamma \, Q_{\theta_{i}^{'}}(s_{t+1}, a') + \alpha \, \mathcal{H} \big[ \pi(a' \vert s') \big]$.

More formally, the Soft Q-value loss for the $i$-th network is obtained by:

$$
    J(\theta^{Q}_{i}) = \mathbb{E}_{(s,a,r,s') \sim \mathcal{D}} \big[ (Q_{\theta_i}(s, a) - y)^2 \big]
$$

**[TODO: add the min over the target Q values]

with the *entropy regularized* Bellman update target
$$
    y = r + \gamma \, Q_{\theta_{i}^{'}}(s', a') + \alpha \, \mathcal{H} \big[ \pi(a' \vert s') \big]
$$, where $a' \sim \pi( \cdot \vert s')$, $a' \sim \pi( \cdot \vert s')$ represents the entropy of the policy, and $\mathcal{D}$ is the replay buffer storing samples of the agent during training.

* `losses/qf_loss`: averages `losses/qf1_loss` and `losses/qf2_loss` for comparison with algorithms using a single Q-value network.

* `losses/actor_loss`:

* `losses/alpha`: $\alpha$ coefficient for *entropy regularization* of the policy.

* `losses/alpha_loss`:

## Implementation details

**TODO**

## Experiment results

PR :material-github: [vwxyzjn/cleanrl#146](https://github.com/vwxyzjn/cleanrl/pull/146) tracks our effort to conduct experiments, and the reprodudction instructions can be found at :material-github: [vwxyzjn/cleanrl/benchmark/sac](https://github.com/vwxyzjn/cleanrl/tree/master/benchmark/sac).

Tracked experiments and game play videos:

