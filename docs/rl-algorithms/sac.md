# Soft Actor-Critic (SAC)

## Overview

The Soft Actor-Critic (SAC) algorithm extends the DDPG algorithm by 1) using a stochastic policy, which in theory can express multi-modal optimal policies.
This also enables the use of 2) *entropy regularization* based on the stochastic policy's entropy. It serves as a built-in, state-dependent exploration heuristic for the agent, instead of relying on non-correlated noise processes as in [DDPG](/rl-algorithms/ddpg/), or [TD3](/rl-algorithms/td3/)
Additionally, it incorporates the 3) usage of two *Soft Q-network* to reduce the overestimation bias issue in Q-network-based methods.

Original papers:
The SAC algorithm's initial proposal, and later updates and improvements can be chronologically traced through the following publications:

* [Reinforcement Learning with Deep Energy-Based Policies](https://arxiv.org/abs/1702.08165)
* [Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor](https://arxiv.org/abs/1801.01290)
* [Composable Deep Reinforcement Learning for Robotic Manipulation](https://arxiv.org/abs/1803.06773)
* [Soft Actor-Critic Algorithms and Applications](https://arxiv.org/abs/1812.05905)
<!-- * [Soft Actor-Critic for Discrete Action Settings](https://arxiv.org/abs/1910.07207) No peer review, preprint only -->

Reference resources:

* :material-github: [haarnoja/sac](https://github.com/haarnoja/sac)
* :material-github: [openai/spinningup](https://github.com/openai/spinningup/tree/master/spinup/algos/tf1/sac)
* :material-github: [pranz24/pytorch-soft-actor-critic](https://github.com/pranz24/pytorch-soft-actor-critic)
* :material-github: [DLR-RM/stable-baselines3](https://github.com/DLR-RM/stable-baselines3/tree/master/stable_baselines3/sac)
* :material-github: [denisyarats/pytorch_sac](https://github.com/denisyarats/pytorch_sac)
* :material-github: [haarnoja/softqlearning](https://github.com/haarnoja/softqlearning)
* :material-github: [rail-berkeley/softlearning](https://github.com/rail-berkeley/softlearning)

| Variants Implemented      | Description |
| ----------- | ----------- |
| :material-github: [`sac_continuous_actions.py`](https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/sac_continuous_action.py), :material-file-document: [docs](/rl-algorithms/sac/#sac_continuous_actionpy) | For continuous action space |

Below is our single-file implementations of SAC:

## `sac_continuous_action.py`

The [sac_continuous_action.py](https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/sac_continuous_action.py) has the following features:

* For continuous action space.
* Works with the `Box` observation space of low-level features.
* Works with the `Box` (continuous) action space.
* Numerically stable stochastic policy based on :material-github: [openai/spinningup](https://github.com/openai/spinningup/tree/master/spinup/algos/tf1/sac) and [pranz24/pytorch-soft-actor-critic](https://github.com/pranz24/pytorch-soft-actor-critic) implementations.
* Supports automatic entropy coefficient $\alpha$ tuning, enabled by default.

### Usage

```bash
poetry install

# Pybullet
poetry install --with pybullet

## Default
python cleanrl/sac_continuous_action.py --env-id HopperBulletEnv-v0

## Without Automatic entropy coef. tuning
python cleanrl/sac_continuous_action.py --env-id HopperBulletEnv-v0 --autotune False --alpha 0.2
```

### Explanation of the logged metrics

Running python cleanrl/ddpg_continuous_action.py will automatically record various metrics such as actor or value losses in Tensorboard. Below is the documentation for these metrics:

* `charts/episodic_return`: the episodic return of the game during training

* `charts/SPS`: number of steps per second

* `losses/qf1_loss`, `losses/qf2_loss`: for each Soft Q-value network $Q_{\theta_i}$, $i \in \{1,2\}$, this metric holds the mean squared error (MSE) between the soft Q-value estimate $Q_{\theta_i}(s, a)$ and the *entropy regularized* Bellman update target estimated as $r_t + \gamma \, Q_{\theta_{i}^{'}}(s', a') + \alpha \, \mathcal{H} \big[ \pi(a' \vert s') \big]$.

More formally, the Soft Q-value loss for the $i$-th network is obtained by:

$$
    J(\theta^{Q}_{i}) = \mathbb{E}_{(s,a,r,s') \sim \mathcal{D}} \big[ (Q_{\theta_i}(s, a) - y)^2 \big]
$$

with the *entropy regularized*, *Soft Bellman update target*:
$$
    y = r(s, a) + \gamma ({\color{orange} \min_{\theta_{1,2}}Q_{\theta_i^{'}}(s',a')} - \alpha \, \text{log} \pi( \cdot \vert s'))
$$ where $a' \sim \pi( \cdot \vert s')$, $\text{log} \pi( \cdot \vert s')$ approximates the entropy of the policy, and $\mathcal{D}$ is the replay buffer storing samples of the agent during training.

Here, $\min_{\theta_{1,2}}Q_{\theta_i^{'}}(s',a')$ takes the minimum *Soft Q-value network* estimate between the two target Q-value networks $Q_{\theta_1^{'}}$ and $Q_{\theta_2^{'}}$ for the next state and action pair, so as to reduce over-estimation bias.

* `losses/qf_loss`: averages `losses/qf1_loss` and `losses/qf2_loss` for comparison with algorithms using a single Q-value network.

* `losses/actor_loss`: Given the stochastic nature of the policy in SAC, the actor (or policy) objective is formulated so as to maximize the likelihood of actions $a \sim \pi( \cdot \vert s)$ that would result in high Q-value estimate $Q(s, a)$. Additionally, the policy objective encourages the policy to maintain its entropy high enough to help explore, discover, and capture multi-modal optimal policies.

The policy's objective function can thus be defined as:

$$
    \text{max}_{\phi} \, J_{\pi}(\phi) = \mathbb{E}_{s \sim \mathcal{D}} \Big[ \text{min}_{i=1,2} Q_{\theta_i}(s, a) - \alpha \, \text{log}\pi_{\phi}(a \vert s) \Big]
$$

where the action is sampled using the reparameterization trick[^1]: $a = \mu_{\phi}(s) + \epsilon \, \sigma_{\phi}(s)$ with $\epsilon \sim \mathcal{N}(0, 1)$, $\text{log} \pi_{\phi}( \cdot \vert s')$ approximates the entropy of the policy, and $\mathcal{D}$ is the replay buffer storing samples of the agent during training.


* `losses/alpha`: $\alpha$ coefficient for *entropy regularization* of the policy.

* `losses/alpha_loss`: In the policy's objective defined above, the coefficient of the _entropy bonus_ $\alpha$ is kept fixed all across the training.
As suggested by the authors in Section 5 of the [_Soft Actor-Critic And Applications_](https://arxiv.org/abs/1812.05905) paper, the original purpose of augmenting the standard reward with the entropy of the policy is to *encourage exploration* of not well enough explored states (thus high entropy).
Conversely, for states where the policy has already learned a near-optimal policy, it would be preferable to reduce the entropy bonus of the policy, so that it does not _become sub-optimal due to the entropy maximization incentive_.

Therefore, having a fixed value for $\alpha$ does not fit this desideratum of matching the entropy bonus with the knowledge of the policy at an arbitrary state during its training.

To mitigate this, the authors proposed a method to dynamically adjust $\alpha$ as the policy is trained, which is as follows:

$$
    \alpha^{*}_t = \text{argmin}_{\alpha_t} \mathbb{E}_{a_t \sim \pi^{*}_t} \big[ -\alpha_t \, \text{log}\pi^{*}_t(a_t \vert s_t; \alpha_t) - \alpha_t \mathcal{H} \big],
$$

where $\mathcal{H}$ represents the _target entropy_, the desired lower bound for the expected entropy of the policy over the trajectory distribution induced by the latter.
As a heuristic for the _target entropy_, the authors use the dimension of the action space of the task.

### Implementation details
CleanRL's [`sac_continuous_action.py`](https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/sac_continuous_action.py) implementation is based on :material-github: [openai/spinningup](https://github.com/openai/spinningup/tree/master/spinup/algos/pytorch/sac).

1. [`sac_continuous_action.py`](https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/sac_continuous_action.py) uses a *numerically stable* estimation method for the standard deviation $\sigma$ of the policy, which squashes it into a range of reasonable values for a standard deviation:

    ```python hl_lines="1-2 19-21"
    LOG_STD_MAX = 2
    LOG_STD_MIN = -5

    class Actor(nn.Module):
        def __init__(self, env):
            super(Actor, self).__init__()
            self.fc1 = nn.Linear(np.array(env.single_observation_space.shape).prod(), 256)
            self.fc2 = nn.Linear(256, 256)
            self.fc_mean = nn.Linear(256, np.prod(env.single_action_space.shape))
            self.fc_logstd = nn.Linear(256, np.prod(env.single_action_space.shape))
            # action rescaling
            self.action_scale = torch.FloatTensor((env.action_space.high - env.action_space.low) / 2.0)
            self.action_bias = torch.FloatTensor((env.action_space.high + env.action_space.low) / 2.0)

        def forward(self, x):
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            mean = self.fc_mean(x)
            log_std = self.fc_logstd(x)
            log_std = torch.tanh(log_std)
            log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1)  # From SpinUp / Denis Yarats

            return mean, log_std

        def get_action(self, x):
            mean, log_std = self(x)
            std = log_std.exp()
            normal = torch.distributions.Normal(mean, std)
            x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
            y_t = torch.tanh(x_t)
            action = y_t * self.action_scale + self.action_bias
            log_prob = normal.log_prob(x_t)
            # Enforcing Action Bound
            log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
            log_prob = log_prob.sum(1, keepdim=True)
            mean = torch.tanh(mean) * self.action_scale + self.action_bias
            return action, log_prob, mean

        def to(self, device):
            self.action_scale = self.action_scale.to(device)
            self.action_bias = self.action_bias.to(device)
            return super(Actor, self).to(device)
    ```
    Note that unlike :material-github: [openai/spinningup](https://github.com/openai/spinningup/tree/master/spinup/algos/tf1/sac)'s implementation which uses `LOG_STD_MIN = -20`, CleanRL's uses `LOG_STD_MIN = -5` instead.

2. [`sac_continuous_action.py`](https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/sac_continuous_action.py) uses different learning rates for the policy and the Soft Q-value networks optimization.

    ```python
        parser.add_argument("--policy-lr", type=float, default=3e-4,
            help="the learning rate of the policy network optimizer")
        parser.add_argument("--q-lr", type=float, default=1e-3,
            help="the learning rate of the Q network network optimizer")
    ```
    while [openai/spinningup](https://github.com/openai/spinningup/blob/038665d62d569055401d91856abb287263096178/spinup/algos/tf1/sac/sac.py#L44)'s uses a single learning rate of `lr=1e-3` for both components.

    Note that in case it is used, the *automatic entropy coefficient* $\alpha$'s tuning shares the `q-lr` learning rate:
    ```python hl_lines="6"
        # Automatic entropy tuning
        if args.autotune:
            target_entropy = -torch.prod(torch.Tensor(envs.single_action_space.shape).to(device)).item()
            log_alpha = torch.zeros(1, requires_grad=True, device=device)
            alpha = log_alpha.exp().item()
            a_optimizer = optim.Adam([log_alpha], lr=args.q_lr)
        else:
            alpha = args.alpha
    ```

3. [`sac_continuous_action.py`](https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/sac_continuous_action.py) uses `--batch-size=256` while :material-github: [openai/spinningup](https://github.com/openai/spinningup/blob/038665d62d569055401d91856abb287263096178/spinup/algos/tf1/sac/sac.py#L44)'s uses `--batch-size=100` by default.

### Experiment results

To run benchmark experiments, see :material-github: [benchmark/sac.sh](https://github.com/vwxyzjn/cleanrl/blob/master/benchmark/sac.sh). Specifically, execute the following command:

<script src="https://emgithub.com/embed.js?target=https%3A%2F%2Fgithub.com%2Fvwxyzjn%2Fcleanrl%2Fblob%2F2e2dc9c6ede5e5e5df3eaea73c458bb9a83507d2%2Fbenchmark%2Fsac.sh%23L1-L7&style=github&showBorder=on&showLineNumbers=on&showFileMeta=on&showCopy=on"></script>

The table below compares the results of CleanRL's [`sac_continuous_action.py`](https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/sac_continuous_action.py) with the [latest published results](https://arxiv.org/abs/1812.05905) by the original authors of the SAC algorithm.

???+ info
    Note that the results table above references the *training episodic return* for [`sac_continuous_action.py`](https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/sac_continuous_action.py), the results of [Soft Actor-Critic Algorithms and Applications](https://arxiv.org/abs/1812.05905) reference *evaluation episodic return* obtained by running the policy in the deterministic mode.

| Environment      | [`sac_continuous_action.py`](https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/sac_continuous_action.py) |[SAC: Algorithms and Applications](https://arxiv.org/abs/1812.05905) @ 1M steps|
| --------------- | ------------------ | ---------------- |
| HalfCheetah-v2  | 10310.37 ± 1873.21       | ~11,250          |
| Walker2d-v2     | 4418.15 ± 592.82         | ~4,800           |
| Hopper-v2       | 2685.76 ± 762.16         | ~3,250           |


Learning curves:

<div class="grid-container">
    <img src="../sac/HalfCheetah-v2.png">
    <img src="../sac/Walker2d-v2.png">
    <img src="../sac/Hopper-v2.png">
</div>

<div></div>


Tracked experiments and game play videos:

<iframe src="https://wandb.ai/openrlbenchmark/openrlbenchmark/reports/MuJoCo-CleanRL-s-SAC--VmlldzoxNzI1NDM0" style="width:100%; height:1200px" title="MuJoCo: CleanRL's DDPG"></iframe>


[^1]:Diederik P Kingma, Max Welling (2016). Auto-Encoding Variational Bayes. ArXiv, abs/1312.6114. https://arxiv.org/abs/1312.6114