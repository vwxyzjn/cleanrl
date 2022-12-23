# Twin Delayed Deep Deterministic Policy Gradient (TD3)


## Overview

TD3 is a popular DRL algorithm for continuous control. It extends DDPG with three techniques: 1) Clipped Double Q-Learning, 2) Delayed Policy Updates, and 3) Target Policy Smoothing Regularization. With these three techniques TD3 shows significantly better performance compared to DDPG.


Original paper: 

* [Addressing Function Approximation Error in Actor-Critic Methods](https://arxiv.org/abs/1802.09477)

Reference resources:

* :material-github: [sfujim/TD3](https://github.com/sfujim/TD3)
* [Twin Delayed DDPG | Spinning Up in Deep RL](https://spinningup.openai.com/en/latest/algorithms/td3.html)

## Implemented Variants


| Variants Implemented      | Description |
| ----------- | ----------- |
| :material-github: [`td3_continuous_action.py`](https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/td3_continuous_action.py), :material-file-document: [docs](/rl-algorithms/td3/#td3_continuous_actionpy) | For continuous action space |


Below are our single-file implementations of TD3:

## `td3_continuous_action.py`

The [td3_continuous_action.py](https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/td3_continuous_action.py) has the following features:

* For continuous action space
* Works with the `Box` observation space of low-level features
* Works with the `Box` (continuous) action space

### Usage

```bash
poetry install
poetry install --with pybullet
python cleanrl/td3_continuous_action.py --help
python cleanrl/td3_continuous_action.py --env-id HopperBulletEnv-v0
poetry install --with mujoco # only works in Linux
python cleanrl/td3_continuous_action.py --env-id Hopper-v3
```

### Explanation of the logged metrics

Running `python cleanrl/td3_continuous_action.py` will automatically record various metrics such as various losses in Tensorboard. Below are the documentation for these metrics:

* `charts/episodic_return`: episodic return of the game
* `charts/SPS`: number of steps per second
* `losses/qf1_loss`: the MSE between the Q values at timestep $t$ and the target Q values at timestep $t+1$, which minimizes temporal difference. 
* `losses/actor_loss`: implemented as `-qf1(data.observations, actor(data.observations)).mean()`; it is the *negative* average Q values calculated based on the 1) observations and the 2) actions computed by the actor based on these observations. By minimizing `actor_loss`, the optimizer updates the actors parameter using the following gradient (Fujimoto et al., 2018, Algorithm 1)[^2]:

$$ \nabla_{\phi} J(\phi)=\left.N^{-1} \sum \nabla_{a} Q_{\theta_{1}}(s, a)\right|_{a=\pi_{\phi}(s)} \nabla_{\phi} \pi_{\phi}(s) $$

* `losses/qf1_values`: implemented as `qf1(data.observations, data.actions).view(-1); it is the average Q values of the sampled data in the replay buffer; useful when gauging if under or over esitmations happen


### Implementation details

Our [`td3_continuous_action.py`](https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/td3_continuous_action.py) is based on the [`TD3.py`](https://github.com/sfujim/TD3/blob/master/TD3.py) from :material-github: [sfujim/TD3](https://github.com/sfujim/TD3). Our [`td3_continuous_action.py`](https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/td3_continuous_action.py) presents the following implementation differences.

1. [`td3_continuous_action.py`](https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/td3_continuous_action.py) uses a two separate objects `qf1` and `qf2` to represents the two Q functions in the Clipped Double Q-learning architecture, whereas  [`TD3.py`](https://github.com/sfujim/TD3/blob/master/TD3.py)  (Fujimoto et al., 2018)[^2] uses a single `Critic` class that contains both Q networks. That said, these two implementations are virtually the same.

2. [`td3_continuous_action.py`](https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/td3_continuous_action.py) also adds support for handling continuous environments where the lower and higher bounds of the action space are not $[-1,1]$, or are asymmetric.
The case where the bounds are not $[-1,1]$ is handled in [`TD3.py`](https://github.com/sfujim/TD3/blob/385b33ac7de4767bab17eb02ade4a268d3e4e24f/TD3.py#L28) (Fujimoto et al., 2018)[^2] as follows:
```python
class Actor(nn.Module):

    ...

	def forward(self, state):
		a = F.relu(self.l1(state))
		a = F.relu(self.l2(a))
		return self.max_action * torch.tanh(self.l3(a)) # Scale from [-1,1] to [-action_high, action_high]
```
 On the other hand, in [`CleanRL's td3_continuous_action.py`](https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/td3_continuous_action.py), the mean and the scale of the action space are computed as `action_bias` and `action_scale` respectively.
 Those scalars are in turn used to scale the output of a `tanh` activation function in the actor to the original action space range:
```python
class Actor(nn.Module):
    def __init__(self, env):
        ...
        # action rescaling
        self.register_buffer("action_scale", torch.FloatTensor((env.action_space.high - env.action_space.low) / 2.0))
        self.register_buffer("action_bias", torch.FloatTensor((env.action_space.high + env.action_space.low) / 2.0))

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = torch.tanh(self.fc_mu(x))
        return x * self.action_scale + self.action_bias # Scale from [-1,1] to [-action_low, action_high]
```

Additionally, when drawing exploration noise that is added to the actions produced by the actor, [`CleanRL's td3_continuous_action.py`](https://github.com/dosssman/cleanrl/blob/10b606e7bd9bd1b06e455e8ef542df2b7699a20c/cleanrl/td3_continuous_action.py#L180) centers the distribution the sampled from at `action_bias`, and the scale of the distribution is set to `action_scale * exploration_noise`.

???+ info

    Note that `Humanoid-v2`, `InvertedPendulum-v2`, `Pusher-v2` have action space bounds that are not the standard `[-1, 1]`. See below and :material-github: [PR #196](https://github.com/vwxyzjn/cleanrl/issues/196)

    ```
    Ant-v2 Observation space: Box(-inf, inf, (111,), float64) Action space: Box(-1.0, 1.0, (8,), float32)
    HalfCheetah-v2 Observation space: Box(-inf, inf, (17,), float64) Action space: Box(-1.0, 1.0, (6,), float32)
    Hopper-v2 Observation space: Box(-inf, inf, (11,), float64) Action space: Box(-1.0, 1.0, (3,), float32)
    Humanoid-v2 Observation space: Box(-inf, inf, (376,), float64) Action space: Box(-0.4, 0.4, (17,), float32)
    InvertedDoublePendulum-v2 Observation space: Box(-inf, inf, (11,), float64) Action space: Box(-1.0, 1.0, (1,), float32)
    InvertedPendulum-v2 Observation space: Box(-inf, inf, (4,), float64) Action space: Box(-3.0, 3.0, (1,), float32)
    Pusher-v2 Observation space: Box(-inf, inf, (23,), float64) Action space: Box(-2.0, 2.0, (7,), float32)
    Reacher-v2 Observation space: Box(-inf, inf, (11,), float64) Action space: Box(-1.0, 1.0, (2,), float32)
    Swimmer-v2 Observation space: Box(-inf, inf, (8,), float64) Action space: Box(-1.0, 1.0, (2,), float32)
    Walker2d-v2 Observation space: Box(-inf, inf, (17,), float64) Action space: Box(-1.0, 1.0, (6,), float32)
    ```

### Experiment results

To run benchmark experiments, see :material-github: [benchmark/td3.sh](https://github.com/vwxyzjn/cleanrl/blob/master/benchmark/td3.sh). Specifically, execute the following command:

<script src="https://emgithub.com/embed.js?target=https%3A%2F%2Fgithub.com%2Fvwxyzjn%2Fcleanrl%2Fblob%2Fmaster%2Fbenchmark%2Ftd3.sh%23L1-L7&style=github&type=code&showBorder=on&showLineNumbers=on&showFileMeta=on&showFullPath=on&showCopy=on"></script>


Below are the average episodic returns for [`td3_continuous_action.py`](https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/td3_continuous_action.py) (3 random seeds). To ensure the quality of the implementation, we compared the results against (Fujimoto et al., 2018)[^2].

| Environment      | [`td3_continuous_action.py`](https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/td3_continuous_action.py) | [`TD3.py`](https://github.com/sfujim/TD3/blob/master/TD3.py) (Fujimoto et al., 2018, Table 1)[^2]  |
| ----------- | ----------- | ----------- | 
| HalfCheetah      | 9449.94 ± 1586.49      |9636.95 ± 859.065  |
| Walker2d   | 3851.55 ± 335.29     |  4682.82 ± 539.64 | 
| Hopper   | 3162.21 ± 261.08        |  3564.07 ± 114.74 | 
| Humanoid |  5011.05 ± 254.89      |  not available | 
| Pusher |  -37.49 ± 10.22      |  not available | 
| InvertedPendulum |    996.81 ± 4.50    | 1000.00 ± 0.00  | 



???+ info

    Note that [`td3_continuous_action.py`](https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/td3_continuous_action.py) uses gym MuJoCo v2 environments while [`TD3.py`](https://github.com/sfujim/TD3/blob/master/TD3.py) (Fujimoto et al., 2018)[^2] uses the gym MuJoCo v1 environments. According to the :material-github: [openai/gym#834](https://github.com/openai/gym/pull/834), gym MuJoCo v2 environments should be equivalent to the gym MuJoCo v1 environments.

    Also note the performance of our `td3_continuous_action.py` seems to be worse than the reference implementation on Walker2d. This is likely due to :material-github: [openai/gym#938](https://github.com/openai/baselines/issues/938). We would have a hard time reproducing gym MuJoCo v1 environments because they have been long deprecated.

    One other thing could cause the performance difference: the original code reported the average episodic return using determinisitc evaluation (i.e., without exploration noise), see [`sfujim/TD3/main.py#L15-L32`](https://github.com/sfujim/TD3/blob/385b33ac7de4767bab17eb02ade4a268d3e4e24f/main.py#L15-L32), whereas we reported the episodic return during training and the policy gets updated between environments steps.

Learning curves:

<div class="grid-container">
<img src="../td3/HalfCheetah-v2.png">

<img src="../td3/Walker2d-v2.png">

<img src="../td3/Hopper-v2.png">

<img src="../td3/Humanoid-v2.png">

<img src="../td3/Pusher-v2.png">

<img src="../td3/InvertedPendulum-v2.png">
</div>


Tracked experiments and game play videos:

<iframe src="https://wandb.ai/openrlbenchmark/openrlbenchmark/reports/MuJoCo-CleanRL-s-TD3--VmlldzoxNjk4Mzk5" style="width:100%; height:500px" title="MuJoCo: CleanRL's TD3"></iframe>




## `td3_continuous_action_jax.py`

The [td3_continuous_action_jax.py](https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/td3_continuous_action_jax.py) has the following features:

* Uses [Jax](https://github.com/google/jax), [Flax](https://github.com/google/flax), and [Optax](https://github.com/deepmind/optax) instead of `torch`.  [td3_continuous_action_jax.py](https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/td3_continuous_action_jax.py) is roughly 2.5-4x faster than  [td3_continuous_action.py](https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/td3_continuous_action.py)
* For continuous action space
* Works with the `Box` observation space of low-level features
* Works with the `Box` (continuous) action space

### Usage

```bash
poetry install --with mujoco,jax
poetry run pip install --upgrade "jax[cuda]==0.3.17" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
poetry run python -c "import mujoco_py"
python cleanrl/td3_continuous_action_jax.py --help
poetry install --with mujoco # only works in Linux
python cleanrl/td3_continuous_action_jax.py --env-id Hopper-v3
```

### Explanation of the logged metrics

See [related docs](/rl-algorithms/td3/#explanation-of-the-logged-metrics) for `td3_continuous_action.py`.


### Implementation details

See [related docs](/rl-algorithms/td3/#implementation-details) for `td3_continuous_action.py`.


### Experiment results

To run benchmark experiments, see :material-github: [benchmark/td3.sh](https://github.com/vwxyzjn/cleanrl/blob/master/benchmark/td3.sh). Specifically, execute the following command:

<script src="https://emgithub.com/embed.js?target=https%3A%2F%2Fgithub.com%2Fvwxyzjn%2Fcleanrl%2Fblob%2Fmaster%2Fbenchmark%2Ftd3.sh%23L9-L16&style=github&type=code&showBorder=on&showLineNumbers=on&showFileMeta=on&showFullPath=on&showCopy=on"></script>

Below are the average episodic returns for [`td3_continuous_action.py`](https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/td3_continuous_action.py) (3 random seeds). To ensure the quality of the implementation, we compared the results against (Fujimoto et al., 2018)[^2].

| Environment      | [`td3_continuous_action_jax.py`](https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/td3_continuous_action_jax.py) (RTX 3060 TI) | [`td3_continuous_action_jax.py`](https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/td3_continuous_action_jax.py) (VM w/ TPU) | [`td3_continuous_action.py`](https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/td3_continuous_action.py) (RTX 3060 TI) | [`TD3.py`](https://github.com/sfujim/TD3/blob/master/TD3.py) (Fujimoto et al., 2018, Table 1)[^2]  |
| ----------- | ----------- | ----------- | ----------- |  ----------- | 
| HalfCheetah |  9408.62 ± 473.23  | 8948.33 ± 1196.87  | 9449.94 ± 1586.49      |9636.95 ± 859.065  |
| Walker2d |  3512.14 ± 1576.59 | 4107.63 ± 173.93  | 3851.55 ± 335.29     |  4682.82 ± 539.64 | 
| Hopper |   2898.62 ± 485.18 | 3151.80 ± 458.68 | 3162.21 ± 261.08        |  3564.07 ± 114.74 | 


???+ info

    Note that the experiments were conducted on different hardwares, so your mileage might vary. This inconsistency is because 1) re-running expeirments on the same hardware is computationally expensive and 2) requiring the same hardware is not inclusive nor feasible to other contributors who might have different hardwares.

    That said, we roughly expect to see a 2-4x speed improvement from using [`td3_continuous_action_jax.py`](https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/td3_continuous_action_jax.py) under the same hardware. And if you disable the `--capture-video` overhead, the speed improvement will be even higher.


Learning curves:

<div class="grid-container">
<img src="../td3-jax/HalfCheetah-v2.png">
<img src="../td3-jax/HalfCheetah-v2-time.png">

<img src="../td3-jax/Walker2d-v2.png">
<img src="../td3-jax/Walker2d-v2-time.png">

<img src="../td3-jax/Hopper-v2.png">
<img src="../td3-jax/Hopper-v2-time.png">
</div>



Tracked experiments and game play videos:

<iframe src="https://wandb.ai/openrlbenchmark/openrlbenchmark/reports/MuJoCo-CleanRL-s-TD3-JAX--VmlldzoyMzU1OTA4" style="width:100%; height:500px" title="MuJoCo: CleanRL's TD3 + JAX"></iframe>

[^1]:Lillicrap, T.P., Hunt, J.J., Pritzel, A., Heess, N.M., Erez, T., Tassa, Y., Silver, D., & Wierstra, D. (2016). Continuous control with deep reinforcement learning. CoRR, abs/1509.02971. https://arxiv.org/abs/1509.02971

[^2]:Fujimoto, S., Hoof, H.V., & Meger, D. (2018). Addressing Function Approximation Error in Actor-Critic Methods. ArXiv, abs/1802.09477. https://arxiv.org/abs/1802.09477
