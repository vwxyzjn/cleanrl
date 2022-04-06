# Twin Delayed Deep Deterministic Policy Gradient (TD3)


## Overview

TD3 is a popular DRL algorithm for continuous control. It extends DQN to work with the continuous action space by introducing a deterministirc actor that directly outputs continuous actions. DDPG also combines techniques from DQN such as the replay buffer and target network.


Original paper: 

* [Addressing Function Approximation Error in Actor-Critic Methods](https://arxiv.org/abs/1802.09477)

Reference resources:

* :material-github: [sfujim/TD3](https://github.com/sfujim/TD3)
* [Twin Delayed DDPG | Spinning Up in Deep RL](https://spinningup.openai.com/en/latest/algorithms/td3.html)

## Implemented Variants


| Variants Implemented      | Description |
| ----------- | ----------- |
| :material-github: [`td3_continuous_action.py`](https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/td3_continuous_action.py), :material-file-document: [docs](/rl-algorithms/td3/#td3_continuous_actionpy) | For continuous action space |


Below are our single-file implementations of PPO:

## `td3_continuous_action.py`

The [td3_continuous_action.py](https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/td3_continuous_action.py) has the following features:

* For continuous action space
* Works with the `Box` observation space of low-level features
* Works with the `Box` (continuous) action space

### Usage

```bash
poetry install
poetry install -E pybullet
python cleanrl/td3_continuous_action.py --help
python cleanrl/td3_continuous_action.py --env-id HopperBulletEnv-v0
poetry install -E mujoco # only works in Linux
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
1. [`td3_continuous_action.py`](https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/td3_continuous_action.py) rescales the gradient so that the norm of the parameters does not exceed `0.5` like done in PPO (:material-github: [ppo2/model.py#L102-L108](https://github.com/openai/baselines/blob/ea25b9e8b234e6ee1bca43083f8f3cf974143998/baselines/ppo2/model.py#L102-L108)). 


### Experiment results

PR :material-github: [vwxyzjn/cleanrl#141](https://github.com/vwxyzjn/cleanrl/pull/141) tracks our effort to conduct experiments, and the reprodudction instructions can be found at :material-github: [vwxyzjn/cleanrl/benchmark/td3](https://github.com/vwxyzjn/cleanrl/tree/master/benchmark/td3).

Below are the average episodic returns for [`td3_continuous_action.py`](https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/td3_continuous_action.py) (3 random seeds). To ensure the quality of the implementation, we compared the results against (Fujimoto et al., 2018)[^2].

| Environment      | [`td3_continuous_action.py`](https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/td3_continuous_action.py) | [`TD3.py`](https://github.com/sfujim/TD3/blob/master/TD3.py) (Fujimoto et al., 2018, Table 1)[^2]  |
| ----------- | ----------- | ----------- | 
| HalfCheetah      | 9391.52 ± 448.54      |9636.95 ± 859.065  |
| Walker2d   | 3895.80 ± 333.89     |  4682.82 ± 539.64 | 
| Hopper   | 3379.25 ± 200.22         |  3564.07 ± 114.74 | 



???+ info

    Note that [`td3_continuous_action.py`](https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/td3_continuous_action.py) uses gym MuJoCo v2 environments while [`TD3.py`](https://github.com/sfujim/TD3/blob/master/TD3.py) (Fujimoto et al., 2018)[^2] uses the gym MuJoCo v1 environments. According to the :material-github: [openai/gym#834](https://github.com/openai/gym/pull/834), gym MuJoCo v2 environments should be equivalent to the gym MuJoCo v1 environments.

    Also note the performance of our `td3_continuous_action.py` seems to perform worse than the reference implementation on Walker2d. This is likely due to :material-github: [openai/gym#938](https://github.com/openai/baselines/issues/938). We would have a hard time reproducing gym MuJoCo v1 environments because they have been long deprecated.

    One other thing could cause the performance difference: the original code reported the average episodic return using determinisitc evaluation (i.e., without exploration noise), see [`sfujim/TD3/main.py#L15-L32`](https://github.com/sfujim/TD3/blob/385b33ac7de4767bab17eb02ade4a268d3e4e24f/main.py#L15-L32), whereas we reported the episodic return during training and the policy gets updated between environments steps.

Learning curves:

<div class="grid-container">
<img src="../td3/HalfCheetah-v2.png">

<img src="../td3/Walker2d-v2.png">

<img src="../td3/Hopper-v2.png">
</div>


Tracked experiments and game play videos:

<iframe src="https://wandb.ai/openrlbenchmark/openrlbenchmark/reports/MuJoCo-CleanRL-s-TD3--VmlldzoxNjk4Mzk5" style="width:100%; height:500px" title="MuJoCo: CleanRL's DDPG"></iframe>


[^1]:Lillicrap, T.P., Hunt, J.J., Pritzel, A., Heess, N.M., Erez, T., Tassa, Y., Silver, D., & Wierstra, D. (2016). Continuous control with deep reinforcement learning. CoRR, abs/1509.02971. https://arxiv.org/abs/1509.02971

[^2]:Fujimoto, S., Hoof, H.V., & Meger, D. (2018). Addressing Function Approximation Error in Actor-Critic Methods. ArXiv, abs/1802.09477. https://arxiv.org/abs/1802.09477
