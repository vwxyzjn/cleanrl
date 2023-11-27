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
| :material-github: [`ppo_atari.py`](https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/ppo_atari.py), :material-file-document: [docs](/rl-algorithms/ppo/#ppo_ataripy) |  For Atari games. It uses convolutional layers and common atari-based pre-processing techniques. |
| :material-github: [`ppo_continuous_action.py`](https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/ppo_continuous_action.py), :material-file-document: [docs](/rl-algorithms/ppo/#ppo_continuous_actionpy) | For continuous action space. Also implemented Mujoco-specific code-level optimizations. |
| :material-github: [`ppo_atari_lstm.py`](https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/ppo_atari_lstm.py), :material-file-document: [docs](/rl-algorithms/ppo/#ppo_atari_lstmpy) | For Atari games using LSTM without stacked frames. |
| :material-github: [`ppo_atari_envpool.py`](https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/ppo_atari_envpool.py), :material-file-document: [docs](/rl-algorithms/ppo/#ppo_atari_envpoolpy) | Uses the blazing fast Envpool Atari vectorized environment. |
| :material-github: [`ppo_atari_envpool_xla_jax.py`](https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/ppo_atari_envpool_xla_jax.py),  :material-file-document: [docs](/rl-algorithms/ppo/#ppo_atari_envpool_xla_jaxpy) | Uses the blazing fast Envpool Atari vectorized environment with EnvPool's XLA interface and JAX. |
| :material-github: [`ppo_atari_envpool_xla_jax_scan.py`](https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/ppo_atari_envpool_xla_jax_scan.py),  :material-file-document: [docs](/rl-algorithms/ppo/#ppo_atari_envpool_xla_jax_scanpy) | Uses native `jax.scan` as opposed to python loops for faster compilation time. |
| :material-github: [`ppo_procgen.py`](https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/ppo_procgen.py), :material-file-document: [docs](/rl-algorithms/ppo/#ppo_procgenpy) | For the procgen environments. |
| :material-github: [`ppo_atari_multigpu.py`](https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/ppo_atari_multigpu.py),  :material-file-document: [docs](/rl-algorithms/ppo/#ppo_atari_multigpupy)| For Atari environments leveraging multi-GPUs. |
| :material-github: [`ppo_pettingzoo_ma_atari.py`](https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/ppo_pettingzoo_ma_atari.py),  :material-file-document: [docs](/rl-algorithms/ppo/#ppo_pettingzoo_ma_ataripy)| For Pettingzoo's multi-agent Atari environments. |

Below are our single-file implementations of PPO:

## `ppo.py`

The [ppo.py](https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/ppo.py) has the following features:

* Works with the `Box` observation space of low-level features
* Works with the `Discrete` action space
* Works with envs like `CartPole-v1`

### Usage

=== "poetry"

    ```bash
    poetry install
    poetry run python cleanrl/ppo.py --help
    poetry run python cleanrl/ppo.py --env-id CartPole-v1
    ```

=== "pip"

    ```bash
    python cleanrl/ppo.py --help
    python cleanrl/ppo.py --env-id CartPole-v1
    ```

### Explanation of the logged metrics

Running `python cleanrl/ppo.py` will automatically record various metrics such as actor or value losses in Tensorboard. Below is the documentation for these metrics:

* `charts/episodic_return`: episodic return of the game
* `charts/episodic_length`: episodic length of the game
* `charts/SPS`: number of steps per second
* `charts/learning_rate`: the current learning rate
* `losses/value_loss`: the mean value loss across all data points
* `losses/policy_loss`: the mean policy loss across all data points
* `losses/entropy`: the mean entropy value across all data points
* `losses/old_approx_kl`: the approximate Kullback–Leibler divergence, measured by `(-logratio).mean()`, which corresponds to the k1 estimator in John Schulman’s blog post on [approximating KL](http://joschu.net/blog/kl-approx.html)
* `losses/approx_kl`: better alternative to `olad_approx_kl` measured by `(logratio.exp() - 1) - logratio`, which corresponds to the k3 estimator in [approximating KL](http://joschu.net/blog/kl-approx.html)
* `losses/clipfrac`: the fraction of the training data that triggered the clipped objective
* `losses/explained_variance`: the explained variance for the value function


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

To run benchmark experiments, see :material-github: [benchmark/ppo.sh](https://github.com/vwxyzjn/cleanrl/blob/master/benchmark/ppo.sh). Specifically, execute the following command:

``` title="benchmark/ppo.sh" linenums="1"
--8<-- "benchmark/ppo.sh:3:8"
```

Below are the average episodic returns for `ppo.py`. To ensure the quality of the implementation, we compared the results against `openai/baselies`' PPO.

| Environment      | `ppo.py` | `openai/baselies`' PPO (Huang et al., 2022)[^1]
| ----------- | ----------- | ----------- |
| CartPole-v1      | 490.04 ± 6.12     |497.54 ± 4.02  |
| Acrobot-v1       | -86.36 ± 1.32     |  -81.82 ± 5.58 |
| MountainCar-v0   | -200.00 ± 0.00         | -200.00 ± 0.00 |


Learning curves:

``` title="benchmark/ppo_plot.sh" linenums="1"
--8<-- "benchmark/ppo_plot.sh::9"
```


<img loading="lazy" src="https://huggingface.co/datasets/cleanrl/benchmark/resolve/main/benchmark/pr-424/ppo.png">
<img loading="lazy" src="https://huggingface.co/datasets/cleanrl/benchmark/resolve/main/benchmark/pr-424/ppo-time.png">


Tracked experiments and game play videos:

<iframe loading="lazy" src="https://wandb.ai/openrlbenchmark/openrlbenchmark/reports/Classic-Control-CleanRL-s-PPO--VmlldzoxODU5MDY1" style="width:100%; height:500px" title="Classic-Control-CleanRL-s-PPO"></iframe>

### Video tutorial

If you'd like to learn `ppo.py` in-depth, consider checking out the following video tutorial:


<div style="text-align: center;"><iframe width="560" height="315" src="https://www.youtube.com/embed/MEt6rrxH8W4" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe></div>


## `ppo_atari.py`

The [ppo_atari.py](https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/ppo_atari.py) has the following features:

* For Atari games. It uses convolutional layers and common atari-based pre-processing techniques.
* Works with the Atari's pixel `Box` observation space of shape `(210, 160, 3)`
* Works with the `Discrete` action space


### Usage

=== "poetry"

    ```bash
    poetry install -E atari
    poetry run python cleanrl/ppo_atari.py --help
    poetry run python cleanrl/ppo_atari.py --env-id BreakoutNoFrameskip-v4
    ```

=== "pip"

    ```bash
    pip install -r requirements/requirements-atari.txt
    python cleanrl/ppo_atari.py --help
    python cleanrl/ppo_atari.py --env-id BreakoutNoFrameskip-v4
    ```

### Explanation of the logged metrics

See [related docs](/rl-algorithms/ppo/#explanation-of-the-logged-metrics) for `ppo.py`.

### Implementation details

[ppo_atari.py](https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/ppo_atari.py) is based on the "9 Atari implementation details" in [The 37 Implementation Details of Proximal Policy Optimization](https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/), which are as follows:

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



To run benchmark experiments, see :material-github: [benchmark/ppo.sh](https://github.com/vwxyzjn/cleanrl/blob/master/benchmark/ppo.sh). Specifically, execute the following command:

``` title="benchmark/ppo.sh" linenums="1"
--8<-- "benchmark/ppo.sh:14:19"
```


Below are the average episodic returns for `ppo_atari.py`. To ensure the quality of the implementation, we compared the results against `openai/baselies`' PPO.

| Environment      | `ppo_atari.py` | `openai/baselies`' PPO (Huang et al., 2022)[^1]
| ----------- | ----------- | ----------- |
| BreakoutNoFrameskip-v4      | 414.66 ± 28.09     | 406.57 ± 31.554  |
| PongNoFrameskip-v4   | 20.36 ± 0.20    |  20.512 ± 0.50 |
| BeamRiderNoFrameskip-v4   | 1915.93 ± 484.58         | 2642.97 ± 670.37 |


Learning curves:

``` title="benchmark/ppo_plot.sh" linenums="1"
--8<-- "benchmark/ppo_plot.sh:11:19"
```

<img loading="lazy" src="https://huggingface.co/datasets/cleanrl/benchmark/resolve/main/benchmark/pr-424/ppo_atari.png">
<img loading="lazy" src="https://huggingface.co/datasets/cleanrl/benchmark/resolve/main/benchmark/pr-424/ppo_atari-time.png">


Tracked experiments and game play videos:

<iframe loading="lazy" src="https://wandb.ai/openrlbenchmark/openrlbenchmark/reports/Atari-CleanRL-s-PPO--VmlldzoxNjk3NjYy" style="width:100%; height:500px" title="Atari-CleanRL-s-PPO"></iframe>

### Video tutorial

If you'd like to learn `ppo_atari.py` in-depth, consider checking out the following video tutorial:


<div style="text-align: center;"><iframe width="560" height="315" src="https://www.youtube.com/embed/05RMTj-2K_Y" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe></div>


## `ppo_continuous_action.py`

The [ppo_continuous_action.py](https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/ppo_continuous_action.py) has the following features:

* For continuous action space. Also implemented Mujoco-specific code-level optimizations
* Works with the `Box` observation space of low-level features
* Works with the `Box` (continuous) action space
* adding experimental support for [Gymnasium](https://gymnasium.farama.org/)
* 🧪 support `dm_control` environments via [Shimmy](https://github.com/Farama-Foundation/Shimmy)


???+ warning

    We are now recommending users to use [`rpo_continuous_action.py`](https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/rpo_continuous_action.py) instead of [`ppo_continuous_action.py`](https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/ppo_continuous_action.py) because `rpo_continuous_action.py` empirically performs better than `ppo_continuous_action.py` in 93% of the environments we tested. Please see [experiment results](/rl-algorithms/rpo/#experiment-results) for detailed analysis.

### Usage

=== "poetry"

    ```bash
    # mujoco v4 environments
    poetry install -E mujoco
    python cleanrl/ppo_continuous_action.py --help
    python cleanrl/ppo_continuous_action.py --env-id Hopper-v4
    # dm_control environments
    poetry install -E "mujoco dm_control"
    python cleanrl/ppo_continuous_action.py --env-id dm_control/cartpole-balance-v0
    ```

=== "pip"

    ```bash
    pip install -r requirements/requirements-mujoco.txt
    python cleanrl/ppo_continuous_action.py --help
    python cleanrl/ppo_continuous_action.py --env-id Hopper-v4
    pip install -r requirements/requirements-dm_control.txt
    python cleanrl/ppo_continuous_action.py --env-id dm_control/cartpole-balance-v0
    ```

???+ warning "dm_control installation issue"

    If you run into error like `AttributeError: 'GLFWContext' object has no attribute '_context'` in Linux, it's because the rendering dependencies are not installed properly. To fix it, try running

    ```
    sudo apt-get update && sudo apt-get -y install libgl1-mesa-glx libosmesa6 libglfw3 
    ```

    See [https://github.com/deepmind/dm_control#rendering](https://github.com/deepmind/dm_control#rendering) for more detail.


### Explanation of the logged metrics

See [related docs](/rl-algorithms/ppo/#explanation-of-the-logged-metrics) for `ppo.py`.

### Implementation details

[ppo_continuous_action.py](https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/ppo_continuous_action.py) is based on the "9 details for continuous action domains (e.g. Mujoco)" in [The 37 Implementation Details of Proximal Policy Optimization](https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/), which are as follows:


1. Continuous actions via normal distributions (:material-github: [common/distributions.py#L103-L104](https://github.com/openai/baselines/blob/9b68103b737ac46bc201dfb3121cfa5df2127e53/baselines/common/distributions.py#L103-L104))
2. State-independent log standard deviation (:material-github: [common/distributions.py#L104](https://github.com/openai/baselines/blob/9b68103b737ac46bc201dfb3121cfa5df2127e53/baselines/common/distributions.py#L104))
3. Independent action components (:material-github: [common/distributions.py#L238-L246](https://github.com/openai/baselines/blob/9b68103b737ac46bc201dfb3121cfa5df2127e53/baselines/common/distributions.py#L238-L246))
4. Separate MLP networks for policy and value functions (:material-github: [common/policies.py#L160](https://github.com/openai/baselines/blob/ea25b9e8b234e6ee1bca43083f8f3cf974143998/baselines/common/policies.py#L160), [baselines/common/models.py#L75-L103](https://github.com/openai/baselines/blob/ea25b9e8b234e6ee1bca43083f8f3cf974143998/baselines/common/models.py#L75-L103)
5. Handling of action clipping to valid range and storage (:material-github: [common/cmd_util.py#L99-L100](https://github.com/openai/baselines/blob/ea25b9e8b234e6ee1bca43083f8f3cf974143998/baselines/common/cmd_util.py#L99-L100)) 
6. Normalization of Observation (:material-github: [common/vec_env/vec_normalize.py#L4](https://github.com/openai/baselines/blob/ea25b9e8b234e6ee1bca43083f8f3cf974143998/baselines/common/vec_env/vec_normalize.py#L4))
7. Observation Clipping (:material-github: [common/vec_env/vec_normalize.py#L39](https://github.com/openai/baselines/blob/ea25b9e8b234e6ee1bca43083f8f3cf974143998/baselines/common/vec_env/vec_normalize.py#L39))
8. Reward Scaling (:material-github: [common/vec_env/vec_normalize.py#L28](https://github.com/openai/baselines/blob/ea25b9e8b234e6ee1bca43083f8f3cf974143998/baselines/common/vec_env/vec_normalize.py#L28))
9. Reward Clipping (:material-github: [common/vec_env/vec_normalize.py#L32](https://github.com/openai/baselines/blob/ea25b9e8b234e6ee1bca43083f8f3cf974143998/baselines/common/vec_env/vec_normalize.py#L32))



### Experiment results

To run benchmark experiments, see :material-github: [benchmark/ppo.sh](https://github.com/vwxyzjn/cleanrl/blob/master/benchmark/ppo.sh). Specifically, execute the following command:


MuJoCo v4

``` title="benchmark/ppo.sh" linenums="1"
--8<-- "benchmark/ppo.sh:25:30"
```

{!benchmark/ppo_continuous_action.md!}

Learning curves:

``` title="benchmark/ppo_plot.sh" linenums="1"
--8<-- "benchmark/ppo_plot.sh:11:19"
```

<img loading="lazy" src="https://huggingface.co/datasets/cleanrl/benchmark/resolve/main/benchmark/pr-424/ppo_continuous_action.png">
<img loading="lazy" src="https://huggingface.co/datasets/cleanrl/benchmark/resolve/main/benchmark/pr-424/ppo_continuous_action-time.png">

Tracked experiments and game play videos:

<iframe loading="lazy" src="https://wandb.ai/costa-huang/cleanRL/reports/MuJoCo-v4-CleanRL-s-PPO--VmlldzozMTIxOTI5" style="width:100%; height:500px" title="MuJoCo-CleanRL-s-PPO"></iframe>



``` title="benchmark/ppo.sh" linenums="1"
--8<-- "benchmark/ppo.sh:36:41"
```

Below are the average episodic returns for `ppo_continuous_action.py` in `dm_control` environments.

|                                       | ppo_continuous_action ({'tag': ['v1.0.0-13-gcbd83f6']})   |
|:--------------------------------------|:----------------------------------------------------------|
| dm_control/acrobot-swingup-v0         | 27.84 ± 9.25                                              |
| dm_control/acrobot-swingup_sparse-v0  | 1.60 ± 1.17                                               |
| dm_control/ball_in_cup-catch-v0       | 900.78 ± 5.26                                             |
| dm_control/cartpole-balance-v0        | 855.47 ± 22.06                                            |
| dm_control/cartpole-balance_sparse-v0 | 999.93 ± 0.10                                             |
| dm_control/cartpole-swingup-v0        | 640.86 ± 11.44                                            |
| dm_control/cartpole-swingup_sparse-v0 | 51.34 ± 58.35                                             |
| dm_control/cartpole-two_poles-v0      | 203.86 ± 11.84                                            |
| dm_control/cartpole-three_poles-v0    | 164.59 ± 3.23                                             |
| dm_control/cheetah-run-v0             | 432.56 ± 82.54                                            |
| dm_control/dog-stand-v0               | 307.79 ± 46.26                                            |
| dm_control/dog-walk-v0                | 120.05 ± 8.80                                             |
| dm_control/dog-trot-v0                | 76.56 ± 6.44                                              |
| dm_control/dog-run-v0                 | 60.25 ± 1.33                                              |
| dm_control/dog-fetch-v0               | 34.26 ± 2.24                                              |
| dm_control/finger-spin-v0             | 590.49 ± 171.09                                           |
| dm_control/finger-turn_easy-v0        | 180.42 ± 44.91                                            |
| dm_control/finger-turn_hard-v0        | 61.40 ± 9.59                                              |
| dm_control/fish-upright-v0            | 516.21 ± 59.52                                            |
| dm_control/fish-swim-v0               | 87.91 ± 6.83                                              |
| dm_control/hopper-stand-v0            | 2.72 ± 1.72                                               |
| dm_control/hopper-hop-v0              | 0.52 ± 0.48                                               |
| dm_control/humanoid-stand-v0          | 6.59 ± 0.18                                               |
| dm_control/humanoid-walk-v0           | 1.73 ± 0.03                                               |
| dm_control/humanoid-run-v0            | 1.11 ± 0.04                                               |
| dm_control/humanoid-run_pure_state-v0 | 0.98 ± 0.03                                               |
| dm_control/humanoid_CMU-stand-v0      | 4.79 ± 0.18                                               |
| dm_control/humanoid_CMU-run-v0        | 0.88 ± 0.05                                               |
| dm_control/manipulator-bring_ball-v0  | 0.50 ± 0.29                                               |
| dm_control/manipulator-bring_peg-v0   | 1.80 ± 1.58                                               |
| dm_control/manipulator-insert_ball-v0 | 35.50 ± 13.04                                             |
| dm_control/manipulator-insert_peg-v0  | 60.40 ± 21.76                                             |
| dm_control/pendulum-swingup-v0        | 242.81 ± 245.95                                           |
| dm_control/point_mass-easy-v0         | 273.95 ± 362.28                                           |
| dm_control/point_mass-hard-v0         | 143.25 ± 38.12                                            |
| dm_control/quadruped-walk-v0          | 239.03 ± 66.17                                            |
| dm_control/quadruped-run-v0           | 180.44 ± 32.91                                            |
| dm_control/quadruped-escape-v0        | 28.92 ± 11.21                                             |
| dm_control/quadruped-fetch-v0         | 193.97 ± 22.20                                            |
| dm_control/reacher-easy-v0            | 626.28 ± 15.51                                            |
| dm_control/reacher-hard-v0            | 443.80 ± 9.64                                             |
| dm_control/stacker-stack_2-v0         | 75.68 ± 4.83                                              |
| dm_control/stacker-stack_4-v0         | 68.02 ± 4.02                                              |
| dm_control/swimmer-swimmer6-v0        | 158.19 ± 10.22                                            |
| dm_control/swimmer-swimmer15-v0       | 131.94 ± 0.88                                             |
| dm_control/walker-stand-v0            | 564.46 ± 235.22                                           |
| dm_control/walker-walk-v0             | 392.51 ± 56.25                                            |
| dm_control/walker-run-v0              | 125.92 ± 10.01                                            |

Note that the dm_control/lqr-lqr_2_1-v0 dm_control/lqr-lqr_6_2-v0 environments are never terminated or truncated. See https://wandb.ai/openrlbenchmark/cleanrl/runs/3tm00923 and https://wandb.ai/openrlbenchmark/cleanrl/runs/1z9us07j as an example.

Learning curves:

![](../ppo/ppo_continuous_action_gymnasium_dm_control.png)

Tracked experiments and game play videos:

<iframe loading="lazy" src="https://wandb.ai/openrlbenchmark/openrlbenchmark/reports/dm_control-CleanRL-s-PPO-part-1---VmlldzozMTI2MjE2" style="width:100%; height:500px" title="dm_control-CleanRL-s-PPO-part-1"></iframe>
<iframe loading="lazy" src="https://wandb.ai/openrlbenchmark/openrlbenchmark/reports/dm_control-CleanRL-s-PPO-part-2---VmlldzozMTI2MjI1" style="width:100%; height:500px" title="dm_control-CleanRL-s-PPO-part-2"></iframe>


???+ info

    In the gymnasium environments, we use the v4 mujoco environments, which roughly results in the same performance as the v2 mujoco environments.

    ![](../ppo/ppo_continuous_action_v2_vs_v4.png)


### Video tutorial

If you'd like to learn `ppo_continuous_action.py` in-depth, consider checking out the following video tutorial:


<div style="text-align: center;"><iframe width="560" height="315" src="https://www.youtube.com/embed/BvZvx7ENZBw" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe></div>


## `ppo_atari_lstm.py`

The [ppo_atari_lstm.py](https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/ppo_atari_lstm.py) has the following features:

* For Atari games using LSTM without stacked frames. It uses convolutional layers and common atari-based pre-processing techniques.
* Works with the Atari's pixel `Box` observation space of shape `(210, 160, 3)`
* Works with the `Discrete` action space

### Usage


=== "poetry"

    ```bash
    poetry install -E atari
    poetry run python cleanrl/ppo_atari_lstm.py --help
    poetry run python cleanrl/ppo_atari_lstm.py --env-id BreakoutNoFrameskip-v4
    ```

=== "pip"

    ```bash
    pip install -r requirements/requirements-atari.txt
    python cleanrl/ppo_atari_lstm.py --help
    python cleanrl/ppo_atari_lstm.py --env-id BreakoutNoFrameskip-v4
    ```


### Explanation of the logged metrics

See [related docs](/rl-algorithms/ppo/#explanation-of-the-logged-metrics) for `ppo.py`.

### Implementation details

[ppo_atari_lstm.py](https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/ppo_atari_lstm.py) is based on the "5 LSTM implementation details" in [The 37 Implementation Details of Proximal Policy Optimization](https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/), which are as follows:

1. Layer initialization for LSTM layers (:material-github: [a2c/utils.py#L84-L86](https://github.com/openai/baselines/blob/ea25b9e8b234e6ee1bca43083f8f3cf974143998/baselines/a2c/utils.py#L84-L86))
2. Initialize the LSTM states to be zeros (:material-github: [common/models.py#L179](https://github.com/openai/baselines/blob/ea25b9e8b234e6ee1bca43083f8f3cf974143998/baselines/common/models.py#L179))
3. Reset LSTM states at the end of the episode (:material-github: [common/models.py#L141](https://github.com/openai/baselines/blob/ea25b9e8b234e6ee1bca43083f8f3cf974143998/baselines/common/models.py#L141))
4. Prepare sequential rollouts in mini-batches (:material-github: [a2c/utils.py#L81](https://github.com/openai/baselines/blob/ea25b9e8b234e6ee1bca43083f8f3cf974143998/baselines/a2c/utils.py#L81))
5. Reconstruct LSTM states during training (:material-github: [a2c/utils.py#L81](https://github.com/openai/baselines/blob/ea25b9e8b234e6ee1bca43083f8f3cf974143998/baselines/a2c/utils.py#L81))

To help test out the memory, we remove the 4 stacked frames from the observation (i.e., using `env = gym.wrappers.FrameStack(env, 1)` instead of `env = gym.wrappers.FrameStack(env, 4)` like in `ppo_atari.py` )



### Experiment results

To run benchmark experiments, see :material-github: [benchmark/ppo.sh](https://github.com/vwxyzjn/cleanrl/blob/master/benchmark/ppo.sh). Specifically, execute the following command:

``` title="benchmark/ppo.sh" linenums="1"
--8<-- "benchmark/ppo.sh:47:52"
```

Below are the average episodic returns for `ppo_atari_lstm.py`. To ensure the quality of the implementation, we compared the results against `openai/baselies`' PPO.


| Environment      | `ppo_atari_lstm.py` | `openai/baselies`' PPO (Huang et al., 2022)[^1]
| ----------- | ----------- | ----------- |
| BreakoutNoFrameskip-v4      | 128.92 ± 31.10    | 138.98 ± 50.76  |
| PongNoFrameskip-v4   | 19.78 ± 1.58    | 19.79 ± 0.67 |
| BeamRiderNoFrameskip-v4   | 1536.20 ± 612.21         | 1591.68 ± 372.95|


Learning curves:

``` title="benchmark/ppo_plot.sh" linenums="1"
--8<-- "benchmark/ppo_plot.sh:11:19"
```

<img loading="lazy" src="https://huggingface.co/datasets/cleanrl/benchmark/resolve/main/benchmark/pr-424/ppo_atari_lstm.png">
<img loading="lazy" src="https://huggingface.co/datasets/cleanrl/benchmark/resolve/main/benchmark/pr-424/ppo_atari_lstm-time.png">

Tracked experiments and game play videos:

<iframe loading="lazy" src="https://wandb.ai/openrlbenchmark/openrlbenchmark/reports/Atari-CleanRL-s-PPO-LSTM--VmlldzoxODcxMzE4" style="width:100%; height:500px" title="Atari-CleanRL-s-PPO-LSTM"></iframe>



## `ppo_atari_envpool.py`

The [ppo_atari_envpool.py](https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/ppo_atari_envpool.py) has the following features:

* Uses the blazing fast [Envpool](https://github.com/sail-sg/envpool) vectorized environment.
* For Atari games. It uses convolutional layers and common atari-based pre-processing techniques.
* Works with the Atari's pixel `Box` observation space of shape `(210, 160, 3)`
* Works with the `Discrete` action space

???+ warning

    Note that `ppo_atari_envpool.py` does not work in Windows :fontawesome-brands-windows: and MacOs :fontawesome-brands-apple:. See envpool's built wheels here: [https://pypi.org/project/envpool/#files](https://pypi.org/project/envpool/#files)

???+ bug

    EnvPool's vectorized environment **does not behave the same** as gym's vectorized environment, which causes a compatibility bug in our PPO implementation. When an action $a$ results in an episode termination or truncation, the environment generates $s_{last}$ as the terminated or truncated state; we then use $s_{new}$ to denote the initial state of the new episodes. Here is how the bahviors differ:

    * Under the vectorized environment of `envpool<=0.6.4`, the `obs` in `obs, reward, done, info = env.step(action)` is the truncated state $s_{last}$
    * Under the vectorized environment of `gym==0.23.1`, the `obs` in `obs, reward, done, info = env.step(action)` is the initial state $s_{new}$.

    This causes the $s_{last}$ to be off by one. 
    See [:material-github: sail-sg/envpool#194](https://github.com/sail-sg/envpool/issues/194) for more detail. However, it does not seem to impact performance, so we take a note here and await for the upstream fix.


### Usage

=== "poetry"

    ```bash
    poetry install -E envpool
    poetry run python cleanrl/ppo_atari_envpool.py --help
    poetry run python cleanrl/ppo_atari_envpool.py --env-id Breakout-v5
    ```

=== "pip"

    ```bash
    pip install -r requirements/requirements-envpool.txt
    python cleanrl/ppo_atari_envpool.py --help
    python cleanrl/ppo_atari_envpool.py --env-id Breakout-v5
    ```

### Explanation of the logged metrics

See [related docs](/rl-algorithms/ppo/#explanation-of-the-logged-metrics) for `ppo.py`.

### Implementation details

[ppo_atari_envpool.py](https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/ppo_atari_envpool.py) uses a customized `RecordEpisodeStatistics` to work with envpool but has the same other implementation details as `ppo_atari.py` (see [related docs](/rl-algorithms/ppo/#implementation-details_1)).

### Experiment results

To run benchmark experiments, see :material-github: [benchmark/ppo.sh](https://github.com/vwxyzjn/cleanrl/blob/master/benchmark/ppo.sh). Specifically, execute the following command:


``` title="benchmark/ppo.sh" linenums="1"
--8<-- "benchmark/ppo.sh:58:63"
```

{!benchmark/ppo_atari_envpool.md!}


Learning curves:

``` title="benchmark/ppo_plot.sh" linenums="1"
--8<-- "benchmark/ppo_plot.sh:51:62"
```

<img loading="lazy" src="https://huggingface.co/datasets/cleanrl/benchmark/resolve/main/benchmark/pr-424/ppo_atari_envpool.png">
<img loading="lazy" src="https://huggingface.co/datasets/cleanrl/benchmark/resolve/main/benchmark/pr-424/ppo_atari_envpool-time.png">


Tracked experiments and game play videos:

<iframe loading="lazy" src="https://wandb.ai/openrlbenchmark/openrlbenchmark/reports/Atari-CleanRL-s-PPO-Envpool--VmlldzoxODcxMzI3" style="width:100%; height:500px" title="Atari-CleanRL-s-PPO-Envpool"></iframe>


## `ppo_atari_envpool_xla_jax.py`

The [ppo_atari_envpool_xla_jax.py](https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/ppo_atari_envpool_xla_jax.py) has the following features:

* Uses the blazing fast [Envpool](https://github.com/sail-sg/envpool) vectorized environment.
    * Uses EnvPool's experimental [XLA interface](https://envpool.readthedocs.io/en/latest/content/xla_interface.html).
* Uses [Jax](https://github.com/google/jax), [Flax](https://github.com/google/flax), and [Optax](https://github.com/deepmind/optax) instead of `torch`.  
* For Atari games. It uses convolutional layers and common atari-based pre-processing techniques.
* Works with the Atari's pixel `Box` observation space of shape `(210, 160, 3)`
* Works with the `Discrete` action space

???+ warning

    Note that `ppo_atari_envpool_xla_jax.py` does not work in Windows :fontawesome-brands-windows: and MacOs :fontawesome-brands-apple:. See envpool's built wheels here: [https://pypi.org/project/envpool/#files](https://pypi.org/project/envpool/#files)


???+ bug

    EnvPool's vectorized environment **does not behave the same** as gym's vectorized environment, which causes a compatibility bug in our PPO implementation. When an action $a$ results in an episode termination or truncation, the environment generates $s_{last}$ as the terminated or truncated state; we then use $s_{new}$ to denote the initial state of the new episodes. Here is how the bahviors differ:

    * Under the vectorized environment of `envpool<=0.6.4`, the `obs` in `obs, reward, done, info = env.step(action)` is the truncated state $s_{last}$
    * Under the vectorized environment of `gym==0.23.1`, the `obs` in `obs, reward, done, info = env.step(action)` is the initial state $s_{new}$.

    This causes the $s_{last}$ to be off by one. 
    See [:material-github: sail-sg/envpool#194](https://github.com/sail-sg/envpool/issues/194) for more detail. However, it does not seem to impact performance, so we take a note here and await for the upstream fix.



### Usage

=== "poetry"

    ```bash
    poetry install -E "envpool jax"
    poetry run pip install --upgrade "jax[cuda11_cudnn82]==0.4.8" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
    poetry run python cleanrl/ppo_atari_envpool_xla_jax.py --help
    poetry run python cleanrl/ppo_atari_envpool_xla_jax.py --env-id Breakout-v5
    ```

=== "pip"

    ```bash
    pip install -r requirements/requirements-envpool.txt
    pip install -r requirements/requirements-jax.txt
    pip install --upgrade "jax[cuda]==0.3.17" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
    python cleanrl/ppo_atari_envpool_xla_jax.py --help
    python cleanrl/ppo_atari_envpool_xla_jax.py --env-id Breakout-v5
    ```

### Explanation of the logged metrics

See [related docs](/rl-algorithms/ppo/#explanation-of-the-logged-metrics) for `ppo.py`. In [ppo_atari_envpool_xla_jax.py](https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/ppo_atari_envpool_xla_jax.py) we omit logging `losses/old_approx_kl` and `losses/clipfrac` for brevity.

Additionally, we record the following metric:

* `charts/avg_episodic_return`: the average value of the *latest* episodic returns of `args.num_envs=8` envs
* `charts/avg_episodic_length`: the average value of the *latest* episodic lengths of `args.num_envs=8` envs

???+ info

    Note that we use `charts/avg_episodic_return` and `charts/avg_episodic_length` in place of `charts/episodic_return` and `charts/episodic_length` because under the EnvPool's XLA interface, we can only record fixed-shape metrics where as there could be a variable number of raw episodic returns / lengths. To resolve this challenge, we create variables (e.g., `returned_episode_returns`, `returned_episode_lengths`) to keep track of the *latest* episodic returns / lengths of each environment and average them for reporting purposes.

### Implementation details

[ppo_atari_envpool_xla_jax.py](https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/ppo_atari_envpool_xla_jax.py) uses the same other implementation details as `ppo_atari.py` (see [related docs](/rl-algorithms/ppo/#implementation-details_1)), with two differences

1. [ppo_atari_envpool_xla_jax.py](https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/ppo_atari_envpool_xla_jax.py) does not use the value function clipping by default, because there is no sufficient evidence that value function clipping actually improves performance.
1. [ppo_atari_envpool_xla_jax.py](https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/ppo_atari_envpool_xla_jax.py) uses a customized `EpisodeStatistics` to record episode statistics instead of the `RecordEpisodeStatistics` used in other variants. `RecordEpisodeStatistics` is a *stateful* python wrapper which is incompatible with EnvPool's *stateless* XLA interface. To address this issue, we used a `EpisodeStatistics` dataclass and simply implement the logic of `RecordEpisodeStatistics`. However, `EpisodeStatistics` comes with a major limitation: its storage has a fixed shape and can only record the *latest* episodic return of the sub-environments. Furthermore, the default episodic return values in `EpisodeStatistics` are set to zeros, which does not necessarily correspond to the episodic return obtained by a random policy. For example, we would report `charts/avg_episodic_return=0` for `Pong-v5`, even if they should have been `charts/avg_episodic_return=-21`. That said, this issue goes away as soon as the sub-environments finished their first episodes, therefore not impacting the reported results.


???+ info
    
    We benchmarked the PPO implementation w/ and w/o value function clipping, finding no significant difference in performance, which is consistent with the findings in Andrychowicz et al.[^2]. See the related report [part 1](https://wandb.ai/costa-huang/cleanRL/reports/CleanRL-PPO-JAX-EnvPool-s-XLA-w-and-w-o-value-loss-clipping-vs-openai-baselins-PPO-part-1---VmlldzoyNzQ3MzQ1) and [part 2](https://wandb.ai/costa-huang/cleanRL/reports/CleanRL-PPO-JAX-EnvPool-s-XLA-w-and-w-o-value-loss-clipping-vs-openai-baselins-PPO-part-2---VmlldzoyNzQ3MzUw).

    ![](../ppo/ppo_atari_envpool_xla_jax/hns_ppo_vs_baselines2.svg)


### Experiment results

To run benchmark experiments, see :material-github: [benchmark/ppo.sh](https://github.com/vwxyzjn/cleanrl/blob/master/benchmark/ppo.sh). Specifically, execute the following command:


``` title="benchmark/ppo.sh" linenums="1"
--8<-- "benchmark/ppo.sh:69:74"
```


{!benchmark/ppo_atari_envpool_xla_jax.md!}


Learning curves:

``` title="benchmark/ppo_plot.sh" linenums="1"
--8<-- "benchmark/ppo_plot.sh:64:85"
```

<img loading="lazy" src="https://huggingface.co/datasets/cleanrl/benchmark/resolve/main/benchmark/pr-424/ppo_atari_envpool_xla_jax_sample_walltime_efficiency.png">
<img loading="lazy" src="https://huggingface.co/datasets/cleanrl/benchmark/resolve/main/benchmark/pr-424/ppo_atari_envpool_xla_jax.png">
<img loading="lazy" src="https://huggingface.co/datasets/cleanrl/benchmark/resolve/main/benchmark/pr-424/ppo_atari_envpool_xla_jax-time.png">



???+ info

    Note the original openai/baselines uses `atari-py==0.2.6` which hangs on `gym.make("DefenderNoFrameskip-v4")` and does not support SurroundNoFrameskip-v4 (see issue [:material-github: openai/atari-py#73](https://github.com/openai/atari-py/issues/73)). To get results on these environments, we use `gym==0.23.1 ale-py==0.7.4 "AutoROM[accept-rom-license]==0.4.2` and [manually register `SurroundNoFrameskip-v4` in our fork](https://github.com/vwxyzjn/baselines/blob/e2cb1c938a62fa8d7fe98187246cde08dfd57bd1/baselines/common/register_all_atari_envs.py#L2). 


Median Human Normalized Score (HNS) compared to SEEDRL's R2D2 (data available [here](https://github.com/google-research/seed_rl/blob/66e8890261f09d0355e8bf5f1c5e41968ca9f02b/docs/seed_r2d2_atari_graphs.csv)). 

![](../ppo/ppo_atari_envpool_xla_jax/hns_ppo_vs_r2d2.svg)

???+ info

    Note the SEEDRL's R2D2's median HNS data does not include learning curves for `Defender` and `Surround` (see [google-research/seed_rl#78](https://github.com/google-research/seed_rl/issues/78)). Also note the SEEDRL's R2D2 uses slightly different Atari preprocessing than our `ppo_atari_envpool_xla_jax.py`, so we may be comparing apples and oranges; however, the results are still informative at the scale of 57 Atari games — we would be at least comparing similar apples.



Tracked experiments and game play videos:

<iframe loading="lazy" src="https://wandb.ai/openrlbenchmark/openrlbenchmark/reports/Atari-CleanRL-PPO-JAX-EnvPool-s-XLA-vs-openai-baselins-PPO-part-1---VmlldzoyNjE2ODMz" style="width:100%; height:500px" title="Atari-CleanRL-s-PPO-Envpool-vs-openai-baselines-Part-1"></iframe>

<iframe loading="lazy" src="https://wandb.ai/openrlbenchmark/openrlbenchmark/reports/Atari-CleanRL-PPO-JAX-EnvPool-s-XLA-vs-openai-baselins-PPO-part-2---VmlldzoyNjE2ODM1" style="width:100%; height:500px" title="Atari-CleanRL-s-PPO-Envpool-vs-openai-baselines-Part-2"></iframe>



## `ppo_atari_envpool_xla_jax_scan.py`

The [ppo_atari_envpool_xla_jax_scan.py](https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/ppo_atari_envpool_xla_jax_scan.py) has the following features:

* Replaces python loops in `compute_gae`, `update_ppo`, and `rollout` functions of [ppo_atari_envpool_xla_jax.py](/rl-algorithms/ppo/#ppo_atari_envpool_xla_jaxpy) with native `jax.scan`
* Warnings and caveats from [ppo_atari_envpool_xla_jax.py](/rl-algorithms/ppo/#ppo_atari_envpool_xla_jaxpy) also apply here

### Usage

=== "poetry"

    ```bash
    poetry install -E "envpool jax"
    poetry run pip install --upgrade "jax[cuda11_cudnn82]==0.4.8" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
    poetry run python cleanrl/ppo_atari_envpool_xla_jax_scan.py --help
    poetry run python cleanrl/ppo_atari_envpool_xla_jax_scan.py --env-id Breakout-v5
    ```

=== "pip"

    ```bash
    pip install -r requirements/requirements-envpool.txt
    pip install -r requirements/requirements-jax.txt
    pip install --upgrade "jax[cuda]==0.3.17" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
    python cleanrl/ppo_atari_envpool_xla_jax_scan.py --help
    python cleanrl/ppo_atari_envpool_xla_jax_scan.py --env-id Breakout-v5
    ```

### Explanation of the logged metrics

See [related docs](/rl-algorithms/ppo/#explanation-of-the-logged-metrics) for `ppo.py`. The metrics are the same as those in [ppo_atari_envpool_xla_jax.py](https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/ppo_atari_envpool_xla_jax.py).

### Implementation details

[ppo_atari_envpool_xla_jax_scan.py](https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/ppo_atari_envpool_xla_jax_scan.py) is a clone of [ppo_atari_envpool_xla_jax.py](https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/ppo_atari_envpool_xla_jax.py) that replaces the python loops with native `jax.scan`.

### Experiment results

To run benchmark experiments, see :material-github: [benchmark/ppo.sh](https://github.com/vwxyzjn/cleanrl/blob/master/benchmark/ppo.sh). Specifically, execute the following command:


``` title="benchmark/ppo.sh" linenums="1"
--8<-- "benchmark/ppo.sh:80:85"
```


{!benchmark/ppo_atari_envpool_xla_jax_scan.md!}


Learning curves:

``` title="benchmark/ppo_plot.sh" linenums="1"
--8<-- "benchmark/ppo_plot.sh:87:96"
```

<img loading="lazy" src="https://huggingface.co/datasets/cleanrl/benchmark/resolve/main/benchmark/pr-424/ppo_atari_envpool_xla_jax_scan.png">
<img loading="lazy" src="https://huggingface.co/datasets/cleanrl/benchmark/resolve/main/benchmark/pr-424/ppo_atari_envpool_xla_jax_scan-time.png">

Learning curves:

???+ info

    The trainig time of this variant and that of [ppo_atari_envpool_xla_jax.py](https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/ppo_atari_envpool_xla_jax.py) are very similar but the compilation time is reduced significantly (see [vwxyzjn/cleanrl#328](https://github.com/vwxyzjn/cleanrl/pull/328#issuecomment-1340474894)). Note that the hardware also affects the speed in the learning curve below. Runs from [`costa-huang`](https://github.com/vwxyzjn/) (red) are slower from those of [`51616`](https://github.com/51616/) (blue and orange) because of hardware differences.

    ![](../ppo/ppo_atari_envpool_xla_jax_scan/compare.png)
    ![](../ppo/ppo_atari_envpool_xla_jax_scan/compare-time.png)


Tracked experiments:

<iframe src="https://wandb.ai/openrlbenchmark/openrlbenchmark/reports/Regression-Report-ppo_atari_envpool_xla_jax_scan--VmlldzozMTk2MzM2" style="width:100%; height:500px" title="Atari-CleanRL-s-PPO-Envpool-Jax-scan"></iframe>


## `ppo_procgen.py`

The [ppo_procgen.py](https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/ppo_procgen.py) has the following features:

* For the procgen environments
* Uses IMPALA-style neural network
* Works with the `Discrete` action space


### Usage

=== "poetry"

    ```bash
    poetry install -E procgen
    poetry run python cleanrl/ppo_procgen.py --help
    poetry run python cleanrl/ppo_procgen.py --env-id starpilot
    ```

=== "pip"

    ```bash
    pip install -r requirements/requirements-procgen.txt
    python cleanrl/ppo_procgen.py --help
    python cleanrl/ppo_procgen.py --env-id starpilot
    ```

### Explanation of the logged metrics

See [related docs](/rl-algorithms/ppo/#explanation-of-the-logged-metrics) for `ppo.py`.

### Implementation details

[ppo_procgen.py](https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/ppo_procgen.py) is based on the details in "Appendix" in [The 37 Implementation Details of Proximal Policy Optimization](https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/), which are as follows:

1. IMPALA-style Neural Network (:material-github: [common/models.py#L28](https://github.com/openai/baselines/blob/ea25b9e8b234e6ee1bca43083f8f3cf974143998/baselines/common/models.py#L28))
1. Use the same `gamma` parameter in the `NormalizeReward` wrapper. Note that the original implementation from [openai/train-procgen](https://github.com/openai/train-procgen) uses the default `gamma=0.99` in [the `VecNormalize` wrapper](https://github.com/openai/train-procgen/blob/1a2ae2194a61f76a733a39339530401c024c3ad8/train_procgen/train.py#L43) but `gamma=0.999` as PPO's parameter. The mismatch between the `gamma`s is technically incorrect. See [#209](https://github.com/vwxyzjn/cleanrl/pull/209)

### Experiment results



To run benchmark experiments, see :material-github: [benchmark/ppo.sh](https://github.com/vwxyzjn/cleanrl/blob/master/benchmark/ppo.sh). Specifically, execute the following command:


``` title="benchmark/ppo.sh" linenums="1"
--8<-- "benchmark/ppo.sh:91:100"
```

We try to match the default setting in [openai/train-procgen](https://github.com/openai/train-procgen) except that we use the `easy` distribution mode and `total_timesteps=25e6` to save compute. Notice [openai/train-procgen](https://github.com/openai/train-procgen) has the following settings:

1. Learning rate annealing is turned off by default
1. Reward scaling and reward clipping is used


Below are the average episodic returns for `ppo_procgen.py`. To ensure the quality of the implementation, we compared the results against `openai/baselies`' PPO.

| Environment      | `ppo_procgen.py` | `openai/baselies`' PPO (Huang et al., 2022)[^1]
| ----------- | ----------- | ----------- |
| StarPilot (easy)      | 30.99 ± 1.96      | 33.97 ± 7.86  |
| BossFight (easy)   | 8.85 ± 0.33    |  9.35 ± 2.04 |
| BigFish  (easy)  | 16.46 ± 2.71         | 20.06 ± 5.34 |



Learning curves:

``` title="benchmark/ppo_plot.sh" linenums="1"
--8<-- "benchmark/ppo_plot.sh:98:106"
```

<img loading="lazy" src="https://huggingface.co/datasets/cleanrl/benchmark/resolve/main/benchmark/pr-424/ppo_procgen.png">
<img loading="lazy" src="https://huggingface.co/datasets/cleanrl/benchmark/resolve/main/benchmark/pr-424/ppo_procgen-time.png">


???+ info

    Note that we have run the procgen experiments using the `easy` distribution for reducing the computational cost.


Tracked experiments and game play videos:

<iframe loading="lazy" src="https://wandb.ai/openrlbenchmark/openrlbenchmark/reports/Procgen-CleanRL-s-PPO--VmlldzoxODcxMzUy" style="width:100%; height:500px" title="Procgen-CleanRL-s-PPO"></iframe>



## `ppo_atari_multigpu.py`

The [ppo_atari_multigpu.py](https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/ppo_atari_multigpu.py) leverages data parallelism to speed up training time *at no cost of sample efficiency*. 

`ppo_atari_multigpu.py` has the following features:

* Allows the users to use do training leveraging data parallelism
* For playing Atari games. It uses convolutional layers and common atari-based pre-processing techniques.
* Works with the Atari's pixel `Box` observation space of shape `(210, 160, 3)`
* Works with the `Discrete` action space

???+ warning

    Note that `ppo_atari_multigpu.py` does not work in Windows :fontawesome-brands-windows: and MacOs :fontawesome-brands-apple:. It will error out with `NOTE: Redirects are currently not supported in Windows or MacOs.` See [pytorch/pytorch#20380](https://github.com/pytorch/pytorch/issues/20380)

### Usage


=== "poetry"

    ```bash
    poetry install -E atari
    poetry run python cleanrl/ppo_atari_multigpu.py --help

    # `--nproc_per_node=2` specifies how many subprocesses we spawn for training with data parallelism
    # note it is possible to run this with a *single GPU*: each process will simply share the same GPU
    poetry run torchrun --standalone --nnodes=1 --nproc_per_node=2 cleanrl/ppo_atari_multigpu.py --env-id BreakoutNoFrameskip-v4

    # by default we use the `gloo` backend, but you can use the `nccl` backend for better multi-GPU performance
    poetry run torchrun --standalone --nnodes=1 --nproc_per_node=2 cleanrl/ppo_atari_multigpu.py --env-id BreakoutNoFrameskip-v4 --backend nccl

    # it is possible to spawn more processes than the amount of GPUs you have via `--device-ids`
    # e.g., the command below spawns two processes using GPU 0 and two processes using GPU 1
    poetry run torchrun --standalone --nnodes=1 --nproc_per_node=2 cleanrl/ppo_atari_multigpu.py --env-id BreakoutNoFrameskip-v4 --device-ids 0 0 1 1
    ```

=== "pip"

    ```bash
    pip install -r requirements/requirements-atari.txt
    python cleanrl/ppo_atari_multigpu.py --help

    # `--nproc_per_node=2` specifies how many subprocesses we spawn for training with data parallelism
    # note it is possible to run this with a *single GPU*: each process will simply share the same GPU
    torchrun --standalone --nnodes=1 --nproc_per_node=2 cleanrl/ppo_atari_multigpu.py --env-id BreakoutNoFrameskip-v4

    # by default we use the `gloo` backend, but you can use the `nccl` backend for better multi-GPU performance
    torchrun --standalone --nnodes=1 --nproc_per_node=2 cleanrl/ppo_atari_multigpu.py --env-id BreakoutNoFrameskip-v4 --backend nccl

    # it is possible to spawn more processes than the amount of GPUs you have via `--device-ids`
    # e.g., the command below spawns two processes using GPU 0 and two processes using GPU 1
    torchrun --standalone --nnodes=1 --nproc_per_node=2 cleanrl/ppo_atari_multigpu.py --env-id BreakoutNoFrameskip-v4 --device-ids 0 0 1 1
    ```


### Explanation of the logged metrics

See [related docs](/rl-algorithms/ppo/#explanation-of-the-logged-metrics) for `ppo.py`.

### Implementation details

[ppo_atari_multigpu.py](https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/ppo_atari_multigpu.py) is based on `ppo_atari.py` (see its [related docs](/rl-algorithms/ppo/#implementation-details_1)).

We use [Pytorch's distributed API](https://pytorch.org/tutorials/intermediate/dist_tuto.html) to implement the data parallelism paradigm. The basic idea is that the user can spawn $N$ processes each running a copy of `ppo_atari.py`,  holding a copy of the model, stepping the environments, and averaging their gradients together for the backward pass. Here are a few note-worthy implementation details.

1. **Local versus global parameters**: All of the parameters in `ppo_atari.py` are global (such as batch size), but in `ppo_atari_multigpu.py` we have local parameters as well. Say we run `torchrun --standalone --nnodes=1 --nproc_per_node=2 cleanrl/ppo_atari_multigpu.py --env-id BreakoutNoFrameskip-v4 --local-num-envs=4`; here are how all multi-gpu related parameters are adjusted:
    * **number of environments**: `num_envs = local_num_envs * world_size = 4 * 2 = 8`
    * **batch size**: `local_batch_size = local_num_envs * num_steps = 4 * 128 = 512`, `batch_size = num_envs * num_steps) = 8 * 128 = 1024`
    * **minibatch size**:  `local_minibatch_size = int(args.local_batch_size // args.num_minibatches) = 512 // 4 = 128`, `minibatch_size = int(args.batch_size // args.num_minibatches) = 1024 // 4 = 256`
    * **number of updates**: `num_iterations = args.total_timesteps // args.batch_size = 10000000 // 1024 = 9765`
1. **Adjust seed per process**: we need be very careful with seeding: we could have used the exact same seed for each subprocess. To ensure this does not happen, we do the following

    ```python hl_lines="2 5 16"
    # CRUCIAL: note that we needed to pass a different seed for each data parallelism worker
    args.seed += local_rank
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed - local_rank)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    # ...

    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, args.seed + i, i, args.capture_video, run_name) for i in range(args.num_envs)]
    )
    assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"

    agent = Agent(envs).to(device)
    torch.manual_seed(args.seed)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)
    ```

    Notice that we adjust the seed with `args.seed += local_rank` (line 2), where `local_rank` is the index of the subprocesses. This ensures we seed packages and envs with uncorrealted seeds. However, we do need to use the same `torch` seed for all process to initialize same weights for the `agent` (line 5), after which we can use a different seed for `torch` (line 16).
1. **Efficient gradient averaging**: PyTorch recommends to average the gradient across the whole world via the following (see [docs](https://pytorch.org/tutorials/intermediate/dist_tuto.html#distributed-training))

    ```python
    for param in agent.parameters():
        dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
        param.grad.data /= world_size
    ```

    However, [@cswinter](https://github.com/cswinter) introduces a more efficient gradient averaging scheme with proper batching (see :material-github: [entity-neural-network/incubator#220](https://github.com/entity-neural-network/incubator/pull/220)), which looks like:

    ```python
    all_grads_list = []
    for param in agent.parameters():
        if param.grad is not None:
            all_grads_list.append(param.grad.view(-1))
    all_grads = torch.cat(all_grads_list)
    dist.all_reduce(all_grads, op=dist.ReduceOp.SUM)
    offset = 0
    for param in agent.parameters():
        if param.grad is not None:
            param.grad.data.copy_(
                all_grads[offset : offset + param.numel()].view_as(param.grad.data) / world_size
            )
            offset += param.numel()
    ```

    In our previous empirical testing (see :material-github: [vwxyzjn/cleanrl#162](https://github.com/vwxyzjn/cleanrl/pull/162#issuecomment-1107909696)), we have found [@cswinter](https://github.com/cswinter)'s implementation to be faster, hence we adopt it in our implementation.




### Experiment results



To run benchmark experiments, see :material-github: [benchmark/ppo.sh](https://github.com/vwxyzjn/cleanrl/blob/master/benchmark/ppo.sh). Specifically, execute the following command:


``` title="benchmark/ppo.sh" linenums="1"
--8<-- "benchmark/ppo.sh:102:107"
```

Below are the average episodic returns for `ppo_atari_multigpu.py`. To ensure no loss of sample efficiency, we compared the results against `ppo_atari.py`.


{!benchmark/ppo_atari_multigpu.md!}


Learning curves:

``` title="benchmark/ppo_plot.sh" linenums="1"
--8<-- "benchmark/ppo_plot.sh:108:117"
```

<img loading="lazy" src="https://huggingface.co/datasets/cleanrl/benchmark/resolve/main/benchmark/pr-424/ppo_atari_multigpu.png">
<img loading="lazy" src="https://huggingface.co/datasets/cleanrl/benchmark/resolve/main/benchmark/pr-424/ppo_atari_multigpu-time.png">




Under the same hardware, we see that `ppo_atari_multigpu.py` is about **30% faster** than `ppo_atari.py` with no loss of sample efficiency. 


???+ info

    The experiments above is to show correctness -- we show that by aligning the same hyperparameters of `ppo_atari.py` and `ppo_atari_multigpu.py`, we can achieve the same sample efficiency. However, we can train even faster by simply running a much larger batch size. For example, we can run `torchrun --standalone --nnodes=1 --nproc_per_node=8 cleanrl/ppo_atari_multigpu.py --env-id BreakoutNoFrameskip-v4 --local-num-envs=8`, which will run 8 x 8 = 64 environments in parallel and achieve a batch size of 64 x 128 = 8192. This will likely result in a sample efficiency but should increase the wall time efficiency.


???+ info

    Although `ppo_atari_multigpu.py` is 30% faster than `ppo_atari.py`, `ppo_atari_multigpu.py` is still slower than `ppo_atari_envpool.py`, as shown below.  This comparison really highlights the different kinds of optimization possible.


    <div class="grid-container">
        <img src="../ppo/Breakout-a.png">
        <img src="../ppo/Breakout-time-a.png">
    </div>

    The purpose of `ppo_atari_multigpu.py` is not (yet) to achieve the fastest PPO + Atari example. Rather, its purpose is to *rigorously validate data paralleism does provide performance benefits*. We could do something like `ppo_atari_multigpu_envpool.py` to possibly obtain the fastest PPO + Atari possible, but that is for another day. Note we may need `numba` to pin the threads `envpool` is using in each subprocess to avoid threads fighting each other and lowering the throughput.


Tracked experiments and game play videos:

<iframe loading="lazy" src="https://wandb.ai/openrlbenchmark/openrlbenchmark/reports/Atari-CleanRL-s-PPO-MultiGPU--VmlldzoxOTM2NDUx" style="width:100%; height:500px" title="Atari-CleanRL-s-PPO"></iframe>





## `ppo_pettingzoo_ma_atari.py`
[ppo_pettingzoo_ma_atari.py](https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/ppo_pettingzoo_ma_atari.py) trains an agent to learn playing Atari games via selfplay. The selfplay environment is implemented as a vectorized environment from [PettingZoo.ml](https://www.pettingzoo.ml/atari). The basic idea is to create vectorized environment $E$ with `num_envs = N`, where $N$ is the number of players in the game. Say $N = 2$, then the 0-th sub environment of $E$ will return the observation for player 0 and 1-th sub environment will return the observation of player 1. Then the two environments takes a batch of 2 actions and execute them for player 0 and player 1, respectively. See "Vectorized architecture" in [The 37 Implementation Details of Proximal Policy Optimization](https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/) for more detail.

`ppo_pettingzoo_ma_atari.py` has the following features:

* For playing the pettingzoo's multi-agent Atari game.
* Works with the pixel-based observation space
* Works with the `Box` action space

???+ warning

    Note that `ppo_pettingzoo_ma_atari.py` does not work in Windows :fontawesome-brands-windows:. See [https://pypi.org/project/multi-agent-ale-py/#files](https://pypi.org/project/multi-agent-ale-py/#files)

### Usage

=== "poetry"

    ```bash
    poetry install -E "pettingzoo atari"
    poetry run AutoROM --accept-license
    poetry run  cleanrl/ppo_pettingzoo_ma_atari.py --help
    poetry run  cleanrl/ppo_pettingzoo_ma_atari.py --env-id pong_v3
    poetry run  cleanrl/ppo_pettingzoo_ma_atari.py --env-id surround_v2
    ```

=== "pip"

    ```bash
    pip install -r requirements/requirements-pettingzoo.txt
    pip install -r requirements/requirements-atari.txt
    AutoROM --accept-license
    python cleanrl/ppo_pettingzoo_ma_atari.py --help
    python cleanrl/ppo_pettingzoo_ma_atari.py --env-id pong_v3
    python cleanrl/ppo_pettingzoo_ma_atari.py --env-id surround_v2
    ```

See [https://www.pettingzoo.ml/atari](https://www.pettingzoo.ml/atari) for a full-list of supported environments such as `basketball_pong_v3`. Notice pettingzoo sometimes introduces breaking changes, so make sure to install the pinned dependencies via `poetry`.

### Explanation of the logged metrics

Additionally, it logs the following metrics

* `charts/episodic_return-player0`: episodic return of the game for player 0
* `charts/episodic_return-player1`: episodic return of the game for player 1
* `charts/episodic_length-player0`: episodic length of the game for player 0
* `charts/episodic_length-player1`: episodic length of the game for player 1

See other logged metrics in the [related docs](/rl-algorithms/ppo/#explanation-of-the-logged-metrics) for `ppo.py`.

### Implementation details

[ppo_pettingzoo_ma_atari.py](https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/ppo_pettingzoo_ma_atari.py) is based on `ppo_atari.py` (see its [related docs](/rl-algorithms/ppo/#implementation-details_1)).

`ppo_pettingzoo_ma_atari.py` additionally has the following implementation details:

1. **`supersuit` wrappers**:  uses preprocessing wrappers from `supersuit` instead of from `stable_baselines3`, which looks like the following. In particular note that the `supersuit` does not offer a wrapper similar to `NoopResetEnv`, and that it uses the `agent_indicator_v0` to add two channels indicating the which player the agent controls.

    ```diff
    -env = gym.make(env_id)
    -env = NoopResetEnv(env, noop_max=30)
    -env = MaxAndSkipEnv(env, skip=4)
    -env = EpisodicLifeEnv(env)
    -if "FIRE" in env.unwrapped.get_action_meanings():
    -    env = FireResetEnv(env)
    -env = ClipRewardEnv(env)
    -env = gym.wrappers.ResizeObservation(env, (84, 84))
    -env = gym.wrappers.GrayScaleObservation(env)
    -env = gym.wrappers.FrameStack(env, 4)
    +env = importlib.import_module(f"pettingzoo.atari.{args.env_id}").parallel_env()
    +env = ss.max_observation_v0(env, 2)
    +env = ss.frame_skip_v0(env, 4)
    +env = ss.clip_reward_v0(env, lower_bound=-1, upper_bound=1)
    +env = ss.color_reduction_v0(env, mode="B")
    +env = ss.resize_v1(env, x_size=84, y_size=84)
    +env = ss.frame_stack_v1(env, 4)
    +env = ss.agent_indicator_v0(env, type_only=False)
    +env = ss.pettingzoo_env_to_vec_env_v1(env)
    +envs = ss.concat_vec_envs_v1(env, args.num_envs // 2, num_cpus=0, base_class="gym")
    ```
1. **A more detailed note on the `agent_indicator_v0` wrapper**: let's dig deeper into how `agent_indicator_v0` works. We do `print(envs.reset(), envs.reset().shape)`
    ```python
    [  0.,   0.,   0., 236.,   1,   0.]],

    [[  0.,   0.,   0., 236.,   0.,   1.],
    [  0.,   0.,   0., 236.,   0.,   1.],
    [  0.,   0.,   0., 236.,   0.,   1.],
    ...,
    [  0.,   0.,   0., 236.,   0.,   1.],
    [  0.,   0.,   0., 236.,   0.,   1.],
    [  0.,   0.,   0., 236.,   0.,   1.]]]]) torch.Size([16, 84, 84, 6])
    ```
    
    So the `agent_indicator_v0` adds the last two columns, where `[  0.,   0.,   0., 236.,   1,   0.]]` means this observation is for player 0, and `[  0.,   0.,   0., 236.,   0.,   1.]` is for player 1. Notice the observation still has the range of $[0, 255]$ but the agent indicator channel has the range of $[0,1]$, so we need to be careful when dividing the observation by 255. In particular, we would only divide the first four channels by 255 and leave the agent indicator channels untouched as follows:

    ```py
    def get_action_and_value(self, x, action=None):
        x = x.clone()
        x[:, :, :, [0, 1, 2, 3]] /= 255.0
        hidden = self.network(x.permute((0, 3, 1, 2)))
    ```


### Experiment results



To run benchmark experiments, see :material-github: [benchmark/ppo.sh](https://github.com/vwxyzjn/cleanrl/blob/master/benchmark/ppo.sh). Specifically, execute the following command:

<script src="https://emgithub.com/embed.js?target=https%3A%2F%2Fgithub.com%2Fvwxyzjn%2Fcleanrl%2Fblob%2Fmaster%2Fbenchmark%2Fppo.sh%23L53-L59&style=github&showBorder=on&showLineNumbers=on&showFileMeta=on&showCopy=on"></script>

???+ info

    Note that evaluation is usually tricker in in selfplay environments. The usual episodic return is not a good indicator of the agent's performance in zero-sum games because the episodic return converges to zero. To evaluate the agent's ability, an intuitive approach is to take a look at the videos of the agents playing the game (included below), visually inspect the agent's behavior. The best scheme, however, is rating systems like [Trueskill](https://www.microsoft.com/en-us/research/project/trueskill-ranking-system/) or [ELO scores](https://en.wikipedia.org/wiki/Elo_rating_system). However, they are more difficult to implement and are outside the scode of `ppo_pettingzoo_ma_atari.py`. 
    
    
    For simplicity, we measure the **episodic length** instead, which in a sense measures how many "back and forth" the agent can create. In other words, the longer the agent can play the game, the better the agent can play. Empirically, we have found episodic length to be a good indicator of the agent's skill, especially in `pong_v3` and `surround_v2`. However, it is not the case for `tennis_v3` and we'd need to visually inspect the agents' game play videos.


Below are the average **episodic length** for `ppo_pettingzoo_ma_atari.py`. To ensure no loss of sample efficiency, we compared the results against `ppo_atari.py`.

| Environment      | `ppo_pettingzoo_ma_atari.py`  | 
| ----------- | ----------- | 
| pong_v3 | 4153.60 ± 190.80      | 
| surround_v2 | 3055.33 ± 223.68  | 
| tennis_v3 | 14538.02 ± 7005.54   | 


Learning curves:

<div class="grid-container">
<img src="../ppo/pong_v3.png">

<img src="../ppo/surround_v2.png">

<img src="../ppo/tennis_v3.png">
</div>



Tracked experiments and game play videos:

<iframe loading="lazy" src="https://wandb.ai/openrlbenchmark/openrlbenchmark/reports/Pettingzoo-s-Multi-agent-Atari-CleanRL-s-PPO--VmlldzoyMDkxNTE5" style="width:100%; height:500px" title="Atari-CleanRL-s-PPO"></iframe>



{!rl-algorithms/ppo-isaacgymenvs.md!}

[^1]: Huang, Shengyi; Dossa, Rousslan Fernand Julien; Raffin, Antonin; Kanervisto, Anssi; Wang, Weixun (2022). The 37 Implementation Details of Proximal Policy Optimization. ICLR 2022 Blog Track https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/

[^2]: Andrychowicz, Marcin, Anton Raichuk, Piotr Stańczyk, Manu Orsini, Sertan Girgin, Raphael Marinier, Léonard Hussenot et al. "What matters in on-policy reinforcement learning? a large-scale empirical study." International Conference on Learning Representations 2021, https://openreview.net/forum?id=nIAxjsniDzg
