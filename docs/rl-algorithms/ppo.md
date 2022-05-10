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
| :material-github: [`ppo_continuous_action.py`](https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/ppo_continuous_action.py), :material-file-document: [docs](/rl-algorithms/ppo/#ppo_continuous_actionpy) | For continuous action space. Also implemented Mujoco-specific code-level optimizations |
| :material-github: [`ppo_atari_lstm.py`](https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/ppo_atari_lstm.py), :material-file-document: [docs](/rl-algorithms/ppo/#ppo_atari_lstmpy) | For Atari games using LSTM without stacked frames. |
| :material-github: [`ppo_atari_envpool.py`](https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/ppo_atari_envpool.py), :material-file-document: [docs](/rl-algorithms/ppo/#ppo_atari_envpoolpy) | Uses the blazing fast Envpool Atari vectorized environment. |
| :material-github: [`ppo_procgen.py`](https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/ppo_procgen.py), :material-file-document: [docs](/rl-algorithms/ppo/#ppo_procgenpy) | For the procgen environments |

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

<script src="https://emgithub.com/embed.js?target=https%3A%2F%2Fgithub.com%2Fvwxyzjn%2Fcleanrl%2Fblob%2F5184afc2b7d5032b56e6689175a17b7bad172771%2Fbenchmark%2Fppo.sh%23L4-L9&style=github&showBorder=on&showLineNumbers=on&showFileMeta=on&showCopy=on"></script>


Below are the average episodic returns for `ppo.py`. To ensure the quality of the implementation, we compared the results against `openai/baselies`' PPO.

| Environment      | `ppo.py` | `openai/baselies`' PPO (Huang et al., 2022)[^1]
| ----------- | ----------- | ----------- |
| CartPole-v1      | 492.40 ± 13.05     |497.54 ± 4.02  |
| Acrobot-v1   | -89.93 ± 6.34     |  -81.82 ± 5.58 |
| MountainCar-v0   | -200.00 ± 0.00         | -200.00 ± 0.00 |


Learning curves:

<div class="grid-container">
<img src="../ppo/CartPole-v1.png">

<img src="../ppo/Acrobot-v1.png">

<img src="../ppo/MountainCar-v0.png">
</div>


Tracked experiments and game play videos:

<iframe src="https://wandb.ai/openrlbenchmark/openrlbenchmark/reports/Classic-Control-CleanRL-s-PPO--VmlldzoxODU5MDY1" style="width:100%; height:500px" title="Classic-Control-CleanRL-s-PPO"></iframe>

### Video tutorial

If you'd like to learn `ppo.py` in-depth, consider checking out the following video tutorial:


<div style="text-align: center;"><iframe width="560" height="315" src="https://www.youtube.com/embed/MEt6rrxH8W4" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe></div>


## `ppo_atari.py`

The [ppo_atari.py](https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/ppo_atari.py) has the following features:

* For Atari games. It uses convolutional layers and common atari-based pre-processing techniques.
* Works with the Atari's pixel `Box` observation space of shape `(210, 160, 3)`
* Works with the `Discrete` action space


### Usage

```bash
poetry install -E atari
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

<script src="https://emgithub.com/embed.js?target=https%3A%2F%2Fgithub.com%2Fvwxyzjn%2Fcleanrl%2Fblob%2Fmaster%2Fbenchmark%2Fppo.sh%23L11-L16&style=github&showBorder=on&showLineNumbers=on&showFileMeta=on&showCopy=on"></script>


Below are the average episodic returns for `ppo_atari.py`. To ensure the quality of the implementation, we compared the results against `openai/baselies`' PPO.

| Environment      | `ppo_atari.py` | `openai/baselies`' PPO (Huang et al., 2022)[^1]
| ----------- | ----------- | ----------- |
| BreakoutNoFrameskip-v4      | 416.31 ± 43.92     | 406.57 ± 31.554  |
| PongNoFrameskip-v4   | 20.59 ± 0.35    |  20.512 ± 0.50 |
| BeamRiderNoFrameskip-v4   | 2445.38 ± 528.91         | 2642.97 ± 670.37 |


Learning curves:

<div class="grid-container">
<img src="../ppo/BreakoutNoFrameskip-v4.png">

<img src="../ppo/PongNoFrameskip-v4.png">

<img src="../ppo/BeamRiderNoFrameskip-v4.png">
</div>


Tracked experiments and game play videos:

<iframe src="https://wandb.ai/openrlbenchmark/openrlbenchmark/reports/Atari-CleanRL-s-PPO--VmlldzoxNjk3NjYy" style="width:100%; height:500px" title="Atari-CleanRL-s-PPO"></iframe>

### Video tutorial

If you'd like to learn `ppo_atari.py` in-depth, consider checking out the following video tutorial:


<div style="text-align: center;"><iframe width="560" height="315" src="https://www.youtube.com/embed/05RMTj-2K_Y" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe></div>


## `ppo_continuous_action.py`

The [ppo_continuous_action.py](https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/ppo_continuous_action.py) has the following features:

* For continuous action space. Also implemented Mujoco-specific code-level optimizations
* Works with the `Box` observation space of low-level features
* Works with the `Box` (continuous) action space

### Usage

```bash
poetry install -E atari
python cleanrl/ppo_continuous_action.py --help
python cleanrl/ppo_continuous_action.py --env-id Hopper-v2
```

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

<script src="https://emgithub.com/embed.js?target=https%3A%2F%2Fgithub.com%2Fvwxyzjn%2Fcleanrl%2Fblob%2F5184afc2b7d5032b56e6689175a17b7bad172771%2Fbenchmark%2Fppo.sh%23L32-L38&style=github&showBorder=on&showLineNumbers=on&showFileMeta=on&showCopy=on"></script>



Below are the average episodic returns for `ppo_continuous_action.py`. To ensure the quality of the implementation, we compared the results against `openai/baselies`' PPO.

| Environment      | `ppo_continuous_action.py` | `openai/baselies`' PPO (Huang et al., 2022)[^1]
| ----------- | ----------- | ----------- |
| Hopper-v2      | 2231.12 ± 656.72     | 2518.95 ± 850.46  |
| Walker2d-v2   | 3050.09 ± 1136.21    |  3208.08 ± 1264.37 |
| HalfCheetah-v2   | 1822.82 ± 928.11         | 2152.26 ± 1159.84 |


Learning curves:

<div class="grid-container">
<img src="../ppo/Hopper-v2.png">

<img src="../ppo/Walker2d-v2.png">

<img src="../ppo/HalfCheetah-v2.png">
</div>


Tracked experiments and game play videos:

<iframe src="https://wandb.ai/openrlbenchmark/openrlbenchmark/reports/MuJoCo-CleanRL-s-PPO--VmlldzoxODAwNjkw" style="width:100%; height:500px" title="MuJoCo-CleanRL-s-PPO"></iframe>

### Video tutorial

If you'd like to learn `ppo_continuous_action.py` in-depth, consider checking out the following video tutorial:


<div style="text-align: center;"><iframe width="560" height="315" src="https://www.youtube.com/embed/BvZvx7ENZBw" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe></div>




## `ppo_atari_lstm.py`

The [ppo_atari_lstm.py](https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/ppo_atari_lstm.py) has the following features:

* For Atari games using LSTM without stacked frames. It uses convolutional layers and common atari-based pre-processing techniques.
* Works with the Atari's pixel `Box` observation space of shape `(210, 160, 3)`
* Works with the `Discrete` action space

### Usage

```bash
poetry install -E atari
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

<script src="https://emgithub.com/embed.js?target=https%3A%2F%2Fgithub.com%2Fvwxyzjn%2Fcleanrl%2Fblob%2F5184afc2b7d5032b56e6689175a17b7bad172771%2Fbenchmark%2Fppo.sh%23L18-L23&style=github&showBorder=on&showLineNumbers=on&showFileMeta=on&showCopy=on"></script>


Below are the average episodic returns for `ppo_atari_lstm.py`. To ensure the quality of the implementation, we compared the results against `openai/baselies`' PPO.


| Environment      | `ppo_atari_lstm.py` | `openai/baselies`' PPO (Huang et al., 2022)[^1]
| ----------- | ----------- | ----------- |
| BreakoutNoFrameskip-v4      | 128.92 ± 31.10    | 138.98 ± 50.76  |
| PongNoFrameskip-v4   | 19.78 ± 1.58    | 19.79 ± 0.67 |
| BeamRiderNoFrameskip-v4   | 1536.20 ± 612.21         | 1591.68 ± 372.95|


Learning curves:

<div class="grid-container">
<img src="../ppo/lstm/BreakoutNoFrameskip-v4.png">

<img src="../ppo/lstm/PongNoFrameskip-v4.png">

<img src="../ppo/lstm/BeamRiderNoFrameskip-v4.png">
</div>


Tracked experiments and game play videos:

<iframe src="https://wandb.ai/openrlbenchmark/openrlbenchmark/reports/Atari-CleanRL-s-PPO-LSTM--VmlldzoxODcxMzE4" style="width:100%; height:500px" title="Atari-CleanRL-s-PPO-LSTM"></iframe>



## `ppo_atari_envpool.py`

The [ppo_atari_envpool.py](https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/ppo_atari_envpool.py) has the following features:

* Uses the blazing fast [Envpool](https://github.com/sail-sg/envpool) vectorized environment.
* For Atari games. It uses convolutional layers and common atari-based pre-processing techniques.
* Works with the Atari's pixel `Box` observation space of shape `(210, 160, 3)`
* Works with the `Discrete` action space

### Usage

```bash
poetry install -E atari
python cleanrl/ppo_atari_envpool.py --help
python cleanrl/ppo_atari_envpool.py --env-id Breakout-v5
```

### Explanation of the logged metrics

See [related docs](/rl-algorithms/ppo/#explanation-of-the-logged-metrics) for `ppo.py`.

### Implementation details

[ppo_atari_envpool.py](https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/ppo_atari_envpool.py) uses a customized `RecordEpisodeStatistics` to work with envpool but has the same other implementation details as `ppo_atari.py` (see [related docs](/rl-algorithms/ppo/#implementation-details_1)).

### Experiment results

To run benchmark experiments, see :material-github: [benchmark/ppo.sh](https://github.com/vwxyzjn/cleanrl/blob/master/benchmark/ppo.sh). Specifically, execute the following command:

<script src="https://emgithub.com/embed.js?target=https%3A%2F%2Fgithub.com%2Fvwxyzjn%2Fcleanrl%2Fblob%2F5184afc2b7d5032b56e6689175a17b7bad172771%2Fbenchmark%2Fppo.sh%23L25-L30&style=github&showBorder=on&showLineNumbers=on&showFileMeta=on&showCopy=on"></script>


Below are the average episodic returns for `ppo_atari_envpool.py`. Notice it has the same sample efficiency as `ppo_atari.py`, but runs about 3x faster.



| Environment      | `ppo_atari_envpool.py` (~80 mins) | `ppo_atari.py` (~220 mins)
| ----------- | ----------- | ----------- |
| BreakoutNoFrameskip-v4 |   389.57 ± 29.62    | 416.31 ± 43.92 
| PongNoFrameskip-v4 | 20.55 ± 0.37   | 20.59 ± 0.35   
| BeamRiderNoFrameskip-v4 |   2039.83 ± 1146.62 | 2445.38 ± 528.91  




Learning curves:

<div class="grid-container">
<img src="../ppo/Breakout.png">
<img src="../ppo/Breakout-time.png">

<img src="../ppo/Pong.png">
<img src="../ppo/Pong-time.png">

<img src="../ppo/BeamRider.png">
<img src="../ppo/BeamRider-time.png">
</div>


Tracked experiments and game play videos:

<iframe src="https://wandb.ai/openrlbenchmark/openrlbenchmark/reports/Atari-CleanRL-s-PPO-Envpool--VmlldzoxODcxMzI3" style="width:100%; height:500px" title="Atari-CleanRL-s-PPO-Envpool"></iframe>




## `ppo_procgen.py`

The [ppo_procgen.py](https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/ppo_procgen.py) has the following features:

* For the procgen environments
* Uses IMPALA-style neural network
* Works with the `Discrete` action space


### Usage

```bash
poetry install -E atari
python cleanrl/ppo_procgen.py --help
python cleanrl/ppo_procgen.py --env-id BreakoutNoFrameskip-v4
```

### Explanation of the logged metrics

See [related docs](/rl-algorithms/ppo/#explanation-of-the-logged-metrics) for `ppo.py`.

### Implementation details

[ppo_procgen.py](https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/ppo_procgen.py) is based on the details in "Appendix" in [The 37 Implementation Details of Proximal Policy Optimization](https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/), which are as follows:

1. IMPALA-style Neural Network (:material-github: [common/models.py#L28](https://github.com/openai/baselines/blob/ea25b9e8b234e6ee1bca43083f8f3cf974143998/baselines/common/models.py#L28))


### Experiment results



To run benchmark experiments, see :material-github: [benchmark/ppo.sh](https://github.com/vwxyzjn/cleanrl/blob/master/benchmark/ppo.sh). Specifically, execute the following command:

<script src="https://emgithub.com/embed.js?target=https%3A%2F%2Fgithub.com%2Fvwxyzjn%2Fcleanrl%2Fblob%2F5184afc2b7d5032b56e6689175a17b7bad172771%2Fbenchmark%2Fppo.sh%23L40-L45&style=github&showBorder=on&showLineNumbers=on&showFileMeta=on&showCopy=on"></script>


We try to match the default setting in [openai/train-procgen](https://github.com/openai/train-procgen) except that we use the `easy` distribution mode and `total_timesteps=25e6` to save compute. Notice [openai/train-procgen](https://github.com/openai/train-procgen) has the following settings:

1. Learning rate annealing is turned off by default
1. Reward scaling and reward clipping is used


Below are the average episodic returns for `ppo_procgen.py`. To ensure the quality of the implementation, we compared the results against `openai/baselies`' PPO.

| Environment      | `ppo_procgen.py` | `openai/baselies`' PPO (Huang et al., 2022)[^1]
| ----------- | ----------- | ----------- |
| StarPilot      | 31.40 ± 11.73     | 33.97 ± 7.86  |
| BossFight   | 9.09 ± 2.35    |  9.35 ± 2.04 |
| BigFish   | 21.44 ± 6.73         | 20.06 ± 5.34 |


Learning curves:

<div class="grid-container">
<img src="../ppo/StarPilot.png">

<img src="../ppo/BossFight.png">

<img src="../ppo/BigFish.png">
</div>


Tracked experiments and game play videos:

<iframe src="https://wandb.ai/openrlbenchmark/openrlbenchmark/reports/Procgen-CleanRL-s-PPO--VmlldzoxODcxMzUy" style="width:100%; height:500px" title="Procgen-CleanRL-s-PPO"></iframe>



[^1]: Huang, Shengyi; Dossa, Rousslan Fernand Julien; Raffin, Antonin; Kanervisto, Anssi; Wang, Weixun (2022). The 37 Implementation Details of Proximal Policy Optimization. ICLR 2022 Blog Track https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/