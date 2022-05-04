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

* For playing Atari games. It uses convolutional layers and common atari-based pre-processing techniques.
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

* For playing Atari games. It uses convolutional layers and common atari-based pre-processing techniques.
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
* For playing Atari games. It uses convolutional layers and common atari-based pre-processing techniques.
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

* For playing Atari games. It uses convolutional layers and common atari-based pre-processing techniques.
* Works with the Atari's pixel `Box` observation space of shape `(210, 160, 3)`
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



## `ppo_atari_multigpu.py`

The [ppo_atari_multigpu.py](https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/ppo_atari_multigpu.py) leverages data parallelism to speed up training time *at no cost of sample efficiency*. 

`ppo_atari_multigpu.py` has the following features:

* Allows the users to use do training leveraging data parallelism
* For playing Atari games. It uses convolutional layers and common atari-based pre-processing techniques.
* Works with the Atari's pixel `Box` observation space of shape `(210, 160, 3)`
* Works with the `Discrete` action space

### Usage

```bash
poetry install -E atari
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

We use [Pytorch's distributed API](https://pytorch.org/tutorials/intermediate/dist_tuto.html) to implement the data parallelism paradigm. The basic idea is that the user can spawn $N$ processes each holding a copy of the model, step the environments, and averages their gradients together for the backward pass. Here are a few note-worthy implementation details.

1. **Shard the environments**: by default, `ppo_atari_multigpu.py` uses `--num-envs=8`. When calling `torchrun --standalone --nnodes=1 --nproc_per_node=2 cleanrl/ppo_atari_multigpu.py --env-id BreakoutNoFrameskip-v4`, it spawns $N=2$ (by `--nproc_per_node=2`) subprocesses and shard the environments across these 2 subprocesses. In particular, each subprocess will have `8/2=4` environments. Implementation wise, we do `args.num_envs = int(args.num_envs / world_size)`. Here `world_size=2` refers to the size of the **world**, which means the group of subprocesses. We also need to adjust various variables as follows:
    * **batch size**: by default it is `(num_envs * num_steps) = 8 * 128 = 1024` and we adjust it to `(num_envs / world_size * num_steps) = (4 * 128) = 512`. 
    * **minibatch size**: by default it is `(num_envs * num_steps) / num_minibatches = (8 * 128) / 4 = 256` and we adjust it to `(num_envs / world_size * num_steps) / num_minibatches = (4 * 128) / 4 = 128`. 
    * **number of updates**: by default it is `total_timesteps // batch_size = 10000000 // (8 * 128) = 9765` and we adjust it to   `total_timesteps // (batch_size * world_size) = 10000000 // (8 * 128 * 2) = 4882`.
    * **global step increment**: by default it is `num_envs`  and we adjust it to `num_envs * world_size`.
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



We can see how `ppo_atari_multigpu.py` can result in no loss of sample efficiency. In this example, the `ppo_atari.py`'s minibatch size is `256` and the `ppo_atari_multigpu.py`'s minibatch size is `128` with world size 2. Because we average gradient across the world, the gradient under  `ppo_atari_multigpu.py` should be virtually the same as the gradient under `ppo_atari.py`.

<!-- 

<script src="https://unpkg.com/monaco-editor@latest/min/vs/loader.js"></script>


<div style="padding-bottom: 20px;">
	<div
	id="ppo_shared"
	style="width: 100%; height: 600px; border: 1px solid grey"
	></div>
</div>

<script>
  require.config({
    paths: { vs: "https://unpkg.com/monaco-editor@latest/min/vs" },
  });
  window.MonacoEnvironment = { getWorkerUrl: () => proxy };

  let proxy = URL.createObjectURL(
    new Blob(
      [
        `
	self.MonacoEnvironment = {
		baseUrl: 'https://unpkg.com/monaco-editor@latest/min/'
	};
	importScripts('https://unpkg.com/monaco-editor@latest/min/vs/base/worker/workerMain.js');
`,
      ],
      { type: "text/javascript" }
    )
  );

  require(["vs/editor/editor.main"], function () {
    var diffEditor = monaco.editor.createDiffEditor(
      document.getElementById("ppo_shared")
    );

	
    Promise.all([
		xhr("https://raw.githubusercontent.com/vwxyzjn/cleanrl/master/cleanrl/ppo_atari.py"),
		xhr("https://raw.githubusercontent.com/vwxyzjn/cleanrl/master/cleanrl/ppo_atari.py")
	]).then(function (r) {
      var originalTxt = r[0].responseText;
      var modifiedTxt = r[1].responseText;

      diffEditor.setModel({
        original: monaco.editor.createModel(originalTxt, "python"),
        modified: monaco.editor.createModel(modifiedTxt, "python"),
        startLineNumber: 104,
      });
      diffEditor.revealPositionInCenter({ lineNumber: 115, column: 0 });
    });
  });
</script>
<script>
  function xhr(url) {
    var req = null;
    return new Promise(
      function (c, e) {
        req = new XMLHttpRequest();
        req.onreadystatechange = function () {
          if (req._canceled) {
            return;
          }

          if (req.readyState === 4) {
            if (
              (req.status >= 200 && req.status < 300) ||
              req.status === 1223
            ) {
              c(req);
            } else {
              e(req);
            }
            req.onreadystatechange = function () {};
          }
        };

        req.open("GET", url, true);
        req.responseType = "";

        req.send(null);
      },
      function () {
        req._canceled = true;
        req.abort();
      }
    );
  }
</script> -->



### Experiment results



To run benchmark experiments, see :material-github: [benchmark/ppo.sh](https://github.com/vwxyzjn/cleanrl/blob/master/benchmark/ppo.sh). Specifically, execute the following command:

<script src="https://emgithub.com/embed.js?target=https%3A%2F%2Fgithub.com%2Fvwxyzjn%2Fcleanrl%2Fblob%2Fc8fe88b7d7daf5be5324c00735885efddb40a252%2Fbenchmark%2Fppo.sh%23L47-L52&style=github&showBorder=on&showLineNumbers=on&showFileMeta=on&showCopy=on"></script>


Below are the average episodic returns for `ppo_atari_multigpu.py`. To ensure no loss of sample efficiency, we compared the results against `ppo_atari.py`.

| Environment      | `ppo_atari_multigpu.py` (in ~160 mins) | `ppo_atari.py` (in ~215 mins)
| ----------- | ----------- | ----------- |
| BreakoutNoFrameskip-v4 | 429.06 ± 52.09      | 416.31 ± 43.92     | 
| PongNoFrameskip-v4 | 20.40 ± 0.46  | 20.59 ± 0.35    |  
| BeamRiderNoFrameskip-v4 | 2454.54 ± 740.49   | 2445.38 ± 528.91         | 


Learning curves:

<div class="grid-container">
<img src="../ppo/BreakoutNoFrameskip-v4multigpu.png">
<img src="../ppo/BreakoutNoFrameskip-v4multigpu-time.png">

<img src="../ppo/PongNoFrameskip-v4multigpu.png">
<img src="../ppo/PongNoFrameskip-v4multigpu-time.png">

<img src="../ppo/BeamRiderNoFrameskip-v4multigpu.png">
<img src="../ppo/BeamRiderNoFrameskip-v4multigpu-time.png">
</div>


Under the same hardware, we see that `ppo_atari_multigpu.py` is about **30% faster** than `ppo_atari.py` with no loss of sample efficiency. 


???+ info

    Although `ppo_atari_multigpu.py` is 30% faster than `ppo_atari.py`, `ppo_atari_multigpu.py` is still slower than `ppo_atari_envpool.py`, as shown below.  This comparison really highlights the different kinds of optimization possible.


    <div class="grid-container">
        <img src="../ppo/Breakout-a.png">
        <img src="../ppo/Breakout-time-a.png">
    </div>

    The purpose of `ppo_atari_multigpu.py` is not (yet) to achieve the fastest PPO + Atari example. Rather, its purpose is to *rigorously validate data paralleism does provide performance benefits*. We could do something like `ppo_atari_multigpu_envpool.py` to possibly obtain the fastest PPO + Atari possible, but that is for another day. Note we may need `numba` to pin the threads `envpool` is using in each subprocess to avoid threads fighting each other and lowering the throughput.


Tracked experiments and game play videos:

<iframe src="https://wandb.ai/openrlbenchmark/openrlbenchmark/reports/Atari-CleanRL-s-PPO-MultiGPU--VmlldzoxOTM2NDUx" style="width:100%; height:500px" title="Atari-CleanRL-s-PPO"></iframe>



[^1]: Huang, Shengyi; Dossa, Rousslan Fernand Julien; Raffin, Antonin; Kanervisto, Anssi; Wang, Weixun (2022). The 37 Implementation Details of Proximal Policy Optimization. ICLR 2022 Blog Track https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/