# Parallel Q Network (PQN)


## Overview

PQN is a parallelized version of the Deep Q-learning algorithm. It is designed to be more efficient than DQN by using multiple agents to interact with the environment in parallel. PQN can be thought of as DQN (1) without replay buffer and target networks, and (2) with layer normalizations and parallel environments.

Original paper: 

* [Simplifying Deep Temporal Difference Learning](https://arxiv.org/html/2407.04811v2)

Reference resources:

* :material-github: [purejaxql](https://github.com/mttga/purejaxql)

## Implemented Variants


| Variants Implemented      | Description |
| ----------- | ----------- |
| :material-github: [`pqn.py`](https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/pqn.py), :material-file-document: [docs](/rl-algorithms/pqn/#pqnpy) | For classic control tasks like `CartPole-v1`. |
| :material-github: [`pqn_atari_envpool.py`](https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/pqn_atari_envpool.py), :material-file-document: [docs](/rl-algorithms/pqn/#pqn_atari_envpoolpy) |  For Atari games. Uses the blazing fast Envpool Atari vectorized environment. It uses convolutional layers and common atari-based pre-processing techniques. |
| :material-github: [`pqn_atari_envpool_lstm.py`](https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/pqn_atari_envpool_lstm.py), :material-file-document: [docs](/rl-algorithms/pqn/#pqn_atari_envpool_lstmpy) | For Atari games. Uses the blazing fast Envpool Atari vectorized environment. Using LSTM without stacked frames. |

Below are our single-file implementations of PQN:

## `pqn.py`

The [pqn.py](https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/pqn.py) has the following features:

* Works with the `Box` observation space of low-level features
* Works with the `Discrete` action space
* Works with envs like `CartPole-v1`

### Usage

=== "poetry"

    ```bash
    poetry install
    poetry run python cleanrl/pqn.py --help
    poetry run python cleanrl/pqn.py --env-id CartPole-v1
    ```

=== "pip"

    ```bash
    python cleanrl/pqn.py --help
    python cleanrl/pqn.py --env-id CartPole-v1
    ```

### Explanation of the logged metrics

Running `python cleanrl/pqn.py` will automatically record various metrics such as actor or value losses in Tensorboard. Below is the documentation for these metrics:

* `charts/episodic_return`: episodic return of the game
* `charts/episodic_length`: episodic length of the game
* `charts/SPS`: number of steps per second
* `charts/learning_rate`: the current learning rate
* `losses/td_loss`: the mean squared error (MSE) between the Q values at timestep $t$ and the Bellman update target estimated using the $Q(\lambda)$ returns.
* `losses/q_values`: it is the average Q values of the sampled data in the replay buffer; useful when gauging if under or over estimation happens.

### Implementation details

1. Vectorized architecture (:material-github: [common/cmd_util.py#L22](https://github.com/openai/baselines/blob/ea25b9e8b234e6ee1bca43083f8f3cf974143998/baselines/common/cmd_util.py#L22))
2. Orthogonal Initialization of Weights and Constant Initialization of biases (:material-github: [a2c/utils.py#L58)](https://github.com/openai/baselines/blob/ea25b9e8b234e6ee1bca43083f8f3cf974143998/baselines/a2c/utils.py#L58))
3. Normalized Q Network (:material-github: [purejaxql/pqn_atari.py#L200](https://github.com/mttga/purejaxql/blob/2205ae5308134d2cedccd749074bff2871832dc8/purejaxql/pqn_atari.py#L200))
4. Uses the RAdam Optimizer with the default epsilon parameter(:material-github: [purejaxql/pqn_atari.py#L362](https://github.com/mttga/purejaxql/blob/2205ae5308134d2cedccd749074bff2871832dc8/purejaxql/pqn_atari.py#L362))
5. Adam Learning Rate Annealing (:material-github: [pqn2/pqn2.py#L133-L135](https://github.com/openai/baselines/blob/ea25b9e8b234e6ee1bca43083f8f3cf974143998/baselines/pqn2/pqn2.py#L133-L135))
6. Q Lambda Returns (:material-github: [purejaxql/pqn_atari.py#L446](https://github.com/mttga/purejaxql/blob/2205ae5308134d2cedccd749074bff2871832dc8/purejaxql/pqn_atari.py#L446))
7. Mini-batch Updates (:material-github: [pqn2/pqn2.py#L157-L166](https://github.com/openai/baselines/blob/ea25b9e8b234e6ee1bca43083f8f3cf974143998/baselines/pqn2/pqn2.py#L157-L166))
8. Global Gradient Clipping (:material-github: [purejaxql/pqn_atari.py#L360](https://github.com/mttga/purejaxql/blob/2205ae5308134d2cedccd749074bff2871832dc8/purejaxql/pqn_atari.py#L360))

### Experiment results

To run benchmark experiments, see :material-github: [benchmark/pqn.sh](https://github.com/vwxyzjn/cleanrl/blob/master/benchmark/pqn.sh). Specifically, execute the following command:

``` title="benchmark/pqn.sh" linenums="1"
--8<-- "benchmark/pqn.sh:0:6"
```

Episode Rewards:

| Environment      | CleanRL PQN       |
|------------------|-------------------|
| CartPole-v1      | 408.14 ± 128.42   |
| Acrobot-v1       | -93.71 ± 2.94     |
| MountainCar-v0   | -200.00 ± 0.00    |

Runtime:

| Environment      | CleanRL PQN          |
|------------------|----------------------|
| CartPole-v1      | 3.619667511995135    |
| Acrobot-v1       | 4.264845468334595    |
| MountainCar-v0   | 3.99800178870078     |

Learning curves:

``` title="benchmark/pqn_plot.sh" linenums="1"
--8<-- "benchmark/pqn_plot.sh:1:9"
```

<img src="../pqn/pqn_state.png">

Tracked experiments: 

<iframe src="https://api.wandb.ai/links/rogercreus/9s50xw0j" style="width:100%; height:500px" title="CleanRL-s-PPO-TrXL"></iframe>

## `pqn_atari_envpool.py`

The [pqn_atari_envpool.py](https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/pqn_atari_envpool.py) has the following features:

* Uses the blazing fast [Envpool](https://github.com/sail-sg/envpool) vectorized environment.
* For Atari games. It uses convolutional layers and common atari-based pre-processing techniques.
* Works with the Atari's pixel `Box` observation space of shape `(210, 160, 3)`
* Works with the `Discrete` action space

???+ warning

    Note that `pqn_atari_envpool.py` does not work in Windows :fontawesome-brands-windows: and MacOs :fontawesome-brands-apple:. See envpool's built wheels here: [https://pypi.org/project/envpool/#files](https://pypi.org/project/envpool/#files)

???+ bug

    EnvPool's vectorized environment **does not behave the same** as gym's vectorized environment, which causes a compatibility bug in our PQN implementation. When an action $a$ results in an episode termination or truncation, the environment generates $s_{last}$ as the terminated or truncated state; we then use $s_{new}$ to denote the initial state of the new episodes. Here is how the bahviors differ:

    * Under the vectorized environment of `envpool<=0.6.4`, the `obs` in `obs, reward, done, info = env.step(action)` is the truncated state $s_{last}$
    * Under the vectorized environment of `gym==0.23.1`, the `obs` in `obs, reward, done, info = env.step(action)` is the initial state $s_{new}$.

    This causes the $s_{last}$ to be off by one. 
    See [:material-github: sail-sg/envpool#194](https://github.com/sail-sg/envpool/issues/194) for more detail. However, it does not seem to impact performance, so we take a note here and await for the upstream fix.


### Usage

=== "poetry"

    ```bash
    poetry install -E envpool
    poetry run python cleanrl/pqn_atari_envpool.py --help
    poetry run python cleanrl/pqn_atari_envpool.py --env-id Breakout-v5
    ```

=== "pip"

    ```bash
    pip install -r requirements/requirements-envpool.txt
    python cleanrl/pqn_atari_envpool.py --help
    python cleanrl/pqn_atari_envpool.py --env-id Breakout-v5
    ```

### Explanation of the logged metrics

See [related docs](/rl-algorithms/pqn/#explanation-of-the-logged-metrics) for `pqn.py`.

### Implementation details

[pqn_atari_envpool.py](https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/pqn_atari_envpool.py) uses a customized `RecordEpisodeStatistics` to work with envpool but has the same other implementation details as `ppo_atari.py`.

### Experiment results

To run benchmark experiments, see :material-github: [benchmark/pqn.sh](https://github.com/vwxyzjn/cleanrl/blob/master/benchmark/pqn.sh). Specifically, execute the following command:

``` title="benchmark/pqn.sh" linenums="1"
--8<-- "benchmark/pqn.sh:12:17"
```

Episode Rewards:

| Environment       | CleanRL PQN        |
|-------------------|--------------------|
| Breakout-v5       | 356.93 ± 7.48      |
| SpaceInvaders-v5  | 900.07 ± 107.95    |
| BeamRider-v5      | 1987.97 ± 24.47    |
| Pong-v5           | 20.44 ± 0.11       |
| MsPacman-v5       | 2437.57 ± 215.01   |

Runtime:

| Environment       | CleanRL PQN           |
|-------------------|-----------------------|
| Breakout-v5       | 41.27235000576079     |
| SpaceInvaders-v5  | 42.191246278536035    |
| BeamRider-v5      | 42.66799268151052     |
| Pong-v5           | 39.35770012905844     |
| MsPacman-v5       | 43.22808379473344     |


Learning curves:

``` title="benchmark/pqn_plot.sh" linenums="1"
--8<-- "benchmark/pqn_plot.sh:11:29"
```

<img src="../pqn/pqn.png">

Tracked experiments: 

<iframe src="https://wandb.ai/rogercreus/cleanRL/reports/PQN-PR-472--Vmlldzo5ODg1NTkx" style="width:100%; height:500px" title="CleanRL-s-PPO-TrXL"></iframe>

## `pqn_atari_envpool_lstm.py`

The [pqn_atari_envpool_lstm.py](https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/pqn_atari_envpool_lstm.py) has the following features:

* Uses the blazing fast [Envpool](https://github.com/sail-sg/envpool) vectorized environment.
* For Atari games using LSTM without stacked frames. It uses convolutional layers and common atari-based pre-processing techniques.
* Works with the Atari's pixel `Box` observation space of shape `(210, 160, 3)`
* Works with the `Discrete` action space

???+ warning

    Note that `pqn_atari_envpool.py` does not work in Windows :fontawesome-brands-windows: and MacOs :fontawesome-brands-apple:. See envpool's built wheels here: [https://pypi.org/project/envpool/#files](https://pypi.org/project/envpool/#files)

???+ bug

    EnvPool's vectorized environment **does not behave the same** as gym's vectorized environment, which causes a compatibility bug in our PQN implementation. When an action $a$ results in an episode termination or truncation, the environment generates $s_{last}$ as the terminated or truncated state; we then use $s_{new}$ to denote the initial state of the new episodes. Here is how the bahviors differ:

    * Under the vectorized environment of `envpool<=0.6.4`, the `obs` in `obs, reward, done, info = env.step(action)` is the truncated state $s_{last}$
    * Under the vectorized environment of `gym==0.23.1`, the `obs` in `obs, reward, done, info = env.step(action)` is the initial state $s_{new}$.

    This causes the $s_{last}$ to be off by one. 
    See [:material-github: sail-sg/envpool#194](https://github.com/sail-sg/envpool/issues/194) for more detail. However, it does not seem to impact performance, so we take a note here and await for the upstream fix.

### Usage


=== "poetry"

    ```bash
    poetry install -E atari
    poetry run python cleanrl/pqn_atari_envpool_lstm.py --help
    poetry run python cleanrl/pqn_atari_envpool_lstm.py --env-id Breakout-v5
    ```

=== "pip"

    ```bash
    pip install -r requirements/requirements-atari.txt
    python cleanrl/pqn_atari_envpool_lstm.py --help
    python cleanrl/pqn_atari_envpool_lstm.py --env-id Breakout-v5
    ```


### Explanation of the logged metrics

See [related docs](/rl-algorithms/pqn/#explanation-of-the-logged-metrics) for `pqn.py`.

### Implementation details

[pqn_atari_envpool_lstm.py](https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/pqn_atari_envpool_lstm.py) is based on the "5 LSTM implementation details" in [The 37 Implementation Details of Proximal Policy Optimization](https://iclr-blog-track.github.io/2022/03/25/pqn-implementation-details/), which are as follows:

1. Layer initialization for LSTM layers (:material-github: [a2c/utils.py#L84-L86](https://github.com/openai/baselines/blob/ea25b9e8b234e6ee1bca43083f8f3cf974143998/baselines/a2c/utils.py#L84-L86))
2. Initialize the LSTM states to be zeros (:material-github: [common/models.py#L179](https://github.com/openai/baselines/blob/ea25b9e8b234e6ee1bca43083f8f3cf974143998/baselines/common/models.py#L179))
3. Reset LSTM states at the end of the episode (:material-github: [common/models.py#L141](https://github.com/openai/baselines/blob/ea25b9e8b234e6ee1bca43083f8f3cf974143998/baselines/common/models.py#L141))
4. Prepare sequential rollouts in mini-batches (:material-github: [a2c/utils.py#L81](https://github.com/openai/baselines/blob/ea25b9e8b234e6ee1bca43083f8f3cf974143998/baselines/a2c/utils.py#L81))
5. Reconstruct LSTM states during training (:material-github: [a2c/utils.py#L81](https://github.com/openai/baselines/blob/ea25b9e8b234e6ee1bca43083f8f3cf974143998/baselines/a2c/utils.py#L81))

To help test out the memory, we remove the 4 stacked frames from the observation (i.e., using `env = gym.wrappers.FrameStack(env, 1)` instead of `env = gym.wrappers.FrameStack(env, 4)` like in `ppo_atari.py` )

### Experiment results

To run benchmark experiments, see :material-github: [benchmark/pqn.sh](https://github.com/vwxyzjn/cleanrl/blob/master/benchmark/pqn.sh). Specifically, execute the following command:

``` title="benchmark/pqn.sh" linenums="1"
--8<-- "benchmark/pqn.sh:23:28"
```


Episode Rewards:

| Environment       | CleanRL PQN        |
|-------------------|--------------------|
| Breakout-v5       | 366.47 ± 2.72      |
| SpaceInvaders-v5  | 681.92 ± 40.15     |
| BeamRider-v5      | 2050.85 ± 38.58    |
| MsPacman-v5       | 1815.20 ± 183.03   |

Runtime:

| Environment       | CleanRL PQN           |
|-------------------|-----------------------|
| Breakout-v5       | 170.30230232607076    |
| SpaceInvaders-v5  | 168.45747969698144    |
| BeamRider-v5      | 172.11561139317593    |
| MsPacman-v5       | 171.66131707108408    |

Learning curves:

``` title="benchmark/pqn_plot.sh" linenums="1"
--8<-- "benchmark/pqn_plot.sh:32:50"
```

<img src="../pqn/pqn_lstm.png">

Tracked experiments: 

<iframe src="https://wandb.ai/rogercreus/cleanRL/reports/PQN-PR-472--Vmlldzo5ODg1NTkx" style="width:100%; height:500px" title="CleanRL-s-PPO-TrXL"></iframe>