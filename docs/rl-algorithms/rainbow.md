# Rainbow

## Overview

The Rainbow algorithm is an extension of DQN that combines multiple improvements:

* Prioritized Experience Replay
* Dueling Network Architecture
* Noisy Networks
* Distributional Q-Learning
* N-step Learning
* Double Q-Learning

Original papers: 

* [Rainbow: Combining Improvements in Deep Reinforcement Learning](https://arxiv.org/abs/1710.02298)

Reference resources:

* :material-github: [Dopamine](https://github.com/google/dopamine)

* :material-github: [Kaixhin](https://github.com/Kaixhin/Rainbow)

## Implemented Variants

| Variants Implemented      | Description |
| ----------- | ----------- |
| :material-github: [`rainbow_atari.py`](https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/rainbow_atari.py), :material-file-document: [docs](/rl-algorithms/rainbow/#rainbow_ataripy) |  For playing Atari games. It uses convolutional layers and common atari-based pre-processing techniques. |


## `rainbow_atari.py`

The [rainbow_atari.py](https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/rainbow_atari.py) has the following features:

* For playing Atari games. It uses convolutional layers and common atari-based pre-processing techniques.
* Works with the Atari's pixel `Box` observation space of shape `(210, 160, 3)`
* Works with the `Discrete` action space

### Usage

```bash
poetry install -E atari
python cleanrl/rainbow_atari.py --env-id BreakoutNoFrameskip-v4
python cleanrl/rainbow_atari.py --env-id PongNoFrameskip-v4
```

=== "poetry"

    ```bash
    poetry install -E atari
    poetry run python cleanrl/rainbow_atari.py --env-id BreakoutNoFrameskip-v4
    poetry run python cleanrl/rainbow_atari.py --env-id PongNoFrameskip-v4
    ```

=== "pip"

    ```bash
    pip install -r requirements/requirements-atari.txt
    python cleanrl/rainbow_atari.py --env-id BreakoutNoFrameskip-v4
    python cleanrl/rainbow_atari.py --env-id PongNoFrameskip-v4
    ```


### Explanation of the logged metrics

Running `python cleanrl/rainbow_atari.py` will automatically record various metrics such as actor or value losses in Tensorboard. Below is the documentation for these metrics:

* `charts/episodic_return`: episodic return of the game
* `charts/SPS`: number of steps per second
* `losses/td_loss`: the n-step distributional TD loss
* `losses/q_values`: the mean Q values of the sampled data in the replay buffer
* `charts/beta`: the beta value of the prioritized experience replay

### Implementation details

[rainbow_atari.py](https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/rainbow_atari.py) is based on (Hessel et al., 2018)[^1], and uses the same hyperparameters as (Hessel et al., 2018)[^1]. See Table 1 in (Hessel et al., 2018)[^1] for the hyperparameters. However, there are a few implementation differences:

1. `rainbow_atari.py` uses the more popular Adam Optimizer with the `--learning-rate=0.0000625` as follows:
    ```python
    optim.Adam(q_network.parameters(), lr=0.0000625)
    ```
    whereas (Hessel et al., 2018)[^1] uses the RMSProp optimizer with `--learning-rate=0.0000625`, gradient momentum `0.95`, squared gradient momentum `0.95`, and min squared gradient `0.01` as follows:
    ```python
    optim.RMSprop(
        q_network.parameters(),
        lr=2.5e-4,
        momentum=0.95,
        # ... PyTorch's RMSprop does not directly support
        # squared gradient momentum and min squared gradient
        # so we are not sure what to put here.
    )
    ``` 

### Experiment results

To run benchmark experiments, see :material-github: [benchmark/rainbow.sh](https://github.com/vwxyzjn/cleanrl/blob/master/benchmark/rainbow.sh). Specifically, execute the following command:

``` title="benchmark/rainbow.sh" linenums="1"
--8<-- "benchmark/rainbow.sh:0:6"
```

Below are the average episodic returns for `rainbow_atari.py`. 

|                             | Rainbow   | C51   | DQN   |
|:----------------------------|:----------------------------------------|:------------------------------------|:------------------------------------|
| AlienNoFrameskip-v4         | 2907.03 ± 355.53                        | 1831.00 ± 98.23                     | 1275.77 ± 65.41                     |
| AssaultNoFrameskip-v4       | 7661.11 ± 226.51                        | 3322.54 ± 94.46                     | 3845.70 ± 443.31                    |
| GopherNoFrameskip-v4        | 8111.07 ± 300.60                        | 8715.60 ± 492.23                    | 10415.53 ± 3438.12                  |
| YarsRevengeNoFrameskip-v4   | 63536.39 ± 5432.22                      | 11010.99 ± 904.27                   | 15290.12 ± 8010.56                  |
| SpaceInvadersNoFrameskip-v4 | 1835.52 ± 205.10                        | 2009.05 ± 226.96                    | 1441.68 ± 23.92                     |
| MsPacmanNoFrameskip-v4      | 3113.30 ± 393.00                        | 2445.13 ± 30.16                     | 2109.43 ± 49.85                     |


Learning curves:

Rainbow shows better performance than C51 and DQN.
<div class="grid-container">
<img src="../rainbow/rainbow_env_curves.png">
</div>

Rainbow is also more sample efficient than C51 and DQN.

<div class="grid-container">
<img src="../rainbow/rainbow_sample_eff.png">
</div>

Rainbow obtains better aggregated performance than C51 and DQN.
<div class="grid-container">
<img src="../rainbow/rainbow_c51_dqn_bars.png">
</div>

<!-- Tracked experiments:

<iframe src="https://wandb.ai/rogercreus/rainbow?nw=nwuserrogercreus" style="width:100%; height:500px" title="CleanRL Rainbow + Atari Tracked Experiments"></iframe> -->


[^1]: Hessel, M., Modayil, J., Hasselt, H.V., Schaul, T., Ostrovski, G., Dabney, W., Horgan, D., Piot, B., Azar, M.G., & Silver, D. (2018). Rainbow: Combining Improvements in Deep Reinforcement Learning. AAAI.