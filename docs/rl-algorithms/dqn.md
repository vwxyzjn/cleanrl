# Deep Q-Learning (DQN)

## Overview

As an extension of the Q-learning, DQN's main technical contribution is the use of replay buffer and target network, both of which would help improve the stability of the algorithm.


Original papers: 

* [Human-level control through deep reinforcement learning
](https://www.nature.com/articles/nature14236)

## Implemented Variants


| Variants Implemented      | Description |
| ----------- | ----------- |
| :material-github: [`dqn.py`](https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/dqn.py), :material-file-document: [docs](/rl-algorithms/dqn/#dqnpy) | For classic control tasks like `CartPole-v1`. |
| :material-github: [`dqn_atari.py`](https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/dqn_atari.py), :material-file-document: [docs](/rl-algorithms/dqn/#dqn_ataripy) |  For playing Atari games. It uses convolutional layers and common atari-based pre-processing techniques. |

Below are our single-file implementations of DQN:

## `dqn.py`

The [dqn.py](https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/dqn.py) has the following features:

* Works with the `Box` observation space of low-level features
* Works with the `Discrete` action space
* Works with envs like `CartPole-v1`

### Implementation details

[dqn.py](https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/dqn.py) includes the 11 core implementation details:



## `dqn_atari.py`

The [dqn_atari.py](https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/dqn_atari.py) has the following features:

* For playing Atari games. It uses convolutional layers and common atari-based pre-processing techniques.
* Works with the Atari's pixel `Box` observation space of shape `(210, 160, 3)`
* Works with the `Discrete` action space