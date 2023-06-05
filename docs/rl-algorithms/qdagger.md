# Qdagger

## Overview

As an extension of the Q-learning, DQN's main technical contribution is the use of replay buffer and target network, both of which would help improve the stability of the algorithm.


Original papers: 

* [Reincarnating Reinforcement Learning: Reusing Prior Computation to Accelerate Progress](https://arxiv.org/abs/2206.01626)

## Implemented Variants

| Variants Implemented      | Description |
| ----------- | ----------- |
| :material-github: [`qdagger_dqn_atari_impalacnn.py`](https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/qdagger_dqn_atari_impalacnn.py), :material-file-document: [docs](/rl-algorithms/qdagger/#qdagger_dqn_atari_impalacnnpy) |  For playing Atari games. It uses convolutional layers and common atari-based pre-processing techniques. |
| :material-github: [`qdagger_dqn_atari_jax_impalacnn.py`](https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/qdagger_dqn_atari_jax_impalacnn.py), :material-file-document: [docs](/rl-algorithms/qdagger/#qdagger_dqn_atari_jax_impalacnnpy) |  For playing Atari games. It uses convolutional layers and common atari-based pre-processing techniques. |


Below are our single-file implementations of Qdagger:


## `qdagger_dqn_atari_impalacnn.py`

The [qdagger_dqn_atari_impalacnn.py](https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/qdagger_dqn_atari_impalacnn.py) has the following features:

* For playing Atari games. It uses convolutional layers and common atari-based pre-processing techniques.
* Works with the Atari's pixel `Box` observation space of shape `(210, 160, 3)`
* Works with the `Discrete` action space

### Usage

```bash
poetry install -E atari
python cleanrl/qdagger_dqn_atari_impalacnn.py --env-id BreakoutNoFrameskip-v4 --teacher-policy-hf-repo cleanrl/BreakoutNoFrameskip-v4-dqn_atari-seed1
python cleanrl/qdagger_dqn_atari_impalacnn.py --env-id PongNoFrameskip-v4 --teacher-policy-hf-repo cleanrl/PongNoFrameskip-v4-dqn_atari-seed1
```

=== "poetry"

    ```bash
    poetry install -E atari
    poetry run python cleanrl/qdagger_dqn_atari_impalacnn.py --env-id BreakoutNoFrameskip-v4 --teacher-policy-hf-repo cleanrl/BreakoutNoFrameskip-v4-dqn_atari-seed1
    poetry run python cleanrl/qdagger_dqn_atari_impalacnn.py --env-id PongNoFrameskip-v4 --teacher-policy-hf-repo cleanrl/PongNoFrameskip-v4-dqn_atari-seed1
    ```

=== "pip"

    ```bash
    pip install -r requirements/requirements-atari.txt
    python cleanrl/qdagger_dqn_atari_impalacnn.py --env-id BreakoutNoFrameskip-v4 --teacher-policy-hf-repo cleanrl/BreakoutNoFrameskip-v4-dqn_atari-seed1
    python cleanrl/qdagger_dqn_atari_impalacnn.py --env-id PongNoFrameskip-v4 --teacher-policy-hf-repo cleanrl/PongNoFrameskip-v4-dqn_atari-seed1
    ```


### Explanation of the logged metrics

Running `python cleanrl/qdagger_dqn_atari_impalacnn.py` will automatically record various metrics such as actor or value losses in Tensorboard. Below is the documentation for these metrics:

* `charts/episodic_return`: episodic return of the game
* `charts/SPS`: number of steps per second
* `losses/td_loss`: the mean squared error (MSE) between the Q values at timestep $t$ and the Bellman update target estimated using the reward $r_t$ and the Q values at timestep $t+1$, thus minimizing the *one-step* temporal difference. Formally, it can be expressed by the equation below.
$$
    J(\theta^{Q}) = \mathbb{E}_{(s,a,r,s') \sim \mathcal{D}} \big[ (Q(s, a) - y)^2 \big],
$$
with the Bellman update target is $y = r + \gamma \, Q^{'}(s', a')$ and the replay buffer is $\mathcal{D}$.
* `losses/q_values`: implemented as `qf1(data.observations, data.actions).view(-1)`, it is the average Q values of the sampled data in the replay buffer; useful when gauging if under or over estimation happens.


### Implementation details

WIP

### Experiment results

WIP

## `qdagger_dqn_atari_jax_impalacnn.py`


The [qdagger_dqn_atari_jax_impalacnn.py](https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/qdagger_dqn_atari_jax_impalacnn.py) has the following features:

* Uses [Jax](https://github.com/google/jax), [Flax](https://github.com/google/flax), and [Optax](https://github.com/deepmind/optax) instead of `torch`.  [qdagger_dqn_atari_jax_impalacnn.py](https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/qdagger_dqn_atari_jax_impalacnn.py) is roughly 25%-50% faster than  [qdagger_dqn_atari_impalacnn.py](https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/qdagger_dqn_atari_impalacnn.py)
* For playing Atari games. It uses convolutional layers and common atari-based pre-processing techniques.
* Works with the Atari's pixel `Box` observation space of shape `(210, 160, 3)`
* Works with the `Discrete` action space

### Usage


=== "poetry"

    ```bash
    poetry install -E "atari jax"
    poetry run pip install --upgrade "jax[cuda]==0.3.17" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
    poetry run python cleanrl/qdagger_dqn_atari_jax_impalacnn.py --env-id BreakoutNoFrameskip-v4 --teacher-policy-hf-repo cleanrl/BreakoutNoFrameskip-v4-dqn_atari_jax-seed1
    poetry run python cleanrl/qdagger_dqn_atari_jax_impalacnn.py --env-id PongNoFrameskip-v4 --teacher-policy-hf-repo cleanrl/PongNoFrameskip-v4-dqn_atari_jax-seed1
    ```

=== "pip"

    ```bash
    pip install -r requirements/requirements-atari.txt
    pip install -r requirements/requirements-jax.txt
    pip install --upgrade "jax[cuda]==0.3.17" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
    python cleanrl/qdagger_dqn_atari_jax_impalacnn.py --env-id BreakoutNoFrameskip-v4 --teacher-policy-hf-repo cleanrl/BreakoutNoFrameskip-v4-dqn_atari_jax-seed1
    python cleanrl/qdagger_dqn_atari_jax_impalacnn.py --env-id PongNoFrameskip-v4 --teacher-policy-hf-repo cleanrl/PongNoFrameskip-v4-dqn_atari_jax-seed1
    ```


???+ warning

    Note that JAX does not work in Windows :fontawesome-brands-windows:. The official [docs](https://github.com/google/jax#installation) recommends using Windows Subsystem for Linux (WSL) to install JAX.

### Explanation of the logged metrics

See [related docs](/rl-algorithms/qdagger/#explanation-of-the-logged-metrics) for `qdagger_dqn_atari_impalacnn.py`.

### Implementation details

See [related docs](/rl-algorithms/qdagger/#implementation-details) for `qdagger_dqn_atari_impalacnn.py`.

### Experiment results

WIP
