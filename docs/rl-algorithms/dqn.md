# Deep Q-Learning (DQN)

As an extension of the Q-learning, DQN's main technical contribution is the use of replay buffer and target network, both of which would help improve the stability of the algorithm.


Original papers: 

* [Playing Atari with Deep Reinforcement Learning
](https://arxiv.org/abs/1312.5602)
* [Human-level control through deep reinforcement learning
](https://www.nature.com/articles/nature14236)

Our single-file implementations of DQN:

* [dqn.py](https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/dqn.py)
    * Works with the `Box` observation space of low-level features
    * Works with the `Discerete` action space
    * Works with envs like `CartPole-v1`
* [dqn_atari.py](https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/dqn_atari.py)
    * For playing Atari games. It uses convolutional layers and common atari-based pre-processing techniques.
    * Works with the Atari's pixel `Box` observation space of shape `(210, 160, 3)`
    * Works with the `Discerete` action space