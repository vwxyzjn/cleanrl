# Random Network Distillation (RND)


## Overview

RND is an exploration bonus for RL methods that's easy to implement and enables significant progress in some hard exploration Atari games such as Montezuma's Revenge. We use [Proximal Policy Gradient](/rl-algorithms/ppo/#ppopy) as our RL method as used by original paper's [implementation](https://github.com/openai/random-network-distillation)


Original paper: 

* [Exploration by Random Network Distillation](https://arxiv.org/abs/1810.12894)

Our single-file implementations of DQN:

* [ppo_rnd.py](https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/ppo_rnd.py)
    * For playing Atari games. It uses convolutional layers and common atari-based pre-processing techniques.
    * Works with the Atari's pixel `Box` observation space of shape `(210, 160, 3)`
    * Works with the `Discerete` action space