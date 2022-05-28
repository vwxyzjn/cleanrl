
Below are the implemented algorithms and their brief descriptions.

- [x] Deep Q-Learning (DQN)
    * [dqn.py](https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/dqn.py)
        * For discrete action space.
    * [dqn_atari.py](https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/dqn_atari.py)
        * For playing Atari games. It uses convolutional layers and common atari-based pre-processing techniques.
- [x] Categorical DQN (C51)
    * [c51.py](https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/c51.py)
        * For discrete action space.
    * [c51_atari.py](https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/c51_atari.py)
        * For playing Atari games. It uses convolutional layers and common atari-based pre-processing techniques.
    * [c51_atari_visual.py](https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/c51_atari_visual.py)
        * Adds return and q-values visulization for `dqn_atari.py`.
- [x] Proximal Policy Gradient (PPO) 
    * All of the PPO implementations below are augmented with some code-level optimizations. See https://costa.sh/blog-the-32-implementation-details-of-ppo.html for more details
    * [ppo.py](https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/ppo.py)
        * For discrete action space.
    * [ppo_continuous_action.py](https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/ppo_continuous_action.py)
        * For continuous action space. Also implemented Mujoco-specific code-level optimizations
    * [ppo_atari.py](https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/ppo_atari.py)
        * For playing Atari games. It uses convolutional layers and common atari-based pre-processing techniques.
- [x] Soft Actor Critic (SAC)
    * [sac_continuous_action.py](https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/sac_continuous_action.py)
        * For continuous action space.
- [x] Deep Deterministic Policy Gradient (DDPG)
    * [ddpg_continuous_action.py](https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/ddpg_continuous_action.py)
        * For continuous action space.
- [x] Twin Delayed Deep Deterministic Policy Gradient (TD3)
    * [td3_continuous_action.py](https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/td3_continuous_action.py)
        * For continuous action space.
- [x] Phasic Policy Gradient (PPG) 
    * [ppg_procgen.py](https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/ppg_procgen.py)
        * PPG implementation for Procgen
