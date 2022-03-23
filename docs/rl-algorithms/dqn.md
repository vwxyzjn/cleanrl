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

### Usage

```bash
poetry install -E atari
python cleanrl/dqn_atari.py --env-id BreakoutNoFrameskip-v4
python cleanrl/dqn_atari.py --env-id PongNoFrameskip-v4
```

### Implementation details

[dqn_atari.py](https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/dqn_atari.py) is based on (Mnih et al., 2015)[^1] but presents a few implementation differences:

1. `dqn_atari.py` use slightly different hyperparameters. Specifically,
    - `dqn_atari.py` uses the more popular Adam Optimizer with the `--learning-rate=1e-4` as follows:
        ```python
        optim.Adam(q_network.parameters(), lr=1e-4)
        ```
       whereas (Mnih et al., 2015)[^1] (Exntended Data Table 1) uses the RMSProp optimizer with `--learning-rate=2.5e-4`, gradient momentum `0.95`, squared gradient momentum `0.95`, and min squared gradient `0.01` as follows:
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
    - `dqn_atari.py` uses `--learning-starts=80000` whereas (Mnih et al., 2015)[^1] (Exntended Data Table 1) uses `--learning-starts=50000`.
    - `dqn_atari.py` uses `--target-network-frequency=1000` whereas (Mnih et al., 2015)[^1] (Exntended Data Table 1) uses `--learning-starts=10000`.
    - `dqn_atari.py` uses `--total-timesteps=10000000` (i.e., 10M timesteps = 40M frames because of frame-skipping) whereas (Mnih et al., 2015)[^1] uses `--total-timesteps=50000000` (i.e., 50M timesteps = 200M frames) (See "Training details" under "METHODS" on page 6 and the related source code [run_gpu#L32](https://github.com/deepmind/dqn/blob/9d9b1d13a2b491d6ebd4d046740c511c662bbe0f/run_gpu#L32), [dqn/train_agent.lua#L81-L82](https://github.com/deepmind/dqn/blob/9d9b1d13a2b491d6ebd4d046740c511c662bbe0f/dqn/train_agent.lua#L81-L82), and [dqn/train_agent.lua#L165-L169](https://github.com/deepmind/dqn/blob/9d9b1d13a2b491d6ebd4d046740c511c662bbe0f/dqn/train_agent.lua#L165-L169)).
    - `dqn_atari.py` uses `--end-e=0.01` (the final exploration epsilon) whereas (Mnih et al., 2015)[^1] (Exntended Data Table 1) uses `--end-e=0.1`.
    - `dqn_atari.py` uses `--exploration-fraction=0.1` whereas (Mnih et al., 2015)[^1] (Exntended Data Table 1) uses `--exploration-fraction=0.02` (all corresponds to 250000 steps or 1M frames being the frame that epsilon is annealed to `--end-e=0.1` ).
    - `dqn_atari.py` treats termination and truncation the same way due to the gym interface[^2] whereas (Mnih et al., 2015)[^1] correctly handles truncation.
1. `dqn_atari.py` use a self-contained evaluation scheme: `dqn_atari.py` reports the episodic returns obtained throughout training, whereas (Mnih et al., 2015)[^1] is trained with `--end-e=0.1` but reported episodic returns using a separate evaluation process with `--end-e=0.01` (See "Evaluation procedure" under "METHODS" on page 6).
1. `dqn_atari.py` rescales the gradient so that the norm of the parameters does not exceed `0.5` like done in PPO (:material-github: [ppo2/model.py#L102-L108](https://github.com/openai/baselines/blob/ea25b9e8b234e6ee1bca43083f8f3cf974143998/baselines/ppo2/model.py#L102-L108)). 


### Experiment results

PR :material-github: [vwxyzjn/cleanrl#124](https://github.com/vwxyzjn/cleanrl/pull/124) tracks our effort to conduct experiments, and the reprodudction instructions can be found at :material-github: [vwxyzjn/cleanrl/benchmark/dqn](https://github.com/vwxyzjn/cleanrl/tree/master/benchmark/dqn).

Below are the average episodic returns for `dqn_atari.py`. 


| Environment      | `dqn_atari.py` 10M steps | (Mnih et al., 2015)[^1] 50M steps | (Hessel et al., 2017, Figure 5)[^3] 
| ----------- | ----------- | ----------- | ---- |
| BreakoutNoFrameskip-v4      | 337.64 ± 69.47      |401.2 ± 26.9  | ~230 at 10M steps, ~300 at 50M steps
| PongNoFrameskip-v4  | 20.293 ± 0.37     |  18.9 ± 1.3 |  ~20 10M steps, ~20 at 50M steps 
| BeamRiderNoFrameskip-v4   | 6207.41 ± 1019.96        | 6846 ± 1619 | ~6000 10M steps, ~7000 at 50M steps 


Learning curves:

<div class="grid-container">
<img src="../dqn/BeamRiderNoFrameskip-v4.png">

<img src="../dqn/BreakoutNoFrameskip-v4.png">

<img src="../dqn/PongNoFrameskip-v4.png">
</div>


Tracked experiments and game play videos:

<iframe src="https://wandb.ai/openrlbenchmark/openrlbenchmark/reports/Atari-CleanRL-s-DQN--VmlldzoxNjk3NjYx" style="width:100%; height:500px" title="CleanRL DQN Tracked Experiments"></iframe>


[^1]:Mnih, V., Kavukcuoglu, K., Silver, D. et al. Human-level control through deep reinforcement learning. Nature 518, 529–533 (2015). https://doi.org/10.1038/nature14236
[^2]:\[Proposal\] Formal API handling of truncation vs termination. https://github.com/openai/gym/issues/2510
[^3]: Hessel, M., Modayil, J., Hasselt, H.V., Schaul, T., Ostrovski, G., Dabney, W., Horgan, D., Piot, B., Azar, M.G., & Silver, D. (2018). Rainbow: Combining Improvements in Deep Reinforcement Learning. AAAI.