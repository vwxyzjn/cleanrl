<!--
 SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 SPDX-License-Identifier: MIT

 Permission is hereby granted, free of charge, to any person obtaining a
 copy of this software and associated documentation files (the "Software"),
 to deal in the Software without restriction, including without limitation
 the rights to use, copy, modify, merge, publish, distribute, sublicense,
 and/or sell copies of the Software, and to permit persons to whom the
 Software is furnished to do so, subject to the following conditions:

 The above copyright notice and this permission notice shall be included in
 all copies or substantial portions of the Software.

 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
 THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 DEALINGS IN THE SOFTWARE.
-->

## `ppo_continuous_action_isaacgym.py`

:octicons-beaker-24: Experimental

The [ppo_continuous_action_isaacgym.py](https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/experimental/ppo_continuous_action_isaacgym.py) has the following features:

- Works with IsaacGymEnvs.
- Works with the `Box` observation space of low-level features
- Works with the `Box` (continuous) action space

[IsaacGymEnvs](https://github.com/NVIDIA-Omniverse/IsaacGymEnvs) is a hardware-accelerated (or GPU-accelerated) robotics simulation environment based on `torch`, which allows us to run thousands of simulation environments at the same time, empowering RL agents to learn many MuJoCo-style robotics tasks in minutes instead of hours. When creating an environment with IsaacGymEnvs via `isaacgymenvs.make("Ant")`, it creates a vectorized environment which produces GPU tensors as observations and take GPU tensors as actions to execute.

???+ info

    `ppo_continuous_action_isaacgym.py` works with most environments in IsaacGymEnvs but it does not work with the following environments yet:

    * AnymalTerrain
    * FrankaCabinet
    * ShadowHandOpenAI_FF
    * ShadowHandOpenAI_LSTM
    * Trifinger
    * Ingenuity Quadcopter

### Usage

Please go to [https://developer.nvidia.com/isaac-gym](https://developer.nvidia.com/isaac-gym) to download and install the latest version of Issac Gym. Make sure your python version is either 3.6, 3.7, or 3.8 (3.9 _not_ supported yet).

```bash
poetry run pip install path-to-isaac-gym
poetry install -E isaacgymenvs
python cleanrl/experimental/ppo_continuous_action_isaacgym.py --help
python cleanrl/experimental/ppo_continuous_action_isaacgym.py --env-id Ant
```

???+ info

    Note that **Isaac Gym** is the underlying core physics engine, and **IssacGymEnvs** is a collection of environments built on Isaac Gym.

### Explanation of the logged metrics

See [related docs](/rl-algorithms/ppo/#explanation-of-the-logged-metrics) for `ppo.py`.

### Implementation details

[ppo_continuous_action_isaacgym.py](https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/experimental/ppo_continuous_action_isaacgym.py) is based on `ppo_continuous_action.py` (see related [docs](/rl-algorithms/ppo/#ppo_continuous_actionpy)), with a few modifications:

1. **Different set of hyperparameters**: `ppo_continuous_action_isaacgym.py` uses hyperparameters primarily derived from [rl-games](https://github.com/Denys88/rl_games)' configuration (see [example](https://github.com/NVIDIA-Omniverse/IsaacGymEnvs/blob/main/isaacgymenvs/cfg/train/AntPPO.yaml)). The basic spirit is to run more `total_timesteps`, with larger `num_envs` and smaller `num_steps`.

| arguments         | `ppo_continuous_action.py` | `ppo_continuous_action_isaacgym.py` |
| ----------------- | -------------------------- | ----------------------------------- |
| --total-timesteps | 1000000                    | 30000000                            |
| --learning-rate   | 3e-4                       | 0.0026                              |
| --num-envs        | 1                          | 4096                                |
| --num-steps       | 2048                       | 16                                  |
| --anneal-lr       | True                       | False                               |
| --num-minibatches | 32                         | 2                                   |
| --update-epochs   | 10                         | 4                                   |
| --clip-vloss      | True                       | False                               |
| --vf-coef         | 0.5                        | 2                                   |
| --max-grad-norm   | 0.5                        | 1                                   |

1. **Slightly larger NN**: `ppo_continuous_action.py` uses the following NN:
   ```python
   self.critic = nn.Sequential(
       layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), 64)),
       nn.Tanh(),
       layer_init(nn.Linear(64, 64)),
       nn.Tanh(),
       layer_init(nn.Linear(64, 1), std=1.0),
   )
   self.actor_mean = nn.Sequential(
       layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), 64)),
       nn.Tanh(),
       layer_init(nn.Linear(64, 64)),
       nn.Tanh(),
       layer_init(nn.Linear(64, np.prod(envs.single_action_space.shape)), std=0.01),
   )
   ```
   while `ppo_continuous_action_isaacgym.py` uses the following NN:
   ```python
   self.critic = nn.Sequential(
       layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), 256)),
       nn.Tanh(),
       layer_init(nn.Linear(256, 256)),
       nn.Tanh(),
       layer_init(nn.Linear(256, 1), std=1.0),
   )
   self.actor_mean = nn.Sequential(
       layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), 256)),
       nn.Tanh(),
       layer_init(nn.Linear(256, 256)),
       nn.Tanh(),
       layer_init(nn.Linear(256, np.prod(envs.single_action_space.shape)), std=0.01),
   )
   ```
1. **No normalization and clipping**: `ppo_continuous_action_isaacgym.py` does _not_ do observation and reward normalization and clipping for simplicity. It does however offer an option to scale the rewards via `--reward-scaler 0.1`, which multiplies all the rewards obtained by `0.1` as an example.
1. **Remove all CPU-related code**: `ppo_continuous_action_isaacgym.py` needs to remove all CPU-related code (e.g. `action.cpu().numpy()`). This is because almost everything in IsaacGymEnvs happens in GPU. To do this, the major modifications include the following:
   1. Create a custom `RecordEpisodeStatisticsTorch` wrapper that records statstics using GPU tensors instead of `numpy` arrays.
   1. Avoid transferring the tensors to CPU. The related code in `ppo_continuous_action.py` looks like
   ```python
   next_obs, reward, done, info = envs.step(action.cpu().numpy())
   rewards[step] = torch.tensor(reward).to(device).view(-1)
   next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(done).to(device)
   ```
   and the related code in `ppo_continuous_action_isaacgym.py` looks like
   ```python
   next_obs, rewards[step], next_done, info = envs.step(action)
   ```

### Experiment results

To run benchmark experiments, see :material-github: [benchmark/ppo.sh](https://github.com/vwxyzjn/cleanrl/blob/master/benchmark/ppo.sh). Specifically, execute the following command:

<script src="https://emgithub.com/embed.js?target=https%3A%2F%2Fgithub.com%2Fvwxyzjn%2Fcleanrl%2Fblob%2F5184afc2b7d5032b56e6689175a17b7bad172771%2Fbenchmark%2Fppo.sh%23L32-L38&style=github&showBorder=on&showLineNumbers=on&showFileMeta=on&showCopy=on"></script>

Below are the average episodic returns for `ppo_continuous_action_isaacgym.py`. To ensure the quality of the implementation, we compared the results against [Denys88/rl_games](https://github.com/Denys88/rl_games)' PPO and present the training time (units being `s (seconds), m (minutes)`). The hardware used is a NVIDIA RTX A6000 in a 24 core machine.

| Environment (training time) | `ppo_continuous_action_isaacgym.py` | [Denys88/rl_games](https://github.com/Denys88/rl_games) |
| --------------------------- | ----------------------------------- | ------------------------------------------------------- |
| Cartpole (40s)              | 440.70 ± 19.75                      | 417.49 (30s)                                            |
| Ant (180s)                  | 4000.61 ± 793.18                    | 5873.05                                                 |
| Humanoid (22m)              | 4751.03 ± 851.21                    | 6254.73                                                 |
| Anymal (12m)                | 51.19 ± 12.72                       | 62.76                                                   |
| BallBalance (140s)          | 176.37 ± 58.42                      | 319.76                                                  |
| ShadowHand (100m)           | 1746.37 ± 179.02                    | 5713.74                                                 |
| AllegroHand (110m)          | 1308.33 ± 440.81                    | 3479.85                                                 |

Learning curves:

<div class="grid-container">
<img src="../ppo/isaacgymenvs/Cartpole.png">
<img src="../ppo/isaacgymenvs/Ant.png">
<img src="../ppo/isaacgymenvs/Humanoid.png">
<img src="../ppo/isaacgymenvs/BallBalance.png">
<img src="../ppo/isaacgymenvs/Anymal.png">
<img src="../ppo/isaacgymenvs/AllegroHand.png">
<img src="../ppo/isaacgymenvs/ShadowHand.png">
</div>

???+ info

    Note `ppo_continuous_action_isaacgym.py`'s performance seems poor compared to [Denys88/rl_games](https://github.com/Denys88/rl_games)' PPO. This is likely due to a few reasons.

    1. [Denys88/rl_games](https://github.com/Denys88/rl_games)' PPO uses different sets of tuned hyperparameters  and neural network architecture configuration for different tasks, whereas `ppo_continuous_action_isaacgym.py` only uses one neural network architecture and 2 set of hyperparameters (ignoring `--total-timesteps`).
    1. `ppo_continuous_action_isaacgym.py` does not use observation normalization (because in my preliminary testing for some reasons it did not help).

    While it should be possible to obtain higher scores with more tuning, the purpose of `ppo_continuous_action_isaacgym.py` is to hit a balance between simplicity and performance. I think `ppo_continuous_action_isaacgym.py` has relatively good performance with a concise codebase, which should be easy to modify and extend for practitioners.

<!-- Tracked experiments and game play videos:


Not available yet. -->
