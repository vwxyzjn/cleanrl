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


???+ warning

    `ppo_continuous_action_isaacgym.py` is temporarily deprecated. Please checkout the code in [https://github.com/vwxyzjn/cleanrl/releases/tag/v1.0.0](https://github.com/vwxyzjn/cleanrl/releases/tag/v1.0.0)


The [ppo_continuous_action_isaacgym.py](https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/ppo_continuous_action_isaacgym/ppo_continuous_action_isaacgym.py) has the following features:

- Works with IsaacGymEnvs.
- Works with the `Box` observation space of low-level features
- Works with the `Box` (continuous) action space

[IsaacGymEnvs](https://github.com/NVIDIA-Omniverse/IsaacGymEnvs) is a hardware-accelerated (or GPU-accelerated) robotics simulation environment based on `torch`, which allows us to run thousands of simulation environments at the same time, empowering RL agents to learn many MuJoCo-style robotics tasks in minutes instead of hours. When creating an environment with IsaacGymEnvs via `isaacgymenvs.make("Ant")`, it creates a vectorized environment which produces GPU tensors as observations and take GPU tensors as actions to execute.

???+ info

    Note that **Isaac Gym** is the underlying core physics engine, and **IssacGymEnvs** is a collection of environments built on Isaac Gym.

???+ info

    `ppo_continuous_action_isaacgym.py` works with most environments in IsaacGymEnvs but it does not work with the following environments yet:

    * AnymalTerrain
    * FrankaCabinet
    * ShadowHandOpenAI_FF
    * ShadowHandOpenAI_LSTM
    * Trifinger
    * Ingenuity Quadcopter

    🔥 we need contributors to work on supporting and tuning our PPO implementation in these envs. If you are interested, please read our [contribution guide](https://github.com/vwxyzjn/cleanrl/blob/master/CONTRIBUTING.md) and reach out!

### Usage

The installation of `isaacgym` requires a bit of work since it's not a standard Python package.

Please go to [https://developer.nvidia.com/isaac-gym](https://developer.nvidia.com/isaac-gym) to download and install the latest version of Issac Gym which should look like `IsaacGym_Preview_4_Package.tar.gz`. Put this `IsaacGym_Preview_4_Package.tar.gz` into the `~/Downloads/` folder. Make sure your python version is either 3.7, or 3.8 (3.9 _not_ supported yet).

```bash
# extract and move the content in `python` folder in the IsaacGym_Preview_4_Package.tar.gz
# into the `cleanrl/ppo_continuous_action_isaacgym/isaacgym/` folder
cp ~/Downloads/IsaacGym_Preview_4_Package.tar.gz IsaacGym_Preview_4_Package.tar.gz 
stat IsaacGym_Preview_4_Package.tar.gz
mkdir temp_isaacgym
tar -xf IsaacGym_Preview_4_Package.tar.gz -C temp_isaacgym
mv temp_isaacgym/isaacgym/python/* cleanrl/ppo_continuous_action_isaacgym/isaacgym
rm -rf temp_isaacgym

# if your global python version is not either 3.7 nor 3.8, you need to tell poetry specifically to use a 3.7 or 3.8 python
# e.g., `poetry env use /home/costa/.pyenv/versions/3.7.8/bin/python`
poetry install --with isaacgym
# if you are using NVIDIA's 30xx GPU, you need to specifically install cuda 11.3 wheels
# `poetry run pip install torch --upgrade --extra-index-url https://download.pytorch.org/whl/cu113`
poetry run python cleanrl/ppo_continuous_action_isaacgym/ppo_continuous_action_isaacgym.py --help
poetry run python cleanrl/ppo_continuous_action_isaacgym/ppo_continuous_action_isaacgym.py --env-id Ant
```

<script id="asciicast-510341" src="https://asciinema.org/a/510341.js" async></script>


???+ warning

    If you encounter the following installation error
    
    ```bash
    Python.h: No such file or directory
    #include <Python.h>
    ```

    or 

    ```bash
    libpython3.8.so.1.0: cannot open shared object file: No such file or directory
    ```

    It usually means your python distribution does not include the shared library files. If you are ubuntu, you can install the following packages:
    
    ```bash
    sudo apt-get install libpython3.8-dev # or sudo apt-get install libpython3.7-dev
    ```

    If you are using [pyenv](https://github.com/pyenv/pyenv), you may try the following:
    
    ```bash
    env PYTHON_CONFIGURE_OPTS="--enable-shared" pyenv install 3.7.8
    ```

### Explanation of the logged metrics

See [related docs](/rl-algorithms/ppo/#explanation-of-the-logged-metrics) for `ppo.py`.

Additionally, `charts/consecutive_successes` means the number of consecutive episodes that the agent has successfully manipulating the rubix cube to the desired state.

### Implementation details

[ppo_continuous_action_isaacgym.py](https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/ppo_continuous_action_isaacgym/ppo_continuous_action_isaacgym.py) is based on `ppo_continuous_action.py` (see related [docs](/rl-algorithms/ppo/#ppo_continuous_actionpy)), with a few modifications:

1. **Different set of hyperparameters**: `ppo_continuous_action_isaacgym.py` uses hyperparameters primarily derived from [rl-games](https://github.com/Denys88/rl_games)' configuration (see [example](https://github.com/NVIDIA-Omniverse/IsaacGymEnvs/blob/main/isaacgymenvs/cfg/train/AntPPO.yaml)). The basic spirit is to run more `total_timesteps`, with larger `num_envs` and smaller `num_steps`.

| arguments         | `ppo_continuous_action.py` | `ppo_continuous_action_isaacgym.py` | `ppo_continuous_action_isaacgym.py` (for `ShadowHand` and `AllegroHand`) |
| ----------------- | -------------------------- | ----------------------------------- | ----------------------------------- |
| --total-timesteps | 1000000                    | 30000000                            | 600000000                           |
| --learning-rate   | 3e-4                       | 0.0026                              | 0.0026                              |
| --num-envs        | 1                          | 4096                                | 8192                                |
| --num-steps       | 2048                       | 16                                  | 8                                   |
| --anneal-lr       | True                       | False                               | False                               |
| --num-minibatches | 32                         | 2                                   | 4                                   |
| --update-epochs   | 10                         | 4                                   | 5                                   |
| --clip-vloss      | True                       | False                               | False                               |
| --vf-coef         | 0.5                        | 2                                   | 2                                   |
| --max-grad-norm   | 0.5                        | 1                                   | 1                                   |
| --reward-scaler   | N/A                        | 1                                   | 0.01                                |

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
1. **No normalization and clipping**: `ppo_continuous_action_isaacgym.py` does _not_ do observation and reward normalization and clipping for simplicity. It does however optionally offer an option to scale the rewards via `--reward-scaler x`, which multiplies all the rewards obtained by `x` as an example.
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

<script src="https://emgithub.com/embed.js?target=https%3A%2F%2Fgithub.com%2Fvwxyzjn%2Fcleanrl%2Fblob%2Fmaster%2Fbenchmark%2Fppo.sh%23L61-L73&style=github&showBorder=on&showLineNumbers=on&showFileMeta=on&showCopy=on"></script>

Below are the average episodic returns for `ppo_continuous_action_isaacgym.py`. To ensure the quality of the implementation, we compared the results against [Denys88/rl_games](https://github.com/Denys88/rl_games)' PPO and present the training time (units being `s (seconds), m (minutes)`). The hardware used is a NVIDIA RTX A6000 in a 24 core machine.

| Environment (training time) | `ppo_continuous_action_isaacgym.py` | [Denys88/rl_games](https://github.com/Denys88/rl_games) |
| --------------------------- | ----------------------------------- | ------------------------------------------------------- |
| Cartpole (40s)              | 413.66 ± 120.93                     | 417.49 (30s)                                            |
| Ant (240s)                  | 3953.30 ± 667.086                   | 5873.05                                                 |
| Humanoid (350s)             | 2987.95 ± 257.60                    | 6254.73                                                 |
| Anymal (317s)               | 29.34 ± 17.80                       | 62.76                                                   |
| BallBalance (160s)          | 161.92 ± 89.20                      | 319.76                                                  |
| AllegroHand (200m)          | 762.93 ± 427.92                     | 3479.85                                                 |
| ShadowHand (130m)           | 427.16 ± 161.79                     | 5713.74                                                 |

Learning curves:

<div class="grid-container">
<img src="../ppo/isaacgymenvs/Cartpole.png">
<img src="../ppo/isaacgymenvs/Cartpole-time.png">
<img src="../ppo/isaacgymenvs/Ant.png">
<img src="../ppo/isaacgymenvs/Ant-time.png">
<img src="../ppo/isaacgymenvs/Humanoid.png">
<img src="../ppo/isaacgymenvs/Humanoid-time.png">
<img src="../ppo/isaacgymenvs/BallBalance.png">
<img src="../ppo/isaacgymenvs/BallBalance-time.png">
<img src="../ppo/isaacgymenvs/Anymal.png">
<img src="../ppo/isaacgymenvs/Anymal-time.png">
<img src="../ppo/isaacgymenvs/AllegroHand.png">
<img src="../ppo/isaacgymenvs/AllegroHand-time.png">
<img src="../ppo/isaacgymenvs/AllegroHand-c.png">
<img src="../ppo/isaacgymenvs/AllegroHand-c-time.png">
<img src="../ppo/isaacgymenvs/ShadowHand.png">
<img src="../ppo/isaacgymenvs/ShadowHand-time.png">
<img src="../ppo/isaacgymenvs/ShadowHand-c.png">
<img src="../ppo/isaacgymenvs/ShadowHand-c-time.png">
</div>

???+ info

    Note `ppo_continuous_action_isaacgym.py`'s performance seems poor compared to [Denys88/rl_games](https://github.com/Denys88/rl_games)' PPO. This is likely due to a few reasons.

    1. [Denys88/rl_games](https://github.com/Denys88/rl_games)' PPO uses different sets of tuned hyperparameters  and neural network architecture configuration for different tasks, whereas `ppo_continuous_action_isaacgym.py` only uses one neural network architecture and 2 set of hyperparameters (ignoring `--total-timesteps`).
    1. `ppo_continuous_action_isaacgym.py` does not use observation normalization (because in my preliminary testing for some reasons it did not help).

    While it should be possible to obtain higher scores with more tuning, the purpose of `ppo_continuous_action_isaacgym.py` is to hit a balance between simplicity and performance. I think `ppo_continuous_action_isaacgym.py` has relatively good performance with a concise codebase, which should be easy to modify and extend for practitioners.


Tracked experiments and game play videos:


<iframe src="https://wandb.ai/openrlbenchmark/openrlbenchmark/reports/Isaac-Gym-CleanRL-s-PPO--VmlldzoyMzQzNzMz" style="width:100%; height:500px" title="Isaac-Gym-CleanRL-s-PPO"></iframe>


Old Learning curves w/ Isaac Gym Preview 3 (no longer available in Nvidia's website for download):

<div class="grid-container">
<img src="../ppo/isaacgymenvs/old/Cartpole.png">
<img src="../ppo/isaacgymenvs/old/Ant.png">
<img src="../ppo/isaacgymenvs/old/Humanoid.png">
<img src="../ppo/isaacgymenvs/old/BallBalance.png">
<img src="../ppo/isaacgymenvs/old/Anymal.png">
<img src="../ppo/isaacgymenvs/old/AllegroHand.png">
<img src="../ppo/isaacgymenvs/old/ShadowHand.png">
</div>

???+ info

    Note the `AllegroHand` and `ShadowHand` experiments used the following command `ppo_continuous_action_isaacgym.py --track --capture_video --num-envs 16384 --num-steps 8 --update-epochs 5 --reward-scaler 0.01 --total-timesteps 600000000 --record-video-step-frequency 3660`. Costa: I was able to run this during my internship at NVIDIA, but in my home setup, the computer has less GPU memory which makes it hard to replicate the results w/ `--num-envs 16384`.


