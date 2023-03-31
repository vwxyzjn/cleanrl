# Random Network Distillation (RND)


## Overview

RND is an exploration bonus for RL methods that's easy to implement and enables significant progress in some hard exploration Atari games such as Montezuma's Revenge. We use [Proximal Policy Gradient](/rl-algorithms/ppo/#ppopy) as our RL method as used by original paper's [implementation](https://github.com/openai/random-network-distillation)


Original paper: 

* [Exploration by Random Network Distillation](https://arxiv.org/abs/1810.12894)

## Implemented Variants


| Variants Implemented      | Description |
| ----------- | ----------- |
| :material-github: [`ppo_rnd_envpool.py`](https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/ppo_rnd_envpool.py), :material-file-document: [docs](/rl-algorithms/ppo-rnd/#ppo_rnd_envpoolpy) | For Atari games, uses EnvPool. |


Below are our single-file implementations of RND:

## `ppo_rnd_envpool.py`

The [ppo_rnd_envpool.py](https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/ppo_rnd_envpool.py) has the following features:

* Uses the blazing fast [Envpool](https://github.com/sail-sg/envpool) vectorized environment.
* For playing Atari games. It uses convolutional layers and common atari-based pre-processing techniques.
* Works with the Atari's pixel `Box` observation space of shape `(210, 160, 3)`
* Works with the `Discerete` action space

???+ warning

    Note that `ppo_rnd_envpool.py` does not work in Windows :fontawesome-brands-windows: and MacOs :fontawesome-brands-apple:. See envpool's built wheels here: [https://pypi.org/project/envpool/#files](https://pypi.org/project/envpool/#files)

???+ bug

    EnvPool's vectorized environment **does not behave the same** as gym's vectorized environment, which causes a compatibility bug in our PPO implementation. When an action $a$ results in an episode termination or truncation, the environment generates $s_{last}$ as the terminated or truncated state; we then use $s_{new}$ to denote the initial state of the new episodes. Here is how the bahviors differ:

    * Under the vectorized environment of `envpool<=0.6.4`, the `obs` in `obs, reward, done, info = env.step(action)` is the truncated state $s_{last}$
    * Under the vectorized environment of `gym==0.23.1`, the `obs` in `obs, reward, done, info = env.step(action)` is the initial state $s_{new}$.

    This causes the $s_{last}$ to be off by one. 
    See [:material-github: sail-sg/envpool#194](https://github.com/sail-sg/envpool/issues/194) for more detail. However, it does not seem to impact performance, so we take a note here and await for the upstream fix.


### Usage

=== "poetry"

    ```bash
    poetry install -E envpool
    poetry run python cleanrl/ppo_rnd_envpool.py --help
    poetry run python cleanrl/ppo_rnd_envpool.py --env-id MontezumaRevenge-v5
    ```

=== "pip"

    ```bash
    pip install -r requirements/requirements-envpool.txt
    python cleanrl/ppo_rnd_envpool.py --help
    python cleanrl/ppo_rnd_envpool.py --env-id MontezumaRevenge-v5
    ```

### Explanation of the logged metrics

See [related docs](/rl-algorithms/ppo/#explanation-of-the-logged-metrics) for `ppo.py`.
Below is the additional metric for RND:

* `charts/episode_curiosity_reward`: episodic intrinsic rewards.
* `losses/fwd_loss`: the prediction error between predict network and target network, can also be viewed as a proxy of the curiosity reward in that batch.

### Implementation details

[ppo_rnd_envpool.py](https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/ppo_rnd_envpool.py) uses a customized `RecordEpisodeStatistics` to work with envpool but has the same other implementation details as `ppo_atari.py` (see [related docs](/rl-algorithms/ppo/#implementation-details_1)). Additionally, it has the following additional details:

1. We initialize the normalization parameters by stepping a random agent in the environment by `args.num_steps * args.num_iterations_obs_norm_init`. `args.num_iterations_obs_norm_init=50` comes from the [original implementation](https://github.com/openai/random-network-distillation/blob/f75c0f1efa473d5109d487062fd8ed49ddce6634/run_atari.py#L69).
1. We uses sticky action from [envpool](https://envpool.readthedocs.io/en/latest/env/atari.html?highlight=repeat_action_probability%20#options) to facilitate the exploration like done in the [original implementation](https://github.com/openai/random-network-distillation/blob/f75c0f1efa473d5109d487062fd8ed49ddce6634/atari_wrappers.py#L204).

### Experiment results

To run benchmark experiments, see :material-github: [benchmark/rnd.sh](https://github.com/vwxyzjn/cleanrl/blob/master/benchmark/rnd.sh). Specifically, execute the following command:

<script src="https://emgithub.com/embed.js?target=https%3A%2F%2Fgithub.com%2Fvwxyzjn%2Fcleanrl%2Fblob%2Fmaster%2Fbenchmark%2Frnd.sh%23L3-L8&style=github&type=code&showBorder=on&showLineNumbers=on&showFileMeta=on&showFullPath=on&showCopy=on"></script>

Below are the average episodic returns for `ppo_rnd_envpool.py`. To ensure the quality of the implementation, we compared the results against `openai/random-network-distillation`' PPO.

| Environment      | `ppo_rnd_envpool.py` | (Burda et al., 2019, Figure 7)[^1] 2000M steps
| ----------- | ----------- | ----------- |
| MontezumaRevengeNoFrameSkip-v4      | 7100 (1 seed)    | 8152 (3 seeds)  |

Note the MontezumaRevengeNoFrameSkip-v4 has same setting to MontezumaRevenge-v5.
Our benchmark has one seed due to limited compute resource and extreme long run time (~250 hours).


Learning curves:

<div class="grid-container">
    <img src="../ppo-rnd/MontezumaRevenge-v5.png">
    <img src="../ppo-rnd/MontezumaRevenge-v5-time.png">
</div>

<div></div>


Tracked experiments and game play videos:

<iframe src="https://wandb.ai/openrlbenchmark/openrlbenchmark/reports/-MontezumaRevenge-CleanRL-s-PPO-RND--VmlldzoyNTIyNjc5" style="width:100%; height:1200px" title="MontezumaRevenge: CleanRL's PPO + RND"></iframe>


[^1]:Burda, Yuri, et al. "Exploration by random network distillation." Seventh International Conference on Learning Representations. 2019.