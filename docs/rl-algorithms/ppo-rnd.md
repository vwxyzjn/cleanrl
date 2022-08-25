# Random Network Distillation (RND)


## Overview

RND is an exploration bonus for RL methods that's easy to implement and enables significant progress in some hard exploration Atari games such as Montezuma's Revenge. We use [Proximal Policy Gradient](/rl-algorithms/ppo/#ppopy) as our RL method as used by original paper's [implementation](https://github.com/openai/random-network-distillation)


Original paper: 

* [Exploration by Random Network Distillation](https://arxiv.org/abs/1810.12894)

Our single-file implementations of RND:

* [ppo_rnd_envpool.py](https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/ppo_rnd_envpool.py)
    * Uses the blazing fast [Envpool](https://github.com/sail-sg/envpool) vectorized environment.
    * For playing Atari games. It uses convolutional layers and common atari-based pre-processing techniques.
    * Works with the Atari's pixel `Box` observation space of shape `(210, 160, 3)`
    * Works with the `Discerete` action space

???+ warning

    Note that `ppo_rnd_envpool.py` does not work in Windows :fontawesome-brands-windows: and MacOs :fontawesome-brands-apple:. See envpool's built wheels here: [https://pypi.org/project/envpool/#files](https://pypi.org/project/envpool/#files)


## Implemented Variants


| Variants Implemented      | Description |
| ----------- | ----------- |
| :material-github: [`ppo_rnd_envpool.py`](https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/ppo_rnd_envpool.py), :material-file-document: [docs](/rl-algorithms/ppo-rnd/#ppo_rnd_envpoolpy) | For continuous action space |


Below are our single-file implementations of TD3:

## `ppo_rnd_envpool.py`

### Usage

```bash
poetry install -E envpool
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

<script src="https://emgithub.com/embed.js?target=https%3A%2F%2Fgithub.com%2Fvwxyzjn%2Fcleanrl%2Fblob%2F64908e7dbb669642d5d4cd04bab505fe29184c88%2Fbenchmark%2Frnd.sh%23L3-L8&style=github&type=code&showBorder=on&showLineNumbers=on&showFileMeta=on&showFullPath=on&showCopy=on"></script>

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