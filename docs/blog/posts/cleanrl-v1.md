---
date: 2022-10-05
authors: [costa]
description: >
  🎉 I am thrilled to announce the v1.0.0b2 CleanRL Beta Release. This new release comes with exciting new features. First, we now support JAX-based learning algorithms, which are usually faster than the `torch` equivalent! Here are the docs of the new JAX-based DQN, TD3, and DDPG implementations:
categories:
  - Blog
---



# CleanRL v1 Release


🎉 We are thrilled to announce the v1.0.0 CleanRL Release. CleanRL has come a long way making high-quality deep reinforcement learning implementations easy to understand and reproducible. Along with our [CleanRL paper's recent publication in Journal of Machine Learning Research](https://www.jmlr.org/papers/v23/21-1342.html), our v1.0.0 release includes reworked documentation, new algorithm variants, support for JAX, hyperparameter tuning utilties, and more. This release is a major milestone for the project and we are excited to share it with you. Over 90 PRs were merged to make this release possible. We would like to thank all the contributors who made this release possible.


<!-- more -->

## Reworked documentation

One of the biggest change of the v1 release is the added documentation at [docs.cleanrl.dev](https://docs.cleanrl.dev). Having great documentation is important for building a reliable and reproducible project. We have reworked the documentation to make it easier to understand and use. For each implemented algorithm, we have documented as much as we can to promote transparency:

* [Short description of the algorithm and references](/rl-algorithms/ppo/#overview)
* [A list of implemented varaint](/rl-algorithms/ppo/#implemented-variants)
* [The usage information](/rl-algorithms/ppo/#usage)
* [The explanation of the logged metrics](/rl-algorithms/ppo/#explanation-of-the-logged-metrics)
* [The documentation of implementation details](/rl-algorithms/ppo/#implementation-details)
* [Experimental results](/rl-algorithms/ppo/#experiment-results)

Here is a list of the algorithm varaints and their documentation:


| Algorithm      | Variants Implemented |
| ----------- | ----------- |
| ✅ [Proximal Policy Gradient (PPO)](https://arxiv.org/pdf/1707.06347.pdf)  | :material-github: [`ppo.py`](https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/ppo.py),  :material-file-document: [docs](/rl-algorithms/ppo/#ppopy) |
| | :material-github: [`ppo_atari.py`](https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/ppo_atari.py),  :material-file-document: [docs](/rl-algorithms/ppo/#ppo_ataripy)
| | :material-github: [`ppo_continuous_action.py`](https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/ppo_continuous_action.py),  :material-file-document: [docs](/rl-algorithms/ppo/#ppo_continuous_actionpy)
| | :material-github: [`ppo_atari_lstm.py`](https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/ppo_atari_lstm.py),  :material-file-document: [docs](/rl-algorithms/ppo/#ppo_atari_lstmpy)
| | :material-github: [`ppo_atari_envpool.py`](https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/ppo_atari_envpool.py),  :material-file-document: [docs](/rl-algorithms/ppo/#ppo_atari_envpoolpy)
| | :material-github: [`ppo_atari_envpool_xla_jax.py`](https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/ppo_atari_envpool_xla_jax.py),  :material-file-document: [docs](/rl-algorithms/ppo/#ppo_atari_envpool_xla_jaxpy)
| | :material-github: [`ppo_procgen.py`](https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/ppo_procgen.py),  :material-file-document: [docs](/rl-algorithms/ppo/#ppo_procgenpy)
| | :material-github: [`ppo_atari_multigpu.py`](https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/ppo_atari_multigpu.py),  :material-file-document: [docs](/rl-algorithms/ppo/#ppo_atari_multigpupy)
| | :material-github: [`ppo_pettingzoo_ma_atari.py`](https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/ppo_pettingzoo_ma_atari.py),  :material-file-document: [docs](/rl-algorithms/ppo/#ppo_pettingzoo_ma_ataripy)
| | :material-github: [`ppo_continuous_action_isaacgym.py`](https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/ppo_continuous_action_isaacgym/ppo_continuous_action_isaacgym.py),  :material-file-document: [docs](/rl-algorithms/ppo/#ppo_continuous_action_isaacgympy)
| ✅ [Deep Q-Learning (DQN)](https://web.stanford.edu/class/psych209/Readings/MnihEtAlHassibis15NatureControlDeepRL.pdf) | :material-github: [`dqn.py`](https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/dqn.py), :material-file-document: [docs](/rl-algorithms/dqn/#dqnpy) |
| | :material-github: [`dqn_atari.py`](https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/dqn_atari.py), :material-file-document: [docs](/rl-algorithms/dqn/#dqn_ataripy) |
| | :material-github: [`dqn_jax.py`](https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/dqn_jax.py), :material-file-document: [docs](/rl-algorithms/dqn/#dqn_jaxpy) |
| | :material-github: [`dqn_atari_jax.py`](https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/dqn_atari_jax.py), :material-file-document: [docs](/rl-algorithms/dqn/#dqn_atari_jaxpy) |
| ✅ [Categorical DQN (C51)](https://arxiv.org/pdf/1707.06887.pdf) | :material-github: [`c51.py`](https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/c51.py), :material-file-document: [docs](/rl-algorithms/c51/#c51py) |
| | :material-github: [`c51_atari.py`](https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/c51_atari.py), :material-file-document: [docs](/rl-algorithms/c51/#c51_ataripy) |
| ✅ [Soft Actor-Critic (SAC)](https://arxiv.org/pdf/1812.05905.pdf) | :material-github: [`sac_continuous_action.py`](https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/sac_continuous_action.py), :material-file-document: [docs](/rl-algorithms/sac/#sac_continuous_actionpy) |
| ✅ [Deep Deterministic Policy Gradient (DDPG)](https://arxiv.org/pdf/1509.02971.pdf) | :material-github: [`ddpg_continuous_action.py`](https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/ddpg_continuous_action.py), :material-file-document: [docs](/rl-algorithms/ddpg/#ddpg_continuous_actionpy) |
| | :material-github: [`ddpg_continuous_action_jax.py`](https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/ddpg_continuous_action_jax.py),  :material-file-document: [docs](/rl-algorithms/ddpg/#ddpg_continuous_action_jaxpy)
| ✅ [Twin Delayed Deep Deterministic Policy Gradient (TD3)](https://arxiv.org/pdf/1802.09477.pdf) | :material-github: [`td3_continuous_action.py`](https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/td3_continuous_action.py), :material-file-document: [docs](/rl-algorithms/td3/#td3_continuous_actionpy) |
|  | :material-github: [`td3_continuous_action_jax.py`](https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/td3_continuous_action_jax.py), :material-file-document: [docs](/rl-algorithms/td3/#td3_continuous_action_jaxpy) |
| ✅ [Phasic Policy Gradient (PPG)](https://arxiv.org/abs/2009.04416) | :material-github: [`ppg_procgen.py`](https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/ppg_procgen.py), :material-file-document: [docs](/rl-algorithms/ppg/#ppg_procgenpy) |
| ✅ [Random Network Distillation (RND)](https://arxiv.org/abs/1810.12894) | :material-github: [`ppo_rnd_envpool.py`](https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/ppo_rnd_envpool.py), :material-file-document: [docs](/rl-algorithms/ppo-rnd/#ppo_rnd_envpoolpy) |


We also improved the [contribution guide](https://github.com/vwxyzjn/cleanrl/blob/master/CONTRIBUTING.md) to make it easier for new contributors to get started. We are still working on improving the documentation. If you have any suggestions, please let us know in the [GitHub Issues](https://github.com/vwxyzjn/cleanrl/issues).


## New algorithm variants, support for JAX

We now support JAX-based learning algorithm varaints, which are usually faster than the `torch` equivalent! Here are the docs of the new JAX-based DQN, TD3, and DDPG implementations:


* [`dqn_atari_jax.py`](https://docs.cleanrl.dev/rl-algorithms/dqn/#dqn_atari_jaxpy) @kinalmehta in [:material-github: vwxyzjn/cleanrl#222](https://github.com/vwxyzjn/cleanrl/pull/222)
* [`dqn_jax.py`](https://docs.cleanrl.dev/rl-algorithms/dqn/#dqn_jaxpy) @kinalmehta in [:material-github: vwxyzjn/cleanrl#222](https://github.com/vwxyzjn/cleanrl/pull/222)
* [`td3_continuous_action_jax.py`](https://docs.cleanrl.dev/rl-algorithms/td3/#td3_continuous_action_jaxpy) by @joaogui1 in [:material-github: vwxyzjn/cleanrl#225](https://github.com/vwxyzjn/cleanrl/pull/225)
* [`ddpg_continuous_action_jax.py`](https://docs.cleanrl.dev/rl-algorithms/td3/#td3_continuous_action_jaxpy) by @vwxyzjn in [:material-github: vwxyzjn/cleanrl#187](https://github.com/vwxyzjn/cleanrl/pull/187)
* [`ppo_atari_envpool_xla_jax.py`](https://docs.cleanrl.dev/rl-algorithms/ppo/#ppo_atari_envpool_xla_jaxpy) by @vwxyzjn in [:material-github: vwxyzjn/cleanrl#227](https://github.com/vwxyzjn/cleanrl/pull/227)

For example, below are the benchmark of DDPG + JAX (see docs [here](/rl-algorithms/ddpg/#ddpg_continuous_action_jaxpy) for further detail):


<div class="grid-container">
<img src="/rl-algorithms/ddpg-jax/HalfCheetah-v2.png">
<img src="/rl-algorithms/ddpg-jax/HalfCheetah-v2-time.png">
</div>

Other new algorithm variants include multi-GPU PPO, PPO prototype that works with [Isaac Gym](https://github.com/NVIDIA-Omniverse/IsaacGymEnvs), multi-agent Atari PPO, and refactored PPG and PPO-RND implementations:

* [`ppo_atari_multigpu.pu`](https://docs.cleanrl.dev/rl-algorithms/ppo/#ppo_atari_multigpupy) by @vwxyzjn in [:material-github: vwxyzjn/cleanrl#178]( https://github.com/vwxyzjn/cleanrl/pull/178)
* [`ppo_continuous_action_isaacgym.py`](https://docs.cleanrl.dev/rl-algorithms/ppo/#ppo_continuous_action_isaacgympy) by @vwxyzjn in [:material-github: vwxyzjn/cleanrl#233](https://github.com/vwxyzjn/cleanrl/pull/233)
* [`ppo_pettingzoo_ma_atari.py`](https://docs.cleanrl.dev/rl-algorithms/ppo/#ppo_pettingzoo_ma_ataripy) by @vwxyzjn in [:material-github: vwxyzjn/cleanrl#188](https://github.com/vwxyzjn/cleanrl/pull/188)
* [`ppg_procgen.py`](https://docs.cleanrl.dev/rl-algorithms/ppg/#ppg_procgenpy) by @Dipamc77 in [:material-github: vwxyzjn/cleanrl#186](https://github.com/vwxyzjn/cleanrl/pull/186)
* [`ppo_rnd_envpoolpy.py`](https://docs.cleanrl.dev/rl-algorithms/ppo-rnd/#ppo_rnd_envpoolpy) by @yooceii in [:material-github: vwxyzjn/cleanrl#151](https://github.com/vwxyzjn/cleanrl/pull/151)



## Tooling improvements

We love tools! The v1.0.0 release comes with a series of DevOps improvements, including pre-commit utilities, CI integration with GitHub to run end-to-end test cases. We also make available a new hyperparameter tuning tool and a new tool for running benchmark experiments.

### DevOps

We added a pre-commit utility to help contributors to format their code, check for spelling, and removing unused variables and imports before submitting a pull request (see [Contribution guide](http://127.0.0.1:8000/contribution/#pre-commit-utilities) for more detail).

<img src="/static/pre-commit.png">


To ensure our single-file implementations can run without error, we also added CI/CD pipeline which now runs end-to-end test cases for all the algorithm varaints. The pipeline also tests builds across different operating systems, such as Linux, macOS, and Windows (see [here](https://github.com/vwxyzjn/cleanrl/actions/runs/3401991711/usage) as an example). GitHub actions are free for open source projects, and we are very happy to have this tool to help us maintain the project. 

<img src="/static/blog/cleanrl-v1/github-action.png">

### Hyperparameter tuning utilties

We now have preliminary support for hyperparameter tuning via `optuna` (see [docs](https://docs.cleanrl.dev/advanced/hyperparameter-tuning/)), which is designed to help researchers to find **a single set** of hyperparameters that work well with a kind of games. The current API looks like below:

```python
import optuna
from cleanrl_utils.tuner import Tuner
tuner = Tuner(
    script="cleanrl/ppo.py",
    metric="charts/episodic_return",
    metric_last_n_average_window=50,
    direction="maximize",
    aggregation_type="average",
    target_scores={
        "CartPole-v1": [0, 500],
        "Acrobot-v1": [-500, 0],
    },
    params_fn=lambda trial: {
        "learning-rate": trial.suggest_loguniform("learning-rate", 0.0003, 0.003),
        "num-minibatches": trial.suggest_categorical("num-minibatches", [1, 2, 4]),
        "update-epochs": trial.suggest_categorical("update-epochs", [1, 2, 4, 8]),
        "num-steps": trial.suggest_categorical("num-steps", [5, 16, 32, 64, 128]),
        "vf-coef": trial.suggest_uniform("vf-coef", 0, 5),
        "max-grad-norm": trial.suggest_uniform("max-grad-norm", 0, 5),
        "total-timesteps": 100000,
        "num-envs": 16,
    },
    pruner=optuna.pruners.MedianPruner(n_startup_trials=5),
    sampler=optuna.samplers.TPESampler(),
)
tuner.tune(
    num_trials=100,
    num_seeds=3,
)
```

### Benchmarking utilities

We also added a new tool for running benchmark experiments. The tool is designed to help researchers to quickly run benchmark experiments across different algorithms environments with some random seeds. The tool lives in the `cleanrl_utils.benchmark` module, and the users can run commands such as:

```bash
OMP_NUM_THREADS=1 xvfb-run -a python -m cleanrl_utils.benchmark \
    --env-ids CartPole-v1 Acrobot-v1 MountainCar-v0 \
    --command "poetry run python cleanrl/ppo.py --cuda False --track --capture-video" \
    --num-seeds 3 \
    --workers 5
```

which will run the `ppo.py` script with `--cuda False --track --capture-video` arguments across 3 random seeds for 3 environments. It uses `multiprocessing` to create a pool of 5 workers run the experiments in parallel.




## What’s next?

It is an exciting time and new improvements are coming to CleanRL. We plan to add more JAX-based implementations, huggingface integration, and some RLops prototypes. CleanRL is a community-based project and we always welcome new contributors. If there is an algorithm or new feature you would like to contribute, feel free to chat with us on our [discord channel](https://discord.gg/D6RCjA6sVT) or raise a GitHub issue. 

### More JAX implementations

More JAX-based implementation are coming. [Antonin Raffin](https://github.com/araffin), the core maintainer of [Stable-baselines3](https://github.com/DLR-RM/stable-baselines3), [SBX](https://github.com/araffin/sbx), and [rl-baselines3-zoo](https://github.com/DLR-RM/rl-baselines3-zoo), is contributing an optimized Soft Actor Critic implementation in JAX ([:material-github: vwxyzjn/cleanrl#300](https://github.com/vwxyzjn/cleanrl/pull/300)) and TD3+TQC, and DroQ ([:material-github: vwxyzjn/cleanrl#272](https://github.com/vwxyzjn/cleanrl/pull/272). These are incredibly exciting new algorithms. For example, DroQ is extremely sample effcient and can obtain ~5000 return in `HalfCheetah-v3` in just 100k steps ([tracked sbx experiment](https://wandb.ai/openrlbenchmark/sbx/runs/1tyzq3tu)).

### Huggingface integration

[Huggingface Hub 🤗](https://huggingface.co/models) is a great platform for sharing and collaborating models. We are working on a new integration with Huggingface Hub to make it easier for researchers to share their RL models and benchmark them against other models ([:material-github: vwxyzjn/cleanrl#292](https://github.com/vwxyzjn/cleanrl/pull/292)). Stay tuned! In the future, we will have a simple snippet for loading models like below:

```python
import random
from typing import Callable

import gym
import numpy as np
import torch


def evaluate(
    model_path: str,
    make_env: Callable,
    env_id: str,
    eval_episodes: int,
    run_name: str,
    Model: torch.nn.Module,
    device: torch.device,
    epsilon: float = 0.05,
    capture_video: bool = True,
):
    envs = gym.vector.SyncVectorEnv([make_env(env_id, 0, 0, capture_video, run_name)])
    model = Model(envs).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    obs = envs.reset()
    episodic_returns = []
    while len(episodic_returns) < eval_episodes:
        if random.random() < epsilon:
            actions = np.array([envs.single_action_space.sample() for _ in range(envs.num_envs)])
        else:
            q_values = model(torch.Tensor(obs).to(device))
            actions = torch.argmax(q_values, dim=1).cpu().numpy()
        next_obs, _, _, infos = envs.step(actions)
        for info in infos:
            if "episode" in info.keys():
                print(f"eval_episode={len(episodic_returns)}, episodic_return={info['episode']['r']}")
                episodic_returns += [info["episode"]["r"]]
        obs = next_obs

    return episodic_returns


if __name__ == "__main__":
    from huggingface_hub import hf_hub_download

    from cleanrl.dqn import QNetwork, make_env

    model_path = hf_hub_download(repo_id="cleanrl/CartPole-v1-dqn-seed1", filename="q_network.pth")
```


<img src="/static/blog/cleanrl-v1/hf.png">

### RLops

How do we know the effect of a new feature / bug fix? DRL is brittle and has a series of reproducibility issues — even bug fixes sometimes could introduce performance regression (e.g., see [how a bug fix of contact force in MuJoCo results in worse performance for PPO](https://github.com/openai/gym/pull/2762#discussion_r853488897)). Therefore, it is essential to understand how the proposed changes impact the performance of the algorithms. 

We built a prototype tool that allows us to compare the performance of the library at different versions of the tracked experiment. With this tool, we can confidently merge new features / bug fixes without worrying about introducing catastrophic regression. The users can run commands such as:

```bash
python -m cleanrl_utils.rlops --exp-name ddpg_continuous_action \
    --wandb-project-name cleanrl \
    --wandb-entity openrlbenchmark \
    --tags 'pr-299' 'rlops-pilot' \
    --env-ids HalfCheetah-v2 Walker2d-v2 Hopper-v2 InvertedPendulum-v2 Humanoid-v2 Pusher-v2 \
    --output-filename compare.png \
    --scan-history \
    --metric-last-n-average-window 100 \
    --report
```

which generates the following image

<img max-width="500px" src="/static/blog/cleanrl-v1/rlops.png">


## New CleanRL Supported Publications

* Weng, Jiayi, Min Lin, Shengyi Huang, Bo Liu, Denys Makoviichuk, Viktor Makoviychuk, Zichen Liu et al. "Envpool: A highly parallel reinforcement learning environment execution engine." In Thirty-sixth Conference on Neural Information Processing Systems Datasets and Benchmarks Track. [https://openreview.net/forum?id=BubxnHpuMbG](https://openreview.net/forum?id=BubxnHpuMbG)
* Huang, Shengyi, Rousslan Fernand Julien Dossa, Antonin Raffin, Anssi Kanervisto, and Weixun Wang. "The 37 Implementation Details of Proximal Policy Optimization." International Conference on Learning Representations 2022 Blog Post Track, [https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/](https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/)
* Huang, Shengyi, and Santiago Ontañón. "A closer look at invalid action masking in policy gradient algorithms." The International FLAIRS Conference Proceedings, 35. [https://journals.flvc.org/FLAIRS/article/view/130584](https://journals.flvc.org/FLAIRS/article/view/130584)
* Schmidt, Dominik, and Thomas Schmied. "Fast and Data-Efficient Training of Rainbow: an Experimental Study on Atari." Deep Reinforcement Learning Workshop at the 35th Conference on Neural Information Processing Systems, [https://arxiv.org/abs/2111.10247](https://arxiv.org/abs/2111.10247)


## Thanks

I would like to cordially thank the core dev members @dosssman @yooceii @Dipamc @kinalmehta @bragajj for their efforts in helping maintain the CleanRL repository.  I would also like to give a shout-out to our new contributors @cool-RR, @Howuhh, @jseppanen, @joaogui1, @ALPH2H, @ElliotMunro200, @WillDudley, and @sdpkjc.

