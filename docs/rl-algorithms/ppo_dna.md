# Proximal Policy Gradient with Dual Network Architecture (PPO-DNA)

## Overview

PPO-DNA is a more sample efficient variant of PPO, based on using separate optimizers and hyperparameters for the actor (policy) and critic (value) networks.

Original paper: 

* [DNA: Proximal Policy Optimization with a Dual Network Architecture](https://arxiv.org/abs/2206.10027)

## Implemented Variants


| Variants Implemented      | Description |
| ----------- | ----------- |
| :material-github: [`ppo_dna_atari_envpool.py`](https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/ppo_dna_atari_envpool.py), :material-file-document: [docs](/rl-algorithms/ppo_dna/#ppo_dna_atari_envpoolpy) | Uses the blazing fast Envpool Atari vectorized environment. |

Below are our single-file implementations of PPO-DNA:

## `ppo_dna_atari_envpool.py`

The [ppo_dna_atari_envpool.py](https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/ppo_dna_atari_envpool.py) has the following features:

* Uses the blazing fast [Envpool](https://github.com/sail-sg/envpool) vectorized environment.
* For Atari games. It uses convolutional layers and common atari-based pre-processing techniques.
* Works with the Atari's pixel `Box` observation space of shape `(210, 160, 3)`
* Works with the `Discrete` action space

???+ warning

    Note that `ppo_dna_atari_envpool.py` does not work in Windows :fontawesome-brands-windows: and MacOs :fontawesome-brands-apple:. See envpool's built wheels here: [https://pypi.org/project/envpool/#files](https://pypi.org/project/envpool/#files)


### Usage

```bash
poetry install -E envpool
python cleanrl/ppo_dna_atari_envpool.py --help
python cleanrl/ppo_dna_atari_envpool.py --env-id Breakout-v5
```

### Explanation of the logged metrics

See [related docs](/rl-algorithms/ppo/#explanation-of-the-logged-metrics) for `ppo.py`.

### Implementation details

[ppo_dna_atari_envpool.py](https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/ppo_dna_atari_envpool.py) uses a customized `RecordEpisodeStatistics` to work with envpool but has the same other implementation details as `ppo_atari.py` (see [related docs](/rl-algorithms/ppo/#implementation-details_1)).

### Experiment results

Below are the average episodic returns for `ppo_dna_atari_envpool.py` compared to `ppo_atari_envpool.py`.


| Environment      | `ppo_dna_atari_envpool.py` | `ppo_atari_envpool.py` |
| ----------- | ----------- | ----------- | 
| BattleZone-v5 (40M steps) | 94800 ± 18300 | 28800 ± 6800 |
| BeamRider-v5 (10M steps) | 5470 ± 850 | 1990 ± 560 |
| Breakout-v5 (10M steps) | 321 ± 63 | 352 ± 52 |
| DoubleDunk-v5 (40M steps) | -4.9 ± 0.3 | -2.0 ± 0.8 |
| NameThisGame-v5 (40M steps) | 8500 ± 2600 | 4400 ± 1200 |
| Phoenix-v5 (45M steps) | 184000 ± 58000 | 10200 ± 2700 |
| Pong-v5 (3M steps) | 19.5 ± 1.1 | 16.6 ± 2.3 |
| Qbert-v5 (45M steps) | 12600 ± 4600 | 10800 ± 3300 |
| Tennis-v5 (10M steps) | 13.0 ± 2.3 | -12.4 ± 2.9 |

Learning curves:

<div class="grid-container">
<img src="../ppo_dna/BattleZone-v5-steps.png">
<img src="../ppo_dna/BattleZone-v5-time.png">
<img src="../ppo_dna/BeamRider-v5-steps.png">
<img src="../ppo_dna/BeamRider-v5-time.png">
<img src="../ppo_dna/Breakout-v5-steps.png">
<img src="../ppo_dna/Breakout-v5-time.png">
<img src="../ppo_dna/DoubleDunk-v5-steps.png">
<img src="../ppo_dna/DoubleDunk-v5-time.png">
<img src="../ppo_dna/NameThisGame-v5-steps.png">
<img src="../ppo_dna/NameThisGame-v5-time.png">
<img src="../ppo_dna/Phoenix-v5-steps.png">
<img src="../ppo_dna/Phoenix-v5-time.png">
<img src="../ppo_dna/Pong-v5-steps.png">
<img src="../ppo_dna/Pong-v5-time.png">
<img src="../ppo_dna/Qbert-v5-steps.png">
<img src="../ppo_dna/Qbert-v5-time.png">
<img src="../ppo_dna/Tennis-v5-steps.png">
<img src="../ppo_dna/Tennis-v5-time.png">
</div>


Tracked experiments:

<iframe src="https://wandb.ai/jseppanen/cleanrl/reports/PPO-DNA-vs-PPO-on-Atari-Envpool--VmlldzoyMzM5Mjcw" style="width:100%; height:500px" title="PPO-DNA vs PPO on Atari Envpool"></iframe>
