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

Note that the original DNA implementation uses the `StickyAction` environment pre-processing wrapper (see (Machado et al., 2018)[^1]), but we did not implement it in [ppo_dna_atari_envpool.py](https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/ppo_dna_atari_envpool.py) because envpool for now does not support `StickyAction`.


### Experiment results

Below are the average episodic returns for `ppo_dna_atari_envpool.py` compared to `ppo_atari_envpool.py`.


| Environment      | `ppo_dna_atari_envpool.py` | `ppo_atari_envpool.py` |
| ----------- | ----------- | ----------- | 
| BattleZone-v5 (40M steps) | 74000 ± 15300 | 28700 ± 6300 |
| BeamRider-v5 (10M steps) | 5200 ± 900 | 1900 ± 530 |
| Breakout-v5 (10M steps) | 319 ± 63 | 349 ± 42 |
| DoubleDunk-v5 (40M steps) | -4.1 ± 1.0 | -2.0 ± 0.8 |
| NameThisGame-v5 (40M steps) | 19100 ± 2300 | 4400 ± 1200 |
| Phoenix-v5 (45M steps) | 186000 ± 67000 | 9900 ± 2700 |
| Pong-v5 (3M steps) | 19.5 ± 1.0 | 16.6 ± 2.4 |
| Qbert-v5 (45M steps) | 12800 ± 4200 | 11400 ± 3600 |
| Tennis-v5 (10M steps) | 19.6 ± 0.0 | -12.4 ± 2.9 |

Learning curves:

<div class="grid-container">
<img src="../ppo_dna/BattleZone-v5-50m-steps.png">
<img src="../ppo_dna/BattleZone-v5-50m-time.png">
<img src="../ppo_dna/BeamRider-v5-10m-steps.png">
<img src="../ppo_dna/BeamRider-v5-10m-time.png">
<img src="../ppo_dna/Breakout-v5-10m-steps.png">
<img src="../ppo_dna/Breakout-v5-10m-time.png">
<img src="../ppo_dna/DoubleDunk-v5-50m-steps.png">
<img src="../ppo_dna/DoubleDunk-v5-50m-time.png">
<img src="../ppo_dna/NameThisGame-v5-50m-steps.png">
<img src="../ppo_dna/NameThisGame-v5-50m-time.png">
<img src="../ppo_dna/Phoenix-v5-50m-steps.png">
<img src="../ppo_dna/Phoenix-v5-50m-time.png">
<img src="../ppo_dna/Pong-v5-3m-steps.png">
<img src="../ppo_dna/Pong-v5-3m-time.png">
<img src="../ppo_dna/Qbert-v5-50m-steps.png">
<img src="../ppo_dna/Qbert-v5-50m-time.png">
<img src="../ppo_dna/Tennis-v5-10m-steps.png">
<img src="../ppo_dna/Tennis-v5-10m-time.png">
</div>


Tracked experiments:

<iframe src="https://wandb.ai/jseppanen/cleanrl/reports/PPO-DNA-vs-PPO-on-Atari-Envpool--VmlldzoyMzM5Mjcw" style="width:100%; height:500px" title="PPO-DNA vs PPO on Atari Envpool"></iframe>




[^1]: Machado, Marlos C., Marc G. Bellemare, Erik Talvitie, Joel Veness, Matthew Hausknecht, and Michael Bowling. "Revisiting the arcade learning environment: Evaluation protocols and open problems for general agents." Journal of Artificial Intelligence Research 61 (2018): 523-562.