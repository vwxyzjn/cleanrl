# Phasic Policy Gradient (PPO)

## Overview

PPG is a DRL algorithm that separates policy and value function training by introducing an auxiliary phase. The training proceeds by running PPO during the policy phase, saving all the experience in a replay buffer. Then the replay buffer is used to train the value function. This makes the algorithm considerably slower than PPO, but improves sample efficiency on Procgen benchmark.

Original paper: 

* [Phasic Policy Gradient](https://arxiv.org/abs/2009.04416)

Reference resources:

* [Code for the paper "Phasic Policy Gradient"](https://github.com/openai/phasic-policy-gradient) - by original authors from OpenAI

The original code has multiple code level details that are not mentioned in the paper. We found these changes to be important for reproducing the results claimed by the paper.

## Implementation 

Below is the single-file implementations of PPG:

## `ppg_procgen.py`

`ppg_procgen.py` works with the Procgen benchmark, which uses 64x64 RGB image observations, and discrete actions

### Usage

```bash
poetry install
python cleanrl/ppo_procgen.py --help
python cleanrl/ppo_procgen.py --env-id "bigfish"
```

### Implementation details

`ppg_procgen.py` includes the <TODO> level implementation details that are different from PPO:

1. Full rollout sampling during auxiliary phase - (:material-github: [phasic_policy_gradient/ppg.py#L173](https://github.com/openai/phasic-policy-gradient/blob/c789b00be58aa704f7223b6fc8cd28a5aaa2e101/phasic_policy_gradient/ppg.py#L173)) - Instead of randomly sampling observations over the entire auxiliary buffer, PPG samples full rullouts from the buffer (Sets of 256 steps). This full rollout sampling is only done during the auxiliary phase. Note that the rollouts will still be at random starting points because PPO truncates the rollouts per env. This change gives a decent performance boost.

1. Batch level advantage normalization - PPG normalizes the full batch of advantage values before PPO updates instead of advantage normalization on each minibatch. (:material-github: [phasic_policy_gradient/ppo.py#L70](https://github.com/openai/phasic-policy-gradient/blob/c789b00be58aa704f7223b6fc8cd28a5aaa2e101/phasic_policy_gradient/ppo.py#L70))

1. Normalized network initialization - (:material-github: [phasic_policy_gradient/impala_cnn.py#L64](https://github.com/openai/phasic-policy-gradient/blob/c789b00be58aa704f7223b6fc8cd28a5aaa2e101/phasic_policy_gradient/impala_cnn.py#L64)) - PPG uses normalized initialization for all layers, with different scales. 
    * Original PPO used orthogonal initialization of only the Policy head and Value heads with scale of 0.01 and 1. respectively.
    * For PPG
        * All weights are initialized with the default torch initialization (Kaiming Uniform)
        * Each layer’s weights are divided by the L2 norm of the weights along the (which axis?), and multiplied by a scale factor.
        * Scale factors for different layers
            * Value head, Policy head, Auxiliary value head - 0.1
            * Fully connected layer after last conv later - 1.4
            * Convolutional layers - Approximately 0.638
1. The Adam Optimizer's Epsilon Parameter -(:material-github: [phasic_policy_gradient/ppg.py#L239](https://github.com/openai/phasic-policy-gradient/blob/c789b00be58aa704f7223b6fc8cd28a5aaa2e101/phasic_policy_gradient/ppg.py#L239)) - Set to torch default of 1e-8 instead of 1e-5 which is used in PPO.


### Experiment results

Below are the average episodic returns for `ppo_procgen.py`, and comparision with `ppo_procgen.py` on 25M timesteps.

| Environment         | `ppg_procgen.py`    | `ppo_procgen.py` |
| -----------         | -----------         | -----------      |
| Bigfish (easy)      | 27.670 ± 9.523      | 21.605 ± 7.996   |
| Starpilot (easy)    |  39.086 ± 11.042    |  34.025 ± 12.535 |

Learning curves:

<div class="grid-container">

<img src="ppg/bigfish-easy-ppg-ppo.svg">

<img src="ppg/starpilot-easy-ppg-ppo.svg">

</div>

Tracked experiments and game play videos:

To be added

### Extra notes

- All the default hyperparameters from the original PPG implementation are used. Except setting 64 for the number of environments.
- The original PPG paper does not report results on easy environments, hence more hyperparameter tuning can give better results.
- Skipping every alternate auxiliary phase gives similar performance on easy environments while saving compute.
- Normalized network initialization scheme seems to matter a lot, but using layernorm with orthogonal initialization also works.
- Using mixed precision for auxiliary phase also works well to save compute, but using on policy phase makes training unstable.


### Differences from the original PPG code

- The original PPG code supports LSTM whereas the CleanRL code does not.
- The original PPG code uses separate optimizers for policy and auxiliary phase, but we do not implement this as we found it to not make too much difference.