# Phasic Policy Gradient (PPG)

## Overview

PPG is a DRL algorithm that separates policy and value function training by introducing an auxiliary phase. The training proceeds by running PPO during the policy phase, saving all the experience in a replay buffer. Then the replay buffer is used to train the value function. This makes the algorithm considerably slower than PPO, but improves sample efficiency on Procgen benchmark.

Original paper: 

* [Phasic Policy Gradient](https://arxiv.org/abs/2009.04416)

Reference resources:

* [Code for the paper "Phasic Policy Gradient"](https://github.com/openai/phasic-policy-gradient) - by original authors from OpenAI

The original code has multiple code level details that are not mentioned in the paper. We found these changes to be important for reproducing the results claimed by the paper.

## Implemented Variants


| Variants Implemented      | Description |
| ----------- | ----------- |
| :material-github: [`ppg_procgen.py`](https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/ppg_procgen.py), :material-file-document: [docs](/rl-algorithms/ppg/#ppg_procgenpy) | For classic control tasks like `CartPole-v1`. |

Below are our single-file implementations of PPG:

## `ppg_procgen.py`

`ppg_procgen.py` works with the Procgen benchmark, which uses 64x64 RGB image observations, and discrete actions

### Usage

=== "poetry"

    ```bash
    poetry install -E procgen
    poetry run python cleanrl/ppg_procgen.py --help
    poetry run python cleanrl/ppg_procgen.py --env-id starpilot
    ```

=== "pip"

    ```bash
    pip install -r requirements/requirements-procgen.txt
    python cleanrl/ppg_procgen.py --help
    python cleanrl/ppg_procgen.py --env-id starpilot
    ```

### Explanation of the logged metrics

Running `python cleanrl/ppg_procgen.py` will automatically record various metrics such as actor or value losses in Tensorboard. Below is the documentation for these metrics:

Same as PPO:

* `charts/episodic_return`: episodic return of the game
* `charts/episodic_length`: episodic length of the game
* `charts/SPS`: number of steps per second (this is initially high but drops off after the auxiliary phase)
* `charts/learning_rate`: the current learning rate (annealing is not done by default)
* `losses/value_loss`: the mean value loss across all data points
* `losses/policy_loss`: the mean policy loss across all data points
* `losses/entropy`: the mean entropy value across all data points
* `losses/old_approx_kl`: the approximate Kullback–Leibler divergence, measured by `(-logratio).mean()`, which corresponds to the k1 estimator in John Schulman’s blog post on [approximating KL](http://joschu.net/blog/kl-approx.html)
* `losses/approx_kl`: better alternative to `olad_approx_kl` measured by `(logratio.exp() - 1) - logratio`, which corresponds to the k3 estimator in [approximating KL](http://joschu.net/blog/kl-approx.html)
* `losses/clipfrac`: the fraction of the training data that triggered the clipped objective
* `losses/explained_variance`: the explained variance for the value function

PPG specific:

* `losses/aux/kl_loss`: the mean value of the KL divergence when distilling the latest policy during the auxiliary phase.
* `losses/aux/aux_value_loss`: the mean value loss on the auxiliary value head
* `losses/aux/real_value_loss`: the mean value loss on the detached value head used to calculate the GAE returns during policy phase

### Implementation details

`ppg_procgen.py` includes the <TODO> level implementation details that are different from PPO:

1. Full rollout sampling during auxiliary phase - (:material-github: [phasic_policy_gradient/ppg.py#L173](https://github.com/openai/phasic-policy-gradient/blob/c789b00be58aa704f7223b6fc8cd28a5aaa2e101/phasic_policy_gradient/ppg.py#L173)) - Instead of randomly sampling observations over the entire auxiliary buffer, PPG samples full rullouts from the buffer (Sets of 256 steps). This full rollout sampling is only done during the auxiliary phase. Note that the rollouts will still be at random starting points because PPO truncates the rollouts per env. This change gives a decent performance boost.

1. Batch level advantage normalization - PPG normalizes the full batch of advantage values before PPO updates instead of advantage normalization on each minibatch. (:material-github: [phasic_policy_gradient/ppo.py#L70](https://github.com/openai/phasic-policy-gradient/blob/c789b00be58aa704f7223b6fc8cd28a5aaa2e101/phasic_policy_gradient/ppo.py#L70))

1. Normalized network initialization - (:material-github: [phasic_policy_gradient/impala_cnn.py#L64](https://github.com/openai/phasic-policy-gradient/blob/c789b00be58aa704f7223b6fc8cd28a5aaa2e101/phasic_policy_gradient/impala_cnn.py#L64)) - PPG uses normalized initialization for all layers, with different scales. 
    * Original PPO used orthogonal initialization of only the Policy head and Value heads with scale of 0.01 and 1. respectively.
    * For PPG
        * All weights are initialized with the default torch initialization (Kaiming Uniform)
        * Each layer’s weights are divided by the L2 norm of the weights such that the weights of `input_channels` axis are individually normalized (axis 1 for linear layers and 1,2,3 for convolutional layers). Then the weights are multiplied by a scale factor.
        * Scale factors for different layers
            * Value head, Policy head, Auxiliary value head - 0.1
            * Fully connected layer after last conv later - 1.4
            * Convolutional layers - Approximately 0.638
1. The Adam Optimizer's Epsilon Parameter -(:material-github: [phasic_policy_gradient/ppg.py#L239](https://github.com/openai/phasic-policy-gradient/blob/c789b00be58aa704f7223b6fc8cd28a5aaa2e101/phasic_policy_gradient/ppg.py#L239)) - Set to torch default of 1e-8 instead of 1e-5 which is used in PPO.
1. Use the same `gamma` parameter in the `NormalizeReward` wrapper. Note that the original implementation from [openai/train-procgen](https://github.com/openai/train-procgen) uses the default `gamma=0.99` in [the `VecNormalize` wrapper](https://github.com/openai/train-procgen/blob/1a2ae2194a61f76a733a39339530401c024c3ad8/train_procgen/train.py#L43) but `gamma=0.999` as PPO's parameter. The mismatch between the `gamma`s is technically incorrect. See [#209](https://github.com/vwxyzjn/cleanrl/pull/209)

Here are some additional notes:

- All the default hyperparameters from the original PPG implementation are used. Except setting 64 for the number of environments.
- The original PPG paper does not report results on easy environments, hence more hyperparameter tuning can give better results.
- Skipping every alternate auxiliary phase gives similar performance on easy environments while saving compute.
- Normalized network initialization scheme seems to matter a lot, but using layernorm with orthogonal initialization also works.
- Using mixed precision for auxiliary phase also works well to save compute, but using on policy phase makes training unstable.


Also, `ppg_procgen.py` differs from the original  `openai/phasic-policy-gradient` implementation in the following ways.

- The original PPG code supports LSTM whereas the CleanRL code does not.
- The original PPG code uses separate optimizers for policy and auxiliary phase, but we do not implement this as we found it to not make too much difference.
- The original PPG code utilizes multiple GPUs but our implementation does not


### Experiment results

To run benchmark experiments, see :material-github: [benchmark/ppg.sh](https://github.com/vwxyzjn/cleanrl/blob/master/benchmark/ppg.sh). Specifically, execute the following command:

<script src="https://emgithub.com/embed.js?target=https%3A%2F%2Fgithub.com%2Fvwxyzjn%2Fcleanrl%2Fblob%2Fmaster%2Fbenchmark%2Fppg.sh%23L3-L8&style=github&showBorder=on&showLineNumbers=on&showFileMeta=on&showCopy=on"></script>


Below are the average episodic returns for `ppg_procgen.py`, and comparison with `ppg_procgen.py` on 25M timesteps.

| Environment      | `ppg_procgen.py` | `ppo_procgen.py` | `openai/phasic-policy-gradient` (easy) |
|------------------|------------------|------------------|----------------------------------------|
| Starpilot (easy) | 34.82 ± 13.77    | 32.47 ± 11.21    | 42.01 ± 9.59                           |
| Bossfight (easy) | 10.78 ± 1.90     | 9.63 ± 2.35      | 10.71 ± 2.05                           |
| Bigfish (easy)   | 24.23 ± 10.73    | 16.80 ± 9.49     | 15.94 ± 10.80                          |


???+ warning

    Note that we have run the procgen experiments using the `easy` distribution for reducing the computational cost. However, the original paper's results were condcuted with the `hard` distribution mode. For convenience, in the learning curves below, we compared the performance of the original code base (`openai/phasic-policy-gradient` the purple curve) in the `easy` distribution. 

Learning curves:

<div class="grid-container">
<img src="../ppg/StarPilot.png">
<img src="../ppg/comparison/StarPilot.png">

<img src="../ppg/BossFight.png">
<img src="../ppg/comparison/BossFight.png">

<img src="../ppg/BigFish.png">
<img src="../ppg/comparison/BigFish.png">
</div>


???+ info

    Also note that our `ppo_procgen.py` which closely matches implementation details of `openai/baselines`' PPO which might not be the same as `openai/phasic-policy-gradient`'s PPO. We take the reported results from (Cobbe et al., 2020)[^1] and (Cobbe et al., 2021)[^2] and compared them in a [google sheet](https://docs.google.com/spreadsheets/d/1ZC_D2WPL6-PzhecM4ZFQWQ6nY6dkXeQDOIgRHVp1BNU/edit?usp=sharing) (screenshot shown below). As shown, the performance seems to diverge a bit. We also note that (Cobbe et al., 2020)[^1] used [`procgen==0.9.2`](https://github.com/openai/train-procgen/blob/1a2ae2194a61f76a733a39339530401c024c3ad8/environment.yml#L10) and (Cobbe et al., 2021)[^2] used [`procgen==0.10.4`](https://github.com/openai/phasic-policy-gradient/blob/7295473f0185c82f9eb9c1e17a373135edd8aacc/environment.yml#L10), which also could cause performance difference. It is for this reason, we ran our own `openai/phasic-policy-gradient` experiments on the `easy` distribution for comparison, but this does mean it's challenging to compare our results against those in the original PPG paper (Cobbe et al., 2021)[^2].

    ![PPG's PPO compared to openai/baselines' PPO](../ppg/ppg-ppo.png)

Tracked experiments and game play videos:


<iframe src="https://wandb.ai/openrlbenchmark/openrlbenchmark/reports/Procgen-CleanRL-s-PPG--VmlldzoyMDc1MDMz" style="width:100%; height:500px" title="Procgen-CleanRL-s-PPG"></iframe>

<iframe src="https://wandb.ai/openrlbenchmark/openrlbenchmark/reports/Procgen-CleanRL-s-PPG-vs-PPO-vs-openai-phasic-policy-gradient--VmlldzoyMDc1MDc3" style="width:100%; height:500px" title="Procgen-CleanRL-s-PPG-PPO-openai-phasic-policy-gradient"></iframe>


[^1]: Cobbe, K., Hesse, C., Hilton, J., & Schulman, J. (2020, November). Leveraging procedural generation to benchmark reinforcement learning. In International conference on machine learning (pp. 2048-2056). PMLR.
[^2]: Cobbe, K. W., Hilton, J., Klimov, O., & Schulman, J. (2021, July). Phasic policy gradient. In International Conference on Machine Learning (pp. 2020-2027). PMLR.

