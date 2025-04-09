# Maximum A Posteriori Optimization (MPO)


## Overview

MPO is an offline actor-critic algorithm best suited for robotics continuous control problems. It learns a critic (the better the method to learn this critic, the better the results), then it uses this critic to compute a non-parametric locally improved policy, and finally just regresses the parametric policy with the non-parametric improved one in a supervised learning fashion. It uses several trust-region constraints, like TRPO and PPO, in order to not rush into some local improvements that may only be local minima.


Original paper: 

* [Maximum a Posteriori Policy Optimisation](https://arxiv.org/abs/1806.06920)
* [Relative Entropy Regularized Policy Iteration](https://arxiv.org/abs/1812.02256.pdf)
* [A Distributional View on Multi-Objective Policy Optimization](https://arxiv.org/abs/2005.07513.pdf)
* [Revisiting Gaussian mixture critics in off-policy reinforcement learning: a sample-based approach](https://arxiv.org/abs/2204.10256)

Reference resources:

* [Acme: A Research Framework for Distributed Reinforcement Learning](https://arxiv.org/abs/2006.00979)
* [Learning Agile Soccer Skills for a Bipedal Robot with Deep Reinforcement Learning](https://arxiv.org/abs/2304.13653.pdf)

We followed the reference implementation in jax of deepmind's acme library [official mpo jax implementation](https://github.com/deepmind/acme/tree/master/acme/agents/jax/mpo) and applied the hyperparameters used in the benchmarks of the [acme introduction paper](https://arxiv.org/abs/2006.00979), that can be found on the [acme library](https://github.com/deepmind/acme/blob/master/examples/baselines/rl_continuous/run_mpo.py).

## Implemented Variants


| Variants Implemented      | Description |
| ----------- | ----------- |
| :material-github: [`mpo_tdn_continuous_action.py`](https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/mpo_tdn_continuous_action.py), :material-file-document: [docs](/rl-algorithms/mpo/#mpo_tdn_continuous_actionpy) | For continuous action space. Also implemented Mujoco-specific code-level optimizations. Uses a TD(n) critic target. | :material-github: [`dmpo_continuous_action.py`](https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/mpo_tdn_continuous_action.py), :material-file-document: [docs](/rl-algorithms/mpo/#mpo_tdn_continuous_actionpy) | For continuous action space. Also implemented Mujoco-specific code-level optimizations. Uses a TD(n) *distributional* critic target. |


Below are our single-file implementations of PPO:

## `mpo_tdn_continuous_action.py`

The [ppo.py](https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/mpo_tdn_continuous_action.py) has the following features:

* For continuous action space
* Works with the `Box` observation space of low-level features
* Works with the `Box` (continuous) action space

### Usage

=== "poetry"

    ```bash
    poetry install
    poetry run python cleanrl/mpo_tdn_continuous_action.py --help
    poetry install -E mujoco_py # only works in Linux
    poetry run python cleanrl/mpo_tdn_continuous_action.py --env-id Hopper-v2
    poetry install -E mujoco
    poetry run python cleanrl/mpo_tdn_continuous_action.py --env-id Hopper-v4
    ```

=== "pip"

    ```bash
    python cleanrl/mpo_tdn_continuous_action.py --help
    pip install -r requirements/requirements-mujoco_py.txt # only works in Linux, you have to pick either `mujoco` or `mujoco_py`
    python cleanrl/mpo_tdn_continuous_action.py --env-id Hopper-v2
    pip install -r requirements/requirements-mujoco.txt
    python cleanrl/mpo_tdn_continuous_action.py --env-id Hopper-v4
    ```

### Explanation of the logged metrics

Running `python cleanrl/ppo.py` will automatically record various metrics such as actor or value losses in Tensorboard. Below is the documentation for these metrics:

* `charts/episodic_return`: episodic return of the game
* `charts/episodic_length`: episodic length of the game
* `charts/episodic_return`: episodic return of the game during evaluation
* `charts/episodic_length`: episodic length of the game during evaluation
* `charts/SPS`: number of steps per second
* `losses/qf_loss`: the mean value loss across all data points
* `losses/policy_loss`: the mean policy loss across all data points
* `losses/dua_loss`: the mean dual loss across all data points
* `losses/qf_values`: the mean qvalue value across all data points
* `losses/log_eta`: the non-parametric KL constraint temperature value
* `losses/log_penalty_temperature`: the action limit constraint temperature value
* `losses/mean_log_alpha_mean`: the mean parametric KL constraint over the mean temperature value across all parametric KL constraint over the mean temperatures
* `losses/mean_log_alpha_min`: the min parametric KL constraint over the mean temperature value across all parametric KL constraint over the mean temperatures
* `losses/mean_log_alpha_std`: the mean parametric KL constraint over the std temperature value across all parametric KL constraint over the std temperatures


### Implementation details

1. We decouple every loss of the policy regression step between mean and std: we first compute all the losses using the mean of the online network and the std of the target network and then compute everything again using the std of the online network and the mean of the target network.
1. We use per dimension KL constraint of the policy regression: "Otherwise the overall KL is constrained, which allows some dimensions to change more at the expense of others staying put."
1. We use an additional out-of-bound quadratic action penalization term when calculating the locally improved non-parametric policy.
1. Actions are clipped to bounds before entering the q-network.
1. LayerNorm followed by tanh is applied after the first layer of both q-network and actor-network.
1. TD(n) is used to get a better critic target. For that, we use a second small buffer that stores the n-last transitions in order for each transition to compute an n-step-discounted reward, which is more precise than the 1-step used in other algorithms. At the end of an episode, we add all the remaining transitions from the mini buffer to the replay buffer, cautiously calculating the discounted reward, as this is a special case and the general case doesn't apply here. For these transitions, we have to store as well the discount factor of the bootstrapped prediction of the q-network, in case of episode ending because of time-limit (episode truncation).
1. Replay buffer handles episode truncation, meaning that in case of the episode ending because of time-limit, we still use the predicted q-value in the target calculation (for normal episode termination this is eliminated).
1. Dual parameters are clipped after each sgd step.

### Experiment results
TBD.
