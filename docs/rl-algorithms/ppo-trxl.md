# Tranformer-XL (PPO-TrXL)

## Overview

Real-world tasks may expose imperfect information (e.g. partial observability). Such tasks require an agent to leverage memory capabilities. One way to do this is to use recurrent neural networks (e.g. LSTM) as seen in :material-github: [`ppo_atari_lstm.py`](https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/ppo_atari_lstm.py), :material-file-document: [docs](/rl-algorithms/ppo/#ppo_atari_lstmpy). Here, Transformer-XL is used as episodic memory in Proximal Policy Optimization (PPO).

Original Paper and Implementation

* :material-file-document: [Memory Gym: Towards Endless Tasks to Benchmark Memory Capabilities of Agents](https://arxiv.org/abs/2309.17207)
* :material-github: [neroRL](https://github.com/MarcoMeter/neroRL)
* :material-github:  [Episodic Transformer Memory PPO](https://github.com/MarcoMeter/episodic-transformer-memory-ppo)
* :material-github: [Endless Memory Gym](https://github.com/MarcoMeter/endless-memory-gym)
* :material-play-circle: [Interactive Visualizations of Trained Agents](https://marcometer.github.io/)

Related Publications and Repositories

* :material-file-document: [Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context](https://arxiv.org/abs/1901.02860)
* :material-file-document: [Stabilizing Transformers for Reinforcement Learning](https://arxiv.org/abs/1910.06764)
* :material-file-document: [Towards mental time travel: a hierarchical memory for reinforcement learning agents](https://arxiv.org/abs/2105.14039)
* :material-file-document: [Grounded Language Learning Fast and Slow](https://arxiv.org/abs/2009.01719)
* :material-github: [transformerXL_PPO_JAX](https://github.com/Reytuag/transformerXL_PPO_JAX)

## Implemented Variants


| Variants Implemented      | Description |
| ----------- | ----------- |
| :material-github: [`ppo_trxl.py`](https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/ppo_trxl/ppo_trxl.py), :material-file-document: [docs](/rl-algorithms/ppo-trxl#ppo_trxlpy) | For training on tasks like `Endless-MortarMayhem-v0`. |

Below is our single-file implementation of PPO-TrXL:

## `ppo_trxl.py`

`ppo_trxl.py` has the following features:

* Works with Memory Gym's environments (84x84 RGB image observation).
* Works with Minigrid Memory (84x84 RGB image observation).
* Works also with environments exposing only game state vector observations (e.g. Proof of Memory Environment).
* Works with just single or multi-discrete action spaces.

### Usage

As the recommended way, the requirements default to PyTorch's CUDA packages.

=== "poetry"

    ```bash
    cd cleanrl/ppo_trxl
    poetry install
    poetry run python ppo_trxl.py --help
    poetry run python ppo_trxl.py --env-id Endless-MortarMayhem-v0
    ```
  
=== "pip"

    ```bash
    pip install -r requirements/requirements-memory_gym.txt
    python cleanrl/ppo_trxl/ppo_trxl.py --help
    python cleanrl/ppo_trxl/ppo_trxl.py --env-id Endless-MortarMayhem-v0
    ```

### Explanation of the logged metrics

* `episode/r_mean`: mean of the episodic return of the game
* `episode/l_mean`: mean of the episode length of the game in steps
* `episode/t_mean`: mean of the episode duration of the game in seconds
* `episode/advantage_mean`: mean of all computed advantages
* `episode/value_mean`: mean of all approximated values
* `charts/SPS`: number of steps per second
* `charts/learning_rate`: the current learning rate
* `charts/entropy_coefficient`: the current entropy coefficient
* `losses/value_loss`: the mean value loss across all data points
* `losses/policy_loss`: the mean policy loss across all data points
* `losses/entropy`: the mean entropy value across all data points
* `losses/reconstruction_loss`: the mean observation reconstruction loss value across all data points
* `losses/loss`: the mean of all summed losses across all data points
* `losses/old_approx_kl`: the approximate Kullback–Leibler divergence, measured by `(-logratio).mean()`, which corresponds to the k1 estimator in John Schulman’s blog post on [approximating KL](http://joschu.net/blog/kl-approx.html)
* `losses/approx_kl`: better alternative to `olad_approx_kl` measured by `(logratio.exp() - 1) - logratio`, which corresponds to the k3 estimator in [approximating KL](http://joschu.net/blog/kl-approx.html)
* `losses/clipfrac`: the fraction of the training data that triggered the clipped objective
* `losses/explained_variance`: the explained variance for the value function

### Implementation details

Most details are derived from [`ppo.py`](/rl-algorithms/ppo#ppopy). These are additional or differing details:

1. The policy and value function share parameters.
2. Multi-head attention is implemented so that all heads share parameters.
3. Absolute positional encoding is used as default. Learned positional encodings are supported.
4. Previously computed hidden states of the TrXL layers are cached and re-used for up to `trxl_memory_length`. Only 1 hidden state is computed anew.
5. TrXL layers adhere to pre-layer normalization.
6. Support for multi-discrete action spaces.
7. Support for an auxiliary observation reconstruction loss, which reconstructs TrXL's output to the fed visual observation.
8. The learning rate and the entropy bonus coefficient linearly decay until reaching a lower threshold.

### Experiment results

Note: When training on potentially endless episodes, the cached hidden states demand a large GPU memory. To reproduce the following experiments a minimum of 40GB is required. One workaround is to cache the hidden states in the buffer with lower precision as bfloat16. This is under examination for future updates.

|                              | PPO-TrXL    |
|:-----------------------------|:------------|
| MortarMayhem-Grid-v0         | 0.99 ± 0.00 |
| MortarMayhem-v0              | 0.99 ± 0.00 |
| Endless-MortarMayhem-v0      | 1.50 ± 0.02 |
| MysteryPath-Grid-v0          | 0.97 ± 0.01 |
| MysteryPath-v0               | 1.67 ± 0.02 |
| Endless-MysteryPath-v0       | 1.84 ± 0.06 |
| SearingSpotlights-v0         | 1.11 ± 0.08 |
| Endless-SearingSpotlights-v0 | 1.60 ± 0.03 |

Learning curves:


<img src="../ppo-trxl/compare.png">


Tracked experiments:

<iframe src="https://api.wandb.ai/links/m-pleines/wo9m43hv" style="width:100%; height:500px" title="CleanRL-s-PPO-TrXL"></iframe>


### Hyperparameters

Memory Gym Environments

Please refer to the defaults in [`ppo_trxl.py`](https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/ppo_trxl/ppo_trxl.py) and the single modifications as found in [`benchmark/ppo_trxl.sh`](https://github.com/vwxyzjn/cleanrl/blob/master/benchmark/ppo_trxl.sh)

ProofofMemory-v0
```bash
poetry run python ppo_trxl.py \
  --env_id ProofofMemory-v0 \
  --total_timesteps 25000 \
  --num_envs 16 \
  --num_steps 128 \
  --num_minibatches 8 \
  --update_epochs 4 \
  --trxl_num_layers 4 \
  --trxl_num_heads 1 \
  --trxl_dim 64 \
  --trxl_memory_length 16 \
  --trxl_positional_encoding none \
  --vf_coef 0.1 \
  --max_grad_norm 0.5 \
  --init_lr 3.0e-4 \
  --init_ent_coef 0.001 \
  --clip_coef 0.2
```

MiniGrid-MemoryS9-v0
```bash
poetry run python ppo_trxl.py \
  --env_id MiniGrid-MemoryS9-v0 \
  --total_timesteps 2048000 \
  --num_envs 16 \
  --num_steps 256 \
  --trxl_num_layers 2 \
  --trxl_num_heads 4 \
  --trxl_dim 256 \
  --trxl_memory_length 64 \
  --max_grad_norm 0.25 \
  --anneal_steps 4096000
  --clip_coef 0.2
```

### Enjoy pre-trained models

Use [`cleanrl/ppo_trxl/enjoy.py`](https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/ppo_trxl/enjoy.py) to watch pre-trained agents.
You can retrieve pre-trained models from [huggingface](https://huggingface.co/LilHairdy/cleanrl_memory_gym).
Note that Memory Gym environments are usually rendered using the `debug_rgb_array` render mode, which shows ground truth information about the current task that the agent cannot observe.


Run models from the hub:
```bash
python cleanrl/ppo_trxl/enjoy.py --hub --name Endless-MortarMayhem-v0_12.nn
python cleanrl/ppo_trxl/enjoy.py --hub --name Endless-MysterPath-v0_11.nn
python cleanrl/ppo_trxl/enjoy.py --hub --name Endless-SearingSpotlights-v0_30.nn
python cleanrl/ppo_trxl/enjoy.py --hub --name MiniGrid-MemoryS9-v0_10.nn
python cleanrl/ppo_trxl/enjoy.py --hub --name ProofofMemory-v0_1.nn
```


Run local models (or download them from the hub manually):
```bash
python cleanrl/ppo_trxl/enjoy.py --name Your.cleanrl_model
```