# ES-C51: Expected Sarsa Based C51 Distributional Reinforcement Learning

This directory contains implementations of ES-C51 (Expected Sarsa based C51) and modified C51 algorithms with softmax action selection, as described in our paper submitted to Neurocomputing.

## Paper Reference

**Title**: ES-C51: Expected Sarsa Based C51 Distributional Reinforcement Learning Algorithm  
**Authors**: Rijul Tandon, Peter Vamplew, Cameron Foale  
**Submitted to**: Neurocomputing  
**ArXiv Link**: [To be added upon publication]

## Overview

This work presents ES-C51, a modification of the standard C51 distributional reinforcement learning algorithm that replaces the greedy Q-learning update with an Expected Sarsa update. The key innovation is using a softmax-weighted expectation over all possible next actions rather than relying solely on the greedy action selection.

## Key Contributions

1. **Algorithmic Innovation**: Introduction of ES-C51 that uses Expected Sarsa updates in distributional RL
2. **Fair Comparison**: Modified standard C51 to use softmax exploration (QL-C51) for fair comparison
3. **Comprehensive Evaluation**: Tested on Gym classic control and Atari-10 environments
4. **Performance Analysis**: Demonstrated superior performance of ES-C51 over QL-C51 across multiple environments

## Files Description

### Core Algorithm Files

- **`c51_expected_sarsa.py`**: ES-C51 implementation for Gym environments
- **`c51_atari_expected_sarsa.py`**: ES-C51 implementation for Atari environments
- **`c51.py`**: Modified C51 with softmax action selection (QL-C51) for Gym environments
- **`c51_atari.py`**: Modified C51 with softmax action selection (QL-C51) for Atari environments

### Key Modifications

#### 1. Action Selection Policy
Both ES-C51 and QL-C51 use **softmax action selection** instead of ε-greedy:
```python
# Softmax policy with temperature τ
policy = torch.softmax(q_values / tau, dim=1)
actions = torch.multinomial(policy, 1).squeeze(1).cpu().numpy()
```

#### 2. Target Distribution Construction (ES-C51 vs QL-C51)

**Standard QL-C51 (Greedy Bellman Backup)**:
```python
_, next_pmfs = target_network.get_action(data.next_observations)  # Greedy action
```

**ES-C51 (Expected Sarsa Backup)**:
```python
# Compute all action distributions
pmfs_all = torch.softmax(logits.view(batch_size, n_actions, n_atoms), dim=2)
q_values = (pmfs_all * atoms).sum(2)

# Use softmax policy to weight all actions
policy = torch.softmax(q_values / tau, dim=1)
expected_next_pmfs = (policy.unsqueeze(2) * pmfs_all).sum(1)  # Expected distribution
```

#### 3. Temperature Decay
Both algorithms use decaying temperature for exploration-exploitation trade-off:
```python
tau = max(1.0 * (1 - global_step / args.total_timesteps), 0.01)
```

## Theoretical Foundation

### Problem with Standard C51
Standard C51 can suffer from instability when multiple actions have similar expected rewards but different variances. The greedy selection can cause:
- Frequent switching between actions with similar Q-values
- Unstable distribution learning
- Policy churn affecting convergence

### ES-C51 Solution
ES-C51 addresses these issues by:
- Using Expected Sarsa updates that consider all possible actions
- Weighting actions by their softmax probabilities
- Reducing variance in target distribution construction
- Providing more stable learning, especially early in training

## Experimental Results

### Environments Tested
- **Gym Classic Control**: CartPole-v1, Acrobot-v1
- **Atari Games**: 10 games from Atari-10 benchmark in both stochastic (v0) and deterministic (NoFrameskip-v4) versions

### Key Findings
- ES-C51 achieves higher mean reward than QL-C51 on **72.7%** of environments
- **Deterministic environments**: More consistent improvements (median ~6-7%, range: -4.74% to 43.98%)
- **Stochastic environments**: More varied results but still positive median improvement (~7%)
- **Runtime**: Comparable to QL-C51, sometimes even faster due to streamlined updates

## Usage

### Running ES-C51 on Gym Environments
```bash
python c51_expected_sarsa.py --env-id CartPole-v1 --total-timesteps 500000
```

### Running ES-C51 on Atari Environments
```bash
python c51_atari_expected_sarsa.py --env-id BreakoutNoFrameskip-v4 --total-timesteps 10000000
```

### Running QL-C51 for Comparison
```bash
python c51.py --env-id CartPole-v1 --total-timesteps 500000
python c51_atari.py --env-id BreakoutNoFrameskip-v4 --total-timesteps 10000000
```

## Hyperparameters

### Gym Environments
- `n_atoms`: 101
- `v_min`: -100, `v_max`: 100
- `learning_rate`: 2.5e-4
- `batch_size`: 128
- `buffer_size`: 10000

### Atari Environments
- `n_atoms`: 51
- `v_min`: -10, `v_max`: 10
- `learning_rate`: 2.5e-4
- `batch_size`: 32
- `buffer_size`: 1000000

## Mathematical Foundation

### Distributional Bellman Operator (ES-C51)
```
𝒯^π Z(s,a) = R(s,a) + γ ∑_{a'∈𝒜} π_τ(a'|s') · Z(s',a')
```

### Cross-Entropy Loss
```
ℒ(θ) = -∑ᵢ Z_target[i] * log(p_θ(s,a,zᵢ))
```

## Citation

If you use this code in your research, please cite our paper:

```bibtex
@article{tandon2025esc51,
  title={ES-C51: Expected Sarsa Based C51 Distributional Reinforcement Learning Algorithm},
  author={Tandon, Rijul and Vamplew, Peter and Foale, Cameron},
  journal={Neurocomputing},
  year={2025},
  note={Submitted}
}
```

## Dependencies

- gymnasium
- torch
- numpy
- tyro
- tensorboard
- cleanrl_utils

## Contact

For questions about this implementation, please contact:
- Rijul Tandon: letscomerijul@gmail.com
- Peter Vamplew: p.vamplew@federation.edu.au
- Cameron Foale: c.foale@federation.edu.au

## License

This code is released under the same license as CleanRL.
