"""
Vanilla policy gradient
dJ = E[\nabla log pi(a|s) * rtg]
where rtg is the reward-to-go
"""

from typing import List, Tuple

import gymnasium as gym
import numpy as np
import torch
import torch.nn

import cleanrl.common


def reward_to_go(r_traj: np.ndarray, gamma: float) -> np.ndarray:
    """
    Compute reward_to_go = sum_{t'} r(s_t', a_t') * gamma^(t')
    """
    T = r_traj.size if r_traj.ndim == 1 else r_traj.shape[1]
    gamma_power = np.power(gamma, np.arange(T))
    return np.flip(np.flip(r_traj * gamma_power, axis=-1).cumsum(axis=-1), axis=-1)


def cal_J(r_traj: np.ndarray, gamma: float, logprob_traj: torch.Tensor):
    """
    Compute the J = sum_t log pi(a_t|s_t) * rtg
    """
    assert r_traj.shape == logprob_traj.shape
    rtg = reward_to_go(r_traj, gamma).copy()
    return torch.sum(
        torch.from_numpy(rtg).to(logprob_traj.device).to(logprob_traj.dtype)
        * logprob_traj
    )
