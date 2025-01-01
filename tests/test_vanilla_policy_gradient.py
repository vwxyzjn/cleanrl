import cleanrl.vanilla_policy_gradient as mut

import pytest
import torch
import numpy as np
import gymnasium as gym


def test_reward_to_go():
    # A single reward traj
    r_traj = np.array([0.5, 0.4, 0.3])
    gamma = 0.99
    rtg = mut.reward_to_go(r_traj, gamma)
    assert rtg.shape == (3,)
    np.testing.assert_allclose(rtg[-1], r_traj[-1] * gamma**2)
    for i in range(2):
        np.testing.assert_allclose(rtg[i], rtg[i + 1] + r_traj[i] * gamma**i)

    # A batch of r_traj
    r_traj = np.array([[-0.5, 0.2, 0.1], [1.2, -0.3, 0.5]])
    rtg = mut.reward_to_go(r_traj, gamma)
    assert rtg.shape == (2, 3)
    for i in range(2):
        rtg_i = mut.reward_to_go(r_traj[i], gamma)
        np.testing.assert_allclose(rtg[i], rtg_i)


def test_calc_J():
    # Test a single r_traj
    r_traj = np.array([0.5, 1, 3])
    gamma = 0.9
    logprob_traj = torch.tensor([0.3, 1.5, 2])
    J = mut.cal_J(r_traj, gamma, logprob_traj)
    rtg = mut.reward_to_go(r_traj, gamma)
    np.testing.assert_allclose(J.item(), np.sum(rtg * logprob_traj.detach().numpy()))

    # Test a batch of r_traj
    r_traj = np.array([[0.5, 1, 1.2], [0.2, 3, -0.1]])
    logprob_traj = torch.tensor([[-0.1, 0.4, 0.3], [0.2, 0.5, 0.4]])
    J = mut.cal_J(r_traj, gamma, logprob_traj)
    J_single_traj = [mut.cal_J(r_traj[i], gamma, logprob_traj[i]) for i in range(2)]
    np.testing.assert_allclose(J.item(), (J_single_traj[0] + J_single_traj[1]).item())
