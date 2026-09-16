import numpy as np


def _compute_gae(dones, values, rewards, next_done, next_value, gamma=0.99, gae_lambda=0.95):
    advantages = np.zeros_like(rewards, dtype=np.float64)
    lastgaelam = 0.0
    for t in reversed(range(len(rewards))):
        if t == len(rewards) - 1:
            nextnonterminal = 1.0 - np.logical_or(dones[t], next_done)
            nextvalues = next_value
        else:
            nextnonterminal = 1.0 - np.logical_or(dones[t], dones[t + 1])
            nextvalues = values[t + 1]
        delta = rewards[t] + gamma * nextvalues * nextnonterminal - values[t]
        lastgaelam = delta + gamma * gae_lambda * nextnonterminal * lastgaelam
        advantages[t] = lastgaelam
    return advantages


def test_terminal_observation_cuts_envpool_dummy_transition():
    """Do not bootstrap from EnvPool's reset state after a terminal frame."""
    dones = np.array([False, True, False])
    values = np.array([1.0, 2.0, 10.0])
    rewards = np.array([0.0, 0.0])

    advantages = _compute_gae(dones, values, rewards, next_done=False, next_value=10.0)

    # Both the terminal transition and the following dummy transition are cut.
    assert np.allclose(advantages, [-1.0, -2.0])
