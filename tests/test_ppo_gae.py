import numpy as np
import torch


def compute_gae_recurrence(rewards, values, dones, next_value, next_done, num_steps, gamma, gae_lambda):
    """Reverse in-place GAE recurrence, mirroring the implementation in cleanrl/ppo.py."""
    advantages = torch.zeros_like(rewards)
    lastgaelam = 0
    for t in reversed(range(num_steps)):
        if t == num_steps - 1:
            nextnonterminal = 1.0 - next_done
            nextvalues = next_value
        else:
            nextnonterminal = 1.0 - dones[t + 1]
            nextvalues = values[t + 1]
        delta = rewards[t] + gamma * nextvalues * nextnonterminal - values[t]
        advantages[t] = lastgaelam = delta + gamma * gae_lambda * nextnonterminal * lastgaelam
    returns = advantages + values
    return advantages, returns


def compute_gae_explicit(rewards, values, dones, next_value, next_done, num_steps, gamma, gae_lambda):
    """Independent reference: GAE as the explicit discounted sum of TD residuals.

    A_t = sum_{k=t}^{T-1} (prod_{j=t}^{k-1} gamma * lambda * nnt_j) * delta_k

    This uses a different accumulation path (forward explicit products) than the
    reverse in-place recurrence above, so agreement validates the recurrence logic.
    """
    rewards = rewards.numpy().astype(np.float64)
    values = values.numpy().astype(np.float64)
    dones = dones.numpy().astype(np.float64)
    next_value = float(next_value)
    next_done = float(next_done)

    # nextnonterminal[t] and nextvalues[t] exactly as used by the recurrence
    nnt = np.empty(num_steps)
    nv = np.empty(num_steps)
    for t in range(num_steps):
        if t == num_steps - 1:
            nnt[t] = 1.0 - next_done
            nv[t] = next_value
        else:
            nnt[t] = 1.0 - dones[t + 1]
            nv[t] = values[t + 1]

    deltas = rewards + gamma * nv * nnt - values

    advantages = np.zeros(num_steps)
    for t in range(num_steps):
        coef = 1.0
        for k in range(t, num_steps):
            advantages[t] += coef * deltas[k]
            coef *= gamma * gae_lambda * nnt[k]
    returns = advantages + values
    return advantages, returns


def test_ppo_gae_matches_explicit_sum():
    """The reverse recurrence in ppo.py must equal the explicit GAE summation."""
    num_steps = 64
    gamma = 0.99
    gae_lambda = 0.95
    g = torch.Generator().manual_seed(42)

    rewards = torch.rand(num_steps, generator=g, dtype=torch.float64) * 2 - 1
    values = torch.rand(num_steps, generator=g, dtype=torch.float64)
    dones = torch.randint(0, 2, (num_steps,), generator=g).to(torch.float64)
    next_value = torch.rand(1, generator=g, dtype=torch.float64).item()
    next_done = float(torch.randint(0, 2, (1,), generator=g).item())

    adv_rec, ret_rec = compute_gae_recurrence(
        rewards, values, dones, next_value, next_done, num_steps, gamma, gae_lambda
    )
    adv_ref, ret_ref = compute_gae_explicit(
        rewards, values, dones, next_value, next_done, num_steps, gamma, gae_lambda
    )

    np.testing.assert_allclose(adv_rec.numpy(), adv_ref, rtol=1e-9, atol=1e-9)
    np.testing.assert_allclose(ret_rec.numpy(), ret_ref, rtol=1e-9, atol=1e-9)


def test_ppo_gae_hand_computed_example():
    """Tiny fixed example with advantages worked out by hand (no episode terminations)."""
    gamma = 0.5
    gae_lambda = 0.5  # gamma * gae_lambda = 0.25
    rewards = torch.tensor([1.0, 1.0, 1.0])
    values = torch.tensor([0.0, 0.0, 0.0])
    dones = torch.tensor([0.0, 0.0, 0.0])
    next_value = torch.tensor([0.0])
    next_done = torch.tensor([0.0])

    # deltas: r_t + gamma*V(s_{t+1}) - V(s_t) = 1 for every t (all values are 0)
    # A_2 = 1
    # A_1 = 1 + 0.25 * A_2 = 1.25
    # A_0 = 1 + 0.25 * A_1 = 1.3125
    expected_adv = np.array([1.3125, 1.25, 1.0])

    adv, ret = compute_gae_recurrence(rewards, values, dones, next_value, next_done, 3, gamma, gae_lambda)
    np.testing.assert_allclose(adv.numpy(), expected_adv, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(ret.numpy(), expected_adv, rtol=1e-6, atol=1e-6)  # returns == adv since values == 0
