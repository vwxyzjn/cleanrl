"""Generalized advantage estimation helpers for EnvPool rollouts.

EnvPool exposes the terminal observation for one extra step after an episode
ends.  The ``done`` value stored with that observation therefore identifies a
dummy transition (terminal observation -> automatically reset observation),
not just the end of the preceding transition.  GAE must cut traces on both
sides of that boundary.
"""

from __future__ import annotations

import torch as th


def compute_gae(
    rewards: th.Tensor,
    values: th.Tensor,
    next_value: th.Tensor,
    dones: th.Tensor,
    next_done: th.Tensor,
    gamma: float,
    gae_lambda: float,
    *,
    bootstrap_on_terminal: bool = True,
) -> tuple[th.Tensor, th.Tensor]:
    """Compute GAE while isolating EnvPool's post-terminal dummy steps.

    Args:
        rewards: Rewards with shape ``[num_steps, num_envs]``.
        values: Value estimates for the stored observations.
        next_value: Value estimate after the final stored transition.
        dones: ``done`` values captured *before* each action.  In EnvPool a
            true value marks an observation from the terminal frame and must
            not be allowed to bootstrap into the automatically reset episode.
        next_done: ``done`` value returned after the final transition.
        gamma: Discount factor.
        gae_lambda: GAE trace factor.
        bootstrap_on_terminal: Keep the historical non-episodic intrinsic
            reward behavior when false, while still cutting dummy steps.

    Returns:
        A tuple ``(advantages, returns)`` with the same shape as ``values``.
    """
    if rewards.ndim != 2 or values.shape != rewards.shape or dones.shape != rewards.shape:
        raise ValueError("rewards, values, and dones must have shape [num_steps, num_envs]")

    next_value = next_value.reshape(-1)
    next_done = next_done.reshape(-1)
    if next_value.shape != rewards.shape[1:] or next_done.shape != rewards.shape[1:]:
        raise ValueError("next_value and next_done must have shape [num_envs]")

    advantages = th.zeros_like(rewards)
    last_gae = th.zeros_like(next_value)

    for step in reversed(range(rewards.shape[0])):
        if step == rewards.shape[0] - 1:
            following_done = next_done
            following_value = next_value
        else:
            following_done = dones[step + 1]
            following_value = values[step + 1]

        # ``dones[step]`` marks EnvPool's terminal observation.  Combining it
        # with the following transition's done flag cuts both the dummy step
        # itself and the transition that produced the terminal observation.
        boundary = dones[step].bool()
        if bootstrap_on_terminal:
            boundary = boundary | following_done.bool()
        continuation = (~boundary).to(dtype=rewards.dtype)

        delta = rewards[step] + gamma * following_value * continuation - values[step]
        last_gae = delta + gamma * gae_lambda * continuation * last_gae
        advantages[step] = last_gae

    return advantages, advantages + values


__all__ = ["compute_gae"]
