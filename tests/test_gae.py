import torch

from cleanrl_utils.gae import compute_gae


def test_envpool_dummy_transition_cuts_gae_trace():
    """A terminal observation must not bootstrap from the reset observation."""
    rewards = torch.tensor([[1.0], [2.0], [3.0], [4.0]])
    values = torch.tensor([[10.0], [20.0], [30.0], [40.0]])
    # The observation at step 2 is EnvPool's post-terminal dummy frame.
    dones = torch.tensor([[0.0], [0.0], [1.0], [0.0]])
    next_value = torch.tensor([50.0])
    next_done = torch.tensor([0.0])

    advantages, returns = compute_gae(
        rewards,
        values,
        next_value,
        dones,
        next_done,
        gamma=0.9,
        gae_lambda=1.0,
    )

    # The dummy transition (step 2) is isolated, and step 1 also stops at the
    # terminal boundary. Without the current-done mask, step 2 would
    # incorrectly bootstrap from the reset observation (advantage 17.1).
    expected = torch.tensor([[-7.2], [-18.0], [-27.0], [9.0]])
    torch.testing.assert_close(advantages, expected)
    torch.testing.assert_close(returns, expected + values)


def test_final_dummy_transition_does_not_bootstrap():
    rewards = torch.tensor([[0.0], [1.0]])
    values = torch.tensor([[2.0], [3.0]])
    dones = torch.tensor([[0.0], [1.0]])

    advantages, _ = compute_gae(
        rewards,
        values,
        next_value=torch.tensor([100.0]),
        dones=dones,
        next_done=torch.tensor([0.0]),
        gamma=0.99,
        gae_lambda=0.95,
    )

    # The final stored observation is terminal, so its transition target is
    # simply reward - value, despite the non-terminal flag returned next.
    torch.testing.assert_close(advantages[-1], torch.tensor([-2.0]))


def test_intrinsic_mode_preserves_terminal_bootstrap_but_cuts_dummy():
    rewards = torch.tensor([[1.0], [1.0], [1.0]])
    values = torch.tensor([[0.0], [0.0], [10.0]])
    dones = torch.tensor([[0.0], [0.0], [1.0]])

    advantages, _ = compute_gae(
        rewards,
        values,
        next_value=torch.tensor([20.0]),
        dones=dones,
        next_done=torch.tensor([1.0]),
        gamma=0.5,
        gae_lambda=1.0,
        bootstrap_on_terminal=False,
    )

    # The true terminal transition still bootstraps for intrinsic rewards,
    # while the current dummy observation is isolated.
    torch.testing.assert_close(advantages, torch.tensor([[1.75], [1.5], [-9.0]]))
