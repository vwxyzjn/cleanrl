import cleanrl.common as mut

import gymnasium as gym
import pytest
import torch


def test_mlp():
    dut = mut.MLP(input_dim=2, hidden_dims=[], output_dim=3)
    assert len(dut.layers) == 1

    dut = mut.MLP(
        input_dim=2, hidden_dims=[4, 8], output_dim=1, activation=torch.nn.ReLU()
    )
    assert len(dut.layers) == 5
    assert isinstance(dut.layers[0], torch.nn.Linear)
    assert isinstance(dut.layers[2], torch.nn.Linear)
    assert isinstance(dut.layers[4], torch.nn.Linear)


def test_mlp_gaussian_actor():
    env = gym.make("MountainCarContinuous-v0")
    dut = mut.MlpGaussianActor(env, [4, 8])

    obs = torch.tensor([1.0, 2.0])

    act, logprob = dut.sample(obs)
    assert act.shape == (1,)
    assert logprob.shape == (1,)

    # A batch of observations
    obs = torch.tensor([[1.0, 2], [3, 0.4], [5, 7]])
    act, logprob = dut.sample(obs)
    assert act.shape == (3, 1)
    assert logprob.shape == (3, 1)
