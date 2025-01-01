from typing import List, Tuple
import gymnasium as gym
import numpy as np
import torch
import torch.nn


class MLP(torch.nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        output_dim: int,
        activation=torch.nn.ReLU(),
    ):
        super().__init__()
        if len(hidden_dims) == 0:
            layers = [torch.nn.Linear(input_dim, output_dim)]
        else:
            layers = [torch.nn.Linear(input_dim, hidden_dims[0]), activation]
            for i in range(len(hidden_dims) - 1):
                layers.append(torch.nn.Linear(hidden_dims[i], hidden_dims[i + 1]))
                layers.append(activation)
            layers.append(torch.nn.Linear(hidden_dims[-1], output_dim))
        self.layers = torch.nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class StochasticActor(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def sample(self, obs: torch.Tensor)-> Tuple[torch.Tensor, torch.Tensor]:
        """
        Given the observation, compute the action and the log probability of the action.
        """
        pass

class MlpGaussianActor(StochasticActor):
    def __init__(
        self, env: gym.Env, hidden_dims: List[int], activation=torch.nn.ReLU()
    ):
        super().__init__()
        self.mlp = MLP(
            np.array(env.observation_space.shape).prod(),
            hidden_dims,
            np.array(env.action_space.shape).prod(),
            activation,
        )
        self.logstd = torch.nn.Parameter(
            torch.zeros(np.array(env.action_space.shape).prod())
        )

    def sample(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Samples an action, together with the log-probability of the action.
        """
        act_mean = self.mlp.forward(obs)
        if act_mean.ndim == 1:
            dist = torch.distributions.normal.Normal(act_mean, torch.exp(self.logstd))
        elif act_mean.ndim == 2:
            dist = torch.distributions.normal.Normal(
                act_mean, torch.exp(self.logstd).expand(act_mean.shape[0], 1)
            )
        else:
            raise Exception(f"Unknown exception, {act_mean.shape}")
        action = dist.sample()
        logprob = dist.log_prob(action)
        return action, logprob
