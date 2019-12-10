# References:
# https://github.com/hill-a/stable-baselines/blob/65ed3969e8859092e32e0cf89ac42959a7f283d6/stable_baselines/common/input.py#L6
# https://github.com/hill-a/stable-baselines/blob/65ed3969e8859092e32e0cf89ac42959a7f283d6/stable_baselines/common/distributions.py#L182

import torch
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
import numpy as np
from gym.spaces import Discrete, Box, MultiBinary, MultiDiscrete, Space

def preprocess_obs_space(obs_space: Space, device: str):
    if isinstance(obs_space, Discrete):
        return (obs_space.n,
                lambda x, obs_space=obs_space: F.one_hot(torch.LongTensor(x), obs_space.n).float().to(device))

    elif isinstance(obs_space, Box):
        return (np.array(obs_space.shape).prod(),
                lambda x, obs_space=obs_space: torch.Tensor(x).float().view(torch.Tensor(x).shape[0], -1).to(device))

    else:
        raise NotImplementedError("Error: the model does not support input space of type {}".format(
            type(obs_space).__name__))

def preprocess_ac_space(ac_space: Space, stochastic=True):
    if isinstance(ac_space, Discrete):
        return ac_space.n

    elif isinstance(ac_space, MultiDiscrete):
        return ac_space.nvec.sum()

    elif isinstance(ac_space, Box):
        return np.prod(ac_space.shape)

    else:
        raise NotImplementedError("Error: the model does not support output space of type {}".format(
            type(ac_space).__name__))

def preprocess_obs_ac_concat( obs_space: Space, ac_space: Space, device: str):
    if isinstance( obs_space, Box) and isinstance( ac_space, Box):
        return (np.array(obs_space.shape).prod() + np.array(ac_space.shape).prod(),
                lambda x, obs_space=obs_space, ac_space=ac_space: torch.Tensor(x).float().view(torch.Tensor(x).shape[0], -1).to(device))
    else:
        raise NotImplementedError("Error: the model does not support obs_space and act_space differnt of Box (yet).")
