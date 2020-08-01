# References:
# https://github.com/hill-a/stable-baselines/blob/65ed3969e8859092e32e0cf89ac42959a7f283d6/stable_baselines/common/input.py#L6
# https://github.com/hill-a/stable-baselines/blob/65ed3969e8859092e32e0cf89ac42959a7f283d6/stable_baselines/common/distributions.py#L182

import torch
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
import numpy as np
from gym.spaces import Discrete, Box, MultiBinary, MultiDiscrete, Space

def preprocess_obs_space(obs_space: Space, device: str):
    """
    The `preprocess_obs_fn` receives the observation `x` in the shape of
    `(batch_num,) + obs_space.shape`.

    1) If the `obs_space` is `Discrete`, `preprocess_obs_fn` outputs a
    preprocessed obs in the shape of
    `(batch_num, obs_space.n)`.

    2) If the `obs_space` is `Box`, `preprocess_obs_fn` outputs a
    preprocessed obs in the shape of
    `(batch_num,) + obs_space.shape`.

    In addition, the preprocessed obs will be sent to `device` (either
    `cpu` or `cuda`)
    """
    if isinstance(obs_space, Discrete):
        def preprocess_obs_fn(x):
            return F.one_hot(torch.LongTensor(x), obs_space.n).float().to(device)
        return (obs_space.n, preprocess_obs_fn)

    elif isinstance(obs_space, Box):
        def preprocess_obs_fn(x):
            return torch.Tensor(x).float().view(torch.Tensor(x).shape[0], -1).to(device)
        return (np.array(obs_space.shape).prod(), preprocess_obs_fn)

    else:
        raise NotImplementedError("Error: the model does not support input space of type {}".format(
            type(obs_space).__name__))

def preprocess_ac_space(ac_space: Space):
    if isinstance(ac_space, Discrete):
        return ac_space.n

    elif isinstance(ac_space, MultiDiscrete):
        return ac_space.nvec.sum()

    elif isinstance(ac_space, Box):
        return np.prod(ac_space.shape)

    else:
        raise NotImplementedError("Error: the model does not support output space of type {}".format(
            type(ac_space).__name__))
