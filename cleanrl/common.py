# References:
# https://github.com/hill-a/stable-baselines/blob/65ed3969e8859092e32e0cf89ac42959a7f283d6/stable_baselines/common/input.py#L6
# https://github.com/hill-a/stable-baselines/blob/65ed3969e8859092e32e0cf89ac42959a7f283d6/stable_baselines/common/distributions.py#L182

import torch
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
import numpy as np
from gym.spaces import Discrete, Box, MultiBinary, MultiDiscrete, Space

def preprocess_obs_space(obs_space: Space):
    if isinstance(obs_space, Discrete):
        return (obs_space.n,
                lambda x, obs_space=obs_space: F.one_hot(torch.LongTensor([x]), obs_space.n).float())

    elif isinstance(obs_space, Box):
        return (np.array(obs_space.shape).prod(),
                lambda x, obs_space=obs_space: torch.Tensor([x]).float())

    # elif isinstance(obs_space, MultiBinary):
    #     observation_ph = tf.placeholder(shape=(batch_size, obs_space.n), dtype=tf.int32, name=name)
    #     processed_observations = tf.cast(observation_ph, tf.float32)
    #     return observation_ph, processed_observations

    # elif isinstance(obs_space, MultiDiscrete):
    #     observation_ph = tf.placeholder(shape=(batch_size, len(obs_space.nvec)), dtype=tf.int32, name=name)
    #     processed_observations = tf.concat([
    #         tf.cast(tf.one_hot(input_split, obs_space.nvec[i]), tf.float32) for i, input_split
    #         in enumerate(tf.split(observation_ph, len(obs_space.nvec), axis=-1))
    #     ], axis=-1)
    #     return observation_ph, processed_observations

    else:
        raise NotImplementedError("Error: the model does not support input space of type {}".format(
            type(obs_space).__name__))

def preprocess_ac_space(ac_space: Space):
    if isinstance(ac_space, Discrete):
        return (ac_space.n,
                lambda logits, ac_space=ac_space: __preprocess_ac_space_discrete(logits, ac_space))

    elif isinstance(ac_space, MultiDiscrete):
        return (ac_space.nvec.sum(),
                lambda logits, ac_space=ac_space: __preprocess_ac_space_multi_discrete(logits, ac_space))

    else:
        raise NotImplementedError("Error: the model does not support output space of type {}".format(
            type(ac_space).__name__))

def __preprocess_ac_space_discrete(logits: torch.Tensor, ac_space: Space):
    probs = Categorical(logits=logits)
    action = probs.sample()
    return probs, action.int(), -probs.log_prob(action), probs.entropy()

def __preprocess_ac_space_multi_discrete(logits: torch.Tensor, ac_space: Space):
    logits_categories = torch.split(logits, ac_space.nvec.tolist(), dim=1)
    action = []
    probs_categories = []
    probs_entropies = torch.tensor(0.)
    neglogprob = torch.tensor(0.)
    for i in range(len(logits_categories)):
        probs_categories.append(Categorical(logits=logits_categories[i]))
        action.append(probs_categories[i].sample().int().squeeze())
        neglogprob -= probs_categories[i].log_prob(action[i]).squeeze()
        probs_entropies += probs_categories[i].entropy().squeeze()
    return probs_categories, action, neglogprob, probs_entropies