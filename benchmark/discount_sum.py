#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 12 17:51:32 2020

@author: costa
"""
import numpy as np
import scipy
import scipy.signal
def discount_cumsum(x, discount):
    """
    magic from rllab for computing discounted cumulative sums of vectors.
    input:
        vector x,
        [x0,
         x1,
         x2]
    output:
        [x0 + discount * x1 + discount^2 * x2,
         x1 + discount * x2,
         x2]
    """
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]


def discount_cumsum1(rewards, gamma):
    returns = np.zeros_like(rewards)
    returns[-1] = rewards[-1]
    for t in reversed(range(rewards.shape[0]-1)):
        returns[t] = rewards[t] + gamma * returns[t+1]
    return returns

episode = np.random.rand(1000)
discount_cumsum(episode, 0.99)
discount_cumsum1(episode, 0.99)