import importlib
import gym
import numpy as np
from gym import error, spaces, utils
from gym.envs.registration import register

"""
First kind of termination: True Done
As an example, the true done of the Breakout game in Atari 2600
comes when you lose all of your lives.
"""
class TestEnv(gym.Env):
    """
    A simple env that always ends after 10 timesteps, which 
    can be considered as the ``true done'' from the environment.
    At each timestep, its observation is its internal timestep
    """
    def __init__(self):
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(low=np.array([-1.]),
            high=np.array([10.]))
    def step(self, action):
        self.t += 1
        return np.array([0.])+self.t, 1, self.t==10, {}
    def reset(self):
        self.t = 0
        return np.array([0.])+self.t

entry_point='__main__:TestEnv'
name = entry_point
mod_name, attr_name = name.split(":")
mod = importlib.import_module(mod_name)
fn = getattr(mod, attr_name)
print(fn)

entry_point='gym.envs.classic_control:CartPoleEnv'
name = entry_point
mod_name, attr_name = name.split(":")
mod = importlib.import_module(mod_name)
fn = getattr(mod, attr_name)
print(fn)

env = fn()
print(env)
print(env.reset())