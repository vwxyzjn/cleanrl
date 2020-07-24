from stable_baselines3.common.vec_env import DummyVecEnv
import gym
import numpy as np
from gym import error, spaces, utils
from gym.envs.registration import register

"""
First kind of termination: true termination
As an example, the true termination of the Breakout game in Atari 2600
comes when you lose all of your lives.
"""
class TestEnv(gym.Env):
    """
    A simple env that always ends after 10 timesteps, which 
    can be considered as the ``true termination'' from the environment.
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

if "TestEnv-v0" not in gym.envs.registry.env_specs:
    register(
        "TestEnv-v0",
        entry_point='__main__:TestEnv'
    )

env = gym.make("TestEnv-v0")
print(f"env is {env}")
for i in range(2):
    all_obs = [env.reset()]
    while True:
        obs, reward, done, info = env.step(env.action_space.sample())
        all_obs += [obs]
        if done:
            print(f"all observation in episode {i}:")
            print(all_obs)
            print("true termination")
            print()
            break
print("=========")

"""
Second kind of termination: time limit termination
As an example, time limit termination comes when the episode of "CartPole-v0"
exceeds length 200.
"""
if "TestEnvTimeLimit3-v0" not in gym.envs.registry.env_specs:
    register(
        "TestEnvTimeLimit3-v0",
        entry_point='__main__:TestEnv',
        max_episode_steps=8
    )

env = gym.make("TestEnvTimeLimit3-v0")
# equivalent to below
# env = TestEnv()
# env = gym.wrappers.TimeLimit(env, max_episode_steps=3)
print(f"env is {env}")
print(f"env's timelimit is {env._max_episode_steps}")
for i in range(2):
    all_obs = [env.reset()]
    while True:
        obs, reward, done, info = env.step(env.action_space.sample())
        all_obs += [obs]
        if done:
            print(f"all observation in episode {i}:")
            print(all_obs)
            print("time limit termination")
            print()
            break
print("=========")

"""
Third kind of termination: early termination by `n_steps`
This is usually combined with TimeLimit wrapped env,
but you can use it without the TimeLimit
"""
n_steps = 5
envs = DummyVecEnv([
    lambda: gym.make("TestEnvTimeLimit3-v0")])
print(f"envs is {envs}")
print(f"envs' timelimit is {envs.envs[0]._max_episode_steps}")
obss = envs.reset()
for i in range(3):
    all_obss = []
    for j in range(n_steps):
        all_obss += [obss.astype("float")]
        obss, rewards, dones, infos = envs.step(np.array([1.,1.]))
        # print(infos)
    
    print(f"all observation in trajectory {i}:")
    print(all_obss)
    print("early termination by `n_steps`")
    print()
print("=========")