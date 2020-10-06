# https://github.com/facebookresearch/torchbeast/blob/master/torchbeast/core/environment.py

import numpy as np
from collections import deque
import gym
from gym import spaces
import cv2
cv2.ocl.setUseOpenCL(False)


class NoopResetEnv(gym.Wrapper):
    def __init__(self, env, noop_max=30):
        """Sample initial states by taking random number of no-ops on reset.
        No-op is assumed to be action 0.
        """
        gym.Wrapper.__init__(self, env)
        self.noop_max = noop_max
        self.override_num_noops = None
        self.noop_action = 0
        assert env.unwrapped.get_action_meanings()[0] == 'NOOP'

    def reset(self, **kwargs):
        """ Do no-op action for a number of steps in [1, noop_max]."""
        self.env.reset(**kwargs)
        if self.override_num_noops is not None:
            noops = self.override_num_noops
        else:
            noops = self.unwrapped.np_random.randint(1, self.noop_max + 1) #pylint: disable=E1101
        assert noops > 0
        obs = None
        for _ in range(noops):
            obs, _, done, _ = self.env.step(self.noop_action)
            if done:
                obs = self.env.reset(**kwargs)
        return obs

    def step(self, ac):
        return self.env.step(ac)

class FireResetEnv(gym.Wrapper):
    def __init__(self, env):
        """Take action on reset for environments that are fixed until firing."""
        gym.Wrapper.__init__(self, env)
        assert env.unwrapped.get_action_meanings()[1] == 'FIRE'
        assert len(env.unwrapped.get_action_meanings()) >= 3

    def reset(self, **kwargs):
        self.env.reset(**kwargs)
        obs, _, done, _ = self.env.step(1)
        if done:
            self.env.reset(**kwargs)
        obs, _, done, _ = self.env.step(2)
        if done:
            self.env.reset(**kwargs)
        return obs

    def step(self, ac):
        return self.env.step(ac)

class EpisodicLifeEnv(gym.Wrapper):
    def __init__(self, env):
        """Make end-of-life == end-of-episode, but only reset on true game over.
        Done by DeepMind for the DQN and co. since it helps value estimation.
        """
        gym.Wrapper.__init__(self, env)
        self.lives = 0
        self.was_real_done  = True

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.was_real_done = done
        # check current lives, make loss of life terminal,
        # then update lives to handle bonus lives
        lives = self.env.unwrapped.ale.lives()
        if lives < self.lives and lives > 0:
            # for Qbert sometimes we stay in lives == 0 condition for a few frames
            # so it's important to keep lives > 0, so that we only reset once
            # the environment advertises done.
            done = True
        self.lives = lives
        return obs, reward, done, info

    def reset(self, **kwargs):
        """Reset only when lives are exhausted.
        This way all states are still reachable even though lives are episodic,
        and the learner need not know about any of this behind-the-scenes.
        """
        if self.was_real_done:
            obs = self.env.reset(**kwargs)
        else:
            # no-op step to advance from terminal/lost life state
            obs, _, _, _ = self.env.step(0)
        self.lives = self.env.unwrapped.ale.lives()
        return obs

class MaxAndSkipEnv(gym.Wrapper):
    def __init__(self, env, skip=4):
        """Return only every `skip`-th frame"""
        gym.Wrapper.__init__(self, env)
        # most recent raw observations (for max pooling across time steps)
        self._obs_buffer = np.zeros((2,)+env.observation_space.shape, dtype=np.uint8)
        self._skip       = skip

    def step(self, action):
        """Repeat action, sum reward, and max over last observations."""
        total_reward = 0.0
        done = None
        for i in range(self._skip):
            obs, reward, done, info = self.env.step(action)
            if i == self._skip - 2: self._obs_buffer[0] = obs
            if i == self._skip - 1: self._obs_buffer[1] = obs
            total_reward += reward
            if done:
                break
        # Note that the observation on the done=True frame
        # doesn't matter
        max_frame = self._obs_buffer.max(axis=0)

        return max_frame, total_reward, done, info

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

class ClipRewardEnv(gym.RewardWrapper):
    def __init__(self, env):
        gym.RewardWrapper.__init__(self, env)

    def reward(self, reward):
        """Bin reward to {+1, 0, -1} by its sign."""
        return np.sign(reward)


class WarpFrame(gym.ObservationWrapper):
    def __init__(self, env, width=84, height=84, grayscale=True, dict_space_key=None):
        """
        Warp frames to 84x84 as done in the Nature paper and later work.
        If the environment uses dictionary observations, `dict_space_key` can be specified which indicates which
        observation should be warped.
        """
        super().__init__(env)
        self._width = width
        self._height = height
        self._grayscale = grayscale
        self._key = dict_space_key
        if self._grayscale:
            num_colors = 1
        else:
            num_colors = 3

        new_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=(self._height, self._width, num_colors),
            dtype=np.uint8,
        )
        if self._key is None:
            original_space = self.observation_space
            self.observation_space = new_space
        else:
            original_space = self.observation_space.spaces[self._key]
            self.observation_space.spaces[self._key] = new_space
        assert original_space.dtype == np.uint8 and len(original_space.shape) == 3

    def observation(self, obs):
        if self._key is None:
            frame = obs
        else:
            frame = obs[self._key]

        if self._grayscale:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        frame = cv2.resize(
            frame, (self._width, self._height), interpolation=cv2.INTER_AREA
        )
        if self._grayscale:
            frame = np.expand_dims(frame, -1)

        if self._key is None:
            obs = frame
        else:
            obs = obs.copy()
            obs[self._key] = frame
        return obs


class FrameStack(gym.Wrapper):
    def __init__(self, env, k):
        """Stack k last frames.
        Returns lazy array, which is much more memory efficient.
        See Also
        --------
        baselines.common.atari_wrappers.LazyFrames
        """
        gym.Wrapper.__init__(self, env)
        self.k = k
        self.frames = deque([], maxlen=k)
        shp = env.observation_space.shape
        self.observation_space = spaces.Box(low=0, high=255, shape=((shp[0] * k,)+shp[1:]), dtype=env.observation_space.dtype)

    def reset(self):
        ob = self.env.reset()
        for _ in range(self.k):
            self.frames.append(ob)
        return self._get_ob()

    def step(self, action):
        ob, reward, done, info = self.env.step(action)
        self.frames.append(ob)
        return self._get_ob(), reward, done, info

    def _get_ob(self):
        assert len(self.frames) == self.k
        return LazyFrames(list(self.frames))

class ScaledFloatFrame(gym.ObservationWrapper):
    def __init__(self, env):
        gym.ObservationWrapper.__init__(self, env)
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=env.observation_space.shape, dtype=np.float32)

    def observation(self, observation):
        # careful! This undoes the memory optimization, use
        # with smaller replay buffers only.
        return np.array(observation).astype(np.float32) / 255.0

class LazyFrames(object):
    def __init__(self, frames):
        """This object ensures that common frames between the observations are only stored once.
        It exists purely to optimize memory usage which can be huge for DQN's 1M frames replay
        buffers.
        This object should only be converted to numpy array before being passed to the model.
        You'd not believe how complex the previous solution was."""
        self._frames = frames
        self._out = None

    def _force(self):
        if self._out is None:
            self._out = np.concatenate(self._frames, axis=0)
            self._frames = None
        return self._out

    def __array__(self, dtype=None):
        out = self._force()
        if dtype is not None:
            out = out.astype(dtype)
        return out

    def __len__(self):
        return len(self._force())

    def __getitem__(self, i):
        return self._force()[i]

    def count(self):
        frames = self._force()
        return frames.shape[frames.ndim - 1]

    def frame(self, i):
        return self._force()[..., i]

def wrap_atari(env, max_episode_steps=None):
    assert 'NoFrameskip' in env.spec.id
    env = NoopResetEnv(env, noop_max=30)
    env = MaxAndSkipEnv(env, skip=4)

    assert max_episode_steps is None

    return env

class ImageToPyTorch(gym.ObservationWrapper):
    """
    Image shape to channels x weight x height
    """

    def __init__(self, env):
        super(ImageToPyTorch, self).__init__(env)
        old_shape = self.observation_space.shape
        self.observation_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=(old_shape[-1], old_shape[0], old_shape[1]),
            dtype=np.uint8,
        )

    def observation(self, observation):
        return np.transpose(observation, axes=(2, 0, 1))

def wrap_deepmind(env, episode_life=True, clip_rewards=True, frame_stack=False, scale=False):
    """Configure environment for DeepMind-style Atari.
    """
    if episode_life:
        env = EpisodicLifeEnv(env)
    if 'FIRE' in env.unwrapped.get_action_meanings():
        env = FireResetEnv(env)
    env = WarpFrame(env)
    if scale:
        env = ScaledFloatFrame(env)
    if clip_rewards:
        env = ClipRewardEnv(env)
    env = ImageToPyTorch(env)
    if frame_stack:
        env = FrameStack(env, 4)
    return env

# https://github.com/openai/baselines/blob/master/baselines/deepq/replay_buffer.py
import operator
class SegmentTree(object):
    def __init__(self, capacity, operation, neutral_element):
        """Build a Segment Tree data structure.
        https://en.wikipedia.org/wiki/Segment_tree
        Can be used as regular array, but with two
        important differences:
            a) setting item's value is slightly slower.
               It is O(lg capacity) instead of O(1).
            b) user has access to an efficient ( O(log segment size) )
               `reduce` operation which reduces `operation` over
               a contiguous subsequence of items in the array.
        Paramters
        ---------
        capacity: int
            Total size of the array - must be a power of two.
        operation: lambda obj, obj -> obj
            and operation for combining elements (eg. sum, max)
            must form a mathematical group together with the set of
            possible values for array elements (i.e. be associative)
        neutral_element: obj
            neutral element for the operation above. eg. float('-inf')
            for max and 0 for sum.
        """
        assert capacity > 0 and capacity & (capacity - 1) == 0, "capacity must be positive and a power of 2."
        self._capacity = capacity
        self._value = [neutral_element for _ in range(2 * capacity)]
        self._operation = operation

    def _reduce_helper(self, start, end, node, node_start, node_end):
        if start == node_start and end == node_end:
            return self._value[node]
        mid = (node_start + node_end) // 2
        if end <= mid:
            return self._reduce_helper(start, end, 2 * node, node_start, mid)
        else:
            if mid + 1 <= start:
                return self._reduce_helper(start, end, 2 * node + 1, mid + 1, node_end)
            else:
                return self._operation(
                    self._reduce_helper(start, mid, 2 * node, node_start, mid),
                    self._reduce_helper(mid + 1, end, 2 * node + 1, mid + 1, node_end)
                )

    def reduce(self, start=0, end=None):
        """Returns result of applying `self.operation`
        to a contiguous subsequence of the array.
            self.operation(arr[start], operation(arr[start+1], operation(... arr[end])))
        Parameters
        ----------
        start: int
            beginning of the subsequence
        end: int
            end of the subsequences
        Returns
        -------
        reduced: obj
            result of reducing self.operation over the specified range of array elements.
        """
        if end is None:
            end = self._capacity
        if end < 0:
            end += self._capacity
        end -= 1
        return self._reduce_helper(start, end, 1, 0, self._capacity - 1)

    def __setitem__(self, idx, val):
        # index of the leaf
        idx += self._capacity
        self._value[idx] = val
        idx //= 2
        while idx >= 1:
            self._value[idx] = self._operation(
                self._value[2 * idx],
                self._value[2 * idx + 1]
            )
            idx //= 2

    def __getitem__(self, idx):
        assert 0 <= idx < self._capacity
        return self._value[self._capacity + idx]


class SumSegmentTree(SegmentTree):
    def __init__(self, capacity):
        super(SumSegmentTree, self).__init__(
            capacity=capacity,
            operation=operator.add,
            neutral_element=0.0
        )

    def sum(self, start=0, end=None):
        """Returns arr[start] + ... + arr[end]"""
        return super(SumSegmentTree, self).reduce(start, end)

    def find_prefixsum_idx(self, prefixsum):
        """Find the highest index `i` in the array such that
            sum(arr[0] + arr[1] + ... + arr[i - i]) <= prefixsum
        if array values are probabilities, this function
        allows to sample indexes according to the discrete
        probability efficiently.
        Parameters
        ----------
        perfixsum: float
            upperbound on the sum of array prefix
        Returns
        -------
        idx: int
            highest index satisfying the prefixsum constraint
        """
        assert 0 <= prefixsum <= self.sum() + 1e-5
        idx = 1
        while idx < self._capacity:  # while non-leaf
            if self._value[2 * idx] > prefixsum:
                idx = 2 * idx
            else:
                prefixsum -= self._value[2 * idx]
                idx = 2 * idx + 1
        return idx - self._capacity


class MinSegmentTree(SegmentTree):
    def __init__(self, capacity):
        super(MinSegmentTree, self).__init__(
            capacity=capacity,
            operation=min,
            neutral_element=float('inf')
        )

    def min(self, start=0, end=None):
        """Returns min(arr[start], ...,  arr[end])"""

        return super(MinSegmentTree, self).reduce(start, end)

class ReplayBuffer(object):
    def __init__(self, size):
        """Create Replay buffer.
        Parameters
        ----------
        size: int
            Max number of transitions to store in the buffer. When the buffer
            overflows the old memories are dropped.
        """
        self._storage = []
        self._maxsize = size
        self._next_idx = 0

    def __len__(self):
        return len(self._storage)

    def add(self, obs_t, action, reward, obs_tp1, done):
        data = (obs_t, action, reward, obs_tp1, done)

        if self._next_idx >= len(self._storage):
            self._storage.append(data)
        else:
            self._storage[self._next_idx] = data
        self._next_idx = (self._next_idx + 1) % self._maxsize

    def _encode_sample(self, idxes):
        obses_t, actions, rewards, obses_tp1, dones = [], [], [], [], []
        for i in idxes:
            data = self._storage[i]
            obs_t, action, reward, obs_tp1, done = data
            obses_t.append(np.array(obs_t, copy=False))
            actions.append(np.array(action, copy=False))
            rewards.append(reward)
            obses_tp1.append(np.array(obs_tp1, copy=False))
            dones.append(done)
        return np.array(obses_t), np.array(actions), np.array(rewards), np.array(obses_tp1), np.array(dones)

    def sample(self, batch_size):
        """Sample a batch of experiences.
        Parameters
        ----------
        batch_size: int
            How many transitions to sample.
        Returns
        -------
        obs_batch: np.array
            batch of observations
        act_batch: np.array
            batch of actions executed given obs_batch
        rew_batch: np.array
            rewards received as results of executing act_batch
        next_obs_batch: np.array
            next set of observations seen after executing act_batch
        done_mask: np.array
            done_mask[i] = 1 if executing act_batch[i] resulted in
            the end of an episode and 0 otherwise.
        """
        idxes = [random.randint(0, len(self._storage) - 1) for _ in range(batch_size)]
        return self._encode_sample(idxes)


class PrioritizedReplayBuffer(ReplayBuffer):
    def __init__(self, size, alpha):
        """Create Prioritized Replay buffer.
        Parameters
        ----------
        size: int
            Max number of transitions to store in the buffer. When the buffer
            overflows the old memories are dropped.
        alpha: float
            how much prioritization is used
            (0 - no prioritization, 1 - full prioritization)
        See Also
        --------
        ReplayBuffer.__init__
        """
        super(PrioritizedReplayBuffer, self).__init__(size)
        assert alpha >= 0
        self._alpha = alpha

        it_capacity = 1
        while it_capacity < size:
            it_capacity *= 2

        self._it_sum = SumSegmentTree(it_capacity)
        self._it_min = MinSegmentTree(it_capacity)
        self._max_priority = 1.0

    def add(self, *args, **kwargs):
        """See ReplayBuffer.store_effect"""
        idx = self._next_idx
        super().add(*args, **kwargs)
        self._it_sum[idx] = self._max_priority ** self._alpha
        self._it_min[idx] = self._max_priority ** self._alpha

    def _sample_proportional(self, batch_size):
        res = []
        p_total = self._it_sum.sum(0, len(self._storage) - 1)
        every_range_len = p_total / batch_size
        for i in range(batch_size):
            mass = random.random() * every_range_len + i * every_range_len
            idx = self._it_sum.find_prefixsum_idx(mass)
            res.append(idx)
        return res

    def sample(self, batch_size, beta):
        """Sample a batch of experiences.
        compared to ReplayBuffer.sample
        it also returns importance weights and idxes
        of sampled experiences.
        Parameters
        ----------
        batch_size: int
            How many transitions to sample.
        beta: float
            To what degree to use importance weights
            (0 - no corrections, 1 - full correction)
        Returns
        -------
        obs_batch: np.array
            batch of observations
        act_batch: np.array
            batch of actions executed given obs_batch
        rew_batch: np.array
            rewards received as results of executing act_batch
        next_obs_batch: np.array
            next set of observations seen after executing act_batch
        done_mask: np.array
            done_mask[i] = 1 if executing act_batch[i] resulted in
            the end of an episode and 0 otherwise.
        weights: np.array
            Array of shape (batch_size,) and dtype np.float32
            denoting importance weight of each sampled transition
        idxes: np.array
            Array of shape (batch_size,) and dtype np.int32
            idexes in buffer of sampled experiences
        """
        assert beta > 0

        idxes = self._sample_proportional(batch_size)

        weights = []
        p_min = self._it_min.min() / self._it_sum.sum()
        max_weight = (p_min * len(self._storage)) ** (-beta)

        for idx in idxes:
            p_sample = self._it_sum[idx] / self._it_sum.sum()
            weight = (p_sample * len(self._storage)) ** (-beta)
            weights.append(weight / max_weight)
        weights = np.array(weights)
        encoded_sample = self._encode_sample(idxes)
        return tuple(list(encoded_sample) + [weights, idxes])

    def update_priorities(self, idxes, priorities):
        """Update priorities of sampled transitions.
        sets priority of transition at index idxes[i] in buffer
        to priorities[i].
        Parameters
        ----------
        idxes: [int]
            List of idxes of sampled transitions
        priorities: [float]
            List of updated priorities corresponding to
            transitions at the sampled idxes denoted by
            variable `idxes`.
        """
        assert len(idxes) == len(priorities)
        for idx, priority in zip(idxes, priorities):
            assert priority > 0
            assert 0 <= idx < len(self._storage)
            self._it_sum[idx] = priority ** self._alpha
            self._it_min[idx] = priority ** self._alpha

            self._max_priority = max(self._max_priority, priority)

# https://github.com/younggyoseo/Ape-X/blob/master/memory.py
class CustomPrioritizedReplayBuffer(PrioritizedReplayBuffer):
    """
    Customized PrioritizedReplayBuffer class
    1. Edited add method to receive priority as input. This enables to enter priority when adding sample.
    This efficiently merges two methods (add, update_priorities) which enables less shared memory lock.
    2. If we save obs as numpy.array, this will decompress LazyFrame which leads to memory explosion.
    To achieve memory efficiency, It is necessary to remove np.array(obs) from _encode_sample.
    """
    def __init__(self, size, alpha):
        super(CustomPrioritizedReplayBuffer, self).__init__(size, alpha)

    def add_with_priority(self, state, action, reward, next_state, done, priority):
        idx = self._next_idx
        data = (state, action, reward, next_state, done)

        if self._next_idx >= len(self._storage):
            self._storage.append(data)
        else:
            self._storage[self._next_idx] = data
        self._next_idx = (self._next_idx + 1) % self._maxsize

        self._it_sum[idx] = priority ** self._alpha
        self._it_min[idx] = priority ** self._alpha
        self._max_priority = max(self._max_priority, priority)

# Reference: https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

import argparse
from distutils.util import strtobool
import collections
import numpy as np
import gym
from gym.wrappers import TimeLimit, Monitor
from gym.spaces import Discrete, Box, MultiBinary, MultiDiscrete, Space
import time
import random
import os

import threading
os.environ["OMP_NUM_THREADS"] = "1"  # Necessary for multithreading.
from torch import multiprocessing as mp
from multiprocessing.managers import SyncManager

import matplotlib
matplotlib.use('Agg')
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image
import glob

class QValueVisualizationWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.env.reset()
        self.image_shape = self.env.render(mode="rgb_array").shape
        self.q_values = [[0.,0.,0.,0.]]
        # self.metadata['video.frames_per_second'] = 60
        
    def set_q_values(self, q_values):
        self.q_values = q_values

    def render(self, mode="human"):
        if mode=="rgb_array":
            env_rgb_array = super().render(mode)
            fig, ax = plt.subplots(figsize=(self.image_shape[1]/100,self.image_shape[0]/100), constrained_layout=True, dpi=100)
            df = pd.DataFrame(np.array(self.q_values).T)
            sns.barplot(x=df.index, y=0, data=df, ax=ax)
            ax.set(xlabel='actions', ylabel='q-values')
            fig.canvas.draw()
            X = np.array(fig.canvas.renderer.buffer_rgba())
            Image.fromarray(X)
            # Image.fromarray(X)
            rgb_image = np.array(Image.fromarray(X).convert('RGB'))
            plt.close(fig)
            q_value_rgb_array = rgb_image
            return np.append(env_rgb_array, q_value_rgb_array, axis=1)
        else:
            super().render(mode)

class Scale(nn.Module):
    def __init__(self, scale):
        super().__init__()
        self.scale = scale

    def forward(self, x):
        return x * self.scale
class QNetwork(nn.Module):
    def __init__(self, env, device="cpu", frames=4):
        super(QNetwork, self).__init__()
        self.device = device
        self.network = nn.Sequential(
            Scale(1/255),
            nn.Conv2d(frames, 32, 8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3136, 512),
            nn.ReLU(),
            nn.Linear(512, env.action_space.n)
        )

    def forward(self, x, device):
        x = torch.Tensor(x).to(self.device)
        return self.network(x)

def linear_schedule(start_e: float, end_e: float, duration: int, t: int):
    slope =  (end_e - start_e) / duration
    if end_e < start_e:
        return max(slope * t + start_e, end_e)
    else:
        return min(slope * t + start_e, end_e)

def act(args, experiment_name, i, q_network, target_network, lock, rollouts_queue, stats_queue, global_step, device):
    env = gym.make(args.gym_id)
    env = wrap_atari(env)
    env = gym.wrappers.RecordEpisodeStatistics(env) # records episode reward in `info['episode']['r']`
    if args.capture_video:
        if i == 0:
            env = QValueVisualizationWrapper(env)
            env = Monitor(env, f'videos/{experiment_name}')
    env = wrap_deepmind(
        env,
        clip_rewards=True,
        frame_stack=True,
        scale=False,
    )
    env.seed(args.seed+i)
    env.action_space.seed(args.seed+i)
    # TRY NOT TO MODIFY: start the game
    obs = env.reset()
    storage = []
    episode_reward = 0
    update_step = 0
    while True:
        update_step += 1
        # global_step *= args.num_actor
        # ALGO LOGIC: put action logic here
        epsilon = linear_schedule(args.start_e, args.end_e, args.exploration_fraction*args.total_timesteps, global_step)
        obs = np.array(obs)
        logits = q_network.forward(obs.reshape((1,)+obs.shape), device)
        if args.capture_video:
            if i == 0:
                env.set_q_values(logits.tolist())
        if random.random() < epsilon:
            action = env.action_space.sample()
        else:
            action = torch.argmax(logits, dim=1).tolist()[0]
    
        # TRY NOT TO MODIFY: execute the game and log data.
        next_obs, reward, done, info = env.step(action)
        episode_reward += reward
        storage += [(obs, action, reward, next_obs, float(done))]
        with lock:
            global_step += 1
        if 'episode' in info.keys():
            stats_queue.put(("charts/episode_reward", info['episode']['r'], info['episode']['l']))
        
        if len(storage) == args.actor_buffer_size:
            obses_t, actions, rewards, obses_tp1, dones = [], [], [], [], []
            for data in storage:
                obs_t, action, reward, obs_tp1, done = data
                obses_t.append(np.array(obs_t, copy=False))
                actions.append(np.array(action, copy=False))
                rewards.append(reward)
                obses_tp1.append(np.array(obs_tp1, copy=False))
                dones.append(done)
            s_obs, s_actions, s_rewards, s_next_obses, s_dones = np.array(obses_t), np.array(actions), np.array(rewards), np.array(obses_tp1), np.array(dones)

            with torch.no_grad():
                # target_max = torch.max(target_network.forward(s_next_obses), dim=1)[0]
                current_value = q_network.forward(s_next_obses, device)
                target_value = target_network.forward(s_next_obses, device)
                target_max = target_value.gather(1, torch.max(current_value, 1)[1].unsqueeze(1)).squeeze(1)
                td_target = torch.Tensor(s_rewards).to(device) + args.gamma * target_max * (1 - torch.Tensor(s_dones).to(device))
                
                old_val = q_network.forward(s_obs, device).gather(1, torch.LongTensor(s_actions).view(-1,1).to(device)).squeeze()
                td_errors = td_target - old_val
            new_priorities = np.abs(td_errors.tolist()) + args.pr_eps
            rollouts_queue.put((storage, new_priorities))
            storage = []

        # TRY NOT TO MODIFY: CRUCIAL step easy to overlook 
        obs = next_obs
        if done:
            # important to note that because `EpisodicLifeEnv` wrapper is applied,
            # the real episode reward is actually the sum of episode reward of 5 lives
            # which we record through `info['episode']['r']` provided by gym.wrappers.RecordEpisodeStatistics
            obs, episode_reward = env.reset(), 0

def data_process(args, i, global_step, rollouts_queue, data_process_queue, data_process_back_queues, device):
    worker_rb_size = args.buffer_size // args.num_data_processors
    rb = CustomPrioritizedReplayBuffer(worker_rb_size, args.pr_alpha)
    while True:
        (storage, new_priorities) = rollouts_queue.get()
        for data, priority in zip(storage, new_priorities):
            obs, action, reward, next_obs, done = data
            rb.add_with_priority(obs, action, reward, next_obs, done, priority)
        if len(rb) > args.learning_starts // args.num_data_processors:
            beta = linear_schedule(args.pr_beta0, 1.0, args.total_timesteps, global_step)
            experience = rb.sample(args.batch_size, beta=beta)
            (s_obs, s_actions, s_rewards, s_next_obses, s_dones, s_weights, s_batch_idxes) = experience
            
            s_obs, s_actions, s_rewards, s_next_obses, s_dones = torch.Tensor(s_obs).to(device), torch.LongTensor(s_actions).to(device), torch.Tensor(s_rewards).to(device), torch.Tensor(s_next_obses).to(device), torch.Tensor(s_dones).to(device)
            data_process_queue.put([i, s_obs, s_actions, s_rewards, s_next_obses, s_dones])
            new_priorities = data_process_back_queues[i].get()
            rb.update_priorities(s_batch_idxes, new_priorities)
            del new_priorities

def learn(args, rb, global_step, data_process_queue, data_process_back_queues, stats_queue, lock, learn_target_network, target_network, learn_q_network, q_network, optimizer, device):
    update_step = 0
    while True:
        update_step += 1
        experience = data_process_queue.get()
        (i, s_obs, s_actions, s_rewards, s_next_obses, s_dones) = experience
        with torch.no_grad():
            current_value = learn_q_network.network(s_next_obses)
            target_value = learn_target_network.network(s_next_obses)
            target_max = target_value.gather(1, torch.max(current_value, 1)[1].unsqueeze(1)).squeeze(1)

            td_target = s_rewards + args.gamma * target_max * (1 - s_dones)

        old_val = learn_q_network.network(s_obs).gather(1, s_actions.view(-1,1)).squeeze()
        td_errors = td_target - old_val
        loss = (td_errors ** 2).mean()
        
        # update the weights in the prioritized replay
        new_priorities = np.abs(td_errors.tolist()) + args.pr_eps
        data_process_back_queues[i].put(new_priorities)
        
        stats_queue.put(("losses/td_loss", loss.item(), update_step+args.learning_starts))
    
        # optimize the midel
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(list(learn_q_network.parameters()), args.max_grad_norm)
        optimizer.step()
        q_network.load_state_dict(learn_q_network.state_dict())
        del i, s_obs, s_actions, s_rewards, s_next_obses, s_dones
    
        # update the target network
        if update_step % args.target_network_frequency == 0:
            learn_target_network.load_state_dict(learn_q_network.state_dict())
            target_network.load_state_dict(learn_q_network.state_dict())

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='DQN agent')
    # Common arguments
    parser.add_argument('--exp-name', type=str, default=os.path.basename(__file__).rstrip(".py"),
                        help='the name of this experiment')
    parser.add_argument('--gym-id', type=str, default="BreakoutNoFrameskip-v4",
                        help='the id of the gym environment')
    parser.add_argument('--learning-rate', type=float, default=1e-4,
                        help='the learning rate of the optimizer')
    parser.add_argument('--seed', type=int, default=2,
                        help='seed of the experiment')
    parser.add_argument('--total-timesteps', type=int, default=10000000,
                        help='total timesteps of the experiments')
    parser.add_argument('--torch-deterministic', type=lambda x:bool(strtobool(x)), default=True, nargs='?', const=True,
                        help='if toggled, `torch.backends.cudnn.deterministic=False`')
    parser.add_argument('--cuda', type=lambda x:bool(strtobool(x)), default=True, nargs='?', const=True,
                        help='if toggled, cuda will not be enabled by default')
    parser.add_argument('--prod-mode', type=lambda x:bool(strtobool(x)), default=False, nargs='?', const=True,
                        help='run the script in production mode and use wandb to log outputs')
    parser.add_argument('--capture-video', type=lambda x:bool(strtobool(x)), default=False, nargs='?', const=True,
                        help='weather to capture videos of the agent performances (check out `videos` folder)')
    parser.add_argument('--wandb-project-name', type=str, default="cleanRL",
                        help="the wandb's project name")
    parser.add_argument('--wandb-entity', type=str, default=None,
                        help="the entity (team) of wandb's project")
    
    # Algorithm specific arguments
    parser.add_argument('--num-actors', type=int, default=4,
                         help='the replay memory buffer size')
    parser.add_argument('--num-data-processors', type=int, default=2,
                         help='the replay memory buffer size')
    parser.add_argument('--actor-buffer-size', type=int, default=50,
                         help='the replay memory buffer size')
    parser.add_argument('--buffer-size', type=int, default=100000,
                         help='the replay memory buffer size')
    parser.add_argument('--pr-alpha', type=float, default=0.6,
                        help='alpha parameter for prioritized replay buffer')
    parser.add_argument('--pr-beta0', type=float, default=0.4,
                        help='initial value of beta for prioritized replay buffer')
    parser.add_argument('--pr-eps', type=float, default=1e-6,
                        help='epsilon to add to the TD errors when updating priorities.')
    parser.add_argument('--gamma', type=float, default=0.99,
                        help='the discount factor gamma')
    parser.add_argument('--target-network-frequency', type=int, default=1000,
                        help="the timesteps it takes to update the target network")
    parser.add_argument('--max-grad-norm', type=float, default=0.5,
                        help='the maximum norm for the gradient clipping')
    parser.add_argument('--batch-size', type=int, default=256,
                        help="the batch size of sample from the reply memory")
    parser.add_argument('--start-e', type=float, default=1.,
                        help="the starting epsilon for exploration")
    parser.add_argument('--end-e', type=float, default=0.02,
                        help="the ending epsilon for exploration")
    parser.add_argument('--exploration-fraction', type=float, default=0.10,
                        help="the fraction of `total-timesteps` it takes from start-e to go end-e")
    parser.add_argument('--learning-starts', type=int, default=80000,
                        help="timestep to start learning")
    parser.add_argument('--train-frequency', type=int, default=4,
                        help="the frequency of training")
    args = parser.parse_args()
    if not args.seed:
        args.seed = int(time.time())

    # TRY NOT TO MODIFY: setup the environment
    experiment_name = f"{args.gym_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    writer = SummaryWriter(f"runs/{experiment_name}")
    writer.add_text('hyperparameters', "|param|value|\n|-|-|\n%s" % (
            '\n'.join([f"|{key}|{value}|" for key, value in vars(args).items()])))
    if args.prod_mode:
        import wandb
        wandb.init(project=args.wandb_project_name, entity=args.wandb_entity, sync_tensorboard=True, config=vars(args), name=experiment_name, monitor_gym=True, save_code=True)
        writer = SummaryWriter(f"/tmp/{experiment_name}")
    
    # TRY NOT TO MODIFY: seeding
    device = torch.device('cuda' if torch.cuda.is_available() and args.cuda else 'cpu')
    env = gym.make(args.gym_id)
    env = wrap_atari(env)
    env = gym.wrappers.RecordEpisodeStatistics(env) # records episode reward in `info['episode']['r']`
    env = wrap_deepmind(
        env,
        clip_rewards=True,
        frame_stack=True,
        scale=False,
    )
    env.seed(args.seed)
    env.action_space.seed(args.seed)
    env.observation_space.seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic
    assert isinstance(env.action_space, Discrete), "only discrete action space is supported"

    m = SyncManager()
    m.start()
    lock = m.Lock()
    rb = CustomPrioritizedReplayBuffer(args.buffer_size, args.pr_alpha)

    q_network = QNetwork(env)
    target_network = QNetwork(env)
    target_network.load_state_dict(q_network.state_dict())
    learn_q_network = QNetwork(env, device).to(device)
    learn_q_network.load_state_dict(q_network.state_dict())
    learn_target_network = QNetwork(env, device).to(device)
    learn_target_network.load_state_dict(q_network.state_dict())
    optimizer = optim.Adam(learn_q_network.parameters(), lr=args.learning_rate)

    print(device.__repr__())
    print(q_network)

    global_step = torch.tensor(0)
    global_step.share_memory_()
    actor_processes = []
    data_processor_processes = []
    ctx = mp.get_context("forkserver")
    stats_queue = ctx.Queue(10)
    rollouts_queue = ctx.Queue(10)
    data_process_queue = ctx.Queue(10)
    data_process_back_queues = []
    
    for i in range(args.num_actors):
        actor = ctx.Process(
            target=act,
            args=(
                args,
                experiment_name,
                i,
                q_network,
                target_network,
                lock,
                rollouts_queue,
                stats_queue,
                global_step,
                "cpu"
            ),
        )
        actor.start()
        actor_processes.append(actor)
        
    for i in range(args.num_data_processors):
        data_process_back_queues += [ctx.Queue(1)]
        data_processor = ctx.Process(
            target=data_process,
            args=(
                args,
                i,
                global_step,
                rollouts_queue,
                data_process_queue,
                data_process_back_queues,
                device
            ),
        )
        data_processor.start()
        data_processor_processes.append(data_processor)


    learner = ctx.Process(
        target=learn,
        args=(
            args, rb, global_step,
            data_process_queue,
            data_process_back_queues, stats_queue, lock, learn_target_network, target_network, learn_q_network, q_network, optimizer, device
        ),
    )
    learner.start()

    import timeit
    timer = timeit.default_timer
    existing_video_files = []
    try:
        while global_step < args.total_timesteps:
            start_global_step = global_step.item()
            start_time = timer()
            m = stats_queue.get()
            if m[0] == "charts/episode_reward":
                r, l = m[1], m[2]
                print(f"global_step={global_step}, episode_reward={r}")
                writer.add_scalar("charts/episode_reward", r, global_step)
                writer.add_scalar("charts/stats_queue_size", stats_queue.qsize(), global_step)
                writer.add_scalar("charts/rollouts_queue_size", rollouts_queue.qsize(), global_step)
                writer.add_scalar("charts/data_process_queue_size", data_process_queue.qsize(), global_step)
                writer.add_scalar("charts/fps", (global_step.item() - start_global_step) / (timer() - start_time), global_step)
                print("FPS: ", (global_step.item() - start_global_step) / (timer() - start_time))
            else:
                # print(m[0], m[1], global_step)
                writer.add_scalar(m[0], m[1], global_step)
            if args.capture_video and args.prod_mode:
                video_files = glob.glob(f'videos/{experiment_name}/*.mp4')
                for video_file in video_files:
                    if video_file not in existing_video_files:
                        existing_video_files += [video_file]
                        print(video_file)
                        if len(existing_video_files) > 1:
                            wandb.log({"video.0": wandb.Video(existing_video_files[-2])})
    except KeyboardInterrupt:
        pass
    finally:
        learner.terminate()
        learner.join(timeout=1)
        for actor in actor_processes:
            actor.terminate()
            actor.join(timeout=1)
        for data_processor in data_processor_processes:
            data_processor.terminate()
            data_processor.join(timeout=1)
        if args.capture_video and args.prod_mode:
            wandb.log({"video.0": wandb.Video(existing_video_files[-1])})
    # env.close()
    writer.close()
