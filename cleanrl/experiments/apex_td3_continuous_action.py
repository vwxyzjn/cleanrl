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

import pybullet_envs

# ALGO LOGIC: initialize agent here:
class QNetwork(nn.Module):
    def __init__(self, device="cpu"):
        super(QNetwork, self).__init__()
        self.device = device
        self.fc1 = nn.Linear(
            np.array(env.observation_space.shape).prod()+np.prod(env.action_space.shape), 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)

    def forward(self, x, a):
        x = torch.Tensor(x).to(self.device)
        return self.process(x, a)

    def process(self, x, a):
        x = torch.cat([x, a], 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

class Actor(nn.Module):
    def __init__(self, device="cpu"):
        super(Actor, self).__init__()
        self.device = device
        self.fc1 = nn.Linear(np.array(env.observation_space.shape).prod(), 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc_mu = nn.Linear(256, np.prod(env.action_space.shape))

    def forward(self, x):
        x = torch.Tensor(x).to(self.device)
        return self.process(x)

    def process(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return torch.tanh(self.fc_mu(x))

def linear_schedule(start_e: float, end_e: float, duration: int, t: int):
    slope =  (end_e - start_e) / duration
    if end_e < start_e:
        return max(slope * t + start_e, end_e)
    else:
        return min(slope * t + start_e, end_e)

def act(args, experiment_name, i, worker_models, lock, rollouts_queue, stats_queue, global_step, device):
    actor, qf1, qf2, qf1_target, qf2_target, target_actor = worker_models
    env = gym.make(args.gym_id)
    env = gym.wrappers.RecordEpisodeStatistics(env) # records episode reward in `info['episode']['r']`
    if args.capture_video:
        if i == 0:
            env = Monitor(env, f'videos/{experiment_name}')
    env.seed(args.seed+i)
    env.action_space.seed(args.seed+i)
    # TRY NOT TO MODIFY: start the game
    obs = env.reset()
    max_action = float(env.action_space.high[0])
    storage = []
    episode_reward = 0
    update_step = 0
    while True:
        update_step += 1
        # ALGO LOGIC: put action logic here
        if global_step < args.learning_starts // args.num_actors:
            action = env.action_space.sample()
        else:
            action = actor.forward(obs.reshape((1,)+obs.shape))
            action = (
                action.tolist()[0]
                + np.random.normal(0, max_action * args.exploration_noise, size=env.action_space.shape[0])
            ).clip(env.action_space.low, env.action_space.high)
    
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
                clipped_noise = (
                    torch.randn_like(torch.Tensor(action)) * args.policy_noise
                ).clamp(-args.noise_clip, args.noise_clip)

                next_state_actions = (
                    target_actor.forward(s_next_obses) + clipped_noise.to(device)
                ).clamp(env.action_space.low[0], env.action_space.high[0])
                qf1_next_target = qf1_target.forward(s_next_obses, next_state_actions)
                qf2_next_target = qf2_target.forward(s_next_obses, next_state_actions)
                min_qf_next_target = torch.min(qf1_next_target, qf2_next_target)
                next_q_value = torch.Tensor(s_rewards).to(device) + (1 - torch.Tensor(s_dones).to(device)) * args.gamma * (min_qf_next_target).view(-1)

                qf1_a_values = qf1.forward(s_obs, torch.Tensor(s_actions).to(device)).view(-1)
                td_errors = qf1_a_values - next_q_value

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
            
            s_obs, s_actions, s_rewards, s_next_obses, s_dones = torch.Tensor(s_obs).to(device), torch.Tensor(s_actions).to(device), torch.Tensor(s_rewards).to(device), torch.Tensor(s_next_obses).to(device), torch.Tensor(s_dones).to(device)
            data_process_queue.put([i, s_obs, s_actions, s_rewards, s_next_obses, s_dones])
            new_priorities = data_process_back_queues[i].get()
            rb.update_priorities(s_batch_idxes, new_priorities)
            del new_priorities

def learn(args, rb, global_step, env, data_process_queue, data_process_back_queues, stats_queue, lock, 
    worker_models,
    learner_models,
    q_optimizer, actor_optimizer,
    device):
    actor, qf1, qf2, qf1_target, qf2_target, target_actor = learner_models
    action = env.action_space.sample()
    loss_fn = nn.MSELoss()
    update_step = 0
    while True:
        update_step += 1
        experience = data_process_queue.get()
        (i, s_obs, s_actions, s_rewards, s_next_obses, s_dones) = experience

        with torch.no_grad():
            clipped_noise = (
				torch.randn_like(torch.Tensor(action)) * args.policy_noise
			).clamp(-args.noise_clip, args.noise_clip)
            
            next_state_actions = (
                target_actor.process(s_next_obses) + clipped_noise.to(device)
            ).clamp(env.action_space.low[0], env.action_space.high[0])
            qf1_next_target = qf1_target.process(s_next_obses, next_state_actions)
            qf2_next_target = qf2_target.process(s_next_obses, next_state_actions)
            min_qf_next_target = torch.min(qf1_next_target, qf2_next_target)
            next_q_value = s_rewards + (1 - s_dones) * args.gamma * (min_qf_next_target).view(-1)

        qf1_a_values = qf1.process(s_obs, s_actions).view(-1)
        qf2_a_values = qf2.process(s_obs, s_actions).view(-1)
        qf1_loss = loss_fn(qf1_a_values, next_q_value)
        qf2_loss = loss_fn(qf2_a_values, next_q_value)
        td_errors = qf1_a_values - next_q_value

        # optimize the midel
        q_optimizer.zero_grad()
        qf1_loss.backward()
        qf2_loss.backward()
        nn.utils.clip_grad_norm_(list(qf1.parameters())+list(qf2.parameters()), args.max_grad_norm)
        q_optimizer.step()

        if global_step % args.policy_frequency == 0:
            actor_loss = -qf1.process(s_obs, actor.process(s_obs)).mean()
            actor_optimizer.zero_grad()
            actor_loss.backward()
            nn.utils.clip_grad_norm_(list(actor.parameters()), args.max_grad_norm)
            actor_optimizer.step()

            # update the target network
            for param, target_param in zip(actor.parameters(), target_actor.parameters()):
                target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)
            for param, target_param in zip(qf1.parameters(), qf1_target.parameters()):
                target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)
            for param, target_param in zip(qf2.parameters(), qf2_target.parameters()):
                target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)

        # update the weights in the prioritized replay
        new_priorities = np.abs(td_errors.tolist()) + args.pr_eps
        data_process_back_queues[i].put(new_priorities)
        
        stats_queue.put(("losses/qf1_loss", qf1_loss.item(), update_step+args.learning_starts))
    
        del i, s_obs, s_actions, s_rewards, s_next_obses, s_dones
    
        for worker_model, learner_model in zip(worker_models, learner_models):
            worker_model.load_state_dict(learner_model.state_dict())

        # update the target network
        # if update_step % args.target_network_frequency == 0:
        #     learn_target_network.load_state_dict(learn_q_network.state_dict())
        #     target_network.load_state_dict(learn_q_network.state_dict())

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='DQN agent')
    # Common arguments
    parser.add_argument('--exp-name', type=str, default=os.path.basename(__file__).rstrip(".py"),
                        help='the name of this experiment')
    parser.add_argument('--gym-id', type=str, default="HopperBulletEnv-v0",
                        help='the id of the gym environment')
    parser.add_argument('--learning-rate', type=float, default=3e-4,
                        help='the learning rate of the optimizer')
    parser.add_argument('--seed', type=int, default=2,
                        help='seed of the experiment')
    parser.add_argument('--total-timesteps', type=int, default=5000000,
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
    parser.add_argument('--tau', type=float, default=0.005,
                        help="target smoothing coefficient (default: 0.005)")
    parser.add_argument('--max-grad-norm', type=float, default=0.5,
                        help='the maximum norm for the gradient clipping')
    parser.add_argument('--batch-size', type=int, default=256,
                        help="the batch size of sample from the reply memory")
    parser.add_argument('--policy-noise', type=float, default=0.2,
                        help='the scale of policy noise')
    parser.add_argument('--exploration-noise', type=float, default=0.1,
                        help='the scale of exploration noise')
    parser.add_argument('--learning-starts', type=int, default=25e3,
                        help="timestep to start learning")
    parser.add_argument('--policy-frequency', type=int, default=2,
                        help="the frequency of training policy (delayed)")
    parser.add_argument('--noise-clip', type=float, default=0.5,
                         help='noise clip parameter of the Target Policy Smoothing Regularization')
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
    env = gym.wrappers.RecordEpisodeStatistics(env) # records episode reward in `info['episode']['r']`
    env.seed(args.seed)
    env.action_space.seed(args.seed)
    env.observation_space.seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic
    assert isinstance(env.action_space, Box), "only discrete action space is supported"

    m = SyncManager()
    m.start()
    lock = m.Lock()
    rb = CustomPrioritizedReplayBuffer(args.buffer_size, args.pr_alpha)

    rb = ReplayBuffer(args.buffer_size)
    actor = Actor()
    qf1 = QNetwork()
    qf2 = QNetwork()
    qf1_target = QNetwork()
    qf2_target = QNetwork()
    target_actor = Actor()
    target_actor.load_state_dict(actor.state_dict())
    qf1_target.load_state_dict(qf1.state_dict())
    qf2_target.load_state_dict(qf2.state_dict())
    worker_models = [actor, qf1, qf2, qf1_target, qf2_target, target_actor]

    learn_actor = Actor(device).to(device)
    learn_qf1 = QNetwork(device).to(device)
    learn_qf2 = QNetwork(device).to(device)
    learn_qf1_target = QNetwork(device).to(device)
    learn_qf2_target = QNetwork(device).to(device)
    learn_target_actor = Actor(device).to(device)
    learn_actor.load_state_dict(actor.state_dict())
    learn_qf1.load_state_dict(qf1.state_dict())
    learn_qf2.load_state_dict(qf2.state_dict())
    learn_target_actor.load_state_dict(actor.state_dict())
    learn_qf1_target.load_state_dict(qf1.state_dict())
    learn_qf2_target.load_state_dict(qf2.state_dict())
    q_optimizer = optim.Adam(list(learn_qf1.parameters()) + list(learn_qf2.parameters()), lr=args.learning_rate)
    actor_optimizer = optim.Adam(list(learn_actor.parameters()), lr=args.learning_rate)
    loss_fn = nn.MSELoss()
    learner_models = [learn_actor, learn_qf1, learn_qf2, learn_qf1_target, learn_qf2_target, learn_target_actor]

    global_step = torch.tensor(0)
    global_step.share_memory_()
    worker_processes = []
    data_processor_processes = []
    ctx = mp.get_context("forkserver")
    stats_queue = ctx.Queue(10)
    rollouts_queue = ctx.Queue(10)
    data_process_queue = ctx.Queue(10)
    data_process_back_queues = []
    

    for i in range(args.num_actors):
        worker = ctx.Process(
            target=act,
            args=(
                args,
                experiment_name,
                i,
                worker_models,
                lock,
                rollouts_queue,
                stats_queue,
                global_step,
                "cpu"
            ),
        )
        worker.start()
        worker_processes.append(worker)
        
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
            args, rb, global_step, env,
            data_process_queue,
            data_process_back_queues, stats_queue, lock, 
            worker_models,
            learner_models,
            q_optimizer, actor_optimizer,
            device
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
        for worker in worker_processes:
            worker.terminate()
            worker.join(timeout=1)
        for data_processor in data_processor_processes:
            data_processor.terminate()
            data_processor.join(timeout=1)
        if args.capture_video and args.prod_mode:
            wandb.log({"video.0": wandb.Video(existing_video_files[-1])})
    # env.close()
    writer.close()
