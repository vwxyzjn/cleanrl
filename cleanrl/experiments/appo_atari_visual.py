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
        return np.concatenate(self.frames)

class ScaledFloatFrame(gym.ObservationWrapper):
    def __init__(self, env):
        gym.ObservationWrapper.__init__(self, env)
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=env.observation_space.shape, dtype=np.float32)

    def observation(self, observation):
        # careful! This undoes the memory optimization, use
        # with smaller replay buffers only.
        return np.array(observation).astype(np.float32) / 255.0

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

# Reference: https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
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

# import threading
os.environ["OMP_NUM_THREADS"] = "1"  # Necessary for multithreading.
from torch import multiprocessing as mp
# mp.set_start_method('forkserver')
from multiprocessing.managers import SyncManager, SharedMemoryManager
from multiprocessing.shared_memory import SharedMemory
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecEnvWrapper
from faster_fifo import Queue as MpQueue
from queue import Empty
import stopwatch

class SharedNDArray(np.ndarray):
    def set_shm(self, shm):
        self.shm = shm

    def close(self):
        self.shm.close()

    def unlink(self):
        self.shm.unlink()


def share_memory(arr):
    shm = SharedMemory(create=True, size=arr.nbytes)
    shm_arr = SharedNDArray(arr.shape, dtype=arr.dtype, buffer=shm.buf)
    shm_arr[:] = arr[:]
    shm_arr.set_shm(shm)
    return shm_arr

class Scale(nn.Module):
    def __init__(self, scale):
        super().__init__()
        self.scale = scale

    def forward(self, x):
        return x * self.scale

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class Agent(nn.Module):
    def __init__(self, n_actions, frames=4):
        super(Agent, self).__init__()
        self.network = nn.Sequential(
            Scale(1/255),
            layer_init(nn.Conv2d(frames, 32, 8, stride=4)),
            nn.ReLU(),
            layer_init(nn.Conv2d(32, 64, 4, stride=2)),
            nn.ReLU(),
            layer_init(nn.Conv2d(64, 64, 3, stride=1)),
            nn.ReLU(),
            nn.Flatten(),
            layer_init(nn.Linear(3136, 512)),
            nn.ReLU()
        )
        self.actor = layer_init(nn.Linear(512, n_actions), std=0.01)
        self.critic = layer_init(nn.Linear(512, 1), std=1)

    def forward(self, x):
        return self.network(x)

    def get_action(self, x, action=None):
        logits = self.actor(self.forward(x))
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy()

    def get_value(self, x):
        return self.critic(self.forward(x))

            
def act(inputs):
    sw = stopwatch.StopWatch()
    i, args, experiment_name, lock, stats_queue, device, \
        next_obs, next_done, obs, actions, logprobs, rewards, dones, values, traj_availables,\
        rollout_task_queues, policy_request_queues, learner_request_queue, new_policy_queues = inputs
    envs = []
    
    def make_env(gym_id, seed, idx):
        env = gym.make(gym_id)
        env = wrap_atari(env)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = wrap_deepmind(
            env,
            clip_rewards=True,
            frame_stack=True,
            scale=False,
        )
        env.seed(seed)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        return env
    envs = [make_env(args.gym_id, args.seed+i, i) for i in range(args.num_envs)]
    envs = np.array(envs, dtype=object)

    # for "Double-buffered" sampling
    policy_request_queue_idx = 0
    for split_idx in range(args.num_env_split):
        policy_request_idxs = []
        for env_idx, env in enumerate(envs[split_idx::args.num_env_split]):
            next_obs[i,split_idx,env_idx,0,0] = env.reset()
            next_done[i,split_idx,env_idx,0,0] = 0
            policy_request_idxs += [[i,split_idx,env_idx,0,0,0]]
        policy_request_queue_idx = (policy_request_queue_idx + 1) % args.num_policy_workers
        policy_request_queues[policy_request_queue_idx].put(policy_request_idxs)

    last_report = last_report_frames = total_env_frames = 0
    
    while True:
        with sw.timer('act'):
            with sw.timer('wait_rollout_task_queue'):
                tasks = []
                while len(tasks) == 0:
                    try:
                        tasks = rollout_task_queues[i].get_many(timeout=0.01)
                    except Empty:
                        pass
            for task in tasks:
                # for "Double-buffered" sampling
                with sw.timer('rollouts'):
                    split_idx, step = task
                    policy_request_idxs = []
                    for env_idx, env in enumerate(envs[split_idx::args.num_env_split]):
                        obs[i,split_idx,env_idx,0,0,step] = next_obs[i,split_idx,env_idx,0,0].copy()
                        dones[i,split_idx,env_idx,0,0,step] = next_done[i,split_idx,env_idx,0,0]
                        next_obs[i,split_idx,env_idx,0,0], r, d, info = env.step(actions[i,split_idx,env_idx,0,0,step])
                        if d:
                            next_obs[i,split_idx,env_idx,0,0] = env.reset()
                        rewards[i,split_idx,env_idx,0,0,step] = r
                        next_done[env_idx] = d
                        
                        next_step = (step + 1) % args.num_steps
                        policy_request_idxs += [[i,split_idx,env_idx,0,0,next_step]]
        
                        num_frames = 1
                        total_env_frames += num_frames
            
                        if 'episode' in info.keys():
                            stats_queue.put(['l', info['episode']['l']])
                            stats_queue.put(["charts/episode_reward", info['episode']['r']])
                with sw.timer('policy_request_queue.put'):
                    if next_step == 0:
                        learner_request_queue.put(policy_request_idxs)
                    else:
                        policy_request_queue_idx = (policy_request_queue_idx + 1) % args.num_policy_workers
                        policy_request_queues[policy_request_queue_idx].put(policy_request_idxs)
        if total_env_frames % 1000 == 0 and i == 0:
            print(stopwatch.format_report(sw.get_last_aggregated_report()))


def start_policy_worker(inputs):
    # raise
    i, args, experiment_name, lock, stats_queue, device, \
        next_obs, next_done, obs, actions, logprobs, rewards, dones, values, traj_availables,\
        rollout_task_queues, policy_request_queues, learner_request_queue, new_policy_queues = inputs
    sw = stopwatch.StopWatch()
    device = torch.device('cuda')
    agent = Agent(4).to(device)
    min_num_requests = 3
    wait_for_min_requests = 0.01
    # time.sleep(5)
    step = 0
    while True:
        step += 1
        with sw.timer('policy_worker'):
            waiting_started = time.time()
            with sw.timer('policy_requests'):
                policy_requests = []
                while len(policy_requests) < min_num_requests and time.time() - waiting_started < wait_for_min_requests:
                    try:
                        policy_requests.extend(policy_request_queues[i].get_many(timeout=0.005))
                    except Empty:
                        pass
                if len(policy_requests) == 0:
                    continue
            with sw.timer('prepare_data'):
                ls = np.concatenate(policy_requests)
                rollout_worker_idxs = ls.T[0,::args.num_envs//args.num_env_split]
                split_idxs = ls.T[1,::args.num_envs//args.num_env_split]
                step_idxs = ls.T[-1,::args.num_envs//args.num_env_split]
                idxs = tuple(ls.T)
            with sw.timer('index'):
                t1 = next_obs[idxs[:-1]]
            with sw.timer('create array at gpu'):
                next_o = torch.tensor(t1, device=device)
            with sw.timer('inference'):
                with torch.no_grad():
                    a, l, _ = agent.get_action(next_o)
            with sw.timer('move_to_cpu'):
                actions[idxs] = a.cpu()
                logprobs[idxs] = l.cpu()
            with sw.timer('execute_action'):
                for j in range(len(rollout_worker_idxs)):
                    rollout_worker_idx = rollout_worker_idxs[j]
                    split_idx = split_idxs[j]
                    step_idx = step_idxs[j]
                    rollout_task_queues[rollout_worker_idx].put([split_idx,step_idx])
            with sw.timer('update_policy'):
                if step % 10 == 0:
                    try:
                        new_policies = new_policy_queues[i].get_many(timeout=0.005)
                        agent.load_state_dict(new_policies[-1])
                    except Empty:
                        pass
        if step % 100 == 0:
            print(ls.shape)
            print(stopwatch.format_report(sw.get_last_aggregated_report()))
        # raise
        
def learn(inputs):
    # raise
    i, args, experiment_name, lock, stats_queue, device, \
        next_obs, next_done, obs, actions, logprobs, rewards, dones, values, traj_availables,\
        rollout_task_queues, policy_request_queues, learner_request_queue, new_policy_queues = inputs
    s_next_obs, s_next_done, s_obs, s_actions, s_logprobs, s_rewards, s_dones, s_values, s_traj_availables = next_obs, next_done, obs, actions, logprobs, rewards, dones, values, traj_availables
    sw = stopwatch.StopWatch()
    device = torch.device('cuda')
    agent = Agent(4).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)
    min_num_requests = 3
    wait_for_min_requests = 0.01
    # time.sleep(5)
    step = 0
    while True:
        with sw.timer('learner'):
            waiting_started = time.time()
            learner_requests = []
            with sw.timer('learner_requests'):
                while len(learner_requests) < min_num_requests and time.time() - waiting_started < wait_for_min_requests:
                    try:
                        learner_requests.extend(learner_request_queue.get_many(timeout=0.005))
                    except Empty:
                        pass
            if len(learner_requests) == 0:
                continue
        with sw.timer('prepare_data'):
            ls = np.concatenate(learner_requests)
            rollout_worker_idxs = ls.T[0,::args.num_envs//args.num_env_split]
            split_idxs = ls.T[1,::args.num_envs//args.num_env_split]
            step_idxs = ls.T[-1,::args.num_envs//args.num_env_split]
            idxs = tuple(ls.T)

        next_done = torch.tensor(s_next_done[idxs[:-1]], device=device)
        next_obs = torch.tensor(s_next_obs[idxs[:-1]], device=device)
        rewards = torch.tensor(s_rewards[idxs[:-1]], device=device).transpose(0, 1)
        obs = torch.tensor(s_obs[idxs[:-1]], device=device).transpose(0, 1)
        logprobs = torch.tensor(s_logprobs[idxs[:-1]], device=device).transpose(0, 1)
        actions = torch.tensor(s_actions[idxs[:-1]], device=device).transpose(0, 1)
        values = torch.tensor(s_values[idxs[:-1]], device=device).transpose(0, 1)
        dones = torch.tensor(s_dones[idxs[:-1]], device=device).transpose(0, 1)

        # bootstrap reward if not done. reached the batch limit
        with torch.no_grad():
            last_value = agent.get_value(next_obs).reshape(1, -1)
            if args.gae:
                advantages = torch.zeros_like(rewards).to(device)
                lastgaelam = 0
                for t in reversed(range(args.num_steps)):
                    if t == args.num_steps - 1:
                        nextnonterminal = 1.0 - next_done
                        nextvalues = last_value
                    else:
                        nextnonterminal = 1.0 - dones[t+1]
                        nextvalues = values[t+1]
                    delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                    advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
                returns = advantages + values
            else:
                returns = torch.zeros_like(rewards).to(device)
                for t in reversed(range(args.num_steps)):
                    if t == args.num_steps - 1:
                        nextnonterminal = 1.0 - next_done
                        next_return = last_value
                    else:
                        nextnonterminal = 1.0 - dones[t+1]
                        next_return = returns[t+1]
                    returns[t] = rewards[t] + args.gamma * nextnonterminal * next_return
                advantages = returns - values
    
        # flatten the batch
        b_obs = obs.reshape((-1,)+(4,84,84))
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape(-1)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)
        # Optimizaing the policy and value network
        # target_agent = Agent(4).to(device)
        inds = np.arange(len(b_obs))
        minibatch_ind = inds
        # for i_epoch_pi in range(args.update_epochs):
        #     np.random.shuffle(inds)
        #     target_agent.load_state_dict(agent.state_dict())
        #     for start in range(0, args.batch_size, args.minibatch_size):
        #         end = start + args.minibatch_size
        #         minibatch_ind = inds[start:end]
        mb_advantages = b_advantages[minibatch_ind]

        if args.norm_adv:
            mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)
        

        _, newlogproba, entropy = agent.get_action(b_obs[minibatch_ind], b_actions[minibatch_ind])
        ratio = (newlogproba - b_logprobs[minibatch_ind]).exp()

        # Stats
        approx_kl = (b_logprobs[minibatch_ind] - newlogproba).mean()

        # Policy loss
        pg_loss1 = -mb_advantages * ratio
        pg_loss2 = -mb_advantages * torch.clamp(ratio, 1-args.clip_coef, 1+args.clip_coef)
        pg_loss = torch.max(pg_loss1, pg_loss2).mean()
        entropy_loss = entropy.mean()

        # Value loss
        new_values = agent.get_value(b_obs[minibatch_ind]).view(-1)
        if args.clip_vloss:
            v_loss_unclipped = ((new_values - b_returns[minibatch_ind]) ** 2)
            v_clipped = b_values[minibatch_ind] + torch.clamp(new_values - b_values[minibatch_ind], -args.clip_coef, args.clip_coef)
            v_loss_clipped = (v_clipped - b_returns[minibatch_ind])**2
            v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
            v_loss = 0.5 * v_loss_max.mean()
        else:
            v_loss = 0.5 * ((new_values - b_returns[minibatch_ind]) ** 2).mean()

        loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
        optimizer.step()

        # raise
        del next_done, next_obs, rewards, obs, logprobs, actions, values, dones
        for learner_request in learner_requests:
            policy_request_queues[0].put(learner_request)
        
        # update the policy in the policy worker
        for new_policy_queue in new_policy_queues:
            new_policy_queue.put(agent.state_dict())

        stats_queue.put(["losses/value_loss", v_loss.item()])
        stats_queue.put(["losses/policy_loss", pg_loss.item()])
        stats_queue.put(["losses/entropy", entropy_loss.item()])
        stats_queue.put(["losses/approx_kl", approx_kl.item()])

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
    parser.add_argument('--total-timesteps', type=int, default=100000000,
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
    parser.add_argument('--num-rollout-workers', type=int, default=4,
                         help='the number of rollout workers')
    parser.add_argument('--num-env-split', type=int, default=2,
                         help='the number of rollout workers')
    parser.add_argument('--num-policy-workers', type=int, default=1,
                         help='the number of policy workers')
    parser.add_argument('--num-envs', type=int, default=8,
                         help='the number of envs per rollout worker')
    parser.add_argument('--num-traj-buffers', type=int, default=1,
                         help='the number of trajectory buffers per rollout worker')
    parser.add_argument('--num-steps', type=int, default=32,
                        help='the number of steps per game environment')
    parser.add_argument('--n-minibatch', type=int, default=4,
                        help='the number of mini batch')
    parser.add_argument('--gamma', type=float, default=0.99,
                        help='the discount factor gamma')
    parser.add_argument('--gae-lambda', type=float, default=0.95,
                        help='the lambda for the general advantage estimation')
    parser.add_argument('--ent-coef', type=float, default=0.01,
                        help="coefficient of the entropy")
    parser.add_argument('--vf-coef', type=float, default=0.5,
                        help="coefficient of the value function")
    parser.add_argument('--max-grad-norm', type=float, default=0.5,
                        help='the maximum norm for the gradient clipping')
    parser.add_argument('--clip-coef', type=float, default=0.1,
                        help="the surrogate clipping coefficient")
    parser.add_argument('--update-epochs', type=int, default=4,
                         help="the K epochs to update the policy")
    parser.add_argument('--kle-stop', type=lambda x:bool(strtobool(x)), default=False, nargs='?', const=True,
                         help='If toggled, the policy updates will be early stopped w.r.t target-kl')
    parser.add_argument('--kle-rollback', type=lambda x:bool(strtobool(x)), default=False, nargs='?', const=True,
                         help='If toggled, the policy updates will roll back to previous policy if KL exceeds target-kl')
    parser.add_argument('--target-kl', type=float, default=0.03,
                         help='the target-kl variable that is referred by --kl')
    parser.add_argument('--gae', type=lambda x:bool(strtobool(x)), default=True, nargs='?', const=True,
                         help='Use GAE for advantage computation')
    parser.add_argument('--norm-adv', type=lambda x:bool(strtobool(x)), default=True, nargs='?', const=True,
                          help="Toggles advantages normalization")
    parser.add_argument('--anneal-lr', type=lambda x:bool(strtobool(x)), default=True, nargs='?', const=True,
                          help="Toggle learning rate annealing for policy and value networks")
    parser.add_argument('--clip-vloss', type=lambda x:bool(strtobool(x)), default=True, nargs='?', const=True,
                          help='Toggles wheter or not to use a clipped loss for the value function, as per the paper.')

    args = parser.parse_args()
    if not args.seed:
        args.seed = int(time.time())
        
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.n_minibatch)

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
    # device = torch.device('cuda' if torch.cuda.is_available() and args.cuda else 'cpu')
    
    def make_env(gym_id, seed, idx):
        env = gym.make(gym_id)
        env = wrap_atari(env)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = wrap_deepmind(
            env,
            clip_rewards=True,
            frame_stack=True,
            scale=False,
        )
        env.seed(seed)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        return env
    
    env = make_env(args.gym_id, args.seed, 0)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic
    assert isinstance(env.action_space, Discrete), "only discrete action space is supported"

    # m = SyncManager()
    # m.start()
    lock = mp.Lock()
    dimensions = (
        args.num_rollout_workers,
        args.num_env_split,
        args.num_envs // args.num_env_split,
        1, # args.num_policy_workers,
        args.num_traj_buffers,
        args.num_steps,
    )
    
    next_obs = share_memory(np.zeros(dimensions[:-1]+env.observation_space.shape, dtype=np.float32))
    next_done = share_memory(np.zeros(dimensions[:-1], dtype=np.float32))
    obs = share_memory(np.zeros(dimensions+env.observation_space.shape, dtype=np.float32))
    actions = share_memory(np.zeros(dimensions+env.action_space.shape, dtype=env.action_space.dtype))
    logprobs = share_memory(np.zeros(dimensions, dtype=np.float32))
    rewards = share_memory(np.zeros(dimensions, dtype=np.float32))
    dones = share_memory(np.zeros(dimensions, dtype=np.float32))
    values = share_memory(np.zeros(dimensions, dtype=np.float32))
    traj_availables = share_memory(np.ones(dimensions, dtype=np.float32))
    actor_processes = []
    policy_workers = []
    stats_queue = MpQueue()
    rollout_task_queues = [MpQueue() for i in range(args.num_rollout_workers)]
    policy_request_queues = [MpQueue() for i in range(args.num_policy_workers)]
    learner_request_queue = MpQueue()
    new_policy_queues = [MpQueue() for i in range(args.num_policy_workers)]
    
    global_args = [
        args, experiment_name, lock, stats_queue, 0,
        next_obs, next_done, obs, actions, logprobs, rewards, dones, values, traj_availables,
        rollout_task_queues, policy_request_queues, learner_request_queue, new_policy_queues
    ]

    for i in range(args.num_rollout_workers):
        actor = mp.Process(
            target=act,
            args=[[i] + global_args],
        )
        actor.start()
        actor_processes.append(actor)


    for i in range(args.num_policy_workers):
        policy_worker = mp.Process(
            target=start_policy_worker,
            args=[[i] + global_args],
        )
        policy_worker.start()
        policy_workers.append(policy_worker)
    learner = mp.Process(
        target=learn,
        args=[[i] + global_args]
    )
    learner.start()

    import timeit
    timer = timeit.default_timer
    existing_video_files = []
    global_step = 0
    global_step_increment = 0
    start_time = time.time()
    update_step = 0
    try:
        while global_step < args.total_timesteps:
            try:
                update_step += 1
                ms = stats_queue.get_many(timeout=1)
                for m in ms:
                    if m[0] == 'l':
                        global_step_increment += m[1]
                        global_step += m[1]
                    else:
                        writer.add_scalar(m[0], m[1], global_step)
                        if m[0] == "charts/episode_reward":
                            print(f"global_step={global_step}, episode_reward={m[1]}")


                if update_step % 10 == 0:
                    # print(f"global_step={global_step}, episode_reward={r}")
                    print(f"global_step={global_step}")
                    # global_step += global_step_increment
                    writer.add_scalar("charts/fps", global_step_increment / (time.time() - start_time), global_step)
                    writer.add_scalar("charts/policy_request_queue", policy_request_queues[0].qsize(), global_step)
                    writer.add_scalar("charts/rollout_task_queues[0]", rollout_task_queues[0].qsize(), global_step)
                    print("FPS: ", global_step_increment / (time.time() - start_time))
                    global_step_increment = 0
                    start_time = time.time()
            except:
                continue

            # writer.add_scalar("charts/episode_reward", r, global_step)
            # writer.add_scalar("charts/stats_queue_size", stats_queue.qsize(), global_step)
            # writer.add_scalar("charts/rollouts_queue_size", rollouts_queue.qsize(), global_step)
            # writer.add_scalar("charts/data_process_queue_size", data_process_queue.qsize(), global_step)


            # else:
            #     # print(m[0], m[1], global_step)
            #     # writer.add_scalar(m[0], m[1], global_step)
            #     pass
            # if args.capture_video and args.prod_mode:
            #     video_files = glob.glob(f'videos/{experiment_name}/*.mp4')
            #     for video_file in video_files:
            #         if video_file not in existing_video_files:
            #             existing_video_files += [video_file]
            #             print(video_file)
            #             if len(existing_video_files) > 1:
            #                 wandb.log({"video.0": wandb.Video(existing_video_files[-2])})
    except KeyboardInterrupt:
        pass
    finally:
        # learner.terminate()
        # learner.join(timeout=1)
        for actor in actor_processes:
            actor.terminate()
            actor.join(timeout=1)
        # for data_processor in data_processor_processes:
        #     data_processor.terminate()
        #     data_processor.join(timeout=1)
        if args.capture_video and args.prod_mode:
            wandb.log({"video.0": wandb.Video(existing_video_files[-1])})
    # env.close()
    writer.close()
