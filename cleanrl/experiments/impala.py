# # https://github.com/facebookresearch/torchbeast/blob/master/torchbeast/core/environment.py
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
        self.observation_space = spaces.Box(low=0, high=255, shape=(shp[:-1] + (shp[-1] * k,)), dtype=env.observation_space.dtype)

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
            self._out = np.concatenate(self._frames, axis=-1)
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

def make_atari(env_id, max_episode_steps=None):
    env = gym.make(env_id)
    assert 'NoFrameskip' in env.spec.id
    env = NoopResetEnv(env, noop_max=30)
    env = MaxAndSkipEnv(env, skip=4)

    assert max_episode_steps is None

    return env

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
    if frame_stack:
        env = FrameStack(env, 4)
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


def wrap_pytorch(env):
    return ImageToPyTorch(env)


"""Naive profiling using timeit. (Used in MonoBeast.)"""

import collections
import timeit


class Timings:
    """Not thread-safe."""

    def __init__(self):
        self._means = collections.defaultdict(int)
        self._vars = collections.defaultdict(int)
        self._counts = collections.defaultdict(int)
        self.reset()

    def reset(self):
        self.last_time = timeit.default_timer()

    def time(self, name):
        """Save an update for event `name`.

        Nerd alarm: We could just store a
            collections.defaultdict(list)
        and compute means and standard deviations at the end. But thanks to the
        clever math in Sutton-Barto
        (http://www.incompleteideas.net/book/first/ebook/node19.html) and
        https://math.stackexchange.com/a/103025/5051 we can update both the
        means and the stds online. O(1) FTW!
        """
        now = timeit.default_timer()
        x = now - self.last_time
        self.last_time = now

        n = self._counts[name]

        mean = self._means[name] + (x - self._means[name]) / (n + 1)
        var = (
            n * self._vars[name] + n * (self._means[name] - mean) ** 2 + (x - mean) ** 2
        ) / (n + 1)

        self._means[name] = mean
        self._vars[name] = var
        self._counts[name] += 1

    def means(self):
        return self._means

    def vars(self):
        return self._vars

    def stds(self):
        return {k: v ** 0.5 for k, v in self._vars.items()}

    def summary(self, prefix=""):
        means = self.means()
        stds = self.stds()
        total = sum(means.values())

        result = prefix
        for k in sorted(means, key=means.get, reverse=True):
            result += f"\n    %s: %.6fms +- %.6fms (%.2f%%) " % (
                k,
                1000 * means[k],
                1000 * stds[k],
                100 * means[k] / total,
            )
        result += "\nTotal: %.6fms" % (1000 * total)
        return result

import argparse
import logging
import os
import pprint
import threading
import time
import timeit
import traceback
import typing

os.environ["OMP_NUM_THREADS"] = "1"  # Necessary for multithreading.

import torch
from torch import multiprocessing as mp
from torch import nn
from torch.nn import functional as F

from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecEnvWrapper

import collections
import torch
import torch.nn.functional as F

# yapf: disable
parser = argparse.ArgumentParser(description="PyTorch Scalable Agent")
# Common arguments
parser.add_argument('--exp-name', type=str, default=os.path.basename(__file__).rstrip(".py"),
                    help='the name of this experiment')
parser.add_argument('--gym-id', type=str, default="PongNoFrameskip-v4",
                    help='the id of the gym environment')
parser.add_argument("--mode", default="train",
                    choices=["train", "test", "test_render"],
                    help="Training or test mode.")
parser.add_argument("--xpid", default=None,
                    help="Experiment id (default: None).")

# Training settings.
parser.add_argument("--disable_checkpoint", action="store_true",
                    help="Disable saving checkpoint.")
parser.add_argument("--savedir", default="~/logs/torchbeast",
                    help="Root dir where experiment data will be saved.")
parser.add_argument("--num_actors", default=45, type=int, metavar="N",
                    help="Number of actors (default: 4).")
parser.add_argument("--total_steps", default=30000000, type=int, metavar="T",
                    help="Total environment steps to train for.")
parser.add_argument("--batch_size", default=4, type=int, metavar="B",
                    help="Learner batch size.")
parser.add_argument("--unroll_length", default=80, type=int, metavar="T",
                    help="The unroll length (time dimension).")
parser.add_argument("--num_buffers", default=60, type=int,
                    metavar="N", help="Number of shared-memory buffers.")
parser.add_argument("--num_learner_threads", "--num_threads", default=4, type=int,
                    metavar="N", help="Number learner threads.")
parser.add_argument("--disable_cuda", action="store_true",
                    help="Disable CUDA.")
parser.add_argument("--use_lstm", action="store_true",
                    help="Use LSTM in agent model.")

# Loss settings.
parser.add_argument("--entropy_cost", default=0.01,
                    type=float, help="Entropy cost/multiplier.")
parser.add_argument("--baseline_cost", default=0.5,
                    type=float, help="Baseline cost/multiplier.")
parser.add_argument("--discounting", default=0.99,
                    type=float, help="Discounting factor.")
parser.add_argument("--reward_clipping", default="abs_one",
                    choices=["abs_one", "none"],
                    help="Reward clipping.")

# Optimizer settings.
parser.add_argument("--learning_rate", default=0.00048,
                    type=float, metavar="LR", help="Learning rate.")
parser.add_argument("--alpha", default=0.99, type=float,
                    help="RMSProp smoothing constant.")
parser.add_argument("--momentum", default=0, type=float,
                    help="RMSProp momentum.")
parser.add_argument("--epsilon", default=0.01, type=float,
                    help="RMSProp epsilon.")
parser.add_argument("--grad_norm_clipping", default=40.0, type=float,
                    help="Global gradient norm clip.")
# yapf: enable


def _format_frame(frame):
    frame = torch.from_numpy(frame)
    return frame.view((1, 1) + frame.shape)  # (...) -> (T,B,...).


class Environment:
    def __init__(self, gym_env):
        self.gym_env = gym_env
        self.episode_return = None
        self.episode_step = None

    def initial(self):
        initial_reward = torch.zeros(1, 1)
        # This supports only single-tensor actions ATM.
        initial_last_action = torch.zeros(1, 1, dtype=torch.int64)
        self.episode_return = torch.zeros(1, 1)
        self.episode_step = torch.zeros(1, 1, dtype=torch.int32)
        initial_done = torch.ones(1, 1, dtype=torch.uint8)
        initial_frame = _format_frame(self.gym_env.reset())
        return dict(
            frame=initial_frame,
            reward=initial_reward,
            done=initial_done,
            episode_return=self.episode_return,
            episode_step=self.episode_step,
            last_action=initial_last_action,
        )

    def step(self, action):
        frame, reward, done, unused_info = self.gym_env.step(action.item())
        self.episode_step += 1
        self.episode_return += reward
        episode_step = self.episode_step
        episode_return = self.episode_return
        if done:
            frame = self.gym_env.reset()
            self.episode_return = torch.zeros(1, 1)
            self.episode_step = torch.zeros(1, 1, dtype=torch.int32)

        frame = _format_frame(frame)
        reward = torch.tensor(reward).view(1, 1)
        done = torch.tensor(done).view(1, 1)

        return dict(
            frame=frame,
            reward=reward,
            done=done,
            episode_return=episode_return,
            episode_step=episode_step,
            last_action=action,
        )

    def close(self):
        self.gym_env.close()

Buffers = typing.Dict[str, typing.List[torch.Tensor]]


def create_buffers(args, obs_shape, num_actions) -> Buffers:
    T = args.unroll_length
    specs = dict(
        frame=dict(size=(T + 1, *obs_shape), dtype=torch.uint8),
        reward=dict(size=(T + 1,), dtype=torch.float32),
        done=dict(size=(T + 1,), dtype=torch.bool),
        episode_return=dict(size=(T + 1,), dtype=torch.float32),
        episode_step=dict(size=(T + 1,), dtype=torch.int32),
        policy_logits=dict(size=(T + 1, num_actions), dtype=torch.float32),
        baseline=dict(size=(T + 1,), dtype=torch.float32),
        last_action=dict(size=(T + 1,), dtype=torch.int64),
        action=dict(size=(T + 1,), dtype=torch.int64),
    )
    buffers: Buffers = {key: [] for key in specs}
    for _ in range(args.num_buffers):
        for key in buffers:
            buffers[key].append(torch.empty(**specs[key]).share_memory_())
    return buffers


def act(
    args,
    actor_index: int,
    free_queue: mp.SimpleQueue,
    full_queue: mp.SimpleQueue,
    model: torch.nn.Module,
    buffers: Buffers,
    initial_agent_state_buffers,
):
    try:
        logging.info("Actor %i started.", actor_index)
        timings = Timings()  # Keep track of how fast things are.

        gym_env = create_env(args)
        seed = actor_index ^ int.from_bytes(os.urandom(4), byteorder="little")
        gym_env.seed(seed)
        env = Environment(gym_env)
        def make_env(args):
            def thunk():
                env = create_env(args)
                return env
            return thunk
        envs = DummyVecEnv([make_env(args) for i in range(1)])
        
        env_output = env.initial()
        envs.reset()
        agent_state = model.initial_state(batch_size=1)
        agent_output, unused_state = model(env_output, agent_state)
        while True:
            index = free_queue.get()
            if index is None:
                break

            # Write old rollout end.
            for key in env_output:
                buffers[key][index][0, ...] = env_output[key]
            for key in agent_output:
                buffers[key][index][0, ...] = agent_output[key]
            for i, tensor in enumerate(agent_state):
                initial_agent_state_buffers[index][i][...] = tensor

            # Do new rollout.
            for t in range(args.unroll_length):
                timings.reset()

                with torch.no_grad():
                    agent_output, agent_state = model(env_output, agent_state)

                # timings.time("model")

                env_output = env.step(agent_output["action"])
                # env_output = env.step(agent_output["action"])
                # envs.step((torch.randint(0, envs.action_space.n, (envs.num_envs,))).numpy())
                assert agent_output["action"] == env_output["last_action"]
                timings.time("step")

                for key in env_output:
                    buffers[key][index][t + 1, ...] = env_output[key]
                for key in agent_output:
                    buffers[key][index][t + 1, ...] = agent_output[key]

                timings.time("write")
            full_queue.put(index)

        if actor_index == 0:
            logging.info("Actor %i: %s", actor_index, timings.summary())

    except KeyboardInterrupt:
        pass  # Return silently.
    except Exception as e:
        logging.error("Exception in worker process %i", actor_index)
        traceback.print_exc()
        print()
        raise e




VTraceFromLogitsReturns = collections.namedtuple(
    "VTraceFromLogitsReturns",
    [
        "vs",
        "pg_advantages",
        "log_rhos",
        "behavior_action_log_probs",
        "target_action_log_probs",
    ],
)

VTraceReturns = collections.namedtuple("VTraceReturns", "vs pg_advantages")


def action_log_probs(policy_logits, actions):
    return -F.nll_loss(
        F.log_softmax(torch.flatten(policy_logits, 0, -2), dim=-1),
        torch.flatten(actions),
        reduction="none",
    ).view_as(actions)


def from_logits(
    behavior_policy_logits,
    target_policy_logits,
    actions,
    discounts,
    rewards,
    values,
    bootstrap_value,
    clip_rho_threshold=1.0,
    clip_pg_rho_threshold=1.0,
):
    """V-trace for softmax policies."""

    target_action_log_probs = action_log_probs(target_policy_logits, actions)
    behavior_action_log_probs = action_log_probs(behavior_policy_logits, actions)
    log_rhos = target_action_log_probs - behavior_action_log_probs
    vtrace_returns = from_importance_weights(
        log_rhos=log_rhos,
        discounts=discounts,
        rewards=rewards,
        values=values,
        bootstrap_value=bootstrap_value,
        clip_rho_threshold=clip_rho_threshold,
        clip_pg_rho_threshold=clip_pg_rho_threshold,
    )
    return VTraceFromLogitsReturns(
        log_rhos=log_rhos,
        behavior_action_log_probs=behavior_action_log_probs,
        target_action_log_probs=target_action_log_probs,
        **vtrace_returns._asdict(),
    )


@torch.no_grad()
def from_importance_weights(
    log_rhos,
    discounts,
    rewards,
    values,
    bootstrap_value,
    clip_rho_threshold=1.0,
    clip_pg_rho_threshold=1.0,
):
    """V-trace from log importance weights."""
    with torch.no_grad():
        rhos = torch.exp(log_rhos)
        if clip_rho_threshold is not None:
            clipped_rhos = torch.clamp(rhos, max=clip_rho_threshold)
        else:
            clipped_rhos = rhos

        cs = torch.clamp(rhos, max=1.0)
        # Append bootstrapped value to get [v1, ..., v_t+1]
        values_t_plus_1 = torch.cat(
            [values[1:], torch.unsqueeze(bootstrap_value, 0)], dim=0
        )
        deltas = clipped_rhos * (rewards + discounts * values_t_plus_1 - values)

        acc = torch.zeros_like(bootstrap_value)
        result = []
        for t in range(discounts.shape[0] - 1, -1, -1):
            acc = deltas[t] + discounts[t] * cs[t] * acc
            result.append(acc)
        result.reverse()
        vs_minus_v_xs = torch.stack(result)

        # Add V(x_s) to get v_s.
        vs = torch.add(vs_minus_v_xs, values)

        # Advantage for policy gradient.
        broadcasted_bootstrap_values = torch.ones_like(vs[0]) * bootstrap_value
        vs_t_plus_1 = torch.cat(
            [vs[1:], broadcasted_bootstrap_values.unsqueeze(0)], dim=0
        )
        if clip_pg_rho_threshold is not None:
            clipped_pg_rhos = torch.clamp(rhos, max=clip_pg_rho_threshold)
        else:
            clipped_pg_rhos = rhos
        pg_advantages = clipped_pg_rhos * (rewards + discounts * vs_t_plus_1 - values)

        # Make sure no gradients backpropagated through the returned values.
        return VTraceReturns(vs=vs, pg_advantages=pg_advantages)



def get_batch(
    args,
    free_queue: mp.SimpleQueue,
    full_queue: mp.SimpleQueue,
    buffers: Buffers,
    initial_agent_state_buffers,
    timings,
    lock=threading.Lock(),
):
    with lock:
        timings.time("lock")
        indices = [full_queue.get() for _ in range(args.batch_size)]
        timings.time("dequeue")
    batch = {
        key: torch.stack([buffers[key][m] for m in indices], dim=1) for key in buffers
    }
    initial_agent_state = (
        torch.cat(ts, dim=1)
        for ts in zip(*[initial_agent_state_buffers[m] for m in indices])
    )
    timings.time("batch")
    for m in indices:
        free_queue.put(m)
    timings.time("enqueue")
    batch = {k: t.to(device=args.device, non_blocking=True) for k, t in batch.items()}
    initial_agent_state = tuple(
        t.to(device=args.device, non_blocking=True) for t in initial_agent_state
    )
    timings.time("device")
    return batch, initial_agent_state


def compute_baseline_loss(advantages):
    return 0.5 * torch.sum(advantages ** 2)


def compute_entropy_loss(logits):
    """Return the entropy loss, i.e., the negative entropy of the policy."""
    policy = F.softmax(logits, dim=-1)
    log_policy = F.log_softmax(logits, dim=-1)
    return torch.sum(policy * log_policy)


def compute_policy_gradient_loss(logits, actions, advantages):
    cross_entropy = F.nll_loss(
        F.log_softmax(torch.flatten(logits, 0, 1), dim=-1),
        target=torch.flatten(actions, 0, 1),
        reduction="none",
    )
    cross_entropy = cross_entropy.view_as(advantages)
    return torch.sum(cross_entropy * advantages.detach())



class AtariNet(nn.Module):
    def __init__(self, observation_shape, num_actions, use_lstm=False):
        super(AtariNet, self).__init__()
        self.observation_shape = observation_shape
        self.num_actions = num_actions

        # Feature extraction.
        self.conv1 = nn.Conv2d(
            in_channels=self.observation_shape[0],
            out_channels=32,
            kernel_size=8,
            stride=4,
        )
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)

        # Fully connected layer.
        self.fc = nn.Linear(3136, 512)

        # FC output size + one-hot of last action + last reward.
        core_output_size = self.fc.out_features + num_actions + 1

        self.use_lstm = use_lstm
        if use_lstm:
            self.core = nn.LSTM(core_output_size, core_output_size, 2)

        self.policy = nn.Linear(core_output_size, self.num_actions)
        self.baseline = nn.Linear(core_output_size, 1)

    def initial_state(self, batch_size):
        if not self.use_lstm:
            return tuple()
        return tuple(
            torch.zeros(self.core.num_layers, batch_size, self.core.hidden_size)
            for _ in range(2)
        )

    def forward(self, inputs, core_state=()):
        x = inputs["frame"]  # [T, B, C, H, W].
        T, B, *_ = x.shape
        x = torch.flatten(x, 0, 1)  # Merge time and batch.
        x = x.float() / 255.0
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(T * B, -1)
        x = F.relu(self.fc(x))

        one_hot_last_action = F.one_hot(
            inputs["last_action"].view(T * B), self.num_actions
        ).float()
        clipped_reward = torch.clamp(inputs["reward"], -1, 1).view(T * B, 1)
        core_input = torch.cat([x, clipped_reward, one_hot_last_action], dim=-1)

        if self.use_lstm:
            core_input = core_input.view(T, B, -1)
            core_output_list = []
            notdone = (~inputs["done"]).float()
            for input, nd in zip(core_input.unbind(), notdone.unbind()):
                # Reset core state to zero whenever an episode ended.
                # Make `done` broadcastable with (num_layers, B, hidden_size)
                # states:
                nd = nd.view(1, -1, 1)
                core_state = tuple(nd * s for s in core_state)
                output, core_state = self.core(input.unsqueeze(0), core_state)
                core_output_list.append(output)
            core_output = torch.flatten(torch.cat(core_output_list), 0, 1)
        else:
            core_output = core_input
            core_state = tuple()

        policy_logits = self.policy(core_output)
        baseline = self.baseline(core_output)

        if self.training:
            action = torch.multinomial(F.softmax(policy_logits, dim=1), num_samples=1)
        else:
            # Don't sample when testing.
            action = torch.argmax(policy_logits, dim=1)

        policy_logits = policy_logits.view(T, B, self.num_actions)
        baseline = baseline.view(T, B)
        action = action.view(T, B)

        return (
            dict(policy_logits=policy_logits, baseline=baseline, action=action),
            core_state,
        )


def batch_and_learn(i, lock=threading.Lock()):
    """Thread target for the learning process."""
    global step, stats
    timings = Timings()
    while step < args.total_steps:
        timings.reset()
        batch, agent_state = get_batch(
            args,
            free_queue,
            full_queue,
            buffers,
            initial_agent_state_buffers,
            timings,
        )
        actor_model = model
        initial_agent_state = agent_state
        """Performs a learning (optimization) step."""
        with lock:
            learner_outputs, unused_state = learner_model(batch, initial_agent_state)

            # Take final value function slice for bootstrapping.
            bootstrap_value = learner_outputs["baseline"][-1]

            # Move from obs[t] -> action[t] to action[t] -> obs[t].
            batch = {key: tensor[1:] for key, tensor in batch.items()}
            learner_outputs = {key: tensor[:-1] for key, tensor in learner_outputs.items()}

            rewards = batch["reward"]
            if args.reward_clipping == "abs_one":
                clipped_rewards = torch.clamp(rewards, -1, 1)
            elif args.reward_clipping == "none":
                clipped_rewards = rewards

            discounts = (~batch["done"]).float() * args.discounting

            vtrace_returns = from_logits(
                behavior_policy_logits=batch["policy_logits"],
                target_policy_logits=learner_outputs["policy_logits"],
                actions=batch["action"],
                discounts=discounts,
                rewards=clipped_rewards,
                values=learner_outputs["baseline"],
                bootstrap_value=bootstrap_value,
            )

            pg_loss = compute_policy_gradient_loss(
                learner_outputs["policy_logits"],
                batch["action"],
                vtrace_returns.pg_advantages,
            )
            baseline_loss = args.baseline_cost * compute_baseline_loss(
                vtrace_returns.vs - learner_outputs["baseline"]
            )
            entropy_loss = args.entropy_cost * compute_entropy_loss(
                learner_outputs["policy_logits"]
            )

            total_loss = pg_loss + baseline_loss + entropy_loss

            episode_returns = batch["episode_return"][batch["done"]]
            stats = {
                "episode_returns": tuple(episode_returns.cpu().numpy()),
                "mean_episode_return": torch.mean(episode_returns).item(),
                "total_loss": total_loss.item(),
                "pg_loss": pg_loss.item(),
                "baseline_loss": baseline_loss.item(),
                "entropy_loss": entropy_loss.item(),
            }

            optimizer.zero_grad()
            total_loss.backward()
            nn.utils.clip_grad_norm_(learner_model.parameters(), args.grad_norm_clipping)
            optimizer.step()
            scheduler.step()

            actor_model.load_state_dict(learner_model.state_dict())
        timings.time("learn")
        with lock:
            step += T * B

    if i == 0:
        logging.info("Batch and learn: %s", timings.summary())

logging.basicConfig(
    format=(
        "[%(levelname)s:%(process)d %(module)s:%(lineno)d %(asctime)s] " "%(message)s"
    ),
    level=0,
)

Net = AtariNet

def create_env(args):
    return wrap_pytorch(
        wrap_deepmind(
            make_atari(args.gym_id),
            clip_rewards=False,
            frame_stack=True,
            scale=False,
        )
    )

args = parser.parse_args()

if args.num_buffers is None:  # Set sensible default for num_buffers.
    args.num_buffers = max(2 * args.num_actors, args.batch_size)
if args.num_actors >= args.num_buffers:
    raise ValueError("num_buffers should be larger than num_actors")
if args.num_buffers < args.batch_size:
    raise ValueError("num_buffers should be larger than batch_size")

T = args.unroll_length
B = args.batch_size

args.device = None
if not args.disable_cuda and torch.cuda.is_available():
    logging.info("Using CUDA.")
    args.device = torch.device("cuda")
else:
    logging.info("Not using CUDA.")
    args.device = torch.device("cpu")

env = create_env(args)

model = Net(env.observation_space.shape, env.action_space.n, args.use_lstm)
buffers = create_buffers(args, env.observation_space.shape, model.num_actions)

model.share_memory()

# Add initial RNN state.
initial_agent_state_buffers = []
for _ in range(args.num_buffers):
    state = model.initial_state(batch_size=1)
    for t in state:
        t.share_memory_()
    initial_agent_state_buffers.append(state)

actor_processes = []
ctx = mp.get_context("fork")
free_queue = ctx.SimpleQueue()
full_queue = ctx.SimpleQueue()

for i in range(args.num_actors):
    actor = ctx.Process(
        target=act,
        args=(
            args,
            i,
            free_queue,
            full_queue,
            model,
            buffers,
            initial_agent_state_buffers,
        ),
    )
    actor.start()
    actor_processes.append(actor)

learner_model = Net(
    env.observation_space.shape, env.action_space.n, args.use_lstm
).to(device=args.device)

optimizer = torch.optim.RMSprop(
    learner_model.parameters(),
    lr=args.learning_rate,
    momentum=args.momentum,
    eps=args.epsilon,
    alpha=args.alpha,
)

def lr_lambda(epoch):
    return 1 - min(epoch * T * B, args.total_steps) / args.total_steps

scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

logger = logging.getLogger("logfile")
stat_keys = [
    "total_loss",
    "mean_episode_return",
    "pg_loss",
    "baseline_loss",
    "entropy_loss",
]
logger.info("# Step\t%s", "\t".join(stat_keys))

step, stats = 0, {}



for m in range(args.num_buffers):
    free_queue.put(m)

threads = []
for i in range(args.num_learner_threads):
    thread = threading.Thread(
        target=batch_and_learn, name="batch-and-learn-%d" % i, args=(i,)
    )
    thread.start()
    threads.append(thread)

timer = timeit.default_timer
try:
    last_checkpoint_time = timer()
    while step < args.total_steps:
        start_step = step
        start_time = timer()
        time.sleep(5)

        sps = (step - start_step) / (timer() - start_time)
        if stats.get("episode_returns", None):
            mean_return = (
                "Return per episode: %.1f. " % stats["mean_episode_return"]
            )
        else:
            mean_return = ""
        total_loss = stats.get("total_loss", float("inf"))
        logging.info(
            "Steps %i @ %.1f SPS. Loss %f. %sStats:\n%s",
            step,
            sps,
            total_loss,
            mean_return,
            pprint.pformat(stats),
        )
except KeyboardInterrupt:
    pass
else:
    for thread in threads:
        thread.join()
    logging.info("Learning finished after %d steps.", step)
finally:
    for _ in range(args.num_actors):
        free_queue.put(None)
    for actor in actor_processes:
        actor.join(timeout=1)
