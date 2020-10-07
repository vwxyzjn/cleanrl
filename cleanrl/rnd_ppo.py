# https://github.com/facebookresearch/torchbeast/blob/master/torchbeast/core/environment.py

import gc
import time
import numpy as np
from collections import deque
import gym
from gym import spaces
import cv2
cv2.ocl.setUseOpenCL(False)


class RunningMeanStd(object):
    # https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
    def __init__(self, epsilon=1e-4, shape=()):
        self.mean = np.zeros(shape, 'float64')
        self.var = np.ones(shape, 'float64')
        self.count = epsilon

    def update(self, x):
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * (self.count)
        m_b = batch_var * (batch_count)
        M2 = m_a + m_b + np.square(delta) * self.count * batch_count / (self.count + batch_count)
        new_var = M2 / (self.count + batch_count)

        new_count = batch_count + self.count

        self.mean = new_mean
        self.var = new_var
        self.count = new_count

class RewardForwardFilter(object):
    def __init__(self, gamma):
        self.rewems = None
        self.gamma = gamma

    def update(self, rews):
        if self.rewems is None:
            self.rewems = rews
        else:
            self.rewems = self.rewems * self.gamma + rews
        return self.rewems

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

class StickyAction(gym.Wrapper):
    def __init__(self, env, sticky_action=True, p=0.25):
        gym.Wrapper.__init__(self, env)
        self.sticky_action = sticky_action
        self.last_action = 0
        self.p = p
    
    def step(self, action):
        if self.sticky_action:
            if np.random.rand() <= self.p:
                action = self.last_action
            self.last_action = action
        return self.env.step(action)

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

def wrap_atari(env, max_episode_steps=None, sticky_action=True):
    assert 'NoFrameskip' in env.spec.id
    env = NoopResetEnv(env, noop_max=30)
    env = MaxAndSkipEnv(env, skip=4)
    if sticky_action:
        env = StickyAction(env)
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

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter

import argparse
from distutils.util import strtobool
import numpy as np
import gym
import vizdoomgym
from gym.wrappers import TimeLimit, Monitor
import pybullet_envs
from gym.spaces import Discrete, Box, MultiBinary, MultiDiscrete, Space
import time
import random
import os
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecEnvWrapper

import matplotlib
matplotlib.use('Agg')
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PPO agent')
    # Common arguments
    parser.add_argument('--exp-name', type=str, default=os.path.basename(__file__).rstrip(".py"),
                        help='the name of this experiment')
    parser.add_argument('--gym-id', type=str, default="BreakoutNoFrameskip-v4",
                        help='the id of the gym environment')
    parser.add_argument('--learning-rate', type=float, default=2.5e-4,
                        help='the learning rate of the optimizer')
    parser.add_argument('--seed', type=int, default=1,
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
    parser.add_argument('--video-interval', type=int, default=50,
                        help='the episode interval for capturing video')
    parser.add_argument('--wandb-project-name', type=str, default="cleanRL",
                        help="the wandb's project name")
    parser.add_argument('--wandb-entity', type=str, default=None,
                        help="the entity (team) of wandb's project")

    # Algorithm specific arguments
    parser.add_argument('--n-minibatch', type=int, default=4,
                        help='the number of mini batch')
    parser.add_argument('--num-envs', type=int, default=8,
                        help='the number of parallel game environment')
    parser.add_argument('--num-steps', type=int, default=128,
                        help='the number of steps per game environment')
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
    parser.add_argument('--sticky-action', type=lambda x:bool(strtobool(x)), default=True, nargs='?', const=True,
                          help='Toggles wheter or not to use sticky action.')

    # RND arguments
    parser.add_argument('--update-proportion', type=float, default=0.25, help="proportion of exp used for predictor update")
    parser.add_argument('--int-coef', type=float, default=1.0, help="coefficient of extrinsic reward")
    parser.add_argument('--ext-coef', type=float, default=2.0, help="coefficient of intrinsic reward")
    parser.add_argument('--int-gamma', type=float, default=0.99, help="Intrinsic reward discount rate")
    


    args = parser.parse_args()
    if not args.seed:
        args.seed = int(time.time())


args.batch_size = int(args.num_envs * args.num_steps)
args.minibatch_size = int(args.batch_size // args.n_minibatch)


class VecPyTorch(VecEnvWrapper):
    def __init__(self, venv, device):
        super(VecPyTorch, self).__init__(venv)
        self.device = device

    def reset(self):
        obs = self.venv.reset()
        obs = torch.from_numpy(obs).float().to(self.device)
        return obs

    def step_async(self, actions):
        actions = actions.cpu().numpy()
        self.venv.step_async(actions)

    def step_wait(self):
        obs, reward, done, info = self.venv.step_wait()
        obs = torch.from_numpy(obs).float().to(self.device)
        reward = torch.from_numpy(reward).unsqueeze(dim=1).float()
        return obs, reward, done, info

class ProbsVisualizationWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.env.reset()
        self.image_shape = self.env.render(mode="rgb_array").shape
        self.probs = [[0.,0.,0.,0.]]
        # self.metadata['video.frames_per_second'] = 60
    def set_probs(self, probs):
        self.probs = probs
    def render(self, mode="human"):
        if mode=="rgb_array":
            env_rgb_array = super().render(mode)
            fig, ax = plt.subplots(figsize=(self.image_shape[1]/100,self.image_shape[0]/100), constrained_layout=True, dpi=100)
            df = pd.DataFrame(np.array(self.probs).T)
            sns.barplot(x=df.index, y=0, data=df, ax=ax)
            ax.set(xlabel='actions', ylabel='probs')
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
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.backends.cudnn.deterministic = args.torch_deterministic
def make_env(gym_id, seed, idx):
    def thunk():
        env = gym.make(gym_id)
        env = wrap_atari(env, sticky_action=args.sticky_action)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        if args.capture_video:
            if idx == 0:
                env = ProbsVisualizationWrapper(env)
                env = Monitor(env, f'videos/{experiment_name}',  video_callable=lambda episode_id: episode_id%args.video_interval==0)
        env = wrap_pytorch(
            wrap_deepmind(
                env,
                episode_life=True,
                clip_rewards=True,
                frame_stack=True,
                scale=False,
            )
        )
        env.seed(seed)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        return env
    return thunk
envs = VecPyTorch(DummyVecEnv([make_env(args.gym_id, args.seed+i, i) for i in range(args.num_envs)]), device)
# if args.prod_mode:
#     envs = VecPyTorch(
#         SubprocVecEnv([make_env(args.gym_id, args.seed+i, i) for i in range(args.num_envs)], "fork"),
#         device
#     )
assert isinstance(envs.action_space, Discrete), "only discrete action space is supported"

# ALGO LOGIC: initialize agent here:
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
    def __init__(self, envs, frames=4):
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
            layer_init(nn.Linear(3136, 256)),
            nn.ReLU(),
            layer_init(nn.Linear(
                256,
                448)),
            nn.ReLU()
        )
        self.extra_layer = nn.Sequential(
            layer_init(nn.Linear(448, 448), std=0.1),
            nn.ReLU()
        )
        self.actor = nn.Sequential(
            layer_init(nn.Linear(448, 448), std=0.01),
            nn.ReLU(),
            layer_init(nn.Linear(448, envs.action_space.n), std=0.01))
        self.critic_ext = layer_init(nn.Linear(448, 1), std=0.01)
        self.critic_int = layer_init(nn.Linear(448, 1), std=0.01)

    def forward(self, x):
        return self.network(x)

    def get_action(self, x, action=None):
        logits = self.actor(self.forward(x))
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy()

    def get_value(self, x):
        features = self.forward(x)
        return self.critic_ext(self.extra_layer(features) + features), self.critic_int(self.extra_layer(features) + features)

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

class RNDModel(nn.Module):
    def __init__(self, input_size, output_size):
        super(RNDModel, self).__init__()

        self.input_size = input_size
        self.output_size = output_size

        feature_output = 7 * 7 * 64

        # Prediction network
        self.predictor = nn.Sequential(
            layer_init(nn.Conv2d(
                in_channels=1,
                out_channels=32,
                kernel_size=8,
                stride=4)),
            nn.LeakyReLU(),
            layer_init(nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=4,
                stride=2)),
            nn.LeakyReLU(),
            layer_init(nn.Conv2d(
                in_channels=64,
                out_channels=64,
                kernel_size=3,
                stride=1)),
            nn.LeakyReLU(),
            Flatten(),
            layer_init(nn.Linear(feature_output, 512)),
            nn.ReLU(),
            layer_init(nn.Linear(512, 512)),
            nn.ReLU(),
            layer_init(nn.Linear(512, 512))
        )

        # Target network
        self.target = nn.Sequential(
            layer_init(nn.Conv2d(
                in_channels=1,
                out_channels=32,
                kernel_size=8,
                stride=4)),
            nn.LeakyReLU(),
            layer_init(nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=4,
                stride=2)),
            nn.LeakyReLU(),
            layer_init(nn.Conv2d(
                in_channels=64,
                out_channels=64,
                kernel_size=3,
                stride=1)),
            nn.LeakyReLU(),
            Flatten(),
            layer_init(nn.Linear(feature_output, 512))
        )

        # Set that target network is not trainable
        for param in self.target.parameters():
            param.requires_grad = False

    def forward(self, next_obs):
        target_feature = self.target(next_obs)
        predict_feature = self.predictor(next_obs)

        return predict_feature, target_feature


agent = Agent(envs).to(device)

rnd_model = RNDModel(4, envs.action_space.n).to(device)

optimizer = optim.Adam(list(agent.parameters()) + list(rnd_model.predictor.parameters()), lr=args.learning_rate, eps=1e-5)
if args.anneal_lr:
    # https://github.com/openai/baselines/blob/ea25b9e8b234e6ee1bca43083f8f3cf974143998/baselines/ppo2/defaults.py#L20
    lr = lambda f: f * args.learning_rate

reward_rms = RunningMeanStd()
obs_rms = RunningMeanStd(shape=(1, 1, 84, 84))

discounted_reward = RewardForwardFilter(args.int_gamma)


# ALGO Logic: Storage for epoch data
obs = torch.zeros((args.num_steps, args.num_envs) + envs.observation_space.shape).to(device)
actions = torch.zeros((args.num_steps, args.num_envs) + envs.action_space.shape).to(device)
logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
curiosity_rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
ext_values = torch.zeros((args.num_steps, args.num_envs)).to(device)
int_values = torch.zeros((args.num_steps, args.num_envs)).to(device)

# TRY NOT TO MODIFY: start the game
global_step = 0
# Note how `next_obs` and `next_done` are used; their usage is equivalent to
# https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail/blob/84a7582477fb0d5c82ad6d850fe476829dddd2e1/a2c_ppo_acktr/storage.py#L60
next_obs = envs.reset()
next_done = torch.zeros(args.num_envs).to(device)
num_updates = args.total_timesteps // args.batch_size

print('Start to initailize observation normalization parameter.....')
next_ob = []
for step in range(args.num_steps * 50):
    acs = torch.from_numpy(np.random.randint(0, envs.action_space.n, size=(args.num_envs,)))
    s, r, d, infos = envs.step(acs)
    next_ob = s[:, 3, :, :].reshape([-1, 1, 84, 84])

    if len(next_ob) % (args.num_steps * args.num_envs) == 0:
        obs_rms.update(next_ob)
        next_ob = []
print('End to initalize...')

for update in range(1, num_updates+1):
    # Annealing the rate if instructed to do so.
    if args.anneal_lr:
        frac = 1.0 - (update - 1.0) / num_updates
        lrnow = lr(frac)
        optimizer.param_groups[0]['lr'] = lrnow

    # TRY NOT TO MODIFY: prepare the execution of the game.
    for step in range(0, args.num_steps):
        global_step += 1 * args.num_envs
        obs[step] = next_obs
        dones[step] = next_done

        # ALGO LOGIC: put action logic here
        with torch.no_grad():
            value_ext, value_int = agent.get_value(obs[step])
            ext_values[step], int_values[step] = value_ext.flatten(), value_int.flatten()
            action, logproba, _ = agent.get_action(obs[step])

            # visualization
            if args.capture_video:
                probs_list = np.array(Categorical(
                    logits=agent.actor(agent.forward(obs[step]))).probs[0:1].tolist())
                envs.env_method("set_probs", probs_list, indices=0)
        
        actions[step] = action
        logprobs[step] = logproba

        # TRY NOT TO MODIFY: execute the game and log data.
        next_obs, rs, ds, infos = envs.step(action)
        rewards[step], next_done = rs.view(-1), torch.Tensor(ds).to(device)
        rnd_next_obs = torch.FloatTensor(((next_obs.data.cpu().numpy()[:,3,:,:].reshape(args.num_envs, 1, 84, 84) - obs_rms.mean) / np.sqrt(obs_rms.var)).clip(-5, 5)).to(device)
        target_next_feature = rnd_model.target(rnd_next_obs)
        predict_next_feature = rnd_model.predictor(rnd_next_obs)
        curiosity_rewards[step] = ((target_next_feature - predict_next_feature).pow(2).sum(1) / 2).data.cpu()

        for idx, info in enumerate(infos):
            if 'episode' in info.keys():
                print(f"global_step={global_step}, episode_reward={info['episode']['r']}, curiosity_reward={curiosity_rewards[step][idx]}")
                writer.add_scalar("charts/episode_reward", info['episode']['r'], global_step)
                writer.add_scalar("charts/episode_curiosity_reward", curiosity_rewards[step][idx], global_step)
                break
    
    total_reward_per_env = np.array([discounted_reward.update(reward_per_step) for reward_per_step in
                                         curiosity_rewards.data.cpu().numpy().T])
    mean, std, count = np.mean(total_reward_per_env), np.std(total_reward_per_env), len(total_reward_per_env)
    reward_rms.update_from_moments(mean, std ** 2, count)

    curiosity_rewards /= np.sqrt(reward_rms.var)

    # bootstrap reward if not done. reached the batch limit
    with torch.no_grad():
        last_value_ext, last_value_int = agent.get_value(next_obs.to(device))
        last_value_ext, last_value_int = last_value_ext.reshape(1, -1), last_value_int.reshape(1, -1)
        if args.gae:
            ext_advantages = torch.zeros_like(rewards).to(device)
            int_advantages = torch.zeros_like(curiosity_rewards).to(device)
            ext_lastgaelam = 0
            int_lastgaelam = 0
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    ext_nextnonterminal = 1.0 - next_done
                    int_nextnonterminal = 1.0
                    ext_nextvalues = last_value_ext
                    int_nextvalues = last_value_int
                else:
                    ext_nextnonterminal = 1.0 - dones[t+1]
                    int_nextnonterminal = 1.0
                    ext_nextvalues = ext_values[t+1]
                    int_nextvalues = int_values[t+1]
                ext_delta = rewards[t] + args.gamma * ext_nextvalues * ext_nextnonterminal - ext_values[t]
                int_delta = curiosity_rewards[t] + args.int_gamma * int_nextvalues * int_nextnonterminal - int_values[t]
                ext_advantages[t] = ext_lastgaelam = ext_delta + args.gamma * args.gae_lambda * ext_nextnonterminal * ext_lastgaelam
                int_advantages[t] = int_lastgaelam = int_delta + args.int_gamma * args.gae_lambda * int_nextnonterminal * int_lastgaelam
            ext_returns = ext_advantages + ext_values
            int_returns = int_advantages + int_values
        else:
            ext_returns = torch.zeros_like(rewards).to(device)
            int_returns = torch.zeros_like(curiosity_rewards).to(device)
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    ext_nextnonterminal = 1.0 - next_done
                    int_nextnonterminal = 1.0
                    ext_next_return = last_value_ext
                    int_next_return = last_value_int
                else:
                    ext_nextnonterminal = 1.0 - dones[t+1]
                    int_nextnonterminal = 1.0
                    ext_next_return = ext_returns[t+1]
                    int_next_return = int_returns[t+1]
                ext_returns[t] = rewards[t] + args.gamma * ext_nextnonterminal * ext_next_return
                int_returns[t] = curiosity_rewards[t] + args.int_gamma * int_nextnonterminal * int_next_return
            ext_advantages = ext_returns - ext_values
            int_advantages = int_returns - int_values

    # flatten the batch
    b_obs = obs.reshape((-1,)+envs.observation_space.shape)
    b_logprobs = logprobs.reshape(-1)
    b_actions = actions.reshape((-1))
    b_ext_advantages = ext_advantages.reshape(-1)
    b_int_advantages = int_advantages.reshape(-1)
    b_ext_returns = ext_returns.reshape(-1)
    b_int_returns = int_returns.reshape(-1)
    b_ext_values = ext_values.reshape(-1)

    b_advantages = b_int_advantages * args.int_coef + b_ext_advantages * args.ext_coef

    obs_rms.update(b_obs.data.cpu().numpy()[:,3,:,:].reshape(-1,1,84,84))

    # Optimizaing the policy and value network
    forward_mse = nn.MSELoss(reduction='none')
    target_agent = Agent(envs).to(device)
    inds = np.arange(args.batch_size,)

    rnd_next_obs = torch.FloatTensor(((b_obs.data.cpu().numpy()[:,3,:,:].reshape(-1, 1, 84, 84) - obs_rms.mean) / np.sqrt(obs_rms.var)).clip(-5, 5)).to(device)

    for i_epoch_pi in range(args.update_epochs):
        target_agent.load_state_dict(agent.state_dict())
        np.random.shuffle(inds)
        for start in range(0, args.batch_size, args.minibatch_size):
            end = start + args.minibatch_size
            minibatch_ind = inds[start:end]
            predict_next_state_feature, target_next_state_feature = rnd_model(rnd_next_obs[minibatch_ind])
            forward_loss = forward_mse(predict_next_state_feature, target_next_state_feature.detach()).mean(-1)
            
            mask = torch.rand(len(forward_loss)).to(device)
            mask = (mask < args.update_proportion).type(torch.FloatTensor).to(device)
            forward_loss = (forward_loss * mask).sum() / torch.max(mask.sum(), torch.Tensor([1]).to(device))

            mb_advantages = b_advantages[minibatch_ind]
            if args.norm_adv:
                mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

            _, newlogproba, entropy = agent.get_action(b_obs[minibatch_ind], b_actions.long()[minibatch_ind])
            ratio = (newlogproba - b_logprobs[minibatch_ind]).exp()

            # Stats
            approx_kl = (b_logprobs[minibatch_ind] - newlogproba).mean()

            # Policy loss
            pg_loss1 = -mb_advantages * ratio
            pg_loss2 = -mb_advantages * torch.clamp(ratio, 1-args.clip_coef, 1+args.clip_coef)
            pg_loss = torch.max(pg_loss1, pg_loss2).mean()
            entropy_loss = entropy.mean()

            # Value loss
            new_ext_values, new_int_values = agent.get_value(b_obs[minibatch_ind])
            new_ext_values, new_int_values = new_ext_values.view(-1), new_int_values.view(-1)
            if args.clip_vloss:
                ext_v_loss_unclipped = ((new_ext_values - b_ext_returns[minibatch_ind]) ** 2)
                ext_v_clipped = b_ext_values[minibatch_ind] + torch.clamp(new_ext_values - b_ext_values[minibatch_ind], -args.clip_coef, args.clip_coef)
                ext_v_loss_clipped = (ext_v_clipped - b_ext_returns[minibatch_ind])**2
                ext_v_loss_max = torch.max(ext_v_loss_unclipped, ext_v_loss_clipped)
                ext_v_loss = 0.5 * ext_v_loss_max.mean()
            else:
                ext_v_loss = 0.5 *((new_ext_values - b_ext_returns[minibatch_ind]) ** 2).mean()

            int_v_loss = 0.5 *((new_int_values - b_int_returns[minibatch_ind]) ** 2).mean()
            v_loss = ext_v_loss + int_v_loss

            optimizer.zero_grad()
            loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef + forward_loss

            loss.backward()
            nn.utils.clip_grad_norm_(list(agent.parameters()) + list(rnd_model.predictor.parameters()), args.max_grad_norm)
            optimizer.step()

        if args.kle_stop:
            if approx_kl > args.target_kl:
                break
        if args.kle_rollback:
            if (b_logprobs[minibatch_ind] - agent.get_action(b_obs[minibatch_ind], b_actions.long()[minibatch_ind])[1]).mean() > args.target_kl:
                agent.load_state_dict(target_agent.state_dict())
                break

    # TRY NOT TO MODIFY: record rewards for plotting purposes
    writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]['lr'], global_step)
    writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
    writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
    writer.add_scalar("losses/fwd_loss", forward_loss.item(), global_step)
    writer.add_scalar("losses/entropy", entropy.mean().item(), global_step)
    writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
    if args.kle_stop or args.kle_rollback:
        writer.add_scalar("debug/pg_stop_iter", i_epoch_pi, global_step)

envs.close()
writer.close()
