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

def wrap_atari(env, max_episode_steps=None):
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
from gym.wrappers import TimeLimit, Monitor
import pybullet_envs
from gym.spaces import Discrete, Box, MultiBinary, MultiDiscrete, Space
import time
import random
import os
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

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
    parser.add_argument('--seed', type=int, default=1,
                        help='seed of the experiment')
    parser.add_argument('--total-timesteps', type=int, default=2000000,
                        help='total timesteps of the experiments')
    parser.add_argument('--torch-deterministic', type=lambda x:bool(strtobool(x)), default=True, nargs='?', const=True,
                        help='if toggled, `torch.backends.cudnn.deterministic=False`')
    parser.add_argument('--cuda', type=lambda x:bool(strtobool(x)), default=True, nargs='?', const=True,
                        help='if toggled, cuda will not be enabled by default')
    parser.add_argument('--prod-mode', type=lambda x:bool(strtobool(x)), default=False, nargs='?', const=True,
                        help='run the script in production mode and use wandb to log outputs')
    parser.add_argument('--capture-video', type=lambda x:bool(strtobool(x)), default=True, nargs='?', const=True,
                        help='weather to capture videos of the agent performances (check out `videos` folder)')
    parser.add_argument('--wandb-project-name', type=str, default="cleanRL",
                        help="the wandb's project name")
    parser.add_argument('--wandb-entity', type=str, default=None,
                        help="the entity (team) of wandb's project")

    # Algorithm specific arguments
    parser.add_argument('--minibatch-size', type=int, default=768,
                        help='the mini batch size of ppo')
    parser.add_argument('--num-envs', type=int, default=24,
                        help='the mini batch size of ppo')
    parser.add_argument('--num-steps', type=int, default=128,
                        help='the mini batch size of ppo')
    parser.add_argument('--gamma', type=float, default=0.99,
                        help='the discount factor gamma')
    parser.add_argument('--gae-lambda', type=float, default=0.95,
                        help='the lambda for the general advantage estimation')
    parser.add_argument('--ent-coef', type=float, default=0.01,
                        help="coefficient of the entropy")
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
    parser.add_argument('--policy-lr', type=float, default=3e-4,
                         help="the learning rate of the policy optimizer")
    parser.add_argument('--value-lr', type=float, default=3e-4,
                         help="the learning rate of the critic optimizer")
    parser.add_argument('--norm-obs', type=lambda x:bool(strtobool(x)), default=True, nargs='?', const=True,
                         help="Toggles observation normalization")
    parser.add_argument('--norm-returns', type=lambda x:bool(strtobool(x)), default=True, nargs='?', const=True,
                         help="Toggles returns normalization")
    parser.add_argument('--norm-adv', type=lambda x:bool(strtobool(x)), default=True, nargs='?', const=True,
                         help="Toggles advantages normalization")
    parser.add_argument('--obs-clip', type=float, default=10.0,
                         help="Value for reward clipping, as per the paper")
    parser.add_argument('--rew-clip', type=float, default=10.0,
                         help="Value for observation clipping, as per the paper")
    parser.add_argument('--anneal-lr', type=lambda x:bool(strtobool(x)), default=True, nargs='?', const=True,
                         help="Toggle learning rate annealing for policy and value networks")
    parser.add_argument('--weights-init', default="orthogonal", choices=["xavier", 'orthogonal'],
                         help='Selects the scheme to be used for weights initialization'),
    parser.add_argument('--clip-vloss', type=lambda x:bool(strtobool(x)), default=True, nargs='?', const=True,
                         help='Toggles wheter or not to use a clipped loss for the value function, as per the paper.')
    parser.add_argument('--pol-layer-norm', type=lambda x:bool(strtobool(x)), default=False, nargs='?', const=True,
                        help='Enables layer normalization in the policy network')

    args = parser.parse_args()
    if not args.seed:
        args.seed = int(time.time())

args.batch_size = int(args.num_envs * args.num_steps)

class RecordRealAtariEpisodeReward(gym.Wrapper):
    def reset(self):
        obs = super().reset()
        self.real_episode_reward = 0
        return obs
    
    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.real_episode_reward += reward
        if self.env.was_real_done:
            info["real_episode_reward"] = self.real_episode_reward
            self.real_episode_reward = 0
        return obs, reward, done, info

class ProbsVisualizationWrapper(gym.Wrapper):
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
def make_env(gym_id, seed):
    def thunk():
        env = gym.make(gym_id)
        env = wrap_atari(env)
        if args.capture_video:
            # env = ProbsVisualizationWrapper(env)
            env = Monitor(env, f'videos/{experiment_name}')
        env = RecordRealAtariEpisodeReward(wrap_pytorch(
            wrap_deepmind(
                env,
                clip_rewards=False,
                frame_stack=True,
                scale=False,
            )
        ))
        env.seed(seed)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        return env
    return thunk
envs = DummyVecEnv([make_env(args.gym_id, args.seed+i) for i in range(args.num_envs)])
# respect the default timelimit
assert isinstance(envs.action_space, Discrete), "only discrete action space is supported"

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

# ALGO LOGIC: initialize agent here:
class Policy(nn.Module):
    def __init__(self, frames=4):
        super(Policy, self).__init__()
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
            nn.ReLU(),
            layer_init(nn.Linear(512, envs.action_space.n), std=0.01)
        )

    def forward(self, x):
        x = torch.Tensor(x).to(device)
        return self.network(x)

    def get_action(self, x, action=None):
        logits = self.forward(x)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy()

class Value(nn.Module):
    def __init__(self, frames=4):
        super(Value, self).__init__()
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
            nn.ReLU(),
            layer_init(nn.Linear(512, 1), std=1)
        )

    def forward(self, x):
        x = torch.Tensor(x).to(device)
        return self.network(x)

def discount_cumsum(x, dones, gamma):
    """
    computing discounted cumulative sums of vectors that resets with dones
    input:
        vector x,  vector dones,
        [x0,       [0,
         x1,        0,
         x2         1,
         x3         0, 
         x4]        0]
    output:
        [x0 + discount * x1 + discount^2 * x2,
         x1 + discount * x2,
         x2,
         x3 + discount * x4,
         x4]
    """
    discount_cumsum = np.zeros_like(x)
    discount_cumsum[-1] = x[-1]
    for t in reversed(range(x.shape[0]-1)):
        discount_cumsum[t] = x[t] + gamma * discount_cumsum[t+1] * (1-dones[t])
    return discount_cumsum

pg = Policy().to(device)
vf = Value().to(device)

# MODIFIED: Separate optimizer and learning rates
pg_optimizer = optim.Adam(list(pg.parameters()), lr=args.policy_lr)
v_optimizer = optim.Adam(list(vf.parameters()), lr=args.value_lr)

# MODIFIED: Initializing learning rate anneal scheduler when need
if args.anneal_lr:
    anneal_fn = lambda f: max(0, 1-f / args.total_timesteps)
    pg_lr_scheduler = optim.lr_scheduler.LambdaLR(pg_optimizer, lr_lambda=anneal_fn)
    vf_lr_scheduler = optim.lr_scheduler.LambdaLR(v_optimizer, lr_lambda=anneal_fn)

loss_fn = nn.MSELoss()

def sf01(arr):
    """
    swap and then flatten axes 0 and 1
    """
    s = arr.shape
    return arr.swapaxes(0, 1).reshape(s[0] * s[1], *s[2:])


# TRY NOT TO MODIFY: start the game
global_step = 0
next_obs = envs.reset()
while global_step < args.total_timesteps:
    # ALGO Logic: Storage for epoch data
    obs = np.empty((args.num_steps, args.num_envs) + envs.observation_space.shape)
    actions = np.empty((args.num_steps, args.num_envs) + envs.action_space.shape)
    logprobs = np.zeros((args.num_steps, args.num_envs))
    rewards = np.zeros((args.num_steps, args.num_envs))
    # real_rewards = []
    dones = np.zeros((args.num_steps, args.num_envs))
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)
    

    # TRY NOT TO MODIFY: prepare the execution of the game.
    for step in range(0, args.num_steps):
        global_step += 1 * args.num_envs
        obs[step] = next_obs.copy()

        # ALGO LOGIC: put action logic here
        with torch.no_grad():
            values[step] = vf.forward(obs[step]).flatten()
            action, logproba, _ = pg.get_action(obs[step])

        actions[step] = action.data.cpu().numpy()
        logprobs[step] = logproba.data.cpu().numpy()

        # TRY NOT TO MODIFY: execute the game and log data.
        next_obs, rewards[step], dones[step], infos = envs.step(action.tolist())

        # Annealing the rate if instructed to do so.
        if args.anneal_lr:
            pg_lr_scheduler.step()
            vf_lr_scheduler.step()
        for info in infos:
            if "real_episode_reward" in info:
                writer.add_scalar("charts/episode_reward", info["real_episode_reward"], global_step)
                print(f"global_step={global_step}, episode_reward={info['real_episode_reward']}")

    # bootstrap reward if not done. reached the batch limit
    # last_value = 0
    # if not dones[step]:
    #     last_value = vf.forward(next_obs.reshape((1,)+next_obs.shape))[0].detach().cpu().numpy()[0]
    # raise
    # last_value = vf.forward(next_obs).detach().cpu().reshape(1, -1)
    # bootstrapped_rewards = np.append(rewards, last_value, 0)

    # # calculate the returns and advantages
    # # TODO: needs verification
    # if args.gae:
    #     bootstrapped_values = np.append(values.detach().cpu().numpy(), last_value, 0)
    #     deltas = bootstrapped_rewards[:-1] + args.gamma * bootstrapped_values[1:] * (1-dones) - bootstrapped_values[:-1]
    #     advantages = discount_cumsum(deltas, dones, args.gamma * args.gae_lambda)
    #     advantages = torch.Tensor(advantages).to(device)
    #     returns = advantages + values
    # else:
    #     returns = discount_cumsum(bootstrapped_rewards, dones, args.gamma)[:-1]
    #     advantages = returns - values.detach().cpu().numpy()
    #     advantages = torch.Tensor(advantages).to(device)
    #     returns = torch.Tensor(returns).to(device)
    last_value = vf.forward(next_obs).detach().cpu().numpy().reshape(1, -1)
    advantages = np.zeros_like(rewards)
    lastgaelam = 0
    for t in reversed(range(args.num_steps)):
        if t == args.num_steps - 1:
            nextnonterminal = 1.0 - dones[step]
            nextvalues = last_value
        else:
            nextnonterminal = 1.0 - dones[t+1]
            nextvalues = values.detach().cpu().numpy()[t+1]
        delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values.detach().cpu().numpy()[t]
        advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
    returns = advantages + values.detach().cpu().numpy()
    # raise
    
    
    # flatten
    # obs = obs.reshape((-1,)+envs.observation_space.shape)
    # logprobs = logprobs.reshape(-1)
    # actions = actions.reshape(-1)
    # advantages = advantages.reshape(-1)
    # returns = returns.reshape(-1)
    # values = values.reshape(-1)
    
    obs = sf01(obs)
    logprobs = sf01(logprobs)
    actions = sf01(actions)
    advantages = sf01(advantages)
    returns = sf01(returns)
    values = values.transpose(0, 1).reshape(-1)
    # raise
    advantages = torch.Tensor(advantages).to(device)
    returns = torch.Tensor(returns).to(device)


    # Advantage normalization
    if args.norm_adv:
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-10)



    # Optimizaing policy network
    # First Tensorize all that is need to be so, clears up the loss computation part
    logprobs = torch.Tensor(logprobs).to(device) # Called 2 times: during policy update and KL bound checked
    entropys = []
    target_pg = Policy().to(device)
    inds = np.arange(args.batch_size,)
    for i_epoch_pi in range(args.update_epochs):
        np.random.shuffle(inds)
        target_pg.load_state_dict(pg.state_dict())
        for start in range(0, args.batch_size, args.minibatch_size):
            end = start + args.minibatch_size
            minibatch_ind = inds[start:end]
            _, newlogproba, entropy = pg.get_action(obs[minibatch_ind], torch.LongTensor(actions.astype(np.int)).to(device)[minibatch_ind])
            ratio = (newlogproba - logprobs[minibatch_ind]).exp()
            
    
            # Policy loss as in OpenAI SpinUp
            clip_adv = torch.where(advantages[minibatch_ind] > 0,
                                    (1.+args.clip_coef) * advantages[minibatch_ind],
                                    (1.-args.clip_coef) * advantages[minibatch_ind]).to(device)
    
            # Entropy computation with resampled actions
            entropys.append(entropy.mean().item())
            policy_loss = (-torch.min(ratio * advantages[minibatch_ind], clip_adv)).mean() + args.ent_coef * entropy.mean()
            
            pg_optimizer.zero_grad()
            policy_loss.backward()
            nn.utils.clip_grad_norm_(pg.parameters(), args.max_grad_norm)
            pg_optimizer.step()
    
            # KEY TECHNIQUE: This will stop updating the policy once the KL has been breached
            approx_kl = (logprobs[minibatch_ind] - newlogproba).mean()
    
            # Resample values
            new_values = vf.forward(obs[minibatch_ind]).view(-1)
    
            # Value loss clipping
            if args.clip_vloss:
                v_loss_unclipped = ((new_values - returns[minibatch_ind]) ** 2)
                v_clipped = values[minibatch_ind] + torch.clamp(new_values - values[minibatch_ind], -args.clip_coef, args.clip_coef)
                v_loss_clipped = (v_clipped - returns[minibatch_ind])**2
                v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                v_loss = 0.5 * v_loss_max.mean()
            else:
                v_loss = torch.mean((returns[minibatch_ind]- new_values).pow(2))
    
            v_optimizer.zero_grad()
            v_loss.backward()
            nn.utils.clip_grad_norm_(vf.parameters(), args.max_grad_norm)
            v_optimizer.step()
        # raise
        if args.kle_stop:
            if approx_kl > args.target_kl:
                break
        if args.kle_rollback:
            if (logprobs[minibatch_ind] - pg.get_action(obs[minibatch_ind], torch.LongTensor(actions.astype(np.int)).to(device)[minibatch_ind])[1]).mean() > args.target_kl:
                pg.load_state_dict(target_pg.state_dict())
                break

    # TRY NOT TO MODIFY: record rewards for plotting purposes
    print(approx_kl)
    writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
    writer.add_scalar("losses/policy_loss", policy_loss.item(), global_step)
    writer.add_scalar("losses/entropy", np.mean(entropys), global_step)
    writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
    if args.kle_stop or args.kle_rollback:
        writer.add_scalar("debug/pg_stop_iter", i_epoch_pi, global_step)

envs.close()
writer.close()
