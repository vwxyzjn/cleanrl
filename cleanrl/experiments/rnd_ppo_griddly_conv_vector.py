# https://github.com/facebookresearch/torchbeast/blob/master/torchbeast/core/environment.py

import gym
import os
import colorsys
from griddly import GymWrapperFactory, gd
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
import griddly
from gym.wrappers import TimeLimit, Monitor
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
    parser.add_argument('--learning-rate', type=float, default=2.5e-4,
                        help='the learning rate of the optimizer')
    parser.add_argument('--seed', type=int, default=1,
                        help='seed of the experiment')
    parser.add_argument('--total-timesteps', type=int, default=1280000,
                        help='total timesteps of the experiments')
    parser.add_argument('--torch-deterministic', type=lambda x: bool(strtobool(x)), default=True, nargs='?', const=True,
                        help='if toggled, `torch.backends.cudnn.deterministic=False`')
    parser.add_argument('--cuda', type=lambda x: bool(strtobool(x)), default=True, nargs='?', const=True,
                        help='if toggled, cuda will not be enabled by default')
    parser.add_argument('--prod-mode', type=lambda x: bool(strtobool(x)), default=False, nargs='?', const=True,
                        help='run the script in production mode and use wandb to log outputs')
    parser.add_argument('--capture-video', type=lambda x: bool(strtobool(x)), default=False, nargs='?', const=True,
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
    parser.add_argument('--kle-stop', type=lambda x: bool(strtobool(x)), default=False, nargs='?', const=True,
                        help='If toggled, the policy updates will be early stopped w.r.t target-kl')
    parser.add_argument('--kle-rollback', type=lambda x: bool(strtobool(x)), default=False, nargs='?', const=True,
                        help='If toggled, the policy updates will roll back to previous policy if KL exceeds target-kl')
    parser.add_argument('--target-kl', type=float, default=0.03,
                        help='the target-kl variable that is referred by --kl')
    parser.add_argument('--gae', type=lambda x: bool(strtobool(x)), default=True, nargs='?', const=True,
                        help='Use GAE for advantage computation')
    parser.add_argument('--norm-adv', type=lambda x: bool(strtobool(x)), default=True, nargs='?', const=True,
                        help="Toggles advantages normalization")
    parser.add_argument('--anneal-lr', type=lambda x: bool(strtobool(x)), default=True, nargs='?', const=True,
                        help="Toggle learning rate annealing for policy and value networks")
    parser.add_argument('--clip-vloss', type=lambda x: bool(strtobool(x)), default=True, nargs='?', const=True,
                        help='Toggles wheter or not to use a clipped loss for the value function, as per the paper.')
    parser.add_argument('--sticky-action', type=lambda x: bool(strtobool(x)), default=True, nargs='?', const=True,
                        help='Toggles wheter or not to use sticky action.')

    # Griddly arguments
    parser.add_argument('--griddly-gdy-file', default='Single-Player/GVGAI/clusters.yaml',
                        help='Toggles wheter or not to use sticky action.')
    parser.add_argument('--griddly-level', type=int, default=0, help='The level number to train')

    # RND arguments
    parser.add_argument('--update-proportion', type=float, default=0.25,
                        help="proportion of exp used for predictor update")
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
        self.image_shape = self.env.render(mode="rgb_array", observer='global').shape
        self.probs = [[0., 0., 0., 0.]]
        # self.metadata['video.frames_per_second'] = 60

        self._scale = 20

        observation_channels = self.observation_space.shape[0]
        HSV_tuples = [(x * 1.0 / (observation_channels + 1), 1.0, 1.0) for x in range(observation_channels + 1)]

        rgb = []
        for hsv in HSV_tuples:
            rgb.append(colorsys.hsv_to_rgb(*hsv))

        self._rgb_pallette = (np.array(rgb) * 255).astype('uint8')

    def set_probs(self, probs):
        self.probs = probs

    def wrap_vector_visualization(self, observation):

        observation = observation.swapaxes(0,2)
        # add extra dimension so argmax does not get confused by 0 index and empty space
        pallette_buffer = np.ones([observation.shape[0] + 1, *observation.shape[1:]]) * 0.5
        pallette_buffer[1:] = observation

        # convert to RGB pallette
        vector_pallette = np.argmax(pallette_buffer, axis=0)

        buffer = self._rgb_pallette[vector_pallette].swapaxes(0, 1)
        # make the observation much bigger by repeating pixels
        observation = buffer.repeat(self._scale, 0).repeat(self._scale, 1)

        return observation

    def render(self, mode="human"):
        if mode == "rgb_array":
            dpi = 100
            env_rgb_array = self.wrap_vector_visualization(super().render(mode, observer='global'))
            fig, ax = plt.subplots(figsize=(self.image_shape[1]*self._scale/dpi, self.image_shape[0]*self._scale/dpi),
                                   constrained_layout=True, dpi=dpi)
            df = pd.DataFrame(np.array(self.probs).T)
            sns.barplot(x=df.index, y=0, data=df, ax=ax)
            ax.set(xlabel='actions', ylabel='probs')
            fig.canvas.draw()
            X = np.array(fig.canvas.renderer.buffer_rgba())
            Image.fromarray(X)
            rgb_image = np.array(Image.fromarray(X).convert('RGB'))
            plt.close(fig)
            q_value_rgb_array = rgb_image
            return np.append(env_rgb_array, q_value_rgb_array, axis=1)
        else:
            super().render(mode)


# TRY NOT TO MODIFY: setup the environment
griddly_gdy_filename = args.griddly_gdy_file
griddly_level = args.griddly_level

wrapper = GymWrapperFactory()
name = os.path.basename(griddly_gdy_filename).replace('.yaml', '')
env_name = f'Griddly-{name}-{griddly_level}'

experiment_name = f"{env_name}__{args.exp_name}__{args.seed}__{int(time.time())}"
writer = SummaryWriter(f"runs/{experiment_name}")
writer.add_text('hyperparameters', "|param|value|\n|-|-|\n%s" % (
    '\n'.join([f"|{key}|{value}|" for key, value in vars(args).items()])))
if args.prod_mode:
    import wandb

    wandb.init(project=args.wandb_project_name, entity=args.wandb_entity, sync_tensorboard=True, config=vars(args),
               name=experiment_name, monitor_gym=True, save_code=True)
    writer = SummaryWriter(f"/tmp/{experiment_name}")

# TRY NOT TO MODIFY: seeding
device = torch.device('cuda' if torch.cuda.is_available() and args.cuda else 'cpu')
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.backends.cudnn.deterministic = args.torch_deterministic

raise
wrapper.build_gym_from_yaml(
    env_name,
    griddly_gdy_filename,
    level=griddly_level,
    global_observer_type=gd.ObserverType.VECTOR,
    player_observer_type=gd.ObserverType.VECTOR,
    max_steps=128,
)


def make_env(gym_id, seed, idx):
    def thunk():

        env = gym.make(f'GDY-{gym_id}-v0')
        env.reset()

        # env = wrap_atari(env, sticky_action=args.sticky_action)
        env = gym.wrappers.RecordEpisodeStatistics(env)

        if args.capture_video:
            if idx == 0:
                env = ProbsVisualizationWrapper(env)
                env = Monitor(env, f'videos/{experiment_name}',
                              video_callable=lambda episode_id: episode_id % args.video_interval == 0)

        env.seed(seed)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        return env

    return thunk


envs = VecPyTorch(DummyVecEnv([make_env(env_name, args.seed + i, i) for i in range(args.num_envs)]), device)
raise

# some important useful layers for generic learning
class Scale(nn.Module):
    def __init__(self, scale):
        super().__init__()
        self.scale = scale

    def forward(self, x):
        return x * self.scale


class GlobalAvePool(nn.Module):

    def __init__(self, final_channels):
        super().__init__()
        self._final_channels = final_channels
        self._pool = nn.Sequential(
            nn.AdaptiveAvgPool3d((final_channels, 1, 1)),
            nn.Flatten(),
        )

    def forward(self, input):
        return self._pool(input)


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    def __init__(self, num_objects):
        super(Agent, self).__init__()
        self.network = nn.Sequential(
            layer_init(nn.Conv2d(num_objects, 32, 3, padding=1)),
            nn.ReLU(),
            layer_init(nn.Conv2d(32, 64, 3, padding=1)),
            nn.ReLU(),
            layer_init(nn.Conv2d(64, 64, 3, padding=1)),
            nn.ReLU(),
            layer_init(nn.Conv2d(64, 64, 3, padding=1)),
            nn.ReLU(),
            GlobalAvePool(2048),
            layer_init(nn.Linear(2048, 512)),
            nn.ReLU(),
            layer_init(nn.Linear(512, 512))
        )
        self.extra_layer = nn.Sequential(
            layer_init(nn.Linear(512, 512), std=0.1),
            nn.ReLU()
        )
        self.actor = nn.Sequential(
            layer_init(nn.Linear(512, 512), std=0.01),
            nn.ReLU(),
            layer_init(nn.Linear(512, envs.action_space.n), std=0.01))
        self.critic_ext = layer_init(nn.Linear(512, 1), std=0.01)
        self.critic_int = layer_init(nn.Linear(512, 1), std=0.01)

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
        return self.critic_ext(self.extra_layer(features) + features), self.critic_int(
            self.extra_layer(features) + features)


class RNDModel(nn.Module):
    def __init__(self, num_objects):
        super(RNDModel, self).__init__()

        # Prediction network
        self.predictor = nn.Sequential(
            layer_init(nn.Conv2d(num_objects, 32, 3, padding=1)),
            nn.ReLU(),
            layer_init(nn.Conv2d(32, 64, 3, padding=1)),
            nn.ReLU(),
            layer_init(nn.Conv2d(64, 64, 3, padding=1)),
            nn.ReLU(),
            layer_init(nn.Conv2d(64, 64, 3, padding=1)),
            nn.ReLU(),
            GlobalAvePool(2048),
            layer_init(nn.Linear(2048, 512)),
            nn.ReLU(),
            layer_init(nn.Linear(512, 512))
        )

        # Target network
        self.target = nn.Sequential(
            layer_init(nn.Conv2d(num_objects, 32, 3, padding=1)),
            nn.ReLU(),
            layer_init(nn.Conv2d(32, 64, 3, padding=1)),
            nn.ReLU(),
            layer_init(nn.Conv2d(64, 64, 3, padding=1)),
            nn.ReLU(),
            layer_init(nn.Conv2d(64, 64, 3, padding=1)),
            nn.ReLU(),
            GlobalAvePool(2048),
            layer_init(nn.Linear(2048, 512)),
            nn.ReLU(),
            layer_init(nn.Linear(512, 512))
        )

        # Set that target network is not trainable
        for param in self.target.parameters():
            param.requires_grad = False

    def forward(self, next_obs):
        target_feature = self.target(next_obs)
        predict_feature = self.predictor(next_obs)

        return predict_feature, target_feature


# The number of objects in the griddly environment is the size of the channels in the VECTOR observer
num_objects = envs.observation_space.shape[0]

agent = Agent(num_objects).to(device)

rnd_model = RNDModel(num_objects).to(device)

optimizer = optim.Adam(list(agent.parameters()) + list(rnd_model.predictor.parameters()), lr=args.learning_rate,
                       eps=1e-5)
if args.anneal_lr:
    # https://github.com/openai/baselines/blob/ea25b9e8b234e6ee1bca43083f8f3cf974143998/baselines/ppo2/defaults.py#L20
    lr = lambda f: f * args.learning_rate

reward_rms = RunningMeanStd()
obs_rms = RunningMeanStd(shape=(1, *envs.observation_space.shape))

discounted_reward = RewardForwardFilter(args.int_gamma)

# ALGO Logic: Storage for epoch data
obs = torch.zeros((args.num_steps, args.num_envs) + envs.observation_space.shape).to(device)
actions = torch.zeros((args.num_steps, args.num_envs)).to(device)
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

for update in range(1, num_updates + 1):
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
        rnd_next_obs = torch.FloatTensor(
            ((next_obs.data.cpu().numpy() - obs_rms.mean) / np.sqrt(obs_rms.var)).clip(-5, 5)).to(device)
        target_next_feature = rnd_model.target(rnd_next_obs)
        predict_next_feature = rnd_model.predictor(rnd_next_obs)
        curiosity_rewards[step] = ((target_next_feature - predict_next_feature).pow(2).sum(1) / 2).data.cpu()

        for idx, info in enumerate(infos):
            if 'episode' in info.keys():
                print(
                    f"global_step={global_step}, episode_reward={info['episode']['r']}, curiosity_reward={curiosity_rewards[step][idx]}")
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
                    ext_nextnonterminal = 1.0 - dones[t + 1]
                    int_nextnonterminal = 1.0
                    ext_nextvalues = ext_values[t + 1]
                    int_nextvalues = int_values[t + 1]
                ext_delta = rewards[t] + args.gamma * ext_nextvalues * ext_nextnonterminal - ext_values[t]
                int_delta = curiosity_rewards[t] + args.int_gamma * int_nextvalues * int_nextnonterminal - int_values[t]
                ext_advantages[
                    t] = ext_lastgaelam = ext_delta + args.gamma * args.gae_lambda * ext_nextnonterminal * ext_lastgaelam
                int_advantages[
                    t] = int_lastgaelam = int_delta + args.int_gamma * args.gae_lambda * int_nextnonterminal * int_lastgaelam
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
                    ext_nextnonterminal = 1.0 - dones[t + 1]
                    int_nextnonterminal = 1.0
                    ext_next_return = ext_returns[t + 1]
                    int_next_return = int_returns[t + 1]
                ext_returns[t] = rewards[t] + args.gamma * ext_nextnonterminal * ext_next_return
                int_returns[t] = curiosity_rewards[t] + args.int_gamma * int_nextnonterminal * int_next_return
            ext_advantages = ext_returns - ext_values
            int_advantages = int_returns - int_values

    # flatten the batch
    b_obs = obs.reshape((-1,) + envs.observation_space.shape)
    b_logprobs = logprobs.reshape(-1)
    b_actions = actions.reshape((-1))
    b_ext_advantages = ext_advantages.reshape(-1)
    b_int_advantages = int_advantages.reshape(-1)
    b_ext_returns = ext_returns.reshape(-1)
    b_int_returns = int_returns.reshape(-1)
    b_ext_values = ext_values.reshape(-1)

    b_advantages = b_int_advantages * args.int_coef + b_ext_advantages * args.ext_coef

    obs_rms.update(b_obs.data.cpu().numpy())

    # Optimizaing the policy and value network
    forward_mse = nn.MSELoss(reduction='none')
    target_agent = Agent(num_objects).to(device)
    inds = np.arange(args.batch_size, )

    rnd_next_obs = torch.FloatTensor(((b_obs.data.cpu().numpy() - obs_rms.mean) / np.sqrt(obs_rms.var)).clip(-5, 5)).to(
        device)

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
            pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
            pg_loss = torch.max(pg_loss1, pg_loss2).mean()
            entropy_loss = entropy.mean()

            # Value loss
            new_ext_values, new_int_values = agent.get_value(b_obs[minibatch_ind])
            new_ext_values, new_int_values = new_ext_values.view(-1), new_int_values.view(-1)
            if args.clip_vloss:
                ext_v_loss_unclipped = ((new_ext_values - b_ext_returns[minibatch_ind]) ** 2)
                ext_v_clipped = b_ext_values[minibatch_ind] + torch.clamp(new_ext_values - b_ext_values[minibatch_ind],
                                                                          -args.clip_coef, args.clip_coef)
                ext_v_loss_clipped = (ext_v_clipped - b_ext_returns[minibatch_ind]) ** 2
                ext_v_loss_max = torch.max(ext_v_loss_unclipped, ext_v_loss_clipped)
                ext_v_loss = 0.5 * ext_v_loss_max.mean()
            else:
                ext_v_loss = 0.5 * ((new_ext_values - b_ext_returns[minibatch_ind]) ** 2).mean()

            int_v_loss = 0.5 * ((new_int_values - b_int_returns[minibatch_ind]) ** 2).mean()
            v_loss = ext_v_loss + int_v_loss

            optimizer.zero_grad()
            loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef + forward_loss

            loss.backward()
            nn.utils.clip_grad_norm_(list(agent.parameters()) + list(rnd_model.predictor.parameters()),
                                     args.max_grad_norm)
            optimizer.step()

        if args.kle_stop:
            if approx_kl > args.target_kl:
                break
        if args.kle_rollback:
            if (b_logprobs[minibatch_ind] - agent.get_action(b_obs[minibatch_ind], b_actions.long()[minibatch_ind])[
                1]).mean() > args.target_kl:
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