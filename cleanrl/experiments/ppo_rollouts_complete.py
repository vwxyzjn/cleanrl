# Primary reference: https://github.com/zhangchuheng123/Reinforcement-Implementation/blob/master/code/ppo.py

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions.normal import Normal
from torch.utils.tensorboard import SummaryWriter

from cleanrl.common import preprocess_obs_space, preprocess_ac_space
import argparse
import numpy as np
import gym
from gym.wrappers import TimeLimit, Monitor
from gym.vector import SyncVectorEnv, AsyncVectorEnv
import pybullet_envs
from gym.spaces import Discrete, Box, MultiBinary, MultiDiscrete, Space
import time
import random
import os

from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
from baselines import bench
from baselines.common.vec_env import VecEnvWrapper
from baselines.common.vec_env.dummy_vec_env import DummyVecEnv
from baselines.common.vec_env.shmem_vec_env import ShmemVecEnv
from baselines.common.vec_env.vec_normalize import \
    VecNormalize as VecNormalize_

def _flatten_helper(T, N, _tensor):
    return _tensor.view(T * N, *_tensor.size()[2:])


class RolloutStorage(object):
    def __init__(self, num_steps, num_processes, obs_shape, action_space,
                 recurrent_hidden_state_size):
        self.obs = torch.zeros(num_steps + 1, num_processes, *obs_shape)
        self.recurrent_hidden_states = torch.zeros(
            num_steps + 1, num_processes, recurrent_hidden_state_size)
        self.rewards = torch.zeros(num_steps, num_processes, 1)
        self.value_preds = torch.zeros(num_steps + 1, num_processes, 1)
        self.returns = torch.zeros(num_steps + 1, num_processes, 1)
        self.action_log_probs = torch.zeros(num_steps, num_processes, 1)
        if action_space.__class__.__name__ == 'Discrete':
            action_shape = 1
        else:
            action_shape = action_space.shape[0]
        self.actions = torch.zeros(num_steps, num_processes, action_shape)
        if action_space.__class__.__name__ == 'Discrete':
            self.actions = self.actions.long()
        self.masks = torch.ones(num_steps + 1, num_processes, 1)

        # Masks that indicate whether it's a true terminal state
        # or time limit end state
        self.bad_masks = torch.ones(num_steps + 1, num_processes, 1)

        self.num_steps = num_steps
        self.step = 0

    def to(self, device):
        self.obs = self.obs.to(device)
        self.recurrent_hidden_states = self.recurrent_hidden_states.to(device)
        self.rewards = self.rewards.to(device)
        self.value_preds = self.value_preds.to(device)
        self.returns = self.returns.to(device)
        self.action_log_probs = self.action_log_probs.to(device)
        self.actions = self.actions.to(device)
        self.masks = self.masks.to(device)
        self.bad_masks = self.bad_masks.to(device)

    def insert(self, obs, recurrent_hidden_states, actions, action_log_probs,
               value_preds, rewards, masks, bad_masks):
        self.obs[self.step + 1].copy_(obs)
        # self.recurrent_hidden_states[self.step +
        #                              1].copy_(recurrent_hidden_states)
        self.actions[self.step].copy_(actions)
        self.action_log_probs[self.step].copy_(action_log_probs)
        self.value_preds[self.step].copy_(value_preds)
        self.rewards[self.step].copy_(rewards)
        self.masks[self.step + 1].copy_(masks)
        self.bad_masks[self.step + 1].copy_(bad_masks)

        self.step = (self.step + 1) % self.num_steps

    def after_update(self):
        self.obs[0].copy_(self.obs[-1])
        self.recurrent_hidden_states[0].copy_(self.recurrent_hidden_states[-1])
        self.masks[0].copy_(self.masks[-1])
        self.bad_masks[0].copy_(self.bad_masks[-1])

    def compute_returns(self,
                        next_value,
                        use_gae,
                        gamma,
                        gae_lambda,
                        use_proper_time_limits=True):
        if use_proper_time_limits:
            if use_gae:
                self.value_preds[-1] = next_value
                gae = 0
                for step in reversed(range(self.rewards.size(0))):
                    delta = self.rewards[step] + gamma * self.value_preds[
                        step + 1] * self.masks[step +
                                               1] - self.value_preds[step]
                    gae = delta + gamma * gae_lambda * self.masks[step +
                                                                  1] * gae
                    gae = gae * self.bad_masks[step + 1]
                    self.returns[step] = gae + self.value_preds[step]
            else:
                self.returns[-1] = next_value
                for step in reversed(range(self.rewards.size(0))):
                    self.returns[step] = (self.returns[step + 1] * \
                        gamma * self.masks[step + 1] + self.rewards[step]) * self.bad_masks[step + 1] \
                        + (1 - self.bad_masks[step + 1]) * self.value_preds[step]
        else:
            if use_gae:
                self.value_preds[-1] = next_value
                gae = 0
                for step in reversed(range(self.rewards.size(0))):
                    delta = self.rewards[step] + gamma * self.value_preds[
                        step + 1] * self.masks[step +
                                               1] - self.value_preds[step]
                    gae = delta + gamma * gae_lambda * self.masks[step +
                                                                  1] * gae
                    self.returns[step] = gae + self.value_preds[step]
            else:
                self.returns[-1] = next_value
                for step in reversed(range(self.rewards.size(0))):
                    self.returns[step] = self.returns[step + 1] * \
                        gamma * self.masks[step + 1] + self.rewards[step]

    def feed_forward_generator(self,
                               advantages,
                               num_mini_batch=None,
                               mini_batch_size=None):
        num_steps, num_processes = self.rewards.size()[0:2]
        batch_size = num_processes * num_steps

        if mini_batch_size is None:
            assert batch_size >= num_mini_batch, (
                "PPO requires the number of processes ({}) "
                "* number of steps ({}) = {} "
                "to be greater than or equal to the number of PPO mini batches ({})."
                "".format(num_processes, num_steps, num_processes * num_steps,
                          num_mini_batch))
            mini_batch_size = batch_size // num_mini_batch
        sampler = BatchSampler(
            SubsetRandomSampler(range(batch_size)),
            mini_batch_size,
            drop_last=True)
        for indices in sampler:
            obs_batch = self.obs[:-1].view(-1, *self.obs.size()[2:])[indices]
            recurrent_hidden_states_batch = self.recurrent_hidden_states[:-1].view(
                -1, self.recurrent_hidden_states.size(-1))[indices]
            actions_batch = self.actions.view(-1,
                                              self.actions.size(-1))[indices]
            value_preds_batch = self.value_preds[:-1].view(-1, 1)[indices]
            return_batch = self.returns[:-1].view(-1, 1)[indices]
            masks_batch = self.masks[:-1].view(-1, 1)[indices]
            old_action_log_probs_batch = self.action_log_probs.view(-1,
                                                                    1)[indices]
            if advantages is None:
                adv_targ = None
            else:
                adv_targ = advantages.view(-1, 1)[indices]

            yield obs_batch, recurrent_hidden_states_batch, actions_batch, \
                value_preds_batch, return_batch, masks_batch, old_action_log_probs_batch, adv_targ

    def recurrent_generator(self, advantages, num_mini_batch):
        num_processes = self.rewards.size(1)
        assert num_processes >= num_mini_batch, (
            "PPO requires the number of processes ({}) "
            "to be greater than or equal to the number of "
            "PPO mini batches ({}).".format(num_processes, num_mini_batch))
        num_envs_per_batch = num_processes // num_mini_batch
        perm = torch.randperm(num_processes)
        for start_ind in range(0, num_processes, num_envs_per_batch):
            obs_batch = []
            recurrent_hidden_states_batch = []
            actions_batch = []
            value_preds_batch = []
            return_batch = []
            masks_batch = []
            old_action_log_probs_batch = []
            adv_targ = []

            for offset in range(num_envs_per_batch):
                ind = perm[start_ind + offset]
                obs_batch.append(self.obs[:-1, ind])
                recurrent_hidden_states_batch.append(
                    self.recurrent_hidden_states[0:1, ind])
                actions_batch.append(self.actions[:, ind])
                value_preds_batch.append(self.value_preds[:-1, ind])
                return_batch.append(self.returns[:-1, ind])
                masks_batch.append(self.masks[:-1, ind])
                old_action_log_probs_batch.append(
                    self.action_log_probs[:, ind])
                adv_targ.append(advantages[:, ind])

            T, N = self.num_steps, num_envs_per_batch
            # These are all tensors of size (T, N, -1)
            obs_batch = torch.stack(obs_batch, 1)
            actions_batch = torch.stack(actions_batch, 1)
            value_preds_batch = torch.stack(value_preds_batch, 1)
            return_batch = torch.stack(return_batch, 1)
            masks_batch = torch.stack(masks_batch, 1)
            old_action_log_probs_batch = torch.stack(
                old_action_log_probs_batch, 1)
            adv_targ = torch.stack(adv_targ, 1)

            # States is just a (N, -1) tensor
            recurrent_hidden_states_batch = torch.stack(
                recurrent_hidden_states_batch, 1).view(N, -1)

            # Flatten the (T, N, ...) tensors to (T * N, ...)
            obs_batch = _flatten_helper(T, N, obs_batch)
            actions_batch = _flatten_helper(T, N, actions_batch)
            value_preds_batch = _flatten_helper(T, N, value_preds_batch)
            return_batch = _flatten_helper(T, N, return_batch)
            masks_batch = _flatten_helper(T, N, masks_batch)
            old_action_log_probs_batch = _flatten_helper(T, N, \
                    old_action_log_probs_batch)
            adv_targ = _flatten_helper(T, N, adv_targ)

            yield obs_batch, recurrent_hidden_states_batch, actions_batch, \
                value_preds_batch, return_batch, masks_batch, old_action_log_probs_batch, adv_targ


def make_env(env_id, seed, rank, log_dir, allow_early_resets):
    def _thunk():
        if env_id.startswith("dm"):
            _, domain, task = env_id.split('.')
            env = dm_control2gym.make(domain_name=domain, task_name=task)
        else:
            env = gym.make(env_id)

        is_atari = hasattr(gym.envs, 'atari') and isinstance(
            env.unwrapped, gym.envs.atari.atari_env.AtariEnv)
        if is_atari:
            env = make_atari(env_id)

        env.seed(seed + rank)

        if str(env.__class__.__name__).find('TimeLimit') >= 0:
            env = TimeLimitMask(env)

        if log_dir is not None:
            env = bench.Monitor(
                env,
                os.path.join(log_dir, str(rank)),
                allow_early_resets=allow_early_resets)

        if is_atari:
            if len(env.observation_space.shape) == 3:
                env = wrap_deepmind(env)
        elif len(env.observation_space.shape) == 3:
            raise NotImplementedError(
                "CNN models work only for atari,\n"
                "please use a custom wrapper for a custom pixel input env.\n"
                "See wrap_deepmind for an example.")

        # If the input has shape (W,H,3), wrap for PyTorch convolutions
        obs_shape = env.observation_space.shape
        if len(obs_shape) == 3 and obs_shape[2] in [1, 3]:
            env = TransposeImage(env, op=[2, 0, 1])

        return env

    return _thunk


def make_vec_envs(env_name,
                  seed,
                  num_processes,
                  gamma,
                  log_dir,
                  device,
                  allow_early_resets,
                  num_frame_stack=None):
    envs = [
        make_env(env_name, seed, i, log_dir, allow_early_resets)
        for i in range(num_processes)
    ]

    if len(envs) > 1:
        envs = ShmemVecEnv(envs, context='fork')
    else:
        envs = DummyVecEnv(envs)

    if len(envs.observation_space.shape) == 1:
        if gamma is None:
            envs = VecNormalize(envs, ret=False)
        else:
            envs = VecNormalize(envs, gamma=gamma)

    envs = VecPyTorch(envs, device)

    if num_frame_stack is not None:
        envs = VecPyTorchFrameStack(envs, num_frame_stack, device)
    elif len(envs.observation_space.shape) == 3:
        envs = VecPyTorchFrameStack(envs, 4, device)

    return envs


# Checks whether done was caused my timit limits or not
class TimeLimitMask(gym.Wrapper):
    def step(self, action):
        obs, rew, done, info = self.env.step(action)
        if done and self.env._max_episode_steps == self.env._elapsed_steps:
            info['bad_transition'] = True

        return obs, rew, done, info

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)


# Can be used to test recurrent policies for Reacher-v2
class MaskGoal(gym.ObservationWrapper):
    def observation(self, observation):
        if self.env._elapsed_steps > 0:
            observation[-2:] = 0
        return observation


class TransposeObs(gym.ObservationWrapper):
    def __init__(self, env=None):
        """
        Transpose observation space (base class)
        """
        super(TransposeObs, self).__init__(env)


class TransposeImage(TransposeObs):
    def __init__(self, env=None, op=[2, 0, 1]):
        """
        Transpose observation space for images
        """
        super(TransposeImage, self).__init__(env)
        assert len(op) == 3, "Error: Operation, " + str(op) + ", must be dim3"
        self.op = op
        obs_shape = self.observation_space.shape
        self.observation_space = Box(
            self.observation_space.low[0, 0, 0],
            self.observation_space.high[0, 0, 0], [
                obs_shape[self.op[0]], obs_shape[self.op[1]],
                obs_shape[self.op[2]]
            ],
            dtype=self.observation_space.dtype)

    def observation(self, ob):
        return ob.transpose(self.op[0], self.op[1], self.op[2])


class VecPyTorch(VecEnvWrapper):
    def __init__(self, venv, device):
        """Return only every `skip`-th frame"""
        super(VecPyTorch, self).__init__(venv)
        self.device = device
        # TODO: Fix data types

    def reset(self):
        obs = self.venv.reset()
        obs = torch.from_numpy(obs).float().to(self.device)
        return obs

    def step_async(self, actions):
        if isinstance(actions, torch.LongTensor):
            # Squeeze the dimension for discrete actions
            actions = actions.squeeze(1)
        actions = actions.cpu().numpy()
        self.venv.step_async(actions)

    def step_wait(self):
        obs, reward, done, info = self.venv.step_wait()
        obs = torch.from_numpy(obs).float().to(self.device)
        reward = torch.from_numpy(reward).unsqueeze(dim=1).float()
        return obs, reward, done, info


class VecNormalize(VecNormalize_):
    def __init__(self, *args, **kwargs):
        super(VecNormalize, self).__init__(*args, **kwargs)
        self.training = True

    def _obfilt(self, obs, update=True):
        if self.ob_rms:
            if self.training and update:
                self.ob_rms.update(obs)
            obs = np.clip((obs - self.ob_rms.mean) /
                          np.sqrt(self.ob_rms.var + self.epsilon),
                          -self.clipob, self.clipob)
            return obs
        else:
            return obs

    def train(self):
        self.training = True

    def eval(self):
        self.training = False


# Derived from
# https://github.com/openai/baselines/blob/master/baselines/common/vec_env/vec_frame_stack.py
class VecPyTorchFrameStack(VecEnvWrapper):
    def __init__(self, venv, nstack, device=None):
        self.venv = venv
        self.nstack = nstack

        wos = venv.observation_space  # wrapped ob space
        self.shape_dim0 = wos.shape[0]

        low = np.repeat(wos.low, self.nstack, axis=0)
        high = np.repeat(wos.high, self.nstack, axis=0)

        if device is None:
            device = torch.device('cpu')
        self.stacked_obs = torch.zeros((venv.num_envs, ) +
                                       low.shape).to(device)

        observation_space = gym.spaces.Box(
            low=low, high=high, dtype=venv.observation_space.dtype)
        VecEnvWrapper.__init__(self, venv, observation_space=observation_space)

    def step_wait(self):
        obs, rews, news, infos = self.venv.step_wait()
        self.stacked_obs[:, :-self.shape_dim0] = \
            self.stacked_obs[:, self.shape_dim0:].clone()
        for (i, new) in enumerate(news):
            if new:
                self.stacked_obs[i] = 0
        self.stacked_obs[:, -self.shape_dim0:] = obs
        return self.stacked_obs, rews, news, infos

    def reset(self):
        obs = self.venv.reset()
        if torch.backends.cudnn.deterministic:
            self.stacked_obs = torch.zeros(self.stacked_obs.shape)
        else:
            self.stacked_obs.zero_()
        self.stacked_obs[:, -self.shape_dim0:] = obs
        return self.stacked_obs

    def close(self):
        self.venv.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PPO agent')
    # Common arguments
    parser.add_argument('--exp-name', type=str, default=os.path.basename(__file__).rstrip(".py"),
                       help='the name of this experiment')
    parser.add_argument('--gym-id', type=str, default="HopperBulletEnv-v0",
                       help='the id of the gym environment')
    parser.add_argument('--learning-rate', type=float, default=7e-4,
                       help='the learning rate of the optimizer')
    parser.add_argument('--seed', type=int, default=1,
                       help='seed of the experiment')
    parser.add_argument('--episode-length', type=int, default=0,
                       help='the maximum length of each episode')
    parser.add_argument('--total-timesteps', type=int, default=100000,
                       help='total timesteps of the experiments')
    parser.add_argument('--no-torch-deterministic', action='store_false', dest="torch_deterministic", default=True,
                       help='if toggled, `torch.backends.cudnn.deterministic=False`')
    parser.add_argument('--no-cuda', action='store_false', dest="cuda", default=True,
                       help='if toggled, cuda will not be enabled by default')
    parser.add_argument('--prod-mode', action='store_true', default=False,
                       help='run the script in production mode and use wandb to log outputs')
    parser.add_argument('--capture-video', action='store_true', default=False,
                       help='weather to capture videos of the agent performances (check out `videos` folder)')
    parser.add_argument('--wandb-project-name', type=str, default="cleanRL",
                       help="the wandb's project name")
    parser.add_argument('--wandb-entity', type=str, default=None,
                       help="the entity (team) of wandb's project")

    # Algorithm specific arguments
    parser.add_argument('--gamma', type=float, default=0.99,
                       help='the discount factor gamma')
    parser.add_argument('--vf-coef', type=float, default=0.25,
                       help="value function's coefficient the loss function")
    parser.add_argument('--max-grad-norm', type=float, default=0.5,
                       help='the maximum norm for the gradient clipping')
    parser.add_argument('--ent-coef', type=float, default=0.01,
                       help="policy entropy's coefficient the loss function")
    parser.add_argument('--clip-coef', type=float, default=0.2,
                       help="the surrogate clipping coefficient")
    parser.add_argument('--update-epochs', type=int, default=4,
                        help="the K epochs to update the policy")
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
    wandb.init(project=args.wandb_project_name, entity=args.wandb_entity, tensorboard=True, config=vars(args), name=experiment_name, monitor_gym=True)
    writer = SummaryWriter(f"/tmp/{experiment_name}")
    wandb.save(os.path.abspath(__file__))

# TRY NOT TO MODIFY: seeding
device = torch.device('cpu')
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.backends.cudnn.deterministic = args.torch_deterministic
envs = make_vec_envs(args.gym_id, args.seed, 1, args.gamma, "/tmp/gym/", device, True)
input_shape, preprocess_obs_fn = preprocess_obs_space(envs.observation_space, device)
output_shape = preprocess_ac_space(envs.action_space)
# # respect the default timelimit
# assert isinstance(env, TimeLimit) or int(args.episode_length), "the gym env does not have a built in TimeLimit, please specify by using --episode-length"
# if isinstance(env, TimeLimit):
#     if int(args.episode_length):
#         env._max_episode_steps = int(args.episode_length)
#     args.episode_length = env._max_episode_steps
# else:
#     env = TimeLimit(env, int(args.episode_length))
# if args.capture_video:
#     env = Monitor(env, f'videos/{experiment_name}')

class Policy(nn.Module):
    def __init__(self):
        super(Policy, self).__init__()
        torch.manual_seed(2)
        torch.cuda.manual_seed_all(2)
        torch.backends.cudnn.deterministic = True
        self.fc1 = nn.Linear(input_shape, 64)
        self.fc2 = nn.Linear(64, 64)
        torch.manual_seed(3)
        torch.cuda.manual_seed_all(3)
        torch.backends.cudnn.deterministic = True
        self.mean = nn.Linear(64, output_shape)
        self.logstd = nn.Parameter(torch.zeros(1, output_shape))

    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        action_mean = self.mean(x)
        action_logstd = self.logstd.expand_as(action_mean)
        return action_mean, action_logstd

    def get_action(self, x):
        mean, logstd = self.forward(x)
        std = torch.exp(logstd)
        probs = Normal(mean, std)
        # torch.manual_seed(4)
        # torch.cuda.manual_seed_all(4)
        action = probs.sample()
        return action, probs.log_prob(action).sum(1)

    def get_logproba(self, x, actions):
        action_mean, action_logstd = self.forward(x)
        action_std = torch.exp(action_logstd)
        dist = Normal(action_mean, action_std)
        return dist.log_prob(actions).sum(1)

class Value(nn.Module):
    def __init__(self):
        super(Value, self).__init__()
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = args.torch_deterministic
        self.fc1 = nn.Linear(input_shape, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 1)

    def forward(self, x):
        x = F.tanh(self.fc1(x))
        x = F.tanh(self.fc2(x))
        x = self.fc3(x)
        return x
pg = Policy().to(device)
vf = Value().to(device)
optimizer = optim.Adam(list(pg.parameters()) + list(vf.parameters()), lr=args.learning_rate)
loss_fn = nn.MSELoss()
#print(pg.logstd.bias)
# TRY NOT TO MODIFY: start the game
num_steps = 5
global_step = 0
episode_rewards = []
rollouts = RolloutStorage(num_steps, 1,
                          envs.observation_space.shape, envs.action_space,
                          1)
obs = envs.reset()
rollouts.obs[0].copy_(obs)
rollouts.to(device)
while global_step < args.total_timesteps:
    
    for step in range(num_steps):
        # Sample actions
        with torch.no_grad():
            value = vf.forward(rollouts.obs[step])
            action, action_log_prob = pg.get_action(rollouts.obs[step])
            # print("sample", action)

        # Obser reward and next obs
        obs, reward, done, infos = envs.step(action)

        for info in infos:
            if 'episode' in info.keys():
                print(f"global_step={global_step}, episode_reward={info['episode']['r']}")
                

        # If done then clean the history of observations.
        masks = torch.FloatTensor(
            [[0.0] if done_ else [1.0] for done_ in done])
        bad_masks = torch.FloatTensor(
            [[0.0] if 'bad_transition' in info.keys() else [1.0]
             for info in infos])
        rollouts.insert(obs, None, action,
                        action_log_prob, value, reward, masks, bad_masks)
        # print("rollouts.value_preds", rollouts.value_preds)
        global_step += 1

    with torch.no_grad():
        next_value = vf.forward(rollouts.obs[-1])

    rollouts.compute_returns(next_value, False, args.gamma,
                             0.95, False)
    
    
    advantages = rollouts.returns[:-1] - rollouts.value_preds[:-1]
    advantages = advantages.detach()
    advantages = (advantages - advantages.mean()) / (
        advantages.std() + 1e-5)
    # print("advantages", advantages)


    for _ in range(args.update_epochs):
        data_generator = rollouts.feed_forward_generator(
            advantages, 1)
        for sample in data_generator:
            obs_batch, recurrent_hidden_states_batch, actions_batch, \
                value_preds_batch, return_batch, masks_batch, old_action_log_probs_batch, \
                    adv_targ = sample
            newlogproba = pg.get_logproba(obs_batch, actions_batch)
            ratio = torch.exp(newlogproba - old_action_log_probs_batch.view(-1))
            surrogate1 = ratio * adv_targ.flatten()
            surrogate2 = ratio.clamp(1 - args.clip_coef, 1 + args.clip_coef) * adv_targ.flatten()
            policy_loss = -torch.mean(torch.min(surrogate1, surrogate2))
            newvalues = vf.forward(obs_batch).flatten()
            vf_loss = torch.mean((newvalues - return_batch.view(-1).detach()).pow(2))
            entropy_loss = torch.mean(torch.exp(newlogproba) * newlogproba)
            total_loss = policy_loss + args.vf_coef * vf_loss + args.ent_coef * entropy_loss
            optimizer.zero_grad()
            total_loss.backward() # total_loss.backward()
            nn.utils.clip_grad_norm_(list(pg.parameters()) + list(vf.parameters()), args.max_grad_norm)
            optimizer.step()
        
    rollouts.after_update()


    # # TRY NOT TO MODIFY: record rewards for plotting purposes
    # writer.add_scalar("charts/episode_reward", rewards.sum(), global_step)
    # writer.add_scalar("losses/value_loss", vf_loss.item(), global_step)
    # writer.add_scalar("losses/entropy", entropys[:step+1].mean().item(), global_step)
    # writer.add_scalar("losses/policy_loss", policy_loss.item(), global_step)
envs.close()
writer.close()