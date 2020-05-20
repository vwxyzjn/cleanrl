# Reference: https://arxiv.org/pdf/1509.02971.pdf
# https://github.com/seungeunrho/minimalRL/blob/master/ddpg.py
# https://github.com/openai/spinningup/blob/master/spinup/algos/pytorch/ddpg/ddpg.py

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from cleanrl.common import preprocess_obs_space, preprocess_ac_space
import argparse
from distutils.util import strtobool
import collections
import numpy as np
import gym
import pybullet_envs
from gym.wrappers import TimeLimit, Monitor
from gym.spaces import Discrete, Box, MultiBinary, MultiDiscrete, Space
import time
import random
import os

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='DDPG agent')
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
    parser.add_argument('--total-timesteps', type=int, default=1000000,
                        help='total timesteps of the experiments')
    parser.add_argument('--torch-deterministic', type=lambda x:bool(strtobool(x)), default=True, nargs='?', const=True,
                        help='if toggled, `torch.backends.cudnn.deterministic=False`')
    parser.add_argument('--cuda', type=lambda x:bool(strtobool(x)), default=True, nargs='?', const=True,
                        help='if toggled, cuda will not be enabled by default')
    parser.add_argument('--prod-mode', type=lambda x:bool(strtobool(x)), default=True, nargs='?', const=True,
                        help='run the script in production mode and use wandb to log outputs')
    parser.add_argument('--capture-video', type=lambda x:bool(strtobool(x)), default=True, nargs='?', const=True,
                        help='weather to capture videos of the agent performances (check out `videos` folder)')
    parser.add_argument('--wandb-project-name', type=str, default="cleanRL",
                        help="the wandb's project name")
    parser.add_argument('--wandb-entity', type=str, default=None,
                        help="the entity (team) of wandb's project")
    
    # Algorithm specific arguments
    parser.add_argument('--buffer-size', type=int, default=int(1e6),
                         help='the replay memory buffer size')
    parser.add_argument('--gamma', type=float, default=0.99,
                        help='the discount factor gamma')
    parser.add_argument('--tau', type=float, default=0.005,
                        help="target smoothing coefficient (default: 0.005)")
    parser.add_argument('--max-grad-norm', type=float, default=0.5,
                        help='the maximum norm for the gradient clipping')
    parser.add_argument('--batch-size', type=int, default=256, # TODO: major discrepency with original
                        help="the batch size of sample from the reply memory")
    parser.add_argument('--action-noise', default="normal", choices=["ou", 'normal'],
                         help='Selects the scheme to be used for weights initialization'),
    parser.add_argument('--start-sigma', type=float, default=0.2,
                        help="the start standard deviation of the action noise for exploration")
    parser.add_argument('--end-sigma', type=float, default=0.05,
                        help="the ending standard deviation of the action noise for exploration")
    parser.add_argument('--exploration-fraction', type=float, default=0.8,
                        help="the fraction of `total-timesteps` it takes from start-sigma to go end-sigma")
    parser.add_argument('--learning-starts', type=int, default=25e3, # TODO: major discrepency with original
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
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.backends.cudnn.deterministic = args.torch_deterministic
env.seed(args.seed)
env.action_space.seed(args.seed)
env.observation_space.seed(args.seed)
input_shape, preprocess_obs_fn = preprocess_obs_space(env.observation_space, device)
output_shape = preprocess_ac_space(env.action_space)
# respect the default timelimit
assert isinstance(env.action_space, Box), "only continuous action space is supported"
assert isinstance(env, TimeLimit) or int(args.episode_length), "the gym env does not have a built in TimeLimit, please specify by using --episode-length"
if isinstance(env, TimeLimit):
    if int(args.episode_length):
        env._max_episode_steps = int(args.episode_length)
    args.episode_length = env._max_episode_steps
else:
    env = TimeLimit(env, int(args.episode_length))
if args.capture_video:
    env = Monitor(env, f'videos/{experiment_name}')

# taken from https://github.com/openai/baselines/blob/master/baselines/ddpg/noise.py
class NormalActionNoise():
    def __init__(self, mu, sigma=0.1):
        self.mu = mu
        self.sigma = sigma

    def __call__(self):
        return np.random.normal(self.mu, self.sigma)

    def reset(self):
        pass

    def __repr__(self):
        return 'NormalActionNoise(mu={}, sigma={})'.format(self.mu, self.sigma)

# modified from https://github.com/seungeunrho/minimalRL/blob/master/dqn.py#
class ReplayBuffer():
    def __init__(self, buffer_limit):
        self.buffer = collections.deque(maxlen=buffer_limit)
    
    def put(self, transition):
        self.buffer.append(transition)
    
    def sample(self, n):
        mini_batch = random.sample(self.buffer, n)
        s_lst, a_lst, r_lst, s_prime_lst, done_mask_lst = [], [], [], [], []
        
        for transition in mini_batch:
            s, a, r, s_prime, done_mask = transition
            s_lst.append(s)
            a_lst.append(a)
            r_lst.append(r)
            s_prime_lst.append(s_prime)
            done_mask_lst.append(done_mask)

        return np.array(s_lst), np.array(a_lst), \
               np.array(r_lst), np.array(s_prime_lst), \
               np.array(done_mask_lst)


class ReplayBuffer(object):
    def __init__(self, max_size=int(1e6)):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0

        self.state = np.zeros((max_size, state_dim))
        self.action = np.zeros((max_size, action_dim))
        self.next_state = np.zeros((max_size, state_dim))
        self.reward = np.zeros((max_size, 1))
        self.not_done = np.zeros((max_size, 1))

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    def add(self, state, action, next_state, reward, done):
        self.state[self.ptr] = state
        self.action[self.ptr] = action
        self.next_state[self.ptr] = next_state
        self.reward[self.ptr] = reward
        self.not_done[self.ptr] = 1. - done

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)


    def sample(self, batch_size):
        ind = np.random.randint(0, self.size, size=batch_size)
        # print(ind)

        return (
            torch.FloatTensor(self.state[ind]).to(self.device),
            torch.FloatTensor(self.action[ind]).to(self.device),
            torch.FloatTensor(self.next_state[ind]).to(self.device),
            torch.FloatTensor(self.reward[ind]).to(self.device),
            torch.FloatTensor(self.not_done[ind]).to(self.device)
        )

state_dim, action_dim = input_shape, output_shape
# ALGO LOGIC: initialize agent here:
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()

        self.l1 = nn.Linear(state_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, action_dim)
        
        self.max_action = max_action
        

    def forward(self, state):
        state = preprocess_obs_fn(state)
        a = F.relu(self.l1(state))
        a = F.relu(self.l2(a))
        return self.max_action * torch.tanh(self.l3(a))


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()

        # Q1 architecture
        self.l1 = nn.Linear(state_dim + action_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, 1)

        # Q2 architecture
        self.l4 = nn.Linear(state_dim + action_dim, 256)
        self.l5 = nn.Linear(256, 256)
        self.l6 = nn.Linear(256, 1)


    def forward(self, state, action):
        state = preprocess_obs_fn(state)
        sa = torch.cat([state, action], 1)

        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)

        q2 = F.relu(self.l4(sa))
        q2 = F.relu(self.l5(q2))
        q2 = self.l6(q2)
        return q1, q2


    def Q1(self, state, action):
        state = preprocess_obs_fn(state)
        sa = torch.cat([state, action], 1)

        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)
        return q1

import copy
max_action = float(env.action_space.high[0])
rb = ReplayBuffer(args.buffer_size)
actor = Actor(state_dim, action_dim, max_action).to(device)
actor_target = copy.deepcopy(actor)
actor_optimizer = torch.optim.Adam(actor.parameters(), lr=3e-4)

critic = Critic(state_dim, action_dim).to(device)
critic_target = copy.deepcopy(critic)
critic_optimizer = torch.optim.Adam(critic.parameters(), lr=3e-4)
loss_fn = nn.MSELoss()
exploration_noise = NormalActionNoise(np.zeros(output_shape))
policy_noise = NormalActionNoise(np.zeros(output_shape))
exploration_noise.sigma = 0.1
policy_noise.sigma = 0.2
exploration_noise = 0.1
policy_noise = 0.2
# TRY NOT TO MODIFY: start the game
global_step = 0
while global_step < args.total_timesteps:
    next_obs = np.array(env.reset())
    actions = np.empty((args.episode_length,), dtype=object)
    rewards, dones = np.zeros((2, args.episode_length))
    td_losses = np.zeros(args.episode_length)
    obs = np.empty((args.episode_length,) + env.observation_space.shape)
    
    # TRY NOT TO MODIFY: prepare the execution of the game.
    for step in range(args.episode_length):
        obs[step] = next_obs.copy()

        # ALGO LOGIC: put action logic here
        if global_step < args.learning_starts:
            actions[step] = env.action_space.sample()
        else:
            # TODO: major discrepency with original
            actions[step] = (actor.forward(obs[step:step+1]).tolist()[0] + 
                np.random.normal(0, max_action * exploration_noise, size=action_dim)
			).clip(-max_action, max_action)
        
        global_step += 1

        # TRY NOT TO MODIFY: execute the game and log data.
        next_obs, rewards[step], dones[step], _ = env.step(actions[step])
        # rb.put((obs[step], actions[step], rewards[step], next_obs, dones[step]))
        rb.add(obs[step], actions[step], next_obs, rewards[step], dones[step])
        next_obs = np.array(next_obs)

        # ALGO LOGIC: training.
        if global_step > args.learning_starts:
            state, action, next_state, reward, not_done = rb.sample(args.batch_size)

            with torch.no_grad():
                # Select action according to policy and add clipped noise
                noise = (
                    torch.randn_like(action) * policy_noise
                ).clamp(-args.noise_clip, args.noise_clip)
                
                next_action = (
                    actor_target(next_state) + noise
                ).clamp(-max_action, max_action)

                # Compute the target Q value
                target_Q1, target_Q2 = critic_target(next_state, next_action)
                target_Q = torch.min(target_Q1, target_Q2)
                target_Q = reward + not_done * args.gamma * target_Q

            # Get current Q estimates
            current_Q1, current_Q2 = critic(state, torch.Tensor(action).to(device))

            # Compute critic loss
            critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)
            
            # Optimize the critic
            critic_optimizer.zero_grad()
            critic_loss.backward()
            critic_optimizer.step()

            # Delayed policy updates
            if global_step % args.policy_frequency == 0:
                # Compute actor losse
                actor_loss = -critic.Q1(state, actor(state)).mean()
                
                # Optimize the actor
                actor_optimizer.zero_grad()
                actor_loss.backward()
                actor_optimizer.step()

                # Update the frozen target models
                for param, target_param in zip(critic.parameters(), critic_target.parameters()):
                    target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)

                for param, target_param in zip(actor.parameters(), actor_target.parameters()):
                    target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)

        if dones[step]:
            break

    # TRY NOT TO MODIFY: record rewards for plotting purposes
    print(f"global_step={global_step}, episode_reward={rewards.sum()}")
    writer.add_scalar("charts/episode_reward", rewards.sum(), global_step)
    writer.add_scalar("losses/td_loss", td_losses[:step+1].mean(), global_step)
env.close()
writer.close()
