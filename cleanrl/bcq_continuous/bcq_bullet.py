import os
import re
import copy
import time
import random
import argparse
import collections
import numpy as np
from distutils.util import strtobool

import gym
import pybullet_envs
import d4rl_pybullet

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from gym.wrappers import TimeLimit, Monitor
from gym.spaces import Box

parser = argparse.ArgumentParser(description='Batch Constrained Q-Learning for Continuous Domains; Uses Pybullet D4RL datasets')
# Common arguments
parser.add_argument('--exp-name', type=str, default=os.path.basename(__file__).rstrip(".py"),
					help='the name of this experiment')
parser.add_argument('--gym-id', type=str, default="hopper-bullet-medium-v0",
					help='the id of the gym environment')
parser.add_argument('--seed', type=int, default=2,
					help='seed of the experiment')
parser.add_argument('--total-timesteps', type=int, default=1000000,
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
parser.add_argument('--eval-frequency', type=int, default=2000,
					help='after how many gradient step is the model evaluated')

# Algorithm specific argumen1ts
parser.add_argument('--gamma', type=float, default=0.99,
					help='the discount factor gamma')
parser.add_argument('--lr', type=float, default=1e-3,
					help='the learning rate of the optimizer for the policy weights')
parser.add_argument('--target-network-frequency', type=int, default=1,
					help="the timesteps it takes to update the target network")
parser.add_argument('--max-grad-norm', type=float, default=0.5,
					help='the maximum norm for the gradient clipping')
parser.add_argument('--batch-size', type=int, default=100,
					help="the batch size of sample from the reply memory")
parser.add_argument('--tau', type=float, default=0.005,
					help="target smoothing coefficient (default: 0.005)")

# BQC specific parameters
parser.add_argument('--offline-dataset-id', type=str, default="expert-v0",
					help='the id of the offline dataset gym environment')
parser.add_argument('--phi', type=float, default=0.05,
					help='maximum perturbation applied over the actions sampled from the VAE Policy')
parser.add_argument('--lmbda', type=float, default=0.75,
					help='coefficient of the min Q term in the Q-network update')
parser.add_argument('--num-actions', type=int, default=10,
					help="how many actions sampled from the VAE to get the maximum value / best action")

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
torch.backends.cudnn.deterministic = args.torch_deterministic
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)

random.seed(args.seed)
np.random.seed(args.seed)

env = gym.make(args.gym_id)
env.seed(args.seed)
env.action_space.seed(args.seed)
env.observation_space.seed(args.seed)

input_shape = env.observation_space.shape[0]
output_shape = env.action_space.shape[0]
max_action = float(env.action_space.high[0])

# respect the default timelimit
assert isinstance(env.action_space, Box), "only continuous action space is supported"
if args.capture_video:
	env = Monitor(env, f'videos/{experiment_name}')

class Actor(nn.Module):
	def __init__(self, state_dim, action_dim, max_action, phi=0.05):
		super().__init__()
		self.max_action, self.phi = max_action, phi
		self.network = nn.Sequential(*[
			nn.Linear(state_dim + action_dim, 400), nn.ReLU(),
			nn.Linear(400, 300), nn.ReLU(),
			nn.Linear(300, action_dim), nn.Tanh()
		])
	
	def forward(self,state, action):
		a = self.phi * self.max_action * self.network(torch.cat([state,action], 1))
		return (a + action).clamp(-self.max_action, self.max_action)

class Critic(nn.Module):
	def __init__(self, state_dim, action_dim):
		super().__init__()
		self.network = nn.Sequential(*[
			nn.Linear(state_dim + action_dim, 400), nn.ReLU(),
			nn.Linear(400, 300), nn.ReLU(),
			nn.Linear(300,1)
		])
	
	def forward(self, state, action):
		return self.network(torch.cat([state, action], 1))

class VAE(nn.Module):
	def __init__(self, state_dim, action_dim, latent_dim, max_action):
		super().__init__()
		self.max_action, self.latent_dim, self.device = max_action, latent_dim, "cpu"
		self.encoder = nn.Sequential(*[
			nn.Linear(state_dim + action_dim, 750), nn.ReLU(),
			nn.Linear(750, 750), nn.ReLU()
		])
		self.encoder_mean = nn.Linear(750, latent_dim)
		self.encoder_log_scale = nn.Linear(750, latent_dim)
		
		self.decoder = nn.Sequential(*[
			nn.Linear(state_dim + latent_dim, 750), nn.ReLU(),
			nn.Linear(750, 750), nn.ReLU(),
			nn.Linear(750, action_dim), nn.Tanh()
		])
	
	def to(self, device, *args, **kwargs):
		super().to(device)
		self.device = device
		return self
	
	def forward(self, state, action):
		feat = self.encoder(torch.cat([state, action], 1))
		
		mean, log_scale = self.encoder_mean(feat), self.encoder_log_scale(feat).clamp(-4,15)
		scale = log_scale.exp()
		
		z = mean + scale * torch.randn_like(scale)
		
		u = self.decoder(torch.cat([state, z], 1)) * self.max_action
		
		return u, mean, scale
	
	def get_actions(self, obs):
		z = torch.randn([obs.shape[0], self.latent_dim]).to(self.device).clamp(-0.5,0.5)
		
		return self.decoder(torch.cat([obs, z],1)) * self.max_action

# ReplayBuffer from original implementation, with data loading and sampling only
class ReplayBuffer(object):
	def __init__(self, state_dim, action_dim, device, max_size=int(1e6)):
		self.max_size = max_size
		self.ptr = 0
		self.size = 0

		self.state = np.zeros((max_size, state_dim))
		self.action = np.zeros((max_size, action_dim))
		self.next_state = np.zeros((max_size, state_dim))
		self.reward = np.zeros(max_size)
		self.done = np.zeros(max_size)

		self.device = device

	def sample(self, batch_size):
		ind = np.random.randint(0, self.size, size=batch_size)

		return (
			torch.FloatTensor(self.state[ind]).to(self.device),
			torch.FloatTensor(self.action[ind]).to(self.device),
			torch.FloatTensor(self.next_state[ind]).to(self.device),
			torch.FloatTensor(self.reward[ind]).to(self.device),
			torch.FloatTensor(self.done[ind]).to(self.device)
		)
		
	def load_d4rl(self, d4rl_dataset, size=-1):
		self.size = int(d4rl_dataset['observations'].shape[0])-1
		
		self.state[:self.size] = d4rl_dataset['observations'][:-1].astype(np.float32)
		self.action[:self.size] = d4rl_dataset['actions'][:-1].astype(np.float32)
		self.next_state[:self.size] = d4rl_dataset['observations'][1:].astype(np.float32)
		self.reward[:self.size] = d4rl_dataset['rewards'][:-1].astype(np.float32)
		self.done[:self.size] = d4rl_dataset['terminals'][:-1].astype(np.float32)

# Network instantiations
actor = Actor(input_shape, output_shape, max_action, args.phi).to(device)
actor_target = copy.deepcopy(actor)

q_kwargs = {"state_dim": input_shape, "action_dim": output_shape}
qf1, qf2 = Critic(**q_kwargs).to(device), Critic(**q_kwargs).to(device)
qf1_target, qf2_target = copy.deepcopy(qf1), copy.deepcopy(qf2)

vae = VAE(input_shape, output_shape, output_shape * 2, max_action).to(device)

# Optimizers
vae_optimizer = optim.Adam(vae.parameters(), lr=args.lr)
actor_optimizer = optim.Adam(actor.parameters(), lr=args.lr)
q_optimizer = optim.Adam(list(qf1.parameters()) + list(qf2.parameters()), lr=args.lr)

# loading dataset into buffer
dataset = env.get_dataset()
replay_buffer = ReplayBuffer(input_shape, output_shape, device)
replay_buffer.load_d4rl(dataset)

for global_step in range(1,args.total_timesteps+1): 
	s_obs, s_actions, s_next_obses, s_rewards, s_dones = replay_buffer.sample(args.batch_size)
	
	# VAE loss: constrain the actions toward the action dist of the dataset:
	reconstructed_actions, latent_mean, latent_scale = vae(s_obs, s_actions)
	rec_loss = F.mse_loss(reconstructed_actions, s_actions) # Eq 28
	kl_loss = -0.5 * (1 + latent_scale.pow(2).log() - latent_mean.pow(2) - latent_scale.pow(2)).mean()
	vae_loss = rec_loss + 0.5 * kl_loss # Eq 30, lambda = .5
	
	vae_optimizer.zero_grad()
	vae_loss.backward()
	vae_optimizer.step()
	
	# Q Function updates
	with torch.no_grad():
		# Eq (27)
		s_next_obses_repeat = s_next_obses.repeat_interleave(args.num_actions,dim=0)
		next_actions = actor_target(s_next_obses_repeat, vae.get_actions(s_next_obses_repeat))
		next_obs_qf1_target = qf1_target(s_next_obses_repeat, next_actions).squeeze(-1)
		next_obs_qf2_target = qf2_target(s_next_obses_repeat, next_actions).squeeze(-1)
		
		qf_target = args.lmbda * torch.min(next_obs_qf1_target, next_obs_qf2_target) + \
			(1. - args.lmbda) * torch.max(next_obs_qf1_target, next_obs_qf2_target)
		qf_target = qf_target.view(args.batch_size, -1).max(1)[0]
		q_backup = s_rewards + (1. - s_dones) * args.gamma * qf_target # Eq 13
	
	qf1_values = qf1(s_obs, s_actions).view(-1)
	qf2_values = qf2(s_obs, s_actions).view(-1)

	qf1_loss = F.mse_loss(qf1_values, q_backup)
	qf2_loss = F.mse_loss(qf2_values, q_backup)
	qf_loss = qf1_loss + qf2_loss

	q_optimizer.zero_grad()
	qf_loss.backward()
	q_optimizer.step()
	
	# Perturbation network loss
	with torch.no_grad():
		raw_actions = vae.get_actions(s_obs)
	actions = actor(s_obs, raw_actions)
	qf1_pi = qf1(s_obs, actions).view(-1)
	actor_loss = - qf1_pi.mean()
	
	actor_optimizer.zero_grad()
	actor_loss.backward()
	actor_optimizer.step()
	
	# update the target networks
	for param, target_param in zip(qf1.parameters(), qf1_target.parameters()):
		target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)
	for param, target_param in zip(qf2.parameters(), qf2_target.parameters()):
		target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)
	for param, target_param in zip(actor.parameters(), actor_target.parameters()):
		target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)
	
	# Evaluated the agent and log results
	if global_step > 0 and global_step % args.eval_frequency == 0:
		eval_env = gym.make(args.gym_id)
		eval_env.seed(args.seed + 100)
		avg_return = 0.
		for _ in range(10):
			done= False
			obs = np.array( eval_env.reset())

			while not done:
				with torch.no_grad():
					obs_tensor = torch.Tensor(obs).unsqueeze(0).repeat(100, 1).to(device)
					action = vae.get_actions(obs_tensor)
					action = actor(obs_tensor, action)
					qf1_values = qf1(obs_tensor, action).view(-1)
					action = action[qf1_values.argmax(0)].cpu().numpy()

				obs, rew, done, _ = eval_env.step(action)
				obs = np.array(obs)
				avg_return += rew
		avg_return /= 10.
		
		# TODO: match logging to other scripts
		print("[I%08d] PertNetLoss: %.3f -- QLoss: %.3f -- VAELoss: %.3f -- Ep.Mean Ret: %.3f" %
			(global_step, actor_loss.item(), qf_loss.item(), vae_loss.item(), avg_return))
		print("[%04d]" % global_step, vae_loss.item(), rec_loss.item(), kl_loss.item(), qf_loss.item(), actor_loss.item())

		writer.add_scalar("global_step", global_step, global_step)
		writer.add_scalar("charts/episode_reward", avg_return, global_step)
		writer.add_scalar("losses/qf_loss", qf_loss.item() / 2., global_step)
		writer.add_scalar("losses/vae_loss", vae_loss.item(), global_step)
		writer.add_scalar("losses/perturb_net_loss", actor_loss.item(), global_step)
		writer.add_scalar("losses/vae_raw_kl_loss", kl_loss.item(), global_step)
		writer.add_scalar("losses/vae_rec_loss", rec_loss.item(), global_step)

env.close()
writer.close()
