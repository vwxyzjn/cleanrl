import os
import re
import time
import copy
import random
import argparse
import collections
import numpy as np
from distutils.util import strtobool

import gym
import d4rl
# try:
#     import mujoco_py
# except Exception as e:
#     print(e)

import torch
import torch as th # TODO: clean up later
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal, Uniform
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

from gym.wrappers import TimeLimit, Monitor
from gym.spaces import Box

parser = argparse.ArgumentParser(description='Batch Constrained Q-Learning for Continuous Domains')
# Common arguments
# parser.add_argument('--exp-name', type=str, default=os.path.basename(__file__).rstrip(".py"),
#                     help='the name of this experiment')
parser.add_argument('--exp-name', type=str, default="bcq_sfujim.ipynb",
                    help='the name of this experiment')
parser.add_argument('--gym-id', type=str, default="Hopper-v2",
                    help='the id of the gym environment')
parser.add_argument('--seed', type=int, default=2,
                    help='seed of the experiment')
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
parser.add_argument('--n-epochs', type=int, default=1000,
                    help='number of epochs (total training iters = n-epochs * epoch-length')
parser.add_argument('--epoch-length', type=int, default=1000,
                    help='number of training steps in an epoch')

# Algorithm specific arguments
parser.add_argument('--gamma', type=float, default=0.99,
                    help='the discount factor gamma')
parser.add_argument('--lr', type=float, default=1e-3,
                    help='the learning rate for the various optimizers')
parser.add_argument('--target-network-frequency', type=int, default=1, # Denis Yarats' implementation delays this by 2.
                    help="the timesteps it takes to update the target network")
parser.add_argument('--max-grad-norm', type=float, default=0.5,
                    help='the maximum norm for the gradient clipping')
parser.add_argument('--batch-size', type=int, default=100,
                    help="the batch size of sample from the reply memory")
parser.add_argument('--tau', type=float, default=0.005,
                    help="target smoothing coefficient (default: 0.005)")

# BQC specific parameters
parser.add_argument('--offline-dataset-id', type=str, default="medium-v0",
                    help='the id of the offline dataset gym environment')
parser.add_argument('--phi', type=float, default=0.05,
                    help='maximum perturbation applied over the actions sampled from the VAE Policy')
parser.add_argument('--lmbda', type=float, default=0.75,
                    help='coefficient of the min Q term in the Q-network update')
parser.add_argument('--num-actions', type=int, default=10,
                    help="how many actions sampled from the VAE to get the maximum value / best action")

# NN Parameterization
parser.add_argument('--weights-init', default='xavier', const='xavier', nargs='?', choices=['xavier', "orthogonal", 'uniform'],
                    help='weight initialization scheme for the neural networks.')
parser.add_argument('--bias-init', default='zeros', const='xavier', nargs='?', choices=['zeros', 'uniform'],
                    help='weight initialization scheme for the neural networks.')

# BCQ original author implementation args
parser.add_argument("--train_behavioral", action="store_true")  # If true, train behavioral (DDPG)
parser.add_argument("--generate_buffer", action="store_true")   # If true, generate buffer
parser.add_argument("--rand_action_p", default=0.3, type=float) # Probability of selecting random action during batch generation
parser.add_argument("--gaussian_std", default=0.3, type=float)  # Std of Gaussian exploration noise (Set to 0.1 if DDPG trains poorly)
parser.add_argument("--eval_freq", default=5e3, type=float)     # How often (time steps) we evaluate
parser.add_argument("--buffer_name", default="Robust")          # Prepends name to filename

args = parser.parse_args()
if not args.seed:
    args.seed = int(time.time())

args.total_steps = args.n_epochs * args.epoch_length

# create offline gym id: 'BeamRiderNoFrameskip-v4' -> 'beam-rider-expert-v0'
args.offline_gym_id = re.sub(r'(?<!^)(?=[A-Z])', '-', args.gym_id).lower().replace(
    "v2", "") + args.offline_dataset_id

# TRY NOT TO MODIFY: setup the environment
experiment_name = f"{args.gym_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
writer = SummaryWriter(f"runs/{experiment_name}")
writer.add_text('hyperparameters', "|param|value|\n|-|-|\n%s" % (
        '\n'.join([f"|{key}|{value}|" for key, value in vars(args).items()])))
if args.prod_mode:
    import wandb
    wandb.init(project=args.wandb_project_name, entity=args.wandb_entity, sync_tensorboard=True, config=vars(args), name=experiment_name, monitor_gym=True, save_code=True)
    writer = SummaryWriter(f"/tmp/{experiment_name}")

class Actor(nn.Module):
	def __init__(self, state_dim, action_dim, max_action, phi=0.05):
		super(Actor, self).__init__()
		self.l1 = nn.Linear(state_dim + action_dim, 400)
		self.l2 = nn.Linear(400, 300)
		self.l3 = nn.Linear(300, action_dim)
		
		self.max_action = max_action
		self.phi = phi


	def forward(self, state, action):
		a = F.relu(self.l1(torch.cat([state, action], 1)))
		a = F.relu(self.l2(a))
		a = self.phi * self.max_action * torch.tanh(self.l3(a))
		return (a + action).clamp(-self.max_action, self.max_action)


class Critic(nn.Module):
	def __init__(self, state_dim, action_dim):
		super(Critic, self).__init__()
		self.l1 = nn.Linear(state_dim + action_dim, 400)
		self.l2 = nn.Linear(400, 300)
		self.l3 = nn.Linear(300, 1)

		self.l4 = nn.Linear(state_dim + action_dim, 400)
		self.l5 = nn.Linear(400, 300)
		self.l6 = nn.Linear(300, 1)


	def forward(self, state, action):
		q1 = F.relu(self.l1(torch.cat([state, action], 1)))
		q1 = F.relu(self.l2(q1))
		q1 = self.l3(q1)

		q2 = F.relu(self.l4(torch.cat([state, action], 1)))
		q2 = F.relu(self.l5(q2))
		q2 = self.l6(q2)
		return q1, q2


	def q1(self, state, action):
		q1 = F.relu(self.l1(torch.cat([state, action], 1)))
		q1 = F.relu(self.l2(q1))
		q1 = self.l3(q1)
		return q1


# Vanilla Variational Auto-Encoder 
class VAE(nn.Module):
	def __init__(self, state_dim, action_dim, latent_dim, max_action, device):
		super(VAE, self).__init__()
		self.e1 = nn.Linear(state_dim + action_dim, 750)
		self.e2 = nn.Linear(750, 750)

		self.mean = nn.Linear(750, latent_dim)
		self.log_std = nn.Linear(750, latent_dim)

		self.d1 = nn.Linear(state_dim + latent_dim, 750)
		self.d2 = nn.Linear(750, 750)
		self.d3 = nn.Linear(750, action_dim)

		self.max_action = max_action
		self.latent_dim = latent_dim
		self.device = device


	def forward(self, state, action):
		z = F.relu(self.e1(torch.cat([state, action], 1)))
		z = F.relu(self.e2(z))

		mean = self.mean(z)
		# Clamped for numerical stability 
		log_std = self.log_std(z).clamp(-4, 15)
		std = torch.exp(log_std)
		z = mean + std * torch.randn_like(std)
		
		u = self.decode(state, z)

		return u, mean, std


	def decode(self, state, z=None):
		# When sampling from the VAE, the latent vector is clipped to [-0.5, 0.5]
		if z is None:
			z = torch.randn((state.shape[0], self.latent_dim)).to(self.device).clamp(-0.5,0.5)

		a = F.relu(self.d1(torch.cat([state, z], 1)))
		a = F.relu(self.d2(a))
		return self.max_action * torch.tanh(self.d3(a))
		

class BCQ(object):
	def __init__(self, state_dim, action_dim, max_action, device, discount=0.99, tau=0.005, lmbda=0.75, phi=0.05):
		latent_dim = action_dim * 2

		self.actor = Actor(state_dim, action_dim, max_action, phi).to(device)
		self.actor_target = copy.deepcopy(self.actor)
		self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=1e-3)

		self.critic = Critic(state_dim, action_dim).to(device)
		self.critic_target = copy.deepcopy(self.critic)
		self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=1e-3)

		self.vae = VAE(state_dim, action_dim, latent_dim, max_action, device).to(device)
		self.vae_optimizer = torch.optim.Adam(self.vae.parameters())

		self.max_action = max_action
		self.action_dim = action_dim
		self.discount = discount
		self.tau = tau
		self.lmbda = lmbda
		self.device = device


	def select_action(self, state):		
		with torch.no_grad():
			state = torch.FloatTensor(state.reshape(1, -1)).repeat(100, 1).to(self.device)
			action = self.actor(state, self.vae.decode(state))
			q1 = self.critic.q1(state, action)
			ind = q1.argmax(0)
		return action[ind].cpu().data.numpy().flatten()


	def train(self, replay_buffer, iterations, batch_size=100):

		for it in range(iterations):
			# Sample replay buffer / batch
			state, action, next_state, reward, not_done = replay_buffer.sample(batch_size)

			# Variational Auto-Encoder Training
			recon, mean, std = self.vae(state, action)
			recon_loss = F.mse_loss(recon, action)
			KL_loss	= -0.5 * (1 + torch.log(std.pow(2)) - mean.pow(2) - std.pow(2)).mean()
			vae_loss = recon_loss + 0.5 * KL_loss

			self.vae_optimizer.zero_grad()
			vae_loss.backward()
			self.vae_optimizer.step()

			# Critic Training
			with torch.no_grad():
				# Duplicate next state 10 times
				next_state = torch.repeat_interleave(next_state, 10, 0)

				# Compute value of perturbed actions sampled from the VAE
				target_Q1, target_Q2 = self.critic_target(next_state, self.actor_target(next_state, self.vae.decode(next_state)))

				# Soft Clipped Double Q-learning 
				target_Q = self.lmbda * torch.min(target_Q1, target_Q2) + (1. - self.lmbda) * torch.max(target_Q1, target_Q2)
				# Take max over each action sampled from the VAE
				target_Q = target_Q.reshape(batch_size, -1).max(1)[0].reshape(-1, 1)

				target_Q = reward + not_done * self.discount * target_Q

			current_Q1, current_Q2 = self.critic(state, action)
			critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

			self.critic_optimizer.zero_grad()
			critic_loss.backward()
			self.critic_optimizer.step()

			# Pertubation Model / Action Training
			sampled_actions = self.vae.decode(state)
			perturbed_actions = self.actor(state, sampled_actions)

			# Update through DPG
			actor_loss = -self.critic.q1(state, perturbed_actions).mean()
		 	 
			self.actor_optimizer.zero_grad()
			actor_loss.backward()
			self.actor_optimizer.step()

			# Update Target Networks
			for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
				target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

			for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
				target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
    
		return vae_loss, recon_loss, KL_loss, critic_loss, actor_loss

# utils and hijack the model loading parts
class ReplayBuffer(object):
	def __init__(self, state_dim, action_dim, device, max_size=int(1e6)):
		self.max_size = max_size
		self.ptr = 0
		self.size = 0

		self.state = np.zeros((max_size, state_dim))
		self.action = np.zeros((max_size, action_dim))
		self.next_state = np.zeros((max_size, state_dim))
		self.reward = np.zeros((max_size, 1))
		self.not_done = np.zeros((max_size, 1))

		self.device = device


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

		return (
			torch.FloatTensor(self.state[ind]).to(self.device),
			torch.FloatTensor(self.action[ind]).to(self.device),
			torch.FloatTensor(self.next_state[ind]).to(self.device),
			torch.FloatTensor(self.reward[ind]).to(self.device),
			torch.FloatTensor(self.not_done[ind]).to(self.device)
		)

	def save(self, save_folder):
		np.save(f"{save_folder}_state.npy", self.state[:self.size])
		np.save(f"{save_folder}_action.npy", self.action[:self.size])
		np.save(f"{save_folder}_next_state.npy", self.next_state[:self.size])
		np.save(f"{save_folder}_reward.npy", self.reward[:self.size])
		np.save(f"{save_folder}_not_done.npy", self.not_done[:self.size])
		np.save(f"{save_folder}_ptr.npy", self.ptr)
        
	def load_d4rl(self, d4rl_dataset, size=-1):
		self.size = int(d4rl_dataset['observations'].shape[0])-1
		
		self.state[:self.size] = d4rl_dataset['observations'][:-1].astype(np.float32)
		self.action[:self.size] = d4rl_dataset['actions'][:-1].astype(np.float32)
		self.next_state[:self.size] = d4rl_dataset['observations'][1:].astype(np.float32)
		self.reward[:self.size] = d4rl_dataset['rewards'][:-1].astype(np.float32).reshape(self.size, 1)
		self.not_done[:self.size] = 1. - d4rl_dataset['terminals'][:-1].astype(np.float32).reshape(self.size, 1)
    
	def load(self, save_folder, size=-1):
		reward_buffer = np.load(f"{save_folder}_reward.npy")
		
		# Adjust crt_size if we're using a custom size
		size = min(int(size), self.max_size) if size > 0 else self.max_size
		self.size = min(reward_buffer.shape[0], size)

		self.state[:self.size] = np.load(f"{save_folder}_state.npy")[:self.size]
		self.action[:self.size] = np.load(f"{save_folder}_action.npy")[:self.size]
		self.next_state[:self.size] = np.load(f"{save_folder}_next_state.npy")[:self.size]
		self.reward[:self.size] = reward_buffer[:self.size]
		self.not_done[:self.size] = np.load(f"{save_folder}_not_done.npy")[:self.size]

# Pytorch config
device = th.device( "cuda" if th.cuda.is_available() and args.cuda else "cpu")
th.backends.cudnn.deterministic = args.torch_deterministic # Props to CleanRL
th.manual_seed(args.seed)
th.cuda.manual_seed_all(args.seed)

# Seeding
random.seed(args.seed)
np.random.seed(args.seed)

# Environment setup
# if args.save_videos:
#     env = Monitor(env, tblogger.get_videos_savedir())

# CQL modification: loading data set into the buffer.
env = gym.make(args.offline_gym_id)
env.seed(args.seed)
env.action_space.seed(args.seed)
env.observation_space.seed(args.seed)

state_dim = env.observation_space.shape[0] # No quite general, however
action_dim = env.action_space.shape[0]
max_action = float(env.action_space.high[0])
# respect the default timelimit
assert isinstance(env.action_space, Box), "only continuous action space is supported"
if args.capture_video:
    env = Monitor(env, f'videos/{experiment_name}')

## loading dataset
dataset = env.get_dataset()

# training related methods
# Handles interactions with the environment, i.e. train behavioral or generate buffer
def interact_with_environment(env, state_dim, action_dim, max_action, device, args):
	# For saving files
	setting = f"{args.gym_id}_{args.seed}"
	buffer_name = f"{args.buffer_name}_{setting}"

	# Initialize and load policy
	policy = DDPG.DDPG(state_dim, action_dim, max_action, device)#, args.gamma, args.tau)
	if args.generate_buffer: policy.load(f"./models/behavioral_{setting}")

	# Initialize buffer
	replay_buffer = ReplayBuffer(state_dim, action_dim, device)
	
	evaluations = []

	state, done = env.reset(), False
	episode_reward = 0
	episode_timesteps = 0
	episode_num = 0

	# Interact with the environment for max_timesteps
	for t in range(int(args.total_steps)):

		episode_timesteps += 1

		# Select action with noise
		if (
			(args.generate_buffer and np.random.uniform(0, 1) < args.rand_action_p) or 
			(args.train_behavioral and t < args.start_timesteps)
		):
			action = env.action_space.sample()
		else:
			action = (
				policy.select_action(np.array(state))
				+ np.random.normal(0, max_action * args.gaussian_std, size=action_dim)
			).clip(-max_action, max_action)
        
		# Perform action
		next_state, reward, done, _ = env.step(action) 
		done_bool = float(done) if episode_timesteps < env._max_episode_steps else 0

		# Store data in replay buffer
		replay_buffer.add(state, action, next_state, reward, done_bool)

		state = next_state
		episode_reward += reward

		# Train agent after collecting sufficient data
		if args.train_behavioral and t >= args.start_timesteps:
			policy.train(replay_buffer, args.batch_size)

		if done:
			# +1 to account for 0 indexing. +0 on ep_timesteps since it will increment +1 even if done=True
			print(f"Total T: {t+1} Episode Num: {episode_num+1} Episode T: {episode_timesteps} Reward: {episode_reward:.3f}")
			# Reset environment
			state, done = env.reset(), False
			episode_reward = 0
			episode_timesteps = 0
			episode_num += 1

		# Evaluate episode
		if args.train_behavioral and (t + 1) % args.eval_freq == 0:
			evaluations.append(eval_policy(policy, args.gym_id, args.seed))
			np.save(f"./results/behavioral_{setting}", evaluations)
			policy.save(f"./models/behavioral_{setting}")

	# Save final policy
	if args.train_behavioral:
		policy.save(f"./models/behavioral_{setting}")

	# Save final buffer and performance
	else:
		evaluations.append(eval_policy(policy, args.gym_id, args.seed))
		np.save(f"./results/buffer_performance_{setting}", evaluations)
		replay_buffer.save(f"./buffers/{buffer_name}")


# Trains BCQ offline
def train_BCQ(state_dim, action_dim, max_action, dataset, device, writer, args):
	# For saving files
	setting = f"{args.gym_id}_{args.seed}"
	buffer_name = f"{args.buffer_name}_{setting}"

	# Initialize policy
	policy = BCQ(state_dim, action_dim, max_action, device, args.gamma, args.tau, args.lmbda, args.phi)

	# Load buffer
	replay_buffer = ReplayBuffer(state_dim, action_dim, device)
	replay_buffer.load_d4rl(dataset)
	# replay_buffer.load(f"./buffers/{buffer_name}")
	
	evaluations = []
	episode_num = 0
	done = True
	training_iters = 0
	
	while training_iters < args.total_steps:
		vae_loss, recon_loss, KL_loss, critic_loss, actor_loss = policy.train(replay_buffer, iterations=int(args.eval_freq), batch_size=args.batch_size)

		avg_return = eval_policy(policy, args.gym_id, args.seed)
		evaluations.append(avg_return)
		# np.save(f"./results/BCQ_{setting}", evaluations)

		training_iters += args.eval_freq
		print(f"Training iterations: {training_iters}")

		# logging to writer / wandb
		writer.add_scalar("global_step", training_iters, training_iters)
		writer.add_scalar("charts/episode_reward", avg_return, training_iters)
		writer.add_scalar("charts/episode_reward_perturbed", avg_return, training_iters)
		writer.add_scalar("losses/qf_loss", critic_loss.item() / 2., training_iters)
		writer.add_scalar("losses/vae_loss", vae_loss.item(), training_iters)
		writer.add_scalar("losses/perturb_net_loss", actor_loss.item(), training_iters)
		writer.add_scalar("losses/vae_raw_kl_loss", KL_loss.item(), training_iters)
		writer.add_scalar("losses/vae_rec_loss", recon_loss.item(), training_iters)

# Runs policy for X episodes and returns average reward
# A fixed seed is used for the eval environment
def eval_policy(policy, env_name, seed, eval_episodes=10):
	eval_env = gym.make(env_name)
	eval_env.seed(seed + 100)

	avg_reward = 0.
	for _ in range(eval_episodes):
		state, done = eval_env.reset(), False
		while not done:
			action = policy.select_action(np.array(state))
			state, reward, done, _ = eval_env.step(action)
			avg_reward += reward

	avg_reward /= eval_episodes

	print("---------------------------------------")
	print(f"Evaluation over {eval_episodes} episodes: {avg_reward:.3f}")
	print("---------------------------------------")
	return avg_reward

print("---------------------------------------")	
if args.train_behavioral:
    print(f"Setting: Training behavioral, Env: {args.gym_id}, Offline Gym Id: {args.offline_gym_id}, Seed: {args.seed}")
elif args.generate_buffer:
    print(f"Setting: Generating buffer, Env: {args.gym_id}, Offline Gym Id: {args.offline_gym_id}, Seed: {args.seed}")
else:
    print(f"Setting: Training BCQ, Env: {args.gym_id}, Offline Gym Id: {args.offline_gym_id}, Seed: {args.seed}")
print("---------------------------------------")

if args.train_behavioral and args.generate_buffer:
    print("Train_behavioral and generate_buffer cannot both be true.")
    exit()

if args.train_behavioral or args.generate_buffer:
    interact_with_environment(env, state_dim, action_dim, max_action, device, args)
else:
    train_BCQ(state_dim, action_dim, max_action, dataset, device, writer, args)