import os
import re
import time
import random
import argparse
import collections
import numpy as np
from distutils.util import strtobool

import gym
import pybullet_envs
import d4rl_pybullet
# import d4rl
# import mujoco_py

import torch
import torch as th # TODO: clean up later
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal, Uniform
from torch.utils.data.dataset import IterableDataset
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from gym.wrappers import TimeLimit, Monitor
from gym.spaces import Box

parser = argparse.ArgumentParser(description='Batch Constrained Q-Learning for Continuous Domains; Uses D4RL datasets')
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
parser.add_argument('--eval-frequency', type=int, default=1000,
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

# NN Parameterization
parser.add_argument('--weights-init', default='xavier', const='xavier', nargs='?', choices=['xavier', "orthogonal", 'uniform'],
                    help='weight initialization scheme for the neural networks.')
parser.add_argument('--bias-init', default='zeros', const='xavier', nargs='?', choices=['zeros', 'uniform'],
                    help='weight initialization scheme for the neural networks.')

args = parser.parse_args()
if not args.seed:
    args.seed = int(time.time())

# create offline gym id: 'BeamRiderNoFrameskip-v4' -> 'beam-rider-expert-v0'
# args.offline_gym_id = re.sub(r'(?<!^)(?=[A-Z])', '-', args.gym_id).lower().replace(
#     "v2", "") + args.offline_dataset_id

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
## DBG: evn override
env = gym.make(args.gym_id)
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.backends.cudnn.deterministic = args.torch_deterministic
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

# Helper for testing an agent in determinstic mode
def test_agent(actor, vae, qf1, n_eval_episodes=10):
    eval_env = gym.make(args.gym_id)
    eval_env.seed(args.seed + 100)
    
    # TODO: verify if the models are evaluated with the perturb network input too.
    returns, lengths = [], []

    for _ in range(n_eval_episodes):
        ret, done, t = 0., False, 0
        obs = np.array( eval_env.reset())

        while not done:
            with torch.no_grad():
                obs_tensor = torch.Tensor(obs).unsqueeze(0).repeat(100, 1).to(device)
                action = vae.get_actions(obs_tensor)
                action = actor(obs_tensor, action)
                qf1_values = qf1(obs_tensor, action).view(-1)
                action = action[qf1_values.argmax(0)].cpu().numpy()

            obs, rew, done, _ = eval_env.step(action)
            obs = np.array( obs)
            ret += rew
            t += 1
        
        returns.append(ret)
        lengths.append(t)
    
    eval_stats = {
        "test_mean_return": np.mean(returns),
        "test_mean_length": np.mean(lengths)
    }

    return eval_stats

# Offline RL data loading
class ExperienceReplayDataset(IterableDataset):
    def __init__(self, env_name):
        self.dataset_env = gym.make(env_name)
        self.dataset = self.dataset_env.get_dataset()
    def __iter__(self):
        while True:
            idx = np.random.choice(len(self.dataset['observations'])-1)
            yield self.dataset['observations'][:-1][idx].astype(np.float32), \
                self.dataset['actions'][:-1][idx].astype(np.float32), \
                self.dataset['rewards'][:-1][idx].astype(np.float32), \
                self.dataset['observations'][1:][idx].astype(np.float32), \
                self.dataset['terminals'][:-1][idx].astype(np.float32)

# data_loader = iter(DataLoader(ExperienceReplayDataset(args.offline_gym_id), batch_size=args.batch_size, num_workers=2))
data_loader = iter(DataLoader(ExperienceReplayDataset(args.gym_id), batch_size=args.batch_size, num_workers=2))

# VAE, Perturbation Network and QFunction definition
# Weight init scheme
def layer_init(layer, weight_gain=1, bias_const=0):
    if isinstance(layer, nn.Linear):
        if args.weights_init == "xavier":
            torch.nn.init.xavier_uniform_(layer.weight, gain=weight_gain)
        elif args.weights_init == "orthogonal":
            torch.nn.init.orthogonal_(layer.weight, gain=weight_gain)
        if args.bias_init == "zeros":
            torch.nn.init.constant_(layer.bias, bias_const)

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
        super().__init__()
        self.l1 = nn.Linear(state_dim + action_dim, 400)
        self.l2 = nn.Linear(400, 300)
        self.l3 = nn.Linear(300, 1)

    def forward(self, state, action):
        q1 = F.relu(self.l1(torch.cat([state, action], 1)))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)

        return q1

class VAE(nn.Module):
    def __init__(self, state_dim, action_dim, latent_dim, max_action):
        super().__init__()
        self.max_action, self.latent_dim, self.device = max_action, latent_dim, "cpu"
        self.e1 = nn.Linear(state_dim + action_dim, 750)
        self.e2 = nn.Linear(750, 750)

        self.mean = nn.Linear(750, latent_dim)
        self.log_std = nn.Linear(750, latent_dim)

        self.d1 = nn.Linear(state_dim + latent_dim, 750)
        self.d2 = nn.Linear(750, 750)
        self.d3 = nn.Linear(750, action_dim)
    
    def to(self, device, *args, **kwargs):
        super().to(device)
        self.device = device
        return self
    
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

    def get_actions(self, obs):
        return self.decode(obs)

# Network instantiations
vae = VAE(input_shape, output_shape, output_shape * 2, max_action).to(device) # TODO: try alternative dim for: (obs_shape + act_shape) // 2
actor = Actor(input_shape, output_shape, max_action, args.phi).to(device)
actor_target = Actor(input_shape, output_shape, max_action, args.phi).to(device).requires_grad_(False)
actor_target.load_state_dict(actor.state_dict())

q_kwargs = {"state_dim": input_shape, "action_dim": output_shape}
qf1, qf1_target = Critic(**q_kwargs).to(device), Critic(**q_kwargs).to(device).requires_grad_(False)
qf2, qf2_target = Critic(**q_kwargs).to(device), Critic(**q_kwargs).to(device).requires_grad_(False)
qf1_target.load_state_dict(qf1.state_dict()), qf2_target.load_state_dict(qf2.state_dict())

# Optimizers
vae_optimizer = optim.Adam(vae.parameters()) # default lr: 1e-3
actor_optimizer = optim.Adam(actor.parameters(), lr=args.lr)
q_optimizer = optim.Adam(list(qf1.parameters()) + list(qf2.parameters()), lr=args.lr)

# Training Loop
for global_step in range(args.total_timesteps): 
    # tenosrize, place and the correct device
    s_obs, s_actions, s_rewards, s_next_obses, s_dones = [i.to(device) for i in next(data_loader)]
        
    # VAE loss: constrain the actions toward the action dist of the dataset:
    reconstructed_actions, latent_mean, latent_scale = vae(s_obs, s_actions)
    rec_loss = F.mse_loss(reconstructed_actions, s_actions) # Eq 28
    # kl_loss = th.distributions.kl.kl_divergence(latent_dist, vae_policy.latent_prior).sum(-1).mean() # Eq 29
    kl_loss = -0.5 * (1 + latent_scale.pow(2).log() - latent_mean.pow(2) - latent_scale.pow(2)).mean()
    vae_loss = rec_loss + 0.5 * kl_loss # Eq 30, lambda = .5
    
    vae_optimizer.zero_grad()
    vae_loss.backward()
    vae_optimizer.step()
    
    # Q Function updates
    with th.no_grad():
        # Eq (27)
        s_next_obses_repeat = s_next_obses.repeat_interleave(args.num_actions,dim=0)
        next_actions = actor_target(s_next_obses_repeat, vae.get_actions(s_next_obses_repeat))
        
        next_obs_qf1_target = qf1_target(s_next_obses_repeat, next_actions).squeeze(-1)
        next_obs_qf2_target = qf2_target(s_next_obses_repeat, next_actions).squeeze(-1)
        
        qf_target = args.lmbda * th.min(next_obs_qf1_target, next_obs_qf2_target) + \
            (1. - args.lmbda) * th.max(next_obs_qf1_target, next_obs_qf2_target)
        qf_target = qf_target.view(args.batch_size, args.num_actions).max(1)[0]

        q_backup = s_rewards + (1. - s_dones) * args.gamma * qf_target # Eq 13
    
    qf1_values = qf1(s_obs, s_actions).view(-1)
    qf2_values = qf2(s_obs, s_actions).view(-1)

    qf1_loss = F.mse_loss(qf1_values, q_backup)
    qf2_loss = F.mse_loss(qf2_values, q_backup)
    qf_loss = (qf1_loss + qf2_loss ) / 2. # for logging purpose mainly

    q_optimizer.zero_grad()
    qf_loss.backward()
    q_optimizer.step()

    # Perturbation network loss
    # TODO: version with "max over 10 actions"
    with th.no_grad():
        raw_actions = vae.get_actions(s_obs)
    actions = actor(s_obs, raw_actions)
    qf1_pi = qf1(s_obs, actions).view(-1)
    # qf2_pi = qf2(s_obs, actions).view(-1)
    # min_qf_pi = th.min(qf1_pi,qf2_pi) # TODO: compare with one network version.
    actor_loss = - qf1_pi.mean()
    
    actor_optimizer.zero_grad()
    actor_loss.backward()
    actor_optimizer.step()
    
    # update the target networks
    if global_step % args.target_network_frequency == 0:
        # TODO: consider refactor as a single function ?
        for param, target_param in zip(qf1.parameters(), qf1_target.parameters()):
            target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)
        for param, target_param in zip(qf2.parameters(), qf2_target.parameters()):
            target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)
        for param, target_param in zip(actor.parameters(), actor_target.parameters()):
            target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)
    
    # Evaluated the agent and log results
    if global_step % args.eval_frequency == 0:
        # TODO: clean up the eval section
        eval_stats = test_agent(actor, vae, qf1)
        # TODO: match logging to other scripts
        print("[I%08d] PertNetLoss: %.3f -- QLoss: %.3f -- VAELoss: %.3f -- Ep.Mean Ret: %.3f" %
            (global_step, actor_loss.item(), qf_loss.item(), vae_loss.item(), eval_stats["test_mean_return"]))

        # TODO: change the names to match the network names.
        writer.add_scalar("global_step", global_step, global_step)
        writer.add_scalar("charts/episode_reward", eval_stats["test_mean_return"], global_step)
        writer.add_scalar("losses/qf_loss", qf_loss.item(), global_step)
        writer.add_scalar("losses/vae_loss", vae_loss.item(), global_step)
        writer.add_scalar("losses/perturb_net_loss", actor_loss.item(), global_step)
        writer.add_scalar("losses/vae_raw_kl_loss", kl_loss.item(), global_step)
        writer.add_scalar("losses/vae_rec_loss", rec_loss.item(), global_step)

# TODO: add writer close too
env.close()
