import os
import re
import time
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

parser = argparse.ArgumentParser(description='Batch Constrained Q-Learning for Continuous Domains; Uses D4RL datasets')
# Common arguments
# parser.add_argument('--exp-name', type=str, default=os.path.basename(__file__).rstrip(".py"),
#                     help='the name of this experiment')
parser.add_argument('--exp-name', type=str, default=os.path.basename(__file__).rstrip(".py"),
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

# Algorithm specific argumen1ts
parser.add_argument('--gamma', type=float, default=0.99,
                    help='the discount factor gamma')
parser.add_argument('--lr', type=float, default=1e-3,
                    help='the learning rate of the optimizer for the policy weights')
parser.add_argument('--q-lr', type=float, default=1e-3,
                    help='the learning rate of the optimizer for the Q netowrks weights')
parser.add_argument('--target-network-frequency', type=int, default=1, # Denis Yarats' implementation delays this by 2.
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

# Logging
parser.add_argument('--log-interval', type=int, default=200,
                    help='determines how many iter the models train before logging the training stats. Also determines how often the agent is evaluated in the env. (time consuming).')

args = parser.parse_args()
if not args.seed:
    args.seed = int(time.time())

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
input_shape = env.observation_space.shape[0]
output_shape = env.action_space.shape[0]
# respect the default timelimit
assert isinstance(env.action_space, Box), "only continuous action space is supported"
if args.capture_video:
    env = Monitor(env, f'videos/{experiment_name}')

# Helper for testing an agent in determinstic mode
def test_agent(env, vae_policy, qf1, perturb_net=None, n_eval_episodes=5):
    # TODO: verify if the models are evaluated with the perturb network input too.
    returns, lengths = [], []

    for _ in range(n_eval_episodes):
        ret, done, t = 0., False, 0
        obs = np.array( env.reset())

        while not done:
            # MaxEntRL Paper argues eval should not be determinsitic
            with torch.no_grad():
                obs_tensor = torch.Tensor(obs).unsqueeze(0).repeat_interleave(args.num_actions, dim=0).to(device)
                action = vae_policy.get_actions(obs_tensor)
                if perturb_net is not None:
                    action += perturb_net(obs_tensor, action)
                    action.clamp(-1., 1.) # TODO: adapt to env actin limits
                qf1_values = qf1(obs_tensor, action).view(-1)
            action = action[th.argmax(qf1_values)].cpu().numpy()
            
            obs, rew, done, _ = env.step(action)
            obs = np.array( obs)
            ret += rew
            t += 1
        
        returns.append(ret)
        lengths.append(t)
    
    eval_stats = {
        "test_mean_return": np.mean(returns),
        "test_mean_length": np.mean(lengths),

        "test_max_return": np.max(returns),
        "test_max_length": np.max(lengths),

        "test_min_return": np.min(returns),
        "test_min_length": np.min(lengths)
    }

    return eval_stats

# Dataset tool and loading methods
from torch.utils.data import Dataset, DataLoader

class ExperienceReplayDataset(Dataset):
    def __init__(self, env_name, device="cpu"):
        self.dataset_env = gym.make(env_name)
        self.dataset = self.dataset_env.get_dataset()
        self.device = device # handles putting the data on the davice when sampling
    def __len__(self):
        return self.dataset['observations'].shape[0]-1
    def __getitem__(self, index):
        return self.dataset['observations'][:-1][index].astype(np.float32), \
               self.dataset['actions'][:-1][index].astype(np.float32), \
               self.dataset['rewards'][:-1][index].astype(np.float32), \
               self.dataset['observations'][1:][index].astype(np.float32), \
               self.dataset['terminals'][:-1][index].astype(np.float32)

data_loader = DataLoader(ExperienceReplayDataset(args.offline_gym_id), batch_size=args.batch_size, num_workers=2)

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

class QNetwork(nn.Module):
    def __init__(self, obs_shape, act_shape, layer_init):
        super().__init__()
        self.network = nn.Sequential(*[
            nn.Linear(obs_shape+act_shape, 400), nn.ReLU(),
            nn.Linear(400, 300), nn.ReLU(),
            nn.Linear(300, 1)
        ])
        self.apply(layer_init)

    def forward(self, x, a):
        return self.network(th.cat([x,a],1))

# VAE Policy
class VAEPolicy(nn.Module):
    def __init__(self, obs_shape, act_shape, latent_shape):
        super().__init__()
        self.obs_shape, self.act_shape, self.latent_shape = obs_shape, act_shape, latent_shape
        
        self.encoder = nn.Sequential(*[
            nn.Linear(obs_shape + act_shape, 750), nn.ReLU(),
            nn.Linear(750, 750), nn.ReLU(),
            nn.Linear(750, latent_shape * 2)
        ])
        
        self.decoder = nn.Sequential(*[
            nn.Linear(obs_shape + latent_shape, 750), nn.ReLU(),
            nn.Linear(750, 750), nn.ReLU(),
            nn.Linear(750, act_shape), nn.Tanh()
        ])
        
        self.register_buffer("latent_prior_mean", th.zeros(latent_shape))
        self.register_buffer("latent_prior_scale", th.ones(latent_shape))
        self.latent_prior = Normal(self.latent_prior_mean, self.latent_prior_scale)
        
        self.apply(layer_init)
    
    def to(self, device, *args, **kwargs):
        super().to(device)
        # after super().to(device) is called, the registered buffers to parameterize
        # the prior will also have been moved to device. Hence, recreate the prior
        # dist using the registedred_buffers that are now on "device".
        self.latent_prior = Normal(self.latent_prior_mean, self.latent_prior_scale) 
        self.device = device
        return self
    
    def forward(self, obs_batch, act_batch):
        batch_size = obs_batch.shape[0]
        
        obs_act = th.cat([obs_batch, act_batch],1)
        latent_mean, latent_scale = th.chunk(self.encoder(obs_act), 2, 1)
        latent_scale = th.sigmoid(latent_scale)
        
        latent_dist = Normal(latent_mean, latent_scale)
        latent = latent_dist.rsample() # z ~ Normal(mu, sigma)
    
        action = self.decoder(th.cat([obs_batch, latent],1))
        
        return action, latent_dist
    
    def get_actions(self, obs):
        latent = self.latent_prior.sample([obs.shape[0]])
        
        return self.decoder(th.cat([obs, latent],1))

class PerturbationNetwork(nn.Module):
    def __init__(self, obs_shape, act_shape, max_perturbation=0.05):
        super().__init__()
        self.obs_shape, self.act_shape, self.max_perturb = obs_shape, act_shape, max_perturbation
        self.network = nn.Sequential(*[
            nn.Linear(obs_shape+act_shape, 400), nn.ReLU(),
            nn.Linear(400,300), nn.ReLU(),
            nn.Linear(300, act_shape), nn.Tanh()
        ])
    
    def forward(self, obs_batch, act_batch):
        return self.network(th.cat([obs_batch, act_batch],1)) * self.max_perturb

# Network instantiations
vae_policy = VAEPolicy(input_shape, output_shape, output_shape * 2).to(device) # TODO: try alternative dim for: (obs_shape + act_shape) // 2
perturb_net = PerturbationNetwork(input_shape, output_shape, args.phi).to(device)
perturb_net_target = PerturbationNetwork(input_shape, output_shape, args.phi).to(device).requires_grad_(False)
perturb_net_target.load_state_dict(perturb_net.state_dict())

q_kwargs = {"obs_shape": input_shape, "act_shape": output_shape, "layer_init": layer_init}
qf1 = QNetwork(**q_kwargs).to(device)
qf2 = QNetwork(**q_kwargs).to(device)

qf1_target = QNetwork(**q_kwargs).to(device).requires_grad_(False)
qf2_target = QNetwork(**q_kwargs).to(device).requires_grad_(False)
qf1_target.load_state_dict(qf1.state_dict())
qf2_target.load_state_dict(qf2.state_dict())

# Optimizers
vae_optimizer = optim.Adam(vae_policy.parameters(), lr=args.lr)
perturb_optimizer = optim.Adam(perturb_net.parameters(), lr=args.lr)
q_optimizer = optim.Adam(list(qf1.parameters()) + list(qf2.parameters()), lr=args.lr)

# Training Loop
global_step = 0 # Tracks the Gradient updates
for epoch in range(args.n_epochs):
    for k, train_batch in enumerate(data_loader):
        global_step += 1
        
        # tenosrize, place and the correct device
        s_obs, s_actions, s_rewards, s_next_obses, s_dones = [i.to(device) for i in train_batch]
        
        # VAE loss: constrain the actions toward the action dist of the dataset:
        reconstructed_actions, latent_dist = vae_policy(s_obs, s_actions)
        rec_loss = F.mse_loss(reconstructed_actions, s_actions) # Eq 28
        kl_loss = th.distributions.kl.kl_divergence(latent_dist, vae_policy.latent_prior).sum(-1).mean() # Eq 29
        vae_loss = rec_loss + args.lmbda * kl_loss # Eq 30
        # NOTE: Original implementation uses 0.5 instead of lambda: https://github.com/sfujim/BCQ/blob/9690927c86b9eade8b50f0e69d46a146c4454851/continuous_BCQ/BCQ.py#L143
        
        vae_optimizer.zero_grad()
        vae_loss.backward()
        vae_optimizer.step()
        
        # Q Function updates
        with th.no_grad():
            # Eq (27)
            s_next_obses_repeat = s_next_obses.repeat_interleave(args.num_actions,dim=0)
            next_actions = vae_policy.get_actions(s_next_obses_repeat)
            next_actions += perturb_net_target(s_next_obses_repeat, next_actions)
            next_actions.clamp_(-1.,1.) # TODO: adapt to env's action space min,max
            
            next_obs_qf1_target = qf1_target(s_next_obses_repeat, next_actions).view(-1)
            next_obs_qf2_target = qf2_target(s_next_obses_repeat, next_actions).view(-1)

            min_qf_target = args.lmbda * th.min(next_obs_qf1_target, next_obs_qf2_target)
            max_qf_target = (1. - args.lmbda) * th.max(next_obs_qf1_target, next_obs_qf2_target)
            qf_target = th.max((min_qf_target + max_qf_target).view(args.batch_size, args.num_actions), -1)[0]

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
            raw_actions = vae_policy.get_actions(s_obs) # in [-1,1]
        actions = raw_actions + perturb_net(s_obs, raw_actions)
        actions.clamp_(-1., 1.) # TODO: adapt to env's action space min,max
        qf1_pi = qf1(s_obs, actions).view(-1)
        qf2_pi = qf2(s_obs, actions).view(-1)
        min_qf_pi = th.min(qf1_pi,qf2_pi) # TODO: compare with one network version.
        perturb_net_loss = - min_qf_pi.mean()
        
        perturb_optimizer.zero_grad()
        perturb_net_loss.backward()
        perturb_optimizer.step()
        
        # update the target networks
        if global_step % args.target_network_frequency == 0:
            # TODO: consider refactor as a single function ?
            for param, target_param in zip(qf1.parameters(), qf1_target.parameters()):
                target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)
            for param, target_param in zip(qf2.parameters(), qf2_target.parameters()):
                target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)
            for param, target_param in zip(perturb_net.parameters(), perturb_net_target.parameters()):
                target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)
        
        # Evaluated the agent and log results
        if global_step % args.log_interval == 0:
            eval_stats = test_agent(env, vae_policy, qf1)
            eval_stats_perturbed = test_agent(env, vae_policy, qf1, perturb_net) # passing the perturb net enables "noisy" evaluation
            print("[E%04d|I%08d] PertNetLoss: %.3f -- QLoss: %.3f -- VAELoss: %.3f -- Determ. Mean Ret: %.3f -- Noisy Mean Ret.: %.3f" %
            (epoch, global_step, perturb_net_loss.item(), qf_loss.item(), vae_loss.item(), eval_stats["test_mean_return"], eval_stats_perturbed["test_mean_return"]))
            
            writer.add_scalar("global_step", global_step, global_step)
            writer.add_scalar("charts/episode_reward", eval_stats["test_mean_return"], global_step)
            writer.add_scalar("charts/episode_reward_perturbed", eval_stats_perturbed["test_mean_return"], global_step)
            writer.add_scalar("losses/qf_loss", qf_loss.item(), global_step)
            writer.add_scalar("losses/vae_loss", vae_loss.item(), global_step)
            writer.add_scalar("losses/perturb_net_loss", perturb_net_loss.item(), global_step)
            writer.add_scalar("losses/vae_raw_kl_loss", kl_loss.item(), global_step)
            writer.add_scalar("losses/vae_rec_loss", rec_loss.item(), global_step)

        # Stop iteration over the dataset
        if k >= args.epoch_length-1:
            break