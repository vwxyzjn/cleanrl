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
import pybullet_envs
from gym.spaces import Discrete, Box, MultiBinary, MultiDiscrete, Space
import time
import random
import os

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
    parser.add_argument('--torch-deterministic', type=bool, default=True,
                       help='whether to set `torch.backends.cudnn.deterministic=True`')
    parser.add_argument('--cuda', type=bool, default=True,
                       help='whether to use CUDA whenever possible')
    parser.add_argument('--prod-mode', type=bool, default=False,
                       help='run the script in production mode and use wandb to log outputs')
    parser.add_argument('--capture-video', type=bool, default=False,
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
    parser.add_argument('--update-epochs', type=int, default=3,
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
if int(args.episode_length):
    if not isinstance(env, TimeLimit):
        env = TimeLimit(env, int(args.episode_length))
    else:
        env._max_episode_steps = int(args.episode_length)
else:
    args.episode_length = env._max_episode_steps if isinstance(env, TimeLimit) else 200
if args.capture_video:
    env = Monitor(env, f'videos/{experiment_name}')

# ALGO LOGIC: initialize agent here:

class Policy(nn.Module):
    def __init__(self):
        super(Policy, self).__init__()
        self.fc1 = nn.Linear(input_shape, 120)
        self.fc2 = nn.Linear(120, 84)
        self.mean = nn.Linear(84, output_shape)
        self.logstd = nn.Parameter(torch.zeros(1, output_shape))

    def forward(self, x):
        x = preprocess_obs_fn(x)
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        action_mean = self.mean(x)
        action_logstd = self.logstd.expand_as(action_mean)
        return action_mean, action_logstd

    def get_action(self, x):
        mean, logstd = self.forward(x)
        std = torch.exp(logstd)
        probs = Normal(mean, std)
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
        self.fc1 = nn.Linear(input_shape, 64)
        self.fc2 = nn.Linear(64, 1)

    def forward(self, x):
        x = preprocess_obs_fn(x)
        x = F.tanh(self.fc1(x))
        x = self.fc2(x)
        return x

pg = Policy().to(device)
vf = Value().to(device)
optimizer = optim.Adam(list(pg.parameters()) + list(vf.parameters()), lr=args.learning_rate)
loss_fn = nn.MSELoss()
#print(pg.logstd.bias)

# TRY NOT TO MODIFY: start the game
global_step = 0
while global_step < args.total_timesteps:
    next_obs = np.array(env.reset())
    actions = np.empty((args.episode_length,) + env.action_space.shape)
    rewards, dones = np.zeros((2, args.episode_length))
    obs = np.empty((args.episode_length,) + env.observation_space.shape)

    # ALGO LOGIC: put other storage logic here
    values = torch.zeros((args.episode_length), device=device)
    logprobs = np.zeros((args.episode_length,),)
    entropys = torch.zeros((args.episode_length,), device=device)

    # TRY NOT TO MODIFY: prepare the execution of the game.
    for step in range(args.episode_length):
        global_step += 1
        obs[step] = next_obs.copy()
        
        # ALGO LOGIC: put action logic here
        values[step] = vf.forward(obs[step:step+1])
        action, logproba = pg.get_action(obs[step:step+1])
        actions[step] = action.data.numpy()[0]
        logprobs[step] = logproba.data.numpy()[0]
        
        # sometimes causes the performance to stay the same for a really long time.. hmmm
        # could be a degenarate seed
        clipped_action = np.clip(action.tolist(), env.action_space.low, env.action_space.high)[0]

        # TRY NOT TO MODIFY: execute the game and log data.
        next_obs, rewards[step], dones[step], _ = env.step(clipped_action)
        next_obs = np.array(next_obs)
        if dones[step]:
            break

    # ALGO LOGIC: training.
    # calculate the discounted rewards, or namely, returns
    returns = np.zeros_like(rewards)
    for t in reversed(range(rewards.shape[0]-1)):
        returns[t] = rewards[t] + args.gamma * returns[t+1] * (1-dones[t])
    # advantages are returns - baseline, value estimates in our case
    advantages = returns - values.detach().cpu().numpy()

    for _ in range(args.update_epochs):
        newlogproba = pg.get_logproba(obs[:step], torch.Tensor(actions[:step]))
        # newvalues = vf.forward(obs[:step]).flatten() DO we generate a new values from the current policy?
        ratio =  torch.exp(newlogproba - torch.Tensor(logprobs[:step]))
        surrogate1 = ratio * torch.Tensor(advantages[:step])
        surrogate2 = ratio.clamp(1 - args.clip_coef, 1 + args.clip_coef) * torch.Tensor(advantages[:step])
        policy_loss = - torch.mean(torch.min(surrogate1, surrogate2))
        vf_loss = torch.mean((values[:step] - torch.Tensor(returns[:step])).pow(2))
        entropy_loss = torch.mean(torch.exp(newlogproba) * newlogproba)
        total_loss = policy_loss + args.vf_coef * vf_loss + args.ent_coef * entropy_loss
        optimizer.zero_grad()
        total_loss.backward(retain_graph=True)
        nn.utils.clip_grad_norm_(list(pg.parameters()) + list(vf.parameters()), args.max_grad_norm)
        optimizer.step()

    # TRY NOT TO MODIFY: record rewards for plotting purposes
    writer.add_scalar("charts/episode_reward", rewards.sum(), global_step)
    writer.add_scalar("losses/value_loss", vf_loss.item(), global_step)
    writer.add_scalar("losses/entropy", entropys[:step].mean().item(), global_step)
    writer.add_scalar("losses/policy_loss", policy_loss.item(), global_step)
env.close()
writer.close()
