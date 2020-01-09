import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
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
    parser = argparse.ArgumentParser(description='A2C agent')
    # Common arguments
    parser.add_argument('--exp-name', type=str, default=os.path.basename(__file__).rstrip(".py"),
                       help='the name of this experiment')
    parser.add_argument('--gym-id', type=str, default="InvertedPendulumBulletEnv-v0",
                       help='the id of the gym environment')
    parser.add_argument('--learning-rate', type=float, default=7e-4,
                       help='the learning rate of the optimizer')
    parser.add_argument('--seed', type=int, default=1,
                       help='seed of the experiment')
    parser.add_argument('--episode-length', type=int, default=0,
                       help='the maximum length of each episode')
    parser.add_argument('--total-timesteps', type=int, default=4000000,
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
        self.fc_mean = nn.Linear(84, output_shape)
        self.logstd = nn.Parameter(torch.zeros(1, output_shape))
        # orthogonal initialization and layer scaling
        self.layer_norm(self.fc1, std=1.0)
        self.layer_norm(self.fc2, std=1.0)
        self.layer_norm(self.fc_mean, std=0.05)

    @staticmethod
    def layer_norm(layer, std=1.0, bias_const=0.0):
        torch.nn.init.orthogonal_(layer.weight, std)
        torch.nn.init.constant_(layer.bias, bias_const)

    def forward(self, x):
        x = preprocess_obs_fn(x)
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        
        action_mean = self.fc_mean(x)
        action_logstd = self.logstd.expand_as(action_mean)
        
        return action_mean, action_logstd.exp()

class Value(nn.Module):
    def __init__(self):
        super(Value, self).__init__()
        self.fc1 = nn.Linear(input_shape, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 1)
        # orthogonal initialization and layer scaling
        self.layer_norm(self.fc1, std=1.0)
        self.layer_norm(self.fc2, std=1.0)
        self.layer_norm(self.fc3, std=0.01)
    @staticmethod
    def layer_norm(layer, std=1.0, bias_const=0.0):
        torch.nn.init.orthogonal_(layer.weight, std)
        torch.nn.init.constant_(layer.bias, bias_const)
    def forward(self, x):
        x = preprocess_obs_fn(x)
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        x = self.fc3(x)
        return x

pg = Policy().to(device)
vf = Value().to(device)
optimizer = optim.Adam(list(pg.parameters()) + list(vf.parameters()), lr=args.learning_rate)
loss_fn = nn.MSELoss()

# TRY NOT TO MODIFY: start the game
global_step = 0
while global_step < args.total_timesteps:
    next_obs = np.array(env.reset())
    actions = np.empty((args.episode_length,), dtype=object)
    rewards, dones = np.zeros((2, args.episode_length))
    obs = np.zeros((args.episode_length,) + env.observation_space.shape)
    
    # ALGO LOGIC: put other storage logic here
    values = torch.zeros((args.episode_length), device=device)
    neglogprobs = torch.zeros((args.episode_length,), device=device)
    entropys = torch.zeros((args.episode_length,), device=device)
    
    # TRY NOT TO MODIFY: prepare the execution of the game.
    for step in range(args.episode_length):
        global_step += 1
        obs[step] = next_obs.copy()
        
        # ALGO LOGIC: put action logic here
        logits, std = pg.forward(obs[step:step+1])
        values[step] = vf.forward(obs[step:step+1])

        # ALGO LOGIC: `env.action_space` specific logic
        if isinstance(env.action_space, Discrete):
            probs = Categorical(logits=logits)
            action = probs.sample()
            actions[step], neglogprobs[step], entropys[step] = action.tolist()[0], -probs.log_prob(action), probs.entropy()

        elif isinstance(env.action_space, Box):
            probs = Normal(logits, std)
            action = probs.sample()
            clipped_action = torch.clamp(action, torch.min(torch.Tensor(env.action_space.low)), torch.min(torch.Tensor(env.action_space.high)))
            actions[step], neglogprobs[step], entropys[step] = clipped_action.tolist()[0], -probs.log_prob(action).sum(), probs.entropy().sum()
    
        elif isinstance(env.action_space, MultiDiscrete):
            logits_categories = torch.split(logits, env.action_space.nvec.tolist(), dim=1)
            action = []
            probs_categories = []
            probs_entropies = torch.zeros((logits.shape[0]))
            neglogprob = torch.zeros((logits.shape[0]))
            for i in range(len(logits_categories)):
                probs_categories.append(Categorical(logits=logits_categories[i]))
                if len(action) != env.action_space.shape:
                    action.append(probs_categories[i].sample())
                neglogprob -= probs_categories[i].log_prob(action[i])
                probs_entropies += probs_categories[i].entropy()
            action = torch.stack(action).transpose(0, 1).tolist()
            actions[step], neglogprobs[step], entropys[step] = action[0], neglogprob, probs_entropies
        
        # TRY NOT TO MODIFY: execute the game and log data.
        next_obs, rewards[step], dones[step], _ = env.step(actions[step])
        next_obs = np.array(next_obs)
        if dones[step]:
            break

    # ALGO LOGIC: training.
    # calculate the discounted rewards, or namely, returns
    num_steps = 5
    c_step = 0
    done_idx = dones.argmax()
    while c_step < done_idx:
        end_idx = min(c_step+num_steps, done_idx+1)
        batch_rewards = rewards[c_step:end_idx]
        with torch.no_grad():
            if c_step+num_steps >= done_idx:
                next_value = torch.Tensor([[0]]).to(device)# vf.forward([next_obs]) 
            else:
                next_value = vf.forward([obs[end_idx]])
        batch_returns = np.append(np.zeros_like(batch_rewards), next_value[0][0].detach().cpu())
        for t in reversed(range(batch_rewards.shape[0])):
            batch_returns[t] = batch_rewards[t] + args.gamma * batch_returns[t+1] * (1-dones[t])
        batch_returns = batch_returns[:-1]
        # advantages are batch_returns - baseline, value estimates in our case
        advantages = batch_returns - values[c_step:end_idx].detach().cpu().numpy()
        
        vf_loss = loss_fn(torch.Tensor(batch_returns).to(device), values[c_step:end_idx]) * args.vf_coef
        pg_loss = torch.Tensor(advantages).to(device) * neglogprobs[c_step:end_idx]
        loss = (pg_loss - entropys[c_step:end_idx] * args.ent_coef).mean() + vf_loss
        if torch.isnan(loss):
            raise Exception()
        
        optimizer.zero_grad()
        loss.backward(retain_graph=True)
        nn.utils.clip_grad_norm_(list(pg.parameters()) + list(vf.parameters()), args.max_grad_norm)
        optimizer.step()
        c_step += num_steps
        
    # returns = np.zeros_like(rewards)
    # for t in reversed(range(rewards.shape[0]-1)):
    #     returns[t] = rewards[t] + args.gamma * returns[t+1] * (1-dones[t])
    # # advantages are returns - baseline, value estimates in our case
    # advantages = returns - values.detach().cpu().numpy()
    
    # vf_loss = loss_fn(torch.Tensor(returns).to(device), values) * args.vf_coef
    # pg_loss = torch.Tensor(advantages).to(device) * neglogprobs
    # loss = (pg_loss - entropys * args.ent_coef).mean() + vf_loss
    
    # optimizer.zero_grad()
    # loss.backward()
    # nn.utils.clip_grad_norm_(list(pg.parameters()) + list(vf.parameters()), args.max_grad_norm)
    # optimizer.step()

    # TRY NOT TO MODIFY: record rewards for plotting purposes
    writer.add_scalar("charts/episode_reward", rewards.sum(), global_step)
    writer.add_scalar("losses/value_loss", vf_loss.item(), global_step)
    writer.add_scalar("losses/entropy", entropys[:step].mean().item(), global_step)
    writer.add_scalar("losses/policy_loss", pg_loss.mean().item(), global_step)
env.close()
writer.close()
