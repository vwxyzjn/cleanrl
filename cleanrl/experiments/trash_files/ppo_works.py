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
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
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
        self.logstd = nn.Linear(output_shape, output_shape)

    def forward(self, x):
        x = preprocess_obs_fn(x)
        x = F.tanh(self.fc1(x))
        x = F.tanh(self.fc2(x))
        
        action_mean = self.mean(x)
        zeros = torch.zeros(action_mean.size(), device=device)
        action_logstd = self.logstd(zeros)
        return action_mean, action_logstd.exp()

    def get_action(self, x):
        mean, std = self.forward(x)
        probs = Normal(mean, std)
        action = probs.sample()
        clipped_action = torch.clamp(action, torch.min(torch.Tensor(env.action_space.low)), torch.min(torch.Tensor(env.action_space.high)))
        return clipped_action, -probs.log_prob(action).sum(1), probs.entropy().sum(1)
    
    def get_logprob(self, x, a):
        mean, std = self.forward(x)
        probs = Normal(mean, std)
        return probs.log_prob(a).sum(1)
from collections import namedtuple
Transition = namedtuple('Transition', ('state', 'value', 'action', 'logproba', 'mask', 'next_state', 'reward'))
EPS = 1e-10
class ActorCritic(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super(ActorCritic, self).__init__()
        
        self.actor_fc1 = nn.Linear(num_inputs, 64)
        self.actor_fc2 = nn.Linear(64, 64)
        self.actor_fc3 = nn.Linear(64, num_outputs)
        self.actor_logstd = nn.Parameter(torch.zeros(1, num_outputs))

        self.critic_fc1 = nn.Linear(num_inputs, 64)
        self.critic_fc2 = nn.Linear(64, 64)
        self.critic_fc3 = nn.Linear(64, 1)

    def forward(self, states):
        """
        run policy network (actor) as well as value network (critic)
        :param states: a Tensor2 represents states
        :return: 3 Tensor2
        """
        action_mean, action_logstd = self._forward_actor(states)
        critic_value = self._forward_critic(states)
        return action_mean, action_logstd, critic_value

    def _forward_actor(self, states):
        x = torch.tanh(self.actor_fc1(states))
        x = torch.tanh(self.actor_fc2(x))
        action_mean = self.actor_fc3(x)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        return action_mean, action_logstd

    def _forward_critic(self, states):
        x = torch.tanh(self.critic_fc1(states))
        x = torch.tanh(self.critic_fc2(x))
        critic_value = self.critic_fc3(x)
        return critic_value

    def select_action(self, action_mean, action_logstd, return_logproba=True):
        """
        given mean and std, sample an action from normal(mean, std)
        also returns probability of the given chosen
        """
        action_std = torch.exp(action_logstd)
        dist = Normal(action_mean, action_std)
        action = dist.sample() # torch.normal(action_mean, action_std)
        if return_logproba:
            logproba = dist.log_prob(action).sum(1) # self._normal_logproba(action, action_mean, action_logstd, action_std)
        return action, logproba

    def get_logproba(self, states, actions):
        """
        return probability of chosen the given actions under corresponding states of current network
        :param states: Tensor
        :param actions: Tensor
        """
        action_mean, action_logstd = self._forward_actor(states)
        action_std = torch.exp(action_logstd)
        dist = Normal(action_mean, action_std)
        return dist.log_prob(actions).sum(1)
    
    def get_entropy(self, states):
        """
        return probability of chosen the given actions under corresponding states of current network
        :param states: Tensor
        :param actions: Tensor
        """
        action_mean, action_logstd = self._forward_actor(states)
        action_std = torch.exp(action_logstd)
        dist = Normal(action_mean, action_std)
        return dist.entropy()
    
class Memory(object):
    def __init__(self):
        self.memory = []

    def push(self, *args):
        self.memory.append(Transition(*args))

    def sample(self):
        return Transition(*zip(*self.memory))

    def __len__(self):
        return len(self.memory)

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
print(pg.logstd.bias)

env = gym.make('HopperBulletEnv-v0')
num_inputs = env.observation_space.shape[0]
num_actions = env.action_space.shape[0]

env.seed(args.seed)
torch.manual_seed(args.seed)

network = ActorCritic(num_inputs, num_actions)
optimizer = optim.Adam(network.parameters(), lr=args.learning_rate)

# TRY NOT TO MODIFY: start the game
global_step = 0
while global_step < args.total_timesteps:
    state = env.reset()
    reward_sum = 0
    memory = Memory()

    next_obs = np.array(env.reset())
    actions = np.empty((args.episode_length,) + env.action_space.shape)
    rewards, dones = np.zeros((2, args.episode_length))
    obs = np.empty((args.episode_length,) + env.observation_space.shape)

    # ALGO LOGIC: put other storage logic here
    values = torch.zeros((args.episode_length), device=device)
    neglogprobs = torch.zeros((args.episode_length,), device=device)
    entropys = torch.zeros((args.episode_length,), device=device)

    # TRY NOT TO MODIFY: prepare the execution of the game.
    for step in range(args.episode_length):
        global_step += 1
        obs[step] = next_obs.copy()
        
        action_mean, action_logstd, value = network(torch.Tensor(obs[step]).unsqueeze(0))
        # value = vf.forward(obs[step:step+1])
        action, logproba = network.select_action(action_mean, action_logstd)
        action = action.data.numpy()[0]
        logproba = logproba.data.numpy()[0]
        next_state, reward, done, _ = env.step(action)
        reward_sum += reward
        mask = 0 if done else 1

        memory.push(obs[step], value, action, logproba, mask, next_state, reward)
        
        if done:
            break
        
        next_obs = np.array(next_state)

    # ALGO LOGIC: training.
    # calculate the discounted rewards, or namely, returns
    returns = np.zeros_like(rewards)
    for t in reversed(range(rewards.shape[0]-1)):
        returns[t] = rewards[t] + args.gamma * returns[t+1] * (1-dones[t])
    # advantages are returns - baseline, value estimates in our case
    advantages = returns - values.detach().cpu().numpy()
    
    
    batch = memory.sample()
    batch_size = len(memory)
    # step2: extract variables from trajectories
    rewards = torch.Tensor(batch.reward)
    values = torch.Tensor(batch.value)
    masks = torch.Tensor(batch.mask)
    actions = torch.Tensor(batch.action)
    states = torch.Tensor(batch.state)
    oldlogproba = torch.Tensor(batch.logproba)
    
    returns = torch.Tensor(batch_size)
    deltas = torch.Tensor(batch_size)
    advantages = torch.Tensor(batch_size)

    prev_return = 0
    prev_value = 0
    prev_advantage = 0
    for i in reversed(range(batch_size)):
        returns[i] = rewards[i] + args.gamma * prev_return * masks[i]
        deltas[i] = rewards[i] + args.gamma * prev_value * masks[i] - values[i]
        # ref: https://arxiv.org/pdf/1506.02438.pdf (generalization advantage estimate)
        advantages[i] = deltas[i] + args.gamma * 0.97 * prev_advantage * masks[i]

        prev_return = returns[i]
        prev_value = values[i]
        prev_advantage = advantages[i]
    # if args.advantage_norm:
    #     advantages = (advantages - advantages.mean()) / (advantages.std() + EPS)

    for i_epoch in range(3):
        # sample from current batch
        minibatch_states = states
        minibatch_actions = actions
        minibatch_oldlogproba = oldlogproba
        minibatch_newlogproba = network.get_logproba(minibatch_states, minibatch_actions)
        minibatch_advantages = advantages
        minibatch_returns = returns
        minibatch_newvalues = network._forward_critic(minibatch_states).flatten()

        ratio =  torch.exp(minibatch_newlogproba - minibatch_oldlogproba)
        surr1 = ratio * minibatch_advantages
        surr2 = ratio.clamp(1 - args.clip_coef, 1 + args.clip_coef) * minibatch_advantages
        loss_surr = - torch.mean(torch.min(surr1, surr2))

        # not sure the value loss should be clipped as well 
        # clip example: https://github.com/Jiankai-Sun/Proximal-Policy-Optimization-in-Pytorch/blob/master/ppo.py
        # however, it does not make sense to clip score-like value by a dimensionless clipping parameter
        # moreover, original paper does not mention clipped value 
        loss_value = torch.mean((minibatch_newvalues - minibatch_returns).pow(2))

        loss_entropy = torch.mean(torch.exp(minibatch_newlogproba) * minibatch_newlogproba)

        total_loss = loss_surr + args.vf_coef * loss_value + args.ent_coef * loss_entropy
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
    # neglogprobs = neglogprobs.detach()
    # non_empty_idx = np.argmax(dones) + 1
    # for _ in range(args.update_epochs):
    #     # ALGO LOGIC: `env.action_space` specific logic
    #     new_neglogprobs = pg.get_logprob(obs[:non_empty_idx], torch.Tensor(actions[:non_empty_idx]).to(device))
    #     ratio = torch.exp(neglogprobs[:non_empty_idx] - new_neglogprobs)
    #     surrogate1 = ratio * torch.Tensor(advantages)[:non_empty_idx].to(device)
    #     surrogate2 = torch.clamp(ratio, 1-args.clip_coef, 1+args.clip_coef) * torch.Tensor(advantages)[:non_empty_idx].to(device)
    #     policy_loss = -torch.min(surrogate1, surrogate2).mean()
    #     vf_loss = loss_fn(torch.Tensor(returns).to(device), values) * args.vf_coef
    #     entropy_loss = torch.mean(torch.exp(new_neglogprobs) * new_neglogprobs)
    #     loss = vf_loss + policy_loss + (entropy_loss * args.ent_coef).mean()
        
        
        
    #     optimizer.zero_grad()
    #     loss.backward(retain_graph=True)
    #     nn.utils.clip_grad_norm_(list(pg.parameters()) + list(vf.parameters()), args.max_grad_norm)
    #     optimizer.step()


    # TRY NOT TO MODIFY: record rewards for plotting purposes
    writer.add_scalar("charts/episode_reward",reward_sum, global_step)
    #writer.add_scalar("losses/value_loss", vf_loss.item(), global_step)
    writer.add_scalar("losses/entropy", entropys[:step+1].mean().item(), global_step)
    #writer.add_scalar("losses/policy_loss", policy_loss.item(), global_step)
env.close()
writer.close()
