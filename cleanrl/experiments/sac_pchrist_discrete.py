# primary references:
# https://arxiv.org/pdf/1910.07207.pdf
# https://github.com/p-christ/Deep-Reinforcement-Learning-Algorithms-with-PyTorch

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
    parser.add_argument('--gym-id', type=str, default="CartPole-v0",
                       help='the id of the gym environment')
    parser.add_argument('--learning-rate', type=float, default=7e-4,
                       help='the learning rate of the optimizer')
    parser.add_argument('--seed', type=int, default=2,
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
    parser.add_argument('--buffer-size', type=int, default=50000,
                        help='the replay memory buffer size')
    parser.add_argument('--gamma', type=float, default=0.99,
                       help='the discount factor gamma')
    parser.add_argument('--target-network-frequency', type=int, default=500,
                       help="the timesteps it takes to update the target network")
    parser.add_argument('--max-grad-norm', type=float, default=0.5,
                       help='the maximum norm for the gradient clipping')
    parser.add_argument('--batch-size', type=int, default=32,
                       help="the batch size of sample from the reply memory")
    parser.add_argument('--tau', type=float, default=0.005,
                       help="target smoothing coefficient (default: 0.005)")
    parser.add_argument('--alpha', type=float, default=0.2,
                       help="Entropy regularization coefficient.")
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
assert isinstance(env.action_space, Discrete), "only discrete action space is supported"

# ALGO LOGIC: initialize agent here:
class Policy(nn.Module):
    def __init__(self):
        super(Policy, self).__init__()
        
        self.linear1 = nn.Linear(input_shape, 120)
        self.linear2 = nn.Linear(120, 84)

        self.fc3 = nn.Linear(84, output_shape)

        # # action rescaling
        # self.action_scale = torch.FloatTensor(
        #     (env.action_space.high - env.action_space.low) / 2.)
        # self.action_bias = torch.FloatTensor(
        #     (env.action_space.high + env.action_space.low) / 2.)
    
    def forward(self, x):
        x = preprocess_obs_fn(x)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = self.fc3(x)
        return x

    def sample(self, state):
        logits = self.forward(state)
        probs = Categorical(logits=logits)
        action = probs.sample()
        # mean = None

        # std = log_std.exp()
        # normal = Normal(mean, std)
        # x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        # y_t = torch.tanh(x_t)
        # action = y_t * self.action_scale + self.action_bias
        # log_prob = normal.log_prob(x_t)
        # # Enforcing Action Bound
        # log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) +  1e-6)
        # log_prob = log_prob.sum(1, keepdim=True)
        # mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, torch.log(probs.probs), probs

    # def to(self, device):
    #     self.action_scale = self.action_scale.to(device)
    #     self.action_bias = self.action_bias.to(device)
    #     return super(Policy, self).to(device)

class SoftQNetwork(nn.Module):
    def __init__(self):
        super(SoftQNetwork, self).__init__()
        self.fc1 = nn.Linear(input_shape, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, output_shape)

    def forward(self, x):
        x = preprocess_obs_fn(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class ReplayBuffer(object):
    def __init__(self, size):
        self._storage = []
        self._maxsize = size
        self._next_idx = 0

    def add(self, obs_t, action, reward, obs_tp1, done):
        data = (obs_t, action, reward, obs_tp1, done)
        if self._next_idx >= len(self._storage):
            self._storage.append(data)
        else:
            self._storage[self._next_idx] = data
        self._next_idx = (self._next_idx + 1) % self._maxsize

    def sample(self, batch_size):
        idxes = [random.randint(0, len(self._storage) - 1) for _ in range(batch_size)]
        obses_t, actions, rewards, obses_tp1, dones = [], [], [], [], []
        for i in idxes:
            data = self._storage[i]
            obs_t, action, reward, obs_tp1, done = data
            obses_t.append(np.array(obs_t, copy=False))
            actions.append(np.array(action, copy=False))
            rewards.append(reward)
            obses_tp1.append(np.array(obs_tp1, copy=False))
            dones.append(done)
        return np.array(obses_t), np.array(actions), np.array(rewards), np.array(obses_tp1), np.array(dones)

er = ReplayBuffer(args.buffer_size)
pg = Policy().to(device)
soft_q_network1 = SoftQNetwork().to(device)
soft_q_network2 = SoftQNetwork().to(device)
soft_q_network1_target = SoftQNetwork().to(device)
soft_q_network2_target = SoftQNetwork().to(device)
soft_q_network1_target.load_state_dict(soft_q_network1.state_dict())
soft_q_network2_target.load_state_dict(soft_q_network2.state_dict())
values_optimizer = optim.Adam(list(soft_q_network1.parameters()) + list(soft_q_network2.parameters()), lr=args.learning_rate)
policy_optimizer = optim.Adam(list(pg.parameters()), lr=args.learning_rate)
loss_fn = nn.MSELoss()
min_Val = torch.tensor(1e-7).float()

# TRY NOT TO MODIFY: start the game
global_step = 0
while global_step < args.total_timesteps:
    next_obs = np.array(env.reset())
    actions = np.empty((args.episode_length,), dtype=object)
    rewards, dones = np.zeros((2, args.episode_length))
    obs = np.empty((args.episode_length,) + env.observation_space.shape)
    
    # ALGO LOGIC: put other storage logic here
    values = torch.zeros((args.episode_length), device=device)
    entropys = torch.zeros((args.episode_length,), device=device)
    
    # TRY NOT TO MODIFY: prepare the execution of the game.
    for step in range(args.episode_length):
        global_step += 1
        obs[step] = next_obs.copy()
        
        # ALGO LOGIC: put action logic here
        action, _, _ = pg.sample(obs[step:step+1])
        #values[step] = vf.forward(obs[step:step+1])

        # ALGO LOGIC: `env.action_space` specific logic
        actions[step] = action.tolist()[0]
    
        # TRY NOT TO MODIFY: execute the game and log data.
        next_obs, rewards[step], dones[step], _ = env.step(action.tolist()[0])
        next_obs = np.array(next_obs)
        er.add(obs[step], actions[step], rewards[step], next_obs, dones[step])
        if dones[step]:
            break
        
        # ALGO LOGIC: training.
        if len(er._storage) > 2000:
            s_obs, s_actions, s_rewards, s_next_obses, s_dones = er.sample(args.batch_size)
            with torch.no_grad():
                next_state_action, next_state_log_pi, probs = pg.sample(s_next_obses)
                qf1_next_target = soft_q_network1_target.forward(s_next_obses)
                qf2_next_target = soft_q_network2_target.forward(s_next_obses)
                min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - args.alpha * next_state_log_pi
                min_qf_next_target *= probs.probs
                min_qf_next_target = min_qf_next_target.mean(dim=1)
                next_q_value = torch.Tensor(s_rewards).to(device) + (1 - torch.Tensor(s_dones).to(device)) * args.gamma * (min_qf_next_target).view(-1)

            qf1 = soft_q_network1.forward(s_obs).gather(1,torch.LongTensor(s_actions).to(device).unsqueeze(1)).view(-1)
            qf2 = soft_q_network2.forward(s_obs).gather(1,torch.LongTensor(s_actions).to(device).unsqueeze(1)).view(-1)
            # qf1, qf2 = critic(state_batch, action_batch)  # Two Q-functions to mitigate positive bias in the policy improvement step
            qf1_loss = loss_fn(qf1, next_q_value)
            qf2_loss = loss_fn(qf2, next_q_value)

            pi, log_pi, probs = pg.sample(s_obs)

            qf1_pi = soft_q_network1.forward(s_obs)
            qf2_pi = soft_q_network2.forward(s_obs)
            min_qf_pi = torch.min(qf1_pi, qf2_pi)
            inside_term = args.alpha * log_pi - min_qf_pi
            policy_loss = probs.probs * inside_term
            policy_loss = policy_loss.mean()

            values_optimizer.zero_grad()
            qf1_loss.backward()
            values_optimizer.step()

            values_optimizer.zero_grad()
            qf2_loss.backward()
            values_optimizer.step()
            
            policy_optimizer.zero_grad()
            policy_loss.backward()
            policy_optimizer.step()

            # writer.add_scalar("losses/soft_value_loss", v_loss.item(), global_step)
            writer.add_scalar("losses/soft_q_value_1_loss", qf1_loss.item(), global_step)
            writer.add_scalar("losses/soft_q_value_2_loss", qf2_loss.item(), global_step)
            writer.add_scalar("losses/policy_loss", policy_loss.item(), global_step)
            #writer.add_scalar("losses/loss", loss.item(), global_step)

            # update the target network
            for param, target_param in zip(soft_q_network1.parameters(), soft_q_network1_target.parameters()):
                target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)
            for param, target_param in zip(soft_q_network2.parameters(), soft_q_network2_target.parameters()):
                target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)

    # TRY NOT TO MODIFY: record rewards for plotting purposes
    writer.add_scalar("charts/episode_reward", rewards.sum(), global_step)
    writer.add_scalar("losses/entropy", entropys[:step+1].mean().item(), global_step)
writer.close()
