# Reference: https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from cleanrl.common import preprocess_obs_space, preprocess_ac_space
import argparse
import collections
import numpy as np
import gym
from gym.wrappers import TimeLimit, Monitor, AtariPreprocessing
from gym.spaces import Discrete, Box, MultiBinary, MultiDiscrete, Space
import time
import random
import os

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='C51 agent')
    # Common arguments
    parser.add_argument('--exp-name', type=str, default=os.path.basename(__file__).rstrip(".py"),
                       help='the name of this experiment')
    parser.add_argument('--gym-id', type=str, default="CartPole-v0",
                       help='the id of the gym environment')
    parser.add_argument('--learning-rate', type=float, default=1e-4,
                       help='the learning rate of the optimizer')
    parser.add_argument('--seed', type=int, default=2,
                       help='seed of the experiment')
    parser.add_argument('--episode-length', type=int, default=200,
                       help='the maximum length of each episode')
    parser.add_argument('--total-timesteps', type=int, default=40000,
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
    parser.add_argument('--buffer-size', type=int, default=10000,
                        help='the replay memory buffer size')
    parser.add_argument('--gamma', type=float, default=0.99,
                       help='the discount factor gamma')
    parser.add_argument('--target-network-frequency', type=int, default=1000,
                       help="the timesteps it takes to update the target network")
    parser.add_argument('--max-grad-norm', type=float, default=0.5,
                       help='the maximum norm for the gradient clipping')
    parser.add_argument('--batch-size', type=int, default=32,
                       help="the batch size of sample from the reply memory")
    parser.add_argument('--start-e', type=float, default=1,
                       help="the starting epsilon for exploration")
    parser.add_argument('--end-e', type=float, default=0.02,
                       help="the ending epsilon for exploration")
    parser.add_argument('--exploration-fraction', type=float, default=0.50,
                       help="the fraction of `total-timesteps` it takes from start-e to go end-e")
    parser.add_argument('--learning-starts', type=int, default=10000,
                       help="timestep to start learning")
    parser.add_argument('--train-frequency', type=int, default=1,
                       help="the frequency of training")
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
assert isinstance(env.action_space, Discrete), "only discrete action space is supported"
assert isinstance(env, TimeLimit) or int(args.episode_length), "the gym env does not have a built in TimeLimit, please specify by using --episode-length"
if args.capture_video:
    env = Monitor(env, f'videos/{experiment_name}')

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

# ALGO LOGIC: initialize agent here:
class QNetwork(nn.Module):
    def __init__(self, frames=4, n_atoms=101, v_min=-100, v_max=100):
        super(QNetwork, self).__init__()
        self.n_atoms = n_atoms
        self.atoms = torch.linspace(v_min, v_max, steps=n_atoms).to(device)
        self.network = nn.Sequential(
            nn.Linear(input_shape, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, env.action_space.n * n_atoms)
        )

    def forward(self, x):
        x = torch.Tensor(x).to(device)
        return self.network(x)

    def get_action(self, x, action=None):
        logits = self.forward(x)
        # probability mass function for each action
        pmfs = torch.softmax(logits.view(len(x), env.action_space.n, self.n_atoms), dim=2)
        q_values = (pmfs*self.atoms).sum(2)
        if action is None:
            action = torch.argmax(q_values, 1)
        return action, pmfs[torch.arange(len(x)), action]

def linear_schedule(start_e: float, end_e: float, duration: int, t: int):
    slope =  (end_e - start_e) / duration
    return max(slope * t + start_e, end_e)

rb = ReplayBuffer(args.buffer_size)
q_network = QNetwork().to(device)
target_network = QNetwork().to(device)
target_network.load_state_dict(q_network.state_dict())
optimizer = optim.Adam(q_network.parameters(), lr=args.learning_rate)
loss_fn = nn.MSELoss()
print(device.__repr__())
print(q_network)

# TRY NOT TO MODIFY: start the game
global_step = 0
while global_step < args.total_timesteps:
    next_obs = np.array(env.reset())
    actions = np.empty((args.episode_length,), dtype=object)
    rewards, dones, td_losses = np.zeros((3, args.episode_length))
    obs = np.empty((args.episode_length,) + env.observation_space.shape)
    
    # TRY NOT TO MODIFY: prepare the execution of the game.
    for step in range(args.episode_length):
        global_step += 1
        obs[step] = next_obs.copy()

        # ALGO LOGIC: put action logic here
        epsilon = linear_schedule(args.start_e, args.end_e, args.exploration_fraction*args.total_timesteps, global_step)
        if random.random() < epsilon:
            actions[step] = env.action_space.sample()
        else:
            action, pmf = target_network.get_action(obs[step:step+1])
            actions[step] = action.tolist()[0]

        # TRY NOT TO MODIFY: execute the game and log data.
        next_obs, rewards[step], dones[step], _ = env.step(actions[step])
        rb.put((obs[step], actions[step], rewards[step], next_obs, dones[step]))
        next_obs = np.array(next_obs)

        # ALGO LOGIC: training.
        if global_step > args.learning_starts and global_step % args.train_frequency == 0:
            s_obs, s_actions, s_rewards, s_next_obses, s_dones = rb.sample(args.batch_size)
            with torch.no_grad():
                _, next_pmfs = q_network.get_action(s_next_obses)
                next_atoms = torch.Tensor(s_rewards).to(device).unsqueeze(-1) + args.gamma * q_network.atoms  * (1 - torch.Tensor(s_dones).to(device).unsqueeze(-1))
    
                # projection
                v_min = -100
                v_max = 100
                delta_z = q_network.atoms[1]-q_network.atoms[0]
                tz = next_atoms.clamp(v_min, v_max)
                
                b = (tz - v_min)/ delta_z
                l = b.floor()
                u = b.ceil()
                d_m_l = (u + (l == u).float() - b) * next_pmfs # why (l == u).float() ?
                d_m_u = (b - l) * next_pmfs
                target_pmfs = torch.zeros_like(next_pmfs)
                for i in range(target_pmfs.size(0)):
                    target_pmfs[i].index_add_(0, l[i].long(), d_m_l[i])
                    target_pmfs[i].index_add_(0, u[i].long(), d_m_u[i])
                print(u[i].long())                

    
                # target_pmfs = next_pmfs * 0
                # b = (tz - v_min) / delta_z
                # l = b.floor().clamp(0, len(next_atoms) - 1)
                # u = b.ceil().clamp(0, len(next_atoms) - 1)
                # offset = (
                #     torch.linspace(0, (args.batch_size - 1) * 101, args.batch_size)
                #     .long()
                #     .unsqueeze(1)
                #     .expand(args.batch_size, 101)
                # )
                # target_pmfs.view(-1).index_add_(
                #     0, (l.long() + offset).view(-1), (next_pmfs * (u - b)).view(-1)
                # )
                # target_pmfs.view(-1).index_add_(
                #     0, (u.long() + offset).view(-1), (next_pmfs * (b - l)).view(-1)
                # )
                
                # log_dist = torch.log(torch.clamp(old_pmfs, min=1.5e-4))
                # log_target_dist = torch.log(torch.clamp(target_pmfs, min=1.5e-4))
                # loss = (target_pmfs * (log_target_dist - log_dist)).sum(dim=-1).mean()
            
            _, old_pmfs = q_network.get_action(s_obs, s_actions)
            # log_dist = torch.log(torch.clamp(old_pmfs, min=1.5e-4))
            # log_target_dist = torch.log(torch.clamp(target_pmfs, min=1.5e-4))
            # loss = (target_pmfs * (log_target_dist - log_dist)).sum(dim=-1).mean()
            loss = (-(target_pmfs.detach() * old_pmfs.log()).sum(-1)).mean()
            td_losses[step] = loss
            # print(loss.item())

            # optimize the midel
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(list(q_network.parameters()), args.max_grad_norm)
            optimizer.step()

            # update the target network
            if global_step % args.target_network_frequency == 0:
                target_network.load_state_dict(q_network.state_dict())

        if dones[step]:
            break

    # TRY NOT TO MODIFY: record rewards for plotting purposes
    print(f"global_step={global_step}, episode_reward={rewards.sum()}")
    writer.add_scalar("charts/episode_reward", rewards.sum(), global_step)
    writer.add_scalar("charts/epsilon", epsilon, global_step)
    writer.add_scalar("losses/td_loss", td_losses[:step+1].mean(), global_step)
env.close()
writer.close()