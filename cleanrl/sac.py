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
import collections

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='SAC with 2 Q functions, Online updates')
    # Common arguments
    parser.add_argument('--exp-name', type=str, default=os.path.basename(__file__).rstrip(".py"),
                       help='the name of this experiment')
    parser.add_argument('--gym-id', type=str, default="Taxi-v3",
                       help='the id of the gym environment')
    parser.add_argument('--learning-rate', type=float, default=7e-4,
                       help='the learning rate of the optimizer')
    parser.add_argument('--seed', type=int, default=1,
                       help='seed of the experiment')
    parser.add_argument('--episode-length', type=int, default=0,
                       help='the maximum length of each episode')
    parser.add_argument('--total-timesteps', type=int, default=int(1e6),
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
    parser.add_argument('--buffer-size', type=int, default=int(1e5),
                        help='the replay memory buffer size')
    parser.add_argument('--gamma', type=float, default=0.99,
                       help='the discount factor gamma')
    parser.add_argument('--target-update-interval', type=int, default=1,
                       help="the timesteps it takes to update the target network")
    parser.add_argument('--batch-size', type=int, default=64,
                       help="the batch size of sample from the reply memory")
    parser.add_argument('--tau', type=float, default=0.005,
                       help="target smoothing coefficient (default: 0.005)")
    parser.add_argument('--alpha', type=float, default=0.2,
                       help="Entropy regularization coefficient.")
    parser.add_argument('--autotune', action='store_true',
                       help='Enables autotuning of the alpha entropy coefficient')
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
# Gaussian Policy
class Policy(nn.Module):
    def __init__(self):
        super(Policy, self).__init__()
        self.fc1 = nn.Linear(input_shape, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, output_shape)
        self.nn_softmax = torch.nn.Softmax(1)
        self.nn_log_softmax = torch.nn.LogSoftmax(1)

    def forward(self, x):
        x = preprocess_obs_fn(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return self.nn_softmax(x), self.nn_log_softmax(x)

    def get_action(self, x):
        action_probs, action_logps = self.forward(x)
        dist = Categorical(probs=action_probs)
        dist.entropy()
        return dist.sample(), action_probs, action_logps, dist.entropy().sum()

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

    def state_action_value(self, x, a):
        x = self.forward(x)
        action_values = x.gather(1, a.view(-1,1))   
        return action_values

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

rb = ReplayBuffer(args.buffer_size)
pg = Policy().to(device)
qf1 = SoftQNetwork().to(device)
qf2 = SoftQNetwork().to(device)
qf1_target = SoftQNetwork().to(device)
qf2_target = SoftQNetwork().to(device)
qf1_target.load_state_dict(qf1.state_dict())
qf2_target.load_state_dict(qf2.state_dict())
values_optimizer = optim.Adam(list(qf1.parameters()) + list(qf2.parameters()), lr=args.learning_rate)
policy_optimizer = optim.Adam(list(pg.parameters()), lr=args.learning_rate)
loss_fn = nn.MSELoss()
# TODO: Explain this part please
# MODIFIED: SAC Automatic Entropy Tuning support
if args.autotune:
    # This is only an Heuristic of the minimal entropy we should constraint to
    target_entropy = - np.prod(env.action_space.shape) # TODO: Better Heuristic target entropy for discrete case
    log_alpha = torch.Tensor([ 0.,]).to(device).requires_grad_()
    # Convoluted, but needed to maintain consistency when not using --autotune
    alpha = log_alpha.exp().cpu().detach().numpy()[0]
    a_optimizer = optim.Adam([log_alpha], lr=args.learning_rate) # TODO: Different learning rate for alpha ?
else:
    alpha = args.alpha

# TRY NOT TO MODIFY: start the game
global_step = 0
while global_step < args.total_timesteps:
    next_obs = np.array(env.reset())
    actions = np.empty((args.episode_length,), dtype=object)
    rewards, dones = np.zeros((2, args.episode_length))
    obs = np.empty((args.episode_length,) + env.observation_space.shape)
    # ALGO LOGIC: put other storage logic here
    entropys = torch.zeros((args.episode_length,), device=device)

    # TRY NOT TO MODIFY: prepare the execution of the game.
    for step in range(args.episode_length):
        global_step += 1
        obs[step] = next_obs.copy()

        # ALGO LOGIC: put action logic here
        action, _, _, entropys[step] = pg.get_action(obs[step:step+1])
        actions[step] = action.tolist()[0]

        # TRY NOT TO MODIFY: execute the game and log data.
        next_obs, rewards[step], dones[step], _ = env.step(action.tolist()[0])
        rb.put((obs[step], actions[step], rewards[step], next_obs, dones[step]))
        next_obs = np.array(next_obs)
        # ALGO LOGIC: training.
        if len(rb.buffer) > 2000:
            s_obs, s_actions, s_rewards, s_next_obses, s_dones = rb.sample(args.batch_size)
            with torch.no_grad():
                _, next_action_probs, next_logprobs, _ = pg.get_action(s_next_obses)

                qf1_target_values = qf1_target.forward(s_next_obses)
                qf2_target_values = qf2_target.forward(s_next_obses)

                min_qf_target_values = torch.min(qf1_target_values, qf2_target_values)
                ent_aug_min_qf_values = min_qf_target_values - alpha * next_logprobs

                v_next_target = (next_action_probs * ent_aug_min_qf_values).sum(1)
                q_backup = torch.Tensor(s_rewards).to(device) + \
                    args.gamma * (1. - torch.Tensor(s_dones).to(device)) * \
                    v_next_target

            qf1_a_values = qf1.state_action_value(s_obs, torch.LongTensor(s_actions).to(device)).view(-1)
            qf2_a_values = qf2.state_action_value(s_obs, torch.LongTensor(s_actions).to(device)).view(-1)

            qf1_loss = loss_fn(qf1_a_values, q_backup)
            qf2_loss = loss_fn(qf2_a_values, q_backup)
            qf_loss = qf1_loss + qf2_loss

            # Q param gradient step
            values_optimizer.zero_grad()
            qf_loss.backward()
            values_optimizer.step()

            # Policy loss
            _, action_probs, logps, _ = pg.get_action(s_obs)
            qf1_values = qf1.forward(s_obs)
            qf2_values = qf2.forward(s_obs)
            min_qf_values = torch.min(qf1_values, qf2_values)

            policy_loss = alpha * logps - min_qf_values
            policy_loss *= action_probs
            policy_loss = policy_loss.sum(1).mean()

            # Policy gradient step
            policy_optimizer.zero_grad()
            policy_loss.backward()
            policy_optimizer.step()

            # TODO: Alpha auto tune
            if args.autotune:
                with torch.no_grad():
                    _, action_probs, logps, _ = pg.get_action(s_obs)
                alpha_loss = action_probs * (- log_alpha.exp() * (logps + target_entropy))
                alpha_loss = alpha_loss.sum(1).mean()
                # Alpha gradient step
                a_optimizer.zero_grad()
                alpha_loss.backward()
                a_optimizer.step()
                # Update the alpha value used in other parts of the algorithm
                # TODO: More elegant way ?
                alpha = log_alpha.exp().cpu().detach().numpy()[0]

            # update the target network
            for param, target_param in zip(qf1.parameters(), qf1_target.parameters()):
                target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)
            for param, target_param in zip(qf2.parameters(), qf2_target.parameters()):
                target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)

            writer.add_scalar("losses/soft_q_value_1_loss", qf1_loss.item(), global_step)
            writer.add_scalar("losses/soft_q_value_2_loss", qf2_loss.item(), global_step)
            writer.add_scalar("losses/policy_loss", policy_loss.item(), global_step)
            
            if args.autotune:
                writer.add_scalar("losses/alpha_entropy_coef", log_alpha.exp(), global_step)

        if dones[step]:
            break
    # TRY NOT TO MODIFY: record rewards for plotting purposes
    writer.add_scalar("losses/entropy", entropys[:step].mean().item(), global_step)
    writer.add_scalar("charts/episode_reward", rewards.sum(), global_step)
    # writer.add_scalar("losses/entropy", entropys[:step+1].mean().item(), global_step)

writer.close()
