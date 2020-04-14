# Reference: https://github.com/seungeunrho/minimalRL/blob/master/acer.py

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
from gym.spaces import Discrete, Box, MultiBinary, MultiDiscrete, Space
import time
import random
import os
import collections

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='ACER agent')
    # Common arguments
    parser.add_argument('--exp-name', type=str, default=os.path.basename(__file__).rstrip(".py"),
                       help='the name of this experiment')
    parser.add_argument('--gym-id', type=str, default="CartPole-v0",
                       help='the id of the gym environment')
    parser.add_argument('--learning-rate', type=float, default=7e-4,
                       help='the learning rate of the optimizer')
    parser.add_argument('--seed', type=int, default=1,
                       help='seed of the experiment')
    parser.add_argument('--episode-length', type=int, default=0,
                       help='the maximum length of each episode')
    parser.add_argument('--total-timesteps', type=int, default=200000,
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

    # Algorithm specific 
    parser.add_argument('--c', type=int, default=1,
                       help="the clip parameter for ACER")
    parser.add_argument('--buffer-size', type=int, default=10000,
                        help='the replay memory buffer size')
    parser.add_argument('--batch-size', type=int, default=4,
                       help="the batch size of sample from the reply memory")
    parser.add_argument('--gamma', type=float, default=0.99,
                       help='the discount factor gamma')
    parser.add_argument('--vf-coef', type=float, default=0.25,
                       help="value function's coefficient the loss function")
    parser.add_argument('--max-grad-norm', type=float, default=0.5,
                       help='the maximum norm for the gradient clipping')
    parser.add_argument('--ent-coef', type=float, default=0.01,
                       help="policy entropy's coefficient the loss function")
    parser.add_argument('--rollout-steps', type=int, default=10,
                       help='the number of rollout steps')
    parser.add_argument('--learning-starts', type=int, default=500,
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
assert isinstance(env, TimeLimit) or int(args.episode_length), "the gym env does not have a built in TimeLimit, please specify by using --episode-length"
if isinstance(env, TimeLimit):
    if int(args.episode_length):
        env._max_episode_steps = int(args.episode_length)
    args.episode_length = env._max_episode_steps
else:
    env = TimeLimit(env, int(args.episode_length))
if args.capture_video:
    env = Monitor(env, f'videos/{experiment_name}')

# ALGO LOGIC: initialize agent here:
# modified from https://github.com/seungeunrho/minimalRL/blob/master/dqn.py#
class ReplayBuffer():
    def __init__(self, buffer_limit):
        self.buffer = collections.deque(maxlen=buffer_limit)

    def put(self, seq_data):
        self.buffer.append(seq_data)
    
    def sample(self, n, on_policy=False):
        if on_policy:
            mini_batch = [self.buffer[-1]]
        else:
            mini_batch = random.sample(self.buffer, n)

        s_lst, a_lst, r_lst, prob_lst, done_lst, is_first_lst = [], [], [], [], [], []
        for seq in mini_batch:
            is_first = True  # Flag for indicating whether the transition is the first item from a sequence
            for transition in seq:
                s, a, r, prob, done = transition

                s_lst.append(s)
                a_lst.append([a])
                r_lst.append(r)
                prob_lst.append(prob)
                done_mask = 0.0 if done else 1.0
                done_lst.append(done_mask)
                is_first_lst.append(is_first)
                is_first = False

        s,a,r,prob,done_mask,is_first = np.array(s_lst), np.array(a_lst), \
                                        np.array(r_lst), np.array(prob_lst), np.array(done_lst), \
                                        np.array(is_first_lst)
        return s,a,r,prob,done_mask,is_first


class Policy(nn.Module):
    def __init__(self):
        super(Policy, self).__init__()
        self.fc1 = nn.Linear(input_shape, 120)
        self.fc2 = nn.Linear(120, 84)
        self.action_fc = nn.Linear(84, output_shape)
        self.q_fc = nn.Linear(84, output_shape)

    def forward(self, x):
        x = preprocess_obs_fn(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return x
    
    def action(self, x):
        x = self.action_fc(self.forward(x))
        return x
    
    def q(self, x):
        x = self.q_fc(self.forward(x))
        return x

rb = ReplayBuffer(args.buffer_size)
pg = Policy().to(device)
optimizer = optim.Adam(list(pg.parameters()), lr=args.learning_rate)
loss_fn = nn.MSELoss()

# TRY NOT TO MODIFY: start the game
global_step = 0
q_retrace = 0
while global_step < args.total_timesteps:
    next_obs = np.array(env.reset())
    obs = np.empty((args.episode_length,) + env.observation_space.shape)

    # ALGO LOGIC: put other storage logic here
    values = torch.zeros((args.episode_length), device=device)
    neglogprobs = torch.zeros((args.episode_length,), device=device)
    entropys = torch.zeros((args.episode_length,), device=device)

    # TRY NOT TO MODIFY: prepare the execution of the game.
    done = False
    episode_reward = 0
    while not done:
        seq_data = []
        for step in range(args.rollout_steps):
            global_step += 1
            obs = next_obs.copy()
    
            # ALGO LOGIC: put action logic here
            logits = pg.action(obs.reshape(1,-1))
            # values[step] = pg.q(obs[step:step+1])
    
            # ALGO LOGIC: `env.action_space` specific logic
            probs = Categorical(logits=logits)
            action = probs.sample().item()

            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, reward, done, _ = env.step(action)
            episode_reward += reward
            seq_data += [(obs, action, reward, probs.probs.detach().cpu().numpy()[0], done)]
            next_obs = np.array(next_obs)
            if done:
                print(f"global_step={global_step}, episode_reward={episode_reward}")
                # print(q_retrace)
                writer.add_scalar("charts/episode_reward", episode_reward, global_step)
                break
        
        rb.put(seq_data)
    
    
        if len(rb.buffer)>500:
            # raise
            # ALGO LOGIC: training.
            s_obs, s_actions, s_rewards, s_probs, s_dones, s_is_first = rb.sample(args.batch_size)
            q_values = pg.q(s_obs)
            action_q_values = q_values.gather(1, torch.LongTensor(s_actions).view(-1,1).to(device)).squeeze()
            logits = pg.action(s_obs)
            probs = Categorical(logits=logits)
            action_log_probs = probs.log_prob(torch.LongTensor(s_actions).flatten().to(device))
            
            with torch.no_grad():
                values = (q_values * probs.probs).sum(1)
                rho = probs.probs / torch.Tensor(s_probs).to(device)
                rho_bar = rho.clamp(max=args.c).gather(1, torch.LongTensor(s_actions).view(-1,1).to(device)).squeeze()

    
            # calculate the retrace Q
            # TODO: remember to change the dones implementation after experiments
            q_retrace = values[-1] * s_dones[-1]
            q_retrace_lst = []
            for i in reversed(range(len(s_rewards))):
                q_retrace = s_rewards[i] + args.gamma * q_retrace
                q_retrace_lst.append(q_retrace.item())
                q_retrace = rho_bar[i] * (q_retrace - action_q_values[i]) + values[i]
                
                if s_is_first[i] and i!=0:
                    q_retrace = values[i-1] * s_dones[i-1] # When a new sequence begins, q_retrace is initialized  
            q_retrace_lst.reverse()
            q_retrace = torch.tensor(q_retrace_lst, dtype=torch.float).to(device)
            loss1 = -rho_bar * action_log_probs * (q_retrace - values)
            
            # correction = (rho- args.c) / rho
            # correction_plus = correction.clamp(min=0)
            # log_probs = nn.LogSoftmax(1)(logits)
            # loss2 = correction_plus * probs.probs.detach() * log_probs * (q_values - values.unsqueeze(1))
            
            q_loss = F.mse_loss(q_retrace, action_q_values)
            
            loss = (loss1 ).mean() + q_loss # + loss2.sum(1)
        
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(list(pg.parameters()), args.max_grad_norm)
            optimizer.step()

    # TRY NOT TO MODIFY: record rewards for plotting purposes
    # writer.add_scalar("losses/value_loss", vf_loss.item(), global_step)
    # writer.add_scalar("losses/entropy", entropys[:step+1].mean().item(), global_step)
    # writer.add_scalar("losses/policy_loss", pg_loss.mean().item(), global_step)
env.close()
writer.close()
