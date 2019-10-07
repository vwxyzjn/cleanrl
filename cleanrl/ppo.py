# Reference: http://inoryy.com/post/tensorflow2-deep-reinforcement-learning/

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tensorboardX import SummaryWriter

from cleanrl.common import preprocess_obs_space, preprocess_ac_space
import argparse
import numpy as np
import gym
import time
import random

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PPO agent')
    # Common arguments
    parser.add_argument('--exp-name', type=str, default="ppo",
                       help='the name of this experiment')
    parser.add_argument('--gym-id', type=str, default="CartPole-v0",
                       help='the id of the gym environment')
    parser.add_argument('--learning-rate', type=float, default=7e-4,
                       help='the learning rate of the optimizer')
    parser.add_argument('--seed', type=int, default=0,
                       help='seed of the experiment')
    parser.add_argument('--episode-length', type=int, default=200,
                       help='the maximum length of each episode')
    parser.add_argument('--total-timesteps', type=int, default=50000,
                       help='total timesteps of the experiments')
    parser.add_argument('--torch-deterministic', type=bool, default=True,
                       help='whether to set `torch.backends.cudnn.deterministic=True`')
    parser.add_argument('--prod-mode', type=bool, default=False,
                       help='run the script in production mode and use wandb to log outputs')
    parser.add_argument('--wandb-project-name', type=str, default="cleanRL",
                       help="the wandb's project name")
    
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
    parser.add_argument('--update-frequency', type=int, default=3,
                        help="the frequency to update the policy network")
    args = parser.parse_args()
    if not args.seed:
        args.seed = int(time.time())

# TRY NOT TO MODIFY: setup the environment
env = gym.make(args.gym_id)
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.backends.cudnn.deterministic = args.torch_deterministic
env.seed(args.seed)
env.action_space.seed(args.seed)
env.observation_space.seed(args.seed)
input_shape, preprocess_obs_fn = preprocess_obs_space(env.observation_space)
output_shape, preprocess_ac_fn = preprocess_ac_space(env.action_space)

# TODO: initialize agent here:
class Policy(nn.Module):
    def __init__(self):
        super(Policy, self).__init__()
        self.fc1 = nn.Linear(input_shape, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, output_shape)

    def forward(self, x):
        x = preprocess_obs_fn(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class Value(nn.Module):
    def __init__(self):
        super(Value, self).__init__()
        self.fc1 = nn.Linear(input_shape, 64)
        self.fc2 = nn.Linear(64, 1)

    def forward(self, x):
        x = preprocess_obs_fn(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return x

pg = Policy()
vf = Value()
optimizer = optim.Adam(list(pg.parameters()) + list(vf.parameters()), lr=args.learning_rate)
loss_fn = nn.MSELoss()

# TRY NOT TO MODIFY: start the game
experiment_name = f"{time.strftime('%b%d_%H-%M-%S')}__{args.exp_name}__{args.seed}"
writer = SummaryWriter(f"runs/{experiment_name}")
writer.add_text('hyperparameters', "|param|value|\n|-|-|\n%s" % (
        '\n'.join([f"|{key}|{value}|" for key, value in vars(args).items()])))
if args.prod_mode:
    import wandb
    wandb.init(project=args.wandb_project_name, tensorboard=True, config=vars(args), name=experiment_name)
    writer = SummaryWriter(f"/tmp/{experiment_name}")
global_step = 0
while global_step < args.total_timesteps:
    next_obs = np.array(env.reset())
    actions = np.empty((args.episode_length,), dtype=object)
    rewards, dones = np.zeros((2, args.episode_length))
    obs = np.empty((args.episode_length,) + env.observation_space.shape)
    
    # TODO: put other storage logic here
    values = torch.zeros((args.episode_length))
    neglogprobs = torch.zeros((args.episode_length,))
    entropys = torch.zeros((args.episode_length,))
    
    # TRY NOT TO MODIFY: prepare the execution of the game.
    for step in range(args.episode_length):
        global_step += 1
        obs[step] = next_obs.copy()
        
        # TODO: put action logic here
        logits = pg.forward([obs[step]])
        values[step] = vf.forward([obs[step]])
        probs, actions[step], neglogprobs[step], entropys[step] = preprocess_ac_fn(logits)
        actions[step] = actions[step][0]
        
        # TRY NOT TO MODIFY: execute the game and log data.
        next_obs, rewards[step], dones[step], _ = env.step(actions[step])
        next_obs = np.array(next_obs)
        if dones[step]:
            break
    
    # TODO: training.
    # calculate the discounted rewards, or namely, returns
    returns = np.zeros_like(rewards)
    for t in reversed(range(rewards.shape[0]-1)):
        returns[t] = rewards[t] + args.gamma * returns[t+1] * (1-dones[t])
    # advantages are returns - baseline, value estimates in our case
    advantages = returns - values.detach().numpy()

    neglogprobs = neglogprobs.detach()
    non_empty_idx = np.argmax(dones) + 1
    for _ in range(args.update_frequency):
        current_probs, _, new_neglogprobs, _ = preprocess_ac_fn(pg.forward(obs[:non_empty_idx]), action=actions[:non_empty_idx])
        ratio = torch.exp(neglogprobs[:non_empty_idx] - new_neglogprobs)
        surrogate1 = ratio * torch.Tensor(advantages)[:non_empty_idx]
        surrogate2 = torch.clamp(ratio, 1-args.clip_coef, 1+args.clip_coef) * torch.Tensor(advantages)[:non_empty_idx]
        clip = torch.min(surrogate1, surrogate2)
        vf_loss = loss_fn(torch.Tensor(returns), torch.Tensor(values)) * args.vf_coef
        loss = vf_loss - (clip + entropys[:non_empty_idx] * args.ent_coef).mean()
        
        optimizer.zero_grad()
        loss.backward(retain_graph=True)
        nn.utils.clip_grad_norm_(list(pg.parameters()) + list(vf.parameters()), args.max_grad_norm)
        optimizer.step()

    # TRY NOT TO MODIFY: record rewards for plotting purposes
    writer.add_scalar("charts/episode_reward", rewards.sum(), global_step)
    writer.add_scalar("losses/value_loss", vf_loss.item(), global_step)
    writer.add_scalar("losses/entropy", entropys.mean().item(), global_step)
env.close()
