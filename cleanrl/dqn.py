# Reference: https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tensorboardX import SummaryWriter
from stable_baselines.deepq.replay_buffer import ReplayBuffer

from cleanrl.common import preprocess_obs_space, preprocess_ac_space
import argparse
import numpy as np
import gym
import time
import random

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='A2C agent')
    # Common arguments
    parser.add_argument('--exp-name', type=str, default="dqn",
                       help='the name of this experiment')
    parser.add_argument('--gym-id', type=str, default="CartPole-v0",
                       help='the id of the gym environment')
    parser.add_argument('--learning-rate', type=float, default=7e-4,
                       help='the learning rate of the optimizer')
    parser.add_argument('--seed', type=int, default=5,
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
    parser.add_argument('--start-e', type=float, default=1,
                       help="the starting epsilon for exploration")
    parser.add_argument('--end-e', type=float, default=0.05,
                       help="the ending epsilon for exploration")
    parser.add_argument('--exploration-duration', type=int, default=25000,
                       help="the time it takes from start-e to go end-e")
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
output_shape, preprocess_ac_fn = preprocess_ac_space(env.action_space, stochastic=False)

# TODO: initialize agent here:
er = ReplayBuffer(args.buffer_size)
class QNetwork(nn.Module):
    def __init__(self):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(input_shape, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, output_shape)

    def forward(self, x):
        x = preprocess_obs_fn(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
def linear_schedule(start_e: float, end_e: float, duration: int, t: int):
    slope =  (end_e - start_e) / duration
    return max(slope * t + start_e, end_e)

q_network = QNetwork()
target_network = QNetwork()
target_network.load_state_dict(q_network.state_dict())
optimizer = optim.Adam(q_network.parameters(), lr=args.learning_rate)
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
        epsilon = linear_schedule(args.start_e, args.end_e, args.exploration_duration, global_step)
        if random.random() < epsilon:
            actions[step] = random.randint(0, env.action_space.n - 1)
        else:
            logits = target_network.forward([obs[step]])
            probs, actions[step], neglogprobs[step], entropys[step] = preprocess_ac_fn(logits)
            actions[step] = actions[step][0]
        
        
        # TRY NOT TO MODIFY: execute the game and log data.
        next_obs, rewards[step], dones[step], _ = env.step(actions[step])
        next_obs = np.array(next_obs)
        done_int = 1 if dones[step] else 0
        er.add(obs[step], actions[step], rewards[step], next_obs, done_int)
        
        # TODO: training.
        if global_step < 1000:
            if done_int:
                break
            continue
        s_obs, s_actions, s_rewards, s_next_obses, s_dones = er.sample(args.batch_size)
        target_max = torch.max(target_network.forward(s_next_obses), dim=1)[0]
        td_target = torch.Tensor(s_rewards) + args.gamma * target_max * (1 - torch.Tensor(s_dones))
        old_val = q_network.forward(s_obs).gather(1, torch.LongTensor(s_actions).view(-1,1)).squeeze()
        loss = loss_fn(td_target, old_val)

        # optimize the midel
        optimizer.zero_grad()
        loss.backward()
        writer.add_scalar("losses/td_loss", loss, global_step)
        nn.utils.clip_grad_norm_(list(q_network.parameters()), args.max_grad_norm)
        optimizer.step()
        
        # update the target network
        if global_step % args.target_network_frequency == 0:
            target_network.load_state_dict(q_network.state_dict())
        
        if done_int:
            break
    
    # TRY NOT TO MODIFY: record rewards for plotting purposes
    writer.add_scalar("charts/episode_reward", rewards.sum(), global_step)
env.close()
