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
import pybullet_envs
from gym.spaces import Discrete, Box, MultiBinary, MultiDiscrete, Space
import time
import random
import os
import collections

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='SAC with 2 Q functions, Online updates')
    # Common arguments
    parser.add_argument('--exp-name', type=str, default=os.path.basename(__file__).strip(".py"),
                       help='the name of this experiment')
    parser.add_argument('--gym-id', type=str, default="CartPole-v0",
                       help='the id of the gym environment')
    parser.add_argument('--learning-rate', type=float, default=7e-4,
                       help='the learning rate of the optimizer')
    parser.add_argument('--seed', type=int, default=1,
                       help='seed of the experiment')
    parser.add_argument('--episode-length', type=int, default=1000,
                       help='the maximum length of each episode')
    parser.add_argument('--total-timesteps', type=int, default=int(1e6),
                       help='total timesteps of the experiments')
    parser.add_argument('--torch-deterministic', type=bool, default=True,
                       help='whether to set `torch.backends.cudnn.deterministic=True`')
    parser.add_argument('--cuda', type=bool, default=True,
                       help='whether to use CUDA whenever possible')
    parser.add_argument('--prod-mode', type=bool, default=False,
                       help='run the script in production mode and use wandb to log outputs')
    parser.add_argument('--wandb-project-name', type=str, default="cleanRL",
                       help="the wandb's project name")
    parser.add_argument('--wandb-entity', type=str, default=None,
                       help="the entity (team) of wandb's project")
    parser.add_argument('--notb', action='store_true',
       help='No Tensorboard logging')

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

    # Neural Network Parametrization
    parser.add_argument('--policy-hid-sizes', nargs='+', type=int, default=(120,84,))
    parser.add_argument('--value-hid-sizes', nargs='+', type=int, default=(120,84,))
    parser.add_argument('--q-hid-sizes', nargs='+', type=int, default=(120,84,))

    args = parser.parse_args()
    if not args.seed:
        args.seed = int(time.time())

# TRY NOT TO MODIFY: setup the environment
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
assert isinstance(env.action_space, Discrete), "only discrete action space is supported"

# ALGO LOGIC: initialize agent here:
EPS = 1e-8
LOG_STD_MAX = 2
LOG_STD_MIN = -20

# Gaussian Policy
class Policy(nn.Module):
    def __init__(self):
        super().__init__()
        self._layers = nn.ModuleList()

        current_dim = input_shape
        for hsize in list(args.policy_hid_sizes) + list([output_shape,]):
            self._layers.append(nn.Linear(current_dim, hsize))
            current_dim = hsize

        self.nn_softmax = torch.nn.Softmax(1)
        self.nn_log_softmax = torch.nn.LogSoftmax(1)

    def forward(self, x):
        # # Workaround Seg. Fault when passing tensor to Q Function
        if not isinstance(x, torch.Tensor):
            x = preprocess_obs_fn(x)

        for layer in self._layers[:-1]:
            x = F.relu(layer(x))

        # Return the logits [batch_size, output_shape]
        logits = self._layers[-1](x)

        return self.nn_softmax(logits), self.nn_log_softmax(logits)

    # TODO: Since we use torch.nn.{Softmax, LogSoftmax}, get_action_probs loses
    # most of it s meaning.
    def get_action_probs(self, observations):
        # action_logits = self.forward(observations)
        action_probs, action_logps = self.forward(observations)

        return action_probs, action_logps

    # TODO: Add support for deterministic action
    def get_actions(self, observations, deterministic = False):
        action_probs, _ = self.forward(observations)

        if deterministic:
            return torch.argmax(action_probs, 1, keepdim=True)

        dist = Categorical(probs=action_probs)

        return dist.sample()

    def get_entropy(self, observations):
        # TODO: Might as well manually compute entropy then
        action_probs, _ = self.forward(observations)
        dist = Categorical(probs=action_probs)

        # entropy = dist.entropy().sum(1)

        return dist.entropy()

class QValue(nn.Module):
    # Custom
    def __init__(self):
        super().__init__()
        self._layers = nn.ModuleList()

        current_dim = input_shape
        for hsize in list(args.q_hid_sizes) + list([output_shape,]):
            self._layers.append(nn.Linear(current_dim, hsize))
            current_dim = hsize

    def forward(self, x):
        # Workaround Seg. Fault when passing tensor to Q Function
        if not isinstance(x, torch.Tensor):
            x = preprocess_obs_fn(x)

        for layer in self._layers[:-1]:
            x = F.relu(layer(x))

        return self._layers[-1](x)

    def get_state_action_values(self, x, a):
        if not isinstance(x, torch.Tensor):
            x = preprocess_obs_fn(x)

        if not isinstance(a, torch.Tensor):
            a = preprocess_obs_fn(a)

        values = self.forward(x)

        action_values = torch.gather(values, 1, a.long())

        return action_values

# From https://github.com/seungeunrho/minimalRL/blob/master/dqn.py
class ReplayBuffer():
    def __init__(self):
        self.buffer = collections.deque(maxlen=args.buffer_size)

    def put(self, transition):
        self.buffer.append(transition)

    def sample(self, n):
        mini_batch = random.sample(self.buffer, n)
        s_lst, a_lst, r_lst, s_prime_lst, done_mask_lst = [], [], [], [], []

        for transition in mini_batch:
            s, a, r, s_prime, done_mask = transition
            s_lst.append(s)
            a_lst.append([a])
            r_lst.append([r])
            s_prime_lst.append(s_prime)
            done_mask_lst.append([done_mask])

        return s_lst, a_lst, r_lst, s_prime_lst, done_mask_lst

    def size(self):
        return len(self.buffer)

buffer = ReplayBuffer()

# Defining the agent's policy: Gaussian
pg = Policy().to(device)

# Defining the agent's policy: Gaussian
qf1 = QValue().to(device)
qf2 = QValue().to(device)

qf1_target = QValue().to(device)
qf2_target = QValue().to(device)

# MODIFIED: Helper function to update target value function network
def update_target_value(vf, vf_target, tau):
    for target_param, param in zip(vf_target.parameters(), vf.parameters()):
        target_param.data.copy_((1. - tau) * target_param.data + tau * param.data)

# Sync weights of the QValues
# Setting tau to 1.0 is equivalent to Hard Update
update_target_value(qf1, qf1_target, 1.0)
update_target_value(qf2, qf2_target, 1.0)

q_optimizer = optim.Adam(list(qf1.parameters()) + list(qf2.parameters()),
    lr=args.learning_rate)
p_optimizer = optim.Adam(list(pg.parameters()), lr=args.learning_rate)

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

mse_loss_fn = nn.MSELoss()

# Helper function to evaluate agent determinisitically
def test_agent(env, policy, eval_episodes=1):
    returns = []
    lengths = []

    for eval_ep in range(eval_episodes):
        ret = 0.
        done = False
        t = 0

        obs = np.array(env.reset())

        while not done:
            with torch.no_grad():
                action = pg.get_actions([obs], True).tolist()[0][0]

            obs, rew, done, _ = env.step(action)
            obs = np.array(obs)
            ret += rew
            t += 1
        # TODO: Break if max episode length is breached

        returns.append(ret)
        lengths.append(t)

    return returns, lengths

# TRY NOT TO MODIFY: start the game
experiment_name = f"{args.gym_id}__{args.exp_name}__{args.seed}__{int(time.time())}"

# MODIFIED: When testing, skip Tensorboard log creation
if not args.notb:
    writer = SummaryWriter(f"runs/{experiment_name}")
    writer.add_text('hyperparameters', "|param|value|\n|-|-|\n%s" % (
            '\n'.join([f"|{key}|{value}|" for key, value in vars(args).items()])))

if args.prod_mode:
    import wandb
    wandb.init(project=args.wandb_project_name, tensorboard=True, config=vars(args), name=experiment_name)
    writer = SummaryWriter(f"/tmp/{experiment_name}")
    wandb.save(os.path.abspath(__file__))

global_step = 0
while global_step < args.total_timesteps:
    next_obs = np.array(env.reset())

    # MODIFIED: Keeping track of train episode returns and lengths
    train_episode_return = 0.
    train_episode_length = 0

    done = False

    # TRY NOT TO MODIFY: prepare the execution of the game.
    for step in range(args.episode_length):
        global_step += 1
        obs = next_obs.copy()

        # ALGO LOGIC: put action logic here
        with torch.no_grad():
            action = pg.get_actions([obs]).tolist()[0]

        # TRY NOT TO MODIFY: execute the game and log data.
        next_obs, rew, done, _ = env.step(action)
        next_obs = np.array(next_obs)

        buffer.put((obs, action, rew, next_obs, done))

        # Keeping track of train episode returns
        train_episode_return += rew
        train_episode_length += 1

        # ALGO LOGIC: training.
        if buffer.size() > args.batch_size:
            # TODO: Cast action batch to longs ?
            observation_batch, action_batch, reward_batch, \
                next_observation_batch, terminals_batch = buffer.sample(args.batch_size)

            # TODO reimplementing algorithm logic
            # Q function losses
            with torch.no_grad():
                next_action_probs, next_logprobs = pg.get_action_probs(next_observation_batch)

                qf1_target_values = qf1_target.forward(next_observation_batch)
                qf2_target_values = qf2_target.forward(next_observation_batch)

                min_qf_target_values = torch.min(qf1_target_values, qf2_target_values)
                ent_aug_min_qf_values = min_qf_target_values - alpha * next_logprobs

                v_next_target = (next_action_probs * ent_aug_min_qf_values).sum(1)
                q_backup = torch.Tensor(reward_batch).to(device) + \
                    args.gamma * (1. - torch.Tensor(terminals_batch).to(device)) * \
                    v_next_target

            qf1_a_values = qf1.get_state_action_values(observation_batch, action_batch).view(-1)
            qf2_a_values = qf2.get_state_action_values(observation_batch, action_batch).view(-1)

            qf1_loss = mse_loss_fn(qf1_a_values, q_backup)
            qf2_loss = mse_loss_fn(qf2_a_values, q_backup)
            qf_loss = qf1_loss + qf2_loss

            # Q param gradient step
            q_optimizer.zero_grad()
            qf_loss.backward()
            q_optimizer.step()

            # Policy loss
            action_probs, logps = pg.get_action_probs(observation_batch)
            qf1_values = qf1.forward(observation_batch)
            qf2_values = qf2.forward(observation_batch)
            min_qf_values = torch.min(qf1_values, qf2_values)

            policy_loss = alpha * logps - min_qf_values
            policy_loss *= action_probs
            policy_loss = policy_loss.sum(1).mean()

            # Policy gradient step
            p_optimizer.zero_grad()
            policy_loss.backward()
            p_optimizer.step()

            with torch.no_grad():
                entropy_batch = pg.get_entropy(observation_batch)

            # TODO: Alpha auto tune
            if args.autotune:
                with torch.no_grad():
                    action_probs, logps = pg.get_action_probs(observation_batch)

                alpha_loss = action_probs * (- log_alpha.exp() * (logps + target_entropy))
                alpha_loss = alpha_loss.sum(1).mean()

                # Alpha gradient step
                a_optimizer.zero_grad()
                alpha_loss.backward()
                a_optimizer.step()

                # Update the alpha value used in other parts of the algorithm
                # TODO: More elegant way ?
                alpha = log_alpha.exp().cpu().detach().numpy()[0]

            if global_step > 0 and global_step % args.target_update_interval == 0:
                update_target_value(qf1, qf1_target, args.tau)
                update_target_value(qf2, qf2_target, args.tau)

            # Some verbosity and logging
            # Evaulating in deterministic mode after one episode
            if global_step % args.episode_length == 0:
                eval_returns, eval_ep_lengths = test_agent(env, pg, 5)
                eval_return_mean = np.mean(eval_returns)
                eval_ep_length_mean = np.mean(eval_ep_lengths)

                # Log to TBoard
                if not args.notb:
                    writer.add_scalar("eval/episode_return", eval_return_mean, global_step)
                    writer.add_scalar("eval/episode_length", eval_ep_length_mean, global_step)

            if not args.notb:
                writer.add_scalar("train/q1_loss", qf1_loss.item(), global_step)
                writer.add_scalar("train/q2_loss", qf2_loss.item(), global_step)
                writer.add_scalar("train/policy_loss", policy_loss.item(), global_step)
                writer.add_scalar("train/entropy", entropy_batch.mean().item(), global_step)
                writer.add_scalar("train/alpha_entropy_coef", log_alpha.exp(), global_step)

            if global_step > 0 and global_step % 100 == 0:
                print("Step %d: Poloss: %.6f -- Q1Loss: %.6f -- Q2Loss: %.6f"
                    % (global_step, policy_loss.item(), qf1_loss.item(), qf2_loss.item()))

        if done:
            # MODIFIED: Logging the trainin episode return and length, then resetting their holders
            if not args.notb:
                writer.add_scalar("eval/train_episode_return", train_episode_return, global_step)
                writer.add_scalar("eval/train_episode_length", train_episode_length, global_step)

            train_episode_return = 0.
            train_episode_length = 0

            break;

writer.close()
