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
import pybullet_envs
from gym.spaces import Discrete, Box, MultiBinary, MultiDiscrete, Space
import time
import random
import os

# MODIFIED: Import buffer with random batch sampling support
from cleanrl.buffers import SimpleReplayBuffer

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='SAC with 2 Q functions, Online updates')
    # Common arguments
    parser.add_argument('--exp-name', type=str, default=os.path.basename(__file__).strip(".py"),
                       help='the name of this experiment')
    parser.add_argument('--gym-id', type=str, default="HopperBulletEnv-v0",
                       help='the id of the gym environment')
    parser.add_argument('--learning-rate', type=float, default=7e-4,
                       help='the learning rate of the optimizer')
    parser.add_argument('--seed', type=int, default=1,
                       help='seed of the experiment')
    parser.add_argument('--episode-length', type=int, default=1000,
                       help='the maximum length of each episode')
    parser.add_argument('--total-timesteps', type=int, default=int( 1e6),
                       help='total timesteps of the experiments')
    parser.add_argument('--torch-deterministic', type=bool, default=True,
                       help='whether to set `torch.backends.cudnn.deterministic=True`')
    parser.add_argument('--cuda', type=bool, default=True,
                       help='whether to use CUDA whenever possible')
    parser.add_argument('--prod-mode', type=bool, default=False,
                       help='run the script in production mode and use wandb to log outputs')
    parser.add_argument('--wandb-project-name', type=str, default="cleanRL",
                       help="the wandb's project name")

    # Algorithm specific arguments
    parser.add_argument('--buffer-size', type=int, default=int( 5e4),
                        help='the replay memory buffer size')
    parser.add_argument('--gamma', type=float, default=0.99,
                       help='the discount factor gamma')
    parser.add_argument('--target-update-interval', type=int, default=1,
                       help="the timesteps it takes to update the target network")
    parser.add_argument('--max-grad-norm', type=float, default=0.5,
                       help='the maximum norm for the gradient clipping')
    parser.add_argument('--batch-size', type=int, default=32,
                       help="the batch size of sample from the reply memory")
    parser.add_argument('--tau', type=float, default=0.005,
                       help="target smoothing coefficient (default: 0.005)")
    parser.add_argument('--alpha', type=float, default=0.2,
                       help="Entropy regularization coefficient.")

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
assert isinstance(env.action_space, Box), "only continuous action space is supported"

# ALGO LOGIC: initialize agent here:
EPS = 1e-8
LOG_STD_MAX = 2
LOG_STD_MIN = -20

# Custom Gaussian Policy
class Policy(nn.Module):
    def __init__(self, squashing = True):
        # Custom
        super().__init__()
        self._layers = nn.ModuleList()
        self._squashing = squashing

        current_dim = input_shape
        for hsize in args.policy_hid_sizes:
            self._layers.append( nn.Linear( current_dim, hsize))
            current_dim = hsize

        self._fc_mean = nn.Linear( args.policy_hid_sizes[-1], output_shape)
        self._fc_logstd = nn.Linear( args.policy_hid_sizes[-1], output_shape)

    def forward(self, x):
        # # Workaround Seg. Fault when passing tensor to Q Function
        if not isinstance( x, torch.Tensor):
            x = preprocess_obs_fn(x)

        for layer in self._layers:
            x = F.relu( layer(x))

        action_means = self._fc_mean( x)
        action_logstds = self._fc_logstd( x)
        action_logstds.clamp_( LOG_STD_MIN, LOG_STD_MAX)

        return action_means, action_logstds

    def get_actions(self, observations, with_logp_pis=True, deterministic=False):
        # Custom
        action_means, action_logstds = self.forward( observations)
        action_stds = action_logstds.exp()

        if deterministic:
            action_stds = torch.zeros_like( action_stds).to( device)

        actions_dist = Normal( action_means, action_stds)
        actions = actions_dist.rsample()
        logp_pis = actions_dist.log_prob( actions)

        if self.squashing:
            actions = torch.tanh( actions)

            logp_pis -= torch.log( 1. - actions.pow(2) + EPS)

        if with_logp_pis:
            return actions, logp_pis.sum(1, keepdim=True)
        else:
            return actions

    def sample(self, observations):
        actions, logp_pis = self.get_actions( observations, True)
        return actions, logp_pis , {}

    def get_entropy( self, observations):
        action_means, action_logstds = self.forward( observations)
        dist = Normal( action_means, action_logstds.exp())
        entropy = dist.entropy().sum(1)

        return entropy

    @property
    def squashing(self):
        return self._squashing

class QValue( nn.Module):
    # Custom
    def __init__(self):
        super().__init__()
        self._layers = nn.ModuleList()

        current_dim = input_shape + output_shape
        for hsize in args.q_hid_sizes:
            self._layers.append( nn.Linear( current_dim, hsize))
            current_dim = hsize

        self._layers.append( nn.Linear( args.q_hid_sizes[-1], 1))

    def forward( self, x, a):
        # Workaround Seg. Fault when passing tensor to Q Function
        if not isinstance( x, torch.Tensor):
            x = preprocess_obs_fn(x)

        if not isinstance( a, torch.Tensor):
            a = preprocess_obs_fn(a)

        x = torch.cat( [x,a], 1)

        for layer in self._layers[:-1]:
            x = F.relu( layer( x))

        return self._layers[-1](x)

class Value( nn.Module):
    def __init__( self):
        super().__init__()
        self._layers = nn.ModuleList()

        # TODO: Unelegant, refactor and use that for all the networks
        current_dim = input_shape
        for hsize in args.value_hid_sizes:
            self._layers.append( nn.Linear( current_dim, hsize))
            current_dim = hsize

        self._layers.append( nn.Linear( args.value_hid_sizes[-1], 1))

    def forward(self, x):
        if not isinstance( x, torch.Tensor):
            x = preprocess_obs_fn(x)

        for layer in self._layers[:-1]:
            x = F.relu( layer( x))

        return self._layers[-1](x)

buffer = SimpleReplayBuffer( env.observation_space, env.action_space, args.buffer_size, args.batch_size)
buffer.set_seed( args.seed) # Seedable buffer for reproducibility

# Defining the agent's policy: Gaussian
pg = Policy().to(device)

# Defining the agent's policy: Gaussian
qf1 = QValue().to(device)
qf2 = QValue().to(device)

vf = Value().to( device)
vf_target = Value().to( device)

# MODIFIED: Helper function to update target value function network
def update_target_value( vf, vf_target, tau):
    for target_param, param in zip( vf_target.parameters(), vf.parameters()):
        target_param.data.copy_( (1. - tau) * target_param.data + tau * param.data)

# Sync weights of the QValues
# Setting tau to 1.0 is equivalent to Hard Update
update_target_value( vf, vf_target, 1.0)

q_optimizer = optim.Adam( list(qf1.parameters()) + list(qf2.parameters()),
    lr=args.learning_rate)
p_optimizer = optim.Adam( list(pg.parameters()), lr=args.learning_rate)
v_optimizer = optim.Adam( list(vf.parameters()), lr=args.learning_rate)

mse_loss_fn = nn.MSELoss()

# Helper function to evaluate agent determinisitically
def test_agent( env, policy, eval_episodes=1):
    returns = []
    lengths = []

    for eval_ep in range( eval_episodes):
        ret = 0.
        done = False
        t = 0

        obs = np.array( env.reset())

        while not done:
            with torch.no_grad():
                action = pg.get_actions([obs], False, True).tolist()[0]

            obs, rew, done, _ = env.step( action)
            obs = np.array( obs)
            ret += rew
            t += 1
        # TODO: Break if max episode length is breached

        returns.append( ret)
        lengths.append( t)

    return returns, lengths

# TRY NOT TO MODIFY: start the game
experiment_name = f"{args.gym_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
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
            action = pg.get_actions([obs], False, False).tolist()[0]

        # TRY NOT TO MODIFY: execute the game and log data.
        next_obs, rew, done, _ = env.step(action)
        next_obs = np.array(next_obs)

        buffer.add_transition(obs, action, rew, done, next_obs)
        # TODO: Add custom buffer

        # Keeping track of train episode returns
        train_episode_return += rew
        train_episode_length += 1

        # ALGO LOGIC: training.
        if buffer.is_ready_for_sample:
            observation_batch, action_batch, reward_batch, \
                terminals_batch, next_observation_batch = buffer.sample(args.batch_size)

            # Value function loss and updates
            with torch.no_grad():
                resampled_actions, resampled_logp_pis = pg.get_actions(observation_batch)

                qf1_values = qf1.forward( observation_batch, resampled_actions)
                qf2_values = qf2.forward( observation_batch, resampled_actions)

                min_qf_values = torch.min( qf1_values, qf2_values)

                v_backup = (min_qf_values - args.alpha * resampled_logp_pis).view(-1)

            v_values = vf.forward( observation_batch).view(-1)

            vf_loss = mse_loss_fn( v_values, v_backup)

            # V gradient steps
            v_optimizer.zero_grad()
            vf_loss.backward()
            v_optimizer.step()

            # Q function losses and updates
            with torch.no_grad():

                vf_target_next_state_values = vf_target.forward( next_observation_batch).view(-1)

                q_backup = torch.Tensor(reward_batch).to(device) + \
                    (1 - torch.Tensor(terminals_batch).to(device)) * args.gamma * \
                        vf_target_next_state_values

            q1_values = qf1.forward(observation_batch, action_batch).view(-1)
            q2_values = qf2.forward(observation_batch, action_batch).view(-1)

            qf1_loss = mse_loss_fn(q1_values, q_backup)
            qf2_loss = mse_loss_fn(q2_values, q_backup)
            qf_loss = qf1_loss + qf2_loss

            # Q functions gradient steps
            q_optimizer.zero_grad()
            qf_loss.backward()
            q_optimizer.step()

            # Policy loss and updates
            resampled_actions, logp_pis = pg.get_actions(observation_batch)

            qf1_pi = qf1.forward(observation_batch, resampled_actions)
            qf2_pi = qf2.forward(observation_batch, resampled_actions)

            min_qf_pi = torch.min(qf1_pi, qf2_pi).view(-1)

            policy_loss = ((args.alpha * logp_pis) - min_qf_pi).mean()

            # Policy gradient step
            p_optimizer.zero_grad()
            policy_loss.backward()
            p_optimizer.step()

            # Measures entropy after the update
            with torch.no_grad():
                entropy_batch = pg.get_entropy( observation_batch)

            if global_step > 0 and global_step % args.target_update_interval == 0:
                update_target_value( vf, vf_target, args.tau)

            # Some verbosity and logging
            # Evaulating in deterministic mode after one episode
            if global_step % args.episode_length == 0:
                eval_returns, eval_ep_lengths = test_agent( env, pg, 5)
                eval_return_mean = np.mean( eval_returns)
                eval_ep_length_mean = np.mean( eval_ep_lengths)

                # Log to TBoard
                writer.add_scalar("eval/episode_return", eval_return_mean, global_step)
                writer.add_scalar("eval/episode_length", eval_ep_length_mean, global_step)

            writer.add_scalar("train/q1_loss", qf1_loss.item(), global_step)
            writer.add_scalar("train/q2_loss", qf2_loss.item(), global_step)
            writer.add_scalar("train/v_loss", vf_loss.item(), global_step)
            writer.add_scalar("train/policy_loss", policy_loss.item(), global_step)
            writer.add_scalar("train/entropy", entropy_batch.mean().item(), global_step)

            if global_step > 0 and global_step % 100 == 0:
                print( "Step %d: Poloss: %.6f -- Q1Loss: %.6f -- Q2Loss: %.6f -- VLoss: %.6f"
                    % ( global_step, policy_loss.item(), qf1_loss.item(), qf2_loss.item(), vf_loss.item()))

        if done:
            # MODIFIED: Logging the trainin episode return and length, then resetting their holders
            writer.add_scalar("eval/train_episode_return", train_episode_return, global_step)
            writer.add_scalar("eval/train_episode_length", train_episode_length, global_step)

            train_episode_return = 0.
            train_episode_length = 0

            break;

writer.close()
