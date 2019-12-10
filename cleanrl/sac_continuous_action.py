import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
from torch.distributions.normal import Normal
from torch.utils.tensorboard import SummaryWriter

from cleanrl.common import preprocess_obs_space, preprocess_ac_space, preprocess_obs_ac_concat
# MODIFIED: Import buffer with random batch sampling support
from cleanrl.buffers import SimpleReplayBuffer
import argparse
import numpy as np
import gym
import pybullet_envs
from gym.spaces import Discrete, Box, MultiBinary, MultiDiscrete, Space
import time
import random
import os

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='SAC agent')
    # Common arguments
    parser.add_argument('--exp-name', type=str, default=os.path.basename(__file__).strip(".py"),
                       help='the name of this experiment')
    parser.add_argument('--gym-id', type=str, default="MountainCarContinuous-v0",
                       help='the id of the gym environment')
    parser.add_argument('--learning-rate', type=float, default=3e-3,
                       help='the learning rate of the optimizer')
    parser.add_argument('--seed', type=int, default=1,
                       help='seed of the experiment')
    parser.add_argument('--episode-length', type=int, default=1000,
                       help='the maximum length of each episode')
    parser.add_argument('--total-timesteps', type=int, default=int( 2e5),
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
    parser.add_argument('--gamma', type=float, default=0.99,
                       help='the discount factor gamma')
    parser.add_argument('--max-grad-norm', type=float, default=0.5,
                       help='the maximum norm for the gradient clipping')
    parser.add_argument('--ent-coef', type=float, default=0.5,
                       help="Coefficient of the entropy added to Q-Values")
    parser.add_argument('--tau', type=float, default=.01,
                       help='Poliak Coefficient for the Value Function Target')
    parser.add_argument('--target_update_interval', type=int, default=16,
                       help="Frequency update of the Target Value Function")
    parser.add_argument('--buffer-max-size', type=int, default=int( 1e5),
                       help="Maximum transistions storable in the buffer")
    parser.add_argument('--batch-size', type=int, default=128,
                       help="Batch size of transition to be used for updates")
    parser.add_argument('--train-iters', type=int, default=1000,
                       help="How many times to we update the NNs after some data sampled")

    # Neural Network Parametrization
    parser.add_argument('--policy-hid-sizes', nargs='+', type=int, default=(128,128,))
    parser.add_argument('--value-hid-sizes', nargs='+', type=int, default=(128,128,))
    parser.add_argument('--q-hid-sizes', nargs='+', type=int, default=(128,128,))

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

# MODIFIED: Buffer config and its seeding
buffer = SimpleReplayBuffer( env.observation_space, env.action_space, args.buffer_max_size, args.batch_size)
# Enables Q-Value special handling of concatenated (obs, ac)
q_input_shape, q_preprocessed_obs_ac_fn = preprocess_obs_ac_concat( env.observation_space, env.action_space, device)

# ALGO LOGIC: initialize agent here:
class Policy( nn.Module):
    def __init__(self):
        super().__init__()
        self._layers = nn.ModuleList()

        # TODO: Unelegant, refactor and use that for all the networks
        current_dim = input_shape
        for hsize in args.policy_hid_sizes:
            self._layers.append( nn.Linear( current_dim, hsize))
            current_dim = hsize

        # TODO: Verify that those two layers' parameters actually get trained
        self._fc_mean = nn.Linear( args.policy_hid_sizes[-1], output_shape)
        self._logstd = nn.Linear( output_shape, output_shape)

    def forward( self, x):
        x = preprocess_obs_fn(x)

        for layer in self._layers:
            x = F.relu( layer(x))

        # TODO: Implement squash too
        action_mean = self._fc_mean( x)
        zeros = torch.zeros( action_mean.size(), device=device)
        action_logstd = self._logstd( zeros)

        return action_mean, action_logstd.exp()

    def get_action( self, obs, eval=True, deterministic=True):
        if eval:
            with torch.no_grad():
                logits , stds = self.forward( [obs])

                if deterministic:
                    stds = torch.zeros_like( stds, device=stds.device)

                return Normal( logits, stds).sample().cpu().numpy()[0]
        else:
            raise NotImplementedError( 'No support for action sampling in training mode (yet)')

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
        x = preprocess_obs_fn(x)
        for layer in self._layers:
            x = F.relu( layer( x))

        return x

class QValue( nn.Module):
    def __init__(self):
        super().__init__()
        self._layers = nn.ModuleList()

        # TODO: Unelegant, refactor and use that for all the networks
        current_dim = input_shape + output_shape
        for hsize in args.q_hid_sizes:
            self._layers.append( nn.Linear( current_dim, hsize))
            current_dim = hsize

        self._layers.append( nn.Linear( args.q_hid_sizes[-1], 1))

    def forward( self, x, a):
        # Shameless copy from Costa's version haha
        x, a = preprocess_obs_fn(x), preprocess_obs_fn(a)
        x = torch.cat( [x,a], 1)

        for layer in self._layers:
            x = F.relu( layer( x))

        return x

pg = Policy().to(device)

vf = Value().to(device)
vf_target = Value().to(device)

qf1 = QValue().to(device)
qf2 = QValue().to(device)

mse_loss_fn = nn.MSELoss()

# MODIFIED: Helper function to update target value function network
def update_target_value( vf, vf_target, tau):
    with torch.no_grad():
        for target_param, param in zip( vf_target.parameters(), vf.parameters()):
            target_param.copy_( (1. - tau) * target_param + tau * param)

# Setting the Target Value's weight to that of the Default Value function
update_target_value( vf, vf_target, 1.0)
# With tau =1.0, copies the weights of the initial value function to the
# target value function

# TODO: Add leraning rate parametrization for each optimizer
p_optimizer = optim.Adam(pg.parameters(), lr=args.learning_rate)
v_optimizer = optim.Adam(vf.parameters(), lr=args.learning_rate)
q_optimizer = optim.Adam(list( qf1.parameters()) + list( qf2.parameters()),
    lr=args.learning_rate)

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

# MODIFIED: In this case, we want to also keep track of the update iterations,
# especially for the loggings
global_iter = 0

# Helper function to evaluate the agent score after each updates
def evaluate( env, policy, eval_episodes=1):
    returns = []

    for eval_ep in range( eval_episodes):
        ret = 0.
        done = False

        obs = np.array( env.reset())

        while not done:
            action = policy.get_action( obs)

            obs, rew, done, _ = env.step( action)
            obs = np.array( obs)
            ret += rew
        # TODO: Break if max episode length is breached

        returns.append( ret)

    return returns

while global_step < args.total_timesteps:
    next_obs = np.array(env.reset())
    actions = np.empty((args.episode_length,), dtype=object)
    rewards, dones = np.zeros((2, args.episode_length))
    obs = np.empty((args.episode_length,) + env.observation_space.shape)

    # ALGO LOGIC: put other storage logic here
    # TODO: Remove the unused tensors
    # neglogprobs = torch.zeros((args.episode_length,), device=device)
    # entropys = torch.zeros((args.episode_length,), device=device)

    # TRY NOT TO MODIFY: prepare the execution of the game.
    for step in range(args.episode_length):
        global_step += 1
        obs[step] = next_obs.copy()

        # ALGO LOGIC: put action logic here
        logits, std = pg.forward([obs[step]])
        # values[step] = vf.forward([obs[step]]) # Compute this during the gradient steps

        # ALGO LOGIC: `env.action_space` specific logic
        if isinstance(env.action_space, Discrete):
            probs = Categorical(logits=logits)
            action = probs.sample()
            actions[step], neglogprobs[step], entropys[step] = action.tolist()[0], -probs.log_prob(action), probs.entropy()

        elif isinstance(env.action_space, Box):
            probs = Normal(logits, std)
            action = probs.sample()
            # DEBUG Notes: Apparently no "pure" action has NaN problem

            # TODO: Figure out why action clipping gave us a NaN on MountainCarContinuous
            # HYP1: The last action in the buffer is None, hence when sampled -> NaN
            # PROPOSED FIX: Add transition one by one instead of the full episode history
            clipped_action = torch.clamp(action, torch.min(torch.Tensor(env.action_space.low)), torch.min(torch.Tensor(env.action_space.high)))
            # actions[step], neglogprobs[step], entropys[step] = clipped_action.cpu().detach().numpy()[0], -probs.log_prob(action).sum(), probs.entropy().sum()
            actions[step] = clipped_action.tolist()[0]

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

        # MODIFIED: Add transition to buffer
        buffer.add_transition( obs[step], actions[step], rewards[step], dones[step], next_obs)

        if dones[step]:
            break

        # ALGO LOGIC: training.
        if buffer.is_ready_for_sample:
            # First, we sample a batch of trajectories
            observation_batch, _, reward_batch, terminals_batch, \
                next_observation_batch = buffer.sample()

            # Resampling only the logprobs
            logits, stds = pg.forward(observation_batch)
            probs = Normal( logits, stds)
            resampled_action = probs.sample()

            with torch.no_grad():
                resampled_action_batch = resampled_action.cpu().detach().numpy() # A more elegant way ? Not doing this -> segmentation error

                # TODO: Refactor this as it is used down below again to update the policy
                q1_values = torch.squeeze( qf1.forward( observation_batch, resampled_action_batch))
                q2_values = torch.squeeze( qf2.forward( observation_batch, resampled_action_batch))
                neglogprob_batch = - probs.log_prob( resampled_action.detach()).sum( 1)

                # Q-Value target for Value function update
                min_q_values = torch.min( q1_values, q2_values)
                min_q_values += args.ent_coef * neglogprob_batch # Add the entropy of the policy

            # Computing Values of the observations in the batch
            values = torch.squeeze( vf.forward( observation_batch))
            # Value Function loss
            # vf_loss = .5 * torch.mean( (values - min_q_values)**2) # TODO: Remove this if Torch;s MSELOss good
            # REMINDER: Detach is crucial, other wise mismatch torch.cuda.FloatTensor and torch.Tensor
            vf_loss = mse_loss_fn( values, min_q_values)

            # Apply gradient step for State Value
            v_optimizer.zero_grad()
            vf_loss.backward()
            v_optimizer.step() # TODO: Consider gradient clipping ?

            # Q Values loss
            # This time, we need it to be with_grad()
            # TODO: Refactor is possible
            # NOTE: The action sampled here are still no_grad.ed, but does it really matter ?
            q1_values = torch.squeeze( qf1.forward( observation_batch, resampled_action_batch))
            q2_values = torch.squeeze( qf2.forward( observation_batch, resampled_action_batch))

            with torch.no_grad():
                q_update_target = torch.tensor( reward_batch).to(device) + \
                    ( 1. - torch.tensor( terminals_batch).to( device)) * \
                    args.gamma * torch.squeeze( vf_target.forward( next_observation_batch))

            q1_loss = mse_loss_fn( q1_values, q_update_target.float())
            q2_loss = mse_loss_fn( q2_values, q_update_target.float())
            q_loss = q1_loss + q2_loss

            # Gradient Step for Q1 and Q2 values
            q_optimizer.zero_grad()
            q_loss.backward()
            q_optimizer.step()

            # Policy loss
            # - neglogprob_batch -> logProbs, right ?

            neglogprob_batch = - probs.log_prob( resampled_action).sum(1)
            entropy_batch = probs.entropy().sum(1)

            with torch.no_grad():
                q1_values = torch.squeeze( qf1.forward( observation_batch, resampled_action.cpu().detach().numpy()))
                q2_values = torch.squeeze( qf2.forward( observation_batch, resampled_action.cpu().detach().numpy()))
                # Segmentation fault when not .cpu().numpy() T_T
                min_q_values = torch.min( q1_values, q2_values)

            # NOTE: This doesn;t look like a KL though
            policy_kl_loss = torch.mean( - neglogprob_batch - min_q_values)

            p_optimizer.zero_grad()
            policy_kl_loss.backward()
            p_optimizer.step()

            # TODO: Consider gradient clipping
            # nn.utils.clip_grad_norm_(list(pg.parameters()) + list(vf.parameters()), args.max_grad_norm)

            # Updating Target Value Function
            if global_step > 0 and global_step % args.target_update_interval == 0:
                update_target_value( vf, vf_target, args.tau)

            # # Evaluating the newly updated policy
            eval_returns = evaluate( env, pg)
            eval_return_mean = np.mean( eval_returns)

            # TODO: Actually include an evaluation phase because we just updated the policy
            # TRY NOT TO MODIFY: record rewards for plotting purposes
            writer.add_scalar("eval/episode_reward", eval_return_mean, global_step)
            writer.add_scalar("losses/value_loss", vf_loss.item(), global_step)
            writer.add_scalar("losses/q1_loss", q1_loss.item(), global_step)
            writer.add_scalar("losses/q2_loss", q2_loss.item(), global_step)
            writer.add_scalar("losses/q_loss", q_loss.item(), global_step)
            writer.add_scalar("losses/entropy", entropy_batch.mean().item(), global_step) # TODO: Is this the correct way to eval entropy ?
            writer.add_scalar("losses/policy_loss", policy_kl_loss.item(), global_step)

            # Some logging
            if global_step > 0 and global_step % 100 == 0:
                # TODO: Print it global_step, with tabulations
                print( "Iter %d: Poloss: %.6f -- Valoss: %.6f -- Q1Loss: %.6f -- Q2Loss: %.6f -- Eval mean: %.3f"
                    % ( global_iter, policy_kl_loss.item(), vf_loss.item(), q1_loss.item(),q2_loss.item(), eval_return_mean))

env.close()
writer.close()
