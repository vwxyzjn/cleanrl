import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
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
    parser.add_argument('--gym-id', type=str, default="CartPole-v0",
                       help='the id of the gym environment')
    parser.add_argument('--learning-rate', type=float, default=7e-4,
                       help='the learning rate of the optimizer')
    parser.add_argument('--seed', type=int, default=1,
                       help='seed of the experiment')
    parser.add_argument('--episode-length', type=int, default=1000,
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
    parser.add_argument('--gae-lambda', type=float, default=0.97,
                       help='the lambda for the general advantage estimation')
    parser.add_argument('--vf-coef', type=float, default=0.25,
                       help="value function's coefficient the loss function")
    parser.add_argument('--max-grad-norm', type=float, default=0.5,
                       help='the maximum norm for the gradient clipping')
    parser.add_argument('--ent-coef', type=float, default=0.01,
                       help="policy entropy's coefficient the loss function")
    parser.add_argument('--clip-coef', type=float, default=0.2,
                       help="the surrogate clipping coefficient")
    parser.add_argument('--update-epochs', type=int, default=100,
                        help="the K epochs to update the policy")
    # MODFIED: Added support for KL Bounding during the updates
    parser.add_argument('--kl', action='store_true',
                        help='If toggled, the policy updates will be early stopped w.r.t target-kl')
    parser.add_argument('--target-kl', type=float, default=0.015)

    # MODIFIED: Separate learning rate for policy and values, according to OpenAI SpinUp
    parser.add_argument('--policy-lr', type=float, default=3e-4)
    parser.add_argument('--value-lr', type=float, default=1e-3)

    # MODIFIED: Parameterization for the tricks in the Implementation Matters paper.
    parser.add_argument('--norm-obs', action='store_true',
                        help="Toggles observation normalization")
    parser.add_argument('--norm-rewards', action='store_true',
                        help="Toggles rewards normalization")
    parser.add_argument('--norm-returns', action='store_true',
                        help="Toggles returns normalization")
    parser.add_argument('--obs-clip', type=float, default=10.0,
                        help="Value for reward clipping, as per the paper")
    parser.add_argument('--rew-clip', type=float, default=5.0,
                        help="Value for observation clipping, as per the paper")
    parser.add_argument('--anneal-lr', action='store_true',
                        help="Toggle learning rate annealing for policy and value networks")
    parser.add_argument('--weights-init', default="xavier", choices=["xavier", 'orthogonal'],
                        help='Selects the scheme to be used for weights initialization')

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
device = torch.device('cpu')
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.backends.cudnn.deterministic = args.torch_deterministic

# MODIFIED: Env Wrapper for Observation and Rewards normalizations from the original repository
from cleanrl.experiments.impl_matters.torch_utils import RunningStat, ZFilter, Identity, StateWithTime, RewardFilter
class CustomEnv(object):
    '''
    A wrapper around the OpenAI gym environment that adds support for the following:
    - Rewards normalization
    - State normalization
    - Adding timestep as a feature with a particular horizon T
    Also provides utility functions/properties for:
    - Whether the env is discrete or continuous
    - Size of feature space
    - Size of action space
    Provides the same API (init, step, reset) as the OpenAI gym
    '''
    def __init__(self, game, add_t_with_horizon=None):
        self.env = gym.make(game)
        self.env.seed(args.seed)
        self.env.action_space.seed(args.seed)
        self.env.observation_space.seed(args.seed)

        # respect the default timelimit
        if int(args.episode_length):
            if not isinstance(self.env, TimeLimit):
                self.env = TimeLimit(self.env, int(args.episode_length))
            else:
                self.env._max_episode_steps = int(args.episode_length)
        else:
            args.episode_length = self.env._max_episode_steps if isinstance(self.env, TimeLimit) else 200

        if args.capture_video:
            self.env = Monitor(self.env, f'videos/{experiment_name}')

        # Environment type
        self.is_discrete = type(self.env.action_space) == Discrete
        assert self.is_discrete or type(self.env.action_space) == Box

        # Number of actions
        action_shape = self.env.action_space.shape
        assert len(action_shape) <= 1 # scalar or vector actions
        self.num_actions = self.env.action_space.n if self.is_discrete else 0 \
                            if len(action_shape) == 0 else action_shape[0]

        # Number of features
        assert len(self.env.observation_space.shape) == 1
        self.num_features = self.env.reset().shape[0]

        # Support for state normalization or using time as a feature
        self.state_filter = Identity()
        if args.norm_obs:
            self.state_filter = ZFilter(self.state_filter, shape=[self.num_features], \
                                            clip=args.obs_clip)
        if add_t_with_horizon is not None:
            self.state_filter = StateWithTime(self.state_filter, horizon=add_t_with_horizon)

        # Support for rewards normalization
        self.reward_filter = Identity()
        if args.norm_rewards:
            self.reward_filter = ZFilter(self.reward_filter, shape=(), center=False, clip=args.rew_clip)
        if args.norm_returns:
            self.reward_filter = RewardFilter(self.reward_filter, shape=(), gamma=args.gamma, clip=args.rew_clip)

        # Running total reward (set to 0.0 at resets)
        self.total_true_reward = 0.0

    def reset(self):
        # Reset the state, and the running total reward
        start_state = self.env.reset()
        self.total_true_reward = 0.0
        self.counter = 0.0
        self.state_filter.reset()
        # MODIFIED: Also reset the reward filter
        self.reward_filter.reset()
        return self.state_filter(start_state, reset=True)

    def step(self, action):
        state, reward, is_done, info = self.env.step(action)
        state = self.state_filter(state)
        self.total_true_reward += reward
        self.counter += 1
        _reward = self.reward_filter(reward)
        if is_done:
            info['done'] = (self.counter, self.total_true_reward)
        return state, _reward, is_done, info

env = CustomEnv(args.gym_id)
# MODIFIED: Moved input_shape and output_shape after the env is created
input_shape, preprocess_obs_fn = preprocess_obs_space(env.env.observation_space, device)
output_shape = preprocess_ac_space(env.env.action_space)

# ALGO LOGIC: initialize agent here:
class Policy(nn.Module):
    def __init__(self):
        super(Policy, self).__init__()
        self.fc1 = nn.Linear(input_shape, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, output_shape)

        if args.weights_init == "orthogonal":
            torch.nn.init.orthogonal_( self.fc1.weight)
            torch.nn.init.orthogonal_( self.fc2.weight)
            torch.nn.init.orthogonal_( self.fc3.weight)
        elif args.weights_init == "xavier":
            torch.nn.init.xavier_uniform_( self.fc1.weight)
            torch.nn.init.xavier_uniform_( self.fc2.weight)
            torch.nn.init.xavier_uniform_( self.fc3.weight)
        else:
            raise NotImplementedError

    def forward(self, x):
        x = preprocess_obs_fn(x)
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        x = self.fc3(x)
        return x

    def get_action(self, x):
        logits = self.forward( x)
        assert isinstance( env.env.action_space, Discrete), "Discrete Action Space only"

        probs = Categorical(logits=logits)
        action = probs.sample()
        return action, probs.log_prob(action)

    def get_logproba(self, x, actions):
        logits = self.forward(x)
        probs = Categorical(logits=logits)
        return probs.log_prob( actions)

class Value(nn.Module):
    def __init__(self):
        super(Value, self).__init__()
        self.fc1 = nn.Linear(input_shape, 64)
        self.fc2 = nn.Linear(64, 1)

        if args.weights_init == "orthogonal":
            torch.nn.init.orthogonal_( self.fc1.weight)
            torch.nn.init.orthogonal_( self.fc2.weight)
        elif args.weights_init == "xavier":
            torch.nn.init.xavier_uniform_( self.fc1.weight)
            torch.nn.init.xavier_uniform_( self.fc2.weight)
        else:
            raise NotImplementedError

    def forward(self, x):
        x = preprocess_obs_fn(x)
        x = torch.tanh(self.fc1(x))
        x = self.fc2(x)
        return x

pg = Policy().to(device)
vf = Value().to(device)

# MODIFIED: Separate optimizer and learning rates
pg_optimizer = optim.Adam( list(pg.parameters()), lr=args.policy_lr)
v_optimizer = optim.Adam( list(vf.parameters()), lr=args.value_lr)

# MODIFIED: Initializing learning rate anneal scheduler when need
if args.anneal_lr:
    anneal_fn = lambda f: 1-f / args.total_timesteps
    pg_lr_scheduler = optim.lr_scheduler.LambdaLR( pg_optimizer, lr_lambda=anneal_fn)
    vf_lr_scheduler = optim.lr_scheduler.LambdaLR( v_optimizer, lr_lambda=anneal_fn)

loss_fn = nn.MSELoss()

# MODIFIED: Buffer for Epoch data
import collections
class ReplayBuffer():
    def __init__(self, buffer_limit):
        self.buffer = collections.deque(maxlen=buffer_limit)

    def put(self, transition):
        self.buffer.append(transition)

    def put_episode( self, episode_data):
        for obs, act, logp, ret, adv in episode_data:
            self.put((obs, act, logp, ret, adv))

    def sample(self, n):
        mini_batch = random.sample(self.buffer, n)
        obs_lst, act_lst, logp_lst, ret_lst, adv_lst = [], [], [], [], []

        for transition in mini_batch:
            obs, act, logp, ret, adv = transition
            obs_lst.append(obs)
            act_lst.append(act)
            logp_lst.append(logp)
            ret_lst.append(ret)
            adv_lst.append(adv)

        # NOTE: Do not Tensor preprocess observations as it is done in the policy / values function
        return obs_lst, \
               torch.tensor( act_lst).to(device), \
               torch.tensor(logp_lst).to(device), \
               torch.tensor(ret_lst).to(device), \
               torch.tensor(adv_lst).to(device)

    def size(self):
        return len(self.buffer)

    def clear(self):
        self.buffer.clear()

buffer = ReplayBuffer(args.episode_length)

# TRY NOT TO MODIFY: start the game
global_step = 0

# MODIFIED: GAE Discount computing following RLLab
import scipy.signal
def discount_cumsum(x, discount):
    """
    magic from rllab for computing discounted cumulative sums of vectors.

    input:
        vector x,
        [x0,
         x1,
         x2]

    output:
        [x0 + discount * x1 + discount^2 * x2,
         x1 + discount * x2,
         x2]
    """
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]

# MODIFIED: Helper function for sampling
# ARGUMENT: Makes the training loop cleaner and easy to follow along the algorithm
# in the paper (TODO: remove this if [un]convinced)
def sample():
    global global_step

    obs = env.reset()
    done = False

    # Temporary storage for episode data. Helps compute the GAE and Returns "lazyly"
    # (wihtout bothering about indices of the buffer)
    ep_obs, ep_acts, ep_logps, ep_rews, ep_vals = [], [], [], [], []

    # Keeping track of some sampling stats for
    ep_count = 0
    ep_lengths = []
    ep_Returns = [] # Keeps track of the undiscounted return for each sampled episode

    for step in range( args.episode_length):
        global_step += 1
        with torch.no_grad():
            action, action_logp = pg.get_action( [obs])
            action = action.numpy()[0]
            action_logp = action_logp.numpy()[0]

            v_obs =  vf.forward( [obs]).numpy()[0][0]

        next_obs, rew, done, _ = env.step( action)

        ep_obs.append( obs)
        ep_acts.append( action)
        ep_rews.append( rew)
        ep_logps.append( action_logp)
        ep_vals.append( v_obs)

        obs = next_obs

        if done:
            # Updateing sampling stats
            ep_count += 1
            ep_lengths.append( step)
            ep_Returns.append( np.sum( ep_rews))

            ep_returns = discount_cumsum( ep_rews, args.gamma)

            # Quick Hack GAE computation, which require np.array data
            ep_rews = np.array( ep_rews)
            ep_vals = np.array( ep_vals)
            deltas = ep_rews[:-1] + args.gamma * ep_vals[1:] - ep_vals[:-1]
            ep_vals = discount_cumsum(deltas, args.gamma * args.gae_lambda)

            buffer.put_episode( zip( ep_obs, ep_acts, ep_logps, ep_returns, ep_vals))

            # Cleanup the tmp holders for fresh episode data
            ep_obs, ep_acts, ep_logps, ep_rews, ep_vals = [], [], [], [], []

            obs = env.reset()
            done = False

        # Reached sampling limit, will cut the episode and use the value function
        # to bootstrap, as per OpenAI SpinUp
        if step == (args.episode_length - 1) and len(ep_obs) > 0:
            # Updateing sampling stats, although they become "unexact"
            ep_count += 1
            ep_lengths.append( step)
            ep_Returns.append( np.sum( ep_rews))

            # NOTE: Last condition: if the episode finished right at the sampling
            # limit, this will be skipped anyway
            with torch.no_grad():
                v_obs = vf.forward( [obs]).numpy()[0][0]

            ep_rews[-1] = v_obs # Boostrapping reward

            ep_returns = discount_cumsum( ep_rews, args.gamma)

            # Quick Hack GAE computation, which require np.array data
            ep_rews = np.array( ep_rews)
            ep_vals = np.array( ep_vals)
            deltas = ep_rews[:-1] + args.gamma * ep_vals[1:] - ep_vals[:-1]
            ep_vals = discount_cumsum(deltas, args.gamma * args.gae_lambda)

            buffer.put_episode( zip( ep_obs, ep_acts, ep_logps, ep_returns, ep_vals))

    sampling_stats = {
        "ep_count": ep_count,
        "train_episode_length": np.mean( ep_lengths),
        "train_episode_return": np.mean( ep_Returns)
    }

    return sampling_stats

while global_step < args.total_timesteps:
    # Will sample args.episode_length transitions, which we consider as the length of an epoch
    sampling_stats = sample()

    # Sampling all the data in the buffer
    obs_batch, act_batch, old_logp_batch, return_batch, adv_batch = buffer.sample( buffer.size())

    # Optimizaing policy network
    for i_epoch_pi in range( args.update_epochs):
        # Resample logps
        logp_a = pg.get_logproba( obs_batch, act_batch)
        ratio = (logp_a - old_logp_batch).exp()

        # Policy loss as in OpenAI SpinUp
        clip_adv = torch.where( adv_batch > 0,
                                (1.+args.clip_coef) * adv_batch,
                                (1.-args.clip_coef) * adv_batch)

        policy_loss = - torch.min( ratio * adv_batch, clip_adv).mean()

        pg_optimizer.zero_grad()
        policy_loss.backward(retain_graph=True)
        nn.utils.clip_grad_norm_(pg.parameters(), args.max_grad_norm)
        pg_optimizer.step()

        # Note: This will stop updating the policy once the KL has been breached
        if args.kl:
            approx_kl = (old_logp_batch - logp_a).mean()
            if approx_kl > args.target_kl:
                break

    # Optimizing value network
    for i_epoch in range( args.update_epochs):
        v_obs_batch = vf.forward( obs_batch).view(-1)
        v_loss = loss_fn( return_batch, v_obs_batch)

        v_optimizer.zero_grad()
        v_loss.backward(retain_graph=True)
        nn.utils.clip_grad_norm_(vf.parameters(), args.max_grad_norm)
        v_optimizer.step()

    # Annealing the rate if instructed to do so.
    if args.anneal_lr:
        pg_lr_scheduler.step()
        vf_lr_scheduler.step()

    # TRY NOT TO MODIFY: record rewards for plotting purposes
    writer.add_scalar("charts/episode_reward",
        sampling_stats["train_episode_return"], global_step)
    writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
    writer.add_scalar("losses/policy_loss", policy_loss.item(), global_step)
    # MODIFIED: After how many iters did the policy udate stop ?
    if args.kl:
        writer.add_scalar("debug/pg_stop_iter", i_epoch_pi, global_step)
    # MODIFIED: Logs some other sampling statitics
    writer.add_scalar("info/episode_count", sampling_stats["ep_count"], global_step)
    writer.add_scalar("info/mean_episode_length", sampling_stats["ep_count"], global_step)

    # TODO: Silcence this
    print( "Step %d -- PLoss: %.6f -- VLoss: %.6f -- Train Mean Return: %.6f" % (global_step,
        policy_loss.item(), v_loss.item(), sampling_stats["train_episode_return"]))

env.close()
writer.close()
