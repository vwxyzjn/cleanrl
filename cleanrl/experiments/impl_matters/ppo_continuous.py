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
    parser.add_argument('--gym-id', type=str, default="HopperBulletEnv-v0",
                       help='the id of the gym environment')
    # TODO: Remove this if consensus reached on using separate learning reate for each component
    # parser.add_argument('--learning-rate', type=float, default=7e-4,
    #                    help='the learning rate of the optimizer')
    parser.add_argument('--seed', type=int, default=1,
                       help='seed of the experiment')
    parser.add_argument('--episode-length', type=int, default=1000,
                       help='the maximum length of each episode')
    parser.add_argument('--total-timesteps', type=int, default=100000,
                       help='total timesteps of the experiments')
    parser.add_argument('--torch-deterministic', type=bool, default=True,
                       help='whether to set `torch.backends.cudnn.deterministic=True`')
    parser.add_argument('--cuda', action="store_true",
                       help='Toggles the use of CUDA whenever possible')
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
    # TODO: Discuss and eventually remove, since we do not use this in the current version
    # parser.add_argument('--vf-coef', type=float, default=0.25,
    #                    help="value function's coefficient the loss function")
    parser.add_argument('--max-grad-norm', type=float, default=0.5,
                       help='the maximum norm for the gradient clipping')
    parser.add_argument('--ent-coef', type=float, default=0.01,
                       help="policy entropy's coefficient the loss function")
    parser.add_argument('--clip-coef', type=float, default=0.2,
                       help="the surrogate clipping coefficient")
    parser.add_argument('--update-epochs', type=int, default=100,
                        help="the K epochs to update the policy")
    # Imposing KL Bound during the policy updates
    parser.add_argument('--kl', action='store_true',
                        help='If toggled, the policy updates will be early stopped w.r.t target-kl')
    parser.add_argument('--target-kl', type=float, default=0.015)
    # GAE based Advantage Estimation toggle
    parser.add_argument('--gae', action='store_true',
                        help='Use GAE for advantage computation')
    # Component wise learning rate, as per OpenAI SpinUp
    parser.add_argument('--policy-lr', type=float, default=3e-4)
    parser.add_argument('--value-lr', type=float, default=1e-3)

    # Parameterization for the tricks in the Implementation Matters paper.
    parser.add_argument('--norm-obs', action='store_true',
                        help="Toggles observation normalization")
    parser.add_argument('--norm-rewards', action='store_true',
                        help="Toggles rewards normalization")
    parser.add_argument('--norm-returns', action='store_true',
                        help="Toggles returns normalization")
    parser.add_argument('--no-obs-reset', action='store_true',
                        help="When passed, the observation filter shall not be reset after the episode")
    parser.add_argument('--no-reward-reset', action='store_true',
                        help="When passed, the reward / return filter shall not be reset after the episode")
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
device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.backends.cudnn.deterministic = args.torch_deterministic

# Helper classes for normalizations
class RunningStat(object):
    '''
    Keeps track of first and second moments (mean and variance)
    of a streaming time series.
     Taken from https://github.com/joschu/modular_rl
     Math in http://www.johndcook.com/blog/standard_deviation/
    '''
    def __init__(self, shape):
        self._n = 0
        self._M = np.zeros(shape)
        self._S = np.zeros(shape)
    def push(self, x):
        x = np.asarray(x)
        assert x.shape == self._M.shape
        self._n += 1
        if self._n == 1:
            self._M[...] = x
        else:
            oldM = self._M.copy()
            self._M[...] = oldM + (x - oldM) / self._n
            self._S[...] = self._S + (x - oldM) * (x - self._M)
    @property
    def n(self):
        return self._n
    @property
    def mean(self):
        return self._M
    @property
    def var(self):
        return self._S / (self._n - 1) if self._n > 1 else np.square(self._M)
    @property
    def std(self):
        return np.sqrt(self.var)
    @property
    def shape(self):
        return self._M.shape1

class Identity:
    '''
    A convenience class which simply implements __call__
    as the identity function
    '''
    def __call__(self, x, *args, **kwargs):
        return x

    def reset(self):
        pass

class RewardFilter:
    """
    Incorrect reward normalization [copied from OAI code]
    update return
    divide reward by std(return) without subtracting and adding back mean
    """
    def __init__(self, prev_filter, shape, gamma, clip=None):
        assert shape is not None
        self.gamma = gamma
        self.prev_filter = prev_filter
        self.rs = RunningStat(shape)
        self.ret = np.zeros(shape)
        self.clip = clip

    def __call__(self, x, **kwargs):
        x = self.prev_filter(x, **kwargs)
        self.ret = self.ret * self.gamma + x
        self.rs.push(self.ret)
        x = x / (self.rs.std + 1e-8)
        if self.clip:
            x = np.clip(x, -self.clip, self.clip)
        return x

    def reset(self):
        self.ret = np.zeros_like(self.ret)
        self.prev_filter.reset()

class ZFilter:
    """
    y = (x-mean)/std
    using running estimates of mean,std
    """
    def __init__(self, prev_filter, shape, center=True, scale=True, clip=None):
        assert shape is not None
        self.center = center
        self.scale = scale
        self.clip = clip
        self.rs = RunningStat(shape)
        self.prev_filter = prev_filter

    def __call__(self, x, **kwargs):
        x = self.prev_filter(x, **kwargs)
        self.rs.push(x)
        if self.center:
            x = x - self.rs.mean
        if self.scale:
            if self.center:
                x = x / (self.rs.std + 1e-8)
            else:
                diff = x - self.rs.mean
                diff = diff/(self.rs.std + 1e-8)
                x = diff + self.rs.mean
        if self.clip:
            x = np.clip(x, -self.clip, self.clip)
        return x

    def reset(self):
        self.prev_filter.reset()
# End of Helper classes for normalizations

# A custom environment wrapper with for observation and action normalization
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
    def __init__(self, game):
        self.env = gym.make(game)
        self.env.seed(args.seed)
        self.env.action_space.seed(args.seed)
        self.env.observation_space.seed(args.seed)

        # Adding references for obs and action space too (?)
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space

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
        if not args.no_obs_reset:
            self.state_filter.reset()
        if not args.no_reward_reset:
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

    def close(self):
        self.env.close()

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
        self.mean = nn.Linear(84, output_shape)
        self.logstd = nn.Parameter(torch.zeros(1, output_shape))

        if args.weights_init == "orthogonal":
            torch.nn.init.orthogonal_(self.fc1.weight)
            torch.nn.init.orthogonal_(self.fc2.weight)
            torch.nn.init.orthogonal_(self.mean.weight)
        elif args.weights_init == "xavier":
            torch.nn.init.xavier_uniform_(self.fc1.weight)
            torch.nn.init.xavier_uniform_(self.fc2.weight)
            torch.nn.init.xavier_uniform_(self.mean.weight)
        else:
            raise NotImplementedError

    def forward(self, x):
        x = preprocess_obs_fn(x)
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        action_mean = self.mean(x)
        action_logstd = self.logstd.expand_as(action_mean)

        return action_mean, action_logstd

    def get_action(self, x):
        mean, logstd = self.forward(x)
        std = torch.exp(logstd)
        probs = Normal(mean, std)
        action = probs.sample()
        return action, probs.log_prob(action).sum(1)

    def get_logproba(self, x, actions):
        # Note: Converts actions to tensor
        if not isinstance(actions, torch.Tensor):
            actions = preprocess_obs_fn(actions)

        action_mean, action_logstd = self.forward(x)
        action_std = torch.exp(action_logstd)
        dist = Normal(action_mean, action_std)
        return dist.log_prob(actions).sum(1)

class Value(nn.Module):
    def __init__(self):
        super(Value, self).__init__()
        self.fc1 = nn.Linear(input_shape, 64)
        self.fc2 = nn.Linear(64, 1)

        if args.weights_init == "orthogonal":
            torch.nn.init.orthogonal_(self.fc1.weight)
            torch.nn.init.orthogonal_(self.fc2.weight)
        elif args.weights_init == "xavier":
            torch.nn.init.xavier_uniform_(self.fc1.weight)
            torch.nn.init.xavier_uniform_(self.fc2.weight)
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
pg_optimizer = optim.Adam(pg.parameters(), lr=args.policy_lr)
v_optimizer = optim.Adam(vf.parameters(), lr=args.value_lr)

# MODIFIED: Initializing learning rate anneal scheduler when need
if args.anneal_lr:
    anneal_fn = lambda f: 1-f / args.total_timesteps
    pg_lr_scheduler = optim.lr_scheduler.LambdaLR(pg_optimizer, lr_lambda=anneal_fn)
    vf_lr_scheduler = optim.lr_scheduler.LambdaLR(v_optimizer, lr_lambda=anneal_fn)

loss_fn = nn.MSELoss()

# TRY NOT TO MODIFY: start the game
global_step = 0
while global_step < args.total_timesteps:
    next_obs = np.array(env.reset())

    # ALGO Logic: Storage for epoch data
    # NOTE: Changed to observations to avoid confusing with single obs
    observations = np.empty((args.episode_length,) + env.observation_space.shape)

    actions = np.empty((args.episode_length,) + env.action_space.shape)
    logprobs = np.zeros((args.episode_length,))

    rewards = np.zeros((args.episode_length,))
    returns = np.zeros((args.episode_length,))

    dones = np.zeros((args.episode_length,))
    values = torch.zeros((args.episode_length,)).to(device)

    episode_lengths = [-1]

    # TRY NOT TO MODIFY: prepare the execution of the game.
    for step in range(args.episode_length):
        global_step += 1
        observations[step] = next_obs.copy()

        # ALGO LOGIC: put action logic here
        values[step] = vf.forward(observations[step:step+1])

        with torch.no_grad():
            action, logproba = pg.get_action(observations[step:step+1])

        actions[step] = action.data.cpu().numpy()[0]
        logprobs[step] = logproba.data.cpu().numpy()[0]

        # COMMENT: Wouldn't it be better to use tanh based squashing ?
        clipped_action = np.clip(action.tolist(), env.action_space.low, env.action_space.high)[0]

        # TRY NOT TO MODIFY: execute the game and log data.
        next_obs, rewards[step], dones[step], _ = env.step(clipped_action)
        next_obs = np.array(next_obs)

        # print( "# DEBUG: Sampling step %d -- Done: %d" % (step,dones[step]))

        if dones[step]:
            # Computing the discounted returns:
            if not args.gae:
                # Classical discounted return computation
                returns[step] = rewards[step]
                for t in reversed(range(episode_lengths[-1], step)):
                    returns[t] = rewards[t] + args.gamma * returns[t+1] * (1-dones[t])
            else:
                # GAE-based discounted return computation
                deltas = np.zeros((args.episode_length,))
                advantages = np.zeros((args.episode_length,))
                prev_return = 0
                prev_value = 0
                prev_advantage = 0
                for i in reversed(range(step)):
                    returns[i] = rewards[i] + args.gamma * prev_return * (1 - dones[i])
                    deltas[i] = rewards[i] + args.gamma * prev_value * (1 - dones[i]) - values[i]
                    # ref: https://arxiv.org/pdf/1506.02438.pdf (generalization advantage estimate)
                    advantages[i] = deltas[i] + args.gamma * args.gae_lambda * prev_advantage * (1 - dones[i])

                    prev_return = returns[i]
                    prev_value = values[i]
                    prev_advantage = advantages[i]

            writer.add_scalar("charts/episode_reward", rewards[(episode_lengths[-1]+1):step].sum(), global_step)

            episode_lengths += [step]
            next_obs = np.array(env.reset())

    if not dones[step]:
        returns = np.append(returns, vf.forward(next_obs.reshape(1, -1))[0].detach().cpu().numpy(), axis=-1)
        if not args.gae:
            for t in reversed(range(episode_lengths[-1], step+1)):
                returns[t] = rewards[t] + args.gamma * returns[t+1] * (1-dones[t])
            returns = returns[:-1]
        else:
            # GAE-based discounted return computation
            deltas = np.zeros((args.episode_length,))
            advantages = np.zeros((args.episode_length,))
            prev_return = vf.forward(next_obs.reshape(1, -1))[0].detach().cpu().numpy()
            prev_value = 0
            prev_advantage = 0
            for i in reversed(range(step)):
                returns[i] = rewards[i] + args.gamma * prev_return * (1 - dones[i])
                deltas[i] = rewards[i] + args.gamma * prev_value * (1 - dones[i]) - values[i]
                # ref: https://arxiv.org/pdf/1506.02438.pdf (generalization advantage estimate)
                advantages[i] = deltas[i] + args.gamma * args.gae_lambda * prev_advantage * (1 - dones[i])

                prev_return = returns[i]
                prev_value = values[i]
                prev_advantage = advantages[i]
            returns = returns[:-1]

    # Tensorizing necessary variables
    if not args.gae:
        advantages = torch.Tensor(returns - values.detach().cpu().numpy()).to(device)
    else:
        advantages = torch.Tensor(advantages).to(device)

    logprobs = torch.Tensor(logprobs).to(device) # Called 2 times: during policy update and KL bound checked
    returns = torch.Tensor(returns).to(device) # Called 1 time when updating the values

    for i_epoch_pi in range(args.update_epochs):
        newlogproba = pg.get_logproba(observations, actions)
        ratio = (newlogproba - logprobs).exp()

        # Policy loss as in OpenAI SpinUp
        clip_adv = torch.where(advantages > 0,
                                (1.+args.clip_coef) * advantages,
                                (1.-args.clip_coef) * advantages).to(device)

        policy_loss = - torch.min(ratio * advantages, clip_adv).mean()

        pg_optimizer.zero_grad()
        policy_loss.backward() # NOTE: If retain_graph is needed for it to work, there is probably a variable that has to be detached() but is not.
        nn.utils.clip_grad_norm_(pg.parameters(), args.max_grad_norm)
        pg_optimizer.step()

        # Note: This will stop updating the policy once the KL has been breached
        # TODO: Roll back to the policy state when it was inside trust region
        if args.kl:
            approx_kl = (logprobs - newlogproba).mean()
            if approx_kl > args.target_kl:
                break

    # Optimizing value network
    for i_epoch in range(args.update_epochs):
        # Resample values
        values = vf.forward(observations).view(-1)
        v_loss = loss_fn(returns, values)

        v_optimizer.zero_grad()
        v_loss.backward()
        nn.utils.clip_grad_norm_(vf.parameters(), args.max_grad_norm)
        v_optimizer.step()

    # Annealing the rate if instructed to do so.
    if args.anneal_lr:
        pg_lr_scheduler.step()
        vf_lr_scheduler.step()

    # TRY NOT TO MODIFY: record rewards for plotting purposes
    writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
    writer.add_scalar("losses/policy_loss", policy_loss.item(), global_step)
    # Additionally, logs after how many iters did the policy udate stop ?
    if args.kl:
        writer.add_scalar("debug/pg_stop_iter", i_epoch_pi, global_step)
        writer.add_scalar("debug/approx_kl", approx_kl.item(), global_step)

env.close()
writer.close()
