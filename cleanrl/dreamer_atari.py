# TODO docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/dreamer/#dreamer_ataripy
import os
import re
import time
import uuid
import copy
import random
import argparse
import datetime
from distutils.util import strtobool

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.distributions as thd
from stable_baselines3.common.atari_wrappers import (
    ClipRewardEnv,
    EpisodicLifeEnv,
    FireResetEnv,
    MaxAndSkipEnv,
    NoopResetEnv,
)
from torch.utils.tensorboard import SummaryWriter

def parse_args():
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", type=str, default=os.path.basename(__file__).rstrip(".py"),
        help="the name of this experiment")
    parser.add_argument("--seed", type=int, default=1,
        help="seed of the experiment")
    parser.add_argument("--torch-deterministic", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, `torch.backends.cudnn.deterministic=False`")
    parser.add_argument("--cuda", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, cuda will be enabled by default")
    parser.add_argument("--track", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="if toggled, this experiment will be tracked with Weights and Biases")
    parser.add_argument("--wandb-project-name", type=str, default="cleanrl-mbrl",
        help="the wandb's project name")
    parser.add_argument("--wandb-entity", type=str, default=None,
        help="the entity (team) of wandb's project")
    parser.add_argument("--capture-video", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="whether to capture videos of the agent performances (check out `videos` folder)")

    # Algorithm specific arguments
    parser.add_argument("--env-id", type=str, default="BreakoutNoFrameskip-v4",
        help="the id of the environment")
    parser.add_argument("--num-envs", type=int, default=1,
        help="Number of parallel environments for sampling")
    parser.add_argument("--env-grayscale", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="use grayscale for the pixel-based environment wrappers")
    parser.add_argument("--env-action-repeats", type=int, default=4,
        help="the number of step for which the action of the agent is repeated onto the env.")
    parser.add_argument("--total-timesteps", type=int, default=5_000_000,
        help="total timesteps of the experiments. The amount of frames is total-timesteps * action_repeats.")
    parser.add_argument("--buffer-size", type=int, default=1_000_000,
        help="the replay memory buffer size. The amount of frames is buffer-size * action_repeats")
    parser.add_argument("--buffer-prefill", type=int, default=50_000,
        help="the number of steps to prefill the buffer with. The amount of frames is thus buffer-prefill * action_repeats.")
    parser.add_argument("--gamma", type=float, default=0.995,
        help="the discount factor gamma")
    parser.add_argument("--lmbda", type=float, default=0.95,
        help="the lambda coeff. for lambda-return computation for value loss")
    parser.add_argument("--batch-size", type=int, default=50,
        help="the batch size of sample from the reply memory")
    parser.add_argument("--batch-length", type=int, default=50,
        help="the sequence length of trajectories in the batch sampled from memory")
    ## World Model specific hyperparameters
    parser.add_argument("--rssm-hid-size", type=int, default=600,
        help="the size of hidden layers of the RSSM networks"),
    parser.add_argument("--rssm-deter-size", type=int, default=600,
        help="the size |H| of the deterministic component H of the latent state S")
    parser.add_argument("--rssm-stoch-size", type=int, default=32,
        help="the size |Y| of the stochastic component Y of the latent state S")
    parser.add_argument("--rssm-discrete", type=int, default=32,
        help="uses vector of categoricals to represent the stochastic component Y of the latent state S")
    parser.add_argument("--rew-pred-hid-size", type=int, default=400,
        help="the size of hidden layers of the reward predictor")
    parser.add_argument("--disc-pred-hid-size", type=int, default=400,
        help="the size of hidden layers of the discount preditor; set to 0 to disable")
    ## World Model KL Loss computaton speicfic
    parser.add_argument("--kl-forward", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="inverts the posterior and prior distribution in the balanced KL loss computation") # TODO: consider just nuking this
    parser.add_argument("--kl-balance", type=float, default=0.8,
        help="the KL balancing coeff. By default, a value closer to 1.0 prioritize update of the prior dist.")
    parser.add_argument("--kl-free", type=float, default=0.0,
        help="the free energy allowed in KL loss computation")
    ## World Model loss sacling coefficient
    parser.add_argument("--kl-scale", type=float, default=0.1,
        help="the scaling coeff. for KL loss")
    parser.add_argument("--rew-scale", type=float, default=1.0,
        help="the scaling coeff. for the reward prediction loss")
    parser.add_argument("--disc-scale", type=float, default=5.0,
        help="the scaling coeff. for the discount prediction loss")
    ## World model optimizer's hyper parameters
    parser.add_argument("--model-lr", type=float, default=2e-4,
        help="the learning rate of the world model Adam optimizer")
    parser.add_argument("--model-eps", type=float, default=1e-5,
        help="the 'epsilon' for the world model Adam optimizer")
    parser.add_argument("--model-wd", type=float, default=1e-6,
        help="the weight decay for the world model Adam optimizer")
    parser.add_argument("--model-grad-clip", type=float, default=100,
        help="the gradient norm clipping threshold for the world model")
    
    ## Actor (policy) specific hyperparameters
    parser.add_argument("--actor-hid-size", type=int, default=400,
        help="the size of the hidden layers of the actor network")
    parser.add_argument("--actor-entropy", type=float, default=1e-3,
        help="the entropy regulirization coefficient for the actor policy") # TODO: add support for decay schedulers
    parser.add_argument("--actor-imagine-grad-mode", type=str, default="both", choices=["dynamics", "reinforce", "both"],
        help="the method used to compute the gradients when updating the actor"
             "over imaginated trajectories")
    parser.add_argument("--actor-imagine-grad-mix", type=float, default=0.1,
        help="the ratio of using world model's reward prediction and 'reinforce' policy gradient"
             "when updating the actor using 'both' method for 'actor-imagine-actor-mode'") # TODO: add scheduled version
    ## Actor optimizer's hyper parameters
    parser.add_argument("--actor-lr", type=float, default=4e-5,
        help="the learning rate of the actor's Adam optimizer")
    parser.add_argument("--actor-eps", type=float, default=1e-5,
        help="the 'epsilon' for the actor's Adam optimizer")
    parser.add_argument("--actor-wd", type=float, default=1e-6,
        help="the weight decay for the actor's Adam optimizer")
    parser.add_argument("--actor-grad-clip", type=float, default=100,
        help="the gradient norm clipping threshold for the actor network")
    
    ## Value (critic) specific hyperparameters
    parser.add_argument("--value-hid-size", type=int, default=400,
        help="the size of the hidden layers of the value network")
    parser.add_argument("--value-slow-target-update", type=int, default=100,
        help="the frequency of update of the slow value network")
    parser.add_argument("--value-slow-target-fraction", type=float, default=1,
        help="the coefficient used to update the lagging slow value network")
    ## Value optimizer's hyper parameters
    parser.add_argument("--value-lr", type=float, default=1e-4,
        help="the learning rate of the value's Adam optimizer")
    parser.add_argument("--value-eps", type=float, default=1e-5,
        help="the 'epsilon' for the value's Adam optimizer")
    parser.add_argument("--value-wd", type=float, default=1e-6,
        help="the weight decay for the value's Adam optimizer")
    parser.add_argument("--value-grad-clip", type=float, default=100,
        help="the gradient norm clipping threshold for the value network")

    ## Dreamer specific hyperparameters
    parser.add_argument("--imagine-horizon", type=int, default=16,
        help="the number of steps to simulate using the world model ('dreaming' horizon)")
    # TODO: train-every in the paper says 4, but in the code says 16
    # Does the code consider train-every frame or train-every step environmetn ?
    parser.add_argument("--train-every", type=int, default=16,
        help="the env. steps interval after which the model is trained.")
    parser.add_argument("--viz-n-videos", type=int, default=3,
        help="the number of video samples to visualize for reconstruction and imagination")
    args = parser.parse_args()
    # fmt: on
    return args

# Helper to show expected training time in human readable form:
# Credits: https://github.com/hevalhazalkurt/codewars_python_solutions/blob/master/4kyuKatas/Human_readable_duration_format.md
# NOTE: While it breaks default logging of cleanrl, because Dreamer (mbrl) takes more time, it is informative for experimetnation
def hrd(seconds): # Human readable duration
    words = ["year", "day", "hr", "min", "sec"]
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    d, h = divmod(h, 24)
    y, d = divmod(d, 365)

    time = [y, d, h, m, s]
    duration = []

    for x, i in enumerate(time):
        if i == 1:
            duration.append(f"{i} {words[x]}")
        elif i > 1:
            duration.append(f"{i} {words[x]}s")

    if len(duration) == 1:
        return duration[0]
    elif len(duration) == 2:
        return f"{duration[0]}, {duration[1]}"
    else:
        return ", ".join(duration[:-1]) + ", " + duration[-1]

# Environment with custom Wrapper to collect datasets
def make_env(env_id, seed, idx, capture_video, run_name, buffer, args):
    # Wrapper to convert pixel observaton shape from HWC to CHW by default
    class ImagePermuteWrapper(gym.ObservationWrapper):
        def __init__(self, env):
            super().__init__(env)
            old_obs_space = env.observation_space
            self.observation_space = gym.spaces.Box(
                low=old_obs_space.low.transpose(2, 0 ,1),
                high=old_obs_space.high.transpose(2, 0, 1),
                shape=old_obs_space.shape[::-1],
                dtype=old_obs_space.dtype
            )
        
        def observation(self, observation):
            return observation.transpose(2, 0, 1)

    # Special wrapper to accumulate episode data and save them to drive for later use
    class SampleCollectionWrapper(gym.Wrapper):
        def __init__(self, env, buffer, buffer_size):
            super().__init__(env)
            self._episode_data = None
            self._train_eps_cache = buffer.train_eps_cache
            self._buffer_size = buffer_size
            # NOTE: inform the buffer about the action space of the task
            buffer.action_space = env.action_space

        def step(self, action):
            observation, reward, done, info = super().step(action)
            # Cache the trajectory data
            self._episode_data["observations"].append(observation)
            # NOTE: For Atari, store the action as one hot vector
            # TODO: handle logic when the space is continuous.
            # Probably just stoe the actions as float32 directly ?
            onehot_action = np.zeros(self.action_space.n)
            onehot_action[action] = 1
            self._episode_data["actions"].append(onehot_action.astype(np.float32))
            self._episode_data["rewards"].append(reward)
            self._episode_data["terminals"].append(done)
            if done:
                self.save_episode() # Reset takes care of cleanup

            return observation, reward, done, info
        
        def reset(self):
            first_obs = super().reset()
            self._episode_data = {
                "observations": [first_obs],
                "actions": [np.zeros(self.action_space.n)],
                "rewards": [0.0],
                "terminals": [False], # done
            }
            return first_obs
        
        def save_episode(self):
            # Prerpocess the episode data into np arrays
            self._episode_data["observations"] = \
                np.array(self._episode_data["observations"], dtype=np.uint8)
            self._episode_data["actions"] = \
                np.array(self._episode_data["actions"], dtype=np.float32)
            self._episode_data["rewards"] = \
                np.array(self._episode_data["rewards"], dtype=np.float32).reshape(-1, 1)
            self._episode_data["terminals"] = \
                np.array(self._episode_data["terminals"], dtype=np.bool8).reshape(-1, 1)
            # TODO: add proper credit for inspiration of this code
            timestamp = datetime.datetime.now().strftime('%Y%m%dT%H%M%S')
            identifier = str(uuid.uuid4().hex)
            length = len(self._episode_data["rewards"])
            filename = f"{timestamp}-{identifier}-{length}.npz"
            # TODO: is write to disk even necessary ?
            # with io.BytesIO() as f1:
            #     np.savez_compressed(f1, **self._episode_data)
            #     f1.seek(0)
            #     with open(filename, "wb") as f2:
            #         f2.write(f1.read())
            # Discard old episodes
            total = 0
            for key in reversed(sorted(self._train_eps_cache.keys())):
                if total <= self._buffer_size - length:
                    total += length - 1
                else:
                    del self._train_eps_cache[key]
            # Append the most recent episode path to the replay buffer
            self._train_eps_cache[filename] = self._episode_data.copy()
            return filename

    def thunk():
        env = gym.make(env_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        if capture_video:
            if idx == 0:
                env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        env = NoopResetEnv(env, noop_max=30)
        env = MaxAndSkipEnv(env, skip=args.env_action_repeats)
        env = EpisodicLifeEnv(env)
        if "FIRE" in env.unwrapped.get_action_meanings():
            env = FireResetEnv(env)
        env = ClipRewardEnv(env)
        env = gym.wrappers.ResizeObservation(env, (64, 64))
        if args.env_grayscale:
            env = gym.wrappers.GrayScaleObservation(env, keep_dim=True)
        env = ImagePermuteWrapper(env) # HWC -> CHW
        # TODO: fix seeding, does not seem to be working
        env.seed(seed + idx)
        env.action_space.seed(seed + idx)
        env.observation_space.seed(seed + idx)
        env.reset(seed=seed + idx)

        env = SampleCollectionWrapper(env, buffer, args.buffer_size)
        return env

    return thunk

# Buffer with support for Truncated Backpropagation Through Time (TBPTT)
class DreamerTBPTTBuffer():
    def __init__(self, config, device):
        self.np_random = np.random.RandomState(config.seed)
        self.train_eps_cache = {}
        self.B = B = config.batch_size
        self.T = config.batch_length
        self.device = device
        self.action_space = None # NOTE: this is assigend later once the env is created
        self.config = config
        # Persits episode data across consecutive batches
        self.episodes_data = [None for _ in range(B)]
        self.episodes_lengths = [0 for _ in range(B)]
        self.episodes_current_idx = [0 for _ in range(B)]

    @property
    def size(self):
        return np.sum([len(v["terminals"])-1  for v in self.train_eps_cache.values()])
    
    def sample(self):
        # TODO: more efficient method to sample that can use multiple workers
        # to hasten the sampling process a bit ?
        B, T = self.B, self.T
        # Placeholder
        C = 1 if self.config.env_grayscale else 3
        obs_list = np.zeros([B, T, C, 64, 64]) # TODO: recover obs shape
        act_list = np.zeros([B, T, self.action_space.n], dtype=np.float32)
        rew_list = np.zeros([B, T, 1], dtype=np.float32)
        ter_list = np.zeros([B, T, 1], dtype=np.bool8)

        for b in range(B):
            ssf = 0 # Steps collected so far current batch trajectory
            while ssf < T:
                edd = self.episodes_data[b]
                ep_length = self.episodes_lengths[b]
                ep_current_idx = self.episodes_current_idx[b]
                if edd is None or ep_length == ep_current_idx:
                    ep_filename = self.np_random.choice(list(self.train_eps_cache.keys()))
                    self.episodes_data[b] = edd = self.train_eps_cache[ep_filename]
                    self.episodes_lengths[b] = ep_length = len(edd["terminals"])
                    self.episodes_current_idx[b] = ep_current_idx = 0
                needed_steps = T - ssf # How many steps needed to fill the traj
                edd_start = ep_current_idx # Where to start slicing from episode data
                avail_steps = ep_length - edd_start # How many steps from the ep. data not used yet
                edd_end = min(edd_start + needed_steps, edd_start + avail_steps)
                subseq_len = edd_end - edd_start
                b_end = ssf + subseq_len
                # Fill up the batch data placeholders with steps from episode data
                obs_list[b, ssf:b_end] = edd["observations"][edd_start:edd_end]
                act_list[b, ssf:b_end] = edd["actions"][edd_start:edd_end]
                rew_list[b, ssf:b_end] = edd["rewards"][edd_start:edd_end]
                ter_list[b, ssf:b_end] = edd["terminals"][edd_start:edd_end]

                ssf = b_end
                self.episodes_current_idx[b] = edd_end
        
        # Tensorize and copy to training batch to (GPU) device
        return {
            "observations": torch.Tensor(obs_list).float().to(self.device) / 255.0 - 0.5,
            "actions": torch.Tensor(act_list).float().to(self.device),
            "rewards": torch.Tensor(rew_list).float().to(self.device),
            "terminals": torch.Tensor(ter_list).bool().to(self.device)
        }

# Bernouilli distribution with custom reparameterization 
class Bernoulli():
    """
        A binary variable distribution that supports straight-through
        gradient backprop, allowing training of the episode termination.
        
        Credits:
         - Danijar Hafner's original TF2.0 implementation: https://github.com/jsikyoon/dreamer-torch/blob/7c2331acd4fa6196d140943e977f23fb177398b3/tools.py#L312
         - Jaesik Yoon's Pytorch adaption: https://github.com/jsikyoon/dreamer-torch/blob/7c2331acd4fa6196d140943e977f23fb177398b3/tools.py#L312

        # TODO:
         - Investigate the benefit of using this custom Bernoulli dist over the default in Pytorch ?
         - Tidy up the code to be more cleanrl-ish
    """
    def __init__(self, dist=None):
        super().__init__()
        self._dist = dist
        self.mean = dist.mean

    def __getattr__(self, name):
        return getattr(self._dist, name)

    def entropy(self):
        return self._dist.entropy()

    def mode(self):
        _mode = torch.round(self._dist.mean)
        return _mode.detach() +self._dist.mean - self._dist.mean.detach()

    def sample(self, sample_shape=()):
        return self._dist.rsample(sample_shape)

    def log_prob(self, x):
        _logits = self._dist.base_dist.logits
        log_probs0 = -F.softplus(_logits)
        log_probs1 = -F.softplus(-_logits)

        return log_probs0 * (1-x) + log_probs1 * x

# Scope that enables gradient computation
class RequiresGrad():
    def __init__(self, model):
        self.model = model

    def __enter__(self):
        self.model.requires_grad_(True)

    def __exit__(self, *args):
        self.model.requires_grad_(False)

# Dreamer Agent
## World model component: handles representation and dynamics learning
class WorldModel(nn.Module):
    def __init__(self, config, num_actions):
        super().__init__()
        self.config = config
        self.num_actions = num_actions
        
        # Pre-compute the state_feat_size: |S| = |H| + |Y|
        self.state_feat_size = config.rssm_deter_size
        self.state_stoch_feat_size = config.rssm_stoch_size * config.rssm_discrete \
            if config.rssm_discrete else config.rssm_stoch_size
        self.state_feat_size += self.state_stoch_feat_size
        
        # Encoder
        C = 1 if config.env_grayscale else 3
        self.encoder = nn.Sequential(
            nn.Conv2d(C, 64, kernel_size=(4,4), stride=(2,2)), nn.ELU(),
            nn.Conv2d(64, 128, kernel_size=(4,4), stride=(2,2)), nn.ELU(),
            nn.Conv2d(128, 256, kernel_size=(4,4), stride=(2,2)), nn.ELU(),
            nn.Conv2d(256, 256, kernel_size=(4,4), stride=(2,2)), nn.ELU()
        )

        # Decoder
        class DeConvReshape(nn.Module):
            def __init__(self):
                super().__init__()
            def forward(self, x):
                return x[:, :, None, None]   # [B * T, 1536, 1, 1]
        
        self.decoder = nn.Sequential(
            nn.Linear(self.state_feat_size, 1536),
            DeConvReshape(),
            nn.ConvTranspose2d(1536, 192, kernel_size=(5,5), stride=(2,2)), nn.ELU(),
            nn.ConvTranspose2d(192, 96, kernel_size=(5,5), stride=(2,2)), nn.ELU(),
            nn.ConvTranspose2d(96, 48, kernel_size=(6,6),stride=(2,2)), nn.ELU(),
            nn.ConvTranspose2d(48, C, kernel_size=(6,6),stride=(2,2))
        )

        # Reward predictor
        N = config.rew_pred_hid_size # 400 by default
        self.reward_pred = nn.Sequential(
            nn.Linear(self.state_feat_size, N), nn.ELU(),
            nn.Linear(N, N), nn.ELU(),
            nn.Linear(N, N), nn.ELU(),
            nn.Linear(N, N), nn.ELU(),
            nn.Linear(N, 1)
        )

        # Terminal state / Discount predictor
        N = config.disc_pred_hid_size # 400 by default
        if N:
            self.disc_pred = nn.Sequential(
                nn.Linear(self.state_feat_size, N), nn.ELU(),
                nn.Linear(N, N), nn.ELU(),
                nn.Linear(N, N), nn.ELU(),
                nn.Linear(N, N), nn.ELU(),
                nn.Linear(N, 1)
            )

        # Custom GRU Cell with layer normalization
        # TODO: consider just using the default GRUCell from Pytorch instead ?
        # Based on: https://github.com/jsikyoon/dreamer-torch/blob/e42d504ea362ad605c7bbe2a0d89df1d2b3e07f2/networks.py#L451
        class GRUCell(nn.Module):
            def __init__(self, inp_size, size, norm=False, act=torch.tanh, update_bias=-1):
                super(GRUCell, self).__init__()
                self._inp_size = inp_size
                self._size = size
                self._act = act
                self._norm = norm
                self._update_bias = update_bias
                self._layer = nn.Linear(inp_size+size, 3*size, bias=norm is not None)
                if norm:
                    self._norm = nn.LayerNorm(3*size)

            def forward(self, inputs, state):
                parts = self._layer(torch.cat([inputs, state], -1))
                if self._norm:
                    parts = self._norm(parts)
                reset, cand, update = torch.split(parts, [self._size]*3, -1)
                reset = torch.sigmoid(reset)
                cand = self._act(reset * cand)
                update = torch.sigmoid(update + self._update_bias)
                output = update * cand + (1 - update) * state
                return output

        N = config.rssm_deter_size # 600 by default
        # Embeds y_{t-1} and a_{t-1} to update h_t
        self.state_action_embed = nn.Sequential(
            nn.Linear(self.state_stoch_feat_size + num_actions, N),
            nn.ELU()
        )
        # RNN for deterministic state component updates
        # h_t <- f_{h-RNN}(y_{t-1}, a_{t-1}, h_{t-1})
        self.update_s_deter = GRUCell(N, N, norm=True)

        # Helper class that returns either logits or mean,std
        # to parameter Categorical or Normal distribution for
        # the stochastic component Y of the latent state S
        class DistributionParameters(nn.Module):
            def __init__(self, input_size, hidden_size, stoch_size, discrete):
                super().__init__()
                self.discrete = discrete
                self.stoch_size = stoch_size
                self.output_size = stoch_size * discrete \
                    if  discrete else int(stoch_size * 2)
                self.network = nn.Sequential(
                    nn.Linear(input_size, hidden_size),
                    nn.ELU(),
                    nn.Linear(hidden_size, self.output_size)
                )
            
            def forward(self, x):
                x = self.network(x)

                if self.discrete:
                    return {"logits": x.view(-1, self.stoch_size, self.discrete)}
                else:
                    mean, std = torch.chunk(x, 2 ,1)
                    std = 2 * torch.sigmoid(std / 2) + 0.1
                    return {"mean": mean, "std": std}
        
        # Posterior distribution over the stochastic state comp. w_t
        self.post_state = DistributionParameters(
            input_size=1024 + config.rssm_deter_size,
            hidden_size=config.rssm_hid_size,
            stoch_size=config.rssm_stoch_size,
            discrete=config.rssm_discrete)

        # Prior distribution ove the stochastic state comp. w_t
        self.prior_state = DistributionParameters(
            input_size=config.rssm_deter_size,
            hidden_size=config.rssm_hid_size,
            stoch_size=config.rssm_stoch_size,
            discrete=config.rssm_discrete)

        # Model optimizer
        self.model_optimizer = optim.Adam(
            params=self.parameters(),
            lr=config.model_lr,
            eps=config.model_eps,
            weight_decay=config.model_wd)
    
    # Helper method that returns a distribution
    # that can be sampled from based for either
    # categorical or continuous latent variable
    def get_dist(self, dist_data):
        if self.config.rssm_discrete:
            logits = dist_data["logits"]
            dist = thd.Independent(thd.OneHotCategoricalStraightThrough(logits=logits), 1)
        else:
            mean, std = dist_data["mean"], dist_data["std"]
            dist = thd.Independent(thd.Normal(mean, std), 1)
        return dist

    # Helper method that computes the KL loss
    # for two distribution. Handles both categorical
    # and continuous variants, as well as KL balancing
    # mechanism introduced in Dreamer v2
    def kl_loss(self, post, prior, forward, balance, free):
        """
            # TODO: add attribution to the code used
            If self.config.rssm_discrete, expects
                - post || prior as a dict containing 'logits': logits list [B * T, stoch_size * disrete_size]
            else:
                - post: {post_mean_list, post_std_list}, where post_mean_list of shape [B * T, stoch-size]
                - prior: {prior_mean_list, prior_std_list}
        """
        kld = thd.kl.kl_divergence
        dist = lambda x: self.get_dist(x) # local shorthand
        sg = lambda x: {k: v.detach() if isinstance(v, list) and len(v) else v for k, v in x.items()} # stop_gradient
        
        # Dreamer v2 KL Balancing
        lhs, rhs = (prior, post) if forward else (post, prior)
        mix = balance if forward else (1. - balance)
        if balance == 0.5:
            value = kld(dist(lhs), dist(rhs))
            # TODO: how does this differ from standard KL div ?
            loss = (torch.maximum(value, torch.Tensor([free])[0])).mean()
        else:
            value_lhs = value = kld(dist(lhs), dist(sg(rhs)))
            value_rhs = kld(dist(sg(lhs)), dist(rhs))
            loss_lhs = torch.maximum(value_lhs.mean(), torch.Tensor([free])[0])
            loss_rhs = torch.maximum(value_rhs.mean(), torch.Tensor([free])[0])
            loss = mix * loss_lhs + (1 - mix) * loss_rhs

        return loss, value

    def loss(self, batch_data_dict, prev_batch_data):
        """
            "batch_data_dict" is a dictionary with the following elements:
                - "observations": [B, T, C, H, W]
                - "actions": [B, T, 1]
                - "rewards": [B, T, 1]
                - "terminals": [B, T, 1]
            
            "prev_batch_data" holds the various intermediate variables to perform TBPTT.
            
            When the "discount predictor" (terminal state prediction) is used, 
            the target data for it is generated as: discount = \gamma * (1 - terminals)
            and is used directly to compute the actor losses

            Intuitively, each batch of trajectories is structured as follows:
            
            actions:            0   a_0   a_1  ...  a_T
                                   ^ |   ^ |         |
                                  /  v  /  v         v
            latent states:     s_0  s_1   s_2  ...  s_T
                                |    |     |         |
                                v    v     v         v
            observations:      o_0  o_1   o_2  ...  o_T

            rewards             0   r_0   r_1  ...  r_T

            discounts           0   d_0   d_1  ...  d_T
        """
        config = self.config
        obs_list, act_list, rew_list, ter_list = \
            batch_data_dict["observations"], \
            batch_data_dict["actions"], \
            batch_data_dict["rewards"], \
            batch_data_dict["terminals"]
        
        B, T = obs_list.shape[:2]
        C_H_W = obs_list.shape[2:]

        if prev_batch_data is None:
            prev_batch_data = {
                "s_deter": obs_list.new_zeros([B, config.rssm_deter_size]),
                "s_stoch": obs_list.new_zeros([B, self.state_stoch_feat_size]),
                "reset_mask": obs_list.new_zeros([B, 1])
            }
        
        # Forward pass of the world model over the batch trajectory
        # Encode images into low-dimensional feature vectors: x_t <- Encoder(o_t)
        obs_feat_list = self.encoder(obs_list.view(B * T, *C_H_W)).view(B, T, -1) # [B, T, 1024]

        s_deter = prev_batch_data["s_deter"] # [B, |H|]
        s_stoch = prev_batch_data["s_stoch"] # [B, |Y|]
        reset_mask = prev_batch_data["reset_mask"] # [B, 1]

        # Placeholder lists
        post_state_dist_data = {k: [] for k in ["logits", "mean", "std"]}
        prior_state_dist_data =  {k: [] for k in ["logits", "mean", "std"]}
        s_deter_list, s_stoch_list = [], []

        for t in range(T):
            # Reset to zero in case of new trajectory
            s_deter = s_deter * reset_mask # h_{t-1}
            s_stoch = s_stoch * reset_mask # y_{t-1}
            
            prev_action = act_list[:, t] # Note that this is a_{t-1}, although it indexes with 't'

            prev_state_action_embed = torch.cat([s_stoch, prev_action], 1) # [B, 32 * 32 + |A|]
            prev_state_action_embed = self.state_action_embed(prev_state_action_embed) # represents {y,a}_{t-1}
            # h_t <- f_{h-RNN}(y_{t-1}, a_{t-1}, h_{t-1})
            s_deter = self.update_s_deter(prev_state_action_embed, s_deter)
            # Obtain logits to predict y_t from the prior dist as a function of h_t
            prior_dist_stats = self.prior_state(s_deter)
            # Obtain logits to predict y_t from the post dist. as a function of o_t and h_t
            post_dist_stats = self.post_state(torch.cat([obs_feat_list[:, t], s_deter], dim=1))
            # Sample y_t ~ q(y_t | h_t, x_t)
            s_stoch = self.get_dist(post_dist_stats).rsample()
            s_stoch = s_stoch.view(B, -1) # Mainly for discrete case, no effect otherwise

            # Store the stochstic component's distribution stats for KL loss computation
            [post_state_dist_data[k].append(v) for k,v in post_dist_stats.items()]
            [prior_state_dist_data[k].append(v) for k,v in prior_dist_stats.items()]

            # Store the s_deter and s_stoch for the imagination later
            s_deter_list.append(s_deter)
            s_stoch_list.append(s_stoch)

            # Prepare the mask to reset the s_deter and s_stoch in the next step, if needs be.
            # NOTE: This will be passed together with the latest s_{deter, stoch} for the next batch's
            reset_mask = 1 - ter_list[:, t].float()
        
        # Stack and tensorize the placeholder lists
        s_deter_list = torch.stack(s_deter_list, dim=1) # [B, T, |H|]
        s_stoch_list = torch.stack(s_stoch_list, dim=1) # [B, T, |Y|]
        s_list = torch.cat([s_deter_list, s_stoch_list], dim=2) # [B, T, |H|+|Y|]

        post_state_dist_data = {k: torch.stack(v, dim=1) if isinstance(v, list) and len(v) else v 
            for k,v in post_state_dist_data.items()} # {k: [B, T, |Y|]}
        prior_state_dist_data = {k: torch.stack(v, dim=1) if isinstance(v, list) and len(v) else v 
            for k,v in prior_state_dist_data.items()} # {k: [B, T, |Y|]}
        # End of the forward pass of the world model

        # Losses computation

        # Reconstruct observation and compute the corresponding loss
        obs_rec_mean_list = self.decoder(s_list.view(B * T, -1)).view(B, T, *C_H_W) # [B, T, C, H, W], same as the "obs_list" target
        obs_rec_dist = thd.Independent(thd.Normal(obs_rec_mean_list, 1), len(C_H_W))
        rec_loss_list = obs_rec_dist.log_prob(obs_list).neg() # [B, T]
        rec_loss = rec_loss_list.mean() # Avg. neg. log likelihood over batch size and length

        # Predict the rewards and compute the corresonding loss
        rew_pred_mean_list = self.reward_pred(s_list) # [B, T, 1], sames as the "rew_list" target
        rew_pred_dist = thd.Independent(thd.Normal(rew_pred_mean_list, 1), 1)
        rew_pred_loss_list = rew_pred_dist.log_prob(rew_list).neg() # [B, T]
        rew_pred_loss = rew_pred_loss_list.mean()

        # Compute the KL loss
        kl_loss, _ = self.kl_loss(
            post=post_state_dist_data,
            prior=prior_state_dist_data,
            forward=config.kl_forward,
            balance=config.kl_balance,
            free=config.kl_free
        )

        # Scale the losses
        kl_loss_scaled = kl_loss * config.kl_scale
        rew_pred_loss_scaled = rew_pred_loss * config.rew_scale

        # Model entropy
        post_ent = self.get_dist(post_state_dist_data).entropy().mean()
        prior_ent = self.get_dist(prior_state_dist_data).entropy().mean()

        # world model loss
        wm_loss = rec_loss + kl_loss_scaled + rew_pred_loss_scaled

        # Compute the discount prediction loss, if applicable
        if config.disc_pred_hid_size:
            discount_pred_logits_list = self.disc_pred(s_list) # [B, T, 1], same as "ter_list" target
            discount_list = (1.0 - ter_list.float()) * config.gamma
            disc_pred_dist = thd.Independent(thd.Bernoulli(logits=discount_pred_logits_list), 1)
            disc_pred_dist = Bernoulli(disc_pred_dist)
            disc_pred_loss_list = disc_pred_dist.log_prob(discount_list).neg() # [B, T]
            disc_pred_loss = disc_pred_loss_list.mean()
        
            # Scaling the loss
            disc_pred_loss_scaled = disc_pred_loss * config.disc_scale

            # Add to the world model loss
            wm_loss += disc_pred_loss_scaled
        
        wm_losses_dict = {
            "wm_loss": wm_loss, # NOTE: used for .backward() later

            # Logging
            ## Unscaled losses
            "kl_loss": kl_loss.item(),
            "rec_loss": rec_loss.item(),
            "rew_pred_loss": rew_pred_loss.item(),
            
            ## Scaled losses
            "kl_loss_scaled": kl_loss_scaled.item(),
            "rew_pred_loss_scaled": rew_pred_loss_scaled.item(),

            ## Model entropies
            "prior_ent": prior_ent.item(),
            "post_ent": post_ent.item(),

            # Coefficient
            "kl_scale": config.kl_scale,
            "kl_free": config.kl_free,
            "kl_balance": config.kl_balance,
        }

        if config.disc_pred_hid_size:
            wm_losses_dict["disc_pred_loss"] = disc_pred_loss.item()
            wm_losses_dict["disc_pred_loss_scaled"] = disc_pred_loss_scaled.item()

        # wd_fwd_dict is used later for the imagination path and actor-critic loss
        # as well as TBPTT for the next batch of trajectories
        wm_fwd_dict = {
            "s_deter_list": s_deter_list,
            "s_stoch_list": s_stoch_list,
            "s_list": s_list,
            "reset_mask": reset_mask
        }
        # Subset of batch observations sequence for video generation
        N = min(B, config.viz_n_videos)
        wm_fwd_dict["obs_rec_mean_list"] = obs_rec_mean_list[:N].detach()

        return wm_losses_dict, wm_fwd_dict

    def _train(self, batch_data_dict, prev_batch_data):
        config = self.config

        with RequiresGrad(self):
            wm_losses_dict, wm_fwd_dict = self.loss(batch_data_dict, prev_batch_data)
            self.model_optimizer.zero_grad()
            wm_losses_dict["wm_loss"].backward()
            if config.model_grad_clip:
                torch.nn.utils.clip_grad_norm_(self.parameters(), config.model_grad_clip)
            self.model_optimizer.step()
        
        return wm_losses_dict, wm_fwd_dict
    
    def imagine(self, actor, imag_traj_start_dict, horizon):
        """
            imag_traj_start_dict: contains the list of determistic (h_t), stochastic (y_t)
            components of the state belief estimated by the World Model, as well
            as the concatenated s_t = cat(h_t, y_t) which we refer to as state belief of 's_feat'.

            # TODO: consider moving imagination into the Dremaer class ? The rationale would be that
            # the method requires both ac.actor and wm, like "sample_action" does.
        """
        wm = self

        s_deter = imag_traj_start_dict["s_deter_list"] # [B * T, |H|]
        s_stoch = imag_traj_start_dict["s_stoch_list"] # [B * T, |Y|]
        s_feat = imag_traj_start_dict["s_list"] # [B * T, |H| + |Y|]
        
        B_T, _ = s_feat.shape

        # Placeholders for state beliefs and actions sampled
        imag_s_list, imag_action_list = [], []

        for _ in range(horizon):
            # Sample the action
            inp = s_feat.detach() # No gradient throug the state belief S
            action = actor(inp).rsample() # [B * T, |A|]

            # Append the sampled action and the state features
            imag_s_list.append(s_feat) # s_h
            imag_action_list.append(action) # a_h

            # Advance the internal state of the model
            state_action_embed = torch.cat([s_stoch, action], 1) # [B * T, |Y| + |A|]
            state_action_embed = wm.state_action_embed(state_action_embed)
            s_deter = wm.update_s_deter(state_action_embed, s_deter) # [B * T, |H|]

            s_dist_stats = wm.prior_state(s_deter)

            # Sample the next state's stoch. comp. y_h ~ p(y_h | h_h)
            s_stoch = wm.get_dist(s_dist_stats).rsample()
            s_stoch = s_stoch.view(B_T, -1)

            # Form the state feats s_h as concat(h_h, y_h)
            s_feat = torch.cat([s_deter, s_stoch], 1)

        imag_s_list = torch.stack(imag_s_list, dim=0) # [H, B * T, |H| + |Y|]
        imag_action_list = torch.stack(imag_action_list, dim=0) # [H, B * T, |A|]

        # Imagine the rewards using the world model;s reward predictor.
        imag_reward_list = wm.reward_pred(imag_s_list) # [H, B*T, 1]

        return {
            "imag_s_list": imag_s_list,
            "imag_action_list": imag_action_list,
            "imag_reward_list": imag_reward_list
        }


class ActorCritic(nn.Module):
    def __init__(self, config, num_actions, wm_fn):
        super().__init__()
        self.config = config
        self.num_actions = num_actions
        self.wm_fn = wm_fn # Callback method that returns the world model
        
        # Pre-compute the state_feat_size: |S| = |H| + |Y|
        self.state_feat_size = config.rssm_deter_size
        self.state_stoch_feat_size = config.rssm_stoch_size * config.rssm_discrete \
            if config.rssm_discrete else config.rssm_stoch_size
        self.state_feat_size += self.state_stoch_feat_size

        # Actor component
        N = config.actor_hid_size
        class ActorHead(nn.Module):
            def __init__(self, input_size, N, num_actions):
                super().__init__()
                self.network = nn.Sequential(
                    nn.Linear(input_size, N), nn.ELU(),
                    nn.Linear(N, N), nn.ELU(),
                    nn.Linear(N, N), nn.ELU(),
                    nn.Linear(N, N), nn.ELU(),
                    nn.Linear(N, num_actions) # logits layer
                )

            def __call__(self, x):
                x = self.network(x) # Obtain logits
                return thd.OneHotCategoricalStraightThrough(logits=x) # Return the action distribution

        self.actor = ActorHead(self.state_feat_size, N, num_actions)

        # Value component
        N = config.value_hid_size
        self.value = nn.Sequential(
            nn.Linear(self.state_feat_size, N), nn.ELU(),
            nn.Linear(N, N), nn.ELU(),
            nn.Linear(N, N), nn.ELU(),
            nn.Linear(N, N), nn.ELU(),
            nn.Linear(N, 1)
        )
        # Lagging value network
        self.slow_value = copy.deepcopy(self.value)
        self.slow_value.requires_grad_(False)

        # Actor optimizer
        self.actor_optimizer = optim.Adam(
            params=self.actor.parameters(),
            lr=config.actor_lr,
            eps=config.actor_eps,
            weight_decay=config.actor_wd)
        
        # Value optimizer
        self.value_optimizer = optim.Adam(
            params=self.value.parameters(),
            lr=config.value_lr,
            eps=config.value_eps,
            weight_decay=config.value_wd)
        
        # Tracking training stats
        self.register_buffer("n_updates", torch.LongTensor([0]))
        self.register_buffer("slow_value_n_updates", torch.LongTensor([0]))
    
    def compute_targets_weights(self, imagine_data_dict):
        """
            Computes the targets and weights to update the actor and value losses later.
                - targets are a form of approximation for the expected returns, shape [H-1, B * T, 1]
                - weights are coefficient that incorporate long-term discounting, shape [H, B * T, 1]
            
            Expects:
                imag_s_list: list of imaginary state featurers: [Hor, B * T, |H| + |Y|]
        """
        config = self.config
        wm = self.wm_fn() # Callback to the worldmodel

        imag_s_list = imagine_data_dict["imag_s_list"]
        imag_reward_list = imagine_data_dict["imag_reward_list"]

        # Compute the discount factor either based on the world model's predictor, or a fixed heuristic
        if config.disc_pred_hid_size:
            # TODO: Does using .Bernoulli_dist.mean result in discount value similar to gamma ?
            imag_discount_dist = thd.Independent(thd.Bernoulli(logits=wm.disc_pred(imag_s_list)), 1)
            imag_discount_list = Bernoulli(imag_discount_dist).mean # [H, B * T, 1]
        else:
            imag_discount_list = config.gamma * torch.ones_like(imag_reward_list) # [H, B * T, 1]
        
        # Compute state value list for the actor target computation first
        imag_actor_state_value_list = self.slow_value(imag_s_list) # [H, B * T, 1]
        
        # Helper to compute the lambda returns for the actor targets
        # TODO: attribution, and clean up, make it easier to understand
        def lambda_return(reward, value, pcont, bootstrap, lambda_):
            # Setting lambda=1 gives a discounted Monte Carlo return.
            # Setting lambda=0 gives a fixed 1-step return.
            assert len(reward.shape) == len(value.shape), (reward.shape, value.shape)
            if bootstrap is None:
                bootstrap = torch.zeros_like(value[-1])
            next_values = torch.cat([value[1:], bootstrap[None]], 0)
            inputs = reward + pcont * next_values * (1 - lambda_)
            
            def static_scan_for_lambda_return(fn, inputs, start):
                last = start
                indices = range(inputs[0].shape[0])
                indices = reversed(indices)
                flag = True
                for index in indices:
                    inp = lambda x: (_input[x] for _input in inputs)
                    last = fn(last, *inp(index))
                    if flag:
                        outputs = last
                        flag = False
                    else:
                        outputs = torch.cat([outputs, last], dim=-1)
                outputs = torch.reshape(outputs, [outputs.shape[0], outputs.shape[1], 1])
                outputs = torch.unbind(outputs, dim=0)
                return outputs

            returns = static_scan_for_lambda_return(lambda agg, cur0, cur1: cur0 + cur1 * lambda_ * agg, (inputs, pcont), bootstrap)
            
            return torch.stack(returns, dim=1) # [H, B * T, 1]

        targets = lambda_return(
            imag_reward_list[:-1],
            imag_actor_state_value_list[:-1],
            imag_discount_list[:-1],
            bootstrap=imag_actor_state_value_list[-1],
            lambda_=config.lmbda) # [H-1, B * T, 1], the last state lost for boostrapping

        weights = torch.cumprod(torch.cat([torch.ones_like(imag_discount_list[:1]), imag_discount_list[:-1]], 0), 0).detach() # [H+1, B * T, 1]

        return targets, weights

    def compute_actor_loss(self, imagine_data_dict):
        config = self.config
        actor, value = self.actor, self.value

        # Recover the necessary simulated trajectory data
        imag_s_list = imagine_data_dict["imag_s_list"] # [Horizon, B * T, |H| + |Y|]
        imag_action_list = imagine_data_dict["imag_action_list"] # [Horizon, B * T, |A|]
        
        # Compute action logprobs over simulated steps
        inp = imag_s_list.detach()
        imag_action_dist = actor(inp) # Get action dist for all imaginary states
        imag_action_logprob_list = imag_action_dist.log_prob(imag_action_list)
        imag_actor_entropy_list = imag_action_dist.entropy() # [H, B * T]

        # Precompute the targets and weights for the actor loss
        # Targets are computed using the lambda-return method
        targets, weights = self.compute_targets_weights(imagine_data_dict)

        # Actor loss computation
        if config.actor_imagine_grad_mode == "dynamics":
            actor_targets = targets
        elif config.actor_imagine_grad_mode in ["reinforce", "both"]:
            # TODO: Original TF2 uses "slow_baseline", but jsikyoon implementation does not
            baseline = value(imag_s_list[:-1])
            advantages = (targets - baseline).detach()
            actor_targets = imag_action_logprob_list[:-1][:, :, None] * advantages
            if config.actor_imagine_grad_mode == "both":
                # NOTE: Default setting prioritize 'reinforce' actor loss
                mix = config.actor_imagine_grad_mix#() # TODO: add decay support
                actor_targets = mix * targets + (1 - mix) * actor_targets
        else:
            raise NotImplementedError(f"Unsupported actor update mode during imagination: {config.actor_imagine_grad_mode}.")

        ## Entropy regularization for the actor loss
        actor_entropy_scale = config.actor_entropy#() # TODO: add suport for decay
        actor_targets += actor_entropy_scale * imag_actor_entropy_list[:-1][:, :, None]

        actor_loss = (weights[:-1] * actor_targets).neg().mean()

        actor_losses_dict = {
            "actor_loss": actor_loss, # Used for .backward()
            # Actor training stats
            # TODO: track the actopyr entropy coefficient in case it is decayed and what not
            "actor_ent": imag_actor_entropy_list.mean().item(),
            "actor_entropy_scale": actor_entropy_scale
        }
        if config.actor_imagine_grad_mode == "both":
            actor_losses_dict["imag_gradient_mix"] = mix.item() if isinstance(mix, torch.Tensor) else mix

        value_train_data = {
            "targets": targets, # Re-used for value loss
            "weights": weights, # Re-used for value loss
        }

        return actor_losses_dict, value_train_data

    def compute_value_loss(self, imagine_data_dict, value_train_data):
        config = self.config
        
        imag_s_list = imagine_data_dict["imag_s_list"]

        # Re-use targets and weights if appropriate
        targets, weights = value_train_data["targets"], value_train_data["weights"]

        values_mean_list = self.value(imag_s_list[:-1].detach()) # [H, B * T, 1]
        values_dist = thd.Independent(thd.Normal(values_mean_list, 1), 1)
        value_loss = values_dist.log_prob(targets.detach()).neg() # [H, B * T]
        value_loss = (value_loss * weights[:-1].squeeze(-1)).mean()

        # Compute the error between the "value" and "slow_value" component
        # This is mostly for debugs
        value_abs_error = (values_mean_list - self.slow_value(imag_s_list[:-1].detach())).abs().mean().item()

        # Update the slow value target
        if self.n_updates % config.value_slow_target_update == 0:
            tau = config.value_slow_target_fraction
            for s, d in zip(self.value.parameters(), self.slow_value.parameters()):
                d.data = tau * s.data + (1 - tau) * d.data
            self.slow_value_n_updates += 1
        
        return {"value_loss": value_loss, "value_abs_error": value_abs_error}

    def _train(self, ac_train_data):
        config = self.config
        actor, value = self.actor, self.value
        wm = self.wm_fn()

        # Actor loss computation
        with RequiresGrad(actor):
            # Use the world model to simulate trajectories
            # This needs to be done under the actor's gradient computation scope
            # to compute the gradient of the action through the world model dynamics
            imagine_data_dict = wm.imagine(actor, ac_train_data, config.imagine_horizon)

            # Compute the actor loss and other related training stats
            actor_losses_dict, value_train_data = \
                self.compute_actor_loss(imagine_data_dict)

            # Optimize the actor network
            self.actor_optimizer.zero_grad()
            actor_losses_dict["actor_loss"].backward()
            if config.actor_grad_clip:
                torch.nn.utils.clip_grad_norm_(actor.parameters(), config.actor_grad_clip)
            self.actor_optimizer.step()
        
        # Value loss computation
        with RequiresGrad(value):
            value_losses_dict = self.compute_value_loss(imagine_data_dict, value_train_data)

            # Optimize the value network
            self.value_optimizer.zero_grad()
            value_losses_dict["value_loss"].backward()
            if config.value_grad_clip:
                torch.nn.utils.clip_grad_norm_(value.parameters(), config.value_grad_clip)
            self.value_optimizer.step()

        # Track training stats of the AC component
        self.n_updates += 1 # Track the number of grad step on the model

        ac_train_losses = {
            **actor_losses_dict,
            **value_losses_dict,
            **{
                "slow_value_n_updates": self.slow_value_n_updates
            }
        }

        return ac_train_losses, {"imag_s_list": imagine_data_dict["imag_s_list"]}


class Dreamer(nn.Module):
    def __init__(self, config, num_actions):
        super().__init__()
        self.config = config
        self.num_actions = num_actions # TODO: adopt a more general notation like "action_dim" ?
        
        self.wm = WorldModel(config=args, num_actions=num_actions)
        self.ac = ActorCritic(config=args, num_actions=num_actions, wm_fn=lambda: self.wm)

        # Tracking training stats
        self.register_buffer("step", torch.LongTensor([0])) # How many env. steps so far, for schedulers
        self.register_buffer("n_updates", torch.LongTensor([0])) # How many updates of the model

        # TODO: add schedules for actor entropy and actor imaginate gradient mixing ?
        # Althoug the paper reports results with fixed hyper parameters for simplicity.

    @torch.no_grad()
    def sample_action(self, obs, prev_data, mode="train"):
        config = self.config
        wm = self.wm # World model shorthand
        actor = self.ac.actor # Actor shorthand
        B = obs.shape[0]

        obs_feat = wm.encoder(obs).view(B, -1) # [B, 1024]

        if prev_data == None: # Dummy previous internal state for the first step
            prev_data = {
                "s_deter": obs.new_zeros([B, config.rssm_deter_size]),
                "s_stoch": obs.new_zeros([B, wm.state_stoch_feat_size]),
                "action": obs.new_zeros([B, self.num_actions]),
                "done": obs.new_zeros([B, 1])
            }
        
        # Reset the internal state in case a new episode has started
        done = prev_data["done"]
        mask = 1. - done.float()
        s_deter = prev_data["s_deter"] * mask
        s_stoch = prev_data["s_stoch"] * mask
        action = prev_data["action"] * mask

        prev_state_action_embed = torch.cat([s_stoch, action], 1)
        prev_state_action_embed = wm.state_action_embed(prev_state_action_embed)
        s_deter = wm.update_s_deter(prev_state_action_embed, s_deter)

        # Sample y_t ~ q(y_t | h_t, x_t)
        s_stoch_dist_stats = wm.post_state(torch.cat([obs_feat, s_deter], 1))
        s_stoch = wm.get_dist(s_stoch_dist_stats).rsample()
        s_stoch = s_stoch.view(B, -1) # Mainly for discrete case, no effect otherwise

        # Concatenate the deter and stoch. componens to form the state features
        s_feat = torch.cat([s_deter, s_stoch], 1)

        # Sample the action
        if mode == "train":
            action = actor(s_feat).rsample()
        elif mode == "eval":
            action = actor(s_feat).mode()
        else:
            raise NotImplementedError(f"Unsupported sampling regime for action: {mode}")

        return action.argmax(dim=1).cpu().numpy(), {"s_deter": s_deter, "s_stoch": s_stoch, "action": action}

    def _train(self, batch_data_dict, prev_batch_data):
        config = self.config

        # NOTE: Pixel-observations are pre-processed at the buffer level already
        wm_losses_dict, wm_fwd_dict = self.wm._train(batch_data_dict, prev_batch_data)
        
        # Batch size and length
        B, T = batch_data_dict["observations"].shape[:2]
        B_T = B * T # How many starting steps for imagination: B * T in general

        # Prepare the data for the ActorCritic updates
        if config.disc_pred_hid_size:
            # Masks to remove the terminal steps for imagination
            ter_list = batch_data_dict["terminals"]
            masks_list = torch.logical_not(ter_list)
            B_T = masks_list.sum()

            # Drop the terminal steps from the data used to 
            # bootstrap the imagination process for actor critic training
            # .detach() is used to block the gradient flow from AC to WM component
            ac_train_data = {k: wm_fwd_dict[k].detach() for k in ["s_deter_list", "s_stoch_list"]}
            ac_train_data = {k: torch.masked_select(v, masks_list).view(B_T, -1) for k,v in ac_train_data.items()}
        else:
            ac_train_data = {k: wm_fwd_dict[k].detach().view(B_T,-1) for k in ["s_deter_list", "s_stoch_list"]}
        
        # Update the ActorCritic component
        ac_losses_dict, ac_fwd_dict = self.ac._train(ac_train_data)

        # Preserve intermediate results for the next batch, required for TBPTT.
        prev_batch_data = {
            "s_deter": wm_fwd_dict["s_deter_list"][:, -1].detach(), # [B, |H|]
            "s_stoch": wm_fwd_dict["s_stoch_list"][:, -1].detach(), # [B, |Y|]
            "reset_mask": wm_fwd_dict["reset_mask"] # [B, 1]
        }

        # Track training stats
        self.n_updates += 1
        
        return wm_losses_dict, ac_losses_dict, wm_fwd_dict, ac_fwd_dict, prev_batch_data

    @torch.no_grad()
    def gen_train_rec_videos(self, batch_data_dict, wm_fwd_dict):
        # Generates video of batch trajectories' ground truth, reconstruction and error
        obs_rec_mean_list = wm_fwd_dict["obs_rec_mean_list"] # [N, T, C, H, W]
        N = obs_rec_mean_list.shape[0]
        orig_obs_list = batch_data_dict["observations"][:N] + 0.5 # [N, T, C ,H, W] in range [0.0,1.0]
        rec_obs_list = (obs_rec_mean_list + 0.5).clamp(0, 1) # [N, T, C, H, W] in range [0.0,1.0]
        error = (rec_obs_list - orig_obs_list + 1) / 2.
        black_strip = orig_obs_list.new_zeros([N, *orig_obs_list.shape[1:-2], 3, orig_obs_list.shape[-1]]) # [N, T, C, 3, W]
        train_video = torch.cat([orig_obs_list, black_strip, rec_obs_list, black_strip, error], 3) # [N, T, C, H * 3 + 6, W]
        train_video = torch.cat([tnsr for tnsr in train_video], dim=3)[None] # [1, T, C, H * 3 + 6, W * N] # Vertical stack
        return train_video.cpu().numpy()

    @torch.no_grad()
    def gen_imag_rec_videos(self, batch_data_dict, ac_fwd_dict):
        # Generates video of reconstructed imaginary trajectories
        N = self.config.viz_n_videos
        Hor = self.config.imagine_horizon # Hor
        C_H_W = batch_data_dict["observations"].shape[2:] # C,H,W
        imag_s_list = ac_fwd_dict["imag_s_list"]
        imag_s_list = torch.stack([imag_s_list[:, i * args.batch_length] for i in range(N)], dim=0) # [N, Hor, |S|]
        imag_s_list = imag_s_list.view(N * Hor, -1) # [N * Hor, |S|]
        imag_traj_video = (self.wm.decoder(imag_s_list).view(N, Hor, *C_H_W) + .5).clamp(0.0, 1.0) # [N, Hor, C, H, W]
        black_strip = imag_traj_video.new_zeros([*imag_traj_video.shape[1:-1], 3])
        imag_traj_video_processed = torch.cat([torch.cat([tnsr, black_strip], 3) for tnsr in imag_traj_video], dim=3)[None] # [1, N, Hor, C, H, (W + 3)* N]
        imag_traj_video_processed = imag_traj_video_processed[:, :, :, :, :-3] # [1, Hor, C, H, (W + 3)* N - 3]
        return imag_traj_video_processed.cpu().numpy()
    
if __name__ == "__main__":
    args = parse_args()
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # Instantiate buffer
    buffer = DreamerTBPTTBuffer(args, device)
    # Env setup
    envs = gym.vector.SyncVectorEnv([make_env(args.env_id, args.seed, i, args.capture_video,
        run_name, buffer, args) for i in range(args.num_envs)])
    assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"
    
    # Instantiate the Dreamer agent
    dreamer = Dreamer(config=args, num_actions=envs.single_action_space.n).to(device)
    print(dreamer) # Useful to check the agent's structure for debug
    from torchinfo import summary
    summary(dreamer)

    # TRY NOT TO MODIFY: start the game
    start_time = time.time()
    obs = envs.reset()
    prev_data, prev_batch_data = None, None
    n_episodes = 0
    total_updates = args.total_timesteps // args.train_every
    for global_step in range(0, args.total_timesteps, args.num_envs):
        # ALGO LOGIC: put action logic here
        if global_step <= args.buffer_prefill:
            action = envs.action_space.sample()
        else:
            # Tensorize observation, pre-process and put on training device
            obs_th = torch.Tensor(obs).to(device) / 255.0 - 0.5
            action, prev_data = dreamer.sample_action(obs_th, prev_data, mode="train")

        # TRY NOT TO MODIFY: execute the game and log data.
        next_obs, rewards, dones, infos = envs.step(action)

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        if int(np.sum(dones)):
            for info in infos:
                if "episode" in info.keys():
                    print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
                    writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
                    writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)
                    writer.add_scalar("charts/episodic_length_frames", info["episode"]["l"] * args.env_action_repeats, global_step)
                    break
            n_episodes += np.sum(dones)
            # writer.add_scalar("charts/buffer_steps", buffer.size, global_step)
            # writer.add_scalar("charts/buffer_frames", buffer.size * args.env_action_repeats, global_step)
            writer.add_scalar("charts/n_episodes", n_episodes, global_step)
        
        # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
        obs = next_obs
        if isinstance(prev_data, dict):
            # If we have started sampling action with the Dreamer agent,
            # need to set "done" to properly mask the previous step's data
            # when a new episode starts
            prev_data["done"] = torch.Tensor(dones)[:, None].to(device) # [NUM_ENVS, 1]

        # Tracks number of env. steps so far, namely for decay schedulers
        dreamer.step += args.num_envs * args.env_action_repeats

        # ALGO LOGIC: training.
        if global_step >= args.buffer_prefill and global_step % args.train_every == 0:
            # TODO: determine the proper relation between train_every and actiopn_repeats
            batch_data_dict = buffer.sample()
            # TODO: consolidate losses into a signle dictionary ?
            wm_losses_dict, ac_losses_dict, wm_fwd_dict, ac_fwd_dict, prev_batch_data = \
                dreamer._train(batch_data_dict, prev_batch_data)

            # Logging training stats every 10 effective model updates (.backward() calls)
            if dreamer.n_updates % 10 == 0:
                # Logging training stats of the WM and AC components in their respective scopes
                [writer.add_scalar(f"wm/{k}", v, global_step) for k,v in wm_losses_dict.items()]
                [writer.add_scalar(f"ac/{k}", v, global_step) for k,v in ac_losses_dict.items()]
                UPS = dreamer.n_updates.item() / (time.time() - start_time) # Update steps per second
                PRGS = dreamer.n_updates / total_updates # Progress rate, monitor progressino rate in Wandb
                print("SPS:", int((global_step - args.buffer_prefill) / (time.time() - start_time)), end=" | ")
                print(f"UPS: {UPS: 0.3f}", end=" | ")
                print("ETA:", hrd(total_updates / UPS))
                writer.add_scalar("global_step", global_step, global_step)
                writer.add_scalar("global_frame", global_step * args.env_action_repeats, global_step)
                writer.add_scalar("charts/SPS", (global_step - args.buffer_prefill) / (time.time() - start_time), global_step)
                writer.add_scalar("charts/FPS", (global_step - args.buffer_prefill) * args.env_action_repeats / (time.time() - start_time), global_step)
                writer.add_scalar("charts/n_updates", dreamer.n_updates, global_step)
                writer.add_scalar("charts/UPS", UPS, global_step)
                writer.add_scalar("charts/PRGS", PRGS, global_step)
                buffer_size = buffer.size
                writer.add_scalar("charts/buffer_steps", buffer_size, global_step)
                writer.add_scalar("charts/buffer_frames", buffer_size * args.env_action_repeats, global_step)

            # NOTE: too frequent video logging will add overhead to training time
            if dreamer.n_updates % 500 == 0:
                # Logging video of trajectory reconstruction from the WM training
                train_rec_video = dreamer.gen_train_rec_videos(batch_data_dict, wm_fwd_dict)
                writer.add_video("train_rec_video", train_rec_video, global_step, fps=16)

                # Logging video of the reconstructed imaginary trajectories from the AC training
                imag_rec_video = dreamer.gen_imag_rec_videos(batch_data_dict, ac_fwd_dict)
                writer.add_video("imag_rec_video", imag_rec_video, global_step, fps=16)

    envs.close()
    writer.close()
