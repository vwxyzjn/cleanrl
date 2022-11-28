# TODO docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/dreamer/#dreamer_ataripy
import os
import io
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
import torch.nn.functional as F
import torch.optim as optim
import torch.distributions as thd
from torch.utils.data import IterableDataset, DataLoader
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
    parser.add_argument("--wandb-project-name", type=str, default="cleanRL",
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
    # TODO: add parameterization for environment action repeat, especially for other tasks
    parser.add_argument("--total-timesteps", type=int, default=10_000_000,
        help="total timesteps of the experiments")
    parser.add_argument("--buffer-size", type=int, default=1_000_000,
        help="the replay memory buffer size")
    parser.add_argument("--buffer-prefill", type=int, default=50_000,
        help="the number of steps to prefill the buffer with ( without action repeat)")
    parser.add_argument("--gamma", type=float, default=0.995,
        help="the discount factor gamma")
    parser.add_argument("--lambda", type=float, default=0.95,
        help="the lambda coeff. for lambda-return computation for value loss")
    parser.add_argument("--batch-size", type=int, default=32,
        help="the batch size of sample from the reply memory")
    parser.add_argument("--batch-length", type=int, default=32,
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
    parser.add_argument("--value-slow-target", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="use a lagging copy of the value network for actor loss computation") 
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
    parser.add_argument("--train-every", type=int, default=16,
        help="the env. step interval after which the model is trained")
    parser.add_argument("--viz-n-videos", type=int, default=3,
        help="the number of video samples to visualize for reconstruction and imagination")
    args = parser.parse_args()
    # fmt: on
    return args

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
    class DatasetCollectionWrapper(gym.Wrapper):
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
            # TODO: ASCII art of how the data is stored
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
        env = MaxAndSkipEnv(env, skip=4) # TODO: parameterize action repeat ?
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

        # TODO: specificy relation with actio repeat, use it instead of // 4
        env = DatasetCollectionWrapper(env, buffer, args.buffer_size // 4)
        return env

    return thunk

# Buffer with support for Truncated Backpropagation Through Time (TBPTT)
class DreamerTBPTTDataset():
    def __init__(self, config, device):
        self.random = np.random.RandomState(config.seed)
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
    def ready_to_sample(self):
        return len(self.train_eps_cache.keys()) > self.B
    
    def sample(self):
        # TODO: more efficient method to sample that can use multiple workers
        # to hasten the sampling process a bit
        assert self.ready_to_sample, \
            "Attempting to sample from buffer without enough episodes data."
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
                    ep_filename = np.random.choice(list(self.train_eps_cache.keys()))
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
        self.state_stoch_feat_size
        
        # Encoder
        C = 1 if config.env_grayscale else 3
        self.encoder = nn.Sequential(
            nn.Conv2d(C, 64, kernel_size=(4,4), stride=(2,2)),
            nn.ELU(),
            nn.Conv2d(64, 128, kernel_size=(4,4), stride=(2,2)),
            nn.ELU(),
            nn.Conv2d(128, 256, kernel_size=(4,4), stride=(2,2)),
            nn.ELU(),
            nn.Conv2d(256, 256, kernel_size=(4,4), stride=(2,2)),
            nn.ELU()
        )

        # Decoder
        class DeConvReshape(nn.Module):
            def __init__(self):
                super().__init__()
            def forward(self, x):
                return x[:, :, None, None]   # [B * T, 2048, 1, 1]
        
        self.decoder = nn.Sequential(
            nn.Linear(self.state_feat_size, 2048),
            DeConvReshape(),
            nn.ConvTranspose2d(2048, 256, kernel_size=(5,5), stride=(2,2)),
            nn.ELU(),
            nn.ConvTranspose2d(256, 128, kernel_size=(5,5), stride=(2,2)),
            nn.ELU(),
            nn.ConvTranspose2d(128, 64, kernel_size=(6,6),stride=(2,2)),
            nn.ELU(),
            nn.ConvTranspose2d(64, C, kernel_size=(6,6),stride=(2,2))
        )

        # Reward predictor
        N = config.rew_pred_hid_size # 400 by default
        self.reward_pred = nn.Sequential(
            nn.Linear(self.state_feat_size, N),
            nn.ELU(),
            nn.Linear(N, N),
            nn.ELU(),
            nn.Linear(N, N),
            nn.ELU(),
            nn.Linear(N, N),
            nn.ELU(),
            nn.Linear(N, 1)
        )

        # Terminal state / Discount predictor
        N = config.disc_pred_hid_size # 400 by default
        if N:
            self.disc_pred = nn.Sequential(
                nn.Linear(self.state_feat_size, N),
                nn.ELU(),
                nn.Linear(N, N),
                nn.ELU(),
                nn.Linear(N, N),
                nn.ELU(),
                nn.Linear(N, N),
                nn.ELU(),
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

            @property
            def state_size(self):
                return self._size

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
                    std = 2 * torch.sigmoid(x / 2) + 0.1 # TODO: consider simpler parameterization ?
                    return {"mean": mean, "std": std}
        
        # Posterior distribution over the stochastic state comp. w_t
        # TODO: find a better way to handle the 1536 obs_feat size ?
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
        class OneHotDist(thd.one_hot_categorical.OneHotCategorical):
            """
                A one-hot categorical distribution that supports straight-through
                gradient backprop, allowing training of discrete latent variabless.
                # TODO: attribution
            """
            def __init__(self, logits=None, probs=None):
                super().__init__(logits=logits, probs=probs)

            def mode(self):
                _mode = F.one_hot(torch.argmax(super().logits, axis=-1), super().logits.shape[-1])
                return _mode.detach() + super().logits - super().logits.detach()

            def rsample(self, sample_shape=(), seed=None):
                if seed is not None:
                    raise ValueError('need to check')
                sample = super().sample(sample_shape)
                probs = super().probs
                while len(probs.shape) < len(sample.shape):
                    probs = probs[None]
                sample += probs - probs.detach()
                return sample

        if self.config.rssm_discrete:
            logits = dist_data["logits"]
            dist = thd.Independent(OneHotDist(logits=logits), 1)
        else:
            mean, std = dist_data["mean"], dist_data["std"]
            dist = thd.Independent(thd.Normal(mean, std), 1)
        return dist

    def forward(self, batch_data_dict, prev_batch_data):
        """
            "batch_data_dict" is a dictionary with the following elements:
                - "observations": [B, T, 1, 64, 64]
                - "actions": [B, T, 1]
                - "rewards": [B, T, 1]
                - "terminals": [B, T, 1]
                - "discount": [B, T, 1] # TODO: precompute this for later
            
            "prev_batch_data" holds the various intermediate variables to 
                perform TBPTT. # TODO: make sure it is detached before being passed into this
            
            When the "discount predictor" (terminal state prediction) is used, 
            the target data for it is generated as: discount = \gamma * (1 - terminals)
            and is used directly to compute the actor losses

            Intuitively, each batch of trajectories is structured as follows:
            
            actions:           -1   a_0   a_1  ...  a_T
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
        obs_list, act_list, ter_list = \
            batch_data_dict["observations"], \
            batch_data_dict["actions"], \
            batch_data_dict["terminals"]
        
        B, T = obs_list.shape[:2]
        C_H_W = obs_list.shape[2:]

        if prev_batch_data is None:
            prev_batch_data = {
                "s_deter": obs_list.new_zeros([B, config.rssm_deter_size]),
                "s_stoch": obs_list.new_zeros([B, self.state_stoch_feat_size]),
                "reset_mask": obs_list.new_zeros([B, 1])
            }
        
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
            reset_mask = 1 - ter_list[:, t].float()
        
        # Stack and tensorize the placeholder lists
        s_deter_list = torch.stack(s_deter_list, dim=1)
        s_stoch_list = torch.stack(s_stoch_list, dim=1)
        post_state_dist_data = {k: torch.stack(v, dim=1) if isinstance(v, list) and len(v) else v 
            for k,v in post_state_dist_data.items()}
        prior_state_dist_data = {k: torch.stack(v, dim=1) if isinstance(v, list) and len(v) else v 
            for k,v in prior_state_dist_data.items()}

        return {
            "s_deter_list": s_deter_list,
            "s_stoch_list": s_stoch_list,
            "post_state_dist_data": post_state_dist_data,
            "prior_state_dist_data": prior_state_dist_data,
            "reset_mask": reset_mask
        }
    
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
        config = self.config
        B, T = batch_data_dict["observations"].shape[:2]
        C_H_W = batch_data_dict["observations"].shape[2:]

        # Run the forward pass of the model with inference and generation path
        wm_fwd_dict = self(batch_data_dict, prev_batch_data)
        s_deter_list = wm_fwd_dict["s_deter_list"] # [B, T, |H|]
        s_stoch_list = wm_fwd_dict["s_stoch_list"] # [B, T, |Y|]
        post_state_dist_data = wm_fwd_dict["post_state_dist_data"] # {k: [B, T, |Y|]}
        prior_state_dist_data = wm_fwd_dict["prior_state_dist_data"] # {k: [B, T, |Y|]}

        s_list = torch.cat([s_deter_list, s_stoch_list], dim=2) # [B, T, |H|+|Y|]

        # Reconstruct observation and compute the corresponding loss
        obs_rec_mean_list = self.decoder(s_list.view(B * T, -1)).view(B, T, *C_H_W) # [B, T, C, H, W]
        obs_list = batch_data_dict["observations"]
        obs_rec_dist = thd.Independent(thd.Normal(obs_rec_mean_list, 1.0), len(C_H_W))
        rec_loss_list = obs_rec_dist.log_prob(obs_list).neg() # [B, T]
        rec_loss = rec_loss_list.mean() # Avg. neg. log likelihood over batch size and length

        # Predict the rewards and compute the corresonding loss
        rew_pred_mean_list = self.reward_pred(s_list) # [B, T, 1]
        rew_list = batch_data_dict["rewards"] # [B, T, 1]
        rew_pred_dist = thd.Independent(thd.Normal(rew_pred_mean_list, 1.0), 1)
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
        wm_loss = rec_loss + rew_pred_loss_scaled

        # Compute the discount prediction loss, if applicable
        if config.disc_pred_hid_size:
            # Straight-through reparameterized Bernoulli distribution
            class Bernoulli():
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
                
            discount_pred_logits_list = self.disc_pred(s_list) # [B, T, 1]
            ter_list = batch_data_dict["terminals"]
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
            "kl_balance": config.kl_balance
        }

        if config.disc_pred_hid_size:
            wm_losses_dict["disc_pred_loss"] = disc_pred_loss.item()
            wm_losses_dict["disc_pred_loss_scaled"] = disc_pred_loss_scaled.item()

        # fwd_dict is used later for the imagination path and actor-critic loss
        wm_fwd_dict["s_list"] = s_list
         # For video generation;
        N = min(B, config.viz_n_videos)
        wm_fwd_dict["obs_rec_mean_list"] = obs_rec_mean_list[:N].detach()

        return wm_losses_dict, wm_fwd_dict

    def _train(self, batch_data_dict, prev_batch_data):
        config = self.config

        with RequiresGrad(self):
            wm_losses_dict, wm_fwd_dict = self.loss(batch_data_dict, prev_batch_data)
            wm_loss = wm_losses_dict["wm_loss"]
            self.model_optimizer.zero_grad()
            wm_loss.backward()
            if config.model_grad_clip:
                torch.nn.utils.clip_grad_norm_(self.parameters(), config.model_grad_clip)
            self.model_optimizer.step()
        
        self.n_updates += 1 # Tracks the number of grad. step on the model
        
        # TODO: might want to pass the data for next batch provessing 
        # a bit more "obsviously", instead of just relying on fwd_dict
        return wm_losses_dict, wm_fwd_dict

    @torch.no_grad()
    def sample_action(self, obs, actor, prev_data, exploration_fn, mode="train"):
        config = self.config
        wm = self # TODO: directly use self.model_component
        B = obs.shape[0]

        obs_feat = wm.encoder(obs)

        if prev_data == None: # Assumes first step of an episode
            prev_data = {
                "s_deter": obs.new_zeros([B, config.rssm_deter_size]),
                "s_stoch": obs.new_zeros([B, self.state_stoch_feat_size]),
                "action": obs.new_zeros([B, self.num_actions]),
                "done": obs.new_zeros([B, 1])
            }
        
        # TODO: also need to add masking so that the s_deter and s_stoch
        # are reset in case a new episode has been started
        done = prev_data["done"]
        mask = 1. - done.float() # TODO: properly handle what comes from the environment
        s_deter = prev_data["s_deter"] * mask
        s_stoch = prev_data["s_stoch"] * mask
        action = prev_data["action"] * mask # TODO: maybe differenciate the first action to -1 vector instead of just 0. Namely, 0 is a valid action

        prev_state_action_embed = torch.cat([s_stoch, action], 1)
        prev_state_action_embed = wm.state_action_embed(prev_state_action_embed)
        s_deter = wm.state_belief(prev_state_action_embed, s_deter)

        # Sample y_t ~ q(y_t | h_t, x_t)
        s_stoch_dist_stats = wm.post_state(torch.cat([obs_feat, s_deter], 1))
        s_stoch = wm.get_dist(s_stoch_dist_stats).rsample()
        s_stoch = s_stoch.view(B, -1) # Mainly for discrete case, no effect otherwise

        # Sample the action
        if mode == "train":
            action = None # TODO implement after actor network has been fleshed out
        elif mode == "eval":
            action = None # TODO implement after actor network has been fleshed out
        else:
            raise NotImplementedError(f"Unsupported sampling regime for action: {mode}")

        return action.cpu().numpy(), {"s_deter": s_deter, "s_stoch": s_stoch, "action": action, "done": done}

class ActorCritic(nn.Module):
    def __init__(self, config, num_actions, wm_fn):
        super().__init__()
        self.config = config
        self.num_actions = num_actions
        self.wm_fn = wm_fn
        
        # Pre-compute the state_feat_size: |S| = |H| + |Y|
        self.state_feat_size = config.rssm_deter_size
        self.state_stoch_feat_size = config.rssm_stoch_size * config.rssm_discrete \
            if config.rssm_discrete else config.rssm_stoch_size
        self.state_feat_size += self.state_stoch_feat_size
        self.state_stoch_feat_size

        # Actor component
        N = config.actor_hid_size
        
        # TODO: make the nn declaration more compact ?
        self.actor = nn.Sequential(
            nn.Linear(self.state_feat_size, N), nn.ELU(),
            nn.Linear(N, N), nn.ELU(),
            nn.Linear(N, N), nn.ELU(),
            nn.Linear(N, N), nn.ELU(),
            nn.Linear(N, num_actions) # logits layer
        )

        # Value component
        N = config.value_hid_size
        self.value = nn.Sequential(
            nn.Linear(self.state_feat_size, N), nn.ELU(),
            nn.Linear(N, N), nn.ELU(),
            nn.Linear(N, N), nn.ELU(),
            nn.Linear(N, N), nn.ELU(),
            nn.Linear(N, 1)
        )
        if config.value_slow_target:
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
        self.register_buffer("slow_value_n_updates", torch.LongTensor([0]))

class Dreamer(nn.Module):
    def __init__(self, config, num_actions):
        super().__init__()
        self.config = config
        self.num_actons = num_actions
        
        self.wm = WorldModel(config=args, num_actions=num_actions)
        self.ac = ActorCritic(config=args, num_actions=num_actions, wm_fn=lambda: self.wm)

        # Tracking training stats
        self.register_buffer("step", torch.LongTensor([0])) # How many env. steps so far, for schedulers
        self.register_buffer("n_updates", torch.LongTensor([0])) # How many updates of the model

        # TODO: add schedules for actor entropy and actor imaginate gradient mixing

    def _train(self, batch_data_dict, prev_batch_data):
        # TODO: mention data precprocessing or move it here instead of in the buffer ?
        wm_losses_dict, wm_fwd_dict = self.wm._train(batch_data_dict, prev_batch_data)

        # Store intermediate results for the next batch
        prev_batch_data = {
            "s_deter": wm_fwd_dict["s_deter_list"][:, -1].detach(), # [B, |H|]
            "s_stoch": wm_fwd_dict["s_stoch_list"][:, -1].detach(), # [B, |Y|]
            "reset_mask": wm_fwd_dict["reset_mask"] # [B, 1]
        }
        return wm_losses_dict, None, wm_fwd_dict, prev_batch_data

    @torch.no_grad()
    def gen_train_rec_videos(self, batch_data_dict, fwd_dict):
        # Generates video of batch trajectories' ground truth, reconstruction and error
        config = self.config
        obs_rec_mean_list = fwd_dict["obs_rec_mean_list"] # [N, T, C, H, W]
        N = obs_rec_mean_list.shape[0]
        orig_obs_list = batch_data_dict["observations"][:N] + 0.5 # [N, T, C ,H, W] in range [0.0,1.0]
        rec_obs_list = (obs_rec_mean_list + 0.5).clamp(0, 1) # [N, T, C, H, W] in range [0.0,1.0]
        error = (rec_obs_list - orig_obs_list + 1) / 2.
        black_strip = orig_obs_list.new_zeros([N, *orig_obs_list.shape[1:-2], 3, orig_obs_list.shape[-1]]) # [N, T, C, 3, W]
        train_video = torch.cat([orig_obs_list, black_strip, rec_obs_list, black_strip, error], 3) # [N, T, C, H * 3 + 6, W]
        train_video = torch.cat([tnsr for tnsr in train_video], dim=3)[None] # [1, T, C, H * 3 + 6, W * N] # Vertical stack
        return train_video.cpu().numpy()

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

    # Instantiate buffer based dataset loader
    buffer = DreamerTBPTTDataset(args, device)
    # Env setup5
    envs = gym.vector.SyncVectorEnv([make_env(args.env_id, args.seed, i, args.capture_video,
        run_name, buffer, args) for i in range(args.num_envs)])
    assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported" # TODO: this needs not be the case ?
    
    # Instantiate the Dreamer agent
    dreamer = Dreamer(config=args, num_actions=envs.single_action_space.n).to(device)

    # TRY NOT TO MODIFY: start the game
    start_time = time.time()
    obs = envs.reset()
    prev_data = None
    prev_batch_data = None
    for global_step in range(0, args.total_timesteps+4, 4): # TODO: explicity add support for action repeat instead of hardcoded '4'
        # ALGO LOGIC: put action logic here
        if global_step <= args.buffer_prefill:
            action = envs.action_space.sample()
        else:
            # TODO: implement action sampling from the Dreamer agent itself
            action = envs.action_space.sample()

        # TRY NOT TO MODIFY: execute the game and log data.
        next_obs, rewards, dones, infos = envs.step(action)

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        for info in infos:
            if "episode" in info.keys():
                print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
                writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
                writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)
                break # TODO: average over finished envs more accurate ?
        
        # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
        obs = next_obs
        # TODO: have to add the real done data to the "prev_data" dictionary
        # so that the sampling is correct
        # TODO: add tracking of the number of env steps to the Dreamer agent
        # dreamer.step += args.num_envs * args.env_action_repeat

        # ALGO LOGIC: training.
        # TODO: add conditiona for when the agent can be trained, and some basic logging
        # TODO also need to add the scheduler for train_every.Does it really need to take the action repeat into account ?
        if global_step >= args.buffer_prefill and buffer.ready_to_sample and global_step % (args.train_every // 4) == 0:
            batch_data_dict = buffer.sample()
            wm_losses_dict, ac_losses_dict, wm_fwd_dict, prev_batch_data = \
                dreamer._train(batch_data_dict, prev_batch_data)

            # Testing video saving
            # TODO: ideally, the saving of training metrics, video etc.. should not be too frequent
            train_rec_video = dreamer.gen_train_rec_videos(batch_data_dict, wm_fwd_dict)
            writer.add_video("train_rec_video", train_rec_video, global_step, fps=16)
            [writer.add_scalar(f"wm/{k}", v, global_step) for k,v in wm_losses_dict.items()]

    envs.close()
    writer.close()
