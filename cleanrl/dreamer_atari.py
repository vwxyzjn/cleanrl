# TODO docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/dreamer/#dreamer_ataripy
import os
import io
import time
import uuid
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
    parser.add_argument("--total-timesteps", type=int, default=10_000_000,
        help="total timesteps of the experiments")
    parser.add_argument("--buffer-size", type=int, default=1_000_000,
        help="the replay memory buffer size")
    parser.add_argument("--buffer-prefill", type=int, default=50_000,
        help="the number of steps to prefill the buffer with ( without action repeat)")
    parser.add_argument("--gamma", type=float, default=0.99,
        help="the discount factor gamma")
    parser.add_argument("--batch-size", type=int, default=32,
        help="the batch size of sample from the reply memory")
    parser.add_argument("--batch-length", type=int, default=32,
        help="the sequence length of trajectories in the batch sampled from memory")
    ## TODO World model specific hyperparameters
    ## TODO Actor (policy) specific hyperparameters
    ## TODO Value specific hyperparameters
    args = parser.parse_args()
    # fmt: on
    return args

# Environment with custom Wrapper to collect datasets
def make_env(env_id, seed, idx, capture_video, run_name, savedir, train_eps_cache, buffer_size):
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
    class DatasetCollectioNrapper(gym.Wrapper):
        def __init__(self, env, savedir, train_eps_cache, buffer_size):
            super().__init__(env)
            self._episode_data = None
            self._savedir = savedir
            self._train_eps_cache = train_eps_cache
            self._buffer_size = buffer_size

        def step(self, action):
            observation, reward, done, info = super().step(action)
            # Cache the trajectory data
            self._episode_data["observations"].append(observation)
            self._episode_data["actions"].append(action)
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
                "actions": [-1], # TODO: distinguish the "first" action for continuous cases ?
                "rewards": [0.0],
                "terminals": [False], # done
            }
            return first_obs
        
        def save_episode(self):
            # Prerpocess the episode data into np arrays
            self._episode_data["observations"] = \
                np.array(self._episode_data["observations"], dtype=np.uint8)
            self._episode_data["actions"] = \
                np.array(self._episode_data["actions"], dtype=np.int64).reshape(-1, 1)
            self._episode_data["rewards"] = \
                np.array(self._episode_data["rewards"], dtype=np.float32).reshape(-1, 1)
            self._episode_data["terminals"] = \
                np.array(self._episode_data["terminals"], dtype=np.bool8).reshape(-1, 1)
            # TODO: add proper credit for inspiration of this code
            timestamp = datetime.datetime.now().strftime('%Y%m%dT%H%M%S')
            identifier = str(uuid.uuid4().hex)
            length = len(self._episode_data["rewards"])
            filename = f"{self._savedir}/{timestamp}-{identifier}-{length}.npz"
            with io.BytesIO() as f1:
                np.savez_compressed(f1, **self._episode_data)
                f1.seek(0)
                # TODO: is write to disk even necessary ?
                with open(filename, "wb") as f2:
                    f2.write(f1.read())
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
        env = MaxAndSkipEnv(env, skip=4)
        env = EpisodicLifeEnv(env)
        if "FIRE" in env.unwrapped.get_action_meanings():
            env = FireResetEnv(env)
        env = ClipRewardEnv(env)
        env = gym.wrappers.ResizeObservation(env, (64, 64))
        env = gym.wrappers.GrayScaleObservation(env, keep_dim=True)
        env = ImagePermuteWrapper(env) # HWC -> CHW
        # TODO: fix seeding, does not seem to be working
        env.seed(seed + idx)
        env.action_space.seed(seed + idx)
        env.observation_space.seed(seed + idx)

        env = DatasetCollectioNrapper(env, savedir, train_eps_cache, buffer_size)
        return env

    return thunk

# Buffer with support for Truncated Backpropagation Through Time (TBPTT)
def make_dataloader(train_eps_cache, batch_size, batch_length, seed, num_workers=1):
    class DreamerTBPTTIterableDataset(IterableDataset):
        def __init__(self, train_eps_cache, batch_length):
            self._batch_length = batch_length
            self._train_eps_cache = train_eps_cache
            # Persists the episode data to maintain continuity of trajectories
            # across consecutive batches
            self._episode_data = None
            self._episode_length = 0
            self._ep_current_idx = 0
            self._ep_filename = None # DEBUG
        
        def __iter__(self):
            T = self._batch_length
            while True:
                # Placeholder
                obs_list = np.zeros([T, 1, 64, 64]) # TODO: recover obs shape
                act_list = np.zeros([T, 1], dtype=np.int64)
                rew_list = np.zeros([T, 1], dtype=np.float32)
                ter_list = np.zeros([T, 1], dtype=np.bool8)
                ssf = 0 # Steps collected so far current batch trajectory
                while ssf < T:
                    if self._episode_data is None or self._episode_length == self._ep_current_idx:
                        idx = torch.randint(len(self._train_eps_cache.keys()), ())
                        self._episode_data = self._train_eps_cache[list(self._train_eps_cache.keys())[idx]]
                        self._ep_filename = list(self._train_eps_cache.keys())[idx] # DEBUG
                        self._episode_length = len(self._episode_data["terminals"])
                        self._ep_current_idx = 0
                    needed_steps = T - ssf # How many steps needed to fill the traj
                    edd_start = self._ep_current_idx # Where to start slicing from episode data
                    avail_steps = self._episode_length - edd_start # How many steps from the ep. data not used yet
                    edd_end = min(edd_start + needed_steps, edd_start + avail_steps)
                    subseq_len = edd_end - edd_start
                    b_end = ssf + subseq_len
                    # Fill up the batch data placeholders with steps from episode data
                    obs_list[ssf:b_end] = self._episode_data["observations"][edd_start:edd_end]
                    act_list[ssf:b_end] = self._episode_data["actions"][edd_start:edd_end]
                    rew_list[ssf:b_end] = self._episode_data["rewards"][edd_start:edd_end]
                    ter_list[ssf:b_end] = self._episode_data["terminals"][edd_start:edd_end]

                    ssf = b_end
                    self._ep_current_idx = edd_end
                
                yield {"observations": obs_list,"actions": act_list,
                    "rewards": rew_list, "terminals": ter_list}

    # TODO: make sure that even if using num_workers > 1, the trajs. are sequential
    # between batches.
    def worker_init_fn(worker_id):
        worker_seed = 133754134 + worker_id
        random.seed(worker_seed)
        np.random.seed(worker_seed)
    th_seed_gen = torch.Generator()
    th_seed_gen.manual_seed(133754134 + seed)

    return iter(
        DataLoader(
            DreamerTBPTTIterableDataset(train_eps_cache=train_eps_cache, batch_length=batch_length),
            batch_size=batch_size, num_workers=num_workers,
            worker_init_fn=worker_init_fn, generator=th_seed_gen
        )
    )

# Dreamer Agent
class WorldModel():
    def __init__(self):
        pass

class ActorCritic():
    def __init__(self):
        pass

class Dreamer():
    def __init__(self):
        pass

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

    # Replay buffer / dataset setup: create the folder to hold sampled episode trajectories
    train_eps_cache = {} # Shared buffer
    train_savedir = os.path.join(f"runs/{run_name}/train_episodes")
    if not os.path.exists(train_savedir):
        os.mkdir(train_savedir)
    
    # env setup
    # NOTE: args.buffer_size // 4 to acount for Atari's action_repeat of 4
    envs = gym.vector.SyncVectorEnv([make_env(args.env_id, args.seed, i, args.capture_video,
        run_name, train_savedir, train_eps_cache, args.buffer_size // 4) for i in range(args.num_envs)])
    assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported" # TODO: this needs not be the case ?

    # Instantiate buffer based dataset loader
    dataloader = make_dataloader(train_eps_cache, batch_size=args.batch_size,
        batch_length=args.batch_length, seed=args.seed)
    
    # TRY NOT TO MODIFY: start the game
    obs = envs.reset()
    for global_step in range(args.total_timesteps):
        # ALGO LOGIC: put action logic here
        # TODO: action spampling
        action = None

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

        # ALGO LOGIC: training.
        # TODO: add conditiona for when the agent can be trained, and some basic logging

    envs.close()
    writer.close()
