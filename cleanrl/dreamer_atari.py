# TODO docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/dreamer/#dreamer_ataripy
import argparse
import os
import random
import time
from distutils.util import strtobool

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
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
    parser.add_argument("--buffer-size", type=int, default=1000000,
        help="the replay memory buffer size")
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

# Wrapper to convert pixel observaton from HWC to CHW by default
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
    def __init__(self, env, callbacks=None):
        super().__init__(env)
        self._callbacks = callbacks or ()
        # NOTE: call reset at least the first time the environment is used
        
    def step(self, action):
        observation, reward, done, info = super().step(action)
        # Cache the trajectory data
        self._episode_data["observations"].append(observation)
        self._episode_data["actions"].append(action)
        self._episode_data["rewards"].append(reward)
        self._episode_data["terminals"].append(done)
        if done:
            # TODO: async saving to disk
            print("Done caught")

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

def make_env(env_id, seed, idx, capture_video, run_name):
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

        env = DatasetCollectioNrapper(env)
        return env

    return thunk

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
    train_dir = os.path.join(f"runs/{run_name}/train_episodes")
    if not os.path.exists(train_dir):
        os.mkdir(train_dir)
    
    # env setup
    envs = gym.vector.SyncVectorEnv([make_env(args.env_id, args.seed, i, args.capture_video, run_name) for i in range(args.num_envs)])
    assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported" # TODO: this needs not be the case ?

    obs = envs.reset()
    done = False
    t = 0

    while not done and t < 100:
        obs, reward, done, info = envs.step(envs.action_space.sample())
        t += 1
    
    pass