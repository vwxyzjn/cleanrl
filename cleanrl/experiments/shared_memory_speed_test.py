# https://github.com/facebookresearch/torchbeast/blob/master/torchbeast/core/environment.py

import numpy as np
from collections import deque
import gym
from gym import spaces
import cv2
cv2.ocl.setUseOpenCL(False)


class NoopResetEnv(gym.Wrapper):
    def __init__(self, env, noop_max=30):
        """Sample initial states by taking random number of no-ops on reset.
        No-op is assumed to be action 0.
        """
        gym.Wrapper.__init__(self, env)
        self.noop_max = noop_max
        self.override_num_noops = None
        self.noop_action = 0
        assert env.unwrapped.get_action_meanings()[0] == 'NOOP'

    def reset(self, **kwargs):
        """ Do no-op action for a number of steps in [1, noop_max]."""
        self.env.reset(**kwargs)
        if self.override_num_noops is not None:
            noops = self.override_num_noops
        else:
            noops = self.unwrapped.np_random.randint(1, self.noop_max + 1) #pylint: disable=E1101
        assert noops > 0
        obs = None
        for _ in range(noops):
            obs, _, done, _ = self.env.step(self.noop_action)
            if done:
                obs = self.env.reset(**kwargs)
        return obs

    def step(self, ac):
        return self.env.step(ac)

class FireResetEnv(gym.Wrapper):
    def __init__(self, env):
        """Take action on reset for environments that are fixed until firing."""
        gym.Wrapper.__init__(self, env)
        assert env.unwrapped.get_action_meanings()[1] == 'FIRE'
        assert len(env.unwrapped.get_action_meanings()) >= 3

    def reset(self, **kwargs):
        self.env.reset(**kwargs)
        obs, _, done, _ = self.env.step(1)
        if done:
            self.env.reset(**kwargs)
        obs, _, done, _ = self.env.step(2)
        if done:
            self.env.reset(**kwargs)
        return obs

    def step(self, ac):
        return self.env.step(ac)

class EpisodicLifeEnv(gym.Wrapper):
    def __init__(self, env):
        """Make end-of-life == end-of-episode, but only reset on true game over.
        Done by DeepMind for the DQN and co. since it helps value estimation.
        """
        gym.Wrapper.__init__(self, env)
        self.lives = 0
        self.was_real_done  = True

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.was_real_done = done
        # check current lives, make loss of life terminal,
        # then update lives to handle bonus lives
        lives = self.env.unwrapped.ale.lives()
        if lives < self.lives and lives > 0:
            # for Qbert sometimes we stay in lives == 0 condition for a few frames
            # so it's important to keep lives > 0, so that we only reset once
            # the environment advertises done.
            done = True
        self.lives = lives
        return obs, reward, done, info

    def reset(self, **kwargs):
        """Reset only when lives are exhausted.
        This way all states are still reachable even though lives are episodic,
        and the learner need not know about any of this behind-the-scenes.
        """
        if self.was_real_done:
            obs = self.env.reset(**kwargs)
        else:
            # no-op step to advance from terminal/lost life state
            obs, _, _, _ = self.env.step(0)
        self.lives = self.env.unwrapped.ale.lives()
        return obs

class MaxAndSkipEnv(gym.Wrapper):
    def __init__(self, env, skip=4):
        """Return only every `skip`-th frame"""
        gym.Wrapper.__init__(self, env)
        # most recent raw observations (for max pooling across time steps)
        self._obs_buffer = np.zeros((2,)+env.observation_space.shape, dtype=np.uint8)
        self._skip       = skip

    def step(self, action):
        """Repeat action, sum reward, and max over last observations."""
        total_reward = 0.0
        done = None
        for i in range(self._skip):
            obs, reward, done, info = self.env.step(action)
            if i == self._skip - 2: self._obs_buffer[0] = obs
            if i == self._skip - 1: self._obs_buffer[1] = obs
            total_reward += reward
            if done:
                break
        # Note that the observation on the done=True frame
        # doesn't matter
        max_frame = self._obs_buffer.max(axis=0)

        return max_frame, total_reward, done, info

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

class ClipRewardEnv(gym.RewardWrapper):
    def __init__(self, env):
        gym.RewardWrapper.__init__(self, env)

    def reward(self, reward):
        """Bin reward to {+1, 0, -1} by its sign."""
        return np.sign(reward)


class WarpFrame(gym.ObservationWrapper):
    def __init__(self, env, width=84, height=84, grayscale=True, dict_space_key=None):
        """
        Warp frames to 84x84 as done in the Nature paper and later work.
        If the environment uses dictionary observations, `dict_space_key` can be specified which indicates which
        observation should be warped.
        """
        super().__init__(env)
        self._width = width
        self._height = height
        self._grayscale = grayscale
        self._key = dict_space_key
        if self._grayscale:
            num_colors = 1
        else:
            num_colors = 3

        new_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=(self._height, self._width, num_colors),
            dtype=np.uint8,
        )
        if self._key is None:
            original_space = self.observation_space
            self.observation_space = new_space
        else:
            original_space = self.observation_space.spaces[self._key]
            self.observation_space.spaces[self._key] = new_space
        assert original_space.dtype == np.uint8 and len(original_space.shape) == 3

    def observation(self, obs):
        if self._key is None:
            frame = obs
        else:
            frame = obs[self._key]

        if self._grayscale:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        frame = cv2.resize(
            frame, (self._width, self._height), interpolation=cv2.INTER_AREA
        )
        if self._grayscale:
            frame = np.expand_dims(frame, -1)

        if self._key is None:
            obs = frame
        else:
            obs = obs.copy()
            obs[self._key] = frame
        return obs


class FrameStack(gym.Wrapper):
    def __init__(self, env, k):
        """Stack k last frames.
        Returns lazy array, which is much more memory efficient.
        See Also
        --------
        baselines.common.atari_wrappers.LazyFrames
        """
        gym.Wrapper.__init__(self, env)
        self.k = k
        self.frames = deque([], maxlen=k)
        shp = env.observation_space.shape
        self.observation_space = spaces.Box(low=0, high=255, shape=((shp[0] * k,)+shp[1:]), dtype=env.observation_space.dtype)

    def reset(self):
        ob = self.env.reset()
        for _ in range(self.k):
            self.frames.append(ob)
        return self._get_ob()

    def step(self, action):
        ob, reward, done, info = self.env.step(action)
        self.frames.append(ob)
        return self._get_ob(), reward, done, info

    def _get_ob(self):
        assert len(self.frames) == self.k
        return LazyFrames(list(self.frames))

class ScaledFloatFrame(gym.ObservationWrapper):
    def __init__(self, env):
        gym.ObservationWrapper.__init__(self, env)
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=env.observation_space.shape, dtype=np.float32)

    def observation(self, observation):
        # careful! This undoes the memory optimization, use
        # with smaller replay buffers only.
        return np.array(observation).astype(np.float32) / 255.0

class LazyFrames(object):
    def __init__(self, frames):
        """This object ensures that common frames between the observations are only stored once.
        It exists purely to optimize memory usage which can be huge for DQN's 1M frames replay
        buffers.
        This object should only be converted to numpy array before being passed to the model.
        You'd not believe how complex the previous solution was."""
        self._frames = frames
        self._out = None

    def _force(self):
        if self._out is None:
            self._out = np.concatenate(self._frames, axis=0)
            self._frames = None
        return self._out

    def __array__(self, dtype=None):
        out = self._force()
        if dtype is not None:
            out = out.astype(dtype)
        return out

    def __len__(self):
        return len(self._force())

    def __getitem__(self, i):
        return self._force()[i]

    def count(self):
        frames = self._force()
        return frames.shape[frames.ndim - 1]

    def frame(self, i):
        return self._force()[..., i]

def wrap_atari(env, max_episode_steps=None):
    assert 'NoFrameskip' in env.spec.id
    env = NoopResetEnv(env, noop_max=30)
    env = MaxAndSkipEnv(env, skip=4)

    assert max_episode_steps is None

    return env

class ImageToPyTorch(gym.ObservationWrapper):
    """
    Image shape to channels x weight x height
    """

    def __init__(self, env):
        super(ImageToPyTorch, self).__init__(env)
        old_shape = self.observation_space.shape
        self.observation_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=(old_shape[-1], old_shape[0], old_shape[1]),
            dtype=np.uint8,
        )

    def observation(self, observation):
        return np.transpose(observation, axes=(2, 0, 1))

def wrap_deepmind(env, episode_life=True, clip_rewards=True, frame_stack=False, scale=False):
    """Configure environment for DeepMind-style Atari.
    """
    if episode_life:
        env = EpisodicLifeEnv(env)
    if 'FIRE' in env.unwrapped.get_action_meanings():
        env = FireResetEnv(env)
    env = WarpFrame(env)
    if scale:
        env = ScaledFloatFrame(env)
    if clip_rewards:
        env = ClipRewardEnv(env)
    env = ImageToPyTorch(env)
    if frame_stack:
        env = FrameStack(env, 4)
    return env



# Reference: https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter

import argparse
from distutils.util import strtobool
import collections
import numpy as np
import gym
from gym.wrappers import TimeLimit, Monitor
from gym.spaces import Discrete, Box, MultiBinary, MultiDiscrete, Space
import time
import random
import os

# import threading
# os.environ["OMP_NUM_THREADS"] = "1"  # Necessary for multithreading.
from torch import multiprocessing as mp
from multiprocessing.managers import SyncManager, SharedMemoryManager
from multiprocessing.shared_memory import SharedMemory
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecEnvWrapper
from faster_fifo import Queue as MpQueue
from torch.multiprocessing import Process as TorchProcess
torch.multiprocessing.set_sharing_strategy('file_system')

class SharedNDArray(np.ndarray):
    def set_shm(self, shm):
        self.shm = shm

    def close(self):
        self.shm.close()

    def unlink(self):
        self.shm.unlink()


def share_memory(arr):
    shm = SharedMemory(create=True, size=arr.nbytes)
    shm_arr = SharedNDArray(arr.shape, dtype=arr.dtype, buffer=shm.buf)
    shm_arr[:] = arr[:]
    shm_arr.set_shm(shm)
    return shm_arr

def share_memory(arr):
    t = torch.tensor(arr)
    t.share_memory_()
    return t


class Actor:
    def __init__(self, inputs):
        self.inputs = inputs
        self.process = TorchProcess(target=self.act, daemon=True)
        self.process.start()

    def act(self):
        # print(torch.ones((12,23,42)).sum())
        torch.multiprocessing.set_sharing_strategy('file_system')
        args, experiment_name, i, lock, stats_queue, device, \
            obs, obs_sm, logprobs, rewards, dones, values = self.inputs
        obs = to_numpy(obs_sm, 5)
        envs = []
        # o = np.ones((210, 160, 3))
        # print(o.sum())
        # print(torch.ones((84,160,3)).sum())
        # raise
        
        def make_env(gym_id, seed, idx):
            env = gym.make(gym_id)
            env = wrap_atari(env)
            env = gym.wrappers.RecordEpisodeStatistics(env)
            env = wrap_deepmind(
                env,
                clip_rewards=True,
                frame_stack=True,
                scale=False,
            )
            env.seed(seed)
            env.action_space.seed(seed)
            env.observation_space.seed(seed)
            return env
        envs = [make_env(args.gym_id, args.seed+i, i) for i in range(args.num_envs)]
        envs = np.array(envs, dtype=object)
    
    
        for env_idx, env in enumerate(envs):
            env.reset()
            # print('Process %d finished resetting %d/%d envs', env_idx + 1, len(envs))
        last_report = last_report_frames = total_env_frames = 0
        while True:
            for env_idx, env in enumerate(envs):
                # os = []
                for step in range(args.num_steps):
    
                    action = env.action_space.sample()
                    o, r, d, info = env.step(action)
                    if d:
                        o = env.reset()
                    # os += [o]
                    # print(obs[i,env_idx,0,0,step].shape, torch.from_numpy(o).shape)
                    # raise
                    # o = np.ones((210, 160, 3))
                    # obs[i,env_idx,0,0,step] = o
                    # print("before", id(obs[i,env_idx,0,0,step]), i, env_idx, step)
                    # t = 
                    obs[i,env_idx,0,0,step].copy_(torch.from_numpy(np.array(o)))
                    # print(torch.from_numpy(np.array(o)).sum(), obs[i,env_idx,0,0,step].sum(), i, env_idx, step)
                    # print(obs[i,env_idx,0,0,step].sum(), i, env_idx, step)
                    # print(id(obs[i,env_idx,0,0,step]), i, env_idx, step)
                    # print(id(obs_sm))
                    # # print(obs_sm.sum())
                    # raise
                    # rewards[i,env_idx,0,0,step] = r
                    # dones[i,env_idx,0,0,step] = d
    
                    num_frames = 1
                    total_env_frames += num_frames
        
                    if 'episode' in info.keys():
                        stats_queue.put(info['episode']['l'])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='DQN agent')
    # Common arguments
    parser.add_argument('--exp-name', type=str, default=os.path.basename(__file__).rstrip(".py"),
                        help='the name of this experiment')
    parser.add_argument('--gym-id', type=str, default="BreakoutNoFrameskip-v4",
                        help='the id of the gym environment')
    parser.add_argument('--learning-rate', type=float, default=1e-4,
                        help='the learning rate of the optimizer')
    parser.add_argument('--seed', type=int, default=2,
                        help='seed of the experiment')
    parser.add_argument('--total-timesteps', type=int, default=100000000,
                        help='total timesteps of the experiments')
    parser.add_argument('--torch-deterministic', type=lambda x:bool(strtobool(x)), default=True, nargs='?', const=True,
                        help='if toggled, `torch.backends.cudnn.deterministic=False`')
    parser.add_argument('--cuda', type=lambda x:bool(strtobool(x)), default=True, nargs='?', const=True,
                        help='if toggled, cuda will not be enabled by default')
    parser.add_argument('--prod-mode', type=lambda x:bool(strtobool(x)), default=False, nargs='?', const=True,
                        help='run the script in production mode and use wandb to log outputs')
    parser.add_argument('--capture-video', type=lambda x:bool(strtobool(x)), default=False, nargs='?', const=True,
                        help='weather to capture videos of the agent performances (check out `videos` folder)')
    parser.add_argument('--wandb-project-name', type=str, default="cleanRL",
                        help="the wandb's project name")
    parser.add_argument('--wandb-entity', type=str, default=None,
                        help="the entity (team) of wandb's project")
    
    # Algorithm specific arguments
    parser.add_argument('--num-rollout-workers', type=int, default=mp.cpu_count(),
                         help='the number of rollout workers')
    parser.add_argument('--num-policy-workers', type=int, default=1,
                         help='the number of policy workers')
    parser.add_argument('--num-envs', type=int, default=20,
                         help='the number of envs per rollout worker')
    parser.add_argument('--num-traj-buffers', type=int, default=1,
                         help='the number of trajectory buffers per rollout worker')
    parser.add_argument('--num-steps', type=int, default=32,
                        help='the number of steps per game environment')
    parser.add_argument('--gamma', type=float, default=0.99,
                        help='the discount factor gamma')
    parser.add_argument('--gae-lambda', type=float, default=0.95,
                        help='the lambda for the general advantage estimation')
    parser.add_argument('--ent-coef', type=float, default=0.01,
                        help="coefficient of the entropy")
    parser.add_argument('--vf-coef', type=float, default=0.5,
                        help="coefficient of the value function")
    parser.add_argument('--max-grad-norm', type=float, default=0.5,
                        help='the maximum norm for the gradient clipping')
    parser.add_argument('--clip-coef', type=float, default=0.1,
                        help="the surrogate clipping coefficient")
    parser.add_argument('--update-epochs', type=int, default=4,
                         help="the K epochs to update the policy")
    parser.add_argument('--kle-stop', type=lambda x:bool(strtobool(x)), default=False, nargs='?', const=True,
                         help='If toggled, the policy updates will be early stopped w.r.t target-kl')
    parser.add_argument('--kle-rollback', type=lambda x:bool(strtobool(x)), default=False, nargs='?', const=True,
                         help='If toggled, the policy updates will roll back to previous policy if KL exceeds target-kl')
    parser.add_argument('--target-kl', type=float, default=0.03,
                         help='the target-kl variable that is referred by --kl')
    parser.add_argument('--gae', type=lambda x:bool(strtobool(x)), default=True, nargs='?', const=True,
                         help='Use GAE for advantage computation')
    parser.add_argument('--norm-adv', type=lambda x:bool(strtobool(x)), default=True, nargs='?', const=True,
                          help="Toggles advantages normalization")
    parser.add_argument('--anneal-lr', type=lambda x:bool(strtobool(x)), default=True, nargs='?', const=True,
                          help="Toggle learning rate annealing for policy and value networks")
    parser.add_argument('--clip-vloss', type=lambda x:bool(strtobool(x)), default=True, nargs='?', const=True,
                          help='Toggles wheter or not to use a clipped loss for the value function, as per the paper.')

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
        wandb.init(project=args.wandb_project_name, entity=args.wandb_entity, sync_tensorboard=True, config=vars(args), name=experiment_name, monitor_gym=True, save_code=True)
        writer = SummaryWriter(f"/tmp/{experiment_name}")
    
    # TRY NOT TO MODIFY: seeding
    device = torch.device('cuda' if torch.cuda.is_available() and args.cuda else 'cpu')
    
    def make_env(gym_id, seed, idx):
        env = gym.make(gym_id)
        env = wrap_atari(env)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = wrap_deepmind(
            env,
            clip_rewards=True,
            frame_stack=True,
            scale=False,
        )
        env.seed(seed)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        return env
    
    env = make_env(args.gym_id, args.seed, 0)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic
    assert isinstance(env.action_space, Discrete), "only discrete action space is supported"

    # m = SyncManager()
    # m.start()
    lock = mp.Lock()
    dimensions = (
        args.num_rollout_workers,
        args.num_envs,
        args.num_policy_workers,
        args.num_traj_buffers,
        args.num_steps,
    )
    
    # smm = SharedMemoryManager()
    # smm.start()
    # rewards = share_memory(np.zeros(dimensions + (args.num_envs,)), smm)
    # def share_memory(a, smm):
    #     """simply put array `a` into shared memory
    #     https://docs.python.org/3/library/multiprocessing.shared_memory.html"""
        # shm = smm.SharedMemory(size=a.nbytes)
        # # Now create a NumPy array backed by shared memory
        # b = np.ndarray(a.shape, dtype=a.dtype, buffer=shm.buf)
        # b[:] = a[:]  # Copy the original data into shared memory
        # del a
        # return b
    # with SharedMemoryManager() as smm:
    # raise

    def to_numpy(t, num_dimensions):
        arr_shape = t.shape[:num_dimensions]
        arr = np.ndarray(arr_shape, dtype=object)
        to_numpy_func(t, arr)
        return arr
    
    
    def to_numpy_func(t, arr):
        if len(arr.shape) == 1:
            for i in range(t.shape[0]):
                arr[i] = t[i]
        else:
            for i in range(t.shape[0]):
                to_numpy_func(t[i], arr[i])
            
    
    obs_sm = share_memory(np.zeros(dimensions + env.observation_space.shape))
    obs = to_numpy(obs_sm, 5)
    actions = share_memory(np.zeros(dimensions + env.action_space.shape))
    logprobs = share_memory(np.zeros(dimensions))
    rewards = share_memory(np.zeros(dimensions))
    dones = share_memory(np.zeros(dimensions))
    values = share_memory(np.zeros(dimensions))
    
    # traj_availables = share_memory(np.zeros(dimensions))
    # raise
    
    # raise
    # global_step = torch.tensor(0)
    # global_step.share_memory_()
    actor_processes = []
    data_processor_processes = []
    # ctx = mp.get_context("forkserver")
    stats_queue = MpQueue()
    # stats_queue = mp.Queue(1000)
    rollouts_queue = mp.Queue(1000)
    data_process_queue = mp.Queue(1000)
    data_process_back_queues = []


    
    
    

    for i in range(args.num_rollout_workers):
        inputs = [args, experiment_name, i, lock, stats_queue, device,
                   obs, obs_sm, logprobs, rewards, dones, values]
        
        a = Actor(inputs)
        # actor = mp.Process(
        #     target=a.act,
        #     daemon=True
        #     # args=[inputs],
        # )
        # actor.start()
        # actor_processes.append(actor)
    
    # learner = ctx.Process(
    #     target=learn,
    #     args=(
    #         args, rb, global_step,
    #         data_process_queue,
    #         data_process_back_queues, stats_queue, lock, learn_target_network, target_network, learn_q_network, q_network, optimizer, device
    #     ),
    # )
    # learner.start()

    import timeit
    timer = timeit.default_timer
    existing_video_files = []
    global_step = 0
    global_step_increment = 0
    start_time = time.time()
    update_step = 0
    try:
        while global_step < args.total_timesteps:
            update_step += 1
            # start_global_step = global_step
            try:
                ls = stats_queue.get_many(timeout=1)
                for l in ls:
                    global_step_increment += l
            except:
                continue

            # writer.add_scalar("charts/episode_reward", r, global_step)
            # writer.add_scalar("charts/stats_queue_size", stats_queue.qsize(), global_step)
            # writer.add_scalar("charts/rollouts_queue_size", rollouts_queue.qsize(), global_step)
            # writer.add_scalar("charts/data_process_queue_size", data_process_queue.qsize(), global_step)
            if update_step % 10 == 0:
                # print(f"global_step={global_step}, episode_reward={r}")
                print(f"global_step={global_step}")
                global_step += global_step_increment
                writer.add_scalar("charts/fps", global_step_increment / (time.time() - start_time), global_step)
                print("FPS: ", global_step_increment / (time.time() - start_time))
                global_step_increment = 0
                start_time = time.time()

            # else:
            #     # print(m[0], m[1], global_step)
            #     # writer.add_scalar(m[0], m[1], global_step)
            #     pass
            # if args.capture_video and args.prod_mode:
            #     video_files = glob.glob(f'videos/{experiment_name}/*.mp4')
            #     for video_file in video_files:
            #         if video_file not in existing_video_files:
            #             existing_video_files += [video_file]
            #             print(video_file)
            #             if len(existing_video_files) > 1:
            #                 wandb.log({"video.0": wandb.Video(existing_video_files[-2])})
    except KeyboardInterrupt:
        pass
    finally:
        # learner.terminate()
        # learner.join(timeout=1)
        for actor in actor_processes:
            actor.terminate()
            actor.join(timeout=1)
        for data_processor in data_processor_processes:
            data_processor.terminate()
            data_processor.join(timeout=1)
        if args.capture_video and args.prod_mode:
            wandb.log({"video.0": wandb.Video(existing_video_files[-1])})
    # env.close()
    writer.close()
