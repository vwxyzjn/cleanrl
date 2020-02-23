import cv2
import gym
import gym.spaces
import numpy as np
import collections


class FireResetEnv(gym.Wrapper):
    def __init__(self, env=None):
        """For environments where the user need to press FIRE for the game to start."""
        super(FireResetEnv, self).__init__(env)
        assert env.unwrapped.get_action_meanings()[1] == 'FIRE'
        assert len(env.unwrapped.get_action_meanings()) >= 3

    def step(self, action):
        return self.env.step(action)

    def reset(self):
        self.env.reset()
        obs, _, done, _ = self.env.step(1)
        if done:
            self.env.reset()
        obs, _, done, _ = self.env.step(2)
        if done:
            self.env.reset()
        return obs


class MaxAndSkipEnv(gym.Wrapper):
    def __init__(self, env=None, skip=4):
        """Return only every `skip`-th frame"""
        super(MaxAndSkipEnv, self).__init__(env)
        # most recent raw observations (for max pooling across time steps)
        self._obs_buffer = collections.deque(maxlen=2)
        self._skip = skip

    def step(self, action):
        total_reward = 0.0
        done = None
        for _ in range(self._skip):
            obs, reward, done, info = self.env.step(action)
            self._obs_buffer.append(obs)
            total_reward += reward
            if done:
                break
        max_frame = np.max(np.stack(self._obs_buffer), axis=0)
        return max_frame, total_reward, done, info

    def reset(self):
        """Clear past frame buffer and init. to first obs. from inner env."""
        self._obs_buffer.clear()
        obs = self.env.reset()
        self._obs_buffer.append(obs)
        return obs


class ProcessFrame84(gym.ObservationWrapper):
    def __init__(self, env=None):
        super(ProcessFrame84, self).__init__(env)
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(84, 84, 1), dtype=np.uint8)

    def observation(self, obs):
        return ProcessFrame84.process(obs)

    @staticmethod
    def process(frame):
        if frame.size == 210 * 160 * 3:
            img = np.reshape(frame, [210, 160, 3]).astype(np.float32)
        elif frame.size == 250 * 160 * 3:
            img = np.reshape(frame, [250, 160, 3]).astype(np.float32)
        else:
            assert False, "Unknown resolution."
        img = img[:, :, 0] * 0.299 + img[:, :, 1] * 0.587 + img[:, :, 2] * 0.114
        resized_screen = cv2.resize(img, (84, 110), interpolation=cv2.INTER_AREA)
        x_t = resized_screen[18:102, :]
        x_t = np.reshape(x_t, [84, 84, 1])
        return x_t.astype(np.uint8)


class ImageToPyTorch(gym.ObservationWrapper):
    def __init__(self, env):
        super(ImageToPyTorch, self).__init__(env)
        old_shape = self.observation_space.shape
        self.observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=(old_shape[-1], old_shape[0], old_shape[1]),
                                                dtype=np.float32)

    def observation(self, observation):
        return np.moveaxis(observation, 2, 0)


class ScaledFloatFrame(gym.ObservationWrapper):
    def observation(self, obs):
        return np.array(obs).astype(np.float32) / 255.0


class BufferWrapper(gym.ObservationWrapper):
    def __init__(self, env, n_steps, dtype=np.float32):
        super(BufferWrapper, self).__init__(env)
        self.dtype = dtype
        old_space = env.observation_space
        self.observation_space = gym.spaces.Box(old_space.low.repeat(n_steps, axis=0),
                                                old_space.high.repeat(n_steps, axis=0), dtype=dtype)

    def reset(self):
        self.buffer = np.zeros_like(self.observation_space.low, dtype=self.dtype)
        return self.observation(self.env.reset())

    def observation(self, observation):
        self.buffer[:-1] = self.buffer[1:]
        self.buffer[-1] = observation
        return self.buffer


def make_env(env_name):
    env = gym.make(env_name)
    env = MaxAndSkipEnv(env)
    env = FireResetEnv(env)
    env = ProcessFrame84(env)
    env = ImageToPyTorch(env)
    env = BufferWrapper(env, 4)
    return ScaledFloatFrame(env)

# Reference: https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from cleanrl.common import preprocess_obs_space, preprocess_ac_space
import argparse
import collections
import numpy as np
import gym
from gym.wrappers import TimeLimit, Monitor, AtariPreprocessing
from gym.spaces import Discrete, Box, MultiBinary, MultiDiscrete, Space
import time
import random
import os

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='A2C agent')
    # Common arguments
    parser.add_argument('--exp-name', type=str, default=os.path.basename(__file__).rstrip(".py"),
                       help='the name of this experiment')
    parser.add_argument('--gym-id', type=str, default="PongNoFrameskip-v4",
                       help='the id of the gym environment')
    parser.add_argument('--learning-rate', type=float, default=1e-4,
                       help='the learning rate of the optimizer')
    parser.add_argument('--seed', type=int, default=2,
                       help='seed of the experiment')
    parser.add_argument('--episode-length', type=int, default=0,
                       help='the maximum length of each episode')
    parser.add_argument('--total-timesteps', type=int, default=10000000,
                       help='total timesteps of the experiments')
    parser.add_argument('--no-torch-deterministic', action='store_false', dest="torch_deterministic", default=True,
                       help='if toggled, `torch.backends.cudnn.deterministic=False`')
    parser.add_argument('--no-cuda', action='store_false', dest="cuda", default=True,
                       help='if toggled, cuda will not be enabled by default')
    parser.add_argument('--prod-mode', action='store_true', default=False,
                       help='run the script in production mode and use wandb to log outputs')
    parser.add_argument('--capture-video', action='store_true', default=False,
                       help='weather to capture videos of the agent performances (check out `videos` folder)')
    parser.add_argument('--wandb-project-name', type=str, default="cleanRL",
                       help="the wandb's project name")
    parser.add_argument('--wandb-entity', type=str, default=None,
                       help="the entity (team) of wandb's project")
    
    # Algorithm specific arguments
    parser.add_argument('--buffer-size', type=int, default=10000,
                        help='the replay memory buffer size')
    parser.add_argument('--gamma', type=float, default=0.99,
                       help='the discount factor gamma')
    parser.add_argument('--target-network-frequency', type=int, default=1000,
                       help="the timesteps it takes to update the target network")
    parser.add_argument('--max-grad-norm', type=float, default=0.5,
                       help='the maximum norm for the gradient clipping')
    parser.add_argument('--batch-size', type=int, default=32,
                       help="the batch size of sample from the reply memory")
    parser.add_argument('--start-e', type=float, default=0.1,
                       help="the starting epsilon for exploration")
    parser.add_argument('--end-e', type=float, default=0.01,
                       help="the ending epsilon for exploration")
    parser.add_argument('--learning-starts', type=int, default=10000,
                       help="timestep to start learning")
    parser.add_argument('--train-frequency', type=int, default=4,
                       help="the frequency of training")
    parser.add_argument('--exploration-fraction', type=float, default=0.10,
                       help="the fraction of `total-timesteps` it takes from start-e to go end-e")
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
device = torch.device('cuda' if torch.cuda.is_available() and args.cuda else 'cpu')
env = make_env(args.gym_id)
#args.episode_length = env._max_episode_steps
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.backends.cudnn.deterministic = args.torch_deterministic
input_shape, preprocess_obs_fn = preprocess_obs_space(env.observation_space, device)
output_shape = preprocess_ac_space(env.action_space)

# ALGO LOGIC: initialize agent here:
EPSILON_DECAY_LAST_FRAME = 10**5
EPSILON_START = 1.0
EPSILON_FINAL = 0.02
class ReplayBuffer():
    def __init__(self, buffer_limit):
        self.buffer = collections.deque(maxlen=buffer_limit)
    
    def put(self, transition):
        self.buffer.append(transition)
    
    def sample(self, n):
        mini_batch = random.sample(self.buffer, n)
        s_lst, a_lst, r_lst, s_prime_lst, done_mask_lst = [], [], [], [], []
        
        for transition in mini_batch:
            s, a, r, s_prime, done_mask = transition
            s_lst.append(s)
            a_lst.append(a)
            r_lst.append(r)
            s_prime_lst.append(s_prime)
            done_mask_lst.append(done_mask)

        return np.array(s_lst), np.array(a_lst), \
               np.array(r_lst), np.array(s_prime_lst), \
               np.array(done_mask_lst)


class DQN(nn.Module):
    def __init__(self, input_shape, n_actions):
        super(DQN, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )
        conv_out_size = self._get_conv_out(input_shape)
        self.fc = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions)
        )

    def _get_conv_out(self, shape):
        o = self.conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))

    def forward(self, x):
        conv_out = self.conv(x).view(x.size()[0], -1)
        return self.fc(conv_out)
class Agent:
    def __init__(self, env, exp_buffer):
        self.env = env
        self.exp_buffer = exp_buffer
        self._reset()

    def _reset(self):
        self.state = env.reset()
        self.total_reward = 0.0

def calc_loss(batch, net, tgt_net, device="cpu"):
    states, actions, rewards, dones, next_states = batch

    states_v = torch.tensor(states).to(device)
    next_states_v = torch.tensor(next_states).to(device)
    actions_v = torch.tensor(actions).to(device)
    rewards_v = torch.tensor(rewards).to(device)
    done_mask = torch.BoolTensor(dones).to(device)

    state_action_values = net(states_v).gather(1, actions_v.unsqueeze(-1)).squeeze(-1)
    next_state_values = tgt_net(next_states_v).max(1)[0]
    next_state_values[done_mask] = 0.0
    next_state_values = next_state_values.detach()

    expected_state_action_values = next_state_values * args.gamma + rewards_v
    return nn.MSELoss()(state_action_values, expected_state_action_values)

net = DQN(env.observation_space.shape, env.action_space.n).to(device)
tgt_net = DQN(env.observation_space.shape, env.action_space.n).to(device)
print(net)

exp_buffer = ReplayBuffer(args.buffer_size)
epsilon = EPSILON_START

optimizer = optim.Adam(net.parameters(), lr=args.learning_rate)
total_rewards = []
ts_frame = 0
ts = time.time()
best_mean_reward = None
    

EPSILON_DECAY_LAST_FRAME = 10**5
EPSILON_START = 1.0
EPSILON_FINAL = 0.02

# q_network = QNetwork().to(device)
# target_network = QNetwork().to(device)
# target_network.load_state_dict(q_network.state_dict())
# optimizer = optim.Adam(q_network.parameters(), lr=args.learning_rate)
loss_fn = nn.MSELoss()
# TRY NOT TO MODIFY: start the game
global_step = 0
state = env.reset()
total_reward = 0
while global_step < args.total_timesteps:
    global_step += 1
    epsilon = max(EPSILON_FINAL, EPSILON_START - global_step / EPSILON_DECAY_LAST_FRAME)

    done_reward = None
    if np.random.random() < epsilon:
        action = env.action_space.sample()
    else:
        state_a = np.array([state], copy=False)
        state_v = torch.tensor(state_a).to(device)
        q_vals_v = net(state_v)
        _, act_v = torch.max(q_vals_v, dim=1)
        action = int(act_v.item())

    # do step in the environment
    new_state, reward, is_done, _ = env.step(action)
    total_reward += reward
    new_state = new_state
    exp_buffer.append((state, action, reward, is_done, new_state))
    state = new_state
    if is_done:
        done_reward = total_reward
        state = env.reset()
        total_reward = 0.0

    
    if done_reward is not None:
        total_rewards.append(done_reward)
        speed = (global_step - ts_frame) / (time.time() - ts)
        ts_frame = global_step
        ts = time.time()
        mean_reward = np.mean(total_rewards[-100:])
        print("%d: done %d games, mean done_reward %.3f, eps %.2f, speed %.2f f/s" % (
            global_step, len(total_rewards), mean_reward, epsilon,
            speed
        ))
        writer.add_scalar("epsilon", epsilon, global_step)
        writer.add_scalar("speed", speed, global_step)
        writer.add_scalar("reward_100", mean_reward, global_step)
        writer.add_scalar("charts/episode_reward", done_reward, global_step)
        if best_mean_reward is None or best_mean_reward < mean_reward:
            torch.save(net.state_dict(), args.gym_id + "-best.dat")
            if best_mean_reward is not None:
                print("Best mean done_reward updated %.3f -> %.3f, model saved" % (best_mean_reward, mean_reward))
            best_mean_reward = mean_reward

    if len(exp_buffer) < args.learning_starts:
        continue

    if global_step % args.target_network_frequency == 0:
        tgt_net.load_state_dict(net.state_dict())

    optimizer.zero_grad()
    batch = exp_buffer.sample(args.batch_size)
    loss_t = calc_loss(batch, net, tgt_net, device=device)
    loss_t.backward()
    optimizer.step()
    # next_obs = np.array(env.reset())
    # actions = np.empty((args.episode_length,), dtype=object)
    # rewards, dones = np.zeros((2, args.episode_length))
    # obs = np.empty((args.episode_length,) + env.observation_space.shape)
    
    # # ALGO LOGIC: put other storage logic here
    # values = torch.zeros((args.episode_length), device=device)
    
    # # TRY NOT TO MODIFY: prepare the execution of the game.
    # for step in range(args.episode_length):
    #     global_step += 1
    #     obs[step] = next_obs.copy()
        
    #     # ALGO LOGIC: put action logic here
    #     epsilon = epsilon = max(EPSILON_FINAL, EPSILON_START - global_step / EPSILON_DECAY_LAST_FRAME)
    #     beta = linear_schedule(0.4, 1.0, args.total_timesteps, global_step)
    #     # ALGO LOGIC: `env.action_space` specific logic
    #     if random.random() < epsilon:
    #         actions[step] = env.action_space.sample()
    #     else:
    #         logits = target_network.forward(obs[step:step+1])
    #         if isinstance(env.action_space, Discrete):
    #             action = torch.argmax(logits, dim=1)
    #             actions[step] = action.tolist()[0]
        
    #     # TRY NOT TO MODIFY: execute the game and log data.
    #     next_obs, rewards[step], dones[step], _ = env.step(actions[step])
    #     rb.put((obs[step], actions[step], rewards[step], next_obs, dones[step]))
    #     next_obs = np.array(next_obs)
        
    #     # ALGO LOGIC: training.
    #     if global_step > args.learning_starts and global_step % args.train_frequency == 0:
    #         s_obs, s_actions, s_rewards, s_next_obses, s_dones = rb.sample(args.batch_size)
    #         target_max = torch.max(target_network.forward(s_next_obses), dim=1)[0]
    #         td_target = torch.Tensor(s_rewards).to(device) + args.gamma * target_max * (1 - torch.Tensor(s_dones).to(device))
    #         old_val = q_network.forward(s_obs).gather(1, torch.LongTensor(s_actions).view(-1,1).to(device)).squeeze()
    #         loss = loss_fn(td_target, old_val)

    #         # optimize the midel
    #         optimizer.zero_grad()
    #         loss.backward()
    #         writer.add_scalar("losses/td_loss", loss, global_step)
    #         nn.utils.clip_grad_norm_(list(q_network.parameters()), args.max_grad_norm)
    #         optimizer.step()
            
    #         # update the target network
    #         if global_step % args.target_network_frequency == 0:
    #             target_network.load_state_dict(q_network.state_dict())
        
    #     if dones[step]:
    #         break
    
    # # TRY NOT TO MODIFY: record rewards for plotting purposes
    # writer.add_scalar("charts/episode_reward", rewards.sum(), global_step)
    # writer.add_scalar("charts/epsilon", epsilon, global_step)
env.close()
writer.close()