# C51 Paper: https://arxiv.org/pdf/1707.06887.pdf
# Paper Implementation: https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/c51_atari.py
# Vectorized : https://github.com/vwxyzjn/cleanrl/blob/experimental-vdqn/cleanrl/atari/vdqn_atari.py

import argparse
import os
import random
import time
from distutils.util import strtobool
import gym

import numpy as np
import torch
import torch.nn as nn
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
    # Common arguments
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp-name', type=str, default=os.path.basename(__file__).rstrip(".py"),
        help='the name of this experiment')
    parser.add_argument('--gym-id', type=str, default="BreakoutNoFrameskip-v4",
        help='the id of the gym environment')
    parser.add_argument('--learning-rate', type=float, default=1e-4,
        help='the learning rate of the optimizer')
    parser.add_argument('--seed', type=int, default=2,
        help='seed of the experiment')
    parser.add_argument('--total-timesteps', type=int, default=10000000,
        help='total timesteps of the experiments')
    parser.add_argument('--torch-deterministic', type=lambda x:bool(strtobool(x)), default=True, nargs='?', const=True,
        help='if toggled, `torch.backends.cudnn.deterministic=False`')
    parser.add_argument('--cuda', type=lambda x:bool(strtobool(x)), default=True, nargs='?', const=True,
        help='if toggled, cuda will be enabled by default')
    parser.add_argument('--track', type=lambda x:bool(strtobool(x)), default=False, nargs='?', const=True,
        help='if toggled, this experiment will be tracked with Weights and Biases')
    parser.add_argument('--wandb-project-name', type=str, default="cleanRL",
        help="the wandb's project name")
    parser.add_argument('--wandb-entity', type=str, default=None,
        help="the entity (team) of wandb's project")
    parser.add_argument('--capture-video', type=lambda x:bool(strtobool(x)), default=False, nargs='?', const=True,
        help='weather to capture videos of the agent performances (check out `videos` folder)')

    # Algorithm specific arguments
    parser.add_argument('--num-envs', type=int, default=8,
        help='the number of parallel game environments')
    parser.add_argument('--num-steps', type=int, default=128,
        help='the number of steps to run in each environment per policy rollout')
    parser.add_argument('--num-minibatches', type=int, default=4,
        help='the number of mini-batches')
    parser.add_argument('--update-epochs', type=int, default=16,
        help="the K epochs to update the policy")
    parser.add_argument('--gamma', type=float, default=0.99,
        help='the discount factor gamma')
    parser.add_argument('--n-atoms', type=int, default=51,
        help="the number of atoms")
    parser.add_argument('--v-min', type=float, default=-10,
        help="the number of atoms")
    parser.add_argument('--v-max', type=float, default=10,
        help="the number of atoms")
    parser.add_argument('--target-network-frequency', type=int, default=1000,
        help="the timesteps it takes to update the target network")
    parser.add_argument('--max-grad-norm', type=float, default=0.5,
        help='the maximum norm for the gradient clipping')
    parser.add_argument('--start-e', type=float, default=1.,
        help="the starting epsilon for exploration")
    parser.add_argument('--end-e', type=float, default=0.01,
        help="the ending epsilon for exploration")
    parser.add_argument('--exploration-fraction', type=float, default=0.10,
        help="the fraction of `total-timesteps` it takes from start-e to go end-e")
    args = parser.parse_args()
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    # fmt: on
    return args


def make_env(gym_id, seed, idx, capture_video, run_name):
    def thunk():
        env = gym.make(gym_id)
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
        env = gym.wrappers.ResizeObservation(env, (84, 84))
        env = gym.wrappers.GrayScaleObservation(env)
        env = gym.wrappers.FrameStack(env, 4)
        env.seed(seed)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        return env

    return thunk


class QNetwork(nn.Module):
    def __init__(self, envs, frames=4, n_atoms=51, v_min=-10, v_max=10):
        super(QNetwork, self).__init__()
        self.n_atoms = n_atoms
        self.atoms = torch.linspace(v_min, v_max, steps=n_atoms).to(device)
        self.network = nn.Sequential(
            nn.Conv2d(frames, 32, 8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3136, 512),
            nn.ReLU(),
            nn.Linear(512, envs.single_action_space.n * n_atoms),
        )

    def forward(self, x):
        return self.network(x / 255.0)

    def get_action(self, envs, x, action=None):
        logits = self.forward(x)
        # probability mass function for each action
        pmfs = torch.softmax(logits.view(len(x), envs.single_action_space.n, self.n_atoms), dim=2)
        q_values = (pmfs * self.atoms).sum(2)
        if action is None:
            action = torch.argmax(q_values, 1)
            random_action = torch.randint(0, envs.single_action_space.n, (len(x),), device=device)
            random_action_flag = torch.rand(len(x), device=device) > epsilon
            action = torch.where(random_action_flag, action, random_action)

        return action, pmfs[torch.arange(len(x)), action]


def linear_schedule(start_e: float, end_e: float, duration: int, t: int):
    slope = (end_e - start_e) / duration
    return max(slope * t + start_e, end_e)


if __name__ == "__main__":
    args = parse_args()
    run_name = f"{args.gym_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
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

    # env setup
    envs = gym.vector.SyncVectorEnv(
        [make_env(args.gym_id, args.seed + i, i, args.capture_video, run_name) for i in range(args.num_envs)]
    )
    assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"

    # ALGO LOGIC: initialize agent here:
    q_network = QNetwork(envs, n_atoms=args.n_atoms, v_min=args.v_min, v_max=args.v_max).to(device)
    target_network = QNetwork(envs, n_atoms=args.n_atoms, v_min=args.v_min, v_max=args.v_max).to(device)
    target_network.load_state_dict(q_network.state_dict())
    optimizer = optim.Adam(q_network.parameters(), lr=args.learning_rate, eps=0.01 / args.batch_size)
    loss_fn = nn.MSELoss()

    # ALGO Logic: Storage setup
    obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()
    next_obs = torch.Tensor(envs.reset()).to(device)
    next_done = torch.zeros(args.num_envs).to(device)
    num_updates = args.total_timesteps // args.batch_size
    num_gradient_updates = 0

    for update in range(1, num_updates + 1):
        # ROLLOUTS
        epsilon = linear_schedule(args.start_e, args.end_e, args.exploration_fraction * args.total_timesteps, global_step)
        writer.add_scalar("charts/epsilon", epsilon, global_step)
        for step in range(0, args.num_steps):
            global_step += 1 * args.num_envs
            obs[step] = next_obs
            dones[step] = next_done
            # ALGO LOGIC: action logic
            with torch.no_grad():
                action, _ = q_network.get_action(envs, next_obs)
            actions[step] = action

            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, reward, done, info = envs.step(action.cpu().numpy())
            rewards[step] = torch.tensor(reward).to(device).view(-1)
            next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(done).to(device)

            for item in info:
                if "episode" in item.keys():
                    print(f"global_step={global_step}, episodic_return={item['episode']['r']}")
                    writer.add_scalar("charts/episodic_return", item["episode"]["r"], global_step)
                    writer.add_scalar("charts/episodic_length", item["episode"]["l"], global_step)
                    break

        # TRAINING
        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_actions = actions.reshape((-1, 1)).long()
        b_rewards = rewards.reshape((-1,))
        b_dones = dones.reshape((-1,))

        # next_obs index manipulation
        b_next_obs = torch.zeros_like(obs).to(device)
        b_next_obs[:-1] = obs[1:]
        b_next_obs[-1] = next_obs
        b_next_obs = b_next_obs.reshape((-1,) + envs.single_observation_space.shape)
        b_inds = np.arange(args.batch_size)

        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                num_gradient_updates += 1
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                with torch.no_grad():
                    _, next_pmfs = target_network.get_action(envs, b_next_obs[mb_inds])
                    next_atoms = b_rewards[mb_inds].unsqueeze(-1) + args.gamma * q_network.atoms * (
                        1 - b_dones[mb_inds].unsqueeze(-1)
                    )
                    # projection
                    delta_z = q_network.atoms[1] - q_network.atoms[0]
                    tz = next_atoms.clamp(args.v_min, args.v_max)
                    b = (tz - args.v_min) / delta_z
                    l = b.floor().clamp(0, args.n_atoms - 1)
                    u = b.ceil().clamp(0, args.n_atoms - 1)
                    # (l == u).float() handles the case where bj is exactly an integer
                    # example bj = 1, then the upper ceiling should be uj= 2, and lj= 1
                    d_m_l = (u + (l == u).float() - b) * next_pmfs
                    d_m_u = (b - l) * next_pmfs
                    target_pmfs = torch.zeros_like(next_pmfs)

                    for i in range(target_pmfs.size(0)):
                        target_pmfs[i].index_add_(0, l[i].long(), d_m_l[i])
                        target_pmfs[i].index_add_(0, u[i].long(), d_m_u[i])

                _, old_pmfs = q_network.get_action(envs, b_obs[mb_inds], b_actions[mb_inds].squeeze())
                loss = (-(target_pmfs * old_pmfs.clamp(min=1e-5).log()).sum(-1)).mean()

                # loss = (target_pmfs * (target_pmfs.clamp(min=1e-5).log() - old_pmfs.clamp(min=1e-5).log())).sum(-1).mean()

                writer.add_scalar("losses/td_loss", loss, global_step)

                # optimize the model
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(list(q_network.parameters()), args.max_grad_norm)
                optimizer.step()

                # update the target network
                if num_gradient_updates % args.target_network_frequency == 0:
                    target_network.load_state_dict(q_network.state_dict())

    envs.close()
    writer.close()
