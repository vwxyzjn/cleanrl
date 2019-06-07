#!/usr/bin/env python3
import gym
import ptan
import time
import random
import numpy as np
import argparse
from tensorboardX import SummaryWriter

import torch
import torch.nn as nn
import torch.nn.utils as nn_utils
import torch.nn.functional as F
import torch.optim as optim

from lib import common

GAMMA = 0.99
LEARNING_RATE = 5e-4
ENTROPY_BETA = 0.01
NUM_ENVS = 16

REWARD_STEPS = 4
CLIP_GRAD = 0.1
IMG_SHAPE = (4, 84, 84)


class AtariA2C(nn.Module):
    def __init__(self, input_shape, n_actions):
        super(AtariA2C, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )

        conv_out_size = self._get_conv_out(input_shape)
        self.policy = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions)
        )

        self.value = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )

    def _get_conv_out(self, shape):
        o = self.conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))

    def forward(self, x):
        fx = x.float() / 256
        conv_out = self.conv(fx).view(fx.size()[0], -1)
        return self.policy(conv_out), self.value(conv_out)


def discount_with_dones(rewards, dones, gamma):
    discounted = []
    r = 0
    for reward, done in zip(rewards[::-1], dones[::-1]):
        r = reward + gamma*r*(1.-done)
        discounted.append(r)
    return discounted[::-1]


def iterate_batches(envs, net, device="cpu"):
    n_actions = envs[0].action_space.n
    act_selector = ptan.actions.ProbabilityActionSelector()
    obs = [e.reset() for e in envs]
    batch_dones = [[False] for _ in range(NUM_ENVS)]
    total_reward = [0.0] * NUM_ENVS
    total_steps = [0] * NUM_ENVS
    mb_obs = np.zeros((NUM_ENVS, REWARD_STEPS) + IMG_SHAPE, dtype=np.uint8)
    mb_rewards = np.zeros((NUM_ENVS, REWARD_STEPS), dtype=np.float32)
    mb_values = np.zeros((NUM_ENVS, REWARD_STEPS), dtype=np.float32)
    mb_actions = np.zeros((NUM_ENVS, REWARD_STEPS), dtype=np.int32)
    mb_probs = np.zeros((NUM_ENVS, REWARD_STEPS, n_actions), dtype=np.float32)

    while True:
        batch_dones = [[dones[-1]] for dones in batch_dones]
        done_rewards = []
        done_steps = []
        for n in range(REWARD_STEPS):
            obs_v = ptan.agent.default_states_preprocessor(obs).to(device)
            mb_obs[:, n] = obs_v.data.cpu().numpy()
            logits_v, values_v = net(obs_v)
            probs_v = F.softmax(logits_v, dim=1)
            probs = probs_v.data.cpu().numpy()
            actions = act_selector(probs)
            mb_probs[:, n] = probs
            mb_actions[:, n] = actions
            mb_values[:, n] = values_v.squeeze().data.cpu().numpy()
            for e_idx, e in enumerate(envs):
                o, r, done, _ = e.step(actions[e_idx])
                total_reward[e_idx] += r
                total_steps[e_idx] += 1
                if done:
                    o = e.reset()
                    done_rewards.append(total_reward[e_idx])
                    done_steps.append(total_steps[e_idx])
                    total_reward[e_idx] = 0.0
                    total_steps[e_idx] = 0
                obs[e_idx] = o
                mb_rewards[e_idx, n] = r
                batch_dones[e_idx].append(done)
        # obtain values for the last observation
        obs_v = ptan.agent.default_states_preprocessor(obs).to(device)
        _, values_v = net(obs_v)
        values_last = values_v.squeeze().data.cpu().numpy()

        for e_idx, (rewards, dones, value) in enumerate(zip(mb_rewards, batch_dones, values_last)):
            rewards = rewards.tolist()
            if not dones[-1]:
                rewards = discount_with_dones(rewards + [value], dones[1:] + [False], GAMMA)[:-1]
            else:
                rewards = discount_with_dones(rewards, dones[1:], GAMMA)
            mb_rewards[e_idx] = rewards

        out_mb_obs = mb_obs.reshape((-1,) + IMG_SHAPE)
        out_mb_rewards = mb_rewards.flatten()
        out_mb_actions = mb_actions.flatten()
        out_mb_values = mb_values.flatten()
        out_mb_probs = mb_probs.flatten()
        yield out_mb_obs, out_mb_rewards, out_mb_actions, out_mb_values, out_mb_probs, \
              np.array(done_rewards), np.array(done_steps)


def train_a2c(net, mb_obs, mb_rewards, mb_actions, mb_values, optimizer, tb_tracker, step_idx, device="cpu"):
    optimizer.zero_grad()
    mb_adv = mb_rewards - mb_values
    adv_v = torch.FloatTensor(mb_adv).to(device)
    obs_v = torch.FloatTensor(mb_obs).to(device)
    rewards_v = torch.FloatTensor(mb_rewards).to(device)
    actions_t = torch.LongTensor(mb_actions).to(device)
    logits_v, values_v = net(obs_v)
    loss_value_v = F.mse_loss(values_v.squeeze(-1), rewards_v)

    log_prob_v = F.log_softmax(logits_v, dim=1)
    log_prob_actions_v = adv_v * log_prob_v[range(len(mb_actions)), actions_t]
    loss_policy_v = -log_prob_actions_v.mean()

    prob_v = F.softmax(logits_v, dim=1)
    entropy_loss_v = (prob_v * log_prob_v).sum(dim=1).mean()
    loss_v = ENTROPY_BETA * entropy_loss_v + loss_value_v + loss_policy_v
    loss_v.backward()
    nn_utils.clip_grad_norm_(net.parameters(), CLIP_GRAD)
    optimizer.step()

    tb_tracker.track("advantage", mb_adv, step_idx)
    tb_tracker.track("values", values_v, step_idx)
    tb_tracker.track("batch_rewards", rewards_v, step_idx)
    tb_tracker.track("loss_entropy", entropy_loss_v, step_idx)
    tb_tracker.track("loss_policy", loss_policy_v, step_idx)
    tb_tracker.track("loss_value", loss_value_v, step_idx)
    tb_tracker.track("loss_total", loss_v, step_idx)
    return obs_v


def set_seed(seed, envs=None, cuda=False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if cuda:
        torch.cuda.manual_seed(seed)

    if envs:
        for idx, env in enumerate(envs):
            env.seed(seed + idx)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda", default=False, action="store_true", help="Enable cuda")
    parser.add_argument("-n", "--name", required=True, help="Name of the run")
    args = parser.parse_args()
    device = torch.device("cuda" if args.cuda else "cpu")

    make_env = lambda: ptan.common.wrappers.wrap_dqn(gym.make("BreakoutNoFrameskip-v4"))
    envs = [make_env() for _ in range(NUM_ENVS)]
    writer = SummaryWriter(comment="-pong-a2c-r2_" + args.name)
    set_seed(20, envs, cuda=args.cuda)

    net = AtariA2C(envs[0].observation_space.shape, envs[0].action_space.n).to(device)
    print(net)

    optimizer = optim.RMSprop(net.parameters(), lr=LEARNING_RATE, eps=1e-5)

    step_idx = 0
    total_steps = 0
    best_reward = None
    ts_start = time.time()

    with common.RewardTracker(writer, stop_reward=18) as tracker:
        with ptan.common.utils.TBMeanTracker(writer, batch_size=10) as tb_tracker:
            for mb_obs, mb_rewards, mb_actions, mb_values, _, done_rewards, done_steps in iterate_batches(envs, net, device=device):
                if len(done_rewards) > 0:
                    total_steps += sum(done_steps)
                    speed = total_steps / (time.time() - ts_start)
                    if best_reward is None:
                        best_reward = done_rewards.max()
                    elif best_reward < done_rewards.max():
                        best_reward = done_rewards.max()
                    tb_tracker.track("total_reward_max", best_reward, step_idx)
                    tb_tracker.track("total_reward", done_rewards, step_idx)
                    tb_tracker.track("total_steps", done_steps, step_idx)
                    print("%d: done %d episodes, mean_reward=%.2f, best_reward=%.2f, speed=%.2f" % (
                        step_idx, len(done_rewards), done_rewards.mean(), best_reward, speed))

                train_a2c(net, mb_obs, mb_rewards, mb_actions, mb_values,
                          optimizer, tb_tracker, step_idx, device=device)
                step_idx += 1
