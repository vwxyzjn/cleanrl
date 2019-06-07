#!/usr/bin/env python3
import gym
import ptan
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
LEARNING_RATE = 1e-4
ENTROPY_BETA = 0.01
NUM_ENVS = 16

REWARD_STEPS = 4
CLIP_GRAD = 0.1


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


def unpack_batch(batch, net, device="cpu"):
    """
    Convert batch into training tensors
    :param batch:
    :param net:
    :return: states variable, actions tensor, reference values variable
    """
    states = []
    actions = []
    rewards = []
    not_done_idx = []
    last_states = []
    for idx, exp in enumerate(batch):
        states.append(np.array(exp.state, copy=False))
        actions.append(int(exp.action))
        rewards.append(exp.reward)
        if exp.last_state is not None:
            not_done_idx.append(idx)
            last_states.append(np.array(exp.last_state, copy=False))
    states_v = torch.FloatTensor(states).to(device)
    actions_t = torch.LongTensor(actions).to(device)

    # handle rewards
    rewards_np = np.array(rewards, dtype=np.float32)
    if not_done_idx:
        last_states_v = torch.FloatTensor(last_states).to(device)
        last_vals_v = net(last_states_v)[1]
        last_vals_np = last_vals_v.data.cpu().numpy()[:, 0]
        rewards_np[not_done_idx] += GAMMA ** REWARD_STEPS * last_vals_np

    ref_vals_v = torch.FloatTensor(rewards_np).to(device)
    return states_v, actions_t, ref_vals_v


def set_seed(seed, envs=None, cuda=False):
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

    make_env = lambda: ptan.common.wrappers.wrap_dqn(gym.make("PongNoFrameskip-v4"))
    envs = [make_env() for _ in range(NUM_ENVS)]
    writer = SummaryWriter(comment="-pong-a2c-rollouts_" + args.name)
    set_seed(20, envs, cuda=args.cuda)

    net = AtariA2C(envs[0].observation_space.shape, envs[0].action_space.n).to(device)
    print(net)

    agent = ptan.agent.ActorCriticAgent(net, apply_softmax=True, device=device)
    exp_source = ptan.experience.ExperienceSourceRollouts(envs, agent, gamma=GAMMA, steps_count=REWARD_STEPS)

    optimizer = optim.RMSprop(net.parameters(), lr=LEARNING_RATE, eps=1e-5)

    step_idx = 0

    with common.RewardTracker(writer, stop_reward=18) as tracker:
        with ptan.common.utils.TBMeanTracker(writer, batch_size=10) as tb_tracker:
            for mb_states, mb_rewards, mb_actions, mb_values in exp_source:
                # handle new rewards
                new_rewards = exp_source.pop_total_rewards()
                if new_rewards:
                    if tracker.reward(np.mean(new_rewards), step_idx):
                        break

                optimizer.zero_grad()
                states_v = torch.FloatTensor(mb_states).to(device)
                mb_adv = mb_rewards - mb_values
                adv_v = torch.FloatTensor(mb_adv).to(device)
                actions_t = torch.LongTensor(mb_actions).to(device)
                vals_ref_v = torch.FloatTensor(mb_rewards).to(device)

                logits_v, value_v = net(states_v)
                loss_value_v = F.mse_loss(value_v.squeeze(-1), vals_ref_v)

                log_prob_v = F.log_softmax(logits_v, dim=1)
                log_prob_actions_v = adv_v * log_prob_v[range(len(mb_states)), actions_t]
                loss_policy_v = -log_prob_actions_v.mean()

                prob_v = F.softmax(logits_v, dim=1)
                entropy_loss_v = (prob_v * log_prob_v).sum(dim=1).mean()

                # apply entropy and value gradients
                loss_v = loss_policy_v + ENTROPY_BETA * entropy_loss_v + loss_value_v
                loss_v.backward()
                nn_utils.clip_grad_norm_(net.parameters(), CLIP_GRAD)
                optimizer.step()

                tb_tracker.track("advantage",       adv_v, step_idx)
                tb_tracker.track("values",          value_v, step_idx)
                tb_tracker.track("batch_rewards",   vals_ref_v, step_idx)
                tb_tracker.track("loss_entropy",    entropy_loss_v, step_idx)
                tb_tracker.track("loss_policy",     loss_policy_v, step_idx)
                tb_tracker.track("loss_value",      loss_value_v, step_idx)
                tb_tracker.track("loss_total",      loss_v, step_idx)
                step_idx += NUM_ENVS * REWARD_STEPS
