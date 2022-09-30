# Proximal Policy Option Critic (PPOC) Original implementation
# Paper: https://arxiv.org/pdf/1712.00004.pdf
# Reference repo: https://github.com/mklissa/PPOC
"""Notes

MLP policy in PPOC is weird:
    1 body shared by critic and termination (but no gradient from termination)
    1 body shared by intra- and inter-option actors (but no gradient from inter-)

1 environment despite using PPO
uses batch_size=32 (if 2 options) or 64 (if 1 option) instead of 32 minibatches
GAE computation based on value of current option (Q(s, w))
Observation normalization but not reward
Per-iteration advantage normalization, no vloss clipping

Consider: PG and termination loss based on s'. Q and policy loss based on s.
    Don't resample termination and option for s' after rollout

Weird tricks:
    rewards are divided by 10 (if using options)
    loops over options, then over epochs (so updates 1 option, then the next, then the next)
    Only updates an option if we have at least 160 datapoints with that option (otherwise save those in a dataset, use it later)
    Termination loss has 5e-7 learning rate (not detaul 3e-4)

Errors:
    - General issues in non-weightsharing derivation being used in weight sharing
    - GAE is computed using Q(s, w) as baseline, bootstrapped from Q(s', w), and used for intra- AND inter-option policies
        - But baseline must be action-independent. Inter-option policy baseline should be V(s) (or maybe U(s', w))
    - Termination advantages should use beta(s', w) and A(s',w), but actually use A(s,w)
    - Inter-option policy advantage should use V(s') as baseline, bootstrap off V(s'')

Execution:
    o = env.reset()  # Start episode
    option = pi.get_option(o)  # Start with option
    loop:
        a, lp_a, Q = pi.act(o, option)
        o, r, new = env.step(a)
        rew /= 10 (if options)
        term = pi.get_term(o)
        if term: option = pi.get_option(o)
        if new: o = env.reset(); option = pi.get_option()
        if t == T: yield accumulated sequences + bootstrap * (1 - new)
"""

import argparse
import os
import random
import time
from collections import deque
from distutils.util import strtobool
from typing import Final, NamedTuple, Tuple

import envpool
import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.distributions import Normal
Tensor = torch.Tensor


def parse_args():
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", type=str, default=os.path.basename(__file__).rstrip(".py"),
        help="the name of this experiment")
    parser.add_argument("--seed", type=int, default=1,
        help="seed of the experiment")
    parser.add_argument("--torch-deterministic", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, `torch.backends.cudnn.deterministic=False`")
    parser.add_argument("--cuda", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="if toggled, cuda will be enabled by default")
    parser.add_argument("--track", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="if toggled, this experiment will be tracked with Weights and Biases")
    parser.add_argument("--wandb-project-name", type=str, default="cleanRL",
        help="the wandb's project name")
    parser.add_argument("--wandb-entity", type=str, default=None,
        help="the entity (team) of wandb's project")
    parser.add_argument("--capture-video", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="weather to capture videos of the agent performances (check out `videos` folder)")

    # Algorithm specific arguments
    parser.add_argument("--env-id", type=str, default="HalfCheetah-v4",
        help="the id of the environment")
    parser.add_argument("--total-timesteps", type=int, default=int(1e6),
        help="total timesteps of the experiments")
    parser.add_argument("--learning-rate", type=float, default=3e-4,
        help="the learning rate of the optimizer")
    parser.add_argument("--num-envs", type=int, default=1,
        help="the number of parallel game environments")
    parser.add_argument("--num-steps", type=int, default=2048,
        help="the number of steps to run in each environment per policy rollout")
    parser.add_argument("--anneal-lr", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Toggle learning rate annealing for policy and value networks")
    parser.add_argument("--gamma", type=float, default=0.99,
        help="the discount factor gamma")
    parser.add_argument("--gae-lambda", type=float, default=0.95,
        help="the lambda for the general advantage estimation")
    parser.add_argument("--num-minibatches", type=int, default=32,
        help="the number of mini-batches")
    parser.add_argument("--update-epochs", type=int, default=10,
        help="the K epochs to update the policy")
    parser.add_argument("--norm-adv", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Toggles advantages normalization")
    parser.add_argument("--clip-coef", type=float, default=0.2,
        help="the surrogate clipping coefficient")
    parser.add_argument("--ent-coef", type=float, default=0.0,
        help="coefficient of the entropy")
    parser.add_argument("--vf-coef", type=float, default=0.5,
        help="coefficient of the value function")
    parser.add_argument("--max-grad-norm", type=float, default=0.5,
        help="the maximum norm for the gradient clipping")
    parser.add_argument("--target-kl", type=float, default=None,
        help="the target KL divergence threshold")
    parser.add_argument("--num-options", type=int, default=2,
        help="the number of options available")
    parser.add_argument("--delib-cost", type=float, default=0.,
        help="cost for switching options")
    args = parser.parse_args()
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    # fmt: on
    return args


def layer_init(layer: nn.Module, std: float = 2 ** 0.5, bias_const: float = 0.):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Tanh(nn.Module):
    """In-place tanh module"""
    def __init__(self):
        super().__init__()

    def forward(self, input: Tensor) -> Tensor:
        return torch.tanh_(input)


class StepOutput(NamedTuple):
    termination: Tensor
    option: Tensor
    action: Tensor
    betas: Tensor
    option_logprobs: Tensor
    logprobs: Tensor
    qs: Tensor
    v: Tensor


@torch.jit.script
def batched_index(idx: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """Return contents of t at n-D array idx. Leading dim of t must match dims of idx"""
    dim = len(idx.shape)
    assert idx.shape == t.shape[:dim]
    num = idx.numel()
    t_flat = t.view((num,) + t.shape[dim:])
    s_flat = t_flat[torch.arange(num, device=t.device), idx.view(-1)]
    return s_flat.view(t.shape[:dim] + t.shape[dim + 1:])


class Agent(nn.Module):
    num_actions: Final[int]
    num_options: Final[int]

    def __init__(self, envs, args):
        super().__init__()
        self.num_actions = int(np.prod(envs.single_action_space.shape))
        self.num_options = args.num_options
        num_obs = int(np.prod(envs.single_observation_space.shape))
        # Critic body shared by termination and critic
        self.critic_body = nn.Sequential(
            layer_init(nn.Linear(num_obs, 64)),
            Tanh(),
            layer_init(nn.Linear(64, 64)),
            Tanh(),
        )
        self.critic = layer_init(nn.Linear(64, self.num_options), 1.)
        self.termination = layer_init(nn.Linear(64, self.num_options), 1e-2)  # is detached in original implement
        # Actor body shared by both actors
        self.actor_body = nn.Sequential(
            layer_init(nn.Linear(num_obs, 64)),
            Tanh(),
            layer_init(nn.Linear(64, 64)),
            Tanh(),
        )
        self.actor_mean = layer_init(nn.Linear(64, int(self.num_actions * self.num_options)), std=0.01)
        self.actor_logstd = nn.Parameter(torch.zeros(1, self.num_options, self.num_actions))
        self.option_actor = layer_init(nn.Linear(64, self.num_options), std=0.01)  # is detached in original implement

    @torch.jit.export
    def get_option_logprobs_and_values(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """Log probability and q value for each option in state x"""
        x = self.critic_body(x)
        pi_lp = torch.log_softmax(self.option_actor(x.detach()), -1)
        q = self.critic(x)
        return pi_lp, q


    def step(self, x: Tensor, option_on_arrival: Tensor, is_done: Tensor) -> StepOutput:
        """Sample termination, then new option if needed. Pick action, return values"""
        x_qt = self.critic_body(x)
        q = self.critic(x_qt)
        beta = F.sigmoid(self.termination(x_qt.detach()))
        term = torch.bernoulli(batched_index(option_on_arrival, beta))  # Sample termination
        x_a = self.actor_body(x)
        option_logprobs = torch.log_softmax(self.option_actor(x_a.detach()), -1)
        v = (option_logprobs.exp() * q).sum(-1)
        option = torch.where((term + is_done) > 0,
                             torch.multinomial(option_logprobs.exp(), 1),
                             option_on_arrival)
        option_logprob = option_logprobs.gather(-1, option).squeeze(-1)
        action_mean = batched_index(option, self.actor_mean(x_a).reshape(-1, self.num_options, self.num_actions))
        action_logstd = batched_index(option.unsqueeze(0), self.actor_logstd).expand_as(action_mean)
        action_std = action_logstd.exp()
        # TODO: Replace with jit-compatible sampling
        probs = Normal(action_mean, action_std)
        action = probs.sample()
        action_logprob = probs.log_prob(action).sum(-1)
        return StepOutput(term, option, action, beta, option_logprob, action_logprob, q, v)



    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        action_mean = self.actor_mean(x)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), self.critic(x)


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

    # env setup
    envs = envpool.make_gym(args.env_id, num_envs=args.num_envs)
    envs.is_vector_env = True
    envs.single_action_space = envs.action_space
    envs.single_observation_space = envs.observation_space
    envs = gym.wrappers.RecordEpisodeStatistics(envs)
    envs = gym.wrappers.NormalizeObservation(envs)
    envs = gym.wrappers.NormalizeReward(envs)

    assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"

    agent = Agent(envs, args).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    # ALGO Logic: Storage setup
    obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape, device=device)
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape, device=device)
    logprobs = torch.zeros((args.num_steps, args.num_envs), device=device)
    rewards = torch.zeros((args.num_steps, args.num_envs), device=device)
    dones = torch.zeros((args.num_steps, args.num_envs), device=device)
    values = torch.zeros((args.num_steps, args.num_envs), device=device)

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()
    next_obs = torch.tensor(envs.reset(), device=device)
    next_done = torch.zeros(args.num_envs, device=device)
    num_updates = args.total_timesteps // args.batch_size

    for update in range(1, num_updates + 1):
        # Annealing the rate if instructed to do so.
        if args.anneal_lr:
            frac = 1.0 - (update - 1.0) / num_updates
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        for step in range(0, args.num_steps):
            global_step += 1 * args.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            # ALGO LOGIC: action logic
            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs)
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob

            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, reward, done, info = envs.step(action.cpu().numpy())
            rewards[step] = torch.tensor(reward, device=device).view(-1)
            next_obs, next_done = torch.Tensor(next_obs, device=device), torch.Tensor(done, device=device)

            for item in info:
                if "episode" in item.keys():
                    print(f"global_step={global_step}, episodic_return={item['episode']['r']}")
                    writer.add_scalar("charts/episodic_return", item["episode"]["r"], global_step)
                    writer.add_scalar("charts/episodic_length", item["episode"]["l"], global_step)
                    break

        # bootstrap value if not done
        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(1, -1)
            if args.gae:
                advantages = torch.zeros_like(rewards, device=device)
                lastgaelam = 0
                for t in reversed(range(args.num_steps)):
                    if t == args.num_steps - 1:
                        nextnonterminal = 1.0 - next_done
                        nextvalues = next_value
                    else:
                        nextnonterminal = 1.0 - dones[t + 1]
                        nextvalues = values[t + 1]
                    delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                    advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
                returns = advantages + values
            else:
                returns = torch.zeros_like(rewards, device=device)
                for t in reversed(range(args.num_steps)):
                    if t == args.num_steps - 1:
                        nextnonterminal = 1.0 - next_done
                        next_return = next_value
                    else:
                        nextnonterminal = 1.0 - dones[t + 1]
                        next_return = returns[t + 1]
                    returns[t] = rewards[t] + args.gamma * nextnonterminal * next_return
                advantages = returns - values

        # flatten the batch
        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        # Optimizing the policy and value network
        b_inds = np.arange(args.batch_size)
        clipfracs = []
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs[mb_inds], b_actions[mb_inds])
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

                mb_advantages = b_advantages[mb_inds]
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                if args.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -args.clip_coef,
                        args.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

            if args.target_kl is not None:
                if approx_kl > args.target_kl:
                    break

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)
        print("SPS:", int(global_step / (time.time() - start_time)))
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

    envs.close()
    writer.close()