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

Weird tricks:
    rewards are divided by 10 (if using options)
    loops over options, then over epochs (so updates 1 option, then the next, then the next)
    Only updates an option if we have at least 160 datapoints with that option (otherwise save those in a dataset, use it later)
    Termination loss has 5e-7 learning rate (not default 3e-4)

Errors:
    - General issues in non-weightsharing derivation being used in weight sharing
    - GAE is computed using Q(s, w) as baseline, bootstrapped from Q(s', w), and used for intra- AND inter-option policies
        - But baseline must be action-independent. Inter-option policy baseline should be V(s) (or maybe U(s', w))
    - Termination advantages should use beta(s', w) and A(s',w), but actually use A(s,w)
    - Inter-option policy advantage should use V(s') as baseline, bootstrap off V(s'')
"""

import argparse
import math
import os
import random
import time
from collections import deque
from distutils.util import strtobool
from typing import Final, NamedTuple, Optional, Tuple

import envpool
import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

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
    parser.add_argument("--option-ent-coef", type=float, default=0.01,
        help="coefficient of the option entropy")
    parser.add_argument("--vf-coef", type=float, default=0.5,
        help="coefficient of the value function")
    parser.add_argument("--term-coef", type=float, default=1. / 600,
        help="coefficient of the termination loss")
    parser.add_argument("--max-grad-norm", type=float, default=1.,
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


class RecordEpisodeStatistics(gym.Wrapper):
    """Simultaneously record statistics and shim to other gym wrappers"""

    def __init__(self, env: gym.Env, deque_size: int = 100):
        super().__init__(env)
        self.num_envs = getattr(env, "num_envs", 1)
        self.episode_count = 0
        self.episode_start_times: np.ndarray = None
        self.episode_returns: Optional[np.ndarray] = None
        self.episode_lengths: Optional[np.ndarray] = None
        self.return_queue = deque(maxlen=deque_size)
        self.length_queue = deque(maxlen=deque_size)
        self.is_vector_env = getattr(env, "is_vector_env", False)

    def reset(self, **kwargs):
        """Resets the environment using kwargs and resets the episode returns and lengths."""
        obs = super().reset(**kwargs)
        self.episode_start_times = np.full(self.num_envs, time.perf_counter(), dtype=np.float32)
        self.episode_returns = np.zeros(self.num_envs, dtype=np.float32)
        self.episode_lengths = np.zeros(self.num_envs, dtype=np.int32)
        return obs, {}

    def step(self, action):
        """Steps through the environment, recording the episode statistics."""
        (
            observations,
            rewards,
            terminations,
            infos,
        ) = self.env.step(action)
        truncations = infos["TimeLimit.truncated"]
        assert isinstance(
            infos, dict
        ), f"`info` dtype is {type(infos)} while supported dtype is `dict`. This may be due to usage of other wrappers in the wrong order."
        self.episode_returns += rewards
        self.episode_lengths += 1
        dones = np.logical_or(terminations, truncations)
        num_dones = np.sum(dones)
        if num_dones:
            if "episode" in infos or "_episode" in infos:
                raise ValueError("Attempted to add episode stats when they already exist")
            else:
                infos["episode"] = {
                    "r": np.where(dones, self.episode_returns, 0.0),
                    "l": np.where(dones, self.episode_lengths, 0),
                    "t": np.where(
                        dones,
                        np.round(time.perf_counter() - self.episode_start_times, 6),
                        0.0,
                    ),
                }
                if self.is_vector_env:
                    infos["_episode"] = np.where(dones, True, False)
            self.return_queue.extend(self.episode_returns[dones])
            self.length_queue.extend(self.episode_lengths[dones])
            self.episode_count += num_dones
            self.episode_lengths[dones] = 0
            self.episode_returns[dones] = 0
            self.episode_start_times[dones] = time.perf_counter()
        return (
            observations,
            rewards,
            terminations,
            truncations,
            infos,
        )


def layer_init(layer: nn.Module, std: float = 2**0.5, bias_const: float = 0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Tanh(nn.Module):
    """In-place tanh module"""

    def forward(self, input: Tensor) -> Tensor:
        return torch.tanh_(input)


class StepOutput(NamedTuple):
    termination: Tensor  # t
    option: Tensor  # w
    action: Tensor  # a
    betas: Tensor  # beta(s, w_)
    option_logprobs: Tensor  # lp(w|s)
    logprobs: Tensor  # lp(a|s, w)
    qs: Tensor  # Q(s, .)
    v: Tensor  # V(s)


@torch.jit.script
def batched_index(idx: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """Return contents of t at n-D array idx. Leading dim of t must match dims of idx"""
    dim = len(idx.shape)
    assert idx.shape == t.shape[:dim]
    num = idx.numel()
    t_flat = t.view((num,) + t.shape[dim:])
    s_flat = t_flat[torch.arange(num, device=t.device), idx.view(-1)]
    return s_flat.view(t.shape[:dim] + t.shape[dim + 1 :])


def body(num_obs: int) -> nn.Sequential:
    """Base 2-layer MLP with Tanh nonlinearities"""
    return nn.Sequential(layer_init(nn.Linear(num_obs, 64)), Tanh(), layer_init(nn.Linear(64, 64)), Tanh())


class Agent(nn.Module):
    num_actions: Final[int]
    num_options: Final[int]

    def __init__(self, envs, args):
        super().__init__()
        self.num_actions = int(np.prod(envs.single_action_space.shape))
        self.num_options = args.num_options
        num_obs = int(np.prod(envs.single_observation_space.shape))
        self.critic_body = body(num_obs)  # shared by critic and termination, termination is detached
        self.critic = layer_init(nn.Linear(64, self.num_options), 1.0)
        self.termination = layer_init(nn.Linear(64, self.num_options), 1e-2)
        self.actor_body = body(num_obs)  # shared by intra- and inter-option actors, inter-option detached
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(64, int(self.num_actions * self.num_options)), std=0.01),
            nn.Unflatten(-1, (self.num_options, self.num_actions)),
        )
        self.actor_logstd = nn.Parameter(torch.zeros(1, self.num_options, self.num_actions))
        self.option_actor = layer_init(nn.Linear(64, self.num_options), std=0.01)

    @torch.jit.export
    def get_value(self, x: Tensor):
        """Return various bootstrap values"""
        q = self.critic(self.critic_body(x))
        option_probs = torch.softmax(self.option_actor(self.actor_body(x).detach()), -1)
        v = (q * option_probs).sum(-1)
        return q, v

    @torch.jit.export
    def step(self, x: Tensor, option_on_arrival: Tensor, is_done: Tensor) -> StepOutput:
        """Sample termination, then new option if needed. Pick action, return values"""
        # Check termination on entering state x
        x_qt = self.critic_body(x)
        x_a = self.actor_body(x)
        betas = torch.sigmoid(self.termination(x_qt.detach()))
        beta = batched_index(option_on_arrival, betas)
        term = torch.bernoulli(beta)
        # Choose new option where terminated (or where new episode)
        option_logprobs = torch.log_softmax(self.option_actor(x_a.detach()), -1)
        option_probs = option_logprobs.exp()
        option = torch.where((term + is_done) > 0, torch.multinomial(option_probs, 1).squeeze(-1), option_on_arrival)
        option_logprob = option_logprobs.gather(-1, option.unsqueeze(-1)).squeeze(-1)
        # Choose action from selected option(s)
        all_action_mean = self.actor_mean(x_a).reshape(-1, self.num_options, self.num_actions)
        all_action_logstd = self.actor_logstd.expand_as(all_action_mean)
        action_mean = batched_index(option, all_action_mean)
        action_logstd = batched_index(option, all_action_logstd)
        action_std = action_logstd.exp()
        action = torch.normal(action_mean, action_std)
        action_logprob = (
            -((action - action_mean) ** 2) / (2 * action_std**2)
            - action_logstd  # from torch.Normal
            - math.log(math.sqrt(2 * math.pi))
        ).sum(-1)
        # Get q value of each option and policy-weighted average V
        q = self.critic(x_qt)
        v = (q * option_probs).sum(-1)
        return StepOutput(term, option, action, beta, option_logprob, action_logprob, q, v)

    @torch.jit.export
    def unroll(self, xs: Tensor, options_on_arrival: Tensor, options: Tensor, actions: Tensor):
        """Compute gradients over a batch of data"""
        x_qts = self.critic_body(xs)
        x_as = self.actor_body(xs)
        # Probability of terminating previous option in each step
        prev_termprobs = torch.sigmoid(batched_index(options_on_arrival, self.termination(x_qts.detach())))
        # Probability of selecting chosen option in each step
        all_option_logprobs = torch.log_softmax(self.option_actor(x_as.detach()), -1)
        option_logprobs = all_option_logprobs.gather(-1, options.unsqueeze(-1)).squeeze(-1)
        option_entropy = (-all_option_logprobs * all_option_logprobs.exp()).sum(-1)
        # Probability of selecting chosen action in each step
        all_action_mean = self.actor_mean(x_as).reshape(-1, self.num_options, self.num_actions)
        all_action_logstd = self.actor_logstd.expand_as(all_action_mean)
        action_mean = batched_index(options, all_action_mean)
        action_logstd = batched_index(options, all_action_logstd)
        action_std = action_logstd.exp()
        action_logprob = (  # from torch.Normal
            -((actions - action_mean) ** 2) / (2 * action_std**2) - action_logstd - math.log(math.sqrt(2 * math.pi))
        ).sum(-1)
        action_entropy = (0.5 + 0.5 * math.log(2 * math.pi) + action_logstd).sum(-1)
        # Value of selected option
        q_sw = batched_index(options, self.critic(x_qts))
        return (prev_termprobs, option_logprobs, option_entropy), (action_logprob, action_entropy), q_sw


@torch.jit.script
def gae(v_tm1_t: Tensor, r_t: Tensor, gamma_t: Tensor, lambda_: float = 1.0) -> Tuple[Tensor, Tensor]:
    """Generalized advantage estimation with lambda-returns"""
    v_tm1, v_t = v_tm1_t[:-1], v_tm1_t[1:]
    deltas = r_t + gamma_t * v_t - v_tm1
    adv = torch.zeros_like(v_t)
    lastgaelam = torch.zeros_like(v_t[0])
    for t in range(v_t.shape[0] - 1, -1, -1):
        lastgaelam = adv[t] = deltas[t] + gamma_t[t] * lambda_ * lastgaelam
    ret = adv + v_tm1
    return adv, ret


@torch.jit.script
def normalize(x: Tensor) -> Tensor:
    """Scripted per-epoch normalization"""
    return (x - x.mean()) / (x.std() + 1e-8)


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
    envs.num_envs = args.num_envs
    envs.single_action_space = envs.action_space
    envs.single_observation_space = envs.observation_space
    envs = RecordEpisodeStatistics(envs)
    envs = gym.wrappers.NormalizeObservation(envs)
    envs = gym.wrappers.TransformReward(envs, lambda x: x / 10.0)  # normalized obs, but scaled reward

    assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"

    agent = Agent(envs, args).to(device)
    # agent = torch.jit.script(agent)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    # ALGO Logic: Storage setup
    obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape, device=device)
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape, device=device)
    options_buffer = torch.zeros((args.num_steps + 1, args.num_envs), dtype=torch.int64, device=device)
    options = options_buffer[1:]
    options_on_arrival = options_buffer[:-1]
    betas_on_arrival = torch.zeros((args.num_steps, args.num_envs), device=device)
    logprobs = torch.zeros((args.num_steps, args.num_envs), device=device)
    option_logprobs = torch.zeros((args.num_steps, args.num_envs), device=device)
    rewards = torch.zeros((args.num_steps, args.num_envs), device=device)
    dones = torch.zeros((args.num_steps + 1, args.num_envs), device=device)
    qvalues = torch.zeros((args.num_steps + 1, args.num_envs, args.num_options), device=device)
    vvalues = torch.zeros((args.num_steps + 1, args.num_envs), device=device)
    uvalues = torch.zeros((args.num_steps + 1, args.num_envs, args.num_options), device=device)
    terminations = torch.zeros((args.num_steps, args.num_envs), device=device)

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()
    next_obs = torch.tensor(envs.reset()[0], device=device, dtype=torch.float32)
    next_done = torch.ones(args.num_envs, device=device)
    next_option = torch.zeros(args.num_envs, device=device, dtype=torch.int64)
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
            options_on_arrival[step] = next_option

            # ALGO LOGIC: action logic
            with torch.no_grad():
                oc_output = agent.step(next_obs, next_option, next_done)
                terminations[step] = oc_output.termination
                next_option = options[step] = oc_output.option
                actions[step] = action = oc_output.action
                betas_on_arrival[step] = oc_output.betas
                option_logprobs[step] = oc_output.option_logprobs
                logprobs[step] = oc_output.logprobs
                qvalues[step], vvalues[step], uvalues[step] = oc_output[-3:]

            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, reward, done, truncated, info = envs.step(action.cpu().numpy())
            rewards[step] = torch.tensor(reward, device=device)
            next_obs, next_done = torch.tensor(next_obs, device=device, dtype=torch.float32), torch.Tensor(done, device=device)

            if "episode" in info:
                first_idx = info["_episode"].nonzero()[0][0]
                print(f"global_step={global_step}, episodic_return={info['episode']['r'][first_idx]}")
                writer.add_scalar("charts/episodic_return", info["episode"]["r"][first_idx], global_step)
                writer.add_scalar("charts/episodic_length", info["episode"]["l"][first_idx], global_step)
                break

        # bootstrap value if not done
        with torch.no_grad():
            # deliberation cost where we terminated without being forced to
            rewards -= args.delib_cost * (terminations * (1.0 - dones[:-1]))
            qvalues[-1], vvalues[-1] = agent.get_value(next_obs)
            # bootstrap both advantages with V(s') (should be Q(s', w') from original)
            q_tm1_t = torch.cat((batched_index(options, qvalues[:-1]), vvalues[-1].unsqueeze(0)), 0)
            dones[-1] = next_done
            gamma_t = 1.0 - dones[1:]
            advantages, returns = gae(q_tm1_t, rewards, gamma_t, args.gae_lambda)
            # Carry over off-by-one error from implementation (q value of selected option instead of prev)
            termination_advantages = q_tm1_t[:-1] - vvalues[:-1] + args.delib_cost

        # flatten the batch
        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_qvalues = q_tm1_t[:-1].reshape(-1)  # q values of selected options
        b_termination_advantages = termination_advantages.reshape(-1)
        b_option_logprobs = option_logprobs.reshape(-1)
        b_options = options.reshape(-1)
        b_options_on_arrival = options_on_arrival.reshape(-1)

        # Optimizing the policy and value network
        b_inds = np.arange(args.batch_size)
        clipfracs = []
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                (newbetas, new_option_lp, option_entropy), (newlogprob, entropy), newvalue = agent.unroll(
                    b_obs[mb_inds], b_options_on_arrival[mb_inds], b_options[mb_inds], b_actions[mb_inds]
                )
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()
                option_logratio = new_option_lp - b_option_logprobs[mb_inds]
                option_ratio = option_logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

                mb_option_advantages = mb_advantages = b_advantages[mb_inds]
                if args.norm_adv:
                    mb_advantages = normalize(mb_advantages)

                # Intra-option policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = torch.maximum(pg_loss1, pg_loss2).mean()

                # Inter-option policy loss
                pg_loss1 = -mb_option_advantages * option_ratio
                pg_loss2 = -mb_option_advantages * torch.clamp(option_ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                option_pg_loss = torch.maximum(pg_loss1, pg_loss2).mean()

                # Termination loss
                termination_loss = (newbetas * b_termination_advantages[mb_inds]).mean()

                # Value loss
                v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                # Both entropy losses
                entropy_loss = -entropy.mean()
                option_entropy_loss = -option_entropy.mean()
                loss = (
                    pg_loss
                    + args.ent_coef * entropy_loss
                    + args.option_ent_coef * option_entropy_loss
                    + v_loss * args.vf_coef
                    + termination_loss * args.term_coef
                )

                optimizer.zero_grad(True)
                loss.backward()
                grad_norm = nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

            if args.target_kl is not None:
                if approx_kl > args.target_kl:
                    break

        y_pred, y_true = b_qvalues.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/option_policy_loss", option_pg_loss.item(), global_step)
        writer.add_scalar("losses/termination_loss", termination_loss.item(), global_step)
        writer.add_scalar("losses/termination_frequency", terminations.mean().item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/option_entropy", option_entropy_loss.item(), global_step)
        writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)
        writer.add_scalar("losses/max_grad_norm", grad_norm.item(), global_step)
        print("SPS:", int(global_step / (time.time() - start_time)))
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

    envs.close()
    writer.close()
