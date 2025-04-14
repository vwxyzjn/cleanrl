# Implementation of Advantage Option Critic (A2OC in paper, but no asynchrony)
# Reference: https://github.com/jeanharb/a2oc_delib
# Paper: https://arxiv.org/pdf/1709.04571.pdf
import argparse
import os
import random
import time
from collections import deque
from distutils.util import strtobool
from typing import Final, Tuple

import envpool
import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
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
    parser.add_argument("--cuda", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
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
    parser.add_argument("--env-id", type=str, default="Amidar-v5",
                        help="the id of the environment")
    parser.add_argument("--total-timesteps", type=int, default=int(8e7),
                        help="total timesteps of the experiments")
    parser.add_argument("--learning-rate", type=float, default=0.0007,
                        help="the learning rate of the optimizer")
    parser.add_argument("--num-envs", type=int, default=16,
                        help="the number of parallel game environments")
    parser.add_argument("--num-steps", type=int, default=10,
                        help="the number of steps to run in each environment per policy rollout")
    parser.add_argument("--anneal-lr", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
                        help="Toggle learning rate annealing for policy and value networks")
    parser.add_argument("--gamma", type=float, default=0.99,
                        help="the discount factor gamma")
    parser.add_argument("--gae-lambda", type=float, default=1.,
                        help="the lambda for the general advantage estimation")
    parser.add_argument("--norm-adv", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
                        help="Toggles advantages normalization")
    parser.add_argument("--option-epsilon", type=float, default=0.1,
                        help="epsilon for epilon-greedy option selection")
    parser.add_argument("--ent-coef", type=float, default=0.01,
                        help="coefficient of the entropy")
    parser.add_argument("--delib-cost", type=float, default=0.02,
                        help="cost for terminating an option. subtracted from reward, added to termination advantage")
    parser.add_argument("--delib-cost-in-reward", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
                        help="if true, subtract cost from immediate reward (in addition to adding to termination advantage")
    parser.add_argument("--vf-coef", type=float, default=1.,
                        help="coefficient of the value function")
    parser.add_argument("--term-coef", type=float, default=1.,
                        help="coefficient of the value function")
    parser.add_argument("--num-options", type=int, default=8,
                        help="the number of options available")
    parser.add_argument("--max-grad-norm", type=float, default=1.,
                        help="the maximum norm for the gradient clipping")
    args = parser.parse_args()
    args.batch_size = int(args.num_envs * args.num_steps)
    # fmt: on
    return args


class RecordEpisodeStatistics(gym.Wrapper):
    def __init__(self, env, deque_size=100):
        super().__init__(env)
        self.num_envs = getattr(env, "num_envs", 1)
        self.episode_returns = None
        self.episode_lengths = None

    def reset(self, **kwargs):
        observations = super().reset(**kwargs)
        self.episode_returns = np.zeros(self.num_envs, dtype=np.float32)
        self.episode_lengths = np.zeros(self.num_envs, dtype=np.int32)
        self.lives = np.zeros(self.num_envs, dtype=np.int32)
        self.returned_episode_returns = np.zeros(self.num_envs, dtype=np.float32)
        self.returned_episode_lengths = np.zeros(self.num_envs, dtype=np.int32)
        return observations

    def step(self, action):
        observations, rewards, dones, infos = super().step(action)
        self.episode_returns += infos["reward"]
        self.episode_lengths += 1
        self.returned_episode_returns[:] = self.episode_returns
        self.returned_episode_lengths[:] = self.episode_lengths
        self.episode_returns *= 1 - infos["terminated"]
        self.episode_lengths *= 1 - infos["terminated"]
        infos["r"] = self.returned_episode_returns
        infos["l"] = self.returned_episode_lengths
        return (
            observations,
            rewards,
            dones,
            infos,
        )


@torch.jit.script
def batched_index(idx: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """Return contents of t at n-D array idx. Leading dim of t must match dims of idx"""
    dim = len(idx.shape)
    assert idx.shape == t.shape[:dim]
    num = idx.numel()
    t_flat = t.view((num,) + t.shape[dim:])
    s_flat = t_flat[torch.arange(num, device=t.device), idx.view(-1)]
    return s_flat.view(t.shape[:dim] + t.shape[dim + 1 :])


def layer_init(layer, std: float = nn.init.calculate_gain("relu"), bias_const: float = 0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    num_options: Final[int]
    num_actions: Final[int]
    option_epsilon: Final[int]

    def __init__(self, envs, args):
        super().__init__()
        # Base A3C CNN
        self.network = nn.Sequential(
            layer_init(nn.Conv2d(4, 16, 8, stride=4)),
            nn.ReLU(inplace=True),
            layer_init(nn.Conv2d(16, 32, 4, stride=2)),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            layer_init(nn.Linear(32 * 9 * 9, 256)),
            nn.ReLU(inplace=True),
        )
        self.num_actions = envs.single_action_space.n
        self.num_options = args.num_options
        self.option_epsilon = args.option_epsilon
        self.actor = layer_init(nn.Linear(256, int(self.num_actions * self.num_options)), std=0.01)
        self.critic = layer_init(nn.Linear(256, self.num_options), std=1)
        self.termination = layer_init(nn.Linear(256, self.num_options), std=0.01)

    def features(self, x: Tensor) -> Tensor:
        """Flattened output of CNN"""
        return self.network(x / 255.0)

    @torch.jit.export
    def get_bsv(self, next_x: Tensor, option_on_arrival: Tensor) -> Tensor:
        """Bootstrap value. U(s', w) = beta(s', w)V(s') + (1 - beta(s', w))Q(s', w)"""
        next_x = self.features(next_x)
        qs = self.critic(next_x)
        v = (1.0 - self.option_epsilon) * qs.max(-1)[0] + self.option_epsilon * qs.mean(-1)
        beta = torch.sigmoid(batched_index(option_on_arrival, self.termination(next_x)))
        u = (1.0 - beta) * batched_index(option_on_arrival, qs) + beta * v
        return u

    def sample_option(
        self,
        option_on_arrival: Tensor,
        qs: Tensor,
        terminations: Tensor,
    ) -> Tensor:
        """Sample option using epsilon-greedy sampling"""
        candidate_option = torch.where(
            torch.rand_like(option_on_arrival, dtype=torch.float32) < self.option_epsilon,
            torch.randint_like(option_on_arrival, 0, self.num_options),
            qs.argmax(-1),
        )
        return torch.where(terminations > 0, candidate_option, option_on_arrival)

    @torch.jit.export
    def step(
        self,
        x: Tensor,
        option_on_arrival: Tensor,
        is_done: Tensor,
    ):
        """Basic single step on receiving new observation.

        Determine terminations, pick new options, get actions and values"""
        x = self.features(x)
        qs = self.critic(x)  # Q for each option
        v = (1.0 - self.option_epsilon) * qs.max(-1)[0] + self.option_epsilon * qs.mean(-1)
        beta_w = torch.sigmoid(batched_index(option_on_arrival, self.termination(x)))  # Termination prob of current option
        term_w = torch.bernoulli(beta_w)  # Sample terminations
        option = self.sample_option(option_on_arrival, qs, term_w + is_done)  # also terminate on episode end
        pi_w = torch.softmax(
            batched_index(option, self.actor(x).reshape(-1, self.num_options, self.num_actions)),
            -1,
        )
        a = torch.multinomial(pi_w, 1)  # action under new option
        return (term_w, option, a.squeeze(-1)), beta_w, (qs, v)

    @torch.jit.export
    def unroll(self, xs: Tensor, options_on_arrival: Tensor, options: Tensor, actions: Tensor):
        """Unroll with gradients. Get logprob of actions, entropy of intra-option policies, value of selected option, termination probability of option on arrival"""
        x = self.features(xs)
        qs = self.critic(x)
        logprobs = torch.log_softmax(
            batched_index(options, self.actor(x).reshape(-1, self.num_options, self.num_actions)),
            -1,
        )
        probs = logprobs.exp()
        entropy = (-logprobs * probs).sum(-1)
        logprob = logprobs.gather(-1, actions.unsqueeze(-1)).squeeze(-1)
        q_sw = batched_index(options, qs)
        betas = torch.sigmoid(batched_index(options_on_arrival, self.termination(x)))
        return logprob, entropy, q_sw, betas


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
            monitor_gym=False,
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
    envs = envpool.make(
        args.env_id,
        env_type="gym",
        num_envs=args.num_envs,
        episodic_life=True,
        reward_clip=True,
        seed=args.seed,
    )
    envs.num_envs = args.num_envs
    envs.single_action_space = envs.action_space
    envs.single_observation_space = envs.observation_space
    envs = RecordEpisodeStatistics(envs)
    assert isinstance(envs.action_space, gym.spaces.Discrete), "only discrete action space is supported"

    agent = Agent(envs, args).to(device)
    agent = torch.jit.script(agent)
    optimizer = optim.Adam(agent.parameters(), args.learning_rate, eps=1e-5)

    # ALGO Logic: Storage setup
    obs = torch.zeros(
        (args.num_steps, args.num_envs) + envs.single_observation_space.shape,
        device=device,
    )
    actions = torch.zeros(
        (args.num_steps, args.num_envs) + envs.single_action_space.shape,
        dtype=torch.int64,
        device=device,
    )
    options_buffer = torch.zeros((args.num_steps + 1, args.num_envs), dtype=torch.int64, device=device)
    options = options_buffer[1:]
    options_on_arrival = options_buffer[:-1]
    betas_on_arrival = torch.zeros((args.num_steps, args.num_envs), device=device)
    rewards = torch.zeros((args.num_steps, args.num_envs), device=device)
    dones = torch.zeros((args.num_steps + 1, args.num_envs), device=device)
    qvalues = torch.zeros((args.num_steps, args.num_envs, args.num_options), device=device)  # for each option
    vvalues = torch.zeros((args.num_steps, args.num_envs), device=device)  # eps-greedy
    terminations = torch.zeros((args.num_steps, args.num_envs), device=device)
    avg_returns = deque(maxlen=20)

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()
    current_option = options_on_arrival[0]
    next_obs = torch.tensor(envs.reset(), device=device)
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
                (
                    (terminations[step], next_option, actions[step]),
                    betas_on_arrival[step],
                    (qvalues[step], vvalues[step]),
                ) = agent.step(next_obs, next_option, next_done)
                options[step] = next_option

            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, reward, done, info = envs.step(actions[step].cpu().numpy())
            rewards[step] = torch.tensor(reward, device=device)
            next_obs, next_done = torch.tensor(next_obs, device=device), torch.tensor(done, device=device, dtype=torch.float32)

            for idx, d in enumerate(done):
                if d and info["lives"][idx] == 0:
                    print(f"global_step={global_step}, episodic_return={info['r'][idx]}")
                    avg_returns.append(info["r"][idx])
                    writer.add_scalar(
                        "charts/avg_episodic_return",
                        np.average(avg_returns),
                        global_step,
                    )
                    writer.add_scalar("charts/episodic_return", info["r"][idx], global_step)
                    writer.add_scalar("charts/episodic_length", info["l"][idx], global_step)

        # bootstrap value if not done
        with torch.no_grad():
            dones[-1] = next_done
            if args.delib_cost_in_reward:  # Switching cost if not forced to terminate
                rewards -= args.delib_cost * (terminations * (1.0 - dones[:-1]))
            q_tm1_t = torch.cat(
                (
                    batched_index(options, qvalues),
                    agent.get_bsv(next_obs, next_option).unsqueeze(0),
                ),
                0,
            )
            # Lambda=1 simplifies to discounted return
            advantages, returns = gae(q_tm1_t, rewards, (1.0 - dones[1:]) * args.gamma, 1.0)
            if args.norm_adv:
                advantages = normalize(advantages)
            termination_advantage = batched_index(options_on_arrival, qvalues) - vvalues + args.delib_cost

        # unclipped a2oc loss
        lp, ent, new_q_sw, betas = agent.unroll(
            obs.reshape((-1,) + envs.single_observation_space.shape),
            options_on_arrival.reshape(-1),
            options.reshape(-1),
            actions.reshape((-1,) + envs.single_action_space.shape),
        )
        pg_loss = (-lp * advantages.reshape(-1)).mean()
        entropy_loss = -ent.mean()
        termination_loss = (betas * termination_advantage.reshape(-1)).mean()
        v_loss = F.mse_loss(new_q_sw, returns.reshape(-1))

        loss = pg_loss + args.ent_coef * entropy_loss + args.vf_coef * v_loss + args.term_coef * termination_loss

        optimizer.zero_grad(True)
        loss.backward()
        grad_norm = nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
        optimizer.step()

        y_pred, y_true = q_tm1_t[:-1].cpu().numpy(), returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/termination_loss", termination_loss.item(), global_step)
        writer.add_scalar("losses/termination_frequency", terminations.mean().item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)
        writer.add_scalar("losses/max_grad_norm", grad_norm, global_step)
        print("SPS:", int(global_step / (time.time() - start_time)))
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

    envs.close()
    writer.close()
