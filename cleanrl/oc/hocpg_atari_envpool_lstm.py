# Hierarchical Option-Critic Policy Gradients (i.e., multi-level option critic with shared-weight derivation)
# Paper: https://arxiv.org/abs/1912.13408 (weight sharing derivation)
# Reference repo: https://github.com/mattriemer/ocpg (private)
# 3 levels with 3 options means 9 total options. Top level chooses among 3 options.
# Each of those chooses among *its own 3* options. Each of those has termination and action policies
# Maintain a critic Q(s, w_1) and Q(s, w_1, w_2) (V(s) computed, Q(s, w_1, w_2, a) sampled)
# TODO: INCOMPLETE
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

Tensor = torch.Tensor
LSTMState = Tuple[torch.Tensor, torch.Tensor]


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
    parser.add_argument("--learning-rate", type=float, default=2.5e-4,
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
    parser.add_argument("--ent-coef", type=float, default=0.01,
                        help="coefficient of the entropy at each level")
    parser.add_argument("--delib-cost", type=float, default=0.02,
                        help="cost for terminating an option. subtracted from reward, added to termination advantage")
    parser.add_argument("--delib-cost-in-reward", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
                        help="if true, subtract cost from immediate reward (in addition to adding to termination advantage")
    parser.add_argument("--vf-coef", type=float, default=0.5,
                        help="coefficient of the value function")
    parser.add_argument("--term-coef", type=float, default=1.,
                        help="coefficient of the value function")
    parser.add_argument("--num-options", type=int, default=3,
                        help="the number of options available at each level")
    parser.add_argument("--num-levels", type=int, default=3,
                        help="the number of levels in hierarchy (2 means option-critic)")
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


def layer_init(layer: nn.Module, std: float = nn.init.calculate_gain("relu"), bias_const: float = 0.0):
    for n, p in layer.named_parameters():
        if "bias" in n:
            torch.nn.init.constant_(p, bias_const)
        elif "weight" in n:
            torch.nn.init.orthogonal_(p, std)
    return layer


class StepOutput(NamedTuple):
    termination: Tensor  # t
    option: Tensor  # w
    action: Tensor  # a
    betas: Tensor  # beta(s, w_)
    option_logprobs: Tensor  # lp(w|s)
    logprobs: Tensor  # lp(a|s, w)
    qs: Tensor  # Q(s, .)
    v: Tensor  # V(s)


class Agent(nn.Module):
    option_levels: Final[int]
    num_options: Final[int]
    num_actions: Final[int]

    def __init__(self, envs, args):
        super().__init__()
        # 4-layer CNN with 2x2 maxpools and relu
        self.network = nn.Sequential(
            layer_init(nn.Conv2d(1, 32, 5, stride=1, padding=2)),
            nn.MaxPool2d(2, 2),
            nn.ReLU(inplace=True),
            layer_init(nn.Conv2d(32, 32, 5, stride=1, padding=1)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            layer_init(nn.Conv2d(32, 64, 4, stride=1, padding=1)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            layer_init(nn.Conv2d(64, 64, 3, stride=1, padding=1)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Flatten(),
        )
        self.lstm = layer_init(nn.LSTMCell(1024, 512), 1.0)
        self.num_actions = envs.single_action_space.n
        self.num_options = args.num_options
        self.option_levels = args.num_levels - 1  # Bottom level is picking actions
        unique_options = int(self.num_options**self.option_levels)  # Unique combinations of various levels
        self.actor = nn.Sequential(
            layer_init(nn.Linear(512, int(self.num_actions * unique_options)), std=0.01),
            nn.Unflatten(-1, (self.num_options,) * self.option_levels + (self.num_actions,)),
        )
        self.option_actor = nn.Sequential(
            layer_init(nn.Linear(512, unique_options), std=0.01), nn.Unflatten(-1, (self.num_options,) * self.option_levels)
        )
        self.critic = nn.Sequential(
            layer_init(nn.Linear(512, unique_options), std=1.0), nn.Unflatten(-1, (self.num_options,) * self.option_levels)
        )
        self.termination = nn.Sequential(
            layer_init(nn.Linear(512, unique_options), std=0.01), nn.Unflatten(-1, (self.num_options,) * self.option_levels)
        )

    def get_state(self, x: Tensor, s: LSTMState) -> LSTMState:
        return self.lstm(self.network(x / 255.0), s)

    @torch.jit.export
    def step(self, x: Tensor, s: LSTMState, option_on_arrival: Tensor, is_done: Tensor) -> Tuple[StepOutput, LSTMState]:
        reset_state = 1.0 - is_done.unsqueeze(-1)
        s = self.get_state(x, (s[0] * reset_state, s[1] * reset_state))
        x = s[0]
        # Termination probs and sample. Sample terminations from lowest level to highest
        betas = torch.sigmoid(self.termination(x))
        for l in reversed(range(self.option_levels)):
            beta_w = batched_index(option_on_arrival[..., l], betas)
            term_w = torch.bernoulli(beta_w)
        # Option logprobs and probs. Sample where terminal or episode start
        option_logprobs = torch.log_softmax(self.option_actor(x), -1)
        option_probs = option_logprobs.exp()
        option = torch.where((term_w + is_done) > 0, torch.multinomial(option_probs, 1).squeeze(-1), option_on_arrival)
        option_logprob = option_logprobs.gather(-1, option.unsqueeze(-1)).squeeze(-1)
        # Sample action using new option
        action_logprobs = torch.log_softmax(batched_index(option, self.actor(x)), -1)
        action_probs = action_logprobs.exp()
        action = torch.multinomial(action_probs, 1).squeeze(-1)
        action_logprob = action_logprobs.gather(-1, action.unsqueeze(-1)).squeeze(-1)
        # Value of each option, average value of ocpg
        qs = self.critic(x)
        v = (qs * option_probs).sum(-1)
        return StepOutput(term_w, option, action, beta_w, option_logprob, action_logprob, qs, v), s

    @torch.jit.export
    def unroll(
        self, xs: Tensor, initial_s: LSTMState, options_on_arrival: Tensor, options: Tensor, actions: Tensor, is_dones: Tensor
    ):
        reset_state = 1.0 - is_dones.unsqueeze(-1)
        T, B = is_dones.shape
        fs = []
        s = initial_s
        # Batch process through cnn
        xs = self.network(xs.view((T * B,) + xs.shape[2:]) / 255.0).view(T, B, -1)
        # loop through lstm with resets where episode terminated
        for x, r in zip(xs, reset_state):
            s = self.lstm(x, (s[0] * r, s[1] * r))
            fs.append(s[0])
        fs = torch.stack(fs, 0)  # (T, B, -1)
        # Probability of terminating previous option in each step
        prev_termprobs = torch.sigmoid(batched_index(options_on_arrival, self.termination(fs)))
        # Probability of selecting chosen option in each step
        all_option_logprobs = torch.log_softmax(self.option_actor(fs), -1)
        option_logprobs = all_option_logprobs.gather(-1, options.unsqueeze(-1)).squeeze(-1)
        option_entropy = (-all_option_logprobs * all_option_logprobs.exp()).sum(-1)
        # Probability of selecting chosen action in each step
        action_logprobs = torch.log_softmax(batched_index(options, self.actor(fs)), -1)
        action_logprob = action_logprobs.gather(-1, actions.unsqueeze(-1)).squeeze(-1)
        action_entropy = (-action_logprobs * action_logprobs.exp()).sum(-1)
        # Value of selected option
        q_sw = batched_index(options, self.critic(fs))
        return (prev_termprobs, option_logprobs, option_entropy), (action_logprob, action_entropy), q_sw

    @torch.jit.export
    def get_bsv(
        self, next_x: Tensor, next_s: LSTMState, option_on_arrival: Tensor, next_is_done: Tensor
    ) -> Tuple[Tensor, Tensor]:
        """Bootstrap intra-option value using U(s', w), inter-option value using V(s')"""
        next_reset = 1.0 - next_is_done.unsqueeze(-1)
        x = self.get_state(next_x, (next_s[0] * next_reset, next_s[1] * next_reset))[0]
        qs = self.critic(x)
        option_probs = torch.softmax(self.option_actor(x), -1)
        v = (qs * option_probs).sum(-1)
        q_w_tm1 = batched_index(option_on_arrival, qs)
        betas = torch.sigmoid(batched_index(option_on_arrival, self.termination(x)))
        u = (1.0 - betas) * q_w_tm1 + betas * v
        return u, v


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
        stack_num=1,
    )
    envs.num_envs = args.num_envs
    envs.single_action_space = envs.action_space
    envs.single_observation_space = envs.observation_space
    envs = RecordEpisodeStatistics(envs)
    assert isinstance(envs.action_space, gym.spaces.Discrete), "only discrete action space is supported"

    agent = Agent(envs, args).to(device)
    # agent = torch.jit.script(agent)
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
    options_buffer = torch.zeros((args.num_steps + 1, args.num_envs, args.num_levels - 1), dtype=torch.int64, device=device)
    options = options_buffer[1:]
    options_on_arrival = options_buffer[:-1]
    betas_on_arrival = torch.zeros((args.num_steps, args.num_envs, args.num_levels - 1), device=device)
    rewards = torch.zeros((args.num_steps, args.num_envs), device=device)
    dones = torch.zeros((args.num_steps + 1, args.num_envs), device=device)
    qvalues = torch.zeros((args.num_steps, args.num_envs, args.num_options), device=device)  # for each option
    vvalues = torch.zeros((args.num_steps + 1, args.num_envs), device=device)  # Average
    terminations = torch.zeros((args.num_steps, args.num_envs, args.num_levels - 1), device=device)
    avg_returns = deque(maxlen=20)

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()
    current_option = options_on_arrival[0]
    next_obs = torch.tensor(envs.reset(), device=device)
    next_done = torch.ones(args.num_envs, device=device)
    next_option = torch.zeros((args.num_envs, args.num_levels - 1), device=device, dtype=torch.int64)
    next_lstm_state = (
        torch.zeros(args.num_envs, agent.lstm.hidden_size, device=device),
        torch.zeros(args.num_envs, agent.lstm.hidden_size, device=device),
    )  # hidden and cell states (see https://youtu.be/8HyCNIVRbSU)
    num_updates = args.total_timesteps // args.batch_size

    for update in range(1, num_updates + 1):
        initial_lstm_state = (next_lstm_state[0].clone(), next_lstm_state[1].clone())
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
                    terminations[step],
                    next_option,
                    actions[step],
                    betas_on_arrival[step],
                    _,
                    _,
                    qvalues[step],
                    vvalues[step],
                ), next_lstm_state = agent.step(next_obs, next_lstm_state, next_option, next_done)
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
            u, vvalues[-1] = agent.get_bsv(next_obs, next_lstm_state, next_option, next_done)
            q_tm1_t = torch.cat((batched_index(options, qvalues), u.unsqueeze(0)), 0)
            # Lambda=1 simplifies to discounted return
            gamma_t = (1.0 - dones[1:]) * args.gamma
            advantages, returns = gae(q_tm1_t, rewards, gamma_t, args.gae_lambda)
            option_advantages = gae(vvalues, rewards, gamma_t, args.gae_lambda)[0]
            if args.norm_adv:
                advantages = normalize(advantages)
                option_advantages = normalize(option_advantages)
            termination_advantage = (batched_index(options_on_arrival, qvalues) - vvalues[:-1] + args.delib_cost) * (
                1.0 - dones[:-1]
            )

        # unclipped a2oc loss
        (betas, option_lp, option_ent), (lp, ent), new_q_sw = agent.unroll(
            obs, initial_lstm_state, options_on_arrival, options, actions, dones[:-1]
        )
        pg_loss = (-lp * advantages).mean()
        entropy_loss = -ent.mean()
        option_pg_loss = (-option_lp * option_advantages).mean()
        option_entropy_loss = -option_ent.mean()
        termination_loss = (betas * termination_advantage).mean()
        v_loss = F.mse_loss(new_q_sw, returns)

        loss = (
            pg_loss
            + args.ent_coef * entropy_loss
            + option_pg_loss
            + args.option_ent_coef * option_entropy_loss
            + args.vf_coef * v_loss
            + args.term_coef * termination_loss
        )

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
        writer.add_scalar("losses/option_policy_loss", option_pg_loss.item(), global_step)
        writer.add_scalar("losses/termination_loss", termination_loss.item(), global_step)
        writer.add_scalar("losses/termination_frequency", terminations.mean().item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/option_entropy", option_entropy_loss.item(), global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)
        writer.add_scalar("losses/max_grad_norm", grad_norm, global_step)
        print("SPS:", int(global_step / (time.time() - start_time)))
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

    envs.close()
    writer.close()
