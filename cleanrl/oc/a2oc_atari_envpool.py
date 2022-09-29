# Implementation of Advantage Option Critic (A2OC in paper, but no asynchrony)
# Reference: https://github.com/jeanharb/a2oc_delib
# Paper: https://arxiv.org/pdf/1709.04571.pdf
"""More notes

Gradient clipping norm is 40...but that's for varied-length summed losses (instead of mean). It's also "global"
They use variable length asynchronous rollouts (5-30, whenever option terminates or episode ends)

"""

import argparse
import os
import random
import time
from collections import deque
from distutils.util import strtobool
from typing import Final

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
    parser.add_argument("--env-id", type=str, default="Pong-v5",
                        help="the id of the environment")
    parser.add_argument("--total-timesteps", type=int, default=int(8e7),
                        help="total timesteps of the experiments")
    parser.add_argument("--learning-rate", type=float, default=0.0007,
                        help="the learning rate of the optimizer")
    parser.add_argument("--num-envs", type=int, default=16,
                        help="the number of parallel game environments")
    parser.add_argument("--num-steps", type=int, default=5,
                        help="the number of steps to run in each environment per policy rollout")
    parser.add_argument("--anneal-lr", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
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
    parser.add_argument("--delib-cost", type=float, default=0.,
                        help="cost for terminating an option. added to reward")
    parser.add_argument("--term-reg", type=float, default=0.,
                        help="regularizer added to termination advantage to discourage terminations")
    parser.add_argument("--vf-coef", type=float, default=0.5,
                        help="coefficient of the value function")
    parser.add_argument("--num-options", type=int, default=8,
                        help="the number of options available")
    parser.add_argument("--max-grad-norm", type=float, default=0.5,
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
        # get if the env has lives
        self.has_lives = False
        env.reset()
        info = env.step(np.zeros(self.num_envs, dtype=int))[-1]
        if info["lives"].sum() > 0:
            self.has_lives = True
            print("env has lives")

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
        all_lives_exhausted = infos["lives"] == 0
        if self.has_lives:
            self.episode_returns *= 1 - all_lives_exhausted
            self.episode_lengths *= 1 - all_lives_exhausted
        else:
            self.episode_returns *= 1 - dones
            self.episode_lengths *= 1 - dones
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
    return s_flat.view(t.shape[:dim] + t.shape[dim + 1:])


def layer_init(
        layer, std: float = nn.init.calculate_gain("relu"), bias_const: float = 0.0
):
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
        return self.network(x / 255.0)

    def sample_option(
            self,
            option_on_arrival: Tensor,
            qs: Tensor,
            terminations: Tensor,
            current_epsilon: float = 0.1,
    ) -> Tensor:
        candidate_option = torch.where(
            torch.rand_like(option_on_arrival, dtype=torch.float32) < current_epsilon,
            torch.randint_like(option_on_arrival, 0, self.num_options),
            qs.argmax(-1),
        )
        return torch.where(terminations > 0, candidate_option, option_on_arrival)

    def step(
            self,
            x: Tensor,
            option_on_arrival: Tensor,
            is_done: Tensor,
            current_epsilon: float = 0.1,
    ):
        """Basic single step on receiving new observation.

        Determine terminations, pick new options, get actions and values"""
        x = self.features(x)
        q = self.critic(x)  # Q for each option
        beta_w = torch.sigmoid(batched_index(option_on_arrival, self.termination(x)))  # Termination prob of current option
        term_w = torch.bernoulli(beta_w)  # Sample terminations
        option = self.sample_option(option_on_arrival, q, term_w + is_done, current_epsilon)  # also terminate on episode end
        pi_w = torch.softmax(
            batched_index(
                option, self.actor(x).reshape(-1, self.num_options, self.num_actions)
            ),
            -1,
        )
        a = torch.multinomial(pi_w, 1)  # action under new option
        return (term_w, option, a.squeeze(-1)), beta_w, q

    def bootstrap_utility(
            self,
            next_x: Tensor,
            option_on_arrival: Tensor,
            is_done: Tensor,
            current_epsilon: float = 0.1,
    ):
        """Single step at end of rollout. Value of next state is function of termination probs"""
        x = self.features(next_x)
        q = self.critic(x)
        beta_w = torch.sigmoid(
            batched_index(option_on_arrival, self.termination(x)))  # Termination prob of current option
        v = q.max(-1)[0] * (1 - current_epsilon) + q.mean(-1) * current_epsilon
        u = (1 - beta_w) * batched_index(option_on_arrival, q) + beta_w * v  # utility on arrival
        return torch.where(is_done > 0, 0, u)  # bootstrap is 0 if next step is a start state

    def unroll(self, xs: Tensor, options_on_arrival: Tensor, options: Tensor, actions: Tensor):
        """Unroll with gradients. Get logprob of actions, entropy of intra-option policies, value of selected option, termination probability of option on arrival"""
        T, B = options.shape
        xs = xs.view((T * B,) + xs.shape[2:])
        x = self.features(xs)
        qs = self.critic(x)
        logprobs = torch.log_softmax(
            batched_index(
                options.flatten(),
                self.actor(x).reshape(-1, self.num_options, self.num_actions),
            ),
            -1,
        )
        probs = logprobs.exp()
        entropy = (-logprobs * probs).sum(-1)
        logprob = logprobs.gather(-1, actions.view(-1, 1)).squeeze(-1)
        q_sw = batched_index(options.flatten(), qs)
        betas = torch.sigmoid(
            batched_index(options_on_arrival.flatten(), self.termination(x))
        )
        return logprob, entropy, q_sw, betas


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
        "|param|value|\n|-|-|\n%s"
        % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
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
    assert isinstance(
        envs.action_space, gym.spaces.Discrete
    ), "only discrete action space is supported"

    agent = Agent(envs, args).to(device)
    optimizer = optim.RMSprop(
        agent.parameters(), args.learning_rate, alpha=0.99, eps=0.1
    )

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
    options_buffer = torch.zeros(
        (args.num_steps + 1, args.num_envs), dtype=torch.int64, device=device
    )
    options = options_buffer[1:]
    options_on_arrival = options_buffer[:-1]
    betas_on_arrival = torch.zeros((args.num_steps, args.num_envs), device=device)
    rewards = torch.zeros((args.num_steps, args.num_envs), device=device)
    dones = torch.zeros((args.num_steps, args.num_envs), device=device)
    values = torch.zeros((args.num_steps, args.num_envs, args.num_options), device=device)  # for each option
    avg_returns = deque(maxlen=20)

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()
    current_option = options_on_arrival[0]
    next_obs = torch.tensor(envs.reset(), device=device)
    next_done = torch.ones(args.num_envs, device=device)
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
                (
                    (terminations, options[step], actions[step]),
                    betas_on_arrival[step],
                    values[step],
                ) = agent.step(
                    next_obs, options_on_arrival[step], next_done, args.option_epsilon
                )

            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, reward, done, info = envs.step(actions[step].cpu().numpy())
            rewards[step] = torch.tensor(reward, device=device)
            rewards[step, terminations > 0] -= (args.delib_cost)  # Subtract deliberation cost from rewards if we chose to terminate
            next_obs, next_done = torch.tensor(next_obs, device=device), torch.tensor(
                done, device=device, dtype=torch.float32
            )

            for idx, d in enumerate(done):
                if d and info["lives"][idx] == 0:
                    print(f"global_step={global_step}, episodic_return={info['r'][idx]}")
                    avg_returns.append(info["r"][idx])
                    writer.add_scalar("charts/avg_episodic_return", np.average(avg_returns), global_step, )
                    writer.add_scalar("charts/episodic_return", info["r"][idx], global_step)
                    writer.add_scalar("charts/episodic_length", info["l"][idx], global_step)

        # bootstrap value if not done
        with torch.no_grad():
            q_sw = batched_index(options, values)  # Select baseline value based on sampled option
            v_s = values.mean(-1)
            next_value = agent.bootstrap_utility(next_obs, options[-1], next_done, args.option_epsilon)
            # Leave in GAE format for max compatibility
            advantages = torch.zeros_like(rewards)
            lastgaelam = 0
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = q_sw[t + 1]
                delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - q_sw[t]
                advantages[t] = lastgaelam = (delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam)
            returns = advantages + q_sw

        # unclipped a2oc loss
        lp, ent, new_q_sw, betas = agent.unroll(obs, options_on_arrival, options, actions)
        pg_loss = (-lp * advantages.flatten()).mean()
        entropy_loss = (-ent).mean()
        beta_loss = (betas * ((q_sw.flatten() - v_s.flatten()) + args.delib_cost)).mean()
        v_loss = F.mse_loss(new_q_sw, returns.flatten())

        loss = pg_loss + args.ent_coef * entropy_loss + v_loss * args.vf_coef + beta_loss

        optimizer.zero_grad(True)
        loss.backward()
        grad_norm = nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
        optimizer.step()
        options_on_arrival[0] = options[-1]  # set new option on arrival

        y_pred, y_true = q_sw.cpu().numpy(), returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/termination_loss", beta_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)
        print("SPS:", int(global_step / (time.time() - start_time)))
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

    envs.close()
    writer.close()
