# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/sac/#sac_continuous_actionpy
import argparse
import math
import os
import random
import time
import typing as t
from distutils.util import strtobool
from types import SimpleNamespace

import gym
import numpy as np
import pybullet_envs  # noqa
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from stable_baselines3.common.buffers import ReplayBuffer
from torch.utils.tensorboard import SummaryWriter


PAPER_N_QUANTILES_TO_DROP = {
    "Hopper-v3": 5,
    "Swimmer-v3": 2,
    "HalfCheetah-v3": 0,
    "Ant-v3": 2,
    "Walker2d-v3": 2,
    "Humanoid-v3": 2,
    "HopperBulletEnv-v0": 5,
    "SwimmerBulletEnv-v0": 2,
    "HalfCheetahBulletEnv-v0": 0,
    "AntBulletEnv-v0": 2,
    "Walker2dBulletEnv-v0": 2,
    "HumanoidBulletEnv-v0": 2,
}


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
    parser.add_argument("--capture-video",
        type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="weather to capture videos of the agent performances (check out `videos` folder)")

    # Algorithm specific arguments
    parser.add_argument("--env-id", type=str, default="HopperBulletEnv-v0",
        help="the id of the environment")
    parser.add_argument("--total-timesteps", type=int, default=5000000,
        help="total timesteps of the experiments")
    parser.add_argument("--buffer-size", type=int, default=int(1e6),
        help="the replay memory buffer size")
    parser.add_argument("--gamma", type=float, default=0.99,
        help="the discount factor gamma")
    parser.add_argument("--tau", type=float, default=0.005,
        help="target smoothing coefficient (default: 0.005)")
    parser.add_argument("--batch-size", type=int, default=256,
        help="the batch size of sample from the reply memory")
    parser.add_argument("--n-quantiles", type=int, default=25,
        help="the number of quantiles used for each Q Network")
    parser.add_argument("--n-critics", type=int, default=5,
        help="the number of Q Networks")
    parser.add_argument("--use-paper-n-quantiles-to-drop",
        type=lambda x: bool(strtobool(x)),
        default=True,
        help="number of quantiles to drop")
    parser.add_argument("--n-quantiles-to-drop", type=int, default=2,
        help="number of quantiles to drop")
    parser.add_argument("--learning-starts", type=int, default=5e3,
        help="timestep to start learning")
    parser.add_argument("--actor_adam_lr", type=float, default=3e-4,
        help="the learning rate of the policy network optimizer")
    parser.add_argument("--critic_adam_lr", type=float, default=3e-4,
        help="the learning rate of the Q network network optimizer")
    parser.add_argument("--alpha_adam_lr", type=float, default=3e-4,
        help="the learning rate to tune target entropy")
    args = parser.parse_args()
    # fmt: on
    return args


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


def make_env(env_id, seed, idx, capture_video, run_name):
    def thunk():
        env = gym.make(env_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        if capture_video:
            if idx == 0:
                env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        env.seed(seed)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        return env

    return thunk


def quantile_huber_loss(
    quantiles: torch.Tensor,
    samples: torch.Tensor,
) -> torch.Tensor:
    pairwise_delta = (
        samples[:, None, None, :] - quantiles[:, :, :, None]  # type: ignore
    )  # batch x nets x quantiles x samples
    abs_pairwise_delta = torch.abs(pairwise_delta)
    huber_loss = torch.where(
        abs_pairwise_delta > 1, abs_pairwise_delta - 0.5, pairwise_delta**2 * 0.5
    )

    n_quantiles = quantiles.shape[2]
    taus = (
        torch.arange(n_quantiles, device=quantiles.device).float().unsqueeze(0)
        / n_quantiles
        + 1 / 2 / n_quantiles
    )
    elementwise_loss = (
        torch.abs(taus[:, None, :, None] - (pairwise_delta < 0).float())  # type: ignore
        * huber_loss
    )
    return elementwise_loss.mean()


# ALGO LOGIC: initialize agent here:
class QuantileCritics(nn.Module):
    def __init__(
        self, envs: gym.vector.VectorEnv, n_quantiles: int, n_critics: int
    ) -> None:
        super().__init__()
        state_dim = np.prod(envs.single_observation_space.shape)
        action_dim = np.prod(envs.single_action_space.shape)
        self.n_critics = n_critics
        self.n_quantiles = n_quantiles
        self.n_total_quantiles = n_quantiles * n_critics

        def make_critic() -> nn.Module:
            return nn.Sequential(
                layer_init(nn.Linear(state_dim + action_dim, 512)),
                nn.ReLU(),
                layer_init(nn.Linear(512, 512)),
                nn.ReLU(),
                layer_init(nn.Linear(512, n_quantiles)),
            )

        self.critics = nn.ModuleList([make_critic() for _ in range(n_critics)])

    def forward(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        state_actions = torch.cat([states, actions], dim=-1)
        return torch.stack(
            tuple(critic(state_actions) for critic in self.critics),
            dim=1,
        )


LOG_STD_MAX = 2
LOG_STD_MIN = -5


class Actor(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.fc1 = nn.Linear(np.array(env.single_observation_space.shape).prod(), 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc_mean = nn.Linear(256, np.prod(env.single_action_space.shape))
        self.fc_logstd = nn.Linear(256, np.prod(env.single_action_space.shape))
        # action rescaling
        self.register_buffer(
            "action_scale",
            torch.FloatTensor((env.action_space.high - env.action_space.low) / 2.0),
        )
        self.register_buffer(
            "action_bias",
            torch.FloatTensor((env.action_space.high + env.action_space.low) / 2.0),
        )

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mean = self.fc_mean(x)
        log_std = self.fc_logstd(x)
        log_std = torch.tanh(log_std)
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (
            log_std + 1
        )  # From SpinUp / Denis Yarats

        return mean, log_std

    def get_action(self, x):
        mean, log_std = self(x)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        # Enforcing Action Bound
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean


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
    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, args.seed, 0, args.capture_video, run_name)]
    )
    assert isinstance(
        envs.single_action_space, gym.spaces.Box
    ), "only continuous action space is supported"

    max_action = float(envs.single_action_space.high[0])

    if args.use_paper_n_quantiles_to_drop and args.env_id in PAPER_N_QUANTILES_TO_DROP:
        args.n_quantiles_to_drop = PAPER_N_QUANTILES_TO_DROP[args.env_id]
        print(
            f"Using paper n_quantiles_to_drop: {args.n_quantiles_to_drop} for env: {args.env_id}"
        )
    actor = Actor(envs).to(device)
    critics = QuantileCritics(envs, args.n_quantiles, args.n_critics).to(device)
    args.n_top_quantiles_to_drop = args.n_quantiles_to_drop * critics.n_critics
    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=args.actor_adam_lr)
    critic_optimizer = torch.optim.Adam(critics.parameters(), lr=args.critic_adam_lr)

    target_critics = QuantileCritics(envs, args.n_quantiles, args.n_critics).to(device)
    target_critics.load_state_dict(critics.state_dict())
    target_critics.requires_grad_(False)

    # Automatic entropy tuning
    target_entropy = -torch.prod(
        torch.Tensor(envs.single_action_space.shape).to(device)
    ).item()
    log_alpha = torch.zeros(1, requires_grad=True, device=device)
    alpha_optimizer = optim.Adam([log_alpha], lr=args.alpha_adam_lr)

    envs.single_observation_space.dtype = np.float32
    rb = ReplayBuffer(
        args.buffer_size,
        envs.single_observation_space,
        envs.single_action_space,
        device,
        handle_timeout_termination=True,
    )
    start_time = time.time()

    # TRY NOT TO MODIFY: start the game
    obs = envs.reset()
    for global_step in range(args.total_timesteps):
        # ALGO LOGIC: put action logic here
        if global_step < args.learning_starts:
            actions = np.array(
                [envs.single_action_space.sample() for _ in range(envs.num_envs)]
            )
        else:
            actions, _, _ = actor.get_action(torch.Tensor(obs).to(device))
            actions = actions.detach().cpu().numpy()

        # TRY NOT TO MODIFY: execute the game and log data.
        next_obs, rewards, dones, infos = envs.step(actions)

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        for info in infos:
            if "episode" in info.keys():
                print(
                    f"global_step={global_step}, episodic_return={info['episode']['r']}"
                )
                writer.add_scalar(
                    "charts/episodic_return", info["episode"]["r"], global_step
                )
                writer.add_scalar(
                    "charts/episodic_length", info["episode"]["l"], global_step
                )
                break

        # TRY NOT TO MODIFY: save data to reply buffer; handle `terminal_observation`
        real_next_obs = next_obs.copy()
        for idx, d in enumerate(dones):
            if d:
                real_next_obs[idx] = infos[idx]["terminal_observation"]
        rb.add(obs, real_next_obs, actions, rewards, dones, infos)

        # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
        obs = next_obs

        # ALGO LOGIC: training.
        if global_step > args.learning_starts:
            sample_batch = rb.sample(args.batch_size)
            alpha = torch.exp(log_alpha)

            batch_size = sample_batch.rewards.size(0)
            # --- Q loss ---
            with torch.no_grad():
                # get policy action
                sampled_next_actions, next_log_pi, _ = actor.get_action(
                    sample_batch.next_observations
                )

                # compute and cut quantiles at the next state
                # batch x nets x quantiles
                next_z = target_critics(
                    sample_batch.next_observations, sampled_next_actions
                )
                sorted_z, _ = torch.sort(next_z.reshape(batch_size, -1))
                sorted_z_part = sorted_z[
                    :, : critics.n_total_quantiles - args.n_top_quantiles_to_drop
                ]

                # compute target
                target = sample_batch.rewards + (
                    1 - sample_batch.dones
                ) * args.gamma * (sorted_z_part - alpha * next_log_pi)
                assert target.shape == (
                    batch_size,
                    critics.n_total_quantiles - args.n_top_quantiles_to_drop,
                ), target.shape

            cur_z = critics(sample_batch.observations, sample_batch.actions)
            # --- Critic update ---
            assert cur_z.shape == (
                batch_size,
                critics.n_critics,
                critics.n_quantiles,
            ), cur_z.shape
            critic_loss = quantile_huber_loss(cur_z, target)
            critic_optimizer.zero_grad()
            critic_loss.backward()
            critic_optimizer.step()
            # --- Policy update ---
            sampled_actions, log_pi, _ = actor.get_action(sample_batch.observations)
            actor_loss = (
                alpha * log_pi
                - critics(sample_batch.observations, sampled_actions)
                .mean(2)
                .mean(1, keepdim=True)
            ).mean()
            actor_optimizer.zero_grad()
            actor_loss.backward()
            actor_optimizer.step()
            # --- Alpha update ---
            alpha_loss = -log_alpha * (log_pi + target_entropy).detach().mean()
            alpha_optimizer.zero_grad()
            alpha_loss.backward()
            alpha_optimizer.step()
            for param, target_param in zip(
                critics.parameters(), target_critics.parameters()
            ):
                target_param.data.copy_(
                    args.tau * param.data + (1 - args.tau) * target_param.data
                )

            if global_step % 100 == 0:
                writer.add_scalar("losses/cur_z", cur_z.mean().item(), global_step)
                writer.add_scalar("losses/critic_loss", critic_loss.item(), global_step)
                writer.add_scalar("losses/actor_loss", actor_loss.item(), global_step)
                writer.add_scalar("losses/alpha", alpha, global_step)
                print("SPS:", int(global_step / (time.time() - start_time)))
                writer.add_scalar(
                    "charts/SPS",
                    int(global_step / (time.time() - start_time)),
                    global_step,
                )
                writer.add_scalar("losses/alpha_loss", alpha_loss.item(), global_step)

    envs.close()
    writer.close()
