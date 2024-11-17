import os
import random
import time
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tyro
from stable_baselines3.common.buffers import ReplayBuffer
from torch.utils.tensorboard import SummaryWriter


@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "crossq"
    """the wandb's project name"""
    wandb_entity: str = None
    """the entity (team) of wandb's project"""
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""

    # Algorithm specific arguments
    env_id: str = "Hopper-v4"
    """the environment id of the task"""
    total_timesteps: int = 1_000_000
    """total timesteps of the experiments"""
    buffer_size: int = 1_000_000
    """the replay memory buffer size"""
    gamma: float = 0.99
    """the discount factor gamma"""
    batch_size: int = 256
    """the batch size of sample from the reply memory"""
    learning_starts: int = 5000
    """timestep to start learning"""
    policy_lr: float = 1e-3
    """the learning rate of the policy network optimizer"""
    q_lr: float = 1e-3
    """the learning rate of the Q network network optimizer"""
    policy_frequency: int = 3
    """the frequency of training policy (delayed)"""
    alpha: float = 0.2
    """Entropy regularization coefficient."""
    autotune: bool = True
    """automatic tuning of the entropy coefficient"""


# BatchRenorm implementation from https://github.com/danielpalen/stable-baselines3-contrib/blob/feat/crossq/sb3_contrib/common/torch_layers.py
class BatchRenorm(torch.jit.ScriptModule):
    """
    BatchRenorm Module (https://arxiv.org/abs/1702.03275).
    Adapted to Pytorch from sbx.sbx.common.jax_layers.BatchRenorm

    BatchRenorm is an improved version of vanilla BatchNorm. Contrary to BatchNorm,
    BatchRenorm uses the running statistics for normalizing the batches after a warmup phase.
    This makes it less prone to suffer from "outlier" batches that can happen
    during very long training runs and, therefore, is more robust during long training runs.

    During the warmup phase, it behaves exactly like a BatchNorm layer. After the warmup phase,
    the running statistics are used for normalization. The running statistics are updated during
    training mode. During evaluation mode, the running statistics are used for normalization but
    not updated.

    :param num_features: Number of features in the input tensor.
    :param eps: A value added to the variance for numerical stability.
    :param momentum: The value used for the ra_mean and ra_var computation.
    :param affine: A boolean value that when set to True, this module has learnable
            affine parameters. Default: True
    :param warmup_steps: Number of warum steps that are performed before the running statistics
            are used form normalization. During the warump phase, the batch statistics are used.
    """

    def __init__(
        self,
        num_features: int,
        eps: float = 0.001,
        momentum: float = 0.01,
        affine: bool = True,
        warmup_steps: int = 100_000,
    ):
        super().__init__()
        # Running average mean and variance
        self.register_buffer("ra_mean", torch.zeros(num_features, dtype=torch.float))
        self.register_buffer("ra_var", torch.ones(num_features, dtype=torch.float))
        self.register_buffer("steps", torch.tensor(0, dtype=torch.long))
        self.scale = torch.nn.Parameter(torch.ones(num_features, dtype=torch.float))
        self.bias = torch.nn.Parameter(torch.zeros(num_features, dtype=torch.float))

        self.affine = affine
        self.eps = eps
        self.step = 0
        self.momentum = momentum
        self.rmax = 3.0
        self.dmax = 5.0
        self.warmup_steps = warmup_steps

    def _check_input_dim(self, x: torch.Tensor) -> None:
        raise NotImplementedError()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Normalize the input tensor.

        :param x: Input tensor
        :return: Normalized tensor.
        """

        if self.training:
            batch_mean = x.mean(0)
            batch_var = x.var(0)
            batch_std = (batch_var + self.eps).sqrt()

            # Use batch statistics during initial warm up phase.
            # Note: in the original paper, after some warmup phase (batch norm phase of 5k steps)
            # the constraints are linearly relaxed to r_max/d_max over 40k steps
            # Here we only have a warmup phase
            if self.steps > self.warmup_steps:

                running_std = (self.ra_var + self.eps).sqrt()
                # scale
                r = (batch_std / running_std).detach()
                r = r.clamp(1 / self.rmax, self.rmax)
                # bias
                d = ((batch_mean - self.ra_mean) / running_std).detach()
                d = d.clamp(-self.dmax, self.dmax)

                # BatchNorm normalization, using minibatch stats and running average stats
                # Because we use _normalize, this is equivalent to
                # ((x - x_mean) / sigma) * r + d = ((x - x_mean) * r + d * sigma) / sigma
                # where sigma = sqrt(var)
                custom_mean = batch_mean - d * batch_var.sqrt() / r
                custom_var = batch_var / (r**2)

            else:
                custom_mean, custom_var = batch_mean, batch_var

            # Update Running Statistics
            self.ra_mean += self.momentum * (batch_mean.detach() - self.ra_mean)
            self.ra_var += self.momentum * (batch_var.detach() - self.ra_var)
            self.steps += 1

        else:
            custom_mean, custom_var = self.ra_mean, self.ra_var

        # Normalize
        x = (x - custom_mean[None]) / (custom_var[None] + self.eps).sqrt()

        if self.affine:
            x = self.scale * x + self.bias

        return x


class BatchRenorm1d(BatchRenorm):
    def _check_input_dim(self, x: torch.Tensor) -> None:
        if x.dim() == 1:
            raise ValueError(f"Expected 2D or 3D input (got {x.dim()}D input)")


def make_env(env_id, seed, idx, capture_video, run_name):
    def thunk():
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        else:
            env = gym.make(env_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env.action_space.seed(seed)
        return env

    return thunk


# ALGO LOGIC: initialize agent here:
class SoftQNetwork(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.bn1 = BatchRenorm1d(
            np.array(env.single_observation_space.shape).prod()
            + np.prod(env.single_action_space.shape)
        )
        self.fc1 = nn.Linear(
            np.array(env.single_observation_space.shape).prod()
            + np.prod(env.single_action_space.shape),
            2048,
        )
        self.bn2 = BatchRenorm1d(2048)
        self.fc2 = nn.Linear(2048, 2048)
        self.bn3 = BatchRenorm1d(2048)
        self.fc3 = nn.Linear(2048, 1)

    def forward(self, x, a, train=False):
        if train:
            self.train()
        else:
            self.eval()

        x = torch.cat([x, a], 1)
        x = F.relu(self.fc1(self.bn1(x)))
        x = F.relu(self.fc2(self.bn2(x)))
        x = self.fc3(self.bn3(x))
        return x


LOG_STD_MIN = -20
LOG_STD_MAX = 2


class Actor(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.bn1 = BatchRenorm1d(np.array(env.single_observation_space.shape).prod())
        self.fc1 = nn.Linear(np.array(env.single_observation_space.shape).prod(), 256)
        self.bn2 = BatchRenorm1d(256)
        self.fc2 = nn.Linear(256, 256)
        self.bn3 = BatchRenorm1d(256)
        self.fc_mean = nn.Linear(256, np.prod(env.single_action_space.shape))
        self.fc_logstd = nn.Linear(256, np.prod(env.single_action_space.shape))
        # action rescaling
        self.register_buffer(
            "action_scale",
            torch.tensor(
                (env.action_space.high - env.action_space.low) / 2.0,
                dtype=torch.float32,
            ),
        )
        self.register_buffer(
            "action_bias",
            torch.tensor(
                (env.action_space.high + env.action_space.low) / 2.0,
                dtype=torch.float32,
            ),
        )

    def forward(self, x):
        x = F.relu(self.fc1(self.bn1(x)))
        x = F.relu(self.fc2(self.bn2(x)))
        x = self.bn3(x)
        mean = self.fc_mean(x)
        log_std = self.fc_logstd(x)
        log_std = torch.tanh(log_std)
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (
            log_std + 1
        )  # From SpinUp / Denis Yarats
        return mean, log_std

    def get_action(self, x, train=False):
        if train:
            self.train()
        else:
            self.eval()
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
    import stable_baselines3 as sb3

    if sb3.__version__ < "2.0":
        raise ValueError(
            """Ongoing migration: run the following command to install the new dependencies:
poetry run pip install "stable_baselines3==2.0.0a1"
"""
        )

    args = tyro.cli(Args)
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

    actor = Actor(envs).to(device)
    qf1 = SoftQNetwork(envs).to(device)
    qf2 = SoftQNetwork(envs).to(device)
    q_optimizer = optim.Adam(
        list(qf1.parameters()) + list(qf2.parameters()),
        lr=args.q_lr,
        betas=(0.5, 0.999),
    )
    actor_optimizer = optim.Adam(list(actor.parameters()), lr=args.policy_lr)

    # Automatic entropy tuning
    if args.autotune:
        target_entropy = -torch.prod(
            torch.Tensor(envs.single_action_space.shape).to(device)
        ).item()
        log_alpha = torch.zeros(1, requires_grad=True, device=device)
        alpha = log_alpha.exp().item()
        a_optimizer = optim.Adam([log_alpha], lr=args.q_lr)
    else:
        alpha = args.alpha

    envs.single_observation_space.dtype = np.float32
    rb = ReplayBuffer(
        args.buffer_size,
        envs.single_observation_space,
        envs.single_action_space,
        device,
        handle_timeout_termination=False,
    )
    start_time = time.time()

    # TRY NOT TO MODIFY: start the game
    obs, _ = envs.reset(seed=args.seed)
    for global_step in range(args.total_timesteps):
        # ALGO LOGIC: put action logic here
        if global_step < args.learning_starts:
            actions = np.array(
                [envs.single_action_space.sample() for _ in range(envs.num_envs)]
            )
        else:
            actions, _, _ = actor.get_action(torch.Tensor(obs).to(device), train=False)
            actions = actions.detach().cpu().numpy()

        # TRY NOT TO MODIFY: execute the game and log data.
        next_obs, rewards, terminations, truncations, infos = envs.step(actions)

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        if "final_info" in infos:
            for info in infos["final_info"]:
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

        # TRY NOT TO MODIFY: save data to reply buffer; handle `final_observation`
        real_next_obs = next_obs.copy()
        for idx, trunc in enumerate(truncations):
            if trunc:
                real_next_obs[idx] = infos["final_observation"][idx]
        rb.add(obs, real_next_obs, actions, rewards, terminations, infos)

        # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
        obs = next_obs

        # ALGO LOGIC: training.
        if global_step > args.learning_starts:
            data = rb.sample(args.batch_size)
            with torch.no_grad():
                next_state_actions, next_state_log_pi, _ = actor.get_action(
                    data.next_observations, train=False
                )

            cat_obs = torch.cat((data.observations, data.next_observations), dim=0)
            cat_actions = torch.cat((data.actions, next_state_actions), dim=0)

            qf1_values = qf1(cat_obs, cat_actions, train=True)
            qf2_values = qf2(cat_obs, cat_actions, train=True)

            qf1_value, qf1_next = torch.chunk(qf1_values, 2)
            qf2_value, qf2_next = torch.chunk(qf2_values, 2)

            qf1_value = qf1_value.view(-1)
            qf2_value = qf2_value.view(-1)

            with torch.no_grad():
                qf1_next = qf1_next.detach()
                qf2_next = qf2_next.detach()
                min_qf_next = torch.min(qf1_next, qf2_next) - alpha * next_state_log_pi
                next_q_value = data.rewards.flatten() + (
                    1 - data.dones.flatten()
                ) * args.gamma * (min_qf_next).view(-1)

            qf1_loss = F.mse_loss(qf1_value, next_q_value)
            qf2_loss = F.mse_loss(qf2_value, next_q_value)
            qf_loss = qf1_loss + qf2_loss

            # optimize the model
            q_optimizer.zero_grad()
            qf_loss.backward()
            q_optimizer.step()

            if global_step % args.policy_frequency == 0:  # TD 3 Delayed update support
                pi, log_pi, _ = actor.get_action(data.observations, train=True)

                qf1_pi = qf1(data.observations, pi, train=False)
                qf2_pi = qf2(data.observations, pi, train=False)

                min_qf_pi = torch.min(qf1_pi, qf2_pi)
                actor_loss = ((alpha * log_pi) - min_qf_pi).mean()

                actor_optimizer.zero_grad()
                actor_loss.backward()
                actor_optimizer.step()

                if args.autotune:
                    with torch.no_grad():
                        _, log_pi, _ = actor.get_action(data.observations)
                    alpha_loss = (-log_alpha.exp() * (log_pi + target_entropy)).mean()

                    a_optimizer.zero_grad()
                    alpha_loss.backward()
                    a_optimizer.step()
                    alpha = log_alpha.exp().item()

            if global_step % 100 == 0:
                writer.add_scalar(
                    "losses/qf1_values", qf1_value.mean().item(), global_step
                )
                writer.add_scalar(
                    "losses/qf2_values", qf2_value.mean().item(), global_step
                )
                writer.add_scalar("losses/qf1_loss", qf1_loss.item(), global_step)
                writer.add_scalar("losses/qf2_loss", qf2_loss.item(), global_step)
                writer.add_scalar("losses/qf_loss", qf_loss.item() / 2.0, global_step)
                writer.add_scalar("losses/actor_loss", actor_loss.item(), global_step)
                writer.add_scalar("losses/alpha", alpha, global_step)
                print("SPS:", int(global_step / (time.time() - start_time)))
                writer.add_scalar(
                    "charts/SPS",
                    int(global_step / (time.time() - start_time)),
                    global_step,
                )
                if args.autotune:
                    writer.add_scalar(
                        "losses/alpha_loss", alpha_loss.item(), global_step
                    )

    envs.close()
    writer.close()
