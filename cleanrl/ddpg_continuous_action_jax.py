import argparse
import functools
import os
import random
import time
import copy
from distutils.util import strtobool
from typing import Sequence

import flax.linen as nn
import gym
import jax
import jax.numpy as jnp
import numpy as np
import optax
import torch
from stable_baselines3.common.buffers import ReplayBuffer
from torch.utils.tensorboard import SummaryWriter


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
    parser.add_argument("--env-id", type=str, default="HalfCheetah-v2",
        help="the id of the environment")
    parser.add_argument("--total-timesteps", type=int, default=1000000,
        help="total timesteps of the experiments")
    parser.add_argument("--learning-rate", type=float, default=3e-4,
        help="the learning rate of the optimizer")
    parser.add_argument("--buffer-size", type=int, default=int(1e6),
        help="the replay memory buffer size")
    parser.add_argument("--gamma", type=float, default=0.99,
        help="the discount factor gamma")
    parser.add_argument("--tau", type=float, default=0.005,
        help="target smoothing coefficient (default: 0.005)")
    parser.add_argument("--batch-size", type=int, default=256,
        help="the batch size of sample from the reply memory")
    parser.add_argument("--exploration-noise", type=float, default=0.1,
        help="the scale of exploration noise")
    parser.add_argument("--learning-starts", type=int, default=25e3,
        help="timestep to start learning")
    parser.add_argument("--policy-frequency", type=int, default=2,
        help="the frequency of training policy (delayed)")
    parser.add_argument("--noise-clip", type=float, default=0.5,
        help="noise clip parameter of the Target Policy Smoothing Regularization")
    args = parser.parse_args()
    # fmt: on
    return args


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


# ALGO LOGIC: initialize agent here:
class QNetwork(nn.Module):
    obs_dim: Sequence[int]
    action_dim: Sequence[int]
    @nn.compact
    def __call__(self, x: jnp.ndarray, a: jnp.ndarray):
        x = jnp.concatenate([x, a], -1)
        x = nn.Dense(256)(x)
        x = nn.relu(x)
        x = nn.Dense(256)(x)
        x = nn.relu(x)
        x = nn.Dense(1)(x)
        return x


class Actor(nn.Module):
    action_dim: Sequence[int]

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(256)(x)
        x = nn.relu(x)
        x = nn.Dense(256)(x)
        x = nn.relu(x)
        x = nn.Dense(self.action_dim)(x)
        x = nn.tanh(x)
        return x


@functools.partial(jax.jit, static_argnames=('actor', 'qf1', 'qf1', 'qf1_optimizer', 'actor_optimizer'))
def forward(
    actor,
    actor_parameters,
    actor_target_parameters,
    qf1,
    qf1_parameters,
    qf1_target_parameters,
    observations,
    actions,
    next_observations,
    rewards,
    dones,
    gamma,
    tau,
    qf1_optimizer,
    qf1_optimizer_state,
    actor_optimizer,
    actor_optimizer_state,
):
    next_state_actions = (actor.apply(actor_target_parameters, next_observations)).clip(-1, 1)
    qf1_next_target = qf1.apply(qf1_target_parameters, next_observations, next_state_actions).reshape(-1)
    next_q_value = (rewards + (1 - dones) * gamma * (qf1_next_target)).reshape(-1)

    def mse_loss(qf1_parameters, observations, actions, next_q_value):
        return ((qf1.apply(qf1_parameters, observations, actions).squeeze() - next_q_value) ** 2).mean()

    qf1_loss_value, grads = jax.value_and_grad(mse_loss)(qf1_parameters, observations, actions, next_q_value)
    updates, qf1_optimizer_state = qf1_optimizer.update(grads, qf1_optimizer_state)
    qf1_parameters = optax.apply_updates(qf1_parameters, updates)

    return qf1_loss_value, 0, qf1_parameters, qf1_target_parameters, qf1_optimizer_state, actor_parameters, actor_target_parameters, actor_optimizer_state


@functools.partial(jax.jit, static_argnames=('actor', 'qf1', 'qf1', 'qf1_optimizer', 'actor_optimizer'))
def forward2(
    actor,
    actor_parameters,
    actor_target_parameters,
    qf1,
    qf1_parameters,
    qf1_target_parameters,
    observations,
    actions,
    next_observations,
    rewards,
    dones,
    gamma,
    tau,
    qf1_optimizer,
    qf1_optimizer_state,
    actor_optimizer,
    actor_optimizer_state,
):
    def actor_loss(actor_parameters, qf1_parameters, observations):
        return -qf1.apply(qf1_parameters, observations, actor.apply(actor_parameters, observations)).mean()

    actor_loss_value, grads = jax.value_and_grad(actor_loss)(actor_parameters, qf1_parameters, observations)
    updates, actor_optimizer_state = actor_optimizer.update(grads, actor_optimizer_state)
    actor_parameters = optax.apply_updates(actor_parameters, updates)

    actor_target_parameters = update_target(actor_parameters, actor_target_parameters, tau)
    qf1_target_parameters = update_target(qf1_parameters, qf1_target_parameters, tau)

    return 0, actor_loss_value, qf1_parameters, qf1_target_parameters, qf1_optimizer_state, actor_parameters, actor_target_parameters, actor_optimizer_state


def update_target(src, dst, tau):
    return jax.tree_map(
        lambda p, tp: p * tau + tp * (1 - tau), src, dst
    )

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
    jaxRNG = jax.random.PRNGKey(0)

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # env setup
    envs = gym.vector.SyncVectorEnv([make_env(args.env_id, args.seed, 0, args.capture_video, run_name)])
    assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"

    max_action = float(envs.single_action_space.high[0])

    # actor = Actor(envs).to(device)

    envs.single_observation_space.dtype = np.float32
    rb = ReplayBuffer(args.buffer_size, envs.single_observation_space, envs.single_action_space, device="cpu")
    start_time = time.time()

    # TRY NOT TO MODIFY: start the game
    obs = envs.reset()

    actor = Actor(action_dim=np.prod(envs.single_action_space.shape))
    actor_parameters = actor.init(jaxRNG, obs)
    actor_target_parameters = actor.init(jaxRNG, obs)
    actor.apply = jax.jit(actor.apply)
    qf1 = QNetwork(obs_dim=np.prod(envs.single_observation_space.shape), action_dim=np.prod(envs.single_action_space.shape))
    qf1_parameters = qf1.init(jaxRNG, obs, envs.action_space.sample())
    qf1_target_parameters = qf1.init(jaxRNG, obs, envs.action_space.sample())
    qf1.apply = jax.jit(qf1.apply)
    actor_target_parameters = update_target(actor_parameters, actor_target_parameters, 1.0)
    qf1_target_parameters = update_target(qf1_parameters, qf1_target_parameters, 1.0)
    actor_optimizer = optax.adam(learning_rate=args.learning_rate)
    actor_optimizer_state = actor_optimizer.init(actor_parameters)
    qf1_optimizer = optax.adam(learning_rate=args.learning_rate)
    qf1_optimizer_state = qf1_optimizer.init(qf1_parameters)

    # raise
    for global_step in range(args.total_timesteps):
        # ALGO LOGIC: put action logic here
        if global_step < args.learning_starts:
            actions = np.array([envs.single_action_space.sample() for _ in range(envs.num_envs)])
        else:
            actions = actor.apply(actor_parameters, obs)
            actions = np.array(
                [
                    (
                        np.array(actions)[0]
                        + np.random.normal(0, max_action * args.exploration_noise, size=envs.single_action_space.shape[0])
                    ).clip(envs.single_action_space.low, envs.single_action_space.high)
                ]
            )

        # TRY NOT TO MODIFY: execute the game and log data.
        next_obs, rewards, dones, infos = envs.step(actions)

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        for info in infos:
            if "episode" in info.keys():
                print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
                writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
                writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)
                break

        # TRY NOT TO MODIFY: save data to reply buffer; handle `terminal_observation`
        real_next_obs = next_obs.copy()
        for idx, d in enumerate(dones):
            if d:
                real_next_obs[idx] = infos[idx]["terminal_observation"]
        rb.add(obs, real_next_obs, actions, rewards, dones, infos)

        # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
        obs = next_obs

        # # ALGO LOGIC: training.
        if global_step > args.learning_starts:
            data = rb.sample(args.batch_size)
            qf1_loss_value, _, qf1_parameters, qf1_target_parameters, qf1_optimizer_state, actor_parameters, actor_target_parameters, actor_optimizer_state = forward(
                actor,
                actor_parameters,
                actor_target_parameters,
                qf1,
                qf1_parameters,
                qf1_target_parameters,
                data.observations.numpy(),
                data.actions.numpy(),
                data.next_observations.numpy(),
                data.rewards.flatten().numpy(),
                data.dones.flatten().numpy(),
                args.gamma,
                args.tau,
                qf1_optimizer,
                qf1_optimizer_state,
                actor_optimizer,
                actor_optimizer_state,
            )

            if global_step % args.policy_frequency == 0:
                _, actor_loss_value, qf1_parameters, qf1_target_parameters, qf1_optimizer_state, actor_parameters, actor_target_parameters, actor_optimizer_state = forward2(
                    actor,
                    actor_parameters,
                    actor_target_parameters,
                    qf1,
                    qf1_parameters,
                    qf1_target_parameters,
                    data.observations.numpy(),
                    data.actions.numpy(),
                    data.next_observations.numpy(),
                    data.rewards.flatten().numpy(),
                    data.dones.flatten().numpy(),
                    args.gamma,
                    args.tau,
                    qf1_optimizer,
                    qf1_optimizer_state,
                    actor_optimizer,
                    actor_optimizer_state,
                )

            
            # print(actor_parameters["params"]["Dense_0"]["kernel"].sum())
            if global_step % 100 == 0:
                # print(qf1_target_parameters["params"]["Dense_0"]["kernel"].sum())
                writer.add_scalar("losses/qf1_loss", qf1_loss_value.item(), global_step)
                writer.add_scalar("losses/actor_loss", actor_loss_value.item(), global_step)
                # writer.add_scalar("losses/qf1_values", qf1_a_values.mean().item(), global_step)
                print("SPS:", int(global_step / (time.time() - start_time)))
                writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

    envs.close()
    writer.close()
