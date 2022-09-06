# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/td3/#td3_continuous_action_jaxpy
import argparse
import os
import random
import time
from distutils.util import strtobool
from typing import Sequence, Optional

import flax
import flax.linen as nn
import gym
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pybullet_envs  # noqa
from flax.training.train_state import TrainState
from stable_baselines3.common.buffers import ReplayBuffer
from torch.utils.tensorboard import SummaryWriter
from stable_baselines3.common.vec_env import DummyVecEnv


def parse_args():
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", type=str, default=os.path.basename(__file__).rstrip(".py"),
        help="the name of this experiment")
    parser.add_argument("--seed", type=int, default=1,
        help="seed of the experiment")
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
    parser.add_argument("-n", "--total-timesteps", type=int, default=1000000,
        help="total timesteps of the experiments")
    parser.add_argument("-lr", "--learning-rate", type=float, default=3e-4,
        help="the learning rate of the optimizer")
    parser.add_argument("--buffer-size", type=int, default=int(1e6),
        help="the replay memory buffer size")
    parser.add_argument("--gamma", type=float, default=0.99,
        help="the discount factor gamma")
    parser.add_argument("--tau", type=float, default=0.005,
        help="target smoothing coefficient (default: 0.005)")
    parser.add_argument("--policy-noise", type=float, default=0.2,
        help="the scale of policy noise")
    parser.add_argument("--batch-size", type=int, default=256,
        help="the batch size of sample from the reply memory")
    parser.add_argument("--exploration-noise", type=float, default=0.1,
        help="the scale of exploration noise")
    parser.add_argument("--learning-starts", type=int, default=1000,
        help="timestep to start learning")
    parser.add_argument("--gradient-steps", type=int, default=1,
        help="Number of gradient steps to perform after each rollout")
    # Argument for dropout rate
    parser.add_argument("--dropout-rate", type=float, default=0.0)
    # Argument for layer normalization
    parser.add_argument("--layer-norm", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True)
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
    use_layer_norm: bool = False
    dropout_rate: Optional[float] = None

    @nn.compact
    def __call__(self, x: jnp.ndarray, a: jnp.ndarray, training: bool = False):
        x = jnp.concatenate([x, a], -1)
        x = nn.Dense(256)(x)
        if self.dropout_rate is not None and self.dropout_rate > 0:
            x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=False)
        if self.use_layer_norm:
            x = nn.LayerNorm()(x)
        x = nn.relu(x)
        x = nn.Dense(256)(x)
        if self.dropout_rate is not None and self.dropout_rate > 0:
            x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=False)
        if self.use_layer_norm:
            x = nn.LayerNorm()(x)
        x = nn.relu(x)
        x = nn.Dense(1)(x)
        return x


class Actor(nn.Module):
    action_dim: Sequence[int]
    action_scale: float
    action_bias: float

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(256)(x)
        x = nn.relu(x)
        x = nn.Dense(256)(x)
        x = nn.relu(x)
        x = nn.Dense(self.action_dim)(x)
        x = nn.tanh(x)
        x = x * self.action_scale + self.action_bias
        return x


class RLTrainState(TrainState):
    target_params: flax.core.FrozenDict


def main():
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
    key = jax.random.PRNGKey(args.seed)
    key, actor_key, qf1_key, qf2_key = jax.random.split(key, 4)
    key, dropout_key1, dropout_key2 = jax.random.split(key, 3)

    # env setup
    envs = DummyVecEnv(
        [make_env(args.env_id, args.seed, 0, args.capture_video, run_name)]
    )
    assert isinstance(
        envs.action_space, gym.spaces.Box
    ), "only continuous action space is supported"

    # Assume that all dimensions share the same bound
    min_action = float(envs.action_space.low[0])
    max_action = float(envs.action_space.high[0])
    envs.observation_space.dtype = np.float32
    rb = ReplayBuffer(
        args.buffer_size,
        envs.observation_space,
        envs.action_space,
        device="cpu",
        handle_timeout_termination=True,
    )

    # TRY NOT TO MODIFY: start the game
    obs = envs.reset()
    actor = Actor(
        action_dim=np.prod(envs.action_space.shape),
        action_scale=(max_action - min_action) / 2.0,
        action_bias=(max_action + min_action) / 2.0,
    )
    actor_state = RLTrainState.create(
        apply_fn=actor.apply,
        params=actor.init(actor_key, obs),
        target_params=actor.init(actor_key, obs),
        tx=optax.adam(learning_rate=args.learning_rate),
    )
    qf = QNetwork(dropout_rate=args.dropout_rate, use_layer_norm=args.layer_norm)
    qf1_state = RLTrainState.create(
        apply_fn=qf.apply,
        params=qf.init(
            {"params": qf1_key, "dropout": dropout_key1},
            obs,
            jnp.array([envs.action_space.sample()]),
        ),
        target_params=qf.init(
            {"params": qf1_key, "dropout": dropout_key1},
            obs,
            jnp.array([envs.action_space.sample()]),
        ),
        tx=optax.adam(learning_rate=args.learning_rate),
    )
    qf2_state = RLTrainState.create(
        apply_fn=qf.apply,
        params=qf.init(
            {"params": qf2_key, "dropout": dropout_key2},
            obs,
            jnp.array([envs.action_space.sample()]),
        ),
        target_params=qf.init(
            {"params": qf2_key, "dropout": dropout_key2},
            obs,
            jnp.array([envs.action_space.sample()]),
        ),
        tx=optax.adam(learning_rate=args.learning_rate),
    )
    actor.apply = jax.jit(actor.apply)
    qf.apply = jax.jit(qf.apply, static_argnames=("dropout_rate", "use_layer_norm"))

    @jax.jit
    def update_critic(
        actor_state: RLTrainState,
        qf1_state: RLTrainState,
        qf2_state: RLTrainState,
        observations: np.ndarray,
        actions: np.ndarray,
        next_observations: np.ndarray,
        rewards: np.ndarray,
        dones: np.ndarray,
        key: jnp.ndarray,
    ):
        # TODO Maybe pre-generate a lot of random keys
        # also check https://jax.readthedocs.io/en/latest/jax.random.html
        key, noise_key, dropout_key_1, dropout_key_2 = jax.random.split(key, 4)
        key, dropout_noise_key = jax.random.split(key, 2)

        clipped_noise = jnp.clip(
            (jax.random.normal(noise_key, actions[0].shape) * args.policy_noise),
            -args.noise_clip,
            args.noise_clip,
        )
        next_state_actions = jnp.clip(
            actor.apply(actor_state.target_params, next_observations) + clipped_noise,
            envs.action_space.low[0],
            envs.action_space.high[0],
        )
        qf1_next_target = qf.apply(
            qf1_state.target_params,
            next_observations,
            next_state_actions,
            True,
            rngs={"dropout": dropout_key_1},
        ).reshape(-1)
        qf2_next_target = qf.apply(
            qf2_state.target_params,
            next_observations,
            next_state_actions,
            True,
            rngs={"dropout": dropout_key_2},
        ).reshape(-1)
        min_qf_next_target = jnp.minimum(qf1_next_target, qf2_next_target)
        next_q_value = (
            rewards + (1 - dones) * args.gamma * (min_qf_next_target)
        ).reshape(-1)

        def mse_loss(params):
            qf_a_values = qf.apply(
                params, observations, actions, True, rngs={"dropout": dropout_noise_key}
            ).squeeze()
            return ((qf_a_values - next_q_value) ** 2).mean(), qf_a_values.mean()

        (qf1_loss_value, qf1_a_values), grads1 = jax.value_and_grad(
            mse_loss, has_aux=True
        )(qf1_state.params)
        (qf2_loss_value, qf2_a_values), grads2 = jax.value_and_grad(
            mse_loss, has_aux=True
        )(qf2_state.params)
        qf1_state = qf1_state.apply_gradients(grads=grads1)
        qf2_state = qf2_state.apply_gradients(grads=grads2)

        return (
            (qf1_state, qf2_state),
            (qf1_loss_value, qf2_loss_value),
            (qf1_a_values, qf2_a_values),
            key,
        )

    @jax.jit
    def update_actor(
        actor_state: RLTrainState,
        qf1_state: RLTrainState,
        qf2_state: RLTrainState,
        observations: np.ndarray,
        key: jnp.ndarray,
    ):
        key, dropout_key = jax.random.split(key, 2)

        def actor_loss(params):
            return -qf.apply(
                qf1_state.params,
                observations,
                actor.apply(params, observations),
                True,
                rngs={"dropout": dropout_key},
            ).mean()

        actor_loss_value, grads = jax.value_and_grad(actor_loss)(actor_state.params)
        actor_state = actor_state.apply_gradients(grads=grads)
        actor_state = actor_state.replace(
            target_params=optax.incremental_update(
                actor_state.params, actor_state.target_params, args.tau
            )
        )

        qf1_state = qf1_state.replace(
            target_params=optax.incremental_update(
                qf1_state.params, qf1_state.target_params, args.tau
            )
        )
        qf2_state = qf2_state.replace(
            target_params=optax.incremental_update(
                qf2_state.params, qf2_state.target_params, args.tau
            )
        )
        return actor_state, (qf1_state, qf2_state), actor_loss_value, key

    start_time = time.time()
    n_updates = 0
    for global_step in range(args.total_timesteps):
        # ALGO LOGIC: put action logic here
        if global_step < args.learning_starts:
            actions = np.array(
                [envs.action_space.sample() for _ in range(envs.num_envs)]
            )
        else:
            actions = actor.apply(actor_state.params, obs)
            actions = np.array(
                [
                    (
                        jax.device_get(actions)[0]
                        + np.random.normal(
                            0,
                            max_action * args.exploration_noise,
                            size=envs.action_space.shape[0],
                        )
                    ).clip(envs.action_space.low, envs.action_space.high)
                ]
            )

        # TRY NOT TO MODIFY: execute the game and log data.
        next_obs, rewards, dones, infos = envs.step(actions)

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        for info in infos:
            if "episode" in info.keys():
                print(
                    f"global_step={global_step + 1}, episodic_return={info['episode']['r']:.2f}"
                )
                writer.add_scalar(
                    "charts/episodic_return", info["episode"]["r"], global_step
                )
                writer.add_scalar(
                    "charts/episodic_length", info["episode"]["l"], global_step
                )
                break

        # TRY NOT TO MODIFY: save data to replay buffer; handle `terminal_observation`
        real_next_obs = next_obs.copy()
        for idx, done in enumerate(dones):
            if done:
                real_next_obs[idx] = infos[idx]["terminal_observation"]
            # Timeout handling done inside the replay buffer
            # if infos[idx].get("TimeLimit.truncated", False) == True:
            #     real_dones[idx] = False

        rb.add(obs, real_next_obs, actions, rewards, dones, infos)

        # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
        obs = next_obs

        # ALGO LOGIC: training.
        if global_step > args.learning_starts:
            for _ in range(args.gradient_steps):
                n_updates += 1
                data = rb.sample(args.batch_size)

                (
                    (qf1_state, qf2_state),
                    (qf1_loss_value, qf2_loss_value),
                    (qf1_a_values, qf2_a_values),
                    key,
                ) = update_critic(
                    actor_state,
                    qf1_state,
                    qf2_state,
                    data.observations.numpy(),
                    data.actions.numpy(),
                    data.next_observations.numpy(),
                    data.rewards.flatten().numpy(),
                    data.dones.flatten().numpy(),
                    key,
                )

                if n_updates % args.policy_frequency == 0:
                    (
                        actor_state,
                        (qf1_state, qf2_state),
                        actor_loss_value,
                        key,
                    ) = update_actor(
                        actor_state,
                        qf1_state,
                        qf2_state,
                        data.observations.numpy(),
                        key,
                    )
            if global_step % 100 == 0:
                writer.add_scalar("losses/qf1_loss", qf1_loss_value.item(), global_step)
                writer.add_scalar("losses/qf2_loss", qf2_loss_value.item(), global_step)
                writer.add_scalar("losses/qf1_values", qf1_a_values.item(), global_step)
                writer.add_scalar("losses/qf2_values", qf2_a_values.item(), global_step)
                writer.add_scalar(
                    "losses/actor_loss", actor_loss_value.item(), global_step
                )
                print("FPS:", int(global_step / (time.time() - start_time)))
                writer.add_scalar(
                    "charts/SPS",
                    int(global_step / (time.time() - start_time)),
                    global_step,
                )

    envs.close()
    writer.close()


if __name__ == "__main__":

    try:
        main()
    except KeyboardInterrupt:
        pass
