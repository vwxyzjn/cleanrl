import argparse
import functools
import os
import time
from distutils.util import strtobool

import flax.linen as nn
import gym
import jax
import jax.numpy as jnp
import numpy as np
import optax
from stable_baselines3.common.atari_wrappers import (
    ClipRewardEnv,
    EpisodicLifeEnv,
    FireResetEnv,
    MaxAndSkipEnv,
    NoopResetEnv,
)
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
    parser.add_argument("--env-id", type=str, default="BreakoutNoFrameskip-v4",
        help="the id of the environment")
    parser.add_argument("--total-timesteps", type=int, default=10000000,
        help="total timesteps of the experiments")
    parser.add_argument("--learning-rate", type=float, default=1e-4,
        help="the learning rate of the optimizer")
    parser.add_argument("--buffer-size", type=int, default=1000000,
        help="the replay memory buffer size")
    parser.add_argument("--gamma", type=float, default=0.99,
        help="the discount factor gamma")
    parser.add_argument("--target-network-frequency", type=int, default=1000,
        help="the timesteps it takes to update the target network")
    parser.add_argument("--max-grad-norm", type=float, default=0.5,
        help="the maximum norm for the gradient clipping")
    parser.add_argument("--batch-size", type=int, default=32,
        help="the batch size of sample from the reply memory")
    parser.add_argument("--start-e", type=float, default=1,
        help="the starting epsilon for exploration")
    parser.add_argument("--end-e", type=float, default=0.01,
        help="the ending epsilon for exploration")
    parser.add_argument("--exploration-fraction", type=float, default=0.10,
        help="the fraction of `total-timesteps` it takes from start-e to go end-e")
    parser.add_argument("--learning-starts", type=int, default=80000,
        help="timestep to start learning")
    parser.add_argument("--train-frequency", type=int, default=4,
        help="the frequency of training")
    parser.add_argument("--mico-weight", type=float, default=0.01,
        help="the weight for mico loss")

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
        env = NoopResetEnv(env, noop_max=30)
        env = MaxAndSkipEnv(env, skip=4)
        env = EpisodicLifeEnv(env)
        if "FIRE" in env.unwrapped.get_action_meanings():
            env = FireResetEnv(env)
        env = ClipRewardEnv(env)
        env = gym.wrappers.ResizeObservation(env, (84, 84))
        env = gym.wrappers.GrayScaleObservation(env)
        env = gym.wrappers.FrameStack(env, 4)
        env.seed(seed)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        return env

    return thunk


# ALGO LOGIC: initialize agent here:
class QNetwork(nn.Module):
    action_dim: int
    initializer: callable

    @nn.compact
    def __call__(self, x: jnp.ndarray):
        x = jnp.transpose(x, (1, 2, 0))
        x = x.astype(jnp.float32) / 255.0
        x = nn.Conv(features=32, kernel_size=(8, 8), strides=(4, 4), kernel_init=self.initializer, padding="VALID")(x)
        x = nn.relu(x)
        x = nn.Conv(features=64, kernel_size=(4, 4), strides=(2, 2), kernel_init=self.initializer, padding="VALID")(x)
        x = nn.relu(x)
        x = nn.Conv(features=64, kernel_size=(3, 3), strides=(1, 1), kernel_init=self.initializer, padding="VALID")(x)
        x = nn.relu(x)
        x = x.reshape(-1)  # flatten
        x = nn.Dense(features=512, kernel_init=self.initializer)(x)
        x = nn.relu(x)
        q_values = nn.Dense(features=self.action_dim, kernel_init=self.initializer)(x)
        return q_values


@functools.partial(jax.jit, static_argnums=(1,))
def linear_schedule(start_e: float, end_e: float, duration: int, t: int):
    slope = (end_e - start_e) / duration
    return jnp.maximum(slope * t + start_e, end_e)


def mse(target, prediction):
    return jnp.power((target - prediction), 2)


@functools.partial(jax.jit, static_argnums=(0, 3, 10))
def train(
    network_def, online_params, target_params, optimizer, optimizer_state, next_obs, obs, rewards, dones, actions, gamma
):
    def loss_fn(online_params, target_params, next_obs, obs, rewards, dones, actions):
        def q_val_fn(observations):
            return network_def.apply(online_params, observations)

        def t_val_fn(observations):
            return network_def.apply(target_params, observations)

        target_max = jnp.max(jax.vmap(t_val_fn)(next_obs), axis=1)
        td_target = jax.lax.stop_gradient(rewards + gamma * target_max * (1 - dones))
        q_val = jax.vmap(q_val_fn)(obs)
        old_val = jax.vmap(lambda x, y: x[y])(q_val, actions)
        return jnp.mean(jax.vmap(mse)(td_target, old_val)), old_val

    # optimize the model
    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, old_val), grad = grad_fn(online_params, target_params, next_obs, obs, rewards, dones, actions)
    updates, optimizer_state = optimizer.update(grad, optimizer_state, params=online_params)
    online_params = optax.apply_updates(online_params, updates)
    return loss, old_val, online_params, optimizer_state


@functools.partial(jax.jit, static_argnums=(2, 3))
def select_action(rng_seed, epsilon, num_actions, network_def, online_params, obs):
    rng, rng_1, rng_2 = jax.random.split(rng_seed, 3)
    return rng, jnp.where(
        jax.random.uniform(rng_1) < epsilon,
        jax.random.randint(rng_2, (), 0, num_actions),
        jnp.argmax(network_def.apply(online_params, obs)),
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
    key = jax.random.PRNGKey(args.seed)
    q_network_seed, t_network_seed, rng_seed = jax.random.split(key, num=3)

    # env setup
    envs = gym.vector.SyncVectorEnv([make_env(args.env_id, args.seed, 0, args.capture_video, run_name)])
    assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"
    obs = envs.reset()
    network_def = QNetwork(envs.single_action_space.n, nn.initializers.xavier_uniform())
    q_network_params = network_def.init(q_network_seed, x=obs.squeeze())
    optimizer = optax.adam(args.learning_rate, b1=0.99, b2=0.999)
    optimizer_state = optimizer.init(q_network_params)
    target_network_params = q_network_params

    rb = ReplayBuffer(args.buffer_size, envs.single_observation_space, envs.single_action_space, optimize_memory_usage=True)
    start_time = time.time()

    # TRY NOT TO MODIFY: start the game
    for global_step in range(args.total_timesteps):
        # ALGO LOGIC: put action logic here
        epsilon = linear_schedule(args.start_e, args.end_e, args.exploration_fraction * args.total_timesteps, global_step)
        rng_seed, actions = select_action(
            rng_seed, epsilon, envs.single_action_space.n, network_def, q_network_params, obs.squeeze()
        )
        actions = np.array([actions])
        # TRY NOT TO MODIFY: execute the game and log data.
        next_obs, rewards, dones, infos = envs.step(actions)
        # TRY NOT TO MODIFY: record rewards for plotting purposes
        for info in infos:
            if "episode" in info.keys():
                print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
                writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
                writer.add_scalar("charts/epsilon", np.asarray(epsilon), global_step)
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
        if global_step > args.learning_starts and global_step % args.train_frequency == 0:
            data = rb.sample(args.batch_size)
            loss, old_vals, q_network_params, optimizer_state = train(
                network_def,
                q_network_params,
                target_network_params,
                optimizer,
                optimizer_state,
                data.next_observations.numpy(),
                data.observations.numpy(),
                data.rewards.flatten().numpy(),
                data.dones.flatten().numpy(),
                data.actions.numpy(),
                args.gamma,
            )
            if global_step % 100 == 0:
                writer.add_scalar("losses/td_loss", np.asarray(loss), global_step)
                writer.add_scalar("losses/q_values", np.asarray(old_vals).mean(), global_step)
                print("SPS:", int(global_step / (time.time() - start_time)))
                writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)
            # update the target network
            if global_step % args.target_network_frequency == 0:
                target_network_params = q_network_params

    envs.close()
    writer.close()
