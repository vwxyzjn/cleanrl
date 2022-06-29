# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/c51/#c51py
import argparse
import os
import random
import time
from distutils.util import strtobool

import flax.linen as nn
import gym
import jax
import jax.numpy as jnp
import numpy as np
import optax
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
    parser.add_argument("--env-id", type=str, default="CartPole-v1",
        help="the id of the environment")
    parser.add_argument("--total-timesteps", type=int, default=500000,
        help="total timesteps of the experiments")
    parser.add_argument("--learning-rate", type=float, default=2.5e-4,
        help="the learning rate of the optimizer")
    parser.add_argument("--n-atoms", type=int, default=101,
        help="the number of atoms")
    parser.add_argument("--v-min", type=float, default=-100,
        help="the number of atoms")
    parser.add_argument("--v-max", type=float, default=100,
        help="the number of atoms")
    parser.add_argument("--buffer-size", type=int, default=10000,
        help="the replay memory buffer size")
    parser.add_argument("--gamma", type=float, default=0.99,
        help="the discount factor gamma")
    parser.add_argument("--target-network-frequency", type=int, default=500,
        help="the timesteps it takes to update the target network")
    parser.add_argument("--batch-size", type=int, default=128,
        help="the batch size of sample from the reply memory")
    parser.add_argument("--start-e", type=float, default=1,
        help="the starting epsilon for exploration")
    parser.add_argument("--end-e", type=float, default=0.05,
        help="the ending epsilon for exploration")
    parser.add_argument("--exploration-fraction", type=float, default=0.5,
        help="the fraction of `total-timesteps` it takes from start-e to go end-e")
    parser.add_argument("--learning-starts", type=int, default=10000,
        help="timestep to start learning")
    parser.add_argument("--train-frequency", type=int, default=10,
        help="the frequency of training")
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
    action_dim: int
    n_atoms: int
    v_min: int
    v_max: int
    
    @nn.compact
    def __call__(self, x):
        x = nn.Dense(120)(x)
        x = nn.relu(x)
        x = nn.Dense(84)(x)
        x = nn.relu(x)
        x = nn.Dense(self.action_dim * self.n_atoms)(x)
        x = x.reshape((x.shape[0], self.action_dim, self.n_atoms))
        x = nn.softmax(x, axis=-1)  # pmfs
        return x


def linear_schedule(start_e: float, end_e: float, duration: int, t: int):
    slope = (end_e - start_e) / duration
    return max(slope * t + start_e, end_e)


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
    key = jax.random.PRNGKey(args.seed)
    key, q_key = jax.random.split(key, 2)

    # env setup
    envs = gym.vector.SyncVectorEnv([make_env(args.env_id, args.seed, 0, args.capture_video, run_name)])
    assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"

    obs = envs.reset()

    q_network = QNetwork(action_dim=envs.single_action_space.n, n_atoms=args.n_atoms, v_min=args.v_min, v_max=args.v_max)
    q_params = q_network.init(q_key, obs)
    target_params = q_network.init(q_key, obs)
    q_network.apply = jax.jit(q_network.apply)
    target_params = optax.incremental_update(q_params, target_params, step_size=1)
    # directly using jnp.linspace leads to numerical errors
    atoms = jnp.asarray(np.linspace(args.v_min, args.v_max, num=args.n_atoms))

    optimizer = optax.adam(learning_rate=args.learning_rate)
    optimizer_state = optimizer.init(q_params)

    rb = ReplayBuffer(
        args.buffer_size,
        envs.single_observation_space,
        envs.single_action_space,
        "cpu",
        handle_timeout_termination=True,
    )

    # def get_action_pmfs(params, obs):
    #     pmfs = q_network(params, obs)  # (batch_size, num_actions, num_atoms)
    #     q_values = (atoms * pmfs).sum(axis=-1)  # (batch_size, num_actions)
    #     actions = jnp.argmax(q_values, axis=-1)  # (batch_size,)

    @jax.jit
    def update(observations, actions, next_observations, rewards, dones, q_params, target_params, optimizer_params):
        next_pmfs = q_network.apply(target_params, next_observations)  # (batch_size, num_actions, num_atoms)
        next_vals = (next_pmfs * atoms).sum(axis=-1)  # (batch_size, num_actions)
        next_action = jnp.argmax(next_vals, axis=-1)  # (batch_size,)
        next_pmfs = next_pmfs[np.arange(next_pmfs.shape[0]), next_action]
        next_atoms = rewards + args.gamma * atoms * (1 - dones)
        # projection
        delta_z = atoms[1]-atoms[0]
        tz = jax.lax.clamp(jnp.float32(args.v_min), next_atoms, jnp.float32(args.v_max))
        b = (tz - args.v_min) / delta_z
        l = jax.lax.clamp(0.0, jnp.floor(b), args.n_atoms-1.0)
        u = jax.lax.clamp(0.0, jnp.ceil(b), args.n_atoms-1.0)
        # (l == u).astype(jnp.float) handles the case where bj is exactly an integer
        # example bj = 1, then the upper ceiling should be uj= 2, and lj= 1
        d_m_l = (u + (l == u).astype(jnp.float32) - b) * next_pmfs
        d_m_u = (b - l) * next_pmfs

        target_pmfs = jnp.zeros_like(next_pmfs)
        for i in range(target_pmfs.shape[0]):
            target_pmfs = target_pmfs.at[l[i].astype(jnp.int32)].add(d_m_l[i])
            target_pmfs = target_pmfs.at[u[i].astype(jnp.int32)].add(d_m_u[i])

        def loss(q_params, observations, actions, target_pmfs):
            pmfs = q_network.apply(q_params, observations)
            old_pmfs = pmfs[np.arange(pmfs.shape[0]), actions.squeeze()]
            # target_params = jnp.clip(target_pmfs, a_min=1e-5, a_max=1 - 1e-5)
            old_pmfs_l = jnp.clip(old_pmfs, a_min=1e-5, a_max=1 - 1e-5)
            loss = (-(target_pmfs * jnp.log(old_pmfs_l)).sum(-1)).mean()
            return loss, (old_pmfs*atoms).sum(-1)
            
        (loss_value, old_values), grads = jax.value_and_grad(loss, has_aux=True)(q_params, observations, actions, target_pmfs)
        updates, optimizer_params = optimizer.update(grads, optimizer_params)
        q_params = optax.apply_updates(q_params, updates)
        return loss_value, old_values, q_params, optimizer_params

    start_time = time.time()

    # TRY NOT TO MODIFY: start the game
    obs = envs.reset()
    for global_step in range(args.total_timesteps):
        # ALGO LOGIC: put action logic here
        epsilon = linear_schedule(args.start_e, args.end_e, args.exploration_fraction * args.total_timesteps, global_step)
        if random.random() < epsilon:
            actions = np.array([envs.single_action_space.sample() for _ in range(envs.num_envs)])
        else:
            pmfs= q_network.apply(q_params, obs)
            q_vals = (pmfs * atoms).sum(axis=-1)
            actions = q_vals.argmax(axis=-1)
            actions = jax.device_get(actions)

        # TRY NOT TO MODIFY: execute the game and log data.
        next_obs, rewards, dones, infos = envs.step(actions)

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        for info in infos:
            if "episode" in info.keys():
                print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
                writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
                writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)
                writer.add_scalar("charts/epsilon", epsilon, global_step)
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
            loss, old_val, q_params, optimizer_params = update(
                data.observations.numpy(),
                data.actions.numpy(),
                data.next_observations.numpy(),
                data.rewards.numpy(),
                data.dones.numpy(),
                q_params,
                target_params,
                optimizer_state,
            )

            if global_step % 100 == 0:
                writer.add_scalar("losses/loss", jax.device_get(loss), global_step)
                writer.add_scalar("losses/q_values", jax.device_get(old_val.mean()), global_step)
                print("SPS:", int(global_step / (time.time() - start_time)))
                writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

            # update the target network
            if global_step % args.target_network_frequency == 0:
                target_params = optax.incremental_update(q_params, target_params, 1)

    envs.close()
    writer.close()
