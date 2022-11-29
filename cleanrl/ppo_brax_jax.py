# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppo_atari_envpool_xla_jaxpy
import argparse
import os
import random
import time
from distutils.util import strtobool
from typing import Sequence

import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax
from brax import envs
from brax.envs import wrappers
from flax.linen.initializers import constant, orthogonal
from flax.training.train_state import TrainState
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
        help="whether to capture videos of the agent performances (check out `videos` folder)")

    # Algorithm specific arguments
    parser.add_argument("--env-id", type=str, default="ant",
        help="the id of the environment")
    parser.add_argument("--total-timesteps", type=int, default=60000000,
        help="total timesteps of the experiments")
    parser.add_argument("--learning-rate", type=float, default=0.0026,
        help="the learning rate of the optimizer")
    parser.add_argument("--num-envs", type=int, default=2048,
        help="the number of parallel game environments")
    parser.add_argument("--num-steps", type=int, default=16,
        help="the number of steps to run in each environment per policy rollout")
    parser.add_argument("--anneal-lr", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="Toggle learning rate annealing for policy and value networks")
    parser.add_argument("--gamma", type=float, default=0.99,
        help="the discount factor gamma")
    parser.add_argument("--gae-lambda", type=float, default=0.95,
        help="the lambda for the general advantage estimation")
    parser.add_argument("--num-minibatches", type=int, default=2,
        help="the number of mini-batches")
    parser.add_argument("--update-epochs", type=int, default=4,
        help="the K epochs to update the policy")
    parser.add_argument("--norm-adv", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Toggles advantages normalization")
    parser.add_argument("--clip-coef", type=float, default=0.2,
        help="the surrogate clipping coefficient")
    parser.add_argument("--clip-vloss", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="Toggles whether or not to use a clipped loss for the value function, as per the paper.")
    parser.add_argument("--ent-coef", type=float, default=0.0,
        help="coefficient of the entropy")
    parser.add_argument("--vf-coef", type=float, default=2,
        help="coefficient of the value function")
    parser.add_argument("--max-grad-norm", type=float, default=1,
        help="the maximum norm for the gradient clipping")
    parser.add_argument("--target-kl", type=float, default=None,
        help="the target KL divergence threshold")
    args = parser.parse_args()
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_updates = args.total_timesteps // args.batch_size
    # fmt: on
    return args


class Critic(nn.Module):
    @nn.compact
    def __call__(self, x):
        critic = nn.Dense(64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(x)
        critic = nn.tanh(critic)
        critic = nn.Dense(64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(critic)
        critic = nn.tanh(critic)
        critic = nn.Dense(1, kernel_init=orthogonal(1), bias_init=constant(0.0))(critic)
        return critic


class Actor(nn.Module):
    action_dim: Sequence[int]

    @nn.compact
    def __call__(self, x):
        actor_mean = nn.Dense(64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(x)
        actor_mean = nn.tanh(actor_mean)
        actor_mean = nn.Dense(64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(actor_mean)
        actor_mean = nn.tanh(actor_mean)
        actor_mean = nn.Dense(self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0))(actor_mean)
        actor_logstd = self.param("actor_logstd", constant(0.0), (1, self.action_dim))
        return actor_mean, actor_logstd


@flax.struct.dataclass
class AgentParams:
    actor_params: flax.core.FrozenDict
    critic_params: flax.core.FrozenDict


@flax.struct.dataclass
class Storage:
    obs: jnp.array
    actions: jnp.array
    logprobs: jnp.array
    dones: jnp.array
    values: jnp.array
    advantages: jnp.array
    returns: jnp.array
    rewards: jnp.array


@flax.struct.dataclass
class EpisodeStatistics:
    episode_returns: jnp.array
    episode_lengths: jnp.array
    returned_episode_returns: jnp.array
    returned_episode_lengths: jnp.array


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
    key, network_key, actor_key, critic_key = jax.random.split(key, 4)
    local_devices_to_use = 1
    key_envs = jax.random.split(key, args.num_envs // 1)
    key_envs = jnp.reshape(key_envs, (local_devices_to_use, -1) + key_envs.shape[1:])

    # env setup
    env = envs.get_environment(env_name=args.env_id)
    env = wrappers.wrap_for_training(env, episode_length=1000, action_repeat=1)
    reset_fn = jax.jit(jax.vmap(env.reset))
    step_env_fn = jax.jit(jax.vmap(env.step))

    @jax.jit
    def reset(key_envs):
        handle = reset_fn(key_envs)
        return handle, handle.obs.squeeze()

    @jax.jit
    def step_env(handle, actions):
        handle = step_env_fn(handle, actions.reshape(1, *actions.shape))
        return handle, (handle.obs.squeeze(), handle.reward.squeeze(), handle.done.squeeze(), handle.info)

    handle, next_obs = reset(key_envs)
    episode_stats = EpisodeStatistics(
        episode_returns=jnp.zeros(args.num_envs, dtype=jnp.float32),
        episode_lengths=jnp.zeros(args.num_envs, dtype=jnp.int32),
        returned_episode_returns=jnp.zeros(args.num_envs, dtype=jnp.float32),
        returned_episode_lengths=jnp.zeros(args.num_envs, dtype=jnp.int32),
    )

    def step_env_wrappeed(episode_stats, handle, action):
        handle, (next_obs, reward, next_done, info) = step_env(handle, action)
        new_episode_return = episode_stats.episode_returns + reward
        new_episode_length = episode_stats.episode_lengths + 1
        episode_stats = episode_stats.replace(
            episode_returns=(new_episode_return) * (1 - next_done),
            episode_lengths=(new_episode_length) * (1 - next_done),
            # only update the `returned_episode_returns` if the episode is done
            returned_episode_returns=jnp.where(next_done, new_episode_return, episode_stats.returned_episode_returns),
            returned_episode_lengths=jnp.where(next_done, new_episode_length, episode_stats.returned_episode_lengths),
        )
        return episode_stats, handle, (next_obs, reward, next_done, info)

    def linear_schedule(count):
        # anneal learning rate linearly after one training iteration which contains
        # (args.num_minibatches * args.update_epochs) gradient updates
        frac = 1.0 - (count // (args.num_minibatches * args.update_epochs)) / args.num_updates
        return args.learning_rate * frac

    actor = Actor(action_dim=env.action_size)
    critic = Critic()
    agent_state = TrainState.create(
        apply_fn=None,
        params=AgentParams(
            actor.init(actor_key, next_obs),
            critic.init(critic_key, next_obs),
        ),
        tx=optax.chain(
            optax.clip_by_global_norm(args.max_grad_norm),
            optax.inject_hyperparams(optax.adam)(
                learning_rate=linear_schedule if args.anneal_lr else args.learning_rate, eps=1e-5
            ),
        ),
    )
    actor.apply = jax.jit(actor.apply)
    critic.apply = jax.jit(critic.apply)

    # ALGO Logic: Storage setup
    storage = Storage(
        obs=jnp.zeros((args.num_steps, args.num_envs) + (env.observation_size,)),
        actions=jnp.zeros((args.num_steps, args.num_envs) + (env.action_size,)),
        logprobs=jnp.zeros((args.num_steps, args.num_envs)),
        dones=jnp.zeros((args.num_steps, args.num_envs)),
        values=jnp.zeros((args.num_steps, args.num_envs)),
        advantages=jnp.zeros((args.num_steps, args.num_envs)),
        returns=jnp.zeros((args.num_steps, args.num_envs)),
        rewards=jnp.zeros((args.num_steps, args.num_envs)),
    )

    @jax.jit
    def get_action_and_value(
        agent_state: TrainState,
        next_obs: np.ndarray,
        next_done: np.ndarray,
        storage: Storage,
        step: int,
        key: jax.random.PRNGKey,
    ):
        action_mean, action_logstd = actor.apply(agent_state.params.actor_params, next_obs)
        action_std = jnp.exp(action_logstd)
        key, subkey = jax.random.split(key)
        action = action_mean + action_std * jax.random.normal(subkey, shape=action_mean.shape)
        logprob = -0.5 * ((action - action_mean) / action_std) ** 2 - 0.5 * jnp.log(2.0 * jnp.pi) - action_logstd
        value = critic.apply(agent_state.params.critic_params, next_obs)
        storage = storage.replace(
            obs=storage.obs.at[step].set(next_obs),
            dones=storage.dones.at[step].set(next_done),
            actions=storage.actions.at[step].set(action),
            logprobs=storage.logprobs.at[step].set(logprob.sum(1)),
            values=storage.values.at[step].set(value.squeeze()),
        )
        return storage, action, key

    @jax.jit
    def get_action_and_value2(
        params: flax.core.FrozenDict,
        x: np.ndarray,
        action: np.ndarray,
    ):
        action_mean, action_logstd = actor.apply(params.actor_params, x)
        action_std = jnp.exp(action_logstd)
        logprob = -0.5 * ((action - action_mean) / action_std) ** 2 - 0.5 * jnp.log(2.0 * jnp.pi) - action_logstd
        entropy = action_logstd + 0.5 * jnp.log(2.0 * jnp.pi * jnp.e)
        value = critic.apply(params.critic_params, x).squeeze()
        return logprob.sum(1), entropy, value

    @jax.jit
    def compute_gae(
        agent_state: TrainState,
        next_obs: np.ndarray,
        next_done: np.ndarray,
        storage: Storage,
    ):
        storage = storage.replace(advantages=storage.advantages.at[:].set(0.0))
        next_value = critic.apply(agent_state.params.critic_params, next_obs).squeeze()
        lastgaelam = 0
        for t in reversed(range(args.num_steps)):
            if t == args.num_steps - 1:
                nextnonterminal = 1.0 - next_done
                nextvalues = next_value
            else:
                nextnonterminal = 1.0 - storage.dones[t + 1]
                nextvalues = storage.values[t + 1]
            delta = storage.rewards[t] + args.gamma * nextvalues * nextnonterminal - storage.values[t]
            lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
            storage = storage.replace(advantages=storage.advantages.at[t].set(lastgaelam))
        storage = storage.replace(returns=storage.advantages + storage.values)
        return storage

    @jax.jit
    def update_ppo(
        agent_state: TrainState,
        storage: Storage,
        key: jax.random.PRNGKey,
    ):
        b_obs = storage.obs.reshape((-1,) + (env.observation_size,))
        b_logprobs = storage.logprobs.reshape(-1)
        b_actions = storage.actions.reshape((-1,) + (env.action_size,))
        b_advantages = storage.advantages.reshape(-1)
        b_returns = storage.returns.reshape(-1)

        def ppo_loss(params, x, a, logp, mb_advantages, mb_returns):
            newlogprob, entropy, newvalue = get_action_and_value2(params, x, a)
            logratio = newlogprob - logp
            ratio = jnp.exp(logratio)
            approx_kl = ((ratio - 1) - logratio).mean()

            if args.norm_adv:
                mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

            # Policy loss
            pg_loss1 = -mb_advantages * ratio
            pg_loss2 = -mb_advantages * jnp.clip(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
            pg_loss = jnp.maximum(pg_loss1, pg_loss2).mean()

            # Value loss
            v_loss = 0.5 * ((newvalue - mb_returns) ** 2).mean()

            entropy_loss = entropy.mean()
            loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef
            return loss, (pg_loss, v_loss, entropy_loss, jax.lax.stop_gradient(approx_kl))

        ppo_loss_grad_fn = jax.value_and_grad(ppo_loss, has_aux=True)
        for _ in range(args.update_epochs):
            key, subkey = jax.random.split(key)
            b_inds = jax.random.permutation(subkey, args.batch_size, independent=True)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]
                (loss, (pg_loss, v_loss, entropy_loss, approx_kl)), grads = ppo_loss_grad_fn(
                    agent_state.params,
                    b_obs[mb_inds],
                    b_actions[mb_inds],
                    b_logprobs[mb_inds],
                    b_advantages[mb_inds],
                    b_returns[mb_inds],
                )
                agent_state = agent_state.apply_gradients(grads=grads)
        return agent_state, loss, pg_loss, v_loss, entropy_loss, approx_kl, key

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()
    # next_obs = next_obs
    next_done = np.zeros(args.num_envs)

    @jax.jit
    def rollout(agent_state, episode_stats, next_obs, next_done, storage, key, handle, global_step):
        for step in range(0, args.num_steps):
            global_step += 1 * args.num_envs
            storage, action, key = get_action_and_value(agent_state, next_obs, next_done, storage, step, key)

            # TRY NOT TO MODIFY: execute the game and log data.
            episode_stats, handle, (next_obs, reward, next_done, _) = step_env_wrappeed(episode_stats, handle, action)
            storage = storage.replace(rewards=storage.rewards.at[step].set(reward))
        return agent_state, episode_stats, next_obs, next_done, storage, key, handle, global_step

    for update in range(1, args.num_updates + 1):
        update_time_start = time.time()
        agent_state, episode_stats, next_obs, next_done, storage, key, handle, global_step = rollout(
            agent_state, episode_stats, next_obs, next_done, storage, key, handle, global_step
        )
        storage = compute_gae(agent_state, next_obs, next_done, storage)
        agent_state, loss, pg_loss, v_loss, entropy_loss, approx_kl, key = update_ppo(
            agent_state,
            storage,
            key,
        )
        avg_episodic_return = np.mean(jax.device_get(episode_stats.returned_episode_returns))
        print(f"global_step={global_step}, avg_episodic_return={avg_episodic_return}")

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        writer.add_scalar("charts/avg_episodic_return", avg_episodic_return, global_step)
        writer.add_scalar(
            "charts/avg_episodic_length", np.mean(jax.device_get(episode_stats.returned_episode_lengths)), global_step
        )
        writer.add_scalar("charts/learning_rate", agent_state.opt_state[1].hyperparams["learning_rate"].item(), global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/loss", loss.item(), global_step)
        print("SPS:", int(global_step / (time.time() - start_time)))
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)
        writer.add_scalar(
            "charts/SPS_update", int(args.num_envs * args.num_steps / (time.time() - update_time_start)), global_step
        )

    # envs.close()
    writer.close()
