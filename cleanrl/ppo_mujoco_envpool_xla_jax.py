# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppo_atari_envpool_xla_jaxpy
import os

os.environ["JAX_DEFAULT_DTYPE_BITS"] = "32"
import argparse
import os
import random
import time
from distutils.util import strtobool
from functools import partial
from typing import Sequence

import envpool
import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.linen.initializers import constant, orthogonal
from flax.training.train_state import TrainState
from jax.config import config
from torch.utils.tensorboard import SummaryWriter

config.update("jax_enable_x64", True)  # envpool only accept double type action input

os.environ[
    "XLA_PYTHON_CLIENT_MEM_FRACTION"
] = "0.7"  # see https://github.com/google/jax/discussions/6332#discussioncomment-1279991


def parse_args():
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", type=str, default=os.path.basename(__file__).rstrip(".py"),
        help="the name of this experiment")
    parser.add_argument("--seed", type=int, default=88,
        help="seed of the experiment")
    parser.add_argument("--track", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="if toggled, this experiment will be tracked with Weights and Biases")
    parser.add_argument("--wandb-project-name", type=str, default="cleanRL",
        help="the wandb's project name")
    parser.add_argument("--wandb-entity", type=str, default=None,
        help="the entity (team) of wandb's project")
    parser.add_argument("--capture-video", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="whether to capture videos of the agent performances (check out `videos` folder)")
    # Algorithm specific arguments
    parser.add_argument("--rew-norm", type=int, default=True,help="Toggles rewards normalization")
    parser.add_argument("--env-id", type=str, default="HalfCheetah-v3",
        help="the id of the environment")
    parser.add_argument("--total-timesteps", type=int, default=3000000,
        help="total timesteps of the experiments")
    parser.add_argument("--learning-rate", type=float, default=3e-4,
        help="the learning rate of the optimizer")
    parser.add_argument("--num-envs", type=int, default=64,
        help="the number of parallel game environments")
    parser.add_argument("--num-steps", type=int, default=32,
        help="the number of steps to run in each environment per policy rollout")
    parser.add_argument("--anneal-lr", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Toggle learning rate annealing for policy and value networks")
    parser.add_argument("--gamma", type=float, default=0.99,
        help="the discount factor gamma")
    parser.add_argument("--gae-lambda", type=float, default=0.95,
        help="the lambda for the general advantage estimation")
    parser.add_argument("--num-minibatches", type=int, default=32,
        help="the number of mini-batches")
    parser.add_argument("--update-epochs", type=int, default=10,
        help="the K epochs to update the policy")
    parser.add_argument("--norm-adv", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=False,
        help="Toggles advantages normalization")
    parser.add_argument("--clip-coef", type=float, default=0.2,
        help="the surrogate clipping coefficient")
    parser.add_argument("--ent-coef", type=float, default=0.0,
        help="coefficient of the entropy")
    parser.add_argument("--vf-coef", type=float, default=0.25,
        help="coefficient of the value function")
    parser.add_argument("--max-grad-norm", type=float, default=0.5,
        help="the maximum norm for the gradient clipping")
    parser.add_argument("--target-kl", type=float, default=None,
        help="the target KL divergence threshold")
    args = parser.parse_args()
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_updates = args.total_timesteps // args.batch_size
    # fmt: on
    return args


# taken from: https://github.com/thu-ml/tianshou/blob/master/tianshou/utils/statistics.py
class RunningMeanStd(flax.struct.PyTreeNode):
    eps: jnp.array = jnp.array(jnp.finfo(jnp.float64).eps.item(), dtype=jnp.float64)
    mean: jnp.array = 0.0
    var: jnp.array = 1.0
    clip_max: jnp.array = jnp.array(10.0, dtype=jnp.float64)
    count: jnp.array = jnp.array(0, dtype=jnp.int64)

    def norm(self, data_array):
        data_array = (data_array - self.mean) / jnp.sqrt(self.var + self.eps)
        data_array = jnp.clip(data_array, -self.clip_max, self.clip_max)
        return data_array

    def update(self, data_array: jnp.ndarray) -> None:
        """Add a batch of item into RMS with the same shape, modify mean/var/count."""
        batch_mean, batch_var = jnp.mean(data_array, axis=0), jnp.var(data_array, axis=0)
        batch_count = len(data_array)

        delta = batch_mean - self.mean
        total_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / total_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m_2 = m_a + m_b + delta**2 * self.count * batch_count / total_count
        new_var = m_2 / total_count
        return self.replace(mean=new_mean, var=new_var, count=total_count)


# EnvWrapper itself is a datacalss, when dealing with a stateful wrapper, we will pass EnvWrapper itself as a handle
@flax.struct.dataclass
class EnvWrapper:
    # recv will modify what env received after a action step
    def recv(self, ret):
        # the first return value should be wrapper itself (or a modified version) as a handle
        return self, ret

    def reset(self, ret):
        return self, ret

    # send will modify the action send to env
    def send(self, action):
        return action


# JAX type VectorEnvWrapper, accept list of wrapper
class VectorEnvWrapper:
    def __init__(self, envs, wrappers):
        self.envs = envs
        self.wrappers = wrappers
        self._handle, self._recv, self._send, self._step = self.envs.xla()

    def reset(self):
        result = self.envs.reset()
        handles = [self._handle]
        for wrapper in self.wrappers:
            handle, result = wrapper.reset(result)
            handles += [handle]
        return handles, result

    def xla(self):
        @jax.jit
        def send(handle: jnp.ndarray, action, env_id=None):
            for wrapper in self.wrappers:
                action = wrapper.send(action)
            return [self._send(handle[0], action, env_id)] + handle[1:]

        @jax.jit
        def recv(handles: jnp.ndarray):
            _handle, ret = self._recv(handles[0])
            new_handles = []
            for handle in reversed(handles[1:]):
                handle, ret = handle.recv(ret)
                new_handles += [handle]  # pass EnvWrapper as a handle
            # the order is reversed
            return [_handle] + list(reversed(new_handles)), ret

        def step(handle, action, env_id=None):
            return recv(send(handle, action, env_id))

        return self._handle, recv, send, step


@flax.struct.dataclass
class VectorEnvNormObs(EnvWrapper):
    obs_rms: RunningMeanStd = RunningMeanStd()

    def recv(self, ret):
        next_obs, reward, next_done, info = ret
        next_truncated = info["TimeLimit.truncated"]
        obs_rms = self.obs_rms.update(next_obs)
        return self.replace(obs_rms=obs_rms), (
            obs_rms.norm(next_obs),
            reward,
            next_done,
            next_truncated,
            info,
        )

    def reset(self, ret):
        obs = ret
        obs_rms = self.obs_rms.update(obs)
        obs = obs_rms.norm(obs).astype(jnp.float32)
        return self.replace(obs_rms=obs_rms), obs


@flax.struct.dataclass
class VectorEnvClipAct(EnvWrapper):
    action_low: jnp.array
    action_high: jnp.array

    def send(self, action):
        action_remap = jnp.clip(action, -1.0, 1.0)
        action_remap = (self.action_low + (action_remap + 1.0) * (self.action_high - self.action_low) / 2.0).astype(
            jnp.float64
        )
        return action


class Critic(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = nn.Dense(64, kernel_init=orthogonal(jnp.float_(np.sqrt(2))), bias_init=constant(0.0))(x)
        x = nn.tanh(x)
        x = nn.Dense(64, kernel_init=orthogonal(jnp.float_(np.sqrt(2))), bias_init=constant(0.0))(x)
        x = nn.tanh(x)
        return nn.Dense(1, kernel_init=orthogonal(jnp.float_(np.sqrt(2))), bias_init=constant(0.0))(x)


class Actor(nn.Module):
    action_dim: Sequence[int]

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(64, kernel_init=orthogonal(jnp.float_(np.sqrt(2))), bias_init=constant(0.0))(x)
        x = nn.tanh(x)
        x = nn.Dense(64, kernel_init=orthogonal(jnp.float_(np.sqrt(2))), bias_init=constant(0.0))(x)
        x = nn.tanh(x)
        x = nn.Dense(
            self.action_dim,
            kernel_init=orthogonal((0.01) * jnp.float_(np.sqrt(2))),
            bias_init=constant(0.0),
        )(x)
        # stdlog = -jnp.ones((self.action_dim,))/2
        stdlog = self.param("stdlog", lambda rng, shape: -jnp.ones(shape) / 2, (self.action_dim,))
        return x, stdlog


# a stateful trainstate
class PPOTrainState(TrainState):
    ret_rms: RunningMeanStd = RunningMeanStd()


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
    truncated: jnp.array
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

    # env setup
    envs = envpool.make(
        args.env_id,
        env_type="gym",
        num_envs=args.num_envs,
        seed=args.seed,
    )
    num_envs = args.num_envs
    single_action_space = envs.action_space
    single_observation_space = envs.observation_space
    envs.is_vector_env = True
    episode_stats = EpisodeStatistics(
        episode_returns=jnp.zeros(args.num_envs, dtype=jnp.float32),
        episode_lengths=jnp.zeros(args.num_envs, dtype=jnp.int32),
        returned_episode_returns=jnp.zeros(args.num_envs, dtype=jnp.float32),
        returned_episode_lengths=jnp.zeros(args.num_envs, dtype=jnp.int32),
    )
    wrappers = [
        VectorEnvNormObs(),
        VectorEnvClipAct(envs.action_space.low, envs.action_space.high),
    ]
    envs = VectorEnvWrapper(envs, wrappers)

    handle, recv, send, step_env = envs.xla()

    def step_env_wrappeed(episode_stats, handle, action):
        handle, (next_obs, reward, next_done, next_truncated, info) = step_env(handle, action)
        new_episode_return = episode_stats.episode_returns + reward
        new_episode_length = episode_stats.episode_lengths + 1
        episode_stats = episode_stats.replace(
            episode_returns=(new_episode_return) * (1 - next_done) * (1 - next_truncated),
            episode_lengths=(new_episode_length) * (1 - next_done) * (1 - next_truncated),
            # only update the `returned_episode_returns` if the episode is done
            returned_episode_returns=jnp.where(
                next_done + next_truncated,
                new_episode_return,
                episode_stats.returned_episode_returns,
            ),
            returned_episode_lengths=jnp.where(
                next_done + next_truncated,
                new_episode_length,
                episode_stats.returned_episode_lengths,
            ),
        )
        return (
            episode_stats,
            handle,
            (next_obs, reward, next_done, next_truncated, info),
        )

    def linear_schedule(count):
        # anneal learning rate linearly after one training iteration which contains
        # (args.num_minibatches * args.update_epochs) gradient updates
        frac = 1.0 - (count // (args.num_minibatches * args.update_epochs)) / args.num_updates
        return args.learning_rate * frac

    actor = Actor(
        action_dim=np.prod(single_action_space.shape),
    )
    critic = Critic()
    agent_state = PPOTrainState.create(
        apply_fn=None,
        params=AgentParams(
            actor.init(
                actor_key,
                np.array([single_observation_space.sample()], dtype=jnp.float32),
            ),
            critic.init(
                critic_key,
                np.array([single_observation_space.sample()], dtype=jnp.float32),
            ),
        ),
        tx=optax.chain(
            optax.clip_by_global_norm(args.max_grad_norm),
            optax.inject_hyperparams(optax.adam)(learning_rate=linear_schedule if args.anneal_lr else args.learning_rate),
        ),
        ret_rms=RunningMeanStd(),
    )
    actor.apply = jax.jit(actor.apply)
    critic.apply = jax.jit(critic.apply)

    @jax.jit
    def get_action_and_value(
        agent_state: TrainState,
        next_obs: np.ndarray,
        key: jax.random.PRNGKey,
    ):
        """sample action, calculate value, logprob, entropy, and update storage"""
        logits, stdlog = actor.apply(agent_state.params.actor_params, next_obs)
        key, subkey = jax.random.split(key)
        u = jax.random.normal(subkey, shape=logits.shape)
        action = logits + u * jnp.exp(stdlog)
        var = jnp.exp(2 * stdlog)
        logprob = (-((action - logits) ** 2) / (2 * var) - stdlog - jnp.log(jnp.sqrt(2 * jnp.pi))).sum(-1)
        value = critic.apply(agent_state.params.critic_params, next_obs)
        return action, logprob, value.squeeze(1), key

    @jax.jit
    def get_action_and_value2(
        params: flax.core.FrozenDict,
        x: np.ndarray,
        action: np.ndarray,
    ):
        """calculate value, logprob of supplied `action`, and entropy"""
        logits, stdlog = actor.apply(params.actor_params, x)
        var = jnp.exp(2 * stdlog)
        logprob = (-((action - logits) ** 2) / (2 * var) - stdlog - jnp.log(jnp.sqrt(2 * jnp.pi))).sum(-1)
        entropy = (2 * stdlog + jnp.log(2 * jnp.pi) + 1) / 2
        value = critic.apply(params.critic_params, x).squeeze()
        return logprob, entropy.sum(-1), value

    def compute_gae_once(carry, inp, gamma, gae_lambda):
        advantages = carry
        nextdone, nexttruncated, nextvalues, curvalues, reward = inp
        nextnonterminal = (1.0 - nextdone) * (1.0 - nexttruncated)

        delta = reward + gamma * nextvalues - curvalues
        advantages = delta + gamma * gae_lambda * nextnonterminal * advantages
        return advantages, advantages

    compute_gae_once = partial(compute_gae_once, gamma=args.gamma, gae_lambda=args.gae_lambda)

    @jax.jit
    def compute_gae(
        agent_state: TrainState,
        next_obs: np.ndarray,
        next_done: np.ndarray,
        next_truncated: np.ndarray,
        storage: Storage,
    ):
        values = critic.apply(
            agent_state.params.critic_params,
            jnp.concatenate([storage.obs, next_obs[None, :]], axis=0),
        ).squeeze()
        if args.rew_norm:
            values = values * jnp.sqrt(agent_state.ret_rms.var).astype(jnp.float32)
        advantages = jnp.zeros((args.num_envs,))
        dones = jnp.concatenate([storage.dones, next_done[None, :]], axis=0)
        truncated = jnp.concatenate([storage.truncated, next_truncated[None, :]], axis=0)
        _, advantages = jax.lax.scan(
            compute_gae_once,
            advantages,
            (
                dones[1:],
                truncated[1:],
                values[1:] * (1.0 - dones[1:]),
                values[:-1],
                storage.rewards,
            ),
            reverse=True,
        )
        returns = advantages + values[:-1]
        if args.rew_norm:
            returns = (returns / jnp.sqrt(agent_state.ret_rms.var + 10e-8)).astype(jnp.float32)
            agent_state = agent_state.replace(ret_rms=agent_state.ret_rms.update(returns.flatten()))

        storage = storage.replace(
            advantages=advantages,
            returns=returns,
        )
        return storage, agent_state

    def ppo_loss(params, x, a, logp, mb_advantages, mb_returns, truncated):
        newlogprob, entropy, newvalue = get_action_and_value2(params, x, a)
        logratio = newlogprob - logp
        ratio = jnp.exp(logratio)
        approx_kl = ((ratio - 1) - logratio).mean()
        if args.norm_adv:
            mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)
        # Policy loss
        pg_loss1 = -mb_advantages * ratio
        pg_loss2 = -mb_advantages * jnp.clip(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
        # mask truncated state
        pg_loss = (jnp.maximum(pg_loss1, pg_loss2) * (1 - truncated)).sum() / (1 - truncated).sum()

        # Value loss
        v_loss = (((newvalue - mb_returns) * (1 - truncated)) ** 2).sum() / (1 - truncated).sum()

        entropy_loss = entropy.mean()
        loss = pg_loss + v_loss * args.vf_coef
        return loss, (pg_loss, v_loss, entropy_loss, jax.lax.stop_gradient(approx_kl))

    ppo_loss_grad_fn = jax.value_and_grad(ppo_loss, has_aux=True)

    @jax.jit
    def update_ppo(
        agent_state: TrainState,
        next_obs: np.ndarray,
        next_done: np.ndarray,
        next_truncated: np.ndarray,
        storage: Storage,
        key: jax.random.PRNGKey,
    ):
        def update_epoch(carry, unused_inp):
            agent_state, key = carry
            key, subkey = jax.random.split(key)

            def flatten(x):
                return x.reshape((-1,) + x.shape[2:])

            # taken from: https://github.com/google/brax/blob/main/brax/training/agents/ppo/train.py
            def convert_data(x: jnp.ndarray):
                x = jax.random.permutation(subkey, x)
                x = jnp.reshape(x, (args.num_minibatches, -1) + x.shape[1:])
                return x

            newstorage, agent_state = compute_gae(agent_state, next_obs, next_done, next_truncated, storage)
            flatten_storage = jax.tree_map(flatten, newstorage)  # seem uneffcient
            shuffled_storage = jax.tree_map(convert_data, flatten_storage)

            def update_minibatch(agent_state, minibatch):
                (loss, (pg_loss, v_loss, entropy_loss, approx_kl),), grads = ppo_loss_grad_fn(
                    agent_state.params,
                    minibatch.obs,
                    minibatch.actions,
                    minibatch.logprobs,
                    minibatch.advantages,
                    minibatch.returns,
                    minibatch.truncated,
                )
                agent_state = agent_state.apply_gradients(grads=grads)
                return agent_state, (
                    loss,
                    pg_loss,
                    v_loss,
                    entropy_loss,
                    approx_kl,
                    grads,
                )

            agent_state, (
                loss,
                pg_loss,
                v_loss,
                entropy_loss,
                approx_kl,
                grads,
            ) = jax.lax.scan(update_minibatch, agent_state, shuffled_storage)
            return (agent_state, key), (
                loss,
                pg_loss,
                v_loss,
                entropy_loss,
                approx_kl,
                grads,
            )

        (agent_state, key), (
            loss,
            pg_loss,
            v_loss,
            entropy_loss,
            approx_kl,
            grads,
        ) = jax.lax.scan(update_epoch, (agent_state, key), (), length=args.update_epochs)
        return agent_state, loss, pg_loss, v_loss, entropy_loss, approx_kl, key

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()
    handle, next_obs = envs.reset()
    next_obs = next_obs.astype(jnp.float32)
    next_done = jnp.zeros(args.num_envs, dtype=jax.numpy.bool_)
    next_truncated = jnp.zeros(args.num_envs, dtype=jax.numpy.bool_)
    # based on https://github.dev/google/evojax/blob/0625d875262011d8e1b6aa32566b236f44b4da66/evojax/sim_mgr.py
    def step_once(carry, step, env_step_fn):
        agent_state, episode_stats, obs, done, truncated, key, handle = carry
        action, logprob, value, key = get_action_and_value(agent_state, obs, key)
        (
            episode_stats,
            handle,
            (next_obs, reward, next_done, next_truncated, _),
        ) = env_step_fn(episode_stats, handle, action.astype(jnp.float64))
        next_obs = next_obs.astype(jnp.float32)
        reward.astype(jnp.float32)
        storage = Storage(
            obs=obs,
            actions=action,
            logprobs=logprob,
            dones=done,
            truncated=truncated,
            values=value,
            rewards=reward,
            returns=jnp.zeros_like(reward),
            advantages=jnp.zeros_like(reward),
        )
        return (
            (
                agent_state,
                episode_stats,
                next_obs,
                next_done,
                next_truncated,
                key,
                handle,
            ),
            storage,
        )

    def rollout(
        agent_state,
        episode_stats,
        next_obs,
        next_done,
        next_truncated,
        key,
        handle,
        step_once_fn,
        max_steps,
    ):
        (agent_state, episode_stats, next_obs, next_done, next_truncated, key, handle,), storage = jax.lax.scan(
            step_once_fn,
            (
                agent_state,
                episode_stats,
                next_obs,
                next_done,
                next_truncated,
                key,
                handle,
            ),
            (),
            max_steps,
        )
        return (
            agent_state,
            episode_stats,
            next_obs,
            next_done,
            next_truncated,
            storage,
            key,
            handle,
        )

    rollout = partial(
        rollout,
        step_once_fn=partial(step_once, env_step_fn=step_env_wrappeed),
        max_steps=args.num_steps,
    )
    for update in range(1, args.num_updates + 1):
        update_time_start = time.time()
        (
            agent_state,
            episode_stats,
            next_obs,
            next_done,
            next_truncated,
            storage,
            key,
            handle,
        ) = rollout(agent_state, episode_stats, next_obs, next_done, next_truncated, key, handle)
        global_step += args.num_steps * args.num_envs
        agent_state, loss, pg_loss, v_loss, entropy_loss, approx_kl, key = update_ppo(
            agent_state,
            next_obs,
            next_done,
            next_truncated,
            storage,
            key,
        )
        print(f"pg_loss={pg_loss.mean()}, loss={loss.mean()}, v_loss={v_loss.mean()}, entropy_loss={entropy_loss.mean()}")
        avg_episodic_return = np.mean(jax.device_get(episode_stats.returned_episode_returns))
        avg_episodic_length = np.mean(jax.device_get(episode_stats.returned_episode_lengths))
        print(
            f"global_step={global_step}, avg_episodic_length={avg_episodic_length}, avg_episodic_return={avg_episodic_return}, SPS_update={int(args.num_steps * args.num_envs / (time.time() - update_time_start))}"
        )

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        writer.add_scalar("charts/avg_episodic_return", avg_episodic_return, global_step)
        writer.add_scalar("charts/avg_episodic_length", avg_episodic_length, global_step)
        writer.add_scalar("charts/learning_rate", agent_state.opt_state[1].hyperparams["learning_rate"].item(), global_step)
        writer.add_scalar("losses/value_loss", v_loss.mean().item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.mean().item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.mean().item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.mean().item(), global_step)
        writer.add_scalar("losses/loss", loss.mean().item(), global_step)
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)
        writer.add_scalar(
            "charts/SPS_update", int(args.num_envs * args.num_steps / (time.time() - update_time_start)), global_step
        )

    envs.close()
    writer.close()
