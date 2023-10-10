# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppo_continuous_actionpy
# https://gregorygundersen.com/blog/2020/02/09/log-sum-exp/
import argparse
from functools import partial
import os
import random
import time
from distutils.util import strtobool
from typing import List, Sequence

import gymnasium as gym
import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax
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
    parser.add_argument("--env-id", type=str, default="Ant-v4",
        help="the id of the environment")
    parser.add_argument("--total-timesteps", type=int, default=10000000,
        help="total timesteps of the experiments")
    parser.add_argument("--learning-rate", type=float, default=0.00295,
        help="the learning rate of the optimizer")
    parser.add_argument("--num-envs", type=int, default=64,
        help="the number of parallel game environments")
    parser.add_argument("--num-steps", type=int, default=64,
        help="the number of steps to run in each environment per policy rollout")
    parser.add_argument("--anneal-lr", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Toggle learning rate annealing for policy and value networks")
    parser.add_argument("--gamma", type=float, default=0.99,
        help="the discount factor gamma")
    parser.add_argument("--gae-lambda", type=float, default=0.95,
        help="the lambda for the general advantage estimation")
    parser.add_argument("--num-minibatches", type=int, default=4,
        help="the number of mini-batches")
    parser.add_argument("--update-epochs", type=int, default=2,
        help="the K epochs to update the policy")
    parser.add_argument("--norm-adv", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Toggles advantages normalization")
    parser.add_argument("--clip-coef", type=float, default=0.2,
        help="the surrogate clipping coefficient")
    parser.add_argument("--clip-vloss", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="Toggles whether or not to use a clipped loss for the value function, as per the paper.")
    parser.add_argument("--ent-coef", type=float, default=0.0,
        help="coefficient of the entropy")
    parser.add_argument("--vf-coef", type=float, default=1.3,
        help="coefficient of the value function")
    parser.add_argument("--max-grad-norm", type=float, default=3.5,
        help="the maximum norm for the gradient clipping")
    parser.add_argument("--target-kl", type=float, default=None,
        help="the target KL divergence threshold")
    args = parser.parse_args()
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_updates = args.total_timesteps // args.batch_size
    # fmt: on
    return args


def make_env(env_id, idx, capture_video, run_name):
    def thunk():
        if capture_video:
            env = gym.make(env_id, render_mode="rgb_array")
        else:
            env = gym.make(env_id)
        env = gym.wrappers.FlattenObservation(env)  # deal with dm_control's Dict observation space
        env = gym.wrappers.RecordEpisodeStatistics(env)
        if capture_video:
            if idx == 0:
                env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        return env

    return thunk



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
            # monitor_gym=True, no longer works for gymnasium
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
    key, actor_key, critic_key = jax.random.split(key, 3)

    envs = gym.vector.AsyncVectorEnv(
        [make_env(args.env_id, i, args.capture_video, run_name) for i in range(args.num_envs)]
    )
    envs = gym.wrappers.ClipAction(envs)
    envs = gym.wrappers.NormalizeObservation(envs)
    envs = gym.wrappers.TransformObservation(envs, lambda obs: np.clip(obs, -10, 10))
    envs = gym.wrappers.NormalizeReward(envs, gamma=args.gamma)
    envs = gym.wrappers.TransformReward(envs, lambda reward: np.clip(reward, -10, 10))
    assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"

    def linear_schedule(count):
        # anneal learning rate linearly after one training iteration which contains
        # (args.num_minibatches * args.update_epochs) gradient updates
        frac = 1.0 - (count // (args.num_minibatches * args.update_epochs)) / args.num_updates
        return args.learning_rate * frac

    actor = Actor(action_dim=np.prod(envs.single_action_space.shape))
    critic = Critic()
    agent_state = TrainState.create(
        apply_fn=None,
        params=AgentParams(
            actor.init(actor_key, envs.single_observation_space.sample()),
            critic.init(critic_key, envs.single_observation_space.sample()),
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

    @jax.jit
    def get_action_and_value(
        agent_state: TrainState,
        next_obs: np.ndarray,
        key: jax.random.PRNGKey,
    ):
        """sample action, calculate value, logprob"""
        action_mean, action_logstd = actor.apply(agent_state.params.actor_params, next_obs)
        action_std = jnp.exp(action_logstd)
        key, subkey = jax.random.split(key)
        action = action_mean + action_std * jax.random.normal(subkey, shape=action_mean.shape)
        logprob = -0.5 * ((action - action_mean) / action_std) ** 2 - 0.5 * jnp.log(2.0 * jnp.pi) - action_logstd
        value = critic.apply(agent_state.params.critic_params, next_obs)
        return action, logprob.sum(1), value.squeeze(1), key

    @jax.jit
    def get_action_and_value2(
        params: flax.core.FrozenDict,
        x: np.ndarray,
        action: np.ndarray,
    ):
        """calculate value, logprob of supplied `action`, and entropy"""
        action_mean, action_logstd = actor.apply(params.actor_params, x)
        action_std = jnp.exp(action_logstd)
        logprob = -0.5 * ((action - action_mean) / action_std) ** 2 - 0.5 * jnp.log(2.0 * jnp.pi) - action_logstd
        entropy = action_logstd + 0.5 * jnp.log(2.0 * jnp.pi * jnp.e)
        value = critic.apply(params.critic_params, x).squeeze()
        return logprob.sum(1), entropy, value

    def compute_gae_once(carry, inp, gamma, gae_lambda):
        advantages = carry
        nextdone, nextvalues, curvalues, reward = inp
        nextnonterminal = 1.0 - nextdone

        delta = reward + gamma * nextvalues * nextnonterminal - curvalues
        advantages = delta + gamma * gae_lambda * nextnonterminal * advantages
        return advantages, advantages

    compute_gae_once = partial(compute_gae_once, gamma=args.gamma, gae_lambda=args.gae_lambda)

    @jax.jit
    def compute_gae(
        agent_state: TrainState,
        next_obs: jnp.ndarray,
        next_done: jnp.ndarray,
        rewards: jnp.ndarray,
        dones: jnp.ndarray,
        values: jnp.ndarray,
    ):
        advantages = jnp.zeros_like(rewards)
        next_value = critic.apply(agent_state.params.critic_params, next_obs).squeeze()
        advantages = jnp.zeros((args.num_envs,))
        dones = jnp.concatenate([dones, next_done[None, :]], axis=0)
        values = jnp.concatenate([values, next_value[None, :]], axis=0)
        _, advantages = jax.lax.scan(
            compute_gae_once, advantages, (dones[1:], values[1:], values[:-1], rewards), reverse=True
        )
        return advantages, advantages + values[:-1]

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

    @jax.jit
    def update_ppo(
        agent_state: TrainState,
        obs: List[np.ndarray],
        actions: List[np.ndarray],
        logprobs: List[np.ndarray],
        dones: List[np.ndarray],
        rewards: List[np.ndarray],
        values: List[np.ndarray],
        next_obs: np.ndarray,
        next_done: np.ndarray,
        key: jax.random.PRNGKey,
    ):
        obs = jnp.asarray(obs)
        actions = jnp.asarray(actions)
        logprobs = jnp.asarray(logprobs)
        dones = jnp.asarray(dones)
        values = jnp.asarray(values)
        rewards = jnp.asarray(rewards)
        advantages, returns = compute_gae(agent_state, next_obs, next_done, rewards, dones, values)
        b_obs = jnp.asarray(obs).reshape((-1,) + envs.single_observation_space.shape)
        b_logprobs = jnp.asarray(logprobs).reshape(-1)
        b_actions = jnp.asarray(actions).reshape((-1,) + envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)

        def update_epoch(carry, _):
            agent_state, key = carry
            key, subkey = jax.random.split(key)

            # taken from: https://github.com/google/brax/blob/main/brax/training/agents/ppo/train.py
            def convert_data(x: jnp.ndarray):
                x = jax.random.permutation(subkey, x)
                x = jnp.reshape(x, (args.num_minibatches, -1) + x.shape[1:])
                return x

            def update_minibatch(agent_state, minibatch):
                mb_obs, mb_actions, mb_logprobs, mb_advantages, mb_returns = minibatch
                (loss, (pg_loss, v_loss, entropy_loss, approx_kl)), grads = ppo_loss_grad_fn(
                    agent_state.params,
                    mb_obs,
                    mb_actions,
                    mb_logprobs,
                    mb_advantages,
                    mb_returns,
                )
                agent_state = agent_state.apply_gradients(grads=grads)
                return agent_state, (loss, pg_loss, v_loss, entropy_loss, approx_kl, grads)

            agent_state, (loss, pg_loss, v_loss, entropy_loss, approx_kl, grads) = jax.lax.scan(
                update_minibatch,
                agent_state,
                (
                    convert_data(b_obs),
                    convert_data(b_actions),
                    convert_data(b_logprobs),
                    convert_data(b_advantages),
                    convert_data(b_returns),
                ),
            )
            return (agent_state, key), (loss, pg_loss, v_loss, entropy_loss, approx_kl, grads)

        (agent_state, key), (loss, pg_loss, v_loss, entropy_loss, approx_kl, _) = jax.lax.scan(
            update_epoch, (agent_state, key), (), length=args.update_epochs
        )
        return agent_state, loss, pg_loss, v_loss, entropy_loss, approx_kl, advantages, returns, key

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()
    next_obs, _ = envs.reset(seed=args.seed)
    next_done = np.zeros(args.num_envs)

    for update in range(1, args.num_updates + 1):
        update_time_start = time.time()
        obs = []
        dones = []
        actions = []
        logprobs = []
        values = []
        rewards = []
        for step in range(0, args.num_steps):
            global_step += 1 * args.num_envs
            obs.append(next_obs)
            dones.append(next_done)
            action, logprob, value, key = get_action_and_value(agent_state, next_obs, key)
            actions.append(action)
            logprobs.append(logprob)
            values.append(value)
            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, next_rewards, terminated, truncated, infos = envs.step(np.array(action))
            next_done = np.logical_or(terminated, truncated)
            rewards.append(next_rewards)

            # Only print when at least 1 env is done
            if "final_info" not in infos:
                continue

            for info in infos["final_info"]:
                # Skip the envs that are not done
                if info is None:
                    continue
                print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
                writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
                writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)

        # advantages, returns = compute_gae(agent_state, next_obs, next_done, rewards, dones, values)
        agent_state, loss, pg_loss, v_loss, entropy_loss, approx_kl, advantages, returns, key = update_ppo(
            agent_state,
            obs,
            actions,
            logprobs,
            dones,
            rewards,
            values,
            next_obs,
            next_done,
            key,
        )

        writer.add_scalar("charts/advantage", np.mean(jax.device_get(advantages)), global_step)
        writer.add_scalar("charts/return", np.mean(jax.device_get(returns)), global_step)

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        writer.add_scalar("charts/learning_rate", agent_state.opt_state[1].hyperparams["learning_rate"].item(), global_step)
        writer.add_scalar("losses/value_loss", v_loss[-1, -1].item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss[-1, -1].item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss[-1, -1].item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl[-1, -1].item(), global_step)
        writer.add_scalar("losses/loss", loss[-1, -1].item(), global_step)
        print("SPS:", int(global_step / (time.time() - start_time)))
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)
        writer.add_scalar(
            "charts/SPS_update", int(args.num_envs * args.num_steps / (time.time() - update_time_start)), global_step
        )

    envs.close()
    writer.close()