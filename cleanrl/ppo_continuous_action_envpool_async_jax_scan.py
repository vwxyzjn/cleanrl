# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppo_atari_envpool_async_jax_scan_impalanet_machadopy
# https://gregorygundersen.com/blog/2020/02/09/log-sum-exp/
import argparse
import os
import random
import time
from distutils.util import strtobool
from typing import Sequence

os.environ[
    "XLA_PYTHON_CLIENT_MEM_FRACTION"
] = "0.7"  # see https://github.com/google/jax/discussions/6332#discussioncomment-1279991

import envpool
import flax
import flax.linen as nn
import gym
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
    parser.add_argument("--save-model", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="whether to save model into the `runs/{run_name}` folder")
    parser.add_argument("--upload-model", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="whether to upload the saved model to huggingface")
    parser.add_argument("--hf-entity", type=str, default="",
        help="the user or org name of the model repository from the Hugging Face Hub")

    # Algorithm specific arguments
    parser.add_argument("--env-id", type=str, default="Ant-v4",
        help="the id of the environment")
    parser.add_argument("--total-timesteps", type=int, default=20000000,
        help="total timesteps of the experiments")
    parser.add_argument("--learning-rate", type=float, default=0.00295,
        help="the learning rate of the optimizer")
    parser.add_argument("--num-envs", type=int, default=128,
        help="the number of parallel game environments")
    parser.add_argument("--async-batch-size", type=int, default=32,
        help="the envpool's batch size in the async mode")
    parser.add_argument("--num-steps", type=int, default=64,
        help="the number of steps to run in each environment per policy rollout")
    parser.add_argument("--anneal-lr", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Toggle learning rate annealing for policy and value networks")
    parser.add_argument("--gae", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Use GAE for advantage computation")
    parser.add_argument("--gamma", type=float, default=0.99,
        help="the discount factor gamma")
    parser.add_argument("--gae-lambda", type=float, default=0.95,
        help="the lambda for the general advantage estimation")
    parser.add_argument("--num-minibatches", type=int, default=2,
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


def make_env(env_id, seed, num_envs, async_batch_size=1):
    def thunk():
        envs = envpool.make(
            env_id,
            env_type="gym",
            num_envs=num_envs,
            batch_size=async_batch_size,
            seed=seed,
        )
        envs = gym.wrappers.FlattenObservation(envs)  # deal with dm_control's Dict observation space
        envs.num_envs = num_envs
        envs.single_action_space = envs.action_space
        envs.single_observation_space = envs.observation_space
        envs.is_vector_env = True
        return envs

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
    key, actor_key, critic_key = jax.random.split(key, 3)

    # env setup
    envs = make_env(args.env_id, args.seed, args.num_envs, args.async_batch_size)()
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
        """sample action, calculate value, logprob, entropy, and update storage"""
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

    def compute_gae_once(carry, x):
        lastvalues, lastdones, advantages, lastgaelam, final_env_ids, final_env_id_checked = carry
        (
            done,
            value,
            eid,
            reward,
        ) = x
        nextnonterminal = 1.0 - lastdones[eid]
        nextvalues = lastvalues[eid]
        delta = jnp.where(final_env_id_checked[eid] == -1, 0, reward + args.gamma * nextvalues * nextnonterminal - value)
        advantages = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam[eid]
        final_env_ids = jnp.where(final_env_id_checked[eid] == 1, 1, 0)
        final_env_id_checked = final_env_id_checked.at[eid].set(
            jnp.where(final_env_id_checked[eid] == -1, 1, final_env_id_checked[eid])
        )

        # the last_ variables keeps track of the actual `num_steps`
        lastgaelam = lastgaelam.at[eid].set(advantages)
        lastdones = lastdones.at[eid].set(done)
        lastvalues = lastvalues.at[eid].set(value)
        return (lastvalues, lastdones, advantages, lastgaelam, final_env_ids, final_env_id_checked), (
            advantages,
            final_env_ids,
        )

    @jax.jit
    def compute_gae(
        env_ids: np.ndarray,
        rewards: np.ndarray,
        values: np.ndarray,
        dones: np.ndarray,
    ):
        dones = jnp.asarray(dones)
        values = jnp.asarray(values)
        env_ids = jnp.asarray(env_ids)
        rewards = jnp.asarray(rewards)

        _, B = env_ids.shape
        final_env_id_checked = jnp.zeros(args.num_envs, jnp.int32) - 1
        final_env_ids = jnp.zeros(B, jnp.int32)
        advantages = jnp.zeros(B)
        lastgaelam = jnp.zeros(args.num_envs)
        lastdones = jnp.zeros(args.num_envs) + 1
        lastvalues = jnp.zeros(args.num_envs)

        (_, _, _, _, final_env_ids, final_env_id_checked), (advantages, final_env_ids) = jax.lax.scan(
            compute_gae_once,
            (
                lastvalues,
                lastdones,
                advantages,
                lastgaelam,
                final_env_ids,
                final_env_id_checked,
            ),
            (
                dones,
                values,
                env_ids,
                rewards,
            ),
            reverse=True,
        )
        return advantages, advantages + values, final_env_id_checked, final_env_ids

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
        obs: list,
        dones: list,
        values: list,
        actions: list,
        logprobs: list,
        env_ids: list,
        rewards: list,
        key: jax.random.PRNGKey,
    ):
        obs = jnp.asarray(obs)
        dones = jnp.asarray(dones)
        values = jnp.asarray(values)
        actions = jnp.asarray(actions)
        logprobs = jnp.asarray(logprobs)
        env_ids = jnp.asarray(env_ids)
        rewards = jnp.asarray(rewards)

        # TODO: in an unlikely event, one of the envs might have not stepped at all, which may results in unexpected behavior
        T, B = env_ids.shape
        index_ranges = jnp.arange(T * B, dtype=jnp.int32)
        next_index_ranges = jnp.zeros_like(index_ranges, dtype=jnp.int32)
        last_env_ids = jnp.zeros(args.num_envs, dtype=jnp.int32) - 1

        def f(carry, x):
            last_env_ids, next_index_ranges = carry
            env_id, index_range = x
            next_index_ranges = next_index_ranges.at[last_env_ids[env_id]].set(
                jnp.where(last_env_ids[env_id] != -1, index_range, next_index_ranges[last_env_ids[env_id]])
            )
            last_env_ids = last_env_ids.at[env_id].set(index_range)
            return (last_env_ids, next_index_ranges), None

        (last_env_ids, next_index_ranges), _ = jax.lax.scan(
            f,
            (last_env_ids, next_index_ranges),
            (env_ids.reshape(-1), index_ranges),
        )

        # rewards is off by one time step
        rewards = rewards.reshape(-1)[next_index_ranges].reshape((args.num_steps) * async_update, args.async_batch_size)
        advantages, returns, _, final_env_ids = compute_gae(env_ids, rewards, values, dones)
        b_inds = jnp.nonzero(final_env_ids.reshape(-1), size=(args.num_steps) * async_update * args.async_batch_size)[0]
        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_logprobs = logprobs.reshape(-1)
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
        return agent_state, loss, pg_loss, v_loss, entropy_loss, approx_kl, advantages, returns, b_inds, final_env_ids, key

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()
    async_update = int(args.num_envs / args.async_batch_size)

    # put data in the last index
    episode_returns = np.zeros((args.num_envs,), dtype=np.float32)
    returned_episode_returns = np.zeros((args.num_envs,), dtype=np.float32)
    episode_lengths = np.zeros((args.num_envs,), dtype=np.float32)
    returned_episode_lengths = np.zeros((args.num_envs,), dtype=np.float32)
    envs.async_reset()
    final_env_ids = np.zeros((async_update, args.async_batch_size), dtype=np.int32)

    for update in range(1, args.num_updates + 2):
        update_time_start = time.time()
        obs = []
        dones = []
        actions = []
        logprobs = []
        values = []
        env_ids = []
        rewards = []
        truncations = []
        terminations = []
        env_recv_time = 0
        inference_time = 0
        storage_time = 0
        env_send_time = 0

        # NOTE: This is a major difference from the sync version:
        # at the end of the rollout phase, the sync version will have the next observation
        # ready for the value bootstrap, but the async version will not have it.
        # for this reason we do `num_steps + 1`` to get the extra states for value bootstrapping.
        # but note that the extra states are not used for the loss computation in the next iteration,
        # while the sync version will use the extra state for the loss computation.
        for step in range(
            async_update, (args.num_steps + 1) * async_update
        ):  # num_steps + 1 to get the states for value bootstrapping.
            env_recv_time_start = time.time()
            next_obs, next_reward, next_done, info = envs.recv()
            if type(next_obs) == dict: # support dict observations
                next_obs = np.concatenate(list(next_obs.values()), -1)
            env_recv_time += time.time() - env_recv_time_start
            global_step += len(next_done)
            env_id = info["env_id"]

            inference_time_start = time.time()
            action, logprob, value, key = get_action_and_value(agent_state, next_obs, key)
            inference_time += time.time() - inference_time_start

            env_send_time_start = time.time()
            envs.send(np.array(action), env_id)
            env_send_time += time.time() - env_send_time_start
            storage_time_start = time.time()
            obs.append(next_obs)
            dones.append(next_done)
            values.append(value)
            actions.append(action)
            logprobs.append(logprob)
            env_ids.append(env_id)
            rewards.append(next_reward)
            truncations.append(info["TimeLimit.truncated"])
            terminations.append(next_done)
            episode_returns[env_id] += next_reward
            returned_episode_returns[env_id] = np.where(
                next_done + info["TimeLimit.truncated"], episode_returns[env_id], returned_episode_returns[env_id]
            )
            episode_returns[env_id] *= (1 - next_done) * (1 - info["TimeLimit.truncated"])
            episode_lengths[env_id] += 1
            returned_episode_lengths[env_id] = np.where(
                next_done + info["TimeLimit.truncated"], episode_lengths[env_id], returned_episode_lengths[env_id]
            )
            episode_lengths[env_id] *= (1 - next_done) * (1 - info["TimeLimit.truncated"])
            storage_time += time.time() - storage_time_start

        avg_episodic_return = np.mean(returned_episode_returns)
        # print(returned_episode_returns)
        print(f"global_step={global_step}, avg_episodic_return={avg_episodic_return}")
        writer.add_scalar("charts/avg_episodic_return", avg_episodic_return, global_step)
        writer.add_scalar("charts/avg_episodic_length", np.mean(returned_episode_lengths), global_step)
        training_time_start = time.time()
        (
            agent_state,
            loss,
            pg_loss,
            v_loss,
            entropy_loss,
            approx_kl,
            advantages,
            returns,
            b_inds,
            final_env_ids,
            key,
        ) = update_ppo(
            agent_state,
            obs,
            dones,
            values,
            actions,
            logprobs,
            env_ids,
            rewards,
            key,
        )
        writer.add_scalar("stats/training_time", time.time() - training_time_start, global_step)
        # writer.add_scalar("stats/advantages", advantages.mean().item(), global_step)
        # writer.add_scalar("stats/returns", returns.mean().item(), global_step)
        writer.add_scalar("stats/truncations", np.sum(truncations), global_step)
        writer.add_scalar("stats/terminations", np.sum(terminations), global_step)

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
        writer.add_scalar("stats/env_recv_time", env_recv_time, global_step)
        writer.add_scalar("stats/inference_time", inference_time, global_step)
        writer.add_scalar("stats/storage_time", storage_time, global_step)
        writer.add_scalar("stats/env_send_time", env_send_time, global_step)
        writer.add_scalar("stats/update_time", time.time() - update_time_start, global_step)

    if args.save_model:
        model_path = f"runs/{run_name}/{args.exp_name}.cleanrl_model"
        with open(model_path, "wb") as f:
            f.write(
                flax.serialization.to_bytes(
                    [
                        vars(args),
                        [
                            agent_state.params.network_params,
                            agent_state.params.actor_params,
                            agent_state.params.critic_params,
                        ],
                    ]
                )
            )
        print(f"model saved to {model_path}")
        from cleanrl_utils.evals.ppo_envpool_jax_eval import evaluate

        episodic_returns = evaluate(
            model_path,
            make_env,
            args.env_id,
            eval_episodes=10,
            run_name=f"{run_name}-eval",
            Model=(Network, Actor, Critic),
        )
        for idx, episodic_return in enumerate(episodic_returns):
            writer.add_scalar("eval/episodic_return", episodic_return, idx)

        if args.upload_model:
            from cleanrl_utils.huggingface import push_to_hub

            repo_name = f"{args.env_id}-{args.exp_name}-seed{args.seed}"
            repo_id = f"{args.hf_entity}/{repo_name}" if args.hf_entity else repo_name
            push_to_hub(args, episodic_returns, repo_id, "PPO", f"runs/{run_name}", f"videos/{run_name}-eval")

    envs.close()
    writer.close()
