# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/muesli/#muesli_atari_envpool_async_jax_scan_impalanet_machadopy
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
        help="weather to capture videos of the agent performances (check out `videos` folder)")
    parser.add_argument("--save-model", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="whether to save model into the `runs/{run_name}` folder")
    parser.add_argument("--upload-model", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="whether to upload the saved model to huggingface")
    parser.add_argument("--hf-entity", type=str, default="",
        help="the user or org name of the model repository from the Hugging Face Hub")

    # Algorithm specific arguments
    parser.add_argument("--env-id", type=str, default="Breakout-v5",
        help="the id of the environment")
    parser.add_argument("--total-timesteps", type=int, default=50000000,
        help="total timesteps of the experiments")
    parser.add_argument("--learning-rate", type=float, default=2.5e-4,
        help="the learning rate of the optimizer")
    parser.add_argument("--num-envs", type=int, default=64,
        help="the number of parallel game environments")
    parser.add_argument("--async-batch-size", type=int, default=16,
        help="the envpool's batch size in the async mode")
    parser.add_argument("--num-steps", type=int, default=32,
        help="the number of steps to run in each environment per policy rollout")
    parser.add_argument("--anneal-lr", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Toggle learning rate annealing for policy and value networks")
    parser.add_argument("--gae", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Use GAE for advantage computation")
    parser.add_argument("--gamma", type=float, default=0.995,  # Hessel et al. 2022, Muesli paper, Table 5
        help="the discount factor gamma")
    parser.add_argument("--gae-lambda", type=float, default=0.95,
        help="the lambda for the general advantage estimation")
    parser.add_argument("--num-minibatches", type=int, default=2,
        help="the number of mini-batches")
    parser.add_argument("--update-epochs", type=int, default=2,
        help="the K epochs to update the policy")
    parser.add_argument("--norm-adv", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Toggles advantages normalization")
    parser.add_argument("--clip-coef", type=float, default=0.1,
        help="the surrogate clipping coefficient")
    parser.add_argument("--ent-coef", type=float, default=0.01,
        help="coefficient of the entropy")
    parser.add_argument("--vf-coef", type=float, default=0.5,
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


def make_env(env_id, seed, num_envs, async_batch_size=1):
    def thunk():
        envs = envpool.make(
            env_id,
            env_type="gym",
            num_envs=num_envs,
            batch_size=async_batch_size,
            stack_num=4,  # Hessel et al. 2022, Muesli paper, Table 10
            img_height=86,  # Hessel et al. 2022, Muesli paper, Table 4
            img_width=86,  # Hessel et al. 2022, Muesli paper, Table 4
            episodic_life=False,  # Hessel et al. 2022, Muesli paper, Table 4
            repeat_action_probability=0.25,  # Hessel et al. 2022, Muesli paper, Table 4
            noop_max=1,  # Hessel et al. 2022, Muesli paper, Table 4
            full_action_space=True,  # Hessel et al. 2022, Muesli paper, Table 4
            max_episode_steps=int(108000 / 4),  # Hessel et al. 2022, Muesli paper, Table 4, we divide by 4 because of the skipped frames
            reward_clip=True,
            seed=seed,
        )
        envs.num_envs = num_envs
        envs.single_action_space = envs.action_space
        envs.single_observation_space = envs.observation_space
        envs.is_vector_env = True
        return envs

    return thunk


class ResidualBlock(nn.Module):
    channels: int

    @nn.compact
    def __call__(self, x):
        inputs = x
        x = nn.relu(x)
        x = nn.Conv(
            self.channels,
            kernel_size=(3, 3),
        )(x)
        x = nn.relu(x)
        x = nn.Conv(
            self.channels,
            kernel_size=(3, 3),
        )(x)
        return x + inputs


class ConvSequence(nn.Module):
    channels: int

    @nn.compact
    def __call__(self, x):
        x = nn.Conv(
            self.channels,
            kernel_size=(3, 3),
        )(x)
        x = nn.max_pool(x, window_shape=(3, 3), strides=(2, 2), padding="SAME")
        x = ResidualBlock(self.channels)(x)
        x = ResidualBlock(self.channels)(x)
        return x


class Network(nn.Module):
    channelss: Sequence[int] = (16, 32, 32)

    @nn.compact
    def __call__(self, x):
        x = jnp.transpose(x, (0, 2, 3, 1))
        x = x / (255.0)
        for channels in self.channelss:
            x = ConvSequence(channels)(x)
        x = nn.relu(x)
        x = x.reshape((x.shape[0], -1))
        x = nn.Dense(256, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(x)
        x = nn.relu(x)
        return x


class Critic(nn.Module):
    @nn.compact
    def __call__(self, x):
        return nn.Dense(1, kernel_init=orthogonal(1), bias_init=constant(0.0))(x)


class Actor(nn.Module):
    action_dim: Sequence[int]

    @nn.compact
    def __call__(self, x):
        return nn.Dense(self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0))(x)


@flax.struct.dataclass
class AgentParams:
    network_params: flax.core.FrozenDict
    actor_params: flax.core.FrozenDict
    critic_params: flax.core.FrozenDict


IntScalar = chex.Array


@chex.dataclass(frozen=True)
class BoundaryPointer:
    head: chex.Array
    length: chex.Array
    max_size: IntScalar

    @classmethod
    def init(cls, num_envs: IntScalar, max_size: IntScalar) -> BoundaryPointer:
        return cls(
            head=jnp.zeros(num_envs, jnp.int32),
            length=jnp.zeros(num_envs, jnp.int32),
            max_size=max_size,
        )

    @jax.jit
    def advance(self, env_ids: chex.Array) -> BoundaryPointer:
        new_head = self.head.at[env_ids].add(1) % self.max_size
        new_length = self.length.at[env_ids].add(jnp.where(self.length[env_ids] == self.max_size, 0, 1))
        return self.replace(head=new_head, length=new_length)

    @property
    @jax.jit
    def tail(self):
        return (self.head - self.length) % self.max_size

    @jax.jit
    def reset(self) -> BoundaryPointer:
        return self.replace(
            length=self.length.at[:].set(0),
        )


@chex.dataclass(frozen=True)
class UniformBuffer:
    """A batched replay buffer, inspired by Hwhitetooth's Jax MuZero implementation and the dejax package.

    See https://github.com/Hwhitetooth/jax_muzero/blob/main/algorithms/replay_buffers.py
    and https://github.com/hr0nix/dejax

    The buffer is designed so that we store a sequence of trajectories for each env.
    Buffer updates can happen asymmetrically across the envs, so that it works with envpool's async mode.
    This is a circular buffer with uniform experience sampling.

    Sequences may contain several adjacent trajectories or just a subsequence of a trajectory.

    Assumes each env trajectory stream is filled at a random rate that is i.i.d.
    """

    data: chex.ArrayTree
    online_queue_ind: BoundaryPointer
    full_buffer_ind: BoundaryPointer
    max_size: IntScalar

    @classmethod
    def init(cls, item_prototype: chex.ArrayTree, num_envs: int, max_size: int):
        chex.assert_tree_has_only_ndarrays(item_prototype)

        data = jax.tree_util.tree_map(lambda t: jnp.tile(t, (num_envs, max_size)), item_prototype)
        return cls(
            data=data,
            online_queue_ind=BoundaryPointer.init(num_envs, max_size),
            full_buffer_ind=BoundaryPointer.init(num_envs, max_size),
            max_size=max_size,
        )

    @jax.jit
    def reset_online_queue(self):
        return self.replace(
            online_queue_ind=self.online_queue_ind.reset(),
        )

    @chex.chexify
    @jax.jit
    def push_env_updates(self, update_batch: chex.ArrayTree, env_ids: chex.Array):
        chex.assert_tree_has_only_ndarrays(update_batch)

        new_data = jax.tree_util.tree_map(
            lambda entry, t: entry.at[env_ids, self.full_buffer_ind.head[env_ids]].set(t), self.data, update_batch
        )
        return self.replace(
            data=new_data,
            online_queue_ind=self.online_queue_ind.advance(env_ids),
            full_buffer_ind=self.full_buffer_ind.advance(env_ids),
        )

    @partial(jax.jit, static_argnums=(5, 6))
    def _sample_sequence(
        self,
        boundary_pointer: BoundaryPointer,
        arange_total_items: chex.Array,
        arange_sequence_length: chex.Array,
        rng: chex.PRNGKey,
        batch_size: int,
        sequence_length: int,
        distribution_power: float = 1,
    ):
        """Sample a sequence of trajectories from the buffer.

        Warning: the sequences are sampled with SRSWR, so they may overlap
        or repeat. Dealing with SRSWOR is annoying when the online queue doesn't have
        enough elements for a good SRSWOR. And then there's error handling and the fact
        that SRSWOR (of subsequences!) cannot be done in parallel easily...
        Using SRSWR is simpler, but it increases the variance of the estimator somewhat.

        Args:
            boundary_pointer: Information about where the stored info begins and ends.
            rng: The PRNG key.
            batch_size: The number of sequences to sample.
            sequence_length: The max length of the sequences to sample.
            distribution_power: Subsequences are sampled according to their length,
                raised to the power of `distribution_power`.
            arange_size: An array containing 0 to the size of the buffer-1

        Returns:
            seqs: The batch of requested sequences.
            seqs_mask: The mask that indicates if sequences are shorter than sequence_length.
        """
        # Get length of sequence if starting at an index
        cum_lengths_per_row = jnp.cumsum(boundary_pointer.length)

        def compute_remaining_sequence_length(carry, x):
            staggered_lengths, length_cutoff = carry
            corresponding_row = staggered_lengths[(staggered_lengths > x).argmax()]
            return (staggered_lengths, length_cutoff), jnp.clip(corresponding_row - x, a_max=length_cutoff)

        _, remaining_sequence_length = jax.lax.scan(
            compute_remaining_sequence_length,
            (cum_lengths_per_row, sequence_length),
            arange_total_items,
        )
        flattened_index_logits = jnp.log(remaining_sequence_length) * distribution_power

        # Sample from the non-empty indices in the buffer, with probability proportional
        # to the length of the index
        rng, index_selection_key = jax.random.split(rng)
        flattened_indices = jax.random.categorical(index_selection_key, logits=flattened_index_logits, shape=(batch_size,))

        # Figure out what indices in the buffer matrix that the flattened indices correspond to
        env_indices = (cum_lengths_per_row.reshape(-1, 1) > flattened_indices).argmax(0)
        env_start_index_to_flattened_index = jnp.concatenate([jnp.zeros(1).astype(jnp.int32), cum_lengths_per_row[:-1]], 0)
        col_indices_before_mod = flattened_indices + (boundary_pointer.tail - env_start_index_to_flattened_index)[env_indices]

        # Find the indices needed to access the sequences in a vectorized manner
        batched_sequence_index = jnp.repeat(arange_sequence_length.reshape(1, -1), batch_size, axis=0)
        expanded_env_indices = jnp.repeat(env_indices.reshape(-1, 1), sequence_length, axis=1)
        expanded_col_indices = (batched_sequence_index + col_indices_before_mod.reshape(-1, 1)) % self.max_size

        sequences = jax.tree_util.tree_map(
            lambda entry: entry[expanded_env_indices, expanded_col_indices],
            self.data,
        )
        sequence_masks = batched_sequence_index < remaining_sequence_length[flattened_indices].reshape(-1, 1)
        return sequences, sequence_masks

    def _sample_sequence_jit_helper(
        self,
        boundary_pointer: BoundaryPointer,
        rng: chex.PRNGKey,
        batch_size: int,
        sequence_length: int,
        distribution_power: float = 1,
    ):
        return self._sample_sequence(
            boundary_pointer,
            jnp.arange(boundary_pointer.length.sum()),
            jnp.arange(sequence_length),
            rng,
            batch_size,
            sequence_length,
            distribution_power,
        )

    def sample_online_queue(
        self,
        rng: chex.PRNGKey,
        batch_size: int,
        sequence_length: int,
        distribution_power: float = 1,
    ):
        return self._sample_sequence_jit_helper(self.online_queue_ind, rng, batch_size, sequence_length, distribution_power)

    def sample_replay_buffer(
        self,
        rng: chex.PRNGKey,
        batch_size: int,
        sequence_length: int,
        distribution_power: float = 1,
    ):
        return self._sample_sequence_jit_helper(self.full_buffer_ind, rng, batch_size, sequence_length, distribution_power)


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
    envs = make_env(args.env_id, args.seed, args.num_envs, args.async_batch_size)()
    assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"

    def linear_schedule(count):
        # anneal learning rate linearly after one training iteration which contains
        # (args.num_minibatches * args.update_epochs) gradient updates
        frac = 1.0 - (count // (args.num_minibatches * args.update_epochs)) / args.num_updates
        return args.learning_rate * frac

    network = Network()
    actor = Actor(action_dim=envs.single_action_space.n)
    critic = Critic()
    network_params = network.init(network_key, np.array([envs.single_observation_space.sample()]))
    agent_state = TrainState.create(
        apply_fn=None,
        params=AgentParams(
            network_params,
            actor.init(actor_key, network.apply(network_params, np.array([envs.single_observation_space.sample()]))),
            critic.init(critic_key, network.apply(network_params, np.array([envs.single_observation_space.sample()]))),
        ),
        tx=optax.chain(
            optax.clip_by_global_norm(args.max_grad_norm),
            optax.inject_hyperparams(optax.adam)(
                learning_rate=linear_schedule if args.anneal_lr else args.learning_rate, eps=1e-5
            ),
        ),
    )

    @jax.jit
    def get_action_and_value(
        agent_state: TrainState,
        next_obs: np.ndarray,
        key: jax.random.PRNGKey,
    ):
        hidden = network.apply(agent_state.params.network_params, next_obs)
        logits = actor.apply(agent_state.params.actor_params, hidden)
        # sample action: Gumbel-softmax trick
        # see https://stats.stackexchange.com/questions/359442/sampling-from-a-categorical-distribution
        key, subkey = jax.random.split(key)
        u = jax.random.uniform(subkey, shape=logits.shape)
        action = jnp.argmax(logits - jnp.log(-jnp.log(u)), axis=1)
        logprob = jax.nn.log_softmax(logits)[jnp.arange(action.shape[0]), action]
        value = critic.apply(agent_state.params.critic_params, hidden)
        return action, logprob, value.squeeze(), key

    @jax.jit
    def get_action_and_value2(
        params: flax.core.FrozenDict,
        x: np.ndarray,
        action: np.ndarray,
    ):
        hidden = network.apply(params.network_params, x)
        logits = actor.apply(params.actor_params, hidden)
        logprob = jax.nn.log_softmax(logits)[jnp.arange(action.shape[0]), action]
        logits = logits - jax.scipy.special.logsumexp(logits, axis=-1, keepdims=True)
        logits = logits.clip(min=jnp.finfo(logits.dtype).min)
        p_log_p = logits * jax.nn.softmax(logits)
        entropy = -p_log_p.sum(-1)
        value = critic.apply(params.critic_params, hidden).squeeze()
        return logprob, entropy, value

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

    def muesli_loss(params, x, a, logp, mb_advantages, mb_returns):
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

    muesli_loss_grad_fn = jax.value_and_grad(muesli_loss, has_aux=True)

    @jax.jit
    def update_muesli(
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
        b_actions = actions.reshape(-1)
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
                (loss, (pg_loss, v_loss, entropy_loss, approx_kl)), grads = muesli_loss_grad_fn(
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
            terminations.append(info["terminated"])
            episode_returns[env_id] += info["reward"]
            returned_episode_returns[env_id] = np.where(
                info["terminated"] + info["TimeLimit.truncated"], episode_returns[env_id], returned_episode_returns[env_id]
            )
            episode_returns[env_id] *= (1 - info["terminated"]) * (1 - info["TimeLimit.truncated"])
            episode_lengths[env_id] += 1
            returned_episode_lengths[env_id] = np.where(
                info["terminated"] + info["TimeLimit.truncated"], episode_lengths[env_id], returned_episode_lengths[env_id]
            )
            episode_lengths[env_id] *= (1 - info["terminated"]) * (1 - info["TimeLimit.truncated"])
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
        ) = update_muesli(
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
        from cleanrl_utils.evals.muesli_envpool_jax_eval import evaluate

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
            push_to_hub(args, episodic_returns, repo_id, "Muesli", f"runs/{run_name}", f"videos/{run_name}-eval")

    envs.close()
    writer.close()
