# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/muesli/#muesli_atari_envpool_async_jax_scan_impalanet_machadopy
# https://gregorygundersen.com/blog/2020/02/09/log-sum-exp/
from __future__ import annotations

import argparse
import os
import random
import time
import typing as tp
from distutils.util import strtobool
from functools import partial

os.environ[
    "XLA_PYTHON_CLIENT_MEM_FRACTION"
] = "0.7"  # see https://github.com/google/jax/discussions/6332#discussioncomment-1279991

import chex
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
    parser.add_argument("--num-envs", type=int, default=96,
        help="the number of parallel game environments")
    parser.add_argument("--async-batch-size", type=int, default=16,
        help="the envpool's batch size in the async mode")
    # The +2 is because async mode means we need to shift the rewards to the right once
    # and the fact that we need the previous action means we need to shift everything to the right once.
    parser.add_argument("--num-steps", type=int, default=30 + 2,  # Hessel et al. 2022, Muesli paper, Table 5
        help="the number of steps to run in each environment per policy rollout")
    parser.add_argument("--anneal-lr", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Toggle learning rate annealing for policy and value networks")
    parser.add_argument("--gamma", type=float, default=0.995,  # Hessel et al. 2022, Muesli paper, Table 5
        help="the discount factor gamma")
    parser.add_argument("--norm-adv", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Toggles advantages normalization")
    parser.add_argument("--clip-coef", type=float, default=0.1,
        help="the surrogate clipping coefficient")
    parser.add_argument("--max-parameter-update", type=float, default=1.0,
        help="the maximum norm for the parameter update")
    parser.add_argument("--prior-network-update-rate", type=float, default=0.1,  # Hessel et al. 2022, Muesli paper, Table 5
                        help="the update rate of the prior/target network")
    parser.add_argument("--adamw-weight-decay", type=float, default=0,  # Hessel et al. 2022, Muesli paper, Tables 5 and 6
                        help="the AdamW weight decay.")
    parser.add_argument("--beta-var", type=float, default=0.99,  # Hessel et al. 2022, Muesli paper, Table 5
        help="the variance moving average decay")
    parser.add_argument("--epsilon-var", type=float, default=1e-12,  # Hessel et al. 2022, Muesli paper, Table 5
                        help="the variance offset")
    parser.add_argument("--update-batch-size", type=int, default=96,  # Hessel et al. 2022, Muesli paper, Table 5
                        help="the minibatch size of each update")
    parser.add_argument("--num-retrace-estimator-samples", type=int, default=16,  # Hessel et al. 2022, Muesli paper, Table 5
                        help="the number of samples to use for the Retrace estimator.")
    parser.add_argument("--num-cmpo-regularizer-samples", type=int, default=16,
                        # Hessel et al. 2022, Muesli paper, Table 5
                        help="the number of samples to use for the CMPO regularizer estimator.")
    parser.add_argument("--cmpo-clipping-threshold", type=float, default=1.0,
                        # Hessel et al. 2022, Muesli paper, Section 4.1, last paragraph.
                        help="the clipping threshold used for the CMPO policy.")
    parser.add_argument("--cmpo-z-init", type=int, default=1.0, # Hessel et al. 2022, Muesli paper, Section 4.2, last paragraph.
                        help="the initial guess for the normalization constant in the sampled CMPO KL-divergence.")
    parser.add_argument("--reward-loss-coeff", type=float, default=1.0,  # Hessel et al. 2022, Muesli paper, Table 5
                        help="the coefficient of the reward model loss")
    parser.add_argument("--value-loss-coeff", type=float, default=0.25,  # Hessel et al. 2022, Muesli paper, Table 5
                        help="the coefficient of the value model loss")
    parser.add_argument("--retrace-lambda", type=float, default=0.95,  # Hessel et al. 2022, Muesli paper, Table 6
                        help="the coefficient of the Retrace importance weight")
    parser.add_argument("--cmpo-exact-kl", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
                        # Hessel et al. 2022, Muesli paper, Figure 3b
                        help="whether to use exact KL-divergence when using the CMPO regularizer")
    parser.add_argument("--cmpo-regularizer-lambda", type=float, default=1.0,  # Hessel et al. 2022, Muesli paper, Figure 3b
                        help="the coefficient of the CMPO regularizer")
    parser.add_argument("--replay-proportion", type=float, default=0.95,  # Hessel et al. 2022, Muesli paper, Table 6
                        help="the proportion of data to sample from the replay buffer.")
    parser.add_argument("--replay-buffer-size", type=int, default=6_000_000,  # Hessel et al. 2022, Muesli paper, Table 5
                        help="the maximum number of frames the replay buffer can store.")
    parser.add_argument("--model-unroll-length", type=int, default=5,
                        # Hessel et al. 2022, Muesli paper, Table 5
                        help="the number of steps to unroll the model.")
    parser.add_argument("--reward-transformation-epsilon", type=float, default=1e-3,  # Schrittwieser et al. 2019, MuZero paper, Appendix F
                        help="the Lipschitz continuity regularization coefficient for the reward transformation.")

    args = parser.parse_args()
    args.batch_sequence_length = args.num_steps-2
    args.batch_size = int(args.num_envs * args.num_steps)
    args.num_updates = args.total_timesteps // args.batch_size
    args.replay_buffer_size = args.replay_buffer_size // args.num_envs  # The width of the replay buffer is `num_envs`
    args.rb_batch_size = int(args.replay_proportion * args.update_batch_size)
    args.oq_batch_size = args.update_batch_size - args.update_batch_size
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
            max_episode_steps=int(
                108000 / 4
            ),  # Hessel et al. 2022, Muesli paper, Table 4, we divide by 4 because of the skipped frames
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


class RepresentationNetwork(nn.Module):
    """Large IMPALA network.

    See Espeholt et al. 2018, IMPALA paper, Figure 4.
    """

    action_dim: tp.Sequence[int]
    channelss: tp.Sequence[int] = (16, 32, 32)
    lstm_dim: int = 256
    reward_clip: int = 1

    @nn.compact
    def __call__(self, carry, x, last_reward, last_action):
        """Encode a batch of observations into the latent space.

        Args:
            carry: The carry to pass to the LSTM. <(batch_size, lstm_dim), (batch_size, lstm_dim)>
            x: The batch of observations. (batch_size, channels, height, width)
            last_reward: The reward from the previous step. (batch_size)
            last_action: The action from the previous step. (batch_size)
        """
        x = jnp.transpose(x, (0, 2, 3, 1))
        x = x / 255.0
        for channels in self.channelss:
            x = ConvSequence(channels)(x)
        x = nn.relu(x)
        batch_size = x.shape[0]
        x = x.reshape(batch_size, -1)
        x = nn.Dense(256, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(x)
        x = nn.relu(x)
        last_action = jax.nn.one_hot(last_action, self.action_dim).reshape(batch_size, -1)
        x = jnp.concatenate([x, last_reward, last_action], axis=-1)
        carry, x = nn.OptimizedLSTMCell()(carry, x)
        return carry, x

    @classmethod
    def initialize_carry(cls, batch_size: tp.Tuple[int, ...] = tuple()):
        init_hidden = jnp.zeros(batch_size + (cls.lstm_dim,))
        return init_hidden, init_hidden


class Critic(nn.Module):
    @nn.compact
    def __call__(self, x):
        return nn.Dense(1, kernel_init=orthogonal(1), bias_init=constant(0.0))(x)


class Actor(nn.Module):
    action_dim: tp.Sequence[int]

    @nn.compact
    def __call__(self, x):
        return nn.Dense(self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0))(x)


class DynamicsProjector(nn.Module):
    dynamics_hidden_dim: int = 1024

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(self.dynamics_hidden_dim, kernel_init=orthogonal(1), bias_init=constant(0.0))(x)
        # An LSTM carry has both the cell and hidden variables.
        return x


class Dynamics(nn.Module):
    action_dim: tp.Sequence[int]

    @nn.remat
    @nn.compact
    def __call__(self, carry, x):
        x = jax.nn.one_hot(x, self.action_dim).reshape(x.shape[0], -1).astype(jnp.float_)  # Work with both 1D and 2D arrays.
        carry, x = nn.OptimizedLSTMCell()(carry, x)
        return carry, x


class RewardValueModel(nn.Module):
    n_atoms: int = 601  # Schrittwieser et al. 2019, MuZero paper, Appendix F
    atoms = jnp.linspace(-300, 300, 601)

    @nn.compact
    def __call__(self, x):
        # Muesli uses MuZero's distributional reward and value representation. See the last paragraph of Section 4.5.
        reward = nn.Dense(self.n_atoms, kernel_init=orthogonal(0.01), bias_init=constant(0.0))(x)
        reward = nn.relu(reward)
        value = nn.Dense(self.n_atoms, kernel_init=orthogonal(0.01), bias_init=constant(0.0))(x)
        value = nn.relu(value)
        return reward, value

    @jax.jit
    def distribution_expectation(self, arr: jnp.ndarray):
        return (arr * self.atoms).sum(-1)

    @jax.jit
    def project_to_atoms(self, scalars):
        """Project the scalars to the distribution of atoms.

        Args:
            scalars: The scalars to project.

        Returns:
            distribution: The distribution of scalars. scalars.shape + (n_atoms,)
        """
        scalars = jnp.clip(scalars, self.atoms.min(), self.atoms.max())

        distribution = jnp.zeros(scalars.shape + self.atoms.shape)
        ceil_scalars = jnp.ceil(scalars)
        # The contribution of the lower atom.
        distribution = jnp.where(
            jnp.floor(scalars)[..., None] == self.atoms,
            jnp.where(
                # Ensure we just have a contribution of 1 when a scalar is equal to the value of an atom.
                (ceil_scalars == scalars),
                jnp.ones_like(scalars),
                ceil_scalars - scalars,
            )[..., None],
            distribution,
        )
        # When a scalar lies strictly between two atoms. Add the contribution of the upper atom.
        distribution = distribution.at[..., 1:].add(
            1 - jnp.where((0 < distribution[..., :-1]) & (distribution[..., :-1] < 1), distribution[..., :-1], 1)
        )
        return distribution

    @jax.jit
    def compute_q(self, pred_reward, pred_value):
        """Compute the q-value.

        pred_reward and pred_value have shape (seq_length, ..., batch_size, n_atoms)
        q_prior has shape (batch_size, seq_length, ...)
        """
        pred_reward_scalar = inverse_reward_transform(RewardValueModel.distribution_expectation(pred_reward))
        pred_value_scalar = inverse_reward_transform(RewardValueModel.distribution_expectation(pred_value))
        q = jnp.moveaxis(pred_reward_scalar + args.gamma * pred_value_scalar, -1, 0)
        return q


class PolicyModel(nn.Module):
    action_dim: tp.Sequence[int]

    @nn.compact
    def __call__(self, x):
        policy = nn.Dense(self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0))(x)
        policy = nn.relu(policy)
        return policy


@flax.struct.dataclass
class AgentParams:
    network_params: flax.core.FrozenDict
    actor_params: flax.core.FrozenDict
    critic_params: flax.core.FrozenDict
    dynamic_proj_params: flax.core.FrozenDict
    dynamics_params: flax.core.FrozenDict
    reward_value_model_params: flax.core.FrozenDict
    policy_model_params: flax.core.FrozenDict


def scale_by_learning_rate(learning_rate: optax.ScalarOrSchedule, flip_sign=True):
    """Scale by the learning rate.

    Taken from https://github.com/deepmind/optax/blob/master/optax/_src/alias.py#L34#L38
    """
    m = -1 if flip_sign else 1
    if callable(learning_rate):
        return optax.scale_by_schedule(lambda count: m * learning_rate(count))
    return optax.scale(m * learning_rate)


def clipped_adamw(
    learning_rate: optax.ScalarOrSchedule,
    b1: float = 0.9,
    b2: float = 0.999,
    eps: float = 1e-8,
    eps_root: float = 0.0,
    mu_dtype: tp.Any | None = None,
    weight_decay: float = 1e-4,
    mask: tp.Optional[tp.Union[tp.Any, tp.Callable[[optax.Params], tp.Any]]] = None,
    max_parameter_update: float = 1.0,
) -> optax.GradientTransformation:
    r"""The AdamW optimizer, except we clip the gradient before scaling by the learning rate.

    See Hessel et al. 2022, Muesli paper, Muesli Supplement F.1: Optimizers for details.

    Args:
        learning_rate: A fixed global scaling factor.
        b1: Exponential decay rate to track the first moment of past gradients.
        b2: Exponential decay rate to track the second moment of past gradients.
        eps: A small constant applied to denominator outside of the square root
          (as in the Adam paper) to avoid dividing by zero when rescaling.
        eps_root: A small constant applied to denominator inside the square root (as
          in RMSProp), to avoid dividing by zero when rescaling. This is needed for
          example when computing (meta-)gradients through Adam.
        mu_dtype: Optional `dtype` to be used for the first order accumulator; if
          `None` then the `dtype` is inferred from `params` and `updates`.
        weight_decay: Strength of the weight decay regularization. Note that this
          weight decay is multiplied with the learning rate. This is consistent
          with other frameworks such as PyTorch, but different from
          (Loshchilov et al, 2019) where the weight decay is only multiplied with
          the "schedule multiplier", but not the base learning rate.
        mask: A tree with same structure as (or a prefix of) the params PyTree,
          or a Callable that returns such a pytree given the params/updates.
          The leaves should be booleans, `True` for leaves/subtrees you want to
          apply the weight decay to, and `False` for those you want to skip. Note
          that the Adam gradient transformations are applied to all parameters.
        max_parameter_update: The amount to clip the parameter update by.
    Returns:
        The corresponding `GradientTransformation`.
    """
    return optax.chain(
        optax.scale_by_adam(b1=b1, b2=b2, eps=eps, eps_root=eps_root, mu_dtype=mu_dtype),
        optax.add_decayed_weights(weight_decay, mask),
        optax.clip(max_parameter_update),
        scale_by_learning_rate(learning_rate),
    )


def update_target_network(state: TrainState, target_state: TrainState, alpha_target: float) -> TrainState:
    new_target_params = jax.tree_multimap(
        lambda p, tp: p * alpha_target + tp * (1 - alpha_target), state.params, target_state.params
    )

    return target_state.replace(params=new_target_params)


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

    def advance(self, env_ids: chex.Array) -> BoundaryPointer:
        new_head = self.head.at[env_ids].add(1) % self.max_size
        new_length = self.length.at[env_ids].add(jnp.where(self.length[env_ids] == self.max_size, 0, 1))
        return self.replace(head=new_head, length=new_length)

    @property
    def tail(self):
        return (self.head - self.length) % self.max_size

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

        data = jax.tree_util.tree_map(
            lambda t: jnp.tile(t[None, None, ...], (num_envs, max_size) + (1,) * t.ndim), item_prototype
        )
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
        col_indices = flattened_indices + (boundary_pointer.tail - env_start_index_to_flattened_index)[env_indices]
        col_indices = (arange_sequence_length.reshape(1, -1) + col_indices.reshape(-1, 1)) % self.max_size

        sequences = jax.tree_util.tree_map(
            lambda entry: entry[env_indices.reshape(-1, 1), col_indices],
            self.data,
        )
        sequence_masks = arange_sequence_length.reshape(1, -1) < remaining_sequence_length[flattened_indices].reshape(-1, 1)
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

    def sample_rb_and_oq(
        self,
        rng: chex.PRNGKey,
        rb_batch_size: int,
        oq_batch_size: int,
        sequence_length: int,
        distribution_power: float = 1,
    ):
        _, rb_rng, oq_rng = jax.random.split(rng, 3)
        rb_sequence, rb_mask = self.sample_replay_buffer(
            rb_rng, rb_batch_size, sequence_length, distribution_power=distribution_power
        )
        oq_sequence, oq_mask = self.sample_online_queue(
            oq_rng, oq_batch_size, sequence_length, distribution_power=distribution_power
        )
        sequence = jax.tree_util.tree_map(
            lambda rb_entry, oq_entry: jnp.vstack([rb_entry, oq_entry]), rb_sequence, oq_sequence
        )
        mask = jnp.vstack([rb_mask, oq_mask])
        return sequence, mask

    @jax.jit
    def peek(self, env_ids: chex.Array):
        """Peek at the top of the buffer.

        The online queue and the replay buffer have the same head.
        """
        return jax.tree_util.tree_map(
            lambda entry: entry[env_ids, (self.full_buffer_ind.head[env_ids] - 1) % self.max_size],
            self.data,
        )


@chex.dataclass
class Storage:
    obs: chex.Array
    action: chex.Array
    logprob: chex.Array
    prev_reward: chex.Array
    done: chex.Array


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
    key, network_key, actor_key, critic_key, dynamics_proj_key, dynamics_key, rv_model_key, p_model_key = jax.random.split(
        key, 8
    )

    # env setup
    envs = make_env(args.env_id, args.seed, args.num_envs, args.async_batch_size)()
    assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"

    def linear_schedule(count):
        # anneal learning rate linearly after one training iteration which contains
        # (args.num_minibatches * args.update_epochs) gradient updates
        frac = 1.0 - (count // args.update_epochs) / args.num_updates
        return args.learning_rate * frac

    network = RepresentationNetwork(action_dim=envs.single_action_space.n)
    actor = Actor(action_dim=envs.single_action_space.n)
    critic = Critic()
    dynamics_proj = DynamicsProjector()
    dynamics = Dynamics(action_dim=envs.single_action_space.n)
    reward_value_model = RewardValueModel()
    policy_model = PolicyModel(action_dim=envs.single_action_space.n)

    example_obs = np.array([envs.single_observation_space.sample()])
    example_carry = RepresentationNetwork.initialize_carry((1,))
    example_reward = np.array([[0.0]])
    example_action = np.array([envs.single_action_space.sample()])
    network_params = network.init(network_key, example_carry, example_obs, example_reward, example_action)

    _, example_latent_obs = network.apply(network_params, example_carry, example_obs, example_reward, example_action)
    actor_params = actor.init(actor_key, example_latent_obs)
    critic_params = critic.init(critic_key, example_latent_obs)
    dynamics_proj_params = dynamics_proj.init(critic_key, example_latent_obs)

    example_dyn_carry = dynamics_proj.apply(dynamics_proj_params, example_latent_obs)
    example_dyn_carry = (example_dyn_carry, example_dyn_carry)
    dynamics_params = dynamics.init(dynamics_key, example_dyn_carry, example_action)

    _, example_predicted_latent_obs = dynamics.apply(dynamics_params, example_dyn_carry, example_action)
    reward_value_model_params = reward_value_model.init(rv_model_key, example_predicted_latent_obs)
    policy_model_params = policy_model.init(p_model_key, example_predicted_latent_obs)

    agent_state = TrainState.create(
        apply_fn=None,
        params=AgentParams(
            network_params,
            actor_params,
            critic_params,
            dynamics_proj_params,
            dynamics_params,
            reward_value_model_params,
            policy_model_params,
        ),
        tx=clipped_adamw(
            learning_rate=linear_schedule if args.anneal_lr else args.learning_rate,
            eps=1e-5,
            weight_decay=args.adamw_weight_decay,
            max_parameter_update=args.max_parameter_update,
        ),
    )
    target_agent_state = TrainState.create(
        apply_fn=None,
        params=AgentParams(
            network_params,
            actor_params,
            critic_params,
            dynamics_proj_params,
            dynamics_params,
            reward_value_model_params,
            policy_model_params,
        ),
        tx=optax.set_to_zero(),
    )

    @jax.jit
    def get_action_and_lstm_carry(
        agent_state: TrainState,
        obs: np.ndarray,
        lstm_carry: np.ndarray,
        prev_reward: np.ndarray,
        prev_action: np.ndarray,
        key: jax.random.PRNGKey,
    ):
        lstm_carry, hidden = network.apply(agent_state.params.network_params, lstm_carry, obs, prev_reward, prev_action)
        logits = normalize_logits(actor.apply(agent_state.params.actor_params, hidden))
        # sample action: Gumbel-softmax trick
        # see https://stats.stackexchange.com/questions/359442/sampling-from-a-categorical-distribution
        key, subkey = jax.random.split(key)
        u = jax.random.uniform(subkey, shape=logits.shape)
        action = jnp.argmax(logits - jnp.log(-jnp.log(u)), axis=1)
        logprob = jax.nn.log_softmax(logits)[jnp.arange(action.shape[0]), action]
        return action, logprob, lstm_carry, key

    @jax.jit
    def normalize_logits(logits: jnp.ndarray) -> jnp.ndarray:
        """Normalize thhe logits.

        # https://gregorygundersen.com/blog/2020/02/09/log-sum-exp/
        """
        logits = logits - jax.scipy.special.logsumexp(logits, axis=-1, keepdims=True)
        logits = logits.clip(min=jnp.finfo(logits.dtype).min)
        return logits

    @jax.jit
    def reward_transform(r: jnp.ndarray) -> jnp.ndarray:
        """Transform the reward.

        h(x): R -> R, epsilon > 0
        x -> sgn(x)(sqrt(abs(x)+1)-1) + epsilon * x

        The function is applied element-wise.

        See the last paragraph of Hessel et al., 2021. Muesli paper. Section 4.5. for details.
        Alternatively, see Pohlen et al., 2018. Observe and Look Further: Achieving Consistent Performance on Atari.
        - Section 3.2: how to use it
        - Proposition A.2: the inverse and some properties of the transformation.
        """
        eps = args.reward_transformation_epsilon
        return jnp.sign(r) * (jnp.sqrt(jnp.abs(r) + 1) - 1) + eps * r

    @jax.jit
    def inverse_reward_transform(tr: jnp.ndarray) -> jnp.ndarray:
        """De-transform the reward.

        h^-1(x): R -> R, epsilon > 0
        x -> sgn(x)(((sqrt(1 + 4*epsilon*(abs(x)+1+epsilon)) - 1)/(2*epsilon))^2-1)

        The function is applied element-wise.

        See the last paragraph of Hessel et al., 2021. Muesli paper. Section 4.5. for details.
        Alternatively, see Pohlen et al., 2018. Observe and Look Further: Achieving Consistent Performance on Atari.
        - Section 3.2: how to use it
        - Proposition A.2: the inverse and some properties of the transformation.
        """
        eps = args.reward_transformation_epsilon
        num = jnp.sqrt(1 + 4 * eps * (jnp.abs(tr) + 1 + eps)) - 1
        denom = 2 * eps
        return jnp.sign(tr)((num / denom) ** 2 - 1)

    @jax.jit
    def unroll_model(carry, act):
        carry, pred_latent_state = dynamics.apply(agent_state.params.dynamics_params, carry, act)
        pred_reward, pred_value = reward_value_model.apply(agent_state.params.reward_value_model_params, pred_latent_state)
        pred_policy = normalize_logits(policy_model.apply(agent_state.params.policy_model_params, pred_latent_state))
        return carry, (pred_reward, pred_value, pred_policy)

    @jax.jit
    def compute_pred_q_and_policy(carry, act):
        _, pred_latent_state = dynamics.apply(agent_state.params.dynamics_params, carry, act)
        pred_reward, pred_value = reward_value_model.apply(agent_state.params.reward_value_model_params, pred_latent_state)
        return carry, (pred_reward, pred_value)

    @jax.jit
    def reanalyze_prior_policies_values_latent_states(carry, x):
        target_params, lstm_carry, key = carry
        obs, prev_reward, prev_action, action, done = x

        # Reset the LSTM carry when necessary
        lstm_carry = jnp.where(done, jnp.zeros_like(lstm_carry), lstm_carry)

        lstm_carry, x = network.apply(target_params.network_params, lstm_carry, obs, prev_reward, prev_action)
        policy_logits = normalize_logits(actor.apply(target_params.actor_params, x))
        value = critic.apply(target_params.critic_params, x)

        logprob = jax.nn.log_softmax(policy_logits)[jnp.arange(action.shape[0]), action]

        if args.cmpo_exact_kl:
            # TODO (shermansiu): Ensure that the batch_size broadcast works
            sample_actions = jnp.arange(policy_logits.shape[-1])[:, None]
        else:
            key, subkey = jax.random.split(key, 2)
            # (n_samples, batch_size, n_actions)
            u = jax.random.uniform(subkey, shape=(args.num_cmpo_regularizer_samples,) + policy_logits.shape)
            sample_actions = jnp.argmax(policy_logits[None, ...] - jnp.log(-jnp.log(u)), axis=-1)

        return (target_params, lstm_carry, key), (
            policy_logits,
            value,
            logprob,
            sample_actions,
        )

    @jax.jit
    def reanalyze_policies_and_model_preds(carry, x):
        params, lstm_carry = carry
        obs, prev_reward, prev_action, act_seq, sample_actions, done = x

        # Reset the LSTM carry when necessary
        lstm_carry = jnp.where(done, jnp.zeros_like(lstm_carry), lstm_carry)

        lstm_carry, x = network.apply(params.network_params, lstm_carry, obs, prev_reward, prev_action)
        policy_logits = normalize_logits(actor.apply(params.actor_params, x))
        logprobs = jax.nn.log_softmax(policy_logits)
        batch_arange = jnp.arange(prev_action.shape[0])
        logprob = logprobs[batch_arange, prev_action]
        lobprob_sample_actions = logprobs[batch_arange.reshape(-1, 1), sample_actions]

        x = dynamics_proj.apply(params.dynamic_proj_params, x)

        _, (pred_reward, pred_value, pred_policy) = jax.lax.scan(
            unroll_model,
            (x, x),
            act_seq.swapaxes(0, 1),
        )
        _, (sample_pred_r, sample_pred_v) = jax.lax.scan(compute_pred_q_and_policy, (x, x), sample_actions)
        return (params, lstm_carry), (
            logprob,
            lobprob_sample_actions,
            pred_reward,
            pred_value,
            pred_policy,
            sample_pred_r,
            sample_pred_v,
        )

    @jax.jit
    def compute_retrace_return_one_step(retrace_return, x):
        reward, q_prior, next_q_prior, log_delta_coeff, should_discard = x
        delta = jnp.exp(log_delta_coeff) * (reward + args.gamma * next_q_prior * should_discard - q_prior)
        retrace_return = jnp.where(should_discard, retrace_return, retrace_return + args.gamma * delta)
        return retrace_return, jnp.array(0)

    @jax.jit
    def compute_retrace_return(carry, x):
        """Compute the Retrace-corrected return G^v(s, a).

        All xs should have the shape (batch_size, window_width).

        Returns:
            retrace_return: The Retrace estimate of the return, bootstrapped with q_prior. (batch_size)
        """
        reward, q_prior, logprob_diff, should_discard = x
        # The Retrace estimator uses the coefficients c_s = lambda * min(1, pi(a_s|x_s)/mu(a_s|x_s)).
        logprob_diff = jnp.cumsum(jnp.log(args.retrace_lambda) * jnp.clip(logprob_diff.at[:, 0].set(0), a_max=0), axis=-1)
        batch_size = reward.shape[0]
        q_prior = q_prior.swapaxes(0, 1)
        retrace_return, _ = jax.lax.scan(
            compute_retrace_return_one_step,
            jnp.zeros(batch_size),
            (
                reward.swapaxes(0, 1)[:-1],
                q_prior[:-1],
                q_prior[1:],
                logprob_diff.swapaxes(0, 1)[:-1],
                should_discard.swapaxes(0, 1)[1:],
            ),
            reverse=True,
        )
        retrace_return = retrace_return + q_prior[0]
        return carry, retrace_return

    @jax.jit
    def muesli_loss(
        params: AgentParams,
        target_params: AgentParams,
        ema_adv_var: jnp.ndarray,
        beta_product: jnp.ndarray,
        seqs: Storage,
        seq_mask: jnp.ndarray,
        key: jax.random.PRNGKey,
    ):
        """Calculate the loss for Muesli.

        We need to shift the rewards to the future by one time step because of the way the async API works.
        Because the LSTM takes the previous rewards, the offset cancels out.
        But since we need the previous action for the LSTM, we end up shifting everything to the right once anyway.

        obs: -1, ..., t+1
        reward: -2, ..., t
        action: -1, ..., t+1
        logprob: -1, ..., t+1
        done: -1, ..., t+1

        Args:
            params: The parameters of the Muesli model.
            target_params: The parameters of the target Muesli model. Used to make actions.
            ema_adv_var: The exponential moving average of the variance of the advantage estimator.
            beta_product: The memoized version of beta_var^t. Used for bias correction.
            seqs: The batch of sequences.
            seq_mask: The mask of "valid sequences."
            key: The pseudo-random number generator key.
        """

        # TODO: Slight problem: When learning, we have to use LSTM hiddens generated from scratch because
        # the original ones are from an outdated network.
        lstm_carry = RepresentationNetwork.initialize_carry((args.update_batch_size,))
        unrolled_actions, _ = make_batched_sliding_windows(
            seqs.action[:, 1:-1], seq_mask[:, 1:-1], args.model_unroll_length
        )  # (batch_size, sequence_length, window_width)

        # Replay the sequences in B through the target/prior network.
        # If only we could scan over axis 1 instead of axis 0...
        (_, _, key), (policy_prior_logits, value_prior, logprob_prior, sample_actions) = jax.lax.scan(
            reanalyze_prior_policies_values_latent_states,
            (target_params, lstm_carry, key),
            (
                seqs.obs.swapaxes(0, 1)[1:-1],
                seqs.prev_reward.swapaxes(0, 1)[1:-1],
                seqs.action.swapaxes(0, 1)[:-2],
                seqs.action.swapaxes(0, 1)[1:-1],
                seqs.done.swapaxes(0, 1)[1:-1],
            ),
            args.batch_sequence_length,
        )

        # Replay the sequences in B through the current network.
        _, (
            logprob_curr,
            logprob_curr_sample_actions,
            pred_reward,
            pred_value,
            pred_policy_logits,
            sample_pred_r,
            sample_pred_v,
        ) = jax.lax.scan(
            reanalyze_policies_and_model_preds,
            (params, lstm_carry),
            (
                seqs.obs.swapaxes(0, 1)[1:-1],
                seqs.prev_reward.swapaxes(0, 1)[1:-1],
                seqs.action.swapaxes(0, 1)[:-2],
                unrolled_actions.swapaxes(0, 1),
                sample_actions.swapaxes(1, 2),  # (seq_length, batch_size, n_samples)
                seqs.done.swapaxes(0, 1)[1:-1],
            ),
            args.batch_sequence_length,
        )

        q_prior = RewardValueModel.compute_q(pred_reward[:, 0], pred_value[:, 0])  # (batch_size, seq_length)
        q_prior_seq, _ = make_batched_sliding_windows(q_prior, seq_mask[:, 1:-1], args.num_retrace_estimator_samples)
        reward_seq, _ = make_batched_sliding_windows(
            seqs.prev_reward[:, 2:], seq_mask[:, 2:], args.num_retrace_estimator_samples
        )
        # log(pi_prior(a|s)/pi_b(a|s))
        logprob_diff_seq, _ = make_batched_sliding_windows(
            logprob_prior.swapaxes(0, 1) - seqs.logprob[:, 1:-1], seq_mask[:, 1:-1], args.num_retrace_estimator_samples
        )

        done_seq, mask_seq = make_batched_sliding_windows(
            # Sync with reward's mask
            seqs.done[:, :-2],
            seq_mask[:, 2:],
            args.num_retrace_estimator_samples,
        )
        # We figure out if the current entry in the window is from the same trajectory as the first entry
        is_from_diff_traj = compute_is_from_diff_traj(done_seq, mask_seq)

        # Calculate Retrace estimates G^v(s,a)
        _, retrace_return = jax.lax.scan(
            compute_retrace_return,
            (),
            (
                reward_seq.swapaxes(0, 1),
                q_prior_seq.swapaxes(0, 1),
                logprob_diff_seq.swapaxes(0, 1),
                is_from_diff_traj.swapaxes(0, 1),
            ),
        )

        retrace_advantage = retrace_return - value_prior
        new_advantage_variance = (retrace_advantage**2).mean()
        ema_adv_var = args.beta_var * ema_adv_var + (1 - args.beta_var) * new_advantage_variance
        beta_product = beta_product * args.beta_var
        ema_adv_var_bias_corrected = ema_adv_var / (1 - beta_product)

        # Advantages of sampled actions (or all actions when using exact KL-divergence)
        sample_q_prior = RewardValueModel.compute_q(sample_pred_r, sample_pred_v)
        sample_pred_advantages = (
            sample_q_prior - value_prior.swapaxes(0, 1)[..., None]
        )  # (batch_size, sequence_length, n_samples)

        if args.norm_adv:
            sample_pred_advantages = sample_pred_advantages / jnp.sqrt(ema_adv_var_bias_corrected + args.epsilon_var)

        # Calculate CMPO policy
        clipped_adv_estimate = jnp.clip(sample_pred_advantages, -args.cmpo_clipping_threshold, args.cmpo_clipping_threshold)
        prior_log_probs = jax.nn.log_softmax(policy_prior_logits.swapaxes(0, 1), -1)
        cmpo_log_probs = jax.nn.log_softmax(prior_log_probs + clipped_adv_estimate, -1)
        cmpo_probs = jnp.exp(cmpo_log_probs)

        # Policy loss
        # NOTE (shermansiu): In Appendix A, the author's use beta-LOO action-dependant baselines to correct for the bias from clipping the importance weight.
        policy_importance_ratio = jnp.clip(jnp.exp(logprob_curr.swapaxes(0, 1) - seqs.logprob[:, 1:-1]), 0, 1)
        pg_loss = -(policy_importance_ratio * retrace_advantage).mean()

        # CMPO regularizer
        if args.cmpo_exact_kl:  # Used for large-scale Atari experiments. See Hessel et al. 2021, Muesli paper, Table 6.
            cmpo_loss = (
                args.cmpo_regularizer_lambda * (cmpo_probs * (cmpo_log_probs - logprob_curr_sample_actions)).sum(-1).mean()
            )
        else:
            clipped_adv_estimate = jnp.exp(clipped_adv_estimate)
            z_tilde_cmpo = (
                args.cmpo_z_init + clipped_adv_estimate.sum(-1) - clipped_adv_estimate
            ) / args.num_cmpo_regularizer_samples
            cmpo_loss = -args.cmpo_regularizer_lambda(clipped_adv_estimate / z_tilde_cmpo * logprob_curr_sample_actions).mean()

        # Reward (model) loss
        unrolled_rewards, _ = make_batched_sliding_windows(seqs.prev_reward[:, 2:], seq_mask[:, 2:], args.model_unroll_length)
        unrolled_done, unrolled_mask = make_batched_sliding_windows(
            seqs.done[:, :-2], seq_mask[:, 2:], args.model_unroll_length
        )
        # We figure out if the current entry in the window is from the same trajectory as the first entry
        r_loss = calculate_categorical_scalar_loss(pred_reward, unrolled_done, unrolled_mask, unrolled_rewards)

        # Value (model) loss
        unrolled_values, _ = make_batched_sliding_windows(value_prior, seq_mask[:, 1:-1], args.model_unroll_length)
        unrolled_done, unrolled_mask = make_batched_sliding_windows(
            seqs.done[:, :-2], seq_mask[:, :-2], args.model_unroll_length
        )
        # We figure out if the current entry in the window is from the same trajectory as the first entry
        v_loss = calculate_categorical_scalar_loss(pred_value, unrolled_done, unrolled_mask, unrolled_values)

        # Policy model loss
        unrolled_cmpo_log_probs, _ = make_batched_sliding_windows(cmpo_log_probs, seq_mask[:, 1:-1], args.model_unroll_length)
        unrolled_done, unrolled_mask = make_batched_sliding_windows(
            seqs.done[:, :-2], seq_mask[:, :-2], args.model_unroll_length
        )
        # We figure out if the current entry in the window is from the same trajectory as the first entry
        unrolled_is_from_diff_traj = compute_is_from_diff_traj(unrolled_done, unrolled_mask)
        pred_policy_logits = pred_policy_logits.transpose(2, 0, 1, 3)  # (batch_size, seq_length, window_width, n_actions)
        pi_model_kl_div = (unrolled_cmpo_log_probs * (unrolled_cmpo_log_probs - pred_policy_logits)).sum(-1)
        m_loss = jnp.where(unrolled_is_from_diff_traj, 0, pi_model_kl_div).mean()

        loss = pg_loss + cmpo_loss + m_loss + args.reward_loss_coeff * r_loss + args.value_loss_coeff * v_loss

        return loss, (loss, pg_loss, cmpo_loss, m_loss, r_loss, v_loss, ema_adv_var, beta_product, key)

    muesli_loss_grad_fn = jax.value_and_grad(muesli_loss, has_aux=True)

    @partial(jax.jit, static_argnums=(2,))
    def make_batched_sliding_windows(batched_sequence, valid_masks, window_width: int):
        """Make batched sliding windows.

        Args:
            batched_sequence: A sequence of batches. (batch_size, ..., sequence_length)
            valid_masks: Which elements in the sequence are valid. Denotes when the sequence ends. (batch_size, sequence_length)
            window_width: The width of the window.

        Returns:
            batched_windows: A batch of windows. (batch_size, ..., sequence_length, window_width)
            valid_masks: Which windows are valid. (batch_size, sequence_length, window_width)
        """
        indices = jnp.arange(window_width).reshape(1, -1) + jnp.arange(batched_sequence.shape[-1]).reshape(-1, 1)
        return (
            batched_sequence.swapaxes(1, -1)[..., indices].swapaxes(1, -2),
            valid_masks[:, indices] & (indices < batched_sequence.shape[-1])[None, ...],
        )

    @jax.jit
    def update_muesli(
        agent_state: TrainState,
        target_agent_state: TrainState,
        ema_adv_var: jnp.ndarray,
        beta_product: jnp.ndarray,
        seqs: Storage,
        seq_mask: jnp.ndarray,
        key: jax.random.PRNGKey,
    ):
        """Update the Muesli model.

        Args:
            agent_state: The state of the Muesli model.
            target_agent_state: The state of the target Muesli model. Used to make actions.
            ema_adv_var: The exponential moving average of the variance of the advantage estimator.
            beta_product: The memoized version of beta_var^t. Used for bias correction.
            seqs: The batch of sequences.
            seq_mask: The mask of "valid sequences."
            key: The pseudo-random number generator key.
        """

        (loss, (loss, pg_loss, cmpo_loss, m_loss, r_loss, v_loss, ema_adv_var, beta_product, key)), grads = muesli_loss(
            agent_state.params,
            target_agent_state.params,
            ema_adv_var,
            beta_product,
            seqs,
            seq_mask,
            key,
        )

        agent_state = agent_state.apply_gradients(grads=grads)
        update_target_network(agent_state, target_agent_state, alpha_target=args.prior_network_update_rate)

        return (
            agent_state,
            target_agent_state,
            ema_adv_var,
            beta_product,
            loss,
            pg_loss,
            cmpo_loss,
            m_loss,
            r_loss,
            v_loss,
            key,
        )

    @jax.jit
    def calculate_categorical_scalar_loss(pred_scalar, unrolled_done, unrolled_mask, unrolled_scalar):
        unrolled_is_from_diff_traj = compute_is_from_diff_traj(unrolled_done, unrolled_mask)
        true_scalar = RewardValueModel.project_to_atoms(reward_transform(unrolled_scalar))
        pred_scalar = jnp.log(pred_scalar.transpose(2, 0, 1, 3))
        cross_entropy = jnp.where(unrolled_is_from_diff_traj, 0, true_scalar * jnp.log(pred_scalar.transpose(2, 0, 1, 3)))
        scalar_loss = -cross_entropy.sum(-1).mean()
        return scalar_loss

    def compute_is_from_diff_traj(done_seq, mask_seq):
        is_from_diff_traj = jnp.concatenate([jnp.full_like(done_seq[..., 0], False)[..., None], done_seq[..., :-1]], -1)
        is_from_diff_traj = jnp.cumsum(is_from_diff_traj, axis=-1).astype(jnp.bool_) | ~mask_seq
        return is_from_diff_traj

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

    ema_adv_var = jnp.asarray(0.0)
    beta_product = jnp.asarray(1.0)

    buffer = UniformBuffer.init(
        [
            Storage(
                obs=jnp.asarray(envs.single_observation_space.sample()),
                action=jnp.asarray(envs.single_action_space.sample()),
                logprob=jnp.asarray(0),
                prev_reward=jnp.asarray(0),
                done=jnp.asarray(True),
            )
        ],
        num_envs=args.num_envs,
        max_size=args.replay_buffer_size,
    )
    lstm_hidden_carryover = UniformBuffer.init(
        RepresentationNetwork.initialize_carry(),
        num_envs=args.num_envs,
        max_size=1,
    )

    for _ in range(args.num_updates + 1):
        update_time_start = time.time()
        truncations = []
        terminations = []
        env_recv_time = 0
        inference_time = 0
        storage_time = 0
        env_send_time = 0
        buffer_peek_time = 0

        # NOTE: This is a major difference from the sync version:
        # at the end of the rollout phase, the sync version will have the next observation
        # ready for the value bootstrap, but the async version will not have it.
        # for this reason we do `num_steps + 1`` to get the extra states for value bootstrapping.
        for step in range(
            async_update, (args.num_steps + 1) * async_update
        ):  # num_steps + 1 to get the states for value bootstrapping.
            env_recv_time_start = time.time()
            obs, prev_reward, done, info = envs.recv()
            env_recv_time += time.time() - env_recv_time_start
            global_step += len(done)
            env_id = info["env_id"]

            buffer_peek_time_start = time.time()
            prev_lstm_hidden = lstm_hidden_carryover.peek(env_id)
            prev_action = buffer.peek(env_id).action
            prev_lstm_hidden = jnp.where(done, jnp.zeros_like(prev_lstm_hidden), prev_lstm_hidden)
            buffer_peek_time += time.time() - buffer_peek_time_start

            inference_time_start = time.time()
            action, logprob, lstm_hidden, key = get_action_and_lstm_carry(
                target_agent_state, obs, prev_lstm_hidden, prev_reward, prev_action, key
            )
            inference_time += time.time() - inference_time_start

            env_send_time_start = time.time()
            envs.send(np.array(action), env_id)
            env_send_time += time.time() - env_send_time_start
            storage_time_start = time.time()

            buffer.push_env_updates(
                update_batch=Storage(
                    obs=jnp.asarray(obs),
                    action=jnp.asarray(action),
                    logprob=jnp.asarray(logprob),
                    prev_reward=jnp.asarray(prev_reward),
                    done=jnp.asarray(done),
                ),
                env_id=jnp.asarray(env_id),
            )
            lstm_hidden_carryover.push_env_updates(
                update_batch=jnp.asarray(lstm_hidden),
                env_id=jnp.asarray(env_id),
            )

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

        # Get sequence for current batch.
        # Sequence sampling is mostly but not fully jitted.
        buffer_sample_time_start = time.time()
        key, sequence_key = jax.random.split(key, 2)
        # The rewards lag by one time step, so we need to shift them right once.
        # And because we need the previous actions for the LSTM, we need to shift everything right once.
        sequence_length = args.batch_sequence_length + 2
        seqs: Storage
        seq_mask: jnp.ndarray
        seqs, seq_mask = buffer.sample_rb_and_oq(
            sequence_key, args.rb_batch_size, args.oq_batch_size, sequence_length=sequence_length, distribution_power=2
        )
        buffer_sample_time = time.time() - buffer_sample_time_start

        avg_episodic_return = np.mean(returned_episode_returns)
        # print(returned_episode_returns)
        print(f"global_step={global_step}, avg_episodic_return={avg_episodic_return}")
        writer.add_scalar("charts/avg_episodic_return", avg_episodic_return, global_step)
        writer.add_scalar("charts/avg_episodic_length", np.mean(returned_episode_lengths), global_step)
        training_time_start = time.time()
        (
            agent_state,
            target_agent_state,
            ema_adv_var,
            beta_product,
            loss,
            pg_loss,
            cmpo_loss,
            m_loss,
            r_loss,
            v_loss,
            key,
        ) = update_muesli(
            agent_state,
            target_agent_state,
            ema_adv_var,
            beta_product,
            seqs,
            seq_mask,
            key,
        )
        writer.add_scalar("stats/training_time", time.time() - training_time_start, global_step)
        # writer.add_scalar("stats/advantages", advantages.mean().item(), global_step)
        # writer.add_scalar("stats/returns", returns.mean().item(), global_step)
        writer.add_scalar("stats/truncations", np.sum(truncations), global_step)
        writer.add_scalar("stats/terminations", np.sum(terminations), global_step)

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        writer.add_scalar("charts/learning_rate", agent_state.opt_state[1].hyperparams["learning_rate"].item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss[-1, -1].item(), global_step)
        writer.add_scalar("losses/cmpo_loss", cmpo_loss[-1, -1].item(), global_step)
        writer.add_scalar("losses/reward_model_loss", r_loss[-1, -1].item(), global_step)
        writer.add_scalar("losses/value_model_loss", v_loss[-1, -1].item(), global_step)
        writer.add_scalar("losses/policy_model_loss", m_loss[-1, -1].item(), global_step)
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
        writer.add_scalar("stats/buffer_peek_time", buffer_peek_time, global_step)
        writer.add_scalar("stats/buffer_sample_time", buffer_peek_time, global_step)
        writer.add_scalar("stats/update_time", time.time() - update_time_start, global_step)

    if args.save_model:
        model_path = f"runs/{run_name}/{args.exp_name}.cleanrl_model"
        with open(model_path, "wb") as f:
            f.write(
                flax.serialization.to_bytes(
                    [
                        vars(args),
                        [
                            target_agent_state.params.network_params,
                            target_agent_state.params.actor_params,
                            target_agent_state.params.critic_params,
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
            Model=(RepresentationNetwork, Actor, Critic, DynamicsProjector, Dynamics, RewardValueModel, PolicyModel),
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
