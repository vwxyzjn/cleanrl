from copy import deepcopy
from functools import partial
from typing import Callable

import flax
import jax
import jax.numpy as jnp
import numpy as np


def test_compute_gae():
    @flax.struct.dataclass
    class Storage:
        dones: jnp.array
        values: jnp.array
        advantages: jnp.array
        returns: jnp.array
        rewards: jnp.array

    def compute_gae_once(carry, inp, gamma, gae_lambda):
        advantages = carry
        nextdone, nextvalues, curvalues, reward = inp
        nextnonterminal = 1.0 - nextdone

        delta = reward + gamma * nextvalues * nextnonterminal - curvalues
        advantages = delta + gamma * gae_lambda * nextnonterminal * advantages
        return advantages, advantages

    def compute_gae_scan(
        next_done: np.ndarray,
        next_value: np.ndarray,
        storage: Storage,
        num_envs: int,
        compute_gae_once_fn: Callable,
    ):
        advantages = jnp.zeros((num_envs,))
        dones = jnp.concatenate([storage.dones, next_done[None, :]], axis=0)
        values = jnp.concatenate([storage.values, next_value[None, :]], axis=0)
        _, advantages = jax.lax.scan(
            compute_gae_once_fn, advantages, (dones[1:], values[1:], values[:-1], storage.rewards), reverse=True
        )
        storage = storage.replace(
            advantages=advantages,
            returns=advantages + storage.values,
        )
        return storage

    def compute_gae_python_loop(
        next_done: np.ndarray, next_value: np.ndarray, storage: Storage, num_steps: int, gamma: float, gae_lambda: float
    ):
        storage = storage.replace(advantages=storage.advantages.at[:].set(0.0))
        lastgaelam = 0
        for t in reversed(range(num_steps)):
            if t == num_steps - 1:
                nextnonterminal = 1.0 - next_done
                nextvalues = next_value
            else:
                nextnonterminal = 1.0 - storage.dones[t + 1]
                nextvalues = storage.values[t + 1]
            delta = storage.rewards[t] + gamma * nextvalues * nextnonterminal - storage.values[t]
            lastgaelam = delta + gamma * gae_lambda * nextnonterminal * lastgaelam
            storage = storage.replace(advantages=storage.advantages.at[t].set(lastgaelam))
        storage = storage.replace(returns=storage.advantages + storage.values)
        return storage

    num_steps = 123
    num_envs = 7
    gamma = 0.99
    gae_lambda = 0.95
    seed = 42
    compute_gae_once_fn = partial(compute_gae_once, gamma=gamma, gae_lambda=gae_lambda)
    compute_gae_scan = jax.jit(partial(compute_gae_scan, num_envs=num_envs, compute_gae_once_fn=compute_gae_once_fn))
    compute_gae_python_loop = jax.jit(
        partial(compute_gae_python_loop, num_steps=num_steps, gamma=gamma, gae_lambda=gae_lambda)
    )
    key = jax.random.PRNGKey(seed)
    key, *k = jax.random.split(key, 6)
    storage1 = Storage(
        dones=jax.random.randint(k[0], (num_steps, num_envs), 0, 2),
        values=jax.random.uniform(k[1], (num_steps, num_envs)),
        advantages=jnp.zeros((num_steps, num_envs)),
        returns=jnp.zeros((num_steps, num_envs)),
        rewards=jax.random.uniform(k[2], (num_steps, num_envs), minval=-1, maxval=1),
    )
    storage2 = deepcopy(storage1)
    next_value = jax.random.uniform(k[3], (num_envs,))
    next_done = jax.random.randint(k[4], (num_envs,), 0, 2)
    storage1 = compute_gae_scan(next_done, next_value, storage1)
    storage2 = compute_gae_python_loop(next_done, next_value, storage2)
    assert (storage1.advantages == storage2.advantages).all()
    assert (storage1.returns == storage2.returns).all()
