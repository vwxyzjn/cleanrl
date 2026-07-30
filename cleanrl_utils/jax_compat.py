"""Compatibility shims for JAX/Flax APIs removed or changed after envpool 0.6.x was released."""

import sys
import types

import jax
import optax
from jax import core
from jax.lib import xla_client

if "jax.abstract_arrays" not in sys.modules:
    abstract_arrays = types.ModuleType("jax.abstract_arrays")
    abstract_arrays.ShapedArray = core.ShapedArray
    sys.modules["jax.abstract_arrays"] = abstract_arrays
    jax.abstract_arrays = abstract_arrays

if not hasattr(xla_client, "register_cpu_custom_call_target"):

    def register_cpu_custom_call_target(name, fn):
        xla_client.register_custom_call_target(name, fn, platform="cpu")

    xla_client.register_cpu_custom_call_target = register_cpu_custom_call_target

from flax.training.train_state import TrainState as _FlaxTrainState


class TrainState(_FlaxTrainState):
    """TrainState that supports flax struct dataclass params (e.g. AgentParams)."""

    @classmethod
    def create(cls, *, apply_fn, params, tx, **kwargs):
        return cls(
            step=0,
            apply_fn=apply_fn,
            params=params,
            tx=tx,
            opt_state=tx.init(params),
            **kwargs,
        )

    def apply_gradients(self, *, grads, **kwargs):
        updates, new_opt_state = self.tx.update(grads, self.opt_state, self.params)
        new_params = optax.apply_updates(self.params, updates)
        return self.replace(
            step=self.step + 1,
            params=new_params,
            opt_state=new_opt_state,
            **kwargs,
        )
