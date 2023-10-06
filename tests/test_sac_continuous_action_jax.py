# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/sac/#sac_continuous_actionpy
# Implementation adapted from https://github.com/araffin/sbx
from typing import Any, Optional

import jax
import jax.numpy as jnp
import numpy as np
import optax

# import pybullet_envs  # noqa
import tensorflow_probability
from flax.training.train_state import TrainState

tfp = tensorflow_probability.substrates.jax
tfd = tfp.distributions
jax.config.update("jax_platform_name", "cpu")

from cleanrl.sac_continuous_action_jax import Actor, sample_action_and_log_prob


class TanhTransformedDistribution(tfd.TransformedDistribution):
    """
    From https://github.com/ikostrikov/walk_in_the_park
    otherwise mode is not defined for Squashed Gaussian
    """

    def __init__(self, distribution: tfd.Distribution, validate_args: bool = False):
        super().__init__(distribution=distribution, bijector=tfp.bijectors.Tanh(), validate_args=validate_args)

    def mode(self) -> jnp.ndarray:
        return self.bijector.forward(self.distribution.mode())

    @classmethod
    def _parameter_properties(cls, dtype: Optional[Any], num_classes=None):
        td_properties = super()._parameter_properties(dtype, num_classes=num_classes)
        del td_properties["bijector"]
        return td_properties


def test_sample_action_and_log_prob():
    batch_szie = 3
    key = jax.random.PRNGKey(1)
    key, actor_key = jax.random.split(key, 2)
    obs = jax.random.normal(key, shape=(batch_szie, 17))
    actor = Actor(action_dim=6)
    actor_state = TrainState.create(
        apply_fn=actor.apply,
        params=actor.init(actor_key, obs),
        tx=optax.adam(learning_rate=3e-4),
    )

    key, subkey = jax.random.split(key, 2)
    mean, logstd = actor.apply(actor_state.params, obs)
    # 1st way: manually implement sampling
    action, log_prob = sample_action_and_log_prob(mean, logstd, subkey)
    print("Manual", action, log_prob)

    # 2nd way: from the `TanhTransformedDistribution` dist
    dist = TanhTransformedDistribution(
        tfd.MultivariateNormalDiag(loc=mean, scale_diag=jnp.exp(logstd)),
    )
    log_prob2 = dist.log_prob(action)
    print("TanhTransformedDistribution", action, log_prob2)
    np.testing.assert_allclose(log_prob, log_prob2, rtol=1e-3, atol=1e-3)
