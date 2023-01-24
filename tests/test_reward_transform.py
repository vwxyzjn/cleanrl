import jax
import jax.numpy as jnp


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
    eps = 1e-2
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
    eps = 1e-2
    num = jnp.sqrt(1 + 4 * eps * (jnp.abs(tr) + 1 + eps)) - 1
    denom = 2 * eps
    return jnp.sign(tr) * ((num / denom) ** 2 - 1)


def test_transform():
    arr = jnp.linspace(-100.0, 100.0, num=2001)
    _, inv_h_h_arr = jax.lax.scan(lambda c, x: (c, inverse_reward_transform(reward_transform(x))), (), arr)
    assert jnp.allclose(arr, inv_h_h_arr, atol=1e-3)
