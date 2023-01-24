import jax.numpy as jnp

atoms = jnp.array([2, 3, 4])


def project_to_atoms(scalars):
    scalars = jnp.clip(scalars, atoms.min(), atoms.max())
    distribution = jnp.zeros(scalars.shape + atoms.shape)
    ceil_scalars = jnp.ceil(scalars)
    distribution = jnp.where(
        jnp.floor(scalars)[..., None] == atoms,
        jnp.where(
            (ceil_scalars == scalars),
            jnp.ones_like(scalars),
            ceil_scalars - scalars,
        )[..., None],
        distribution,
    )
    distribution = distribution.at[..., 1:].add(
        1 - jnp.where((0 < distribution[..., :-1]) & (distribution[..., :-1] < 1), distribution[..., :-1], 1)
    )
    return distribution


def distribution_expectation(arr):
    return (arr * atoms).sum(-1)


def test_project_to_atoms():
    scalars = jnp.linspace(1.9, 4.1, 23)
    assert jnp.allclose(jnp.clip(scalars, atoms.min(), atoms.max()), distribution_expectation(project_to_atoms(scalars)))
