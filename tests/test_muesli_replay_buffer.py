import jax
import jax.numpy as jnp

from cleanrl.muesli_atari_envpool_async_jax_scan_impalanet_machado import (
    BoundaryPointer,
    UniformBuffer,
)


def test_boundary_pointer():
    pointer = BoundaryPointer.init(3, 2)

    pointer = pointer.advance(jnp.array([2, 0]))
    assert (pointer.head == jnp.array([1, 0, 1])).all()
    assert (pointer.length == jnp.array([1, 0, 1])).all()

    pointer = pointer.advance(jnp.array([2, 0]))
    assert (pointer.head == jnp.array([0, 0, 0])).all()
    assert (pointer.tail == jnp.array([0, 0, 0])).all()
    assert (pointer.length == jnp.array([2, 0, 2])).all()

    pointer = pointer.advance(jnp.array([2, 0]))
    assert (pointer.head == jnp.array([1, 0, 1])).all()
    assert (pointer.tail == jnp.array([1, 0, 1])).all()
    assert (pointer.length == jnp.array([2, 0, 2])).all()

    pointer = pointer.advance(jnp.array([0, 1]))
    assert (pointer.head == jnp.array([0, 1, 1])).all()
    assert (pointer.tail == jnp.array([0, 0, 1])).all()
    assert (pointer.length == jnp.array([2, 1, 2])).all()

    pointer = pointer.reset()
    assert (pointer.head == jnp.array([0, 1, 1])).all()
    assert (pointer.tail == jnp.array([0, 1, 1])).all()
    assert (pointer.length == jnp.array([0, 0, 0])).all()

    pointer = BoundaryPointer.init(1, 3)
    assert (pointer.head == jnp.array([0])).all()
    assert (pointer.length == jnp.array([0])).all()

    pointer = pointer.advance(jnp.array([0]))
    assert (pointer.head == jnp.array([1])).all()
    assert (pointer.length == jnp.array([1])).all()

    pointer = pointer.advance(jnp.array([0]))
    assert (pointer.head == jnp.array([2])).all()
    assert (pointer.length == jnp.array([2])).all()


def test_buffer():
    buffer = UniformBuffer.init([jnp.array(0), jnp.array(1)], 3, 5)
    buffer = buffer.push_env_updates([jnp.array([3, 4]), jnp.array([5, 6])], jnp.array([0, 1]))
    buffer = buffer.push_env_updates([jnp.array([7, 8]), jnp.array([9, 10])], jnp.array([2, 1]))

    assert buffer.data[0][0, 0] == 3
    assert buffer.data[0][1, 0] == 4
    assert buffer.data[0][1, 1] == 8
    assert buffer.data[0][2, 0] == 7
    assert buffer.data[1][0, 0] == 5
    assert buffer.data[1][1, 0] == 6
    assert buffer.data[1][1, 1] == 10
    assert buffer.data[1][2, 0] == 9

    buffer = buffer.reset_online_queue()
    for _ in range(4):
        buffer = buffer.push_env_updates([jnp.array([99]), jnp.array([99])], jnp.array([1]))

    assert buffer.data[0][1, 2] == 99
    assert buffer.data[0][1, 3] == 99
    assert buffer.data[0][1, 4] == 99
    assert buffer.data[0][1, 0] == 99
    assert (buffer.online_queue_ind.head == buffer.full_buffer_ind.head).all()
    assert (buffer.online_queue_ind.length == jnp.array([0, 4, 0])).all()
    assert (buffer.full_buffer_ind.length == jnp.array([1, 5, 1])).all()

    for _ in range(2):
        buffer = buffer.push_env_updates([jnp.array([100]), jnp.array([100])], jnp.array([1]))
    assert buffer.data[0][1, 1] == 100
    assert buffer.data[0][1, 2] == 100

    seqs, seq_mask = buffer.sample_online_queue(jax.random.PRNGKey(0), 3, 5)
    for seq in seqs:
        assert seq.shape == (3, 5)
        assert (seq >= 99).all()
    assert seq_mask.shape == (3, 5)

    seqs, seq_mask = buffer.sample_replay_buffer(jax.random.PRNGKey(0), 3, 5)
    for seq in seqs:
        assert seq.shape == (3, 5)
    assert seq_mask.shape == (3, 5)
