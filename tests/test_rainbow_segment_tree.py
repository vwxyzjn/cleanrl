import numpy as np

from cleanrl.rainbow_atari import MinSegmentTree, SumSegmentTree


def test_sum_tree_retrieve_prefix_order():
    # https://github.com/vwxyzjn/cleanrl/issues/546: with equal priorities, prefix masses
    # 0.5, 1.5, 2.5 must map to indices 0, 1, 2 (the unpadded layout returned [1, 2, 0])
    tree = SumSegmentTree(3)
    for idx in range(3):
        tree.update(idx, 1.0)
    assert [tree.retrieve(value) for value in (0.5, 1.5, 2.5)] == [0, 1, 2]


def test_sum_tree_retrieve_matches_cumulative_sum():
    for capacity in range(1, 34):
        rng = np.random.default_rng(capacity)
        priorities = rng.uniform(0.1, 2.0, size=capacity).astype(np.float32)
        tree = SumSegmentTree(capacity)
        for idx, priority in enumerate(priorities):
            tree.update(idx, priority)
        assert np.isclose(tree.total(), priorities.sum())
        cumulative = np.cumsum(priorities, dtype=np.float32)
        starts = np.concatenate([[np.float32(0.0)], cumulative[:-1]])
        midpoints = (starts + cumulative) / 2
        assert [tree.retrieve(value) for value in midpoints] == list(range(capacity)), capacity


def test_sum_tree_get_returns_stored_priority():
    capacity = 6
    tree = SumSegmentTree(capacity)
    priorities = [0.5, 1.0, 0.125, 2.0, 0.25, 1.5]
    for idx, priority in enumerate(priorities):
        tree.update(idx, priority)
    assert [tree.get(idx) for idx in range(capacity)] == priorities


def test_sum_tree_sampling_proportional_to_priorities():
    capacity = 6
    rng = np.random.default_rng(0)
    priorities = rng.uniform(0.1, 2.0, size=capacity).astype(np.float32)
    tree = SumSegmentTree(capacity)
    for idx, priority in enumerate(priorities):
        tree.update(idx, priority)
    draws = 200000
    values = rng.uniform(0.0, tree.total(), size=draws)
    counts = np.bincount([tree.retrieve(value) for value in values], minlength=capacity)
    np.testing.assert_allclose(counts / draws, priorities / priorities.sum(), atol=0.01)


def test_sum_tree_never_retrieves_empty_slots():
    # a partially filled buffer must never sample an index that was not written
    tree = SumSegmentTree(5)
    for idx in range(3):
        tree.update(idx, 1.0)
    values = np.linspace(1e-6, tree.total(), num=1000)
    assert {tree.retrieve(value) for value in values} <= {0, 1, 2}


def test_min_tree_tracks_minimum():
    for capacity in range(1, 34):
        rng = np.random.default_rng(capacity)
        priorities = rng.uniform(0.1, 2.0, size=capacity).astype(np.float32)
        tree = MinSegmentTree(capacity)
        for idx, priority in enumerate(priorities):
            tree.update(idx, priority)
        assert tree.min() == priorities.min()
        min_idx = int(np.argmin(priorities))
        tree.update(min_idx, 5.0)
        priorities[min_idx] = 5.0
        assert tree.min() == priorities.min()
