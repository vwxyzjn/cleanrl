"""Compatibility for Gymnasium final observation info keys.

Older AutoReset / vector wrappers expose ``final_observation``; Gymnasium
vector SAME_STEP autoreset (needed for MuJoCo v5 / Gymnasium >= 1.0) uses
``final_obs`` / ``_final_obs`` instead. CleanRL off-policy scripts must accept
both (see issue #545).
"""

import numpy as np


def apply_final_obs(next_obs, truncations, infos):
    """Mirror the dual-key handling used in CleanRL off-policy scripts."""
    real_next_obs = next_obs.copy()
    if "final_observation" in infos:
        for idx, trunc in enumerate(truncations):
            if trunc:
                real_next_obs[idx] = infos["final_observation"][idx]
    elif "final_obs" in infos:
        final_mask = infos.get("_final_obs", truncations)
        for idx, has_final in enumerate(final_mask):
            if has_final:
                real_next_obs[idx] = infos["final_obs"][idx]
    return real_next_obs


def test_legacy_final_observation_key():
    next_obs = np.array([[0.0, 0.0], [1.0, 1.0]], dtype=np.float32)
    truncations = np.array([False, True])
    final = np.array([[9.0, 9.0], [8.0, 8.0]], dtype=np.float32)
    infos = {"final_observation": final}
    out = apply_final_obs(next_obs, truncations, infos)
    np.testing.assert_array_equal(out[0], next_obs[0])
    np.testing.assert_array_equal(out[1], final[1])


def test_same_step_final_obs_key():
    next_obs = np.array([[0.0, 0.0], [1.0, 1.0]], dtype=np.float32)
    truncations = np.array([False, True])
    final = np.array([np.array([7.0, 7.0]), np.array([6.0, 6.0])], dtype=object)
    infos = {
        "final_obs": final,
        "_final_obs": np.array([False, True]),
    }
    out = apply_final_obs(next_obs, truncations, infos)
    np.testing.assert_array_equal(out[0], next_obs[0])
    np.testing.assert_array_equal(out[1], final[1])


def test_no_final_keys_leaves_next_obs():
    next_obs = np.array([[0.0, 0.0], [1.0, 1.0]], dtype=np.float32)
    truncations = np.array([True, True])
    out = apply_final_obs(next_obs, truncations, {})
    np.testing.assert_array_equal(out, next_obs)


def test_missing_final_observation_does_not_raise():
    """Regression for KeyError: 'final_observation' under Gymnasium SAME_STEP."""
    next_obs = np.array([[0.0], [1.0]], dtype=np.float32)
    truncations = np.array([True, False])
    infos = {
        "final_obs": np.array([np.array([3.0]), None], dtype=object),
        "_final_obs": np.array([True, False]),
    }
    out = apply_final_obs(next_obs, truncations, infos)
    np.testing.assert_array_equal(out[0], np.array([3.0]))
    np.testing.assert_array_equal(out[1], next_obs[1])
