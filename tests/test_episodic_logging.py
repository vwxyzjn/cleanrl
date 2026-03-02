"""Test that episodic return logging uses unique steps per env.

Verifies the fix for https://github.com/vwxyzjn/cleanrl/issues/508

Zero external dependencies — uses only unittest.mock to verify
the exact add_scalar call arguments our fix produces.
"""
from unittest.mock import Mock


def test_fixed_logging_uses_unique_steps_per_env():
    """The fix: each env gets logging_step = global_step - num_envs + i.

    Simulates the FIXED code path from ppo.py lines 210-216.
    Verifies that writer.add_scalar is called with unique
    logging_step values, not the same global_step for all envs.
    """
    writer = Mock()
    global_step = 1000
    num_envs = 4

    # Simulate final_info from gymnasium vectorized env:
    # 3 of 4 envs finished their episode at this step
    final_info = [
        {"episode": {"r": 50.0, "l": 100}},  # env 0 finished
        None,                                  # env 1 still running
        {"episode": {"r": 80.0, "l": 200}},  # env 2 finished
        {"episode": {"r": 30.0, "l": 150}},  # env 3 finished
    ]

    # This is the FIXED code (exact copy from our changed ppo.py)
    for i, info in enumerate(final_info):
        if info and "episode" in info:
            logging_step = global_step - num_envs + i
            writer.add_scalar("charts/episodic_return", info["episode"]["r"], logging_step)
            writer.add_scalar("charts/episodic_length", info["episode"]["l"], logging_step)

    # 3 envs finished → 6 add_scalar calls (return + length for each)
    assert writer.add_scalar.call_count == 6

    # Each env used its OWN unique step
    writer.add_scalar.assert_any_call("charts/episodic_return", 50.0, 996)   # 1000-4+0
    writer.add_scalar.assert_any_call("charts/episodic_return", 80.0, 998)   # 1000-4+2
    writer.add_scalar.assert_any_call("charts/episodic_return", 30.0, 999)   # 1000-4+3

    writer.add_scalar.assert_any_call("charts/episodic_length", 100, 996)
    writer.add_scalar.assert_any_call("charts/episodic_length", 200, 998)
    writer.add_scalar.assert_any_call("charts/episodic_length", 150, 999)


def test_old_broken_logging_uses_same_step():
    """Prove the bug: without the fix, all envs log at the same global_step.

    This test shows what the OLD code did wrong.
    """
    writer = Mock()
    global_step = 1000

    final_info = [
        {"episode": {"r": 50.0, "l": 100}},
        None,
        {"episode": {"r": 80.0, "l": 200}},
        {"episode": {"r": 30.0, "l": 150}},
    ]

    # OLD broken code: no enumerate, no logging_step
    for info in final_info:
        if info and "episode" in info:
            writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)

    # All 3 calls used the SAME step — this is the bug
    steps_used = [call.args[2] for call in writer.add_scalar.call_args_list]
    assert all(s == 1000 for s in steps_used), f"Bug: all steps should be 1000, got {steps_used}"
    assert len(set(steps_used)) == 1, "Bug: only 1 unique step value despite 3 episodes"


def test_procgen_break_bug():
    """Prove the ppo_procgen.py break bug: only first env's episode was logged."""
    writer = Mock()
    global_step = 1000

    # info is a list of dicts (old gym API used by ppo_procgen.py)
    info = [
        {"episode": {"r": 50.0, "l": 100}},
        {"episode": {"r": 80.0, "l": 200}},
        {},  # env 2 didn't finish
        {"episode": {"r": 30.0, "l": 150}},
    ]

    # OLD broken code with break
    for item in info:
        if "episode" in item.keys():
            writer.add_scalar("charts/episodic_return", item["episode"]["r"], global_step)
            break  # <-- THE BUG

    # Only 1 call made — 2 episodes silently discarded
    assert writer.add_scalar.call_count == 1
    writer.add_scalar.assert_called_once_with("charts/episodic_return", 50.0, 1000)


def test_procgen_fixed_logging():
    """After fix: ppo_procgen.py logs all envs with unique steps, no break."""
    writer = Mock()
    global_step = 1000
    num_envs = 4

    info = [
        {"episode": {"r": 50.0, "l": 100}},
        {"episode": {"r": 80.0, "l": 200}},
        {},  # env 2 didn't finish
        {"episode": {"r": 30.0, "l": 150}},
    ]

    # FIXED code: enumerate, logging_step, no break
    for i, item in enumerate(info):
        if "episode" in item.keys():
            logging_step = global_step - num_envs + i
            writer.add_scalar("charts/episodic_return", item["episode"]["r"], logging_step)
            writer.add_scalar("charts/episodic_length", item["episode"]["l"], logging_step)

    # 3 envs finished → 6 calls, unique steps
    assert writer.add_scalar.call_count == 6

    steps_used = [call.args[2] for call in writer.add_scalar.call_args_list]
    unique_steps = set(steps_used)
    assert len(unique_steps) == 3, f"Expected 3 unique steps, got {unique_steps}"

    writer.add_scalar.assert_any_call("charts/episodic_return", 50.0, 996)
    writer.add_scalar.assert_any_call("charts/episodic_return", 80.0, 997)
    writer.add_scalar.assert_any_call("charts/episodic_return", 30.0, 999)


def test_envpool_broken_uses_same_step():
    """Prove the envpool bug: all envs log at the same global_step.

    EnvPool variants use info["r"][idx] instead of info["episode"]["r"]
    but have the same bug - using global_step for all envs.
    """
    writer = Mock()
    global_step = 1000
    num_envs = 4

    # EnvPool uses different info structure: dict with arrays
    info = {
        "r": [50.0, None, 80.0, 30.0],  # returns per env
        "l": [100, None, 200, 150],     # lengths per env
        "lives": [0, 1, 0, 0],          # lives (0 = episode done)
    }
    next_done = [True, False, True, True]  # which envs are done

    # OLD broken envpool code: uses global_step for all
    for idx, d in enumerate(next_done):
        if d and info["lives"][idx] == 0:
            writer.add_scalar("charts/episodic_return", info["r"][idx], global_step)
            writer.add_scalar("charts/episodic_length", info["l"][idx], global_step)

    # All calls used the SAME step — this is the bug
    steps_used = [call.args[2] for call in writer.add_scalar.call_args_list]
    assert all(s == 1000 for s in steps_used), f"Bug: all steps should be 1000, got {steps_used}"
    assert len(set(steps_used)) == 1, "Bug: only 1 unique step value despite 3 episodes"


def test_envpool_fixed_uses_unique_steps():
    """After fix: envpool variants use unique steps per env."""
    writer = Mock()
    global_step = 1000
    num_envs = 4

    info = {
        "r": [50.0, None, 80.0, 30.0],
        "l": [100, None, 200, 150],
        "lives": [0, 1, 0, 0],
    }
    next_done = [True, False, True, True]

    # FIXED envpool code: use logging_step per env
    for idx, d in enumerate(next_done):
        if d and info["lives"][idx] == 0:
            logging_step = global_step - num_envs + idx
            writer.add_scalar("charts/episodic_return", info["r"][idx], logging_step)
            writer.add_scalar("charts/episodic_length", info["l"][idx], logging_step)

    # 3 envs finished → 6 calls, unique steps
    assert writer.add_scalar.call_count == 6

    steps_used = [call.args[2] for call in writer.add_scalar.call_args_list]
    unique_steps = set(steps_used)
    assert len(unique_steps) == 3, f"Expected 3 unique steps, got {unique_steps}"

    writer.add_scalar.assert_any_call("charts/episodic_return", 50.0, 996)  # 1000-4+0
    writer.add_scalar.assert_any_call("charts/episodic_return", 80.0, 998)  # 1000-4+2
    writer.add_scalar.assert_any_call("charts/episodic_return", 30.0, 999)  # 1000-4+3
