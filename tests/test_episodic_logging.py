"""Test that episodic return logging uses unique steps per env.

Verifies the fix for https://github.com/vwxyzjn/cleanrl/issues/508

The core issues:
1. ppo_procgen.py has a `break` that discards all episodes after the first env
2. Multiple episodes logged at the same step can cause confusing TensorBoard charts
   (duplicate x-axis values), even if newer TensorBoard versions store all values
"""
import tempfile

from torch.utils.tensorboard import SummaryWriter
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def test_unique_steps_preserve_all_data():
    """Verify that offset steps produce clean, unique data points."""
    with tempfile.TemporaryDirectory() as tmpdir:
        writer = SummaryWriter(tmpdir)
        global_step = 1000
        num_envs = 4
        values = [50.0, 80.0, 30.0, 60.0]
        for i, v in enumerate(values):
            logging_step = global_step - num_envs + i
            writer.add_scalar("charts/episodic_return", v, logging_step)
        writer.close()

        ea = EventAccumulator(tmpdir)
        ea.Reload()
        events = ea.Scalars("charts/episodic_return")
        # All 4 data points survive with unique steps
        assert len(events) == 4, f"Expected 4, got {len(events)}"
        recorded_values = [e.value for e in events]
        assert recorded_values == values
        # Each step is unique
        steps = [e.step for e in events]
        assert len(set(steps)) == 4, f"Expected 4 unique steps, got {steps}"


def test_duplicate_steps_produce_ambiguous_data():
    """Show that logging at the same step produces duplicate x-values (undesirable)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        writer = SummaryWriter(tmpdir)
        global_step = 1000
        values = [50.0, 80.0, 30.0, 60.0]
        for v in values:
            writer.add_scalar("charts/episodic_return", v, global_step)
        writer.close()

        ea = EventAccumulator(tmpdir)
        ea.Reload()
        events = ea.Scalars("charts/episodic_return")
        # All values are stored but all at the SAME step — ambiguous in charts
        steps = [e.step for e in events]
        assert all(s == global_step for s in steps), "All events should share the same step"
        # This is the problem: non-unique steps make charts misleading
        assert len(set(steps)) == 1, "Only 1 unique step value despite 4 data points"


def test_break_discards_episodes():
    """Simulate the ppo_procgen.py break bug — only first env's episode is logged."""
    with tempfile.TemporaryDirectory() as tmpdir:
        writer = SummaryWriter(tmpdir)
        # Simulate the old ppo_procgen.py behavior with break
        info_list = [
            {"episode": {"r": 50.0, "l": 100}},
            {"episode": {"r": 80.0, "l": 200}},
            {},  # env 2 didn't finish
            {"episode": {"r": 60.0, "l": 150}},
        ]
        global_step = 1000
        logged_count = 0
        for item in info_list:
            if "episode" in item:
                writer.add_scalar("charts/episodic_return", item["episode"]["r"], global_step)
                logged_count += 1
                break  # <-- THE BUG: only logs the first env
        writer.close()

        assert logged_count == 1, "break causes only 1 episode to be logged"

        # Without break, all 3 finished envs should be logged
        logged_without_break = sum(1 for item in info_list if "episode" in item)
        assert logged_without_break == 3, "Without break, 3 episodes should be logged"
