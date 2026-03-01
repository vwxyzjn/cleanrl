"""Test that episodic return logging uses unique steps per env.

Verifies the fix for https://github.com/vwxyzjn/cleanrl/issues/508
"""
import tempfile

from torch.utils.tensorboard import SummaryWriter
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def test_duplicate_steps_lose_data():
    """Prove the bug: writing multiple scalars at the same step loses data."""
    with tempfile.TemporaryDirectory() as tmpdir:
        writer = SummaryWriter(tmpdir)
        # Simulate 4 envs finishing at the same global_step (the bug)
        global_step = 1000
        values = [50.0, 80.0, 30.0, 60.0]
        for v in values:
            writer.add_scalar("charts/episodic_return", v, global_step)
        writer.close()

        ea = EventAccumulator(tmpdir)
        ea.Reload()
        events = ea.Scalars("charts/episodic_return")
        # BUG: only 1 data point survives (the last one overwrites the rest)
        assert len(events) == 1, f"Expected 1 (bug: data lost), got {len(events)}"
        assert events[0].value == 60.0


def test_unique_steps_preserve_all_data():
    """Prove the fix: writing scalars at offset steps preserves all data."""
    with tempfile.TemporaryDirectory() as tmpdir:
        writer = SummaryWriter(tmpdir)
        # Simulate the fix: each env gets a unique logging_step
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
        # FIX: all 4 data points survive
        assert len(events) == 4, f"Expected 4, got {len(events)}"
        recorded_values = [e.value for e in events]
        assert recorded_values == values
