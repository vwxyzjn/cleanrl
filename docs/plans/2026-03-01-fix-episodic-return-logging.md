# Fix Episodic Return Logging in PPO Implementations — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Fix episodic return logging so that all completed episodes are recorded with unique step values in TensorBoard, preventing data loss when multiple envs finish at the same `global_step`.

**Architecture:** In each PPO file's logging block, replace raw `global_step` with a per-env `logging_step = global_step - num_envs + i` using `enumerate()`. Also remove the `break` in `ppo_procgen.py`. This ensures every episode gets a unique TensorBoard step, preventing overwrites.

**Tech Stack:** Python, TensorBoard (`SummaryWriter.add_scalar`), Gymnasium vectorized envs

---

## Proposed Changes

### Summary of the Bug

When multiple parallel envs finish an episode on the same step, all episodic returns are logged at the same `global_step`. TensorBoard overwrites previous values at the same step, so only the last env's data survives. Additionally, `ppo_procgen.py` has a `break` that skips all envs after the first finished one.

### Files to Modify

All files share the same pattern. The fix is identical in each: add `enumerate()` and compute `logging_step`.

| File | Lines | Extra Issue |
|---|---|---|
| `cleanrl/ppo.py` | 210–215 | — |
| `cleanrl/ppo_atari.py` | 227–232 | — |
| `cleanrl/ppo_procgen.py` | 244–249 | Has `break` to remove |
| `cleanrl/ppo_atari_lstm.py` | 259–264 | — |
| `cleanrl/ppo_continuous_action.py` | 225–230 | — |
| `cleanrl/ppo_atari_multigpu.py` | 277–282 | — |

---

### Task 1: Write the failing test

**Files:**
- Create: `tests/test_episodic_logging.py`

This test proves the bug exists by simulating TensorBoard logging with duplicate steps vs unique steps.

**Step 1: Write the test**

```python
"""Test that episodic return logging uses unique steps per env.

Verifies the fix for https://github.com/vwxyzjn/cleanrl/issues/508
"""
import os
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
```

**Step 2: Run the test to verify it passes (it tests the bug, not our code yet)**

Run: `python -m pytest tests/test_episodic_logging.py -v`
Expected: PASS (both tests should pass — they prove the TensorBoard behavior)

**Step 3: Commit**

```bash
git add tests/test_episodic_logging.py
git commit -m "test(logging): add tests proving episodic return logging bug #508

ROOT CAUSE:
Multiple envs finishing at the same global_step causes TensorBoard
to overwrite all but the last value.

CHANGES:
- Add test proving duplicate steps lose data
- Add test proving unique offset steps preserve all data

IMPACT:
Establishes test evidence for the fix

FILES MODIFIED:
- tests/test_episodic_logging.py [NEW]"
```

---

### Task 2: Fix `ppo.py`

**Files:**
- Modify: `cleanrl/ppo.py:210-215`

**Step 1: Apply the fix**

Change:
```python
            if "final_info" in infos:
                for info in infos["final_info"]:
                    if info and "episode" in info:
                        print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
                        writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
                        writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)
```

To:
```python
            if "final_info" in infos:
                for i, info in enumerate(infos["final_info"]):
                    if info and "episode" in info:
                        logging_step = global_step - args.num_envs + i
                        print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
                        writer.add_scalar("charts/episodic_return", info["episode"]["r"], logging_step)
                        writer.add_scalar("charts/episodic_length", info["episode"]["l"], logging_step)
```

**Step 2: Verify no syntax errors**

Run: `python -c "import ast; ast.parse(open('cleanrl/ppo.py').read()); print('OK')"`
Expected: `OK`

**Step 3: Commit**

```bash
git add cleanrl/ppo.py
git commit -m "fix(ppo): use per-env logging_step for episodic return logging #508

ROOT CAUSE:
Multiple envs logging at same global_step causes TensorBoard overwrites

CHANGES:
- Add enumerate() to final_info loop
- Compute logging_step = global_step - num_envs + i

FILES MODIFIED:
- cleanrl/ppo.py"
```

---

### Task 3: Fix `ppo_atari.py`

**Files:**
- Modify: `cleanrl/ppo_atari.py:227-232`

**Step 1: Apply the fix**

Same pattern as Task 2. Change `for info in` to `for i, info in enumerate(` and use `logging_step = global_step - args.num_envs + i`.

**Step 2: Verify no syntax errors**

Run: `python -c "import ast; ast.parse(open('cleanrl/ppo_atari.py').read()); print('OK')"`
Expected: `OK`

**Step 3: Commit**

```bash
git add cleanrl/ppo_atari.py
git commit -m "fix(ppo_atari): use per-env logging_step for episodic return logging #508

FILES MODIFIED:
- cleanrl/ppo_atari.py"
```

---

### Task 4: Fix `ppo_procgen.py` (remove `break` + add offset)

**Files:**
- Modify: `cleanrl/ppo_procgen.py:244-249`

**Step 1: Apply the fix**

Change:
```python
            for item in info:
                if "episode" in item.keys():
                    print(f"global_step={global_step}, episodic_return={item['episode']['r']}")
                    writer.add_scalar("charts/episodic_return", item["episode"]["r"], global_step)
                    writer.add_scalar("charts/episodic_length", item["episode"]["l"], global_step)
                    break
```

To:
```python
            for i, item in enumerate(info):
                if "episode" in item.keys():
                    logging_step = global_step - args.num_envs + i
                    print(f"global_step={global_step}, episodic_return={item['episode']['r']}")
                    writer.add_scalar("charts/episodic_return", item["episode"]["r"], logging_step)
                    writer.add_scalar("charts/episodic_length", item["episode"]["l"], logging_step)
```

> [!IMPORTANT]
> The `break` is removed entirely — it was discarding all envs after the first.

**Step 2: Verify no syntax errors**

Run: `python -c "import ast; ast.parse(open('cleanrl/ppo_procgen.py').read()); print('OK')"`
Expected: `OK`

**Step 3: Commit**

```bash
git add cleanrl/ppo_procgen.py
git commit -m "fix(ppo_procgen): remove break and use per-env logging_step #508

ROOT CAUSE:
break statement discarded all episodes after the first env.
All episodes logged at same global_step caused overwrites.

CHANGES:
- Remove break statement
- Add enumerate() and compute logging_step

FILES MODIFIED:
- cleanrl/ppo_procgen.py"
```

---

### Task 5: Fix `ppo_atari_lstm.py`

**Files:**
- Modify: `cleanrl/ppo_atari_lstm.py:259-264`

**Step 1: Apply the fix**

Same pattern: `for i, info in enumerate(...)` + `logging_step = global_step - args.num_envs + i`.

**Step 2: Verify no syntax errors**

Run: `python -c "import ast; ast.parse(open('cleanrl/ppo_atari_lstm.py').read()); print('OK')"`
Expected: `OK`

**Step 3: Commit**

```bash
git add cleanrl/ppo_atari_lstm.py
git commit -m "fix(ppo_atari_lstm): use per-env logging_step for episodic return logging #508

FILES MODIFIED:
- cleanrl/ppo_atari_lstm.py"
```

---

### Task 6: Fix `ppo_continuous_action.py`

**Files:**
- Modify: `cleanrl/ppo_continuous_action.py:225-230`

**Step 1: Apply the fix**

Same pattern.

**Step 2: Verify no syntax errors**

Run: `python -c "import ast; ast.parse(open('cleanrl/ppo_continuous_action.py').read()); print('OK')"`
Expected: `OK`

**Step 3: Commit**

```bash
git add cleanrl/ppo_continuous_action.py
git commit -m "fix(ppo_continuous_action): use per-env logging_step for episodic return logging #508

FILES MODIFIED:
- cleanrl/ppo_continuous_action.py"
```

---

### Task 7: Fix `ppo_atari_multigpu.py`

**Files:**
- Modify: `cleanrl/ppo_atari_multigpu.py:277-282`

**Step 1: Apply the fix**

Same pattern, but note this file uses `args.local_num_envs` for the local env count. However `global_step` increments by `args.num_envs` (total across all GPUs). The `infos["final_info"]` list has `local_num_envs` entries. So the correct formula is `logging_step = global_step - args.num_envs + i`.

**Step 2: Verify no syntax errors**

Run: `python -c "import ast; ast.parse(open('cleanrl/ppo_atari_multigpu.py').read()); print('OK')"`
Expected: `OK`

**Step 3: Commit**

```bash
git add cleanrl/ppo_atari_multigpu.py
git commit -m "fix(ppo_atari_multigpu): use per-env logging_step for episodic return logging #508

FILES MODIFIED:
- cleanrl/ppo_atari_multigpu.py"
```

---

## Verification Plan

### Automated Tests

1. **TensorBoard behavior test** (no gym/envs needed):
   ```
   python -m pytest tests/test_episodic_logging.py -v
   ```
   Proves that duplicate steps lose data and offset steps preserve it.

2. **Syntax check on all modified files**:
   ```
   python -c "import ast; [ast.parse(open(f).read()) for f in ['cleanrl/ppo.py', 'cleanrl/ppo_atari.py', 'cleanrl/ppo_procgen.py', 'cleanrl/ppo_atari_lstm.py', 'cleanrl/ppo_continuous_action.py', 'cleanrl/ppo_atari_multigpu.py']]; print('All OK')"
   ```

3. **Grep verification** — confirm no file still uses raw `global_step` in episodic logging:
   ```
   grep -n 'add_scalar.*episodic_return.*global_step' cleanrl/ppo.py cleanrl/ppo_atari.py cleanrl/ppo_procgen.py cleanrl/ppo_atari_lstm.py cleanrl/ppo_continuous_action.py cleanrl/ppo_atari_multigpu.py
   ```
   Expected: **no matches** (all should use `logging_step` now)

4. **Grep verification** — confirm `break` is removed from `ppo_procgen.py`:
   ```
   grep -n 'break' cleanrl/ppo_procgen.py
   ```
   Expected: no `break` inside the episodic logging block

### Manual Verification (optional, for user)

Run `ppo.py` with a short training and check TensorBoard:
```bash
python cleanrl/ppo.py --env-id CartPole-v1 --num-envs 4 --total-timesteps 5000
tensorboard --logdir runs/
```
Then open TensorBoard → `charts/episodic_return` and verify that data points appear densely (not sparse with big gaps). Compare against a run from the `main` branch to see more data points recorded.
