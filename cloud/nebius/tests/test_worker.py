import json

import pytest

from cloud.nebius.worker import child_environment, command


def test_shell_free_command_and_credential_separation(journal, monkeypatch):
    config = json.loads(journal.rows()[0]["worker"])
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "storage-secret")
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "storage-key")
    monkeypatch.setenv("NEBIUS_IAM_TOKEN", "controller-token")
    monkeypatch.setenv("WANDB_API_KEY", "tracking-secret")
    monkeypatch.setenv("WANDB_RESUME", "allow")
    env = child_environment(config)
    assert not any(k.startswith(("AWS_", "NEBIUS_")) for k in env)
    assert "WANDB_API_KEY" not in env and "WANDB_RESUME" not in env
    args = command(config, "python")
    assert "--save-model" in args and "--cuda" in args and "--track" not in args
    assert "bash" not in args and "-c" not in args


def test_tracking_is_explicit(journal, monkeypatch):
    config = json.loads(journal.rows()[0]["worker"])
    config["tracking"] = {"mode": "online", "project": "project", "entity": "team"}
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    with pytest.raises(ValueError):
        child_environment(config)
    monkeypatch.setenv("WANDB_API_KEY", "example-key")
    env = child_environment(config)
    assert env["WANDB_API_KEY"] == "example-key"
    assert env["WANDB_RUN_ID"] == config["attempt"]
    assert "--track" in command(config, "python")


@pytest.mark.parametrize("outcome", ["success", "failure", "timeout", "signal"])
def test_real_child_exit_timeout_signal_and_export(journal, tmp_path, monkeypatch, outcome):
    import os
    import signal
    import sys
    import threading
    from types import SimpleNamespace

    from cloud.nebius import worker
    from cloud.nebius.artifacts import LocalStore, download
    from cloud.nebius.config import digest

    config = json.loads(journal.rows()[0]["worker"])
    config["worker_timeout"] = 1 if outcome == "timeout" else 10
    fake_torch = SimpleNamespace(__version__="test", version=SimpleNamespace(cuda=None))
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setattr(worker.shutil, "which", lambda name: "/test/ffmpeg")
    monkeypatch.setattr(worker, "validate_model", lambda path: None)
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "test-secret-do-not-export")
    script = "from pathlib import Path; import time; Path('runs/r').mkdir(parents=True); "
    script += "Path('runs/r/dqn.cleanrl_model').write_bytes(b'policy'); print('test-secret-do-not-export',flush=True); "
    script += (
        "time.sleep(30)" if outcome in {"timeout", "signal"} else "raise SystemExit(1)" if outcome == "failure" else "pass"
    )
    monkeypatch.setattr(worker, "command", lambda *a: [sys.executable, "-u", "-c", script])
    store = LocalStore(tmp_path / "remote")
    timer = None
    if outcome == "signal":
        timer = threading.Timer(0.5, lambda: os.kill(os.getpid(), signal.SIGTERM))
        timer.start()
    try:
        result = worker.run(config, store, cpu=True)
    finally:
        if timer:
            timer.cancel()
            timer.join()
    assert result == (0 if outcome == "success" else 1)
    manifest = json.loads(store.get(config["prefix"] + "manifest.json", 100000))
    assert manifest["success"] == (outcome == "success")
    assert manifest["metadata"]["cancelled"] == (outcome in {"timeout", "signal"})
    output = store.get(config["prefix"] + "files/stdout.txt", 100000)
    assert b"test-secret-do-not-export" not in output
    if outcome == "success":
        download(
            store, config["prefix"], tmp_path / "download", {"attempt": config["attempt"], "config_sha256": digest(config)}
        )
    else:
        assert not store.exists(config["prefix"] + "complete.json")


def test_worker_config_rejects_output_escape(journal):
    from cloud.nebius.config import validate_worker

    config = json.loads(journal.rows()[0]["worker"])
    config["prefix"] = "../../outside/"
    with pytest.raises(ValueError):
        validate_worker(config)
