import json
import os
import runpy
import shlex
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import requests


def test_submit_exp_no_build():
    subprocess.run(
        "uv run python -m cleanrl_utils.submit_exp --docker-tag vwxyzjn/cleanrl:latest --wandb-key xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx",
        shell=True,
        check=True,
    )


@pytest.mark.parametrize(
    "original_program,code_path,expected_filename,remove_entity",
    [
        pytest.param("/original-machine/work/cleanrl/ppo.py", "cleanrl/ppo.py", "ppo.py", False, id="absolute-original-path"),
        pytest.param("cleanrl/ppo.py", "cleanrl/ppo.py", "ppo.py", False, id="relative-original-path"),
        pytest.param("workspace/agents/ppo.py", "experiments/nested/ppo.py", "ppo.py", False, id="nested-code-path"),
        pytest.param(
            "/original machine/work/cleanrl/ppo.py", "cleanrl/ppo.py", "ppo.py", False, id="spaces-in-original-directory"
        ),
        pytest.param("ppo.py", "cleanrl/ppo.py", "ppo.py", False, id="existing-basename-compatible"),
        pytest.param("./ppo.py", "experiments/nested/ppo.py", "ppo.py", False, id="dot-relative-compatible"),
        pytest.param("ppo.py", "cleanrl/ppo.py", "ppo.py", True, id="default-entity-removal-compatible"),
        pytest.param(
            "/original-machine/ppo.py", "archived/agents/train.py", "train.py", False, id="different-logged-script-name"
        ),
    ],
)
def test_reproduce_downloaded_program(
    monkeypatch, tmp_path, capsys, original_program, code_path, expected_filename, remove_entity
):
    recorded_args = ["--env-id", "CartPole-v1", "--seed", "17", "--track", "False"]
    metadata = {
        "program": original_program,
        "codePath": code_path,
        "args": recorded_args + ["--wandb-entity", "original-team"],
    }
    calls = []

    def metadata_response(*, url):
        calls.append(url)
        return SimpleNamespace(json=lambda: metadata)

    monkeypatch.setattr(requests, "get", metadata_response)
    arguments = ["reproduce", "--run", "public/project/runs/run123"]
    if not remove_entity:
        arguments += ["--remove-entity", "false"]
    monkeypatch.setattr(sys, "argv", arguments)
    repo = Path(__file__).resolve().parents[1]
    monkeypatch.syspath_prepend(str(repo))
    namespace = runpy.run_module("cleanrl_utils.reproduce", run_name="__main__")
    assert Path(namespace["__file__"]) == repo / "cleanrl_utils/reproduce.py"
    assert calls == ["https://api.wandb.ai/files/public/project/run123/wandb-metadata.json"]
    lines = capsys.readouterr().out.strip().splitlines()
    download = shlex.split(next(line for line in lines if line.startswith("curl ")))
    assert download[-1] == "https://api.wandb.ai/files/public/project/run123/code/" + code_path

    source = tmp_path / "logged-code" / code_path
    source.parent.mkdir(parents=True)
    source.write_text("import json, sys; print(json.dumps(sys.argv[1:]))\n")
    destination = tmp_path / "reproduction with spaces"
    destination.mkdir()
    download[-1] = source.as_uri()
    downloaded = subprocess.run(download, cwd=destination, capture_output=True, text=True, timeout=10)
    assert downloaded.returncode == 0, downloaded.stderr
    assert (destination / expected_filename).read_bytes() == source.read_bytes()

    environment = dict(os.environ)
    environment["PATH"] = str(Path(sys.executable).parent) + os.pathsep + environment["PATH"]
    executed = subprocess.run(
        ["/bin/sh", "-c", lines[-1]], cwd=destination, env=environment, capture_output=True, text=True, timeout=10
    )
    assert executed.returncode == 0, executed.stderr
    assert json.loads(executed.stdout) == (recorded_args if remove_entity else metadata["args"])
