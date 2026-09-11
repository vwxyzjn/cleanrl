import json
import runpy
import shlex
import subprocess
import sys
from unittest.mock import Mock

import pytest
import requests


@pytest.mark.parametrize("program", ["/fsx/costa/cleanrl/cleanrl/ppo_atari.py", "cleanrl/ppo_atari.py"])
@pytest.mark.parametrize("code_path", ["cleanrl/ppo_atari.py", "ppo_atari.py"])
def test_reproduce_runs_downloaded_script(tmp_path, monkeypatch, capsys, program, code_path):
    metadata = {
        "program": program,
        "codePath": code_path,
        "args": ["--track", "--env-id", "BreakoutNoFrameskip-v4", "--seed", "3"],
    }
    response = Mock()
    response.json.return_value = metadata
    get = Mock(return_value=response)
    monkeypatch.setattr(requests, "get", get)
    monkeypatch.setattr(sys, "argv", ["reproduce", "--run", "openrlbenchmark/cleanrl/runs/c1y1qnz4"])

    runpy.run_module("cleanrl_utils.reproduce", run_name="__main__")

    output = capsys.readouterr().out
    get.assert_called_once_with(url="https://api.wandb.ai/files/openrlbenchmark/cleanrl/c1y1qnz4/wandb-metadata.json")
    assert f"curl -OL https://api.wandb.ai/files/openrlbenchmark/cleanrl/c1y1qnz4/code/{code_path}" in output
    command = shlex.split(output.strip().splitlines()[-1])
    assert command[0] == "python"
    # curl -O saves the artifact under its basename in the current directory.
    (tmp_path / "ppo_atari.py").write_text("import json, sys\nprint(json.dumps(sys.argv[1:]))\n")
    result = subprocess.run([sys.executable, *command[1:]], cwd=tmp_path, capture_output=True, text=True, check=True)
    assert json.loads(result.stdout) == metadata["args"]
