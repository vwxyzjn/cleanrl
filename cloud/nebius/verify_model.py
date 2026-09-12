"""Reload a downloaded DQN final-policy checkpoint in the worker environment."""

import argparse
import json
from pathlib import Path

from cloud.nebius.artifacts import sha256
from cloud.nebius.worker import validate_model

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory")
    args = parser.parse_args()
    root = Path(args.directory)
    manifest = json.loads((root / "manifest.json").read_text())
    models = list(root.glob("runs/*/dqn.cleanrl_model"))
    if len(models) != 1 or not manifest.get("success"):
        raise SystemExit("Expected one checkpoint from a successful manifest")
    item = next(i for i in manifest["files"] if i["path"] == models[0].relative_to(root).as_posix())
    if sha256(models[0].read_bytes()) != item["sha256"]:
        raise SystemExit("Checkpoint checksum mismatch")
    validate_model(models[0])
    print("Checkpoint hash, parameter shapes and finite inference output verified")
