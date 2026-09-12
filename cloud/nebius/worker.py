"""Run the unmodified DQN script and export its final policy and diagnostics."""

import argparse
import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

from cloud.nebius.artifacts import LocalStore, S3Store, export
from cloud.nebius.config import digest, encode, validate_worker

ROOT = Path(__file__).resolve().parents[2]


def command(config, python, cpu=False):
    args = [
        python,
        "-u",
        str(ROOT / "cleanrl/dqn.py"),
        "--env-id",
        "CartPole-v1",
        "--num-envs",
        "1",
        "--seed",
        str(config["seed"]),
        "--no-cuda" if cpu else "--cuda",
        "--save-model",
    ]
    for key, value in config["training"].items():
        args += ["--" + key.replace("_", "-"), str(value)]
    if config["tracking"]["mode"] != "disabled":
        args += [
            "--track",
            "--wandb-project-name",
            config["tracking"]["project"],
            "--wandb-entity",
            config["tracking"]["entity"],
        ]
    return args


def validate_model(path):
    import gymnasium as gym
    import torch

    from cleanrl.dqn import QNetwork, make_env

    envs = gym.vector.SyncVectorEnv([make_env("CartPole-v1", 1, 0, False, "check")])
    try:
        model = QNetwork(envs)
        state = torch.load(path, map_location="cpu", weights_only=True)
        model.load_state_dict(state, strict=True)
        if not all(torch.isfinite(t).all() for t in state.values()):
            raise ValueError("Checkpoint contains non-finite tensors")
        obs, _ = envs.reset(seed=1)
        with torch.no_grad():
            output = model(torch.tensor(obs))
        if output.shape != (1, 2) or not torch.isfinite(output).all():
            raise ValueError("Checkpoint inference failed")
    finally:
        envs.close()


def child_environment(config):
    env = dict(os.environ)
    # Only the parent exporter requires Object Storage credentials.
    for key in list(env):
        if key.startswith(("AWS_", "NEBIUS_")) or key in {"WANDB_API_KEY", "WANDB_BASE_URL", "WANDB_RESUME"}:
            env.pop(key)
    env.update(
        PYTHONPATH=str(ROOT),
        PYTHONUNBUFFERED="1",
        OMP_NUM_THREADS="1",
        SDL_VIDEODRIVER="dummy",
        WANDB_MODE=config["tracking"]["mode"],
        WANDB_RUN_ID=config["attempt"],
        WANDB_RUN_GROUP=config["experiment"],
        WANDB_INIT_TIMEOUT="30",
        WANDB_SILENT="true",
    )
    if config["tracking"]["mode"] == "online":
        if not os.environ.get("WANDB_API_KEY"):
            raise ValueError("Online tracking requires WANDB_API_KEY")
        env["WANDB_API_KEY"] = os.environ["WANDB_API_KEY"]
    return env


def run(config, store, cpu=False):
    import torch

    validate_worker(config)
    if store.exists(config["prefix"] + "manifest.json") or store.exists(config["prefix"] + "complete.json"):
        raise ValueError("Attempt already exported; use a fresh attempt")
    if not cpu and (not torch.cuda.is_available() or not torch.isfinite(torch.ones(1, device="cuda") * 2).all()):
        raise RuntimeError("A working CUDA device is required; CPU fallback is disabled")
    if not shutil.which("ffmpeg"):
        raise RuntimeError("FFmpeg is required by DQN's save-model evaluation")
    revision_file = ROOT / "cloud/nebius/source-revision.txt"
    if revision_file.exists() and revision_file.read_text().strip() != config["source_revision"]:
        raise ValueError("Image source revision does not match plan")
    metadata = {
        "attempt": config["attempt"],
        "config_sha256": digest(config),
        "seed": config["seed"],
        "image": config["image"],
        "source_revision": config["source_revision"],
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "device": "cpu-test" if cpu else torch.cuda.get_device_name(0),
    }
    print(json.dumps(metadata), flush=True)
    env = child_environment(config)
    env["IMAGEIO_FFMPEG_EXE"] = shutil.which("ffmpeg")
    secrets = [
        os.environ[k].encode() for k in ("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "WANDB_API_KEY") if os.environ.get(k)
    ]
    cancelled = False

    def request_stop(signum, frame):
        nonlocal cancelled
        cancelled = True

    handlers = {s: signal.signal(s, request_stop) for s in (signal.SIGINT, signal.SIGTERM)}
    try:
        with tempfile.TemporaryDirectory(prefix="cleanrl-output-") as work:
            path = Path(work)
            raw = path / "stdout.raw"
            with raw.open("wb") as stream:
                process = subprocess.Popen(
                    command(config, sys.executable, cpu),
                    cwd=work,
                    env=env,
                    stdout=stream,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
                started = time.monotonic()
                progress_stop = threading.Event()

                def progress():
                    while not progress_stop.wait(30):
                        try:
                            data = raw.read_bytes()
                            for secret in secrets:
                                data = data.replace(secret, b"[REDACTED]")
                            store.put(config["prefix"] + "progress/stdout.txt", data[-(1024**2) :])
                            for event in path.glob("runs/*/events.out.tfevents.*"):
                                if event.stat().st_size <= 1024**2:
                                    store.put(config["prefix"] + "progress/" + event.name, event.read_bytes())
                        except Exception:
                            print("Progress export unavailable", flush=True)

                threading.Thread(target=progress, daemon=True).start()
                try:
                    while process.poll() is None:
                        if (
                            cancelled
                            or time.monotonic() - started >= config["worker_timeout"]
                            or raw.stat().st_size > 128 * 1024**2
                        ):
                            cancelled = True
                            os.killpg(process.pid, signal.SIGTERM)
                            try:
                                process.wait(timeout=10)
                            except subprocess.TimeoutExpired:
                                os.killpg(process.pid, signal.SIGKILL)
                                process.wait(timeout=10)
                            break
                        time.sleep(0.2)
                    code = process.wait(timeout=10)
                finally:
                    progress_stop.set()
                    if process.poll() is None:
                        os.killpg(process.pid, signal.SIGKILL)
                        process.wait(timeout=10)
            data = raw.read_bytes()
            for secret in secrets:
                data = data.replace(secret, b"[REDACTED]")
            raw.unlink()
            (path / "stdout.txt").write_bytes(data)
            print(data.decode(errors="replace")[-8000:], flush=True)
            # W&B diagnostic logs/config may contain account information; only export
            # offline run data when explicitly selected, with credentials absent.
            if config["tracking"]["mode"] != "offline":
                shutil.rmtree(path / "wandb", ignore_errors=True)
            else:
                for item in (path / "wandb").rglob("*"):
                    if item.is_symlink():
                        item.unlink()
            models = list(path.glob("runs/*/dqn.cleanrl_model"))
            valid = False
            if len(models) == 1:
                try:
                    validate_model(models[0])
                    valid = True
                except Exception:
                    models[0].unlink()
            success = code == 0 and not cancelled and valid
            metadata.update(training_exit_code=code, cancelled=cancelled, checkpoint_valid=valid)
            (path / "run.json").write_bytes(encode(config))
            export(store, config["prefix"], path, metadata, success)
            return 0 if success else 1
    finally:
        for sig, handler in handlers.items():
            signal.signal(sig, handler)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="/etc/cleanrl/run.json")
    parser.add_argument("--local-output", help="Explicit local test mode; never uses Object Storage")
    parser.add_argument("--cpu-test", action="store_true", help="Allowed only with --local-output")
    args = parser.parse_args()
    if args.cpu_test and not args.local_output:
        parser.error("--cpu-test requires --local-output")
    config = json.loads(Path(args.config).read_text())
    expected = os.environ.get("CLEANRL_CONFIG_SHA256")
    if not args.local_output and (not expected or digest(config) != expected):
        raise ValueError("Injected run configuration does not match receipt")
    if not args.local_output and not all(os.environ.get(k) for k in ("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY")):
        raise ValueError("Storage credentials must be injected explicitly")
    store = LocalStore(args.local_output) if args.local_output else S3Store(**config["storage"])
    return run(config, store, args.cpu_test)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        # SDK/storage errors can contain credentials or signed URLs.
        print(f"Worker failed ({type(exc).__name__}); inspect local configuration and safe logs", file=sys.stderr)
        raise SystemExit(1) from None
