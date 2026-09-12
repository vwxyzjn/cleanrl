"""Closed, non-secret configuration for a deliberately small DQN example."""

import hashlib
import json
import re
from pathlib import Path

TERMINAL = {"COMPLETED", "FAILED", "ERROR", "CANCELLED"}
DEFAULT_TRAINING = {
    "total_timesteps": 20000,
    "learning_starts": 1000,
    "train_frequency": 4,
    "buffer_size": 10000,
    "batch_size": 128,
}


def encode(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def digest(value):
    return hashlib.sha256(encode(value)).hexdigest()


def fields(value, required, optional=()):
    if not isinstance(value, dict) or set(value) - set(required) - set(optional) or set(required) - set(value):
        raise ValueError("Missing or unknown configuration fields; see cartpole.json")


def text(value, pattern, label):
    if not isinstance(value, str) or not re.fullmatch(pattern, value):
        raise ValueError(f"Invalid {label}")
    return value


def integer(value, low, high, label):
    if type(value) is not int or not low <= value <= high:
        raise ValueError(f"{label} must be an integer in [{low}, {high}]")
    return value


def validate(config):
    fields(
        config,
        ("project_id", "subnet_id", "image", "source_revision", "storage", "tracking", "training"),
        ("registry_secret_version",),
    )
    text(config["project_id"], r"project-[a-z0-9]+", "project ID")
    text(config["subnet_id"], r"vpcsubnet-[a-z0-9]+", "subnet ID")
    text(config["image"], r"[a-z0-9][a-z0-9./:_-]*@sha256:[0-9a-f]{64}", "digest-pinned image")
    if len(config["image"]) > 128:
        raise ValueError("Use a shorter image repository (maximum reference length 128 for this example)")
    text(config["source_revision"], r"[0-9a-f]{40}", "CleanRL source revision")
    fields(config["storage"], ("bucket", "region", "secret_version"))
    text(config["storage"]["bucket"], r"[a-z0-9][a-z0-9.-]{1,61}[a-z0-9]", "bucket")
    text(config["storage"]["region"], r"[a-z]+-[a-z]+[0-9]+", "storage region")
    text(config["storage"]["secret_version"], r"mbsecver-[a-z0-9]+", "storage secret version")
    if config.get("registry_secret_version"):
        text(config["registry_secret_version"], r"mbsecver-[a-z0-9]+", "registry secret version")
    fields(config["tracking"], ("mode",), ("project", "entity", "secret_version"))
    tracking = config["tracking"]
    if tracking["mode"] not in ("disabled", "offline", "online"):
        raise ValueError("Tracking mode must be disabled, offline or online")
    if tracking["mode"] != "disabled":
        for name in ("project", "entity"):
            text(tracking.get(name), r"[a-zA-Z0-9][a-zA-Z0-9_.-]{0,63}", f"tracking {name}")
    if tracking["mode"] == "online":
        text(tracking.get("secret_version"), r"mbsecver-[a-z0-9]+", "tracking secret version")
    elif "secret_version" in tracking:
        raise ValueError("Only online tracking accepts a secret reference")
    fields(config["training"], DEFAULT_TRAINING)
    for name, high in (
        ("total_timesteps", 20000),
        ("learning_starts", 10000),
        ("train_frequency", 128),
        ("buffer_size", 10000),
        ("batch_size", 1024),
    ):
        integer(config["training"][name], 1, high, name)
    if not config["training"]["batch_size"] <= config["training"]["buffer_size"]:
        raise ValueError("Batch must fit in the replay buffer")
    if config["training"]["learning_starts"] >= config["training"]["total_timesteps"]:
        raise ValueError("The workload must perform training updates")
    return config


def load(path):
    return validate(json.loads(Path(path).read_text()))


def worker_config(config, experiment, attempt, seed):
    # No secret values or references are needed by the worker's JSON file.
    return {
        "schema": 1,
        "experiment": experiment,
        "attempt": attempt,
        "seed": seed,
        "image": config["image"],
        "source_revision": config["source_revision"],
        "training": config["training"],
        "tracking": {k: v for k, v in config["tracking"].items() if k != "secret_version"},
        "storage": {k: v for k, v in config["storage"].items() if k != "secret_version"},
        "prefix": f"cleanrl/{experiment}/seed-{seed}/{attempt}/",
        "worker_timeout": 600,
    }


def validate_worker(config):
    fields(
        config,
        (
            "schema",
            "experiment",
            "attempt",
            "seed",
            "image",
            "source_revision",
            "training",
            "tracking",
            "storage",
            "prefix",
            "worker_timeout",
        ),
    )
    if config["schema"] != 1:
        raise ValueError("Unsupported worker schema")
    for name in ("experiment", "attempt"):
        text(config[name], r"[0-9a-f]{32}", name)
    integer(config["seed"], 1, 2147483647, "seed")
    integer(config["worker_timeout"], 1, 600, "worker timeout")
    if config["prefix"] != f"cleanrl/{config['experiment']}/seed-{config['seed']}/{config['attempt']}/":
        raise ValueError("Output prefix does not match attempt identity")
    fields(config["storage"], ("bucket", "region"))
    fields(config["tracking"], ("mode",), ("project", "entity"))
    tracking = dict(config["tracking"])
    if tracking["mode"] == "online":
        tracking["secret_version"] = "mbsecver-placeholder"
    validate(
        dict(
            project_id="project-placeholder",
            subnet_id="vpcsubnet-placeholder",
            image=config["image"],
            source_revision=config["source_revision"],
            training=config["training"],
            tracking=tracking,
            storage=dict(config["storage"], secret_version="mbsecver-placeholder"),
        )
    )
    return config
