"""One persistent journal and one controller process per experiment."""

import json
import os
import sqlite3
import time
import uuid
from contextlib import contextmanager
from pathlib import Path

from cloud.nebius.config import TERMINAL, digest, integer, validate, worker_config


@contextmanager
def locked(path):
    # POSIX only, including the controller's supported Linux/macOS hosts.
    import fcntl

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(str(path) + ".lock", os.O_CREAT | os.O_RDWR, 0o600)
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("Another controller owns this journal") from None
        yield
    finally:
        os.close(fd)


class Journal:
    def __init__(self, path):
        self.path = Path(path)
        self.db = sqlite3.connect(path)
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA synchronous=FULL")
        self.db.execute("CREATE TABLE IF NOT EXISTS experiment (config TEXT NOT NULL, experiment TEXT NOT NULL, cap INTEGER)")
        self.db.execute(
            """CREATE TABLE IF NOT EXISTS attempts (
            id TEXT PRIMARY KEY, seed INTEGER, worker TEXT, fingerprint TEXT, idempotency TEXT,
            phase TEXT, operation_id TEXT, job_id TEXT, cancel_operation TEXT,
            submitted REAL, cancel_requested INTEGER DEFAULT 0, detail TEXT DEFAULT '')"""
        )
        os.chmod(path, 0o600)
        self.db.commit()

    def close(self):
        self.db.close()

    def plan(self, config, seeds, cap):
        validate(config)
        integer(cap, 1, 2, "max-in-flight")
        if not 1 <= len(seeds) <= 3 or len(set(seeds)) != len(seeds):
            raise ValueError("Choose 1 to 3 distinct seeds")
        for seed in seeds:
            integer(seed, 1, 2147483647, "seed")
        if self.db.execute("SELECT 1 FROM experiment").fetchone():
            raise ValueError("Journal already contains an experiment; use its existing plan")
        experiment = uuid.uuid4().hex
        with self.db:
            self.db.execute("INSERT INTO experiment VALUES (?, ?, ?)", (json.dumps(config), experiment, cap))
            for seed in seeds:
                self.add(seed, config, experiment)

    def add(self, seed, config, experiment):
        attempt = uuid.uuid4().hex
        worker = worker_config(config, experiment, attempt, seed)
        self.db.execute(
            """INSERT INTO attempts
            (id,seed,worker,fingerprint,idempotency,phase) VALUES (?,?,?,?,?,?)""",
            (attempt, seed, json.dumps(worker), digest(worker), str(uuid.uuid4()), "PLANNED"),
        )
        return attempt

    @property
    def settings(self):
        row = self.db.execute("SELECT * FROM experiment").fetchone()
        if not row:
            raise ValueError("Journal has no plan")
        return json.loads(row["config"]), row["experiment"], row["cap"]

    def rows(self):
        return [dict(row) for row in self.db.execute("SELECT * FROM attempts ORDER BY rowid")]

    def row(self, attempt):
        row = self.db.execute("SELECT * FROM attempts WHERE id=?", (attempt,)).fetchone()
        if row is None:
            raise ValueError("Unknown attempt ID")
        return dict(row)

    def update(self, attempt, **values):
        allowed = {"phase", "operation_id", "job_id", "cancel_operation", "submitted", "cancel_requested", "detail"}
        if not values or set(values) - allowed:
            raise ValueError("Invalid journal update")
        with self.db:
            self.db.execute(
                "UPDATE attempts SET " + ",".join(f"{k}=?" for k in values) + " WHERE id=?", (*values.values(), attempt)
            )

    def reserve(self, attempt):
        _, _, cap = self.settings
        with self.db:
            self.db.execute("BEGIN IMMEDIATE")
            rows = self.rows()
            if self.row(attempt)["phase"] != "PLANNED":
                return False
            if sum(r["phase"] not in TERMINAL | {"PLANNED", "SKIPPED"} for r in rows) >= cap:
                return False
            self.db.execute("UPDATE attempts SET phase='DISPATCHING', submitted=? WHERE id=?", (time.time(), attempt))
        return True

    def retry(self, attempt):
        row = self.row(attempt)
        if row["phase"] not in {"FAILED", "ERROR", "CANCELLED"}:
            raise ValueError("Retry requires a reconciled unsuccessful terminal Job")
        if any(
            r["seed"] == row["seed"] and r["id"] != attempt and r["phase"] not in TERMINAL | {"SKIPPED"} for r in self.rows()
        ):
            raise ValueError("Seed already has a pending or active attempt")
        config, experiment, _ = self.settings
        with self.db:
            return self.add(row["seed"], config, experiment)
