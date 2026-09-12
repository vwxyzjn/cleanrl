import copy
import json

import pytest

from cloud.nebius.config import TERMINAL, validate
from cloud.nebius.controller import Controller
from cloud.nebius.journal import Journal, locked
from cloud.nebius.transport import Nebius


class Fake:
    def __init__(self):
        self.created, self.cancelled, self.jobs = [], [], {}
        self.lose_ack = False

    async def create(self, config, row):
        self.created.append(row["id"])
        self.jobs[row["id"]] = {"id": row["id"], "state": "RUNNING", "code": "", "owned": True}
        if self.lose_ack:
            raise TimeoutError("unknown")
        return {"operation_id": "op-" + row["id"], "job_id": row["id"]}

    async def get(self, job_id):
        return self.jobs[job_id]

    async def cancel(self, job_id, key):
        self.cancelled.append(job_id)
        self.jobs[job_id]["state"] = "CANCELLING"
        return {"operation_id": "cancel-" + job_id, "job_id": job_id}

    def owned(self, job, config, row):
        return job["owned"]

    async def candidates(self, config, row):
        return [{"id": k, "state": v["state"], "matches": v["owned"]} for k, v in self.jobs.items() if k == row["id"]]


@pytest.mark.asyncio
async def test_lost_ack_never_replayed_after_reopen(journal):
    fake = Fake()
    fake.lose_ack = True
    controller = Controller(journal, fake)
    with pytest.raises(TimeoutError):
        await controller.submit()
    row = journal.rows()[0]
    assert row["phase"] == "SUBMISSION_UNKNOWN"
    reopened = Journal(journal.path)
    try:
        controller = Controller(reopened, fake)
        await controller.submit()
        assert fake.created == [row["id"]]
        found = await controller.reconcile(row["id"])
        assert found[0]["matches"]
        assert reopened.row(row["id"])["job_id"] is None
        await controller.reconcile(row["id"], adopt=row["id"])
        await controller.refresh()
        assert reopened.row(row["id"])["phase"] == "RUNNING"
    finally:
        reopened.close()


@pytest.mark.asyncio
async def test_crash_before_dispatch_keeps_slot(journal):
    row = journal.rows()[0]
    assert journal.reserve(row["id"])
    fake = Fake()
    controller = Controller(journal, fake)
    await controller.refresh()
    await controller.submit()
    assert not fake.created
    assert await controller.reconcile(row["id"]) == []
    with pytest.raises(ValueError):
        journal.retry(row["id"])


@pytest.mark.asyncio
async def test_cap_counts_provisioning_cancel_and_failed_reads(journal):
    fake = Fake()
    controller = Controller(journal, fake)
    await controller.submit()
    row = journal.rows()[0]
    for state in ("PROVISIONING", "STARTING", "IMAGE_PULLING", "RUNNING", "CANCELLING", "DELETING", "UNKNOWN_NEW_STATE"):
        fake.jobs[row["id"]]["state"] = state
        await controller.refresh()
        await controller.submit()
        assert len(fake.created) == 1
    del fake.jobs[row["id"]]
    await controller.refresh()
    await controller.submit()
    assert len(fake.created) == 1
    fake.jobs[row["id"]] = {"id": row["id"], "state": "COMPLETED", "code": "", "owned": True}
    await controller.refresh()
    await controller.submit()
    assert len(fake.created) == 2


@pytest.mark.asyncio
async def test_cancel_once_and_completion_race(journal):
    fake = Fake()
    controller = Controller(journal, fake)
    await controller.submit()
    row = journal.rows()[0]
    await controller.cancel(row["id"])
    await controller.cancel(row["id"])
    assert fake.cancelled == [row["id"]]
    assert journal.row(row["id"])["phase"] not in TERMINAL
    fake.jobs[row["id"]]["state"] = "COMPLETED"
    await controller.cancel(row["id"], repeat=True)
    assert len(fake.cancelled) == 1
    assert journal.row(row["id"])["phase"] == "COMPLETED"


@pytest.mark.asyncio
async def test_refuses_unowned_cancel_and_adoption(journal):
    fake = Fake()
    controller = Controller(journal, fake)
    await controller.submit()
    row = journal.rows()[0]
    fake.jobs[row["id"]]["owned"] = False
    with pytest.raises(ValueError):
        await controller.cancel(row["id"])
    with pytest.raises(ValueError):
        await controller.reconcile(row["id"], adopt=row["id"])
    await controller.refresh()
    assert journal.row(row["id"])["phase"] == "OWNERSHIP_UNKNOWN"
    assert not fake.cancelled


@pytest.mark.asyncio
async def test_deadline_cancel_and_planned_skip(journal):
    fake = Fake()
    controller = Controller(journal, fake)
    await controller.submit()
    row = journal.rows()[0]
    await controller.enforce_deadlines(row["submitted"] + 2701)
    assert len(fake.cancelled) == 1
    await controller.cancel(journal.rows()[1]["id"])
    assert journal.rows()[1]["phase"] == "SKIPPED"


@pytest.mark.asyncio
async def test_explicit_retry_is_fresh_identity(journal):
    fake = Fake()
    controller = Controller(journal, fake)
    await controller.submit()
    old = journal.rows()[0]
    fake.jobs[old["id"]]["state"] = "FAILED"
    await controller.refresh()
    new = journal.row(journal.retry(old["id"]))
    assert new["id"] != old["id"] and new["idempotency"] != old["idempotency"]
    assert json.loads(new["worker"])["prefix"] != json.loads(old["worker"])["prefix"]
    assert new["phase"] == "PLANNED"
    with pytest.raises(ValueError):
        journal.retry(old["id"])


def test_real_file_lock_excludes_second_controller(tmp_path):
    path = tmp_path / "receipt.sqlite"
    with locked(path):
        with pytest.raises(RuntimeError):
            with locked(path):
                pass


@pytest.mark.parametrize(
    "change",
    [
        lambda c: c.update(secret="do-not-accept"),
        lambda c: c.update(image="latest"),
        lambda c: c["training"].update(total_timesteps=0),
        lambda c: c["training"].update(total_timesteps=True),
        lambda c: c["training"].update(total_timesteps=10000000),
        lambda c: c["storage"].update(region="x; echo bad"),
        lambda c: c["tracking"].update(mode="online"),
    ],
)
def test_invalid_config(config, change):
    change(config)
    with pytest.raises(ValueError):
        validate(config)


def test_error_summary_keeps_identifiers_without_secret_message():
    from types import SimpleNamespace

    from cloud.nebius.config import error_summary

    error = Exception("secret-value must never appear")
    error.status = SimpleNamespace(
        code=SimpleNamespace(name="INVALID_ARGUMENT"), request_id="f49fb676-2dac-4395-8b91-16a19ab02cdb"
    )
    summary = error_summary(error)
    assert "INVALID_ARGUMENT" in summary and error.status.request_id in summary
    assert "secret-value" not in summary
    error.status.request_id = "secret-value"
    assert "secret-value" not in error_summary(error)


def test_job_spec_and_ownership(journal):
    from nebius.api.nebius.ai.v1 import Job

    config = journal.settings[0]
    row = journal.rows()[0]
    request = Nebius.request(config, row)
    assert request.spec.restart_attempts == 0
    assert not request.spec.preemptible
    assert request.spec.timeout.total_seconds() == 3600
    assert len(request.spec.injected_files[0].content) < 65536
    assert not request.spec.ports
    job = Job(metadata=request.metadata, spec=request.spec)
    transport = object.__new__(Nebius)
    snapshot = transport.snapshot(job)
    assert transport.owned(snapshot, config, row)
    snapshot["labels"].update({"vmapp-id": "aijob-service-added", "vmapp-image": config["image"]})
    assert transport.owned(snapshot, config, row)
    changed_label = copy.deepcopy(snapshot)
    changed_label["labels"]["cleanrl-config"] = "different"
    assert not transport.owned(changed_label, config, row)
    for env in job.spec.environment_variables:
        if env.mysterybox_secret.version_id:
            env.mysterybox_secret.secret_id = "mbsec-resolved-parent"
    assert transport.owned(transport.snapshot(job), config, row)
    job.spec.environment_variables[1].mysterybox_secret.version_id = "mbsecver-different"
    assert not transport.owned(transport.snapshot(job), config, row)
    changed = copy.deepcopy(snapshot)
    changed["spec"]["image"] = "other-image"
    assert not transport.owned(changed, config, row)
    assert all(e.name != "WANDB_API_KEY" for e in request.spec.environment_variables)


@pytest.mark.asyncio
async def test_retry_wait_uses_latest_attempt(journal):
    from cloud.nebius.submit import latest_rows

    fake = Fake()
    controller = Controller(journal, fake)
    await controller.submit()
    old = journal.rows()[0]
    fake.jobs[old["id"]]["state"] = "FAILED"
    await controller.refresh()
    new = journal.retry(old["id"])
    assert next(r for r in latest_rows(journal) if r["seed"] == old["seed"])["id"] == new


@pytest.mark.asyncio
async def test_lost_cancel_ack_requires_explicit_repeat(journal):
    class Lost(Fake):
        async def cancel(self, job_id, key):
            await super().cancel(job_id, key)
            raise TimeoutError("lost cancel acknowledgment")

    fake = Lost()
    controller = Controller(journal, fake)
    await controller.submit()
    attempt = journal.rows()[0]["id"]
    with pytest.raises(TimeoutError):
        await controller.cancel(attempt)
    await controller.cancel(attempt)
    assert len(fake.cancelled) == 1
    with pytest.raises(TimeoutError):
        await controller.cancel(attempt, repeat=True)
    assert len(fake.cancelled) == 2
