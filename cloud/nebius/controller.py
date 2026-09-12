"""Lifecycle orchestration; unknown submissions retain their concurrency slot."""

import time
import uuid

from cloud.nebius.config import TERMINAL, error_summary


class Controller:
    def __init__(self, journal, transport):
        self.journal = journal
        self.transport = transport

    async def refresh(self):
        config, _, _ = self.journal.settings
        for row in self.journal.rows():
            if row["phase"] in {"PLANNED", "SKIPPED"}:
                continue
            try:
                if not row["job_id"] and row["operation_id"]:
                    operation = await self.transport.operation(row["operation_id"])
                    if operation["job_id"]:
                        self.journal.update(row["id"], job_id=operation["job_id"])
                        row = self.journal.row(row["id"])
                if not row["job_id"]:
                    self.journal.update(row["id"], phase="SUBMISSION_UNKNOWN", detail="Reconcile before further action")
                    continue
                job = await self.transport.get(row["job_id"])
                if not self.transport.owned(job, config, row):
                    self.journal.update(row["id"], phase="OWNERSHIP_UNKNOWN", detail="Job does not match receipt")
                    continue
                state = job["state"]
                if state not in TERMINAL | {"PROVISIONING", "STARTING", "IMAGE_PULLING", "RUNNING", "CANCELLING", "DELETING"}:
                    state = "STATE_UNSPECIFIED"
                code = job["code"]
                safe_code = code if code.isalnum() and len(code) <= 64 else ""
                self.journal.update(row["id"], phase=state, detail=safe_code)
            except Exception as exc:
                # Authentication, not-found and network failures are not evidence of termination.
                self.journal.update(row["id"], detail=f"Observation failed ({type(exc).__name__}); state retained")

    async def submit(self):
        config, _, _ = self.journal.settings
        for row in self.journal.rows():
            if not self.journal.reserve(row["id"]):
                continue
            try:
                result = await self.transport.create(config, row)
                self.journal.update(row["id"], phase="SUBMITTED", operation_id=result["operation_id"], job_id=result["job_id"])
            except BaseException as exc:
                self.journal.update(
                    row["id"],
                    phase="SUBMISSION_UNKNOWN",
                    detail=f"Create outcome unknown; never replay ({error_summary(exc)})",
                )
                raise

    async def cancel(self, attempt, repeat=False):
        row = self.journal.row(attempt)
        if row["phase"] in TERMINAL | {"SKIPPED"}:
            return
        if row["phase"] == "PLANNED":
            self.journal.update(attempt, phase="SKIPPED")
            return
        config, _, _ = self.journal.settings
        if not row["job_id"]:
            raise ValueError("No acknowledged Job ID; reconcile before cancellation")
        job = await self.transport.get(row["job_id"])
        if not self.transport.owned(job, config, row):
            raise ValueError("Refusing cancellation: Job does not match receipt")
        if job["state"] in TERMINAL:
            self.journal.update(attempt, phase=job["state"])
            return
        if row["cancel_requested"] and not repeat:
            return
        self.journal.update(attempt, cancel_requested=1, phase="CANCEL_REQUESTED")
        try:
            key = str(uuid.uuid5(uuid.UUID(row["idempotency"]), "cancel"))
            result = await self.transport.cancel(row["job_id"], key)
            self.journal.update(attempt, cancel_operation=result["operation_id"])
        except BaseException:
            self.journal.update(attempt, detail="Cancel acknowledgment unknown; inspect Job before explicit repeat")
            raise

    async def reconcile(self, attempt, adopt=None):
        row = self.journal.row(attempt)
        if row["phase"] in {"PLANNED", "SKIPPED"}:
            raise ValueError("This attempt has never been dispatched")
        config, _, _ = self.journal.settings
        candidates = await self.transport.candidates(config, row)
        if adopt:
            if len(candidates) != 1 or candidates[0]["id"] != adopt or not candidates[0]["matches"]:
                raise ValueError("Adoption requires exactly one matching candidate; resolve all other candidates first")
            job = await self.transport.get(adopt)
            if not self.transport.owned(job, config, row):
                raise ValueError("Candidate changed or is not owned")
            self.journal.update(attempt, job_id=adopt, phase="SUBMITTED", detail="Explicitly adopted after reconciliation")
        return candidates

    async def enforce_deadlines(self, now=None):
        now = time.time() if now is None else now
        for row in self.journal.rows():
            if row["submitted"] and now - row["submitted"] >= 2700 and row["phase"] not in TERMINAL:
                await self.cancel(row["id"])
