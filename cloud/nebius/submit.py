"""Plan and control a bounded CleanRL experiment on Nebius Serverless Jobs."""

import argparse
import asyncio
import json
import random
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from cloud.nebius.config import TERMINAL, encode, load
from cloud.nebius.controller import Controller
from cloud.nebius.journal import Journal, locked


def summary(journal):
    config, experiment, cap = journal.settings
    print(
        json.dumps(
            {
                "experiment": experiment,
                "image": config["image"],
                "max_in_flight": cap,
                "gpu_per_job": 1,
                "job_timeout_hours": 1,
                "attempts": [
                    {k: r[k] for k in ("id", "seed", "phase", "job_id", "operation_id", "cancel_operation", "detail")}
                    for r in journal.rows()
                ],
            },
            indent=2,
        )
    )


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--journal", required=True, help="Persistent local SQLite receipt; reuse after restart")
    commands = result.add_subparsers(dest="command", required=True)
    p = commands.add_parser("plan", help="Offline only; writes receipts without cloud calls")
    p.add_argument("--config", required=True)
    p.add_argument("--seeds", nargs="+", type=int, default=[1])
    p.add_argument("--max-in-flight", type=int, default=1)
    p = commands.add_parser("export-config", help="Write one worker configuration for local testing")
    p.add_argument("--attempt", required=True)
    p.add_argument("--output", required=True)
    commands.add_parser("show", help="Read local receipts only")
    for name in ("submit", "status", "wait", "reconcile", "cancel"):
        p = commands.add_parser(name)
        p.add_argument("--credentials-file", required=True, help="Dedicated renewable service-account credentials file")
        if name in ("reconcile", "cancel"):
            p.add_argument("--attempt", required=True)
        if name == "reconcile":
            p.add_argument("--adopt-job-id", help="Explicitly adopt exactly one fully matching candidate")
        if name == "cancel":
            p.add_argument("--repeat", action="store_true", help="Explicitly repeat cancellation of a verified owned Job")
        if name == "wait":
            p.add_argument("--poll-seconds", type=int, default=10)
    p = commands.add_parser("retry", help="Plan a fresh attempt for a reconciled unsuccessful seed")
    p.add_argument("--attempt", required=True)
    p = commands.add_parser("logs", help="Best-effort CLI logs for an acknowledged Job")
    p.add_argument("--attempt", required=True)
    p.add_argument("--profile", required=True, help="CLI profile authorized for the same project")
    p = commands.add_parser("results", help="Download and hash-verify a completed attempt")
    p.add_argument("--attempt", required=True)
    p.add_argument("--directory", required=True)
    return result


def latest_rows(journal):
    return list({r["seed"]: r for r in journal.rows()}.values())


async def remote(args, journal):
    from nebius.sdk import SDK

    from cloud.nebius.transport import Nebius

    if args.command == "wait" and not 1 <= args.poll_seconds <= 60:
        raise ValueError("poll-seconds must be between 1 and 60")
    async with SDK(
        credentials_file_name=args.credentials_file,
        options=[("grpc.enable_retries", 0)],
        user_agent_prefix="cleanrl-example/1",
    ) as sdk:
        controller = Controller(journal, Nebius(sdk))
        if args.command == "reconcile":
            await controller.refresh()
            print(json.dumps(await controller.reconcile(args.attempt, args.adopt_job_id), indent=2))
        elif args.command == "cancel":
            await controller.cancel(args.attempt, args.repeat)
            await controller.refresh()
        else:
            try:
                while True:
                    await controller.refresh()
                    if args.command == "submit":
                        await controller.submit()
                    if args.command != "wait":
                        break
                    rows = latest_rows(journal)
                    if any(r["phase"] in {"DISPATCHING", "SUBMISSION_UNKNOWN", "OWNERSHIP_UNKNOWN"} for r in rows):
                        summary(journal)
                        return 3
                    await controller.enforce_deadlines()
                    # A cancelled/timed-out or failed attempt halts automatic seed expansion.
                    if any(r["cancel_requested"] or r["phase"] in {"FAILED", "ERROR", "CANCELLED", "SKIPPED"} for r in rows):
                        if all(r["phase"] in TERMINAL | {"PLANNED", "SKIPPED"} for r in rows):
                            break
                    else:
                        await controller.submit()
                    rows = latest_rows(journal)
                    if all(r["phase"] in TERMINAL | {"SKIPPED"} for r in rows):
                        break
                    if any(
                        r["submitted"] and time.time() - r["submitted"] > 3000 and r["phase"] not in TERMINAL for r in rows
                    ):
                        summary(journal)
                        print("Cleanup is still pending; retain this journal and inspect the recorded IDs", file=sys.stderr)
                        return 3
                    summary(journal)
                    await asyncio.sleep(args.poll_seconds + random.random())
            except asyncio.CancelledError:
                for row in journal.rows():
                    try:
                        await controller.cancel(row["id"])
                    except Exception:
                        pass
                raise
        summary(journal)
        if args.command == "wait":
            return 0 if all(r["phase"] == "COMPLETED" for r in latest_rows(journal)) else 1
        return 0


def main():
    args = parser().parse_args()
    # No credential imports, SDK construction or network calls for these local commands.
    if args.command != "plan" and not Path(args.journal).is_file():
        raise ValueError("Journal does not exist")
    with locked(args.journal):
        journal = Journal(args.journal)
        try:
            if args.command == "plan":
                journal.plan(load(args.config), args.seeds, args.max_in_flight)
                summary(journal)
            elif args.command == "show":
                summary(journal)
            elif args.command == "export-config":
                row = journal.row(args.attempt)
                with Path(args.output).open("xb") as output:
                    output.write(encode(json.loads(row["worker"])))
            elif args.command == "retry":
                print(journal.retry(args.attempt))
            elif args.command == "logs":
                row = journal.row(args.attempt)
                if not row["job_id"] or not row["submitted"]:
                    raise ValueError("Logs require an acknowledged Job")
                since = datetime.fromtimestamp(row["submitted"], timezone.utc).isoformat()
                subprocess.run(
                    [
                        "nebius",
                        "--profile",
                        args.profile,
                        "ai",
                        "job",
                        "logs",
                        row["job_id"],
                        "--since",
                        since,
                        "--timestamps",
                    ],
                    check=True,
                    timeout=60,
                )
            elif args.command == "results":
                from cloud.nebius.artifacts import S3Store, download

                row = journal.row(args.attempt)
                if row["phase"] != "COMPLETED":
                    raise ValueError("Refresh status first; results require Job COMPLETED")
                config = json.loads(row["worker"])
                store = S3Store(**config["storage"])
                manifest = download(
                    store, config["prefix"], args.directory, {"attempt": row["id"], "config_sha256": row["fingerprint"]}
                )
                print(
                    json.dumps(
                        {
                            "hashes_verified": True,
                            "directory": args.directory,
                            "files": len(manifest["files"]),
                            "next": "Reload the policy using verify_model.py in the worker environment",
                        }
                    )
                )
            else:
                return asyncio.run(remote(args, journal))
        finally:
            journal.close()
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("Interrupted; retain the journal and reconcile before any new submission", file=sys.stderr)
        raise SystemExit(130) from None
    except (ValueError, RuntimeError) as exc:
        # These are local validation errors with static messages, not raw SDK diagnostics.
        print(str(exc), file=sys.stderr)
        raise SystemExit(2) from None
    except Exception as exc:
        print(f"Command failed ({type(exc).__name__}); inspect the receipt with show/status", file=sys.stderr)
        raise SystemExit(1) from None
