# CleanRL on Nebius Serverless Jobs

Run the existing DQN CartPole example in a container on one regular GPU per seed.
The optional controller plans experiments locally, records submission identity before
sending a request, and exports final policy weights and diagnostics to Nebius Object
Storage. CleanRL's algorithms, AWS launcher and base dependencies are unchanged.

This experimental example has offline SDK tests, a local Linux CPU container smoke
test and live validation on an L40S in `eu-north1` on September 12, 2026. Private
image pulls, secret injection, CUDA training, Object Storage exports, offline
tracking and cancellation were exercised. See the validation scope below before
using it in another environment.

CartPole environments run on CPU; the network updates use CUDA. This small workload
checks the execution path and is not a GPU throughput or cost-efficiency benchmark.
The final `.cleanrl_model` contains policy weights, **not resumable training state**.
An interrupted seed must be rerun from the beginning in a fresh attempt.

## Requirements

- Linux or macOS for the controller, Python 3.12 and [uv](https://docs.astral.sh/uv/).
- Docker with Linux amd64 build support. An ARM laptop can use local emulation.
- A Nebius project, subnet, and available `gpu-l40s-a` / `1gpu-8vcpu-32gb` quota.
- A dedicated service account with permissions for Jobs and referenced resources.
  Follow [Jobs prerequisites](https://docs.nebius.com/serverless/jobs/manage) and
  [service-account authentication](https://docs.nebius.com/iam/service-accounts/authentication).
- An existing Nebius Object Storage bucket and access keys stored in a versioned
  SecretStash secret, with exactly `AWS_ACCESS_KEY_ID` and `AWS_SECRET_ACCESS_KEY`
  payload keys, using `string_value` entries rather than `binary_value`. The worker
  identity needs read/write access to this bucket.
- A registry reachable from Serverless. Private pulls may need a separate secret
  version with `REGISTRY_USERNAME` and `REGISTRY_PASSWORD` string payload keys.

The controller only uses Serverless Jobs APIs. It does not manage Compute VMs,
Endpoints, Token Factory or AI Studio. No resources are provisioned by `plan`.
Bucket, secrets, registry and IAM setup are external prerequisites.

## Install and build

Run all commands from the CleanRL repository root:

```sh
uv venv cloud/nebius/.venv-controller --python 3.12
uv pip sync --python cloud/nebius/.venv-controller/bin/python \
  cloud/nebius/requirements-controller.txt
CONTROLLER=cloud/nebius/.venv-controller/bin/python

CLEANRL_REVISION=$(git rev-parse HEAD)
docker build --platform linux/amd64 \
  --build-arg CLEANRL_REVISION="$CLEANRL_REVISION" \
  -f cloud/nebius/Dockerfile -t cleanrl-nebius:local .
```

The image installs Python 3.10, CleanRL's frozen dependency lock and the existing
`cloud` extra for its S3 client. It includes FFmpeg because `dqn.py --save-model`
performs video evaluation even when training video capture is off. No packages or
source are downloaded at Job startup. The Docker build context excludes credentials,
local receipts, training output and Git history.

Build/push the image to your registry, then resolve the **registry manifest digest**.
Use `registry/path@sha256:...`, not a mutable tag or a local Docker image ID. Keep the
full reference at most 128 characters for this example. Put the same
`CLEANRL_REVISION` in your configuration; the worker checks it against the image.
Pinning source, dependencies and seeds does not guarantee identical results across
GPU architectures. Validate the Torch CUDA 12.1 runtime against the selected GPU host.

## Configure and plan

Copy `cloud/nebius/cartpole.json` to `cloud/nebius/config.json`. Replace the example
project/subnet/bucket/secret IDs, image digest and source revision. Do not add secret
values. An optional top-level `registry_secret_version` sets private-registry auth.

The default is one seed, 20,000 steps, one environment, and tracking disabled.
At most three distinct seeds and two in-flight Jobs are allowed. Each Job requests
one L40S, 8 vCPUs, 32 GiB RAM and a 250 GiB ephemeral disk. It has no public ports;
`public_ip` is enabled for outbound registry, storage and optional W&B access.

```sh
"$CONTROLLER" -m cloud.nebius.submit --journal cloud/nebius/run.sqlite plan \
  --config cloud/nebius/config.json --seeds 1 --max-in-flight 1
"$CONTROLLER" -m cloud.nebius.submit --journal cloud/nebius/run.sqlite show
```

`plan` validates the configuration and writes immutable per-seed receipts without
reading credentials or contacting Nebius. It prints the attempt ID used by later
commands. The journal records the image, configuration hash, idempotency key,
operation/Job IDs and concurrency slots. Keep it on durable local storage. Only one
controller process may use it; the file lock is for one host, not a shared network
filesystem or multiple schedulers. Do not delete or replace it while Jobs may exist.

## Submit and inspect

Keep the service-account credentials file outside the repository. It is used by the
controller and is never injected into the worker:

```sh
CREDENTIALS_FILE=/absolute/path/to/cleanrl-service-account.json
"$CONTROLLER" -m cloud.nebius.submit --journal cloud/nebius/run.sqlite validate \
  --credentials-file "$CREDENTIALS_FILE"
"$CONTROLLER" -m cloud.nebius.submit --journal cloud/nebius/run.sqlite submit \
  --credentials-file "$CREDENTIALS_FILE"
"$CONTROLLER" -m cloud.nebius.submit --journal cloud/nebius/run.sqlite wait \
  --credentials-file "$CREDENTIALS_FILE"
```

`validate` calls the live API with `dry_run=true` for planned attempts. It checks
the image and secret configuration without allocating a Job or changing receipts.
It is optional and requires credentials, unlike the offline `plan` command.
A successful dry run does not guarantee later capacity or runtime success.

`submit` dispatches only available slots; `wait` polls and dispatches remaining
planned seeds within the cap. Provisioning, image pulling, unknown submission and
cancellation all retain their slots. The controller refreshes its token before a
mutation, then uses a static bearer with SDK and gRPC retries disabled. It never
replays Create automatically, including after an authentication error.

The Job timeout is one hour (the service minimum), with zero restart attempts and
no preemptible execution. The worker limits the training/evaluation process to ten
minutes. Export and shutdown take additional time. A 45-minute controller deadline
requests cancellation; after a further five minutes, `wait` reports cleanup pending
and exits. Neither limit is a hard billing cap. If the controller is lost, restart
it with the same journal; unattended operation needs a separate watchdog.

`status` observes without submitting more seeds. Best-effort live logs use an
explicit CLI profile authorized for the same project:

```sh
ATTEMPT_ID=replace-with-attempt-id-from-plan
"$CONTROLLER" -m cloud.nebius.submit --journal cloud/nebius/run.sqlite status \
  --credentials-file "$CREDENTIALS_FILE"
"$CONTROLLER" -m cloud.nebius.submit --journal cloud/nebius/run.sqlite logs \
  --attempt "$ATTEMPT_ID" --profile your-nebius-cli-profile
```

Live logs require the Nebius CLI (`0.12.265` checked); the SDK has no Job log-stream
method. The worker also exports periodic stdout/TensorBoard snapshots when storage
is available. Those snapshots may be incomplete and are not completion markers.

## Download and verify results

The worker stages outputs locally and uploads to
`cleanrl/<experiment>/seed-<seed>/<attempt>/`. It uploads files before `manifest.json`
and writes `complete.json` last, only after a successful training/evaluation exit,
a valid checkpoint and successful object read-back checks. Failed attempts can have
partial diagnostic manifests, but never receive a completion marker. Existing
manifests cannot be overwritten by restarting a worker with the same attempt.

After `status` reports `COMPLETED`, use separate read-authorized Object Storage
credentials (for example an explicitly selected AWS credentials profile). The client
sets the Nebius regional S3 endpoint; it does not send outputs to AWS:

```sh
AWS_PROFILE=cleanrl-results "$CONTROLLER" -m cloud.nebius.submit \
  --journal cloud/nebius/run.sqlite results --attempt "$ATTEMPT_ID" \
  --directory cloud/nebius/outputs/verified

docker run --rm --platform linux/amd64 --network none \
  --entrypoint /opt/cleanrl/.venv/bin/python \
  --mount type=bind,src="$PWD/cloud/nebius/outputs/verified",dst=/results,readonly \
  cleanrl-nebius:local -m cloud.nebius.verify_model /results
```

`results` verifies attempt identity, manifest hash, every file size/hash and safe
paths. It requires a new destination directory; a failed download remains incomplete.
The separate model check loads only tensors, validates expected parameter shapes
and checks finite inference output. A Job status alone is not an artifact check.

Outputs include the final `runs/.../dqn.cleanrl_model`, TensorBoard events, evaluation
videos, safe stdout and the exact run configuration. TensorBoard logs episodic
returns/lengths, TD loss, estimated Q values and training steps per second; evaluation
returns describe the final policy. A short smoke run need not solve CartPole.
Limits are 128 MiB per file and 512 MiB per final bundle.

## Tracking credentials

Tracking is optional and independent of durable checkpoints:

```json
{"mode": "online", "project": "cleanrl-nebius", "entity": "your-team",
 "secret_version": "mbsecver-replace"}
```

Use that value for `tracking`, with a string secret payload key `WANDB_API_KEY`. The worker
passes only that tracking credential to the training child, not its S3 keys or the
controller's IAM credential. Each attempt has a distinct W&B run ID and the
experiment ID is the W&B group. No `.netrc` lookup or secret-bearing Docker script
is generated by this controller.

For disconnected tracking, choose `mode: offline`, `project` and `entity`, with no
secret reference. Offline W&B files are exported for later manual synchronization.
An online tracking initialization failure fails the attempt; there is no silent
change to offline mode. W&B's initialization timeout is 30 seconds; the process
wall-clock limit also bounds a hung tracker. W&B resume does not restore DQN state.

## Cancellation and ambiguous submission

```sh
"$CONTROLLER" -m cloud.nebius.submit --journal cloud/nebius/run.sqlite cancel \
  --attempt "$ATTEMPT_ID" --credentials-file "$CREDENTIALS_FILE"
```

Cancellation verifies the Job against its receipt. A planned seed is skipped locally.
For a submitted Job, Cancel acknowledgment is not proof of termination: rerun `status`
until terminal. If completion wins the race, the Job remains `COMPLETED`. An
interrupted or timed-out cancel is not automatically repeated; inspect status before
using explicit `cancel --repeat`. SIGINT during `wait` attempts cancellation of owned
Jobs and skips planned seeds, but a killed process cannot guarantee cleanup.

If Create times out or its acknowledgment is lost, the receipt becomes
`SUBMISSION_UNKNOWN`. Do **not** rerun a bare Create command or discard the journal:

```sh
"$CONTROLLER" -m cloud.nebius.submit --journal cloud/nebius/run.sqlite reconcile \
  --attempt "$ATTEMPT_ID" --credentials-file "$CREDENTIALS_FILE"
```

Known operation IDs are checked first. Otherwise reconciliation reads every Job
list page and reports same-name/attempt candidates with matching-spec evidence.
Names are not unique. Zero candidates is not proof that nothing was created;
multiple candidates require manual investigation and owned-resource cleanup. The
controller never adopts by name. To adopt one fully matching candidate explicitly,
repeat `reconcile` with `--adopt-job-id <exact-id>`. Changed/ambiguous candidates are
refused. Normal reads never request secret payload views.

After an acknowledged Job is `FAILED`, `ERROR` or `CANCELLED`, `retry --attempt ...`
plans a new attempt for that seed, with a new ID, output prefix and tracking ID.
Then submit it explicitly. Training restarts from step zero. Correcting image or
other immutable configuration requires a new plan after reconciling old resources.

The example never deletes Jobs, VMs, disks, shared buckets or secrets. After verifying
results and termination, remove retained Job records through the console/CLI and
independently verify cleanup. Retain output under your bucket's lifecycle policy.
See [Serverless lifecycle](https://docs.nebius.com/serverless/lifecycle).

## Local tests and live validation

Offline controller tests use the released SDK against a local gRPC fake (including
lost acknowledgments, authentication failures and duplicate pagination), plus Moto
S3 tests and artifact corruption checks. They require no cloud credentials:

```sh
uv pip sync --python "$CONTROLLER" cloud/nebius/requirements-test.txt
"$CONTROLLER" -m pytest -c cloud/nebius/pytest.ini cloud/nebius/tests
```

To smoke-test the worker on CPU, export one planned worker config using
`export-config --attempt ... --output /tmp/cleanrl-run.json`. For a short check,
change its `training` to 512 steps, learning starts 64, batch 32 and buffer 1024.
Use a fresh local output directory for each attempt:

```sh
mkdir -p /tmp/cleanrl-smoke-output
docker run --rm --platform linux/amd64 --network none --cpus 2 --memory 4g \
  --mount type=bind,src=/tmp/cleanrl-run.json,dst=/etc/cleanrl/run.json,readonly \
  --mount type=bind,src=/tmp/cleanrl-smoke-output,dst=/output \
  cleanrl-nebius:local --local-output /output --cpu-test
```

`--cpu-test` requires explicit local output and is never set by the Job controller.
Without it, a missing CUDA device fails before training. The image smoke checks
FFmpeg evaluation and policy export; it cannot validate the Nebius runtime.

Live validation on September 12, 2026 used six single-GPU Jobs with the frozen
Python 3.10 / Torch 2.4.1+cu121 worker built from CleanRL revision
`0652c16c2289d20c5caaaf358887b774f782b45c` plus this example. The controller includes
the compatibility fixes found during those runs:

| Check | Observed result |
|---|---|
| Tracking disabled | 20,000 DQN steps on NVIDIA L40S; ten evaluation episodes; verified export and independent checkpoint reload |
| Offline W&B | Two 1,024-step seeds completed, with 19 verified output files each, including offline run data |
| Concurrency and reconnect | Second seed stayed planned until the first completed under a one-Job cap; a new controller process resumed the first experiment from its receipt |
| Private registry and secrets | Digest-pinned private image pulled; string secret payloads delivered; known temporary credentials absent from checked logs and downloaded bundles |
| Running cancellation | Job reached CANCELLED; worker recorded SIGTERM and a partial manifest without a success marker |
| Worker deadline | A one-second test deadline stopped the child; Job failed and only a partial manifest was exported |
| Invalid storage credentials | Worker failed its initial storage check; no manifest or success marker |
| Corrupt output | Independent downloader rejected a corrupted completion marker in live Object Storage; the original marker was restored |
| API preflight | Binary secret entries and a missing image were rejected before allocation |
| Cleanup | All six Job records, test storage/registry/secrets and temporary keys removed; a final paginated listing found no validation Jobs |

The API fills in parent secret IDs for version references and adds its own labels.
Ownership checks normalize the resolved reference and require the CleanRL identity
labels and workload fields, allowing additional server labels.

Online W&B authentication, a storage outage after training starts, the service's
full one-hour timeout, cancellation during provisioning and a cancellation/completion
race were not exercised live. Offline tests cover several of these failure paths.
The live seed test used a one-Job cap; simultaneous two-Job dispatch within one
experiment remains unvalidated. Other regions, GPU types, least-privilege IAM policies
and performance comparisons also require their own validation.

There is no multi-node, distributed training, automatic retry, full-state resume,
Endpoint serving or cross-host scheduler support in this example.
