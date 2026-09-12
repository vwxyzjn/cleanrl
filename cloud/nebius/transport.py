"""Nebius SDK transport. Every mutation uses one static-token RPC attempt."""

import json

from cloud.nebius.config import digest, encode


class Nebius:
    def __init__(self, sdk, mutation_factory=None):
        from nebius.api.nebius.ai.v1 import JobServiceClient
        from nebius.sdk import SDK

        self.sdk = sdk
        self.jobs = JobServiceClient(sdk)
        self.mutation_factory = mutation_factory or (
            lambda token: SDK(credentials=token, options=[("grpc.enable_retries", 0)], user_agent_prefix="cleanrl-example/1")
        )

    @staticmethod
    def request(config, row):
        from datetime import timedelta

        from nebius.api.nebius.ai.v1 import CreateJobRequest, JobSpec
        from nebius.api.nebius.common.v1 import ResourceMetadata
        from nebius.api.nebius.compute.v1 import DiskSpec

        worker = json.loads(row["worker"])
        if digest(worker) != row["fingerprint"]:
            raise ValueError("Journal config fingerprint mismatch")
        content = encode(worker)
        if len(content) > 65536:
            raise ValueError("Injected config is too large")
        secrets = {
            "AWS_ACCESS_KEY_ID": config["storage"]["secret_version"],
            "AWS_SECRET_ACCESS_KEY": config["storage"]["secret_version"],
        }
        if config["tracking"]["mode"] == "online":
            secrets["WANDB_API_KEY"] = config["tracking"]["secret_version"]
        env = [JobSpec.EnvironmentVariable(name="CLEANRL_CONFIG_SHA256", value=row["fingerprint"])]
        env += [
            JobSpec.EnvironmentVariable(name=name, mysterybox_secret=JobSpec.MysteryBoxSecretRef(version_id=version))
            for name, version in sorted(secrets.items())
        ]
        spec = JobSpec(
            image=config["image"],
            container_command="/opt/cleanrl/.venv/bin/python",
            args="-m cloud.nebius.worker",
            working_dir="/opt/cleanrl",
            platform="gpu-l40s-a",
            preset="1gpu-8vcpu-32gb",
            subnet_id=config["subnet_id"],
            public_ip=True,
            preemptible=False,
            restart_attempts=0,
            timeout=timedelta(hours=1),
            disk=JobSpec.DiskSpec(type=DiskSpec.DiskType.NETWORK_SSD, size_bytes=250 * 1024**3),
            environment_variables=env,
            injected_files=[JobSpec.FileInjection(container_path="/etc/cleanrl/run.json", content=content)],
        )
        if config.get("registry_secret_version"):
            spec.registry_credentials = JobSpec.RegistryCredentials(
                mysterybox_secret_version=config["registry_secret_version"]
            )
        return CreateJobRequest(
            metadata=ResourceMetadata(
                parent_id=config["project_id"],
                name="cleanrl-" + row["id"],
                labels={"cleanrl-attempt": row["id"], "cleanrl-config": row["fingerprint"]},
            ),
            spec=spec,
        )

    async def mutate(self, method, request, key):
        from nebius.api.nebius.ai.v1 import JobServiceClient

        # Refresh before sending. A static bearer cannot retry an UNAUTHENTICATED RPC.
        token = await self.sdk.get_token(timeout=30)
        async with self.mutation_factory(token.token) as sdk:
            service = JobServiceClient(sdk)
            operation = await getattr(service, method)(
                request, retries=0, timeout=60, auth_timeout=30, metadata=[("x-idempotency-key", key)]
            )
            return {"operation_id": operation.id, "job_id": operation.resource_id}

    async def create(self, config, row):
        return await self.mutate("create", self.request(config, row), row["idempotency"])

    async def cancel(self, job_id, key):
        from nebius.api.nebius.ai.v1 import CancelJobRequest

        return await self.mutate("cancel", CancelJobRequest(id=job_id), key)

    async def operation(self, operation_id):
        from nebius.api.nebius.common.v1 import GetOperationRequest

        op = await self.jobs.operation_service().get(GetOperationRequest(id=operation_id), timeout=30, retries=2)
        return {
            "job_id": op.resource_id,
            "done": op.status() is not None,
            "code": op.status().code.name if op.status() is not None else None,
        }

    @staticmethod
    def snapshot(job):
        spec = job.spec
        return {
            "id": job.metadata.id,
            "parent": job.metadata.parent_id,
            "name": job.metadata.name,
            "labels": dict(job.metadata.labels),
            "state": job.status.state.name,
            # Do not persist free-form service errors: they may echo request data.
            "code": job.status.state_details.code,
            "spec": {
                "image": spec.image,
                "platform": spec.platform,
                "preset": spec.preset,
                "subnet_id": spec.subnet_id,
                "restart_attempts": spec.restart_attempts,
                "preemptible": spec.preemptible,
                "public_ip": spec.public_ip,
                "container_command": spec.container_command,
                "args": spec.args,
                "working_dir": spec.working_dir,
                "disk_size": spec.disk.size_bytes,
                "disk_type": spec.disk.type.name,
                "timeout": spec.timeout.total_seconds(),
                "shm_size": spec.shm_size_bytes,
                "ports": len(spec.ports),
                "volumes": len(spec.volumes),
                "ssh_keys": list(spec.ssh_authorized_keys),
                "files": [f.container_path for f in spec.injected_files],
                "env": sorted(
                    (e.name, e.value, e.mysterybox_secret.secret_id, e.mysterybox_secret.version_id)
                    for e in spec.environment_variables
                ),
                "registry_version": spec.registry_credentials.mysterybox_secret_version,
                "registry_username": spec.registry_credentials.username,
            },
        }

    def owned(self, job, config, row):
        from nebius.api.nebius.ai.v1 import Job

        expected = self.request(config, row)
        view = self.snapshot(Job(metadata=expected.metadata, spec=expected.spec))
        return all(job[key] == view[key] for key in ("parent", "name", "labels", "spec"))

    async def get(self, job_id):
        from nebius.api.nebius.ai.v1 import GetJobRequest

        return self.snapshot(await self.jobs.get(GetJobRequest(id=job_id), timeout=30, retries=2))

    async def candidates(self, config, row):
        from nebius.api.nebius.ai.v1 import ListJobsRequest

        found, token, seen = [], "", set()
        while True:
            page = await self.jobs.list(
                ListJobsRequest(parent_id=config["project_id"], page_size=100, page_token=token), timeout=30, retries=2
            )
            for item in page.items:
                if item.metadata.name == "cleanrl-" + row["id"] or item.metadata.labels.get("cleanrl-attempt") == row["id"]:
                    job = self.snapshot(item)
                    found.append({"id": job["id"], "state": job["state"], "matches": self.owned(job, config, row)})
            token = page.next_page_token
            if not token:
                return found
            if token in seen:
                raise RuntimeError("Repeated pagination token; reconciliation incomplete")
            seen.add(token)
