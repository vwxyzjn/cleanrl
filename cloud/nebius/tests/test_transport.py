"""Exercise the released SDK over local gRPC, including lost acknowledgments."""

import asyncio
from contextlib import asynccontextmanager

import grpc
import pytest
from nebius.aio.channel import INSECURE
from nebius.api.nebius.ai.v1 import Job, JobServiceClient, JobStatus, ListJobsResponse
from nebius.api.nebius.common.v1 import Operation, OperationServiceClient
from nebius.base.resolver import Constant
from nebius.sdk import SDK

from cloud.nebius.controller import Controller
from cloud.nebius.transport import Nebius


def add_service(server, client, implementation):
    descriptor = client.get_descriptor()
    registry = client.__registry__
    handlers = {}
    for method in descriptor.methods:
        function = getattr(implementation, method.name, None)
        if function:
            handlers[method.name] = grpc.unary_unary_rpc_method_handler(
                function,
                request_deserializer=registry.message_class(method.input_type.full_name).FromString,
                response_serializer=registry.message_class(method.output_type.full_name).SerializeToString,
            )
    server.add_generic_rpc_handlers((grpc.method_handlers_generic_handler(descriptor.full_name, handlers),))


class Service:
    def __init__(self, failure=None):
        self.failure, self.creates, self.jobs, self.tokens = failure, [], [], []
        self.cancel_count = 0
        self.missing_resource_id = False

    async def Create(self, request, context):
        self.creates.append(dict(context.invocation_metadata()))
        job = Job(metadata=request.metadata, spec=request.spec, status=JobStatus(state=JobStatus.State.RUNNING))
        job.metadata.id = "aijob-local"
        self.jobs.append(job)
        if self.failure:
            await context.abort(self.failure, "simulated lost acknowledgment")
        return Operation(id="operation-local", resource_id="" if self.missing_resource_id else job.metadata.id)

    async def Get(self, request, context):
        return next(j for j in self.jobs if j.metadata.id == request.id)

    async def List(self, request, context):
        self.tokens.append(request.page_token)
        index = int(request.page_token or "0")
        following = str(index + 1) if index + 1 < len(self.jobs) else ""
        return ListJobsResponse(items=self.jobs[index : index + 1], next_page_token=following)

    async def Cancel(self, request, context):
        self.cancel_count += 1
        return Operation(id="operation-cancel", resource_id=request.id)


@asynccontextmanager
async def transport(service):
    server = grpc.aio.server()
    port = server.add_insecure_port("127.0.0.1:0")
    add_service(server, JobServiceClient, service)

    class Operations:
        async def Get(self, request, context):
            return Operation(id=request.id, resource_id="aijob-local")

    add_service(server, OperationServiceClient, Operations())
    await server.start()

    def factory(token):
        return SDK(
            resolver=Constant(f"127.0.0.1:{port}"),
            credentials=token,
            options=[(INSECURE, True), ("grpc.enable_retries", 0)],
            user_agent_prefix="cleanrl-local-test/1",
        )

    try:
        async with factory("local-test-token") as sdk:
            yield Nebius(sdk, mutation_factory=factory)
    finally:
        await server.stop(0)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "failure",
    [
        grpc.StatusCode.UNAVAILABLE,
        grpc.StatusCode.DEADLINE_EXCEEDED,
        grpc.StatusCode.UNAUTHENTICATED,
        grpc.StatusCode.INVALID_ARGUMENT,
    ],
)
async def test_one_wire_create_even_after_error_and_controller_restart(journal, failure):
    service = Service(failure)
    async with transport(service) as adapter:
        controller = Controller(journal, adapter)
        with pytest.raises(Exception):
            await controller.submit()
        await asyncio.sleep(0.05)
        assert len(service.creates) == 1
        row = journal.rows()[0]
        assert row["phase"] == "SUBMISSION_UNKNOWN"
        assert service.creates[0]["x-idempotency-key"] == row["idempotency"]
        assert service.creates[0]["authorization"] == "Bearer local-test-token"
        restarted = Controller(journal, adapter)
        await restarted.submit()
        assert len(service.creates) == 1
        candidates = await restarted.reconcile(row["id"])
        assert candidates == [{"id": "aijob-local", "state": "RUNNING", "matches": True}]
        await restarted.reconcile(row["id"], adopt="aijob-local")
        await restarted.refresh()
        assert journal.row(row["id"])["phase"] == "RUNNING"


@pytest.mark.asyncio
async def test_real_sdk_ack_cancel_and_multipage_duplicates(journal):
    service = Service()
    async with transport(service) as adapter:
        controller = Controller(journal, adapter)
        await controller.submit()
        row = journal.rows()[0]
        assert row["job_id"] == "aijob-local" and row["operation_id"] == "operation-local"
        await controller.refresh()
        assert journal.row(row["id"])["phase"] == "RUNNING"
        await controller.cancel(row["id"])
        await controller.cancel(row["id"])
        assert service.cancel_count == 1
        duplicate = Job.FromString(service.jobs[0].SerializeToString())
        duplicate.metadata.id = "aijob-duplicate"
        service.jobs.append(duplicate)
        candidates = await controller.reconcile(row["id"])
        assert len(candidates) == 2 and all(c["matches"] for c in candidates)
        assert service.tokens[-2:] == ["", "1"]
        with pytest.raises(ValueError):
            await controller.reconcile(row["id"], adopt="aijob-local")


@pytest.mark.asyncio
async def test_acknowledged_operation_recovers_missing_job_id(journal):
    service = Service()
    service.missing_resource_id = True
    async with transport(service) as adapter:
        controller = Controller(journal, adapter)
        await controller.submit()
        row = journal.rows()[0]
        assert row["operation_id"] == "operation-local" and not row["job_id"]
        operation = await adapter.operation(row["operation_id"])
        assert operation == {"job_id": "aijob-local", "done": False, "code": None}
        await controller.refresh()
        assert journal.row(row["id"])["job_id"] == "aijob-local"
        assert journal.row(row["id"])["phase"] == "RUNNING"
