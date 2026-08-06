from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import pytest

from agentic_rag.domain.models import JobStatus
from agentic_rag.ingestion.state import IngestionJobClaim
from agentic_rag.ingestion.worker import IngestionWorker
from agentic_rag.persistence.redis_queue import StreamMessage
from agentic_rag.persistence.repositories import LeaseLost


class JobStore:
    def __init__(self, *, attempt_count: int = 0) -> None:
        self.status = JobStatus.QUEUED
        self.attempt_count = attempt_count
        self.owner: str | None = None
        self.heartbeats = 0

    async def claim(
        self, job_id: str, owner: str, lease_seconds: int
    ) -> IngestionJobClaim | None:
        if self.status is not JobStatus.QUEUED:
            return None
        self.status = JobStatus.RUNNING
        self.owner = owner
        return IngestionJobClaim(
            job_id=job_id,
            user_id="user-1",
            document_id="doc-1",
            document_version_id="version-1",
            owner=owner,
            claim_generation=self.attempt_count,
        )

    async def heartbeat(self, claim: IngestionJobClaim, lease_seconds: int) -> None:
        if self.owner != claim.owner or self.status is not JobStatus.RUNNING:
            raise LeaseLost(claim.job_id)
        self.heartbeats += 1

    async def get_status(self, job_id: str) -> JobStatus | None:
        return self.status

    async def record_failure(
        self, claim: IngestionJobClaim, error_code: str, *, max_attempts: int
    ) -> JobStatus:
        self.attempt_count += 1
        self.owner = None
        self.status = (
            JobStatus.FAILED if self.attempt_count >= max_attempts else JobStatus.QUEUED
        )
        return self.status


class Broker:
    def __init__(self) -> None:
        self.acked: list[str] = []
        self.dead: list[tuple[str, str]] = []

    async def consume(self, *args: Any, **kwargs: Any) -> list[StreamMessage]:
        return []

    async def publish(self, *args: Any, **kwargs: Any) -> str:
        return "published-1"

    async def reclaim(self, *args: Any, **kwargs: Any) -> list[StreamMessage]:
        return []

    async def ack(self, stream: str, group: str, message_id: str) -> None:
        self.acked.append(message_id)

    async def dead_letter(
        self, dead_stream: str, message: StreamMessage, reason: str
    ) -> None:
        self.dead.append((message.id, reason))


class Graph:
    def __init__(self, outcome: BaseException | None = None) -> None:
        self.outcome = outcome
        self.claims: list[IngestionJobClaim] = []

    async def run(self, claim: IngestionJobClaim) -> dict[str, object]:
        self.claims.append(claim)
        if self.outcome is not None:
            raise self.outcome
        return {"terminal_status": "completed"}


def _message() -> StreamMessage:
    return StreamMessage(id="1-0", aggregate_id="job-1", enqueued_at=datetime.now(UTC))


@pytest.mark.asyncio
async def test_worker_acks_only_after_graph_reaches_durable_terminal_status() -> None:
    jobs = JobStore()
    broker = Broker()
    graph = Graph()
    worker = IngestionWorker(
        jobs=jobs,
        broker=broker,
        graph=graph,
        reconciler=None,
        worker_id="worker-1",
        heartbeat_interval_seconds=0.01,
    )

    jobs.status = JobStatus.COMPLETED
    await worker.process_message(_message())
    assert broker.acked == ["1-0"]

    broker.acked.clear()
    jobs.status = JobStatus.QUEUED
    await worker.process_message(_message())
    assert broker.acked == []
    assert jobs.status is JobStatus.RUNNING


@pytest.mark.asyncio
async def test_third_failed_attempt_dead_letters_then_acks_terminal_job() -> None:
    jobs = JobStore(attempt_count=2)
    broker = Broker()
    worker = IngestionWorker(
        jobs=jobs,
        broker=broker,
        graph=Graph(RuntimeError("transient provider failure")),
        reconciler=None,
        worker_id="worker-1",
        heartbeat_interval_seconds=0.01,
    )

    await worker.process_message(_message())

    assert jobs.status is JobStatus.FAILED
    assert broker.dead == [("1-0", "RuntimeError")]
    assert broker.acked == ["1-0"]


@pytest.mark.asyncio
async def test_nonterminal_failure_and_lease_loss_leave_message_pending() -> None:
    for error in (RuntimeError("retry"), LeaseLost("job-1")):
        jobs = JobStore()
        broker = Broker()
        worker = IngestionWorker(
            jobs=jobs,
            broker=broker,
            graph=Graph(error),
            reconciler=None,
            worker_id="worker-1",
            heartbeat_interval_seconds=0.01,
        )

        await worker.process_message(_message())

        assert broker.acked == []
        assert broker.dead == []
