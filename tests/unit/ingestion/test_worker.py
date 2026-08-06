from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from typing import Any, Literal

import pytest

from agentic_rag.domain.models import JobStatus
from agentic_rag.ingestion.state import IngestionJobClaim, JobDeliveryState
from agentic_rag.ingestion.worker import IngestionWorker
from agentic_rag.persistence.redis_queue import StreamMessage
from agentic_rag.persistence.repositories import LeaseLost


class JobStore:
    def __init__(self, *, attempt_count: int = 0) -> None:
        self.status = JobStatus.QUEUED
        self.attempt_count = attempt_count
        self.owner: str | None = None
        self.heartbeats = 0
        self.dead_letter_status: Literal["pending", "published"] | None = None
        self.dead_letter_reason: str | None = None

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

    async def get_delivery_state(self, job_id: str) -> JobDeliveryState | None:
        return JobDeliveryState(
            status=self.status,
            attempt_count=self.attempt_count,
            dead_letter_status=self.dead_letter_status,
            dead_letter_reason=self.dead_letter_reason,
        )

    async def mark_dead_letter_published(self, job_id: str, reason: str) -> None:
        self.dead_letter_status = "published"
        self.dead_letter_reason = reason

    async def record_failure(
        self, claim: IngestionJobClaim, error_code: str, *, max_attempts: int
    ) -> JobStatus:
        self.attempt_count += 1
        self.owner = None
        self.status = (
            JobStatus.FAILED if self.attempt_count >= max_attempts else JobStatus.QUEUED
        )
        if self.status is JobStatus.FAILED:
            self.dead_letter_status = "pending"
            self.dead_letter_reason = error_code
        return self.status


class Broker:
    def __init__(self) -> None:
        self.acked: list[str] = []
        self.dead: list[tuple[str, str]] = []
        self.fail_dead_letter = False

    async def consume(self, *args: Any, **kwargs: Any) -> list[StreamMessage]:
        return []

    async def publish(self, *args: Any, **kwargs: Any) -> str:
        return "published-1"

    async def reclaim(self, *args: Any, **kwargs: Any) -> list[StreamMessage]:
        return []

    async def ack(self, stream: str, group: str, message_id: str) -> None:
        self.acked.append(message_id)

    async def dead_letter(
        self,
        dead_stream: str,
        message: StreamMessage,
        reason: str,
        dedupe_key: str | None = None,
    ) -> None:
        if self.fail_dead_letter:
            self.fail_dead_letter = False
            raise RuntimeError("dead stream unavailable")
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


class CompletingGraph(Graph):
    def __init__(self, jobs: JobStore, stop: asyncio.Event) -> None:
        super().__init__()
        self._jobs = jobs
        self._stop = stop

    async def run(self, claim: IngestionJobClaim) -> dict[str, object]:
        self._jobs.status = JobStatus.COMPLETED
        self._stop.set()
        return await super().run(claim)


class TransientBroker(Broker):
    def __init__(self, message: StreamMessage) -> None:
        super().__init__()
        self.message = message
        self.reclaim_calls = 0
        self.delivered = False

    async def reclaim(self, *args: Any, **kwargs: Any) -> list[StreamMessage]:
        self.reclaim_calls += 1
        if self.reclaim_calls == 1:
            raise RuntimeError("temporary redis failure")
        return []

    async def consume(self, *args: Any, **kwargs: Any) -> list[StreamMessage]:
        if self.delivered:
            return []
        self.delivered = True
        return [self.message]


class SingleMessageBroker(Broker):
    def __init__(self, message: StreamMessage) -> None:
        super().__init__()
        self.message = message
        self.delivered = False

    async def consume(self, *args: Any, **kwargs: Any) -> list[StreamMessage]:
        if self.delivered:
            return []
        self.delivered = True
        return [self.message]


class BatchBroker(Broker):
    def __init__(self, messages: list[StreamMessage]) -> None:
        super().__init__()
        self.messages = messages
        self.delivered = False

    async def consume(self, *args: Any, **kwargs: Any) -> list[StreamMessage]:
        if self.delivered:
            return []
        self.delivered = True
        return self.messages


class DrainingGraph(Graph):
    def __init__(self, jobs: JobStore) -> None:
        super().__init__()
        self._jobs = jobs
        self.started = asyncio.Event()
        self.release = asyncio.Event()

    async def run(self, claim: IngestionJobClaim) -> dict[str, object]:
        self.started.set()
        await self.release.wait()
        self._jobs.status = JobStatus.COMPLETED
        return await super().run(claim)


class CancellationGraph(Graph):
    def __init__(self) -> None:
        super().__init__()
        self.cancelled = asyncio.Event()

    async def run(self, claim: IngestionJobClaim) -> dict[str, object]:
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            self.cancelled.set()
            raise
        raise AssertionError("cancellation graph unexpectedly resumed")


class FailingDrainGraph(Graph):
    def __init__(self) -> None:
        super().__init__()
        self.started = asyncio.Event()
        self.release = asyncio.Event()

    async def run(self, claim: IngestionJobClaim) -> dict[str, object]:
        self.started.set()
        await self.release.wait()
        raise RuntimeError("drained message failed")


class HeartbeatLosingJobStore(JobStore):
    async def heartbeat(self, claim: IngestionJobClaim, lease_seconds: int) -> None:
        raise LeaseLost(claim.job_id)


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


@pytest.mark.asyncio
async def test_terminal_fast_path_repairs_durable_dead_letter_before_ack() -> None:
    jobs = JobStore(attempt_count=3)
    jobs.status = JobStatus.FAILED
    jobs.dead_letter_status = "pending"
    broker = Broker()
    broker.fail_dead_letter = True
    worker = IngestionWorker(
        jobs=jobs,
        broker=broker,
        graph=Graph(),
        reconciler=None,
        worker_id="worker-1",
        heartbeat_interval_seconds=0.01,
    )

    with pytest.raises(RuntimeError, match="dead stream unavailable"):
        await worker.process_message(_message())
    assert broker.acked == []
    assert jobs.dead_letter_status == "pending"

    await worker.process_message(_message())
    assert jobs.dead_letter_status == "published"
    assert broker.acked == ["1-0"]


@pytest.mark.asyncio
async def test_run_forever_retries_broker_error_and_processes_later_message() -> None:
    jobs = JobStore()
    stop = asyncio.Event()
    broker = TransientBroker(_message())
    worker = IngestionWorker(
        jobs=jobs,
        broker=broker,
        graph=CompletingGraph(jobs, stop),
        reconciler=None,
        worker_id="worker-1",
        heartbeat_interval_seconds=0.01,
        retry_backoff_seconds=0.001,
    )

    await worker.run_forever(stop_event=stop)

    assert broker.reclaim_calls >= 2
    assert broker.acked == ["1-0"]


@pytest.mark.asyncio
async def test_stop_signal_drains_current_message_before_worker_exits() -> None:
    jobs = JobStore()
    stop = asyncio.Event()
    broker = SingleMessageBroker(_message())
    graph = DrainingGraph(jobs)
    worker = IngestionWorker(
        jobs=jobs,
        broker=broker,
        graph=graph,
        reconciler=None,
        worker_id="worker-1",
        heartbeat_interval_seconds=0.01,
    )

    task = asyncio.create_task(worker.run_forever(stop_event=stop))
    await asyncio.wait_for(graph.started.wait(), timeout=1)
    stop.set()
    await asyncio.sleep(0)
    assert not task.done()

    graph.release.set()
    await asyncio.wait_for(task, timeout=1)
    assert broker.acked == ["1-0"]


@pytest.mark.asyncio
async def test_stop_signal_does_not_claim_later_messages_in_consumed_batch() -> None:
    jobs = JobStore()
    stop = asyncio.Event()
    second = StreamMessage(
        id="2-0", aggregate_id="job-2", enqueued_at=datetime.now(UTC)
    )
    broker = BatchBroker([_message(), second])
    graph = CompletingGraph(jobs, stop)
    worker = IngestionWorker(
        jobs=jobs,
        broker=broker,
        graph=graph,
        reconciler=None,
        worker_id="worker-1",
        heartbeat_interval_seconds=0.01,
    )

    await worker.run_forever(stop_event=stop)

    assert [claim.job_id for claim in graph.claims] == ["job-1"]
    assert broker.acked == ["1-0"]


@pytest.mark.asyncio
async def test_heartbeat_lease_loss_cancels_graph_and_leaves_message_pending() -> None:
    jobs = HeartbeatLosingJobStore()
    broker = Broker()
    graph = CancellationGraph()
    worker = IngestionWorker(
        jobs=jobs,
        broker=broker,
        graph=graph,
        reconciler=None,
        worker_id="worker-1",
        heartbeat_interval_seconds=0.001,
    )

    await worker.process_message(_message())

    assert graph.cancelled.is_set()
    assert broker.acked == []


def test_broker_retry_backoff_saturates_before_large_exponent_overflows() -> None:
    worker = IngestionWorker(
        jobs=JobStore(),
        broker=Broker(),
        graph=Graph(),
        reconciler=None,
        worker_id="worker-1",
        heartbeat_interval_seconds=0.01,
        retry_backoff_seconds=0.25,
        max_retry_backoff_seconds=5.0,
    )

    delay = worker._bounded_retry_delay(10**100)  # noqa: SLF001

    assert 0.25 <= delay <= 5.0
    assert delay == 5.0


@pytest.mark.asyncio
async def test_cancellation_preserves_cancelled_error_after_drained_message_fails() -> (
    None
):
    jobs = JobStore(attempt_count=2)
    broker = SingleMessageBroker(_message())
    broker.fail_dead_letter = True
    graph = FailingDrainGraph()
    worker = IngestionWorker(
        jobs=jobs,
        broker=broker,
        graph=graph,
        reconciler=None,
        worker_id="worker-1",
        heartbeat_interval_seconds=0.01,
    )
    task = asyncio.create_task(worker.run_forever())
    await asyncio.wait_for(graph.started.wait(), timeout=1)

    task.cancel()
    await asyncio.sleep(0)
    graph.release.set()

    with pytest.raises(asyncio.CancelledError):
        await task
    assert broker.acked == []
