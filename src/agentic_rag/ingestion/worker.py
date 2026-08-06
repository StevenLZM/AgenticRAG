"""Redis Streams ingestion worker with MySQL lease fencing and recovery."""

from __future__ import annotations

import asyncio
from contextlib import suppress
from datetime import UTC, datetime, timedelta
from typing import Protocol

from sqlalchemy import select, update
from sqlalchemy.sql.elements import ColumnElement
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from agentic_rag.domain.models import DocumentStatus, DocumentVersionStatus
from agentic_rag.domain.models import JobStatus
from agentic_rag.ingestion.state import IngestionJobClaim, IngestionRuntime
from agentic_rag.persistence.redis_queue import StreamBroker, StreamMessage
from agentic_rag.persistence.repositories import (
    LeaseLost,
    document_versions,
    documents,
    ingestion_jobs,
    task_outbox,
)


INGESTION_STREAM = "agenticrag:jobs:ingestion"
INGESTION_GROUP = "agenticrag-ingestion-workers"
INGESTION_DEAD_STREAM = "agenticrag:jobs:ingestion:dead"
TERMINAL_JOB_STATUSES = {
    JobStatus.COMPLETED,
    JobStatus.QUARANTINED,
    JobStatus.FAILED,
}


class IngestionJobStore(Protocol):
    async def claim(
        self, job_id: str, owner: str, lease_seconds: int
    ) -> IngestionJobClaim | None: ...

    async def heartbeat(self, claim: IngestionJobClaim, lease_seconds: int) -> None: ...

    async def get_status(self, job_id: str) -> JobStatus | None: ...

    async def record_failure(
        self,
        claim: IngestionJobClaim,
        error_code: str,
        *,
        max_attempts: int,
    ) -> JobStatus: ...


class IngestionGraphRunner(Protocol):
    async def run(self, claim: IngestionJobClaim) -> dict[str, object]: ...


class ReconcilerRunner(Protocol):
    async def run_once(self) -> object: ...


class DeterministicIngestionError(RuntimeError):
    """Input cannot succeed on retry and should fail immediately."""


class IngestionWorker:
    """Consume notifications while keeping MySQL as the lifecycle authority."""

    def __init__(
        self,
        *,
        jobs: IngestionJobStore,
        broker: StreamBroker,
        graph: IngestionGraphRunner,
        reconciler: ReconcilerRunner | None,
        worker_id: str,
        lease_seconds: int = 30,
        heartbeat_interval_seconds: float = 10,
        reclaim_idle_ms: int = 30_000,
        reconcile_interval_seconds: float = 30,
        block_ms: int = 1_000,
        max_attempts: int = 3,
    ) -> None:
        if not worker_id.strip():
            raise ValueError("worker_id must not be blank")
        if lease_seconds <= 0 or heartbeat_interval_seconds <= 0:
            raise ValueError("lease and heartbeat intervals must be positive")
        if heartbeat_interval_seconds >= lease_seconds:
            raise ValueError("heartbeat interval must be shorter than the lease")
        if reclaim_idle_ms < 0 or reconcile_interval_seconds <= 0 or block_ms < 0:
            raise ValueError("worker intervals must be non-negative and bounded")
        if max_attempts <= 0:
            raise ValueError("max_attempts must be positive")
        self._jobs = jobs
        self._broker = broker
        self._graph = graph
        self._reconciler = reconciler
        self._worker_id = worker_id
        self._lease_seconds = lease_seconds
        self._heartbeat_interval = heartbeat_interval_seconds
        self._reclaim_idle_ms = reclaim_idle_ms
        self._reconcile_interval = reconcile_interval_seconds
        self._block_ms = block_ms
        self._max_attempts = max_attempts

    async def process_message(self, message: StreamMessage) -> None:
        claim = await self._jobs.claim(
            message.aggregate_id, self._worker_id, self._lease_seconds
        )
        if claim is None:
            if (
                await self._jobs.get_status(message.aggregate_id)
                in TERMINAL_JOB_STATUSES
            ):
                await self._ack(message)
            return

        try:
            await self._run_with_heartbeat(claim)
        except LeaseLost:
            return
        except DeterministicIngestionError as error:
            status = await self._jobs.record_failure(
                claim, type(error).__name__, max_attempts=1
            )
            if status in TERMINAL_JOB_STATUSES:
                await self._ack(message)
            return
        except Exception as error:
            try:
                status = await self._jobs.record_failure(
                    claim, type(error).__name__, max_attempts=self._max_attempts
                )
            except LeaseLost:
                return
            if status is JobStatus.FAILED:
                await self._broker.dead_letter(
                    INGESTION_DEAD_STREAM, message, type(error).__name__
                )
                await self._ack(message)
            return

        if await self._jobs.get_status(message.aggregate_id) in TERMINAL_JOB_STATUSES:
            await self._ack(message)

    async def run_forever(self) -> None:
        reconcile_task: asyncio.Task[None] | None = None
        if self._reconciler is not None:
            reconcile_task = asyncio.create_task(self._reconcile_forever())
        try:
            while True:
                reclaimed = await self._broker.reclaim(
                    INGESTION_STREAM,
                    INGESTION_GROUP,
                    self._worker_id,
                    self._reclaim_idle_ms,
                )
                fresh = await self._broker.consume(
                    INGESTION_STREAM,
                    INGESTION_GROUP,
                    self._worker_id,
                    self._block_ms,
                )
                for message in (*reclaimed, *fresh):
                    await self.process_message(message)
        finally:
            if reconcile_task is not None:
                reconcile_task.cancel()
                with suppress(asyncio.CancelledError):
                    await reconcile_task

    async def _run_with_heartbeat(self, claim: IngestionJobClaim) -> None:
        graph_task = asyncio.create_task(self._graph.run(claim))
        try:
            while True:
                done, _ = await asyncio.wait(
                    {graph_task}, timeout=self._heartbeat_interval
                )
                if done:
                    await graph_task
                    return
                try:
                    await self._jobs.heartbeat(claim, self._lease_seconds)
                except BaseException:
                    graph_task.cancel()
                    with suppress(asyncio.CancelledError):
                        await graph_task
                    raise
        finally:
            if not graph_task.done():
                graph_task.cancel()

    async def _reconcile_forever(self) -> None:
        assert self._reconciler is not None
        while True:
            try:
                await self._reconciler.run_once()
            except Exception:
                pass
            await asyncio.sleep(self._reconcile_interval)

    async def _ack(self, message: StreamMessage) -> None:
        await self._broker.ack(INGESTION_STREAM, INGESTION_GROUP, message.id)


class SqlAlchemyIngestionJobStore:
    """Transactional Job lease, runtime projection and quarantine-review adapter."""

    def __init__(self, session_factory: async_sessionmaker[AsyncSession]) -> None:
        self._session_factory = session_factory

    async def claim(
        self, job_id: str, owner: str, lease_seconds: int
    ) -> IngestionJobClaim | None:
        if lease_seconds <= 0:
            raise ValueError("lease_seconds must be positive")
        now = datetime.now(UTC)
        async with self._session_factory.begin() as session:
            result = await session.execute(
                update(ingestion_jobs)
                .where(
                    ingestion_jobs.c.id == job_id,
                    ingestion_jobs.c.status == JobStatus.QUEUED.value,
                )
                .values(
                    status=JobStatus.RUNNING.value,
                    lease_owner=owner,
                    lease_expires_at=now + timedelta(seconds=lease_seconds),
                    heartbeat_at=now,
                    updated_at=now,
                )
            )
            if getattr(result, "rowcount", 0) != 1:
                return None
            row = (
                (
                    await session.execute(
                        select(ingestion_jobs).where(ingestion_jobs.c.id == job_id)
                    )
                )
                .mappings()
                .one()
            )
        return _claim_from_row(row, owner)

    async def heartbeat(self, claim: IngestionJobClaim, lease_seconds: int) -> None:
        now = datetime.now(UTC)
        async with self._session_factory.begin() as session:
            result = await session.execute(
                update(ingestion_jobs)
                .where(*_live_claim_predicates(claim, now))
                .values(
                    heartbeat_at=now,
                    lease_expires_at=now + timedelta(seconds=lease_seconds),
                    updated_at=now,
                )
            )
            if getattr(result, "rowcount", 0) != 1:
                raise LeaseLost(claim.job_id)

    async def assert_lease(self, claim: IngestionJobClaim) -> None:
        now = datetime.now(UTC)
        async with self._session_factory() as session:
            value = await session.scalar(
                select(ingestion_jobs.c.id).where(*_live_claim_predicates(claim, now))
            )
        if value is None:
            raise LeaseLost(claim.job_id)

    async def get_status(self, job_id: str) -> JobStatus | None:
        async with self._session_factory() as session:
            value = await session.scalar(
                select(ingestion_jobs.c.status).where(ingestion_jobs.c.id == job_id)
            )
        return JobStatus(value) if value is not None else None

    async def record_failure(
        self,
        claim: IngestionJobClaim,
        error_code: str,
        *,
        max_attempts: int,
    ) -> JobStatus:
        now = datetime.now(UTC)
        next_attempt = claim.claim_generation + 1
        status = JobStatus.FAILED if next_attempt >= max_attempts else JobStatus.QUEUED
        async with self._session_factory.begin() as session:
            result = await session.execute(
                update(ingestion_jobs)
                .where(*_live_claim_predicates(claim, now))
                .values(
                    status=status.value,
                    attempt_count=next_attempt,
                    lease_owner=None,
                    lease_expires_at=None,
                    heartbeat_at=None,
                    error_code=error_code[:128],
                    updated_at=now,
                )
            )
            if getattr(result, "rowcount", 0) != 1:
                raise LeaseLost(claim.job_id)
        return status

    async def load_job(self, claim: IngestionJobClaim) -> IngestionRuntime:
        now = datetime.now(UTC)
        async with self._session_factory() as session:
            row = (
                (
                    await session.execute(
                        select(
                            ingestion_jobs,
                            documents.c.filename,
                            documents.c.mime_type,
                            documents.c.source_type,
                            documents.c.content_hash,
                            documents.c.source_trust,
                            document_versions.c.version_no,
                            document_versions.c.parser_version,
                            document_versions.c.pipeline_version,
                            document_versions.c.embedding_version,
                            document_versions.c.index_generation,
                        )
                        .select_from(
                            ingestion_jobs.join(
                                documents,
                                documents.c.id == ingestion_jobs.c.document_id,
                            ).join(
                                document_versions,
                                document_versions.c.id
                                == ingestion_jobs.c.document_version_id,
                            )
                        )
                        .where(*_live_claim_predicates(claim, now))
                    )
                )
                .mappings()
                .one_or_none()
            )
        if row is None:
            raise LeaseLost(claim.job_id)
        return IngestionRuntime(
            job_id=claim.job_id,
            user_id=str(row["user_id"]),
            document_id=str(row["document_id"]),
            document_version_id=str(row["document_version_id"]),
            version_no=int(row["version_no"]),
            filename=str(row["filename"]),
            mime_type=str(row["mime_type"]),
            source_type=str(row["source_type"]),  # type: ignore[arg-type]
            content_hash=str(row["content_hash"]),
            parser_version=str(row["parser_version"]),
            pipeline_version=str(row["pipeline_version"]),
            embedding_version=str(row["embedding_version"]),
            index_generation=str(row["index_generation"]),
            source_trust=str(row["source_trust"]),
        )

    async def complete(self, claim: IngestionJobClaim) -> None:
        await self._finish_claim(claim, JobStatus.COMPLETED)

    async def quarantine(self, claim: IngestionJobClaim, reasons: list[str]) -> None:
        now = datetime.now(UTC)
        async with self._session_factory.begin() as session:
            result = await session.execute(
                update(ingestion_jobs)
                .where(*_live_claim_predicates(claim, now))
                .values(
                    status=JobStatus.QUARANTINED.value,
                    lease_owner=None,
                    lease_expires_at=None,
                    heartbeat_at=None,
                    error_code=(reasons[0] if reasons else "content_quarantined")[:128],
                    updated_at=now,
                )
            )
            if getattr(result, "rowcount", 0) != 1:
                raise LeaseLost(claim.job_id)
            await session.execute(
                update(document_versions)
                .where(document_versions.c.id == claim.document_version_id)
                .values(status=DocumentVersionStatus.QUARANTINED.value)
            )

    async def review_quarantine(self, version_id: str, *, approve: bool) -> str:
        """Apply one explicit trusted-local approval/rejection and return the Job ID."""
        now = datetime.now(UTC)
        async with self._session_factory.begin() as session:
            row = (
                await session.execute(
                    select(
                        ingestion_jobs.c.id,
                        ingestion_jobs.c.document_id,
                    )
                    .where(
                        ingestion_jobs.c.document_version_id == version_id,
                        ingestion_jobs.c.status == JobStatus.QUARANTINED.value,
                    )
                    .with_for_update()
                )
            ).one_or_none()
            if row is None:
                raise KeyError(version_id)
            job_id, document_id = str(row.id), str(row.document_id)
            if approve:
                await session.execute(
                    update(documents)
                    .where(documents.c.id == document_id)
                    .values(source_trust="trusted_curated", updated_at=now)
                )
                await session.execute(
                    update(document_versions)
                    .where(document_versions.c.id == version_id)
                    .values(status=DocumentVersionStatus.UPLOADED.value)
                )
                await session.execute(
                    update(ingestion_jobs)
                    .where(ingestion_jobs.c.id == job_id)
                    .values(
                        status=JobStatus.QUEUED.value,
                        error_code=None,
                        lease_owner=None,
                        lease_expires_at=None,
                        heartbeat_at=None,
                        updated_at=now,
                    )
                )
                await session.execute(
                    update(task_outbox)
                    .where(
                        task_outbox.c.aggregate_type == "ingestion_job",
                        task_outbox.c.aggregate_id == job_id,
                    )
                    .values(status="pending", next_attempt_at=now, dispatched_at=None)
                )
            else:
                await session.execute(
                    update(document_versions)
                    .where(document_versions.c.id == version_id)
                    .values(status=DocumentVersionStatus.FAILED.value)
                )
                await session.execute(
                    update(ingestion_jobs)
                    .where(ingestion_jobs.c.id == job_id)
                    .values(
                        status=JobStatus.FAILED.value,
                        error_code="review_rejected",
                        updated_at=now,
                    )
                )
                await session.execute(
                    update(documents)
                    .where(
                        documents.c.id == document_id,
                        documents.c.active_version_id.is_(None),
                    )
                    .values(status=DocumentStatus.FAILED.value, updated_at=now)
                )
        return job_id

    async def _finish_claim(self, claim: IngestionJobClaim, status: JobStatus) -> None:
        now = datetime.now(UTC)
        async with self._session_factory.begin() as session:
            result = await session.execute(
                update(ingestion_jobs)
                .where(*_live_claim_predicates(claim, now))
                .values(
                    status=status.value,
                    lease_owner=None,
                    lease_expires_at=None,
                    heartbeat_at=None,
                    error_code=None,
                    updated_at=now,
                )
            )
            if getattr(result, "rowcount", 0) != 1:
                raise LeaseLost(claim.job_id)


def _live_claim_predicates(
    claim: IngestionJobClaim, now: datetime
) -> tuple[ColumnElement[bool], ...]:
    return (
        ingestion_jobs.c.id == claim.job_id,
        ingestion_jobs.c.user_id == claim.user_id,
        ingestion_jobs.c.document_id == claim.document_id,
        ingestion_jobs.c.document_version_id == claim.document_version_id,
        ingestion_jobs.c.status == JobStatus.RUNNING.value,
        ingestion_jobs.c.lease_owner == claim.owner,
        ingestion_jobs.c.attempt_count == claim.claim_generation,
        ingestion_jobs.c.lease_expires_at > now,
    )


def _claim_from_row(row: object, owner: str) -> IngestionJobClaim:
    values = row
    return IngestionJobClaim(
        job_id=str(values["id"]),  # type: ignore[index]
        user_id=str(values["user_id"]),  # type: ignore[index]
        document_id=str(values["document_id"]),  # type: ignore[index]
        document_version_id=str(values["document_version_id"]),  # type: ignore[index]
        owner=owner,
        claim_generation=int(values["attempt_count"]),  # type: ignore[index]
    )
