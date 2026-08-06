"""Redis-backed regressions for quarantine approval delivery generations."""

from __future__ import annotations

import os
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import urlparse
from uuid import uuid4

import pytest
from redis.asyncio import Redis
from sqlalchemy import insert, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from agentic_rag.domain.models import JobStatus
from agentic_rag.ingestion.state import IngestionJobClaim
from agentic_rag.ingestion.worker import (
    INGESTION_STREAM,
    IngestionWorker,
    SqlAlchemyIngestionJobStore,
)
from agentic_rag.persistence.outbox import OutboxDispatcher
from agentic_rag.persistence.redis_queue import RedisStreamsBroker
from agentic_rag.persistence.repositories import (
    OutboxRecord,
    SqlAlchemyOutboxRepository,
    document_versions,
    documents,
    ingestion_jobs,
    metadata,
    task_outbox,
)


pytestmark = pytest.mark.integration


class _TransactionalOutbox:
    def __init__(self, factory: async_sessionmaker[AsyncSession]) -> None:
        self._factory = factory

    async def list_pending(self, limit: int) -> list[OutboxRecord]:
        async with self._factory() as session:
            return await SqlAlchemyOutboxRepository(session).list_pending(limit)

    async def claim_pending(self, limit: int) -> list[OutboxRecord]:
        async with self._factory.begin() as session:
            return await SqlAlchemyOutboxRepository(session).claim_pending(limit)

    async def mark_dispatched(self, outbox_id: str) -> None:
        async with self._factory.begin() as session:
            await SqlAlchemyOutboxRepository(session).mark_dispatched(outbox_id)

    async def schedule_retry(self, outbox_id: str) -> None:
        async with self._factory.begin() as session:
            await SqlAlchemyOutboxRepository(session).schedule_retry(outbox_id)


class _CompletingGraph:
    def __init__(self, jobs: SqlAlchemyIngestionJobStore) -> None:
        self._jobs = jobs
        self.claims: list[IngestionJobClaim] = []

    async def run(self, claim: IngestionJobClaim) -> dict[str, object]:
        self.claims.append(claim)
        await self._jobs.complete(claim)
        return {"terminal_status": "completed"}


def _local_redis_dsn() -> str:
    dsn = os.getenv("AGENTIC_RAG_TEST_REDIS_DSN")
    if not dsn:
        pytest.skip("set AGENTIC_RAG_TEST_REDIS_DSN to run approval delivery tests")
    parsed = urlparse(dsn)
    if parsed.scheme not in {"redis", "rediss"} or parsed.hostname not in {
        "127.0.0.1",
        "::1",
        "localhost",
    }:
        pytest.skip("AGENTIC_RAG_TEST_REDIS_DSN must target explicit local Redis")
    return dsn


@pytest.mark.parametrize(
    "quarantine_reason", ("upload_quarantined", "instruction_like_content")
)
@pytest.mark.asyncio
async def test_approval_after_old_ack_delivers_new_claimable_generation(
    tmp_path: Path,
    quarantine_reason: str,
) -> None:
    redis_dsn = _local_redis_dsn()
    suffix = uuid4().hex
    stream = INGESTION_STREAM
    group = f"approval-workers-{suffix}"
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / f'{suffix}.sqlite'}")
    async with engine.begin() as connection:
        await connection.run_sync(metadata.create_all)
    factory = async_sessionmaker(engine, expire_on_commit=False)
    redis = Redis.from_url(redis_dsn, decode_responses=True)
    broker = RedisStreamsBroker(redis)
    jobs = SqlAlchemyIngestionJobStore(factory)
    dispatcher = OutboxDispatcher(_TransactionalOutbox(factory), broker)
    now = datetime.now(UTC)
    try:
        # Once explicitly configured, an unavailable Redis is an integration failure.
        await redis.ping()
        async with factory.begin() as session:
            await session.execute(
                insert(documents).values(
                    id=f"doc-{suffix}",
                    user_id="user-1",
                    source_type="pdf",
                    filename="review.pdf",
                    mime_type="application/pdf",
                    content_hash="a" * 64,
                    status="processing",
                    source_trust="untrusted",
                    created_at=now,
                    updated_at=now,
                )
            )
            await session.execute(
                insert(document_versions).values(
                    id=f"version-{suffix}",
                    document_id=f"doc-{suffix}",
                    version_no=1,
                    parser_version="docling-v1",
                    pipeline_version="ingestion-v1",
                    embedding_version="text-embedding-v3",
                    index_generation="index-v2",
                    status="quarantined",
                    created_at=now,
                )
            )
            await session.execute(
                insert(ingestion_jobs).values(
                    id=f"job-{suffix}",
                    user_id="user-1",
                    document_id=f"doc-{suffix}",
                    document_version_id=f"version-{suffix}",
                    status="quarantined",
                    error_code=quarantine_reason,
                    attempt_count=0,
                    created_at=now,
                    updated_at=now,
                )
            )
            await session.execute(
                insert(task_outbox).values(
                    id=f"outbox-{suffix}",
                    aggregate_type="ingestion_job",
                    aggregate_id=f"job-{suffix}",
                    stream_name=stream,
                    status="pending",
                    attempt_count=0,
                    next_attempt_at=now,
                    created_at=now,
                )
            )

        assert await dispatcher.dispatch_once() == 1
        old = await broker.consume(stream, group, "worker-old", 100)
        assert len(old) == 1
        await broker.ack(stream, group, old[0].id)

        await jobs.review_quarantine(f"version-{suffix}", approve=True)
        assert await dispatcher.dispatch_once() == 1
        fresh = await broker.consume(stream, group, "worker-new", 100)
        assert len(fresh) == 1
        assert fresh[0].id != old[0].id

        graph = _CompletingGraph(jobs)
        worker = IngestionWorker(
            jobs=jobs,
            broker=broker,
            graph=graph,
            reconciler=None,
            worker_id="worker-new",
            heartbeat_interval_seconds=0.01,
        )
        await worker.process_message(fresh[0])

        assert await jobs.get_status(f"job-{suffix}") is JobStatus.COMPLETED
        assert len(graph.claims) == 1
        async with factory() as session:
            generation = await session.scalar(
                select(task_outbox.c.attempt_count).where(
                    task_outbox.c.id == f"outbox-{suffix}"
                )
            )
        assert generation == 1
    finally:
        await redis.delete(stream, f"{stream}:dedupe")
        await redis.aclose()
        await engine.dispose()
