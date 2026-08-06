from __future__ import annotations

from collections.abc import AsyncIterator
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from sqlalchemy import insert, select, update
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from agentic_rag.domain.models import JobStatus
from agentic_rag.ingestion.state import IngestionJobClaim
from agentic_rag.ingestion.worker import SqlAlchemyIngestionJobStore
from agentic_rag.persistence.repositories import (
    document_versions,
    documents,
    ingestion_jobs,
    metadata,
    task_outbox,
)


@pytest.fixture
async def store(
    tmp_path: Path,
) -> AsyncIterator[
    tuple[SqlAlchemyIngestionJobStore, async_sessionmaker[AsyncSession]]
]:
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'jobs.sqlite'}")
    async with engine.begin() as connection:
        await connection.run_sync(metadata.create_all)
    factory = async_sessionmaker(engine, expire_on_commit=False)
    try:
        yield SqlAlchemyIngestionJobStore(factory), factory
    finally:
        await engine.dispose()


async def _seed(
    factory: async_sessionmaker[AsyncSession],
    *,
    job_status: str,
    version_status: str,
    attempt_count: int = 0,
    include_outbox: bool = True,
) -> None:
    now = datetime.now(UTC)
    async with factory.begin() as session:
        await session.execute(
            insert(documents).values(
                id="doc-1",
                user_id="user-1",
                source_type="text",
                filename="sample.txt",
                mime_type="text/plain",
                content_hash="a" * 64,
                status="processing",
                source_trust="untrusted",
                created_at=now,
                updated_at=now,
            )
        )
        await session.execute(
            insert(document_versions).values(
                id="version-1",
                document_id="doc-1",
                version_no=1,
                parser_version="docling-v1",
                pipeline_version="ingestion-v1",
                parent_count=0,
                child_count=0,
                embedding_version="text-embedding-v3",
                index_generation="index-v2",
                status=version_status,
                created_at=now,
            )
        )
        await session.execute(
            insert(ingestion_jobs).values(
                id="job-1",
                user_id="user-1",
                document_id="doc-1",
                document_version_id="version-1",
                status=job_status,
                lease_owner="worker-1" if job_status == "running" else None,
                lease_expires_at=(
                    now + timedelta(minutes=1) if job_status == "running" else None
                ),
                heartbeat_at=now if job_status == "running" else None,
                attempt_count=attempt_count,
                created_at=now,
                updated_at=now,
            )
        )
        if include_outbox:
            await session.execute(
                insert(task_outbox).values(
                    id="outbox-1",
                    aggregate_type="ingestion_job",
                    aggregate_id="job-1",
                    stream_name="agenticrag:jobs:ingestion",
                    status="dispatched",
                    attempt_count=0,
                    next_attempt_at=now,
                    created_at=now,
                    dispatched_at=now,
                )
            )


@pytest.mark.asyncio
async def test_approval_rotates_outbox_delivery_generation(
    store: tuple[SqlAlchemyIngestionJobStore, async_sessionmaker[AsyncSession]],
) -> None:
    jobs, factory = store
    await _seed(factory, job_status="quarantined", version_status="quarantined")

    await jobs.review_quarantine("version-1", approve=True)

    async with factory() as session:
        row = (await session.execute(select(task_outbox))).mappings().one()
    assert row["status"] == "pending"
    assert row["attempt_count"] == 1


@pytest.mark.asyncio
async def test_approval_recreates_missing_outbox_row(
    store: tuple[SqlAlchemyIngestionJobStore, async_sessionmaker[AsyncSession]],
) -> None:
    jobs, factory = store
    await _seed(
        factory,
        job_status="quarantined",
        version_status="quarantined",
        include_outbox=False,
    )

    await jobs.review_quarantine("version-1", approve=True)

    async with factory() as session:
        row = (await session.execute(select(task_outbox))).mappings().one()
    assert row["aggregate_id"] == "job-1"
    assert row["status"] == "pending"


@pytest.mark.asyncio
async def test_terminal_failure_atomically_fails_unpublished_version_and_document(
    store: tuple[SqlAlchemyIngestionJobStore, async_sessionmaker[AsyncSession]],
) -> None:
    jobs, factory = store
    await _seed(
        factory, job_status="running", version_status="uploaded", attempt_count=2
    )
    claim = IngestionJobClaim(
        job_id="job-1",
        user_id="user-1",
        document_id="doc-1",
        document_version_id="version-1",
        owner="worker-1",
        claim_generation=2,
    )

    status = await jobs.record_failure(claim, "parse_failed", max_attempts=3)

    async with factory() as session:
        job = (await session.execute(select(ingestion_jobs))).mappings().one()
        version = (await session.execute(select(document_versions))).mappings().one()
        document = (await session.execute(select(documents))).mappings().one()
    assert status is JobStatus.FAILED
    assert job["dead_letter_status"] == "pending"
    assert version["status"] == "failed"
    assert document["status"] == "failed"


@pytest.mark.asyncio
async def test_failure_after_publication_converges_job_to_completed(
    store: tuple[SqlAlchemyIngestionJobStore, async_sessionmaker[AsyncSession]],
) -> None:
    jobs, factory = store
    await _seed(factory, job_status="running", version_status="active", attempt_count=2)
    async with factory.begin() as session:
        await session.execute(
            update(documents)
            .where(documents.c.id == "doc-1")
            .values(status="active", active_version_id="version-1")
        )
    claim = IngestionJobClaim(
        job_id="job-1",
        user_id="user-1",
        document_id="doc-1",
        document_version_id="version-1",
        owner="worker-1",
        claim_generation=2,
    )

    status = await jobs.record_failure(claim, "post_publish_crash", max_attempts=3)

    assert status is JobStatus.COMPLETED
    assert await jobs.get_status("job-1") is JobStatus.COMPLETED


@pytest.mark.asyncio
async def test_dead_letter_mark_repairs_legacy_missing_intent_fields(
    store: tuple[SqlAlchemyIngestionJobStore, async_sessionmaker[AsyncSession]],
) -> None:
    jobs, factory = store
    await _seed(factory, job_status="failed", version_status="failed")

    await jobs.mark_dead_letter_published("job-1", "legacy_failure")

    state = await jobs.get_delivery_state("job-1")
    assert state is not None
    assert state.dead_letter_status == "published"
    assert state.dead_letter_reason == "legacy_failure"
