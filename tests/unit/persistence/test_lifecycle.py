"""Offline SQL lifecycle branch coverage using an ephemeral database."""

from __future__ import annotations

from collections.abc import AsyncIterator
from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy import insert, select, update
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from agentic_rag.domain.models import DocumentStatus, DocumentVersionStatus, JobStatus
from agentic_rag.persistence.lifecycle import (
    SqlAlchemyPublicationRepository,
    SqlAlchemyReconciliationRepository,
)
from agentic_rag.persistence.repositories import (
    document_versions,
    documents,
    ingestion_jobs,
    metadata,
    task_outbox,
)


@pytest.fixture
async def lifecycle_store() -> AsyncIterator[
    tuple[SqlAlchemyReconciliationRepository, async_sessionmaker[AsyncSession]]
]:
    engine: AsyncEngine = create_async_engine("sqlite+aiosqlite://")
    async with engine.begin() as connection:
        await connection.run_sync(metadata.create_all)
    factory = async_sessionmaker(engine, expire_on_commit=False)
    try:
        yield (
            SqlAlchemyReconciliationRepository(
                factory, stale_after=timedelta(minutes=15)
            ),
            factory,
        )
    finally:
        await engine.dispose()


async def _document(
    factory: async_sessionmaker[AsyncSession],
    *,
    document_id: str,
    pointer: str | None,
    status: DocumentStatus | None = None,
    deletion_status: str | None = None,
    deletion_fenced_at: datetime | None = None,
) -> None:
    now = datetime.now(UTC)
    async with factory.begin() as session:
        await session.execute(
            insert(documents).values(
                id=document_id,
                user_id="user-1",
                source_type="text",
                filename="test.txt",
                mime_type="text/plain",
                content_hash="a" * 64,
                status=(
                    status
                    or (
                        DocumentStatus.ACTIVE
                        if pointer is not None
                        else DocumentStatus.PROCESSING
                    )
                ).value,
                active_version_id=pointer,
                deletion_status=deletion_status,
                deletion_fenced_at=deletion_fenced_at,
                source_trust="untrusted",
                created_at=now,
                updated_at=now,
            )
        )


async def _version(
    factory: async_sessionmaker[AsyncSession],
    *,
    version_id: str,
    document_id: str,
    version_no: int,
    status: DocumentVersionStatus,
    manifest: bool = True,
) -> None:
    async with factory.begin() as session:
        await session.execute(
            insert(document_versions).values(
                id=version_id,
                document_id=document_id,
                version_no=version_no,
                parser_version="docling-v1",
                pipeline_version="pipeline-v1",
                canonical_ast_path=("artifact://canonical" if manifest else None),
                canonical_ast_hash=("a" * 64 if manifest else None),
                manifest_path=("artifact://manifest" if manifest else None),
                manifest_hash=("b" * 64 if manifest else None),
                parent_count=1 if manifest else 0,
                child_count=1 if manifest else 0,
                embedding_version="text-embedding-v3",
                index_generation="index-v2",
                status=status.value,
                created_at=datetime.now(UTC) - timedelta(hours=1),
            )
        )


async def test_live_or_reclaimed_job_excludes_building_version_from_repairs(
    lifecycle_store: tuple[
        SqlAlchemyReconciliationRepository, async_sessionmaker[AsyncSession]
    ],
) -> None:
    repository, factory = lifecycle_store
    await _document(factory, document_id="document-live", pointer=None)
    await _version(
        factory,
        version_id="version-live",
        document_id="document-live",
        version_no=1,
        status=DocumentVersionStatus.BUILDING,
    )
    now = datetime.now(UTC)
    async with factory.begin() as session:
        await session.execute(
            insert(ingestion_jobs).values(
                id="job-live",
                user_id="user-1",
                document_id="document-live",
                document_version_id="version-live",
                status=JobStatus.RUNNING.value,
                lease_owner="worker-1",
                lease_expires_at=now + timedelta(minutes=5),
                heartbeat_at=now,
                attempt_count=1,
                created_at=now,
                updated_at=now,
            )
        )
        await session.execute(
            insert(task_outbox).values(
                id="outbox-live",
                aggregate_type="ingestion_job",
                aggregate_id="job-live",
                stream_name="jobs",
                status="dispatched",
                attempt_count=0,
                next_attempt_at=now,
                created_at=now,
                dispatched_at=now,
            )
        )

    assert await repository.list_stale_building_versions(10) == ()
    assert await repository.list_pointer_mismatches(10) == ()

    async with factory.begin() as session:
        await session.execute(
            update(ingestion_jobs)
            .where(ingestion_jobs.c.id == "job-live")
            .values(lease_expires_at=now - timedelta(seconds=1))
        )
    assert await repository.reclaim_expired_jobs(10) == ("job-live",)
    assert await repository.list_stale_building_versions(10) == ()
    assert await repository.list_pointer_mismatches(10) == ()
    async with factory() as session:
        outbox = (
            (
                await session.execute(
                    select(task_outbox).where(task_outbox.c.id == "outbox-live")
                )
            )
            .mappings()
            .one()
        )
    assert outbox["status"] == "pending"
    assert outbox["attempt_count"] == 1


async def test_pointer_scan_classifies_older_physical_losers_for_deactivation(
    lifecycle_store: tuple[
        SqlAlchemyReconciliationRepository, async_sessionmaker[AsyncSession]
    ],
) -> None:
    repository, factory = lifecycle_store
    await _document(factory, document_id="document-race", pointer=None)
    await _version(
        factory,
        version_id="version-2",
        document_id="document-race",
        version_no=2,
        status=DocumentVersionStatus.BUILDING,
    )
    await _version(
        factory,
        version_id="version-3",
        document_id="document-race",
        version_no=3,
        status=DocumentVersionStatus.ACTIVE,
    )
    async with factory.begin() as session:
        await session.execute(
            update(documents)
            .where(documents.c.id == "document-race")
            .values(
                active_version_id="version-3",
                status=DocumentStatus.ACTIVE.value,
            )
        )

    mismatches = await repository.list_pointer_mismatches(10)

    assert [(item.context.document_version_id, item.action) for item in mismatches] == [
        ("version-2", "deactivate")
    ]
    await repository.resolve_deactivated_version("version-2")
    async with factory() as session:
        status = await session.scalar(
            select(document_versions.c.status).where(
                document_versions.c.id == "version-2"
            )
        )
    assert status == DocumentVersionStatus.INACTIVE.value


async def test_pointer_to_inactive_complete_version_is_republished(
    lifecycle_store: tuple[
        SqlAlchemyReconciliationRepository, async_sessionmaker[AsyncSession]
    ],
) -> None:
    repository, factory = lifecycle_store
    await _document(factory, document_id="document-pointer", pointer=None)
    await _version(
        factory,
        version_id="version-pointer",
        document_id="document-pointer",
        version_no=1,
        status=DocumentVersionStatus.INACTIVE,
    )
    async with factory.begin() as session:
        await session.execute(
            update(documents)
            .where(documents.c.id == "document-pointer")
            .values(
                active_version_id="version-pointer",
                status=DocumentStatus.ACTIVE.value,
            )
        )

    mismatches = await repository.list_pointer_mismatches(10)

    assert [(item.context.document_version_id, item.action) for item in mismatches] == [
        ("version-pointer", "publish")
    ]
    publisher_repository = SqlAlchemyPublicationRepository(factory)
    target = await publisher_repository.get_target("version-pointer")
    assert target is not None
    await publisher_repository.finalize(target)
    async with factory() as session:
        status = await session.scalar(
            select(document_versions.c.status).where(
                document_versions.c.id == "version-pointer"
            )
        )
    assert status == DocumentVersionStatus.ACTIVE.value


async def test_outbox_claim_is_ingestion_only_and_keeps_delivery_attempt_stable(
    lifecycle_store: tuple[
        SqlAlchemyReconciliationRepository, async_sessionmaker[AsyncSession]
    ],
) -> None:
    repository, factory = lifecycle_store
    now = datetime.now(UTC)
    async with factory.begin() as session:
        for outbox_id, aggregate_type in (
            ("outbox-ingestion", "ingestion_job"),
            ("outbox-query", "query_run"),
        ):
            await session.execute(
                insert(task_outbox).values(
                    id=outbox_id,
                    aggregate_type=aggregate_type,
                    aggregate_id=(
                        "job-1" if aggregate_type == "ingestion_job" else "run-1"
                    ),
                    stream_name="jobs",
                    status="pending",
                    attempt_count=0,
                    next_attempt_at=now - timedelta(seconds=1),
                    created_at=now,
                )
            )

    first = await repository.claim_pending_outbox(10)
    assert [(row.id, row.attempt_count) for row in first] == [("outbox-ingestion", 0)]

    async with factory.begin() as session:
        await session.execute(
            update(task_outbox)
            .where(task_outbox.c.id == "outbox-ingestion")
            .values(next_attempt_at=now - timedelta(seconds=1))
        )
    second = await repository.claim_pending_outbox(10)
    assert [(row.id, row.attempt_count) for row in second] == [("outbox-ingestion", 0)]


async def test_pointer_scan_restores_document_status_for_pointed_active_version(
    lifecycle_store: tuple[
        SqlAlchemyReconciliationRepository, async_sessionmaker[AsyncSession]
    ],
) -> None:
    repository, factory = lifecycle_store
    await _document(
        factory,
        document_id="document-status-drift",
        pointer=None,
        status=DocumentStatus.FAILED,
    )
    await _version(
        factory,
        version_id="version-active",
        document_id="document-status-drift",
        version_no=1,
        status=DocumentVersionStatus.ACTIVE,
    )
    async with factory.begin() as session:
        await session.execute(
            update(documents)
            .where(documents.c.id == "document-status-drift")
            .values(active_version_id="version-active")
        )

    mismatches = await repository.list_pointer_mismatches(10)

    assert [(item.context.document_version_id, item.action) for item in mismatches] == [
        ("version-active", "restore")
    ]
    assert await repository.restore_active_document("version-active") is True
    async with factory() as session:
        row = (
            await session.execute(
                select(documents.c.status, documents.c.active_version_id).where(
                    documents.c.id == "document-status-drift"
                )
            )
        ).one()
    assert row.status == DocumentStatus.ACTIVE.value
    assert row.active_version_id == "version-active"


async def test_pointerless_active_document_fails_closed_after_active_versions_deactivate(
    lifecycle_store: tuple[
        SqlAlchemyReconciliationRepository, async_sessionmaker[AsyncSession]
    ],
) -> None:
    repository, factory = lifecycle_store
    await _document(
        factory,
        document_id="document-active-without-pointer",
        pointer=None,
        status=DocumentStatus.ACTIVE,
    )
    for version_id, version_no in (("version-active-1", 1), ("version-active-2", 2)):
        await _version(
            factory,
            version_id=version_id,
            document_id="document-active-without-pointer",
            version_no=version_no,
            status=DocumentVersionStatus.ACTIVE,
            manifest=False,
        )

    mismatches = await repository.list_pointer_mismatches(10)
    assert [item.action for item in mismatches] == ["deactivate", "deactivate"]
    for mismatch in mismatches:
        await repository.resolve_deactivated_version(
            mismatch.context.document_version_id
        )

    async with factory() as session:
        document = (
            await session.execute(
                select(documents.c.status, documents.c.active_version_id).where(
                    documents.c.id == "document-active-without-pointer"
                )
            )
        ).one()
        statuses = (
            (
                await session.execute(
                    select(document_versions.c.status)
                    .where(
                        document_versions.c.document_id
                        == "document-active-without-pointer"
                    )
                    .order_by(document_versions.c.version_no)
                )
            )
            .scalars()
            .all()
        )
    assert document.status == DocumentStatus.FAILED.value
    assert document.active_version_id is None
    assert statuses == [
        DocumentVersionStatus.INACTIVE.value,
        DocumentVersionStatus.INACTIVE.value,
    ]


async def test_running_job_with_missing_expiry_is_reclaimed(
    lifecycle_store: tuple[
        SqlAlchemyReconciliationRepository, async_sessionmaker[AsyncSession]
    ],
) -> None:
    repository, factory = lifecycle_store
    now = datetime.now(UTC)
    await _document(factory, document_id="document-invalid-lease", pointer=None)
    await _version(
        factory,
        version_id="version-invalid-lease",
        document_id="document-invalid-lease",
        version_no=1,
        status=DocumentVersionStatus.BUILDING,
    )
    async with factory.begin() as session:
        await session.execute(
            insert(ingestion_jobs).values(
                id="job-invalid-lease",
                user_id="user-1",
                document_id="document-invalid-lease",
                document_version_id="version-invalid-lease",
                status=JobStatus.RUNNING.value,
                lease_owner="worker-1",
                lease_expires_at=None,
                heartbeat_at=now,
                attempt_count=1,
                created_at=now,
                updated_at=now,
            )
        )
        await session.execute(
            insert(task_outbox).values(
                id="outbox-invalid-lease",
                aggregate_type="ingestion_job",
                aggregate_id="job-invalid-lease",
                stream_name="jobs",
                status="dispatched",
                attempt_count=0,
                next_attempt_at=now,
                created_at=now,
                dispatched_at=now,
            )
        )

    assert await repository.reclaim_expired_jobs(10) == ("job-invalid-lease",)
    async with factory() as session:
        job = (
            (
                await session.execute(
                    select(ingestion_jobs).where(
                        ingestion_jobs.c.id == "job-invalid-lease"
                    )
                )
            )
            .mappings()
            .one()
        )
    assert job["status"] == JobStatus.QUEUED.value
    assert job["lease_owner"] is None
    assert job["attempt_count"] == 2


async def test_third_expired_lease_fails_lifecycle_and_records_dlq_intent(
    lifecycle_store: tuple[
        SqlAlchemyReconciliationRepository, async_sessionmaker[AsyncSession]
    ],
) -> None:
    repository, factory = lifecycle_store
    now = datetime.now(UTC)
    await _document(factory, document_id="document-expired-third", pointer=None)
    await _version(
        factory,
        version_id="version-expired-third",
        document_id="document-expired-third",
        version_no=1,
        status=DocumentVersionStatus.BUILDING,
    )
    async with factory.begin() as session:
        await session.execute(
            insert(ingestion_jobs).values(
                id="job-expired-third",
                user_id="user-1",
                document_id="document-expired-third",
                document_version_id="version-expired-third",
                status=JobStatus.RUNNING.value,
                lease_owner="dead-worker",
                lease_expires_at=now - timedelta(seconds=1),
                heartbeat_at=now - timedelta(minutes=1),
                attempt_count=2,
                created_at=now,
                updated_at=now,
            )
        )
        await session.execute(
            insert(task_outbox).values(
                id="outbox-expired-third",
                aggregate_type="ingestion_job",
                aggregate_id="job-expired-third",
                stream_name="agenticrag:jobs:ingestion",
                status="dispatched",
                attempt_count=2,
                next_attempt_at=now,
                created_at=now,
                dispatched_at=now,
            )
        )

    assert await repository.reclaim_expired_jobs(10) == ("job-expired-third",)
    async with factory() as session:
        job = (
            (
                await session.execute(
                    select(ingestion_jobs).where(
                        ingestion_jobs.c.id == "job-expired-third"
                    )
                )
            )
            .mappings()
            .one()
        )
        version_status = await session.scalar(
            select(document_versions.c.status).where(
                document_versions.c.id == "version-expired-third"
            )
        )
        document_status = await session.scalar(
            select(documents.c.status).where(documents.c.id == "document-expired-third")
        )
        outbox = (
            (
                await session.execute(
                    select(task_outbox).where(
                        task_outbox.c.id == "outbox-expired-third"
                    )
                )
            )
            .mappings()
            .one()
        )
    assert job["status"] == JobStatus.FAILED.value
    assert job["attempt_count"] == 3
    assert job["dead_letter_status"] == "pending"
    assert outbox["status"] == "pending"
    assert outbox["attempt_count"] == 3
    assert version_status == DocumentVersionStatus.FAILED.value
    assert document_status == DocumentStatus.FAILED.value


async def test_deleted_document_with_only_inactive_versions_remains_pending(
    lifecycle_store: tuple[
        SqlAlchemyReconciliationRepository, async_sessionmaker[AsyncSession]
    ],
) -> None:
    repository, factory = lifecycle_store
    await _document(
        factory,
        document_id="document-inactive-delete",
        pointer=None,
        status=DocumentStatus.DELETED,
        deletion_status="fenced",
        deletion_fenced_at=datetime.now(UTC) - timedelta(hours=1),
    )
    await _version(
        factory,
        version_id="version-inactive-delete",
        document_id="document-inactive-delete",
        version_no=1,
        status=DocumentVersionStatus.INACTIVE,
    )

    deleted = await repository.list_deleted_documents(10)

    assert len(deleted) == 1
    assert deleted[0].document_id == "document-inactive-delete"
    assert deleted[0].user_id == "user-1"
    assert [item.document_version_id for item in deleted[0].versions] == [
        "version-inactive-delete"
    ]


async def test_completed_deleted_document_remains_eligible_for_idempotent_resweep(
    lifecycle_store: tuple[
        SqlAlchemyReconciliationRepository, async_sessionmaker[AsyncSession]
    ],
) -> None:
    repository, factory = lifecycle_store
    await _document(
        factory,
        document_id="document-completed-delete",
        pointer=None,
        status=DocumentStatus.DELETED,
        deletion_status="completed",
        deletion_fenced_at=datetime.now(UTC) - timedelta(hours=1),
    )
    await _version(
        factory,
        version_id="version-completed-delete",
        document_id="document-completed-delete",
        version_no=1,
        status=DocumentVersionStatus.INACTIVE,
    )

    deleted = await repository.list_deleted_documents(10)

    assert [item.document_id for item in deleted] == ["document-completed-delete"]
    assert (
        await repository.mark_deletion_reconciled("document-completed-delete") is False
    )


async def test_deletion_completion_terminates_running_worker_lease(
    lifecycle_store: tuple[
        SqlAlchemyReconciliationRepository, async_sessionmaker[AsyncSession]
    ],
) -> None:
    repository, factory = lifecycle_store
    now = datetime.now(UTC)
    await _document(
        factory,
        document_id="document-live-delete",
        pointer=None,
        status=DocumentStatus.DELETED,
        deletion_status="pending",
    )
    await _version(
        factory,
        version_id="version-live-delete",
        document_id="document-live-delete",
        version_no=1,
        status=DocumentVersionStatus.BUILDING,
    )
    async with factory.begin() as session:
        await session.execute(
            insert(ingestion_jobs).values(
                id="job-live-delete",
                user_id="user-1",
                document_id="document-live-delete",
                document_version_id="version-live-delete",
                status=JobStatus.RUNNING.value,
                lease_owner="worker-1",
                lease_expires_at=now + timedelta(minutes=5),
                heartbeat_at=now,
                attempt_count=1,
                created_at=now,
                updated_at=now,
            )
        )

    assert await repository.list_deleted_documents(10) == ()
    assert await repository.fence_pending_deletions(10) == ("document-live-delete",)
    async with factory() as session:
        job = (
            (
                await session.execute(
                    select(ingestion_jobs).where(
                        ingestion_jobs.c.id == "job-live-delete"
                    )
                )
            )
            .mappings()
            .one()
        )
        deletion_status = await session.scalar(
            select(documents.c.deletion_status).where(
                documents.c.id == "document-live-delete"
            )
        )
    assert job["status"] == JobStatus.FAILED.value
    assert job["lease_owner"] is None
    assert job["lease_expires_at"] is None
    assert job["error_code"] == "document_deleted"
    assert deletion_status == "fenced"
    async with factory.begin() as session:
        await session.execute(
            update(documents)
            .where(documents.c.id == "document-live-delete")
            .values(deletion_fenced_at=now - timedelta(hours=1))
        )
    assert [
        item.document_id for item in await repository.list_deleted_documents(10)
    ] == ["document-live-delete"]
    await repository.mark_deletion_reconciled("document-live-delete")
    async with factory() as session:
        deletion_status = await session.scalar(
            select(documents.c.deletion_status).where(
                documents.c.id == "document-live-delete"
            )
        )
    assert deletion_status == "completed"
