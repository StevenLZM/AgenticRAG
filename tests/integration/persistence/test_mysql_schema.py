"""MySQL migration and repository integration tests.

Set ``AGENTIC_RAG_TEST_MYSQL_DSN`` to a disposable MySQL database. The module
is skipped when the explicit integration DSN is absent; it never guesses at or
mutates a developer database.
"""

from __future__ import annotations

import asyncio
import os
from collections.abc import AsyncIterator, Iterator
from datetime import UTC, datetime, timedelta, timezone

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import inspect, insert, select, update
from sqlalchemy.exc import IntegrityError, SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from agentic_rag.domain.models import RunStatus, UserScope
from agentic_rag.persistence.mysql import create_mysql_engine, create_session_factory
from agentic_rag.persistence import repositories
from agentic_rag.persistence.repositories import (
    ActiveRunConflict,
    AgentEvent,
    EventKeyConflict,
    LeaseLost,
    SqlAlchemyDocumentRepository,
    SqlAlchemyEventRepository,
    SqlAlchemyIngestionJobRepository,
    SqlAlchemyRunRepository,
    agent_runs,
    ingestion_jobs,
    task_outbox,
)
from agentic_rag.runtime.models import RuntimeConfigSnapshot


pytestmark = pytest.mark.integration

EXPECTED_TABLES = {
    "documents",
    "document_versions",
    "parent_chunks",
    "ingestion_jobs",
    "agent_runs",
    "task_outbox",
    "messages",
    "agent_events",
    "memory_tombstones",
}

EXPECTED_INDEXES = {
    "documents": {
        "ix_documents_user_status": ("user_id", "status"),
        "ix_documents_user_content_hash": ("user_id", "content_hash"),
    },
    "document_versions": {"ix_versions_document_status": ("document_id", "status")},
    "parent_chunks": {
        "ix_parents_user_status": ("user_id", "status"),
        "ix_parents_user_document": ("user_id", "document_id"),
    },
    "ingestion_jobs": {
        "ix_jobs_status_lease": ("status", "lease_expires_at"),
        "ix_jobs_user_created": ("user_id", "created_at"),
    },
    "agent_runs": {
        "ix_runs_status_lease": ("status", "lease_expires_at"),
        "ix_runs_user_thread_created": ("user_id", "thread_id", "created_at"),
    },
    "task_outbox": {"ix_outbox_pending": ("status", "next_attempt_at", "id")},
    "messages": {
        "ix_messages_user_thread_created": ("user_id", "thread_id", "created_at")
    },
    "agent_events": {"ix_events_user_run_id": ("user_id", "run_id", "id")},
    "memory_tombstones": {"ix_tombstones_status_requested": ("status", "requested_at")},
}

EXPECTED_FOREIGN_KEYS = {
    "documents": {
        "fk_documents_active_version": (
            ("active_version_id",),
            "document_versions",
            ("id",),
            "SET NULL",
        )
    },
    "document_versions": {
        "fk_versions_document": (("document_id",), "documents", ("id",), "CASCADE")
    },
    "parent_chunks": {
        "fk_parents_document": (("document_id",), "documents", ("id",), "CASCADE"),
        "fk_parents_version": (
            ("document_version_id",),
            "document_versions",
            ("id",),
            "CASCADE",
        ),
    },
    "ingestion_jobs": {
        "fk_jobs_document": (("document_id",), "documents", ("id",), "CASCADE"),
        "fk_jobs_version": (
            ("document_version_id",),
            "document_versions",
            ("id",),
            "CASCADE",
        ),
    },
    "agent_runs": {},
    "task_outbox": {},
    "messages": {"fk_messages_run": (("run_id",), "agent_runs", ("id",), "CASCADE")},
    "agent_events": {"fk_events_run": (("run_id",), "agent_runs", ("id",), "CASCADE")},
    "memory_tombstones": {},
}

SNAPSHOT = RuntimeConfigSnapshot(
    app_version="0.1.0",
    graph_version="graph-v1",
    prompt_version="prompt-v1",
    main_model_id="deepseek-v4-pro",
    light_model_id="deepseek-v4-flash",
    embedding_model="text-embedding-v3",
    embedding_dimensions=1024,
    reranker_version="bge-reranker-v2-m3",
    retrieval_config_version="retrieval-v1",
    index_generation="index-v1",
    memory_config_version="memory-v1",
)


@pytest.fixture(scope="module", autouse=True)
def migrated_schema(mysql_dsn: str) -> Iterator[None]:
    config = Config("alembic.ini")
    config.set_main_option("sqlalchemy.url", mysql_dsn)
    command.upgrade(config, "head")
    command.downgrade(config, "base")
    command.upgrade(config, "head")
    yield
    command.downgrade(config, "base")
    command.upgrade(config, "head")


@pytest.fixture(scope="module")
def mysql_dsn() -> str:
    dsn = os.getenv("AGENTIC_RAG_TEST_MYSQL_DSN")
    if not dsn:
        pytest.skip("set AGENTIC_RAG_TEST_MYSQL_DSN to a disposable MySQL database")
    engine = create_mysql_engine(dsn, pool_pre_ping=True)

    async def probe() -> None:
        try:
            async with engine.connect():
                pass
        finally:
            await engine.dispose()

    try:
        asyncio.run(probe())
    except (OSError, RuntimeError, SQLAlchemyError) as error:
        pytest.skip(f"disposable MySQL database is unavailable: {type(error).__name__}")
    return dsn


@pytest.fixture(scope="module")
async def session_factory(
    mysql_dsn: str, migrated_schema: None
) -> AsyncIterator[async_sessionmaker[AsyncSession]]:
    engine = create_mysql_engine(mysql_dsn, pool_pre_ping=True)
    factory = create_session_factory(engine)
    try:
        yield factory
    finally:
        await engine.dispose()


@pytest.mark.asyncio
async def test_migration_creates_exact_schema_with_required_keys(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    async with session_factory() as session:
        schema = await session.connection()

        def inspect_schema(sync_connection):
            inspector = inspect(sync_connection)
            tables = set(inspector.get_table_names()) - {"alembic_version"}
            run_uniques = {
                tuple(item["column_names"])
                for item in inspector.get_unique_constraints("agent_runs")
            }
            event_uniques = {
                tuple(item["column_names"])
                for item in inspector.get_unique_constraints("agent_events")
            }
            indexes = {
                table: {
                    item["name"]: tuple(item["column_names"])
                    for item in inspector.get_indexes(table)
                    if not item["unique"]
                }
                for table in EXPECTED_TABLES
            }
            foreign_keys = {
                table: {
                    item["name"]: (
                        tuple(item["constrained_columns"]),
                        item["referred_table"],
                        tuple(item["referred_columns"]),
                        item.get("options", {}).get("ondelete"),
                    )
                    for item in inspector.get_foreign_keys(table)
                }
                for table in EXPECTED_TABLES
            }
            return tables, run_uniques, event_uniques, indexes, foreign_keys

        (
            tables,
            run_uniques,
            event_uniques,
            indexes,
            foreign_keys,
        ) = await schema.run_sync(inspect_schema)

    assert tables == EXPECTED_TABLES
    assert ("user_id", "thread_id", "active_slot") in run_uniques
    assert ("event_key",) in event_uniques
    required_indexes = {
        table: {name: indexes[table].get(name) for name in definitions}
        for table, definitions in EXPECTED_INDEXES.items()
    }
    assert required_indexes == EXPECTED_INDEXES
    assert foreign_keys == EXPECTED_FOREIGN_KEYS


@pytest.mark.asyncio
async def test_only_one_active_run_per_user_thread(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    scope = UserScope(user_id="active-slot-user")
    async with session_factory.begin() as transaction:
        repository = SqlAlchemyRunRepository(transaction)
        first = await repository.create_queued(scope, "thread-1", SNAPSHOT)

    with pytest.raises(ActiveRunConflict):
        async with session_factory.begin() as transaction:
            repository = SqlAlchemyRunRepository(transaction)
            await repository.create_queued(scope, "thread-1", SNAPSHOT)

    async with session_factory.begin() as transaction:
        repository = SqlAlchemyRunRepository(transaction)
        claimed = await repository.claim(first.id, "worker:first", lease_seconds=30)
        assert claimed is not None
        await repository.finish(
            first.id,
            RunStatus.COMPLETED,
            result_ref="artifact://answer",
            error_code=None,
            owner="worker:first",
            claim_generation=claimed.claim_generation,
        )
        second = await repository.create_queued(scope, "thread-1", SNAPSHOT)

    assert second.id != first.id


@pytest.mark.asyncio
async def test_scoped_run_read_cannot_cross_user_boundary(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    owner = UserScope(user_id="owner-user")
    async with session_factory.begin() as transaction:
        repository = SqlAlchemyRunRepository(transaction)
        run = await repository.create_queued(owner, "private-thread", SNAPSHOT)

    async with session_factory() as transaction:
        repository = SqlAlchemyRunRepository(transaction)
        assert await repository.get(run.id, owner) is not None
        assert await repository.get(run.id, UserScope(user_id="other-user")) is None


@pytest.mark.asyncio
async def test_queued_cancel_finalizes_and_allows_next_run(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    scope = UserScope(user_id="queued-cancel-user")
    async with session_factory.begin() as transaction:
        repository = SqlAlchemyRunRepository(transaction)
        first = await repository.create_queued(scope, "cancel-thread", SNAPSHOT)
        assert await repository.request_cancel(first.id, scope) is RunStatus.CANCELLED
        second = await repository.create_queued(scope, "cancel-thread", SNAPSHOT)

    assert second.id != first.id


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("status", "active_slot"),
    [(RunStatus.QUEUED, None), (RunStatus.COMPLETED, 1)],
)
async def test_database_rejects_status_active_slot_mismatch(
    session_factory: async_sessionmaker[AsyncSession],
    status: RunStatus,
    active_slot: int | None,
) -> None:
    """MySQL itself protects the exclusivity invariant in both directions."""
    with pytest.raises(IntegrityError):
        async with session_factory.begin() as transaction:
            await transaction.execute(
                insert(agent_runs).values(
                    id=f"invalid-slot-{status.value}",
                    user_id="invalid-slot-user",
                    thread_id=f"thread-{status.value}",
                    checkpoint_thread_id=f"query:invalid:{status.value}",
                    status=status.value,
                    active_slot=active_slot,
                    attempt_count=0,
                    runtime_config_snapshot_id=SNAPSHOT.snapshot_id,
                    runtime_config_snapshot=SNAPSHOT.model_dump(mode="json"),
                    created_at=datetime.now(UTC),
                )
            )


@pytest.mark.asyncio
async def test_claim_is_single_winner_and_cancel_requested_is_reclaimable(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    scope = UserScope(user_id="claim-user")
    async with session_factory.begin() as transaction:
        repository = SqlAlchemyRunRepository(transaction)
        run = await repository.create_queued(scope, "claim-thread", SNAPSHOT)
        first = await repository.claim(run.id, "shared-owner", lease_seconds=30)
        assert first is not None

    async with session_factory.begin() as transaction:
        repository = SqlAlchemyRunRepository(transaction)
        assert await repository.claim(run.id, "shared-owner", lease_seconds=30) is None
        assert (
            await repository.request_cancel(run.id, scope) is RunStatus.CANCEL_REQUESTED
        )
        await transaction.execute(
            update(agent_runs)
            .where(agent_runs.c.id == run.id)
            .values(lease_expires_at=datetime.now(UTC) - timedelta(seconds=1))
        )

    async with session_factory.begin() as transaction:
        repository = SqlAlchemyRunRepository(transaction)
        reclaimed = await repository.claim(run.id, "worker-2", lease_seconds=30)
        assert reclaimed is not None
        assert reclaimed.status is RunStatus.CANCEL_REQUESTED
        assert reclaimed.claim_generation == first.claim_generation + 1
        await repository.finish(
            run.id,
            RunStatus.CANCELLED,
            result_ref=None,
            error_code=None,
            owner="worker-2",
            claim_generation=reclaimed.claim_generation,
        )


@pytest.mark.asyncio
async def test_expired_heartbeat_and_stale_finish_are_fenced(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    scope = UserScope(user_id="fence-user")
    async with session_factory.begin() as transaction:
        repository = SqlAlchemyRunRepository(transaction)
        run = await repository.create_queued(scope, "fence-thread", SNAPSHOT)
        first = await repository.claim(run.id, "worker-1", lease_seconds=30)
        assert first is not None
        await transaction.execute(
            update(agent_runs)
            .where(agent_runs.c.id == run.id)
            .values(lease_expires_at=datetime.now(UTC) - timedelta(seconds=1))
        )
        with pytest.raises(LeaseLost):
            await repository.heartbeat(
                run.id,
                "worker-1",
                30,
                claim_generation=first.claim_generation,
            )

    async with session_factory.begin() as transaction:
        repository = SqlAlchemyRunRepository(transaction)
        second = await repository.claim(run.id, "worker-2", lease_seconds=30)
        assert second is not None
        with pytest.raises(LeaseLost):
            await repository.finish(
                run.id,
                RunStatus.COMPLETED,
                result_ref="artifact://stale",
                error_code=None,
                owner="worker-1",
                claim_generation=first.claim_generation,
            )
        await repository.finish(
            run.id,
            RunStatus.COMPLETED,
            result_ref="artifact://fresh",
            error_code=None,
            owner="worker-2",
            claim_generation=second.claim_generation,
        )

    async with session_factory.begin() as transaction:
        assert (
            await SqlAlchemyRunRepository(transaction).request_cancel(run.id, scope)
            is RunStatus.COMPLETED
        )


@pytest.mark.asyncio
async def test_event_append_is_idempotent_and_rejects_key_conflicts(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    scope = UserScope(user_id="event-user")
    async with session_factory.begin() as transaction:
        run = await SqlAlchemyRunRepository(transaction).create_queued(
            scope, "event-thread", SNAPSHOT
        )
        repository = SqlAlchemyEventRepository(transaction)
        event = AgentEvent(
            event_key="event-key:integration",
            trace_id="trace-1",
            run_id=run.id,
            user_id=scope.user_id,
            event_type="RETRIEVAL_COMPLETED",
            summary="retrieval complete",
            runtime_config_snapshot_id=SNAPSHOT.snapshot_id,
        )
        first_id = await repository.append(event)
        assert await repository.append(event) == first_id
        with pytest.raises(EventKeyConflict):
            await repository.append(
                AgentEvent(
                    event_key=event.event_key,
                    trace_id=event.trace_id,
                    run_id=event.run_id,
                    user_id=event.user_id,
                    event_type=event.event_type,
                    summary="conflicting summary",
                    runtime_config_snapshot_id=event.runtime_config_snapshot_id,
                )
            )
        explicit_at = datetime(2026, 8, 5, 10, 0, tzinfo=timezone(timedelta(hours=8)))
        timestamped = AgentEvent(
            event_key="event-key:timestamped",
            trace_id="trace-1",
            run_id=run.id,
            user_id=scope.user_id,
            event_type="TODO_UPDATED",
            summary="todo updated",
            runtime_config_snapshot_id=SNAPSHOT.snapshot_id,
            created_at=explicit_at,
        )
        timestamped_id = await repository.append(timestamped)
        assert (
            await repository.append(
                AgentEvent(
                    event_key=timestamped.event_key,
                    trace_id=timestamped.trace_id,
                    run_id=timestamped.run_id,
                    user_id=timestamped.user_id,
                    event_type=timestamped.event_type,
                    summary=timestamped.summary,
                    runtime_config_snapshot_id=(timestamped.runtime_config_snapshot_id),
                    created_at=datetime(2026, 8, 5, 2, 0, tzinfo=UTC),
                )
            )
            == timestamped_id
        )
        with pytest.raises(EventKeyConflict):
            await repository.append(
                AgentEvent(
                    event_key=timestamped.event_key,
                    trace_id=timestamped.trace_id,
                    run_id=timestamped.run_id,
                    user_id=timestamped.user_id,
                    event_type=timestamped.event_type,
                    summary=timestamped.summary,
                    runtime_config_snapshot_id=(timestamped.runtime_config_snapshot_id),
                    created_at=explicit_at + timedelta(seconds=1),
                )
            )


@pytest.mark.asyncio
async def test_outbox_insert_failure_rolls_back_run(
    session_factory: async_sessionmaker[AsyncSession], monkeypatch: pytest.MonkeyPatch
) -> None:
    duplicate_outbox_id = "00000000-0000-7000-8000-000000000001"
    run_id = "00000000-0000-7000-8000-000000000002"
    now = datetime.now(UTC)
    async with session_factory.begin() as transaction:
        await transaction.execute(
            insert(task_outbox).values(
                id=duplicate_outbox_id,
                aggregate_type="query_run",
                aggregate_id="existing-run",
                stream_name="agenticrag:jobs:query",
                status="pending",
                attempt_count=0,
                next_attempt_at=now,
                created_at=now,
            )
        )

    ids = iter([run_id, duplicate_outbox_id])
    monkeypatch.setattr(repositories, "new_id", lambda: next(ids))
    with pytest.raises(IntegrityError):
        async with session_factory.begin() as transaction:
            await SqlAlchemyRunRepository(transaction).create_queued(
                UserScope(user_id="rollback-user"), "rollback-thread", SNAPSHOT
            )

    async with session_factory() as transaction:
        assert (
            await transaction.execute(
                select(agent_runs.c.id).where(agent_runs.c.id == run_id)
            )
        ).one_or_none() is None


@pytest.mark.asyncio
async def test_outbox_insert_failure_rolls_back_ingestion_job(
    session_factory: async_sessionmaker[AsyncSession], monkeypatch: pytest.MonkeyPatch
) -> None:
    scope = UserScope(user_id="job-rollback-user")
    duplicate_outbox_id = "00000000-0000-7000-8000-000000000003"
    job_id = "00000000-0000-7000-8000-000000000004"
    now = datetime.now(UTC)
    async with session_factory.begin() as transaction:
        document, version = await SqlAlchemyDocumentRepository(transaction).create(
            scope,
            source_type="text",
            filename="rollback.txt",
            mime_type="text/plain",
            content_hash="b" * 64,
            parser_version="parser-v1",
            pipeline_version="pipeline-v1",
            embedding_version="embedding-v1",
            index_generation="index-v1",
        )
        await transaction.execute(
            insert(task_outbox).values(
                id=duplicate_outbox_id,
                aggregate_type="ingestion_job",
                aggregate_id="existing-job",
                stream_name="agenticrag:jobs:ingestion",
                status="pending",
                attempt_count=0,
                next_attempt_at=now,
                created_at=now,
            )
        )

    ids = iter([job_id, duplicate_outbox_id])
    monkeypatch.setattr(repositories, "new_id", lambda: next(ids))
    with pytest.raises(IntegrityError):
        async with session_factory.begin() as transaction:
            await SqlAlchemyIngestionJobRepository(transaction).create_queued(
                scope, document.id, version.id
            )

    async with session_factory() as transaction:
        assert (
            await transaction.execute(
                select(ingestion_jobs.c.id).where(ingestion_jobs.c.id == job_id)
            )
        ).one_or_none() is None
        assert (
            await transaction.execute(
                select(task_outbox.c.id).where(task_outbox.c.aggregate_id == job_id)
            )
        ).one_or_none() is None


@pytest.mark.asyncio
async def test_run_and_job_creation_commit_matching_outbox_rows(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    scope = UserScope(user_id="atomic-user")
    async with session_factory.begin() as transaction:
        document, version = await SqlAlchemyDocumentRepository(transaction).create(
            scope,
            source_type="text",
            filename="notes.txt",
            mime_type="text/plain",
            content_hash="a" * 64,
            parser_version="parser-v1",
            pipeline_version="pipeline-v1",
            embedding_version="embedding-v1",
            index_generation="index-v1",
        )
        run = await SqlAlchemyRunRepository(transaction).create_queued(
            scope, "atomic-thread", SNAPSHOT
        )
        job = await SqlAlchemyIngestionJobRepository(transaction).create_queued(
            scope, document.id, version.id
        )

    async with session_factory() as transaction:
        run_row = (
            (
                await transaction.execute(
                    select(agent_runs).where(agent_runs.c.id == run.id)
                )
            )
            .mappings()
            .one()
        )
        job_row = (
            (
                await transaction.execute(
                    select(ingestion_jobs).where(ingestion_jobs.c.id == job.id)
                )
            )
            .mappings()
            .one()
        )
        outbox_rows = (
            (
                await transaction.execute(
                    select(task_outbox).where(
                        task_outbox.c.aggregate_id.in_([run.id, job.id])
                    )
                )
            )
            .mappings()
            .all()
        )

    assert run_row["status"] == "queued"
    assert job_row["status"] == "queued"
    assert {(row["aggregate_type"], row["aggregate_id"]) for row in outbox_rows} == {
        ("query_run", run.id),
        ("ingestion_job", job.id),
    }
