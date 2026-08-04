"""MySQL migration and repository integration tests.

Set ``AGENTIC_RAG_TEST_MYSQL_DSN`` to a disposable MySQL database. The module
is skipped when the explicit integration DSN is absent; it never guesses at or
mutates a developer database.
"""

from __future__ import annotations

import asyncio
import os
from collections.abc import AsyncIterator, Iterator

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import inspect, select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from agentic_rag.domain.models import UserScope
from agentic_rag.persistence.mysql import create_mysql_engine, create_session_factory
from agentic_rag.persistence.repositories import (
    ActiveRunConflict,
    SqlAlchemyDocumentRepository,
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
            foreign_keys = {
                table: inspector.get_foreign_keys(table) for table in EXPECTED_TABLES
            }
            return tables, run_uniques, event_uniques, foreign_keys

        tables, run_uniques, event_uniques, foreign_keys = await schema.run_sync(
            inspect_schema
        )

    assert tables == EXPECTED_TABLES
    assert ("user_id", "thread_id", "active_slot") in run_uniques
    assert ("event_key",) in event_uniques
    assert foreign_keys["document_versions"]
    assert foreign_keys["parent_chunks"]
    assert foreign_keys["ingestion_jobs"]
    assert foreign_keys["messages"]
    assert foreign_keys["agent_events"]


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
        await repository.mark_completed(first.id, result_ref="artifact://answer")
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
