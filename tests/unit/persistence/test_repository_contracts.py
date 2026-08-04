"""Unit contracts for the MySQL persistence boundary."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, cast

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from agentic_rag.domain.models import JobStatus, RunStatus, UserScope
from agentic_rag.persistence.mysql import create_mysql_engine, create_session_factory
from agentic_rag.persistence.repositories import (
    DocumentRepository,
    EventRepository,
    IngestionJobRepository,
    MemoryTombstoneRepository,
    OutboxRepository,
    ParentRepository,
    RunRepository,
    SqlAlchemyDocumentRepository,
    SqlAlchemyEventRepository,
    SqlAlchemyIngestionJobRepository,
    SqlAlchemyMemoryTombstoneRepository,
    SqlAlchemyOutboxRepository,
    SqlAlchemyParentRepository,
    SqlAlchemyRunRepository,
    active_slot_for_status,
)
from agentic_rag.runtime.models import RuntimeConfigSnapshot


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


class RecordingSession:
    """Small AsyncSession stand-in that records SQLAlchemy statements."""

    def __init__(self, rows: Sequence[dict[str, Any]] = ()) -> None:
        self.statements: list[Any] = []
        self._rows = rows
        self.commit_called = False

    async def execute(self, statement: Any) -> Any:
        self.statements.append(statement)
        return RecordingResult(self._rows)

    async def commit(self) -> None:
        self.commit_called = True


class RecordingResult:
    def __init__(self, rows: Sequence[dict[str, Any]]) -> None:
        self._rows = rows

    def mappings(self) -> "RecordingResult":
        return self

    def all(self) -> Sequence[dict[str, Any]]:
        return self._rows


def _statement_tables(session: RecordingSession) -> list[str]:
    return [statement.table.name for statement in session.statements]


@pytest.mark.asyncio
async def test_async_mysql_engine_and_session_factory_use_asyncmy() -> None:
    """A sync driver must not leak across the async persistence boundary."""
    engine = create_mysql_engine("mysql+asyncmy://rag:secret@127.0.0.1:3306/rag")
    factory = create_session_factory(engine)

    try:
        assert engine.url.drivername == "mysql+asyncmy"
        assert factory.class_ is AsyncSession
        assert factory.kw["expire_on_commit"] is False
    finally:
        await engine.dispose()


def test_all_repository_adapters_satisfy_their_runtime_protocols() -> None:
    """Removing a required repository operation breaks the public port."""
    transaction = cast(AsyncSession, RecordingSession())

    assert isinstance(SqlAlchemyRunRepository(transaction), RunRepository)
    assert isinstance(
        SqlAlchemyIngestionJobRepository(transaction), IngestionJobRepository
    )
    assert isinstance(SqlAlchemyDocumentRepository(transaction), DocumentRepository)
    assert isinstance(SqlAlchemyParentRepository(transaction), ParentRepository)
    assert isinstance(SqlAlchemyEventRepository(transaction), EventRepository)
    assert isinstance(SqlAlchemyOutboxRepository(transaction), OutboxRepository)
    assert isinstance(
        SqlAlchemyMemoryTombstoneRepository(transaction), MemoryTombstoneRepository
    )


@pytest.mark.asyncio
async def test_run_creation_stages_run_and_matching_outbox_in_same_transaction() -> (
    None
):
    """Dropping the outbox insert would make a committed run undispatchable."""
    transaction = RecordingSession()
    repository = SqlAlchemyRunRepository()

    run = await repository.create_queued(
        scope=UserScope(user_id="user-1"),
        thread_id="thread-1",
        snapshot=SNAPSHOT,
        transaction=cast(AsyncSession, transaction),
    )

    assert run.status is RunStatus.QUEUED
    assert run.active_slot == 1
    assert _statement_tables(transaction) == ["agent_runs", "task_outbox"]
    assert transaction.commit_called is False
    outbox_values = transaction.statements[1].compile().params
    assert outbox_values["aggregate_type"] == "query_run"
    assert outbox_values["aggregate_id"] == run.id


@pytest.mark.asyncio
async def test_job_creation_stages_job_and_matching_outbox_in_same_transaction() -> (
    None
):
    """Dropping the outbox insert would strand a queued ingestion job."""
    transaction = RecordingSession()
    repository = SqlAlchemyIngestionJobRepository()

    job = await repository.create_queued(
        scope=UserScope(user_id="user-1"),
        document_id="document-1",
        document_version_id="version-1",
        transaction=cast(AsyncSession, transaction),
    )

    assert job.status is JobStatus.QUEUED
    assert _statement_tables(transaction) == ["ingestion_jobs", "task_outbox"]
    assert transaction.commit_called is False
    outbox_values = transaction.statements[1].compile().params
    assert outbox_values["aggregate_type"] == "ingestion_job"
    assert outbox_values["aggregate_id"] == job.id


@pytest.mark.parametrize(
    ("status", "expected"),
    [
        (RunStatus.QUEUED, 1),
        (RunStatus.RUNNING, 1),
        (RunStatus.CANCEL_REQUESTED, 1),
        (RunStatus.CANCELLED, None),
        (RunStatus.COMPLETED, None),
        (RunStatus.FAILED, None),
    ],
)
def test_active_slot_tracks_only_non_terminal_run_states(
    status: RunStatus, expected: int | None
) -> None:
    """A terminal run retaining slot 1 would block all later thread runs."""
    assert active_slot_for_status(status) == expected
