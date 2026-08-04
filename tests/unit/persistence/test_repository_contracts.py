"""Unit contracts for the MySQL persistence boundary."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime, timedelta, timezone
from typing import Any, cast

import pytest
from sqlalchemy import CheckConstraint
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from agentic_rag.domain.models import JobStatus, RunStatus, UserScope
from agentic_rag.persistence.mysql import create_mysql_engine, create_session_factory
from agentic_rag.persistence.repositories import (
    DocumentRepository,
    EventKeyConflict,
    EventRepository,
    IngestionJobRepository,
    MemoryTombstoneRepository,
    LeaseLost,
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
    agent_runs,
    AgentEvent,
    ActiveRunConflict,
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

    def __init__(
        self,
        rows: Sequence[dict[str, Any]] = (),
        scripted: Sequence[Any] = (),
    ) -> None:
        self.statements: list[Any] = []
        self._rows = rows
        self._scripted = list(scripted)
        self.commit_called = False

    async def execute(self, statement: Any) -> Any:
        self.statements.append(statement)
        if self._scripted:
            outcome = self._scripted.pop(0)
            if isinstance(outcome, BaseException):
                raise outcome
            return outcome
        return RecordingResult(self._rows)

    async def commit(self) -> None:
        self.commit_called = True


class RecordingResult:
    def __init__(
        self, rows: Sequence[Any] = (), *, rowcount: int = 1, lastrowid: int = 1
    ) -> None:
        self._rows = rows
        self.rowcount = rowcount
        self.lastrowid = lastrowid

    def mappings(self) -> "RecordingResult":
        return self

    def all(self) -> Sequence[dict[str, Any]]:
        return self._rows

    def one_or_none(self) -> Any | None:
        return self._rows[0] if self._rows else None

    def one(self) -> Any:
        return self._rows[0]


def _statement_tables(session: RecordingSession) -> list[str]:
    return [statement.table.name for statement in session.statements]


def _run_row(
    *, status: RunStatus, attempt_count: int, owner: str | None = None
) -> dict[str, Any]:
    return {
        "id": "run-1",
        "user_id": "user-1",
        "thread_id": "thread-1",
        "checkpoint_thread_id": "query:user-1:thread-1",
        "status": status.value,
        "active_slot": active_slot_for_status(status),
        "lease_owner": owner,
        "lease_expires_at": datetime.now(UTC) + timedelta(seconds=30),
        "attempt_count": attempt_count,
        "runtime_config_snapshot_id": SNAPSHOT.snapshot_id,
        "runtime_config_snapshot": SNAPSHOT.model_dump(mode="json"),
        "result_ref": None,
        "error_code": None,
    }


def _event(
    *, summary: str = "retrieval complete", created_at: datetime | None = None
) -> AgentEvent:
    return AgentEvent(
        event_key="event-key-1",
        trace_id="trace-1",
        run_id="run-1",
        user_id="user-1",
        event_type="RETRIEVAL_COMPLETED",
        summary=summary,
        runtime_config_snapshot_id=SNAPSHOT.snapshot_id,
        created_at=created_at,
    )


def _event_row(event: AgentEvent) -> dict[str, Any]:
    return {
        "id": 41,
        "event_key": event.event_key,
        "trace_id": event.trace_id,
        "run_id": event.run_id,
        "user_id": event.user_id,
        "node_name": event.node_name,
        "event_type": event.event_type,
        "summary": event.summary,
        "payload_ref": event.payload_ref,
        "runtime_config_snapshot_id": event.runtime_config_snapshot_id,
        "created_at": event.created_at or datetime.now(UTC),
    }


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


@pytest.mark.asyncio
async def test_outbox_claim_and_retry_leave_transaction_commit_to_the_caller() -> None:
    """Dispatcher coordination must not silently commit its caller's transaction."""
    now = datetime.now(UTC)
    transaction = RecordingSession(
        rows=[
            {
                "id": "outbox-1",
                "aggregate_type": "query_run",
                "aggregate_id": "run-1",
                "stream_name": "agenticrag:jobs:query",
                "status": "pending",
                "attempt_count": 0,
                "next_attempt_at": now,
                "created_at": now,
            }
        ]
    )
    repository = SqlAlchemyOutboxRepository(cast(AsyncSession, transaction))

    records = await repository.claim_pending(limit=1)
    await repository.schedule_retry("outbox-1")

    assert records[0].aggregate_id == "run-1"
    assert records[0].created_at == now
    assert transaction.statements[0].get_final_froms()[0].name == "task_outbox"
    assert transaction.statements[0]._for_update_arg.skip_locked is True
    assert transaction.statements[1].table.name == "task_outbox"
    assert "next_attempt_at" in transaction.statements[1].compile().params
    assert transaction.statements[2].table.name == "task_outbox"
    assert transaction.commit_called is False


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


def test_agent_runs_schema_binds_status_to_its_only_valid_active_slot() -> None:
    """The durable constraint must reject both directions of slot drift."""
    constraint = next(
        item
        for item in agent_runs.constraints
        if isinstance(item, CheckConstraint) and item.name == "ck_active_slot"
    )

    assert str(constraint.sqltext) == (
        "((status IN ('queued','running','cancel_requested') "
        "AND active_slot IS NOT NULL AND active_slot = 1) "
        "OR (status IN ('cancelled','completed','failed') AND active_slot IS NULL))"
    )


@pytest.mark.asyncio
async def test_queued_cancellation_atomically_releases_active_slot() -> None:
    """Cancelling before Claim must not leave an unclaimable active run."""
    transaction = RecordingSession(scripted=[RecordingResult(rowcount=1)])
    repository = SqlAlchemyRunRepository(cast(AsyncSession, transaction))

    status = await repository.request_cancel("run-1", UserScope(user_id="user-1"))

    assert status is RunStatus.CANCELLED
    values = transaction.statements[0].compile().params
    assert values["status"] == RunStatus.CANCELLED.value
    assert values["active_slot"] is None


@pytest.mark.asyncio
async def test_cancel_requested_run_can_be_reclaimed_with_new_generation() -> None:
    """A crashed cancelling worker must not strand the active slot."""
    row = _run_row(status=RunStatus.CANCEL_REQUESTED, attempt_count=2, owner="worker-2")
    transaction = RecordingSession(
        scripted=[RecordingResult(rowcount=1), RecordingResult([row])]
    )
    repository = SqlAlchemyRunRepository(cast(AsyncSession, transaction))

    claimed = await repository.claim("run-1", "worker-2", lease_seconds=30)

    assert claimed is not None
    assert claimed.status is RunStatus.CANCEL_REQUESTED
    assert claimed.claim_generation == 2


@pytest.mark.asyncio
async def test_claim_requires_its_compare_and_set_to_change_one_row() -> None:
    """A competing same-owner Claim cannot succeed from a stale reread."""
    transaction = RecordingSession(scripted=[RecordingResult(rowcount=0)])
    repository = SqlAlchemyRunRepository(cast(AsyncSession, transaction))

    assert await repository.claim("run-1", "worker-1", lease_seconds=30) is None
    assert len(transaction.statements) == 1


@pytest.mark.asyncio
async def test_expired_heartbeat_is_rejected_by_owner_and_generation_fence() -> None:
    """A worker cannot resurrect its lease after it has expired."""
    transaction = RecordingSession(scripted=[RecordingResult(rowcount=0)])
    repository = SqlAlchemyRunRepository(cast(AsyncSession, transaction))

    with pytest.raises(LeaseLost):
        await repository.heartbeat("run-1", "worker-1", 30, claim_generation=1)


@pytest.mark.asyncio
async def test_stale_finish_is_rejected_by_owner_and_generation_fence() -> None:
    """A reclaimed run cannot be overwritten by its previous worker."""
    transaction = RecordingSession(scripted=[RecordingResult(rowcount=0)])
    repository = SqlAlchemyRunRepository(cast(AsyncSession, transaction))

    with pytest.raises(LeaseLost):
        await repository.finish(
            "run-1",
            RunStatus.COMPLETED,
            "artifact://answer",
            None,
            owner="worker-1",
            claim_generation=1,
        )


@pytest.mark.asyncio
async def test_identical_event_replay_returns_existing_cursor() -> None:
    """Checkpoint replay of the same event is repository-idempotent."""
    event = _event()
    transaction = RecordingSession(
        scripted=[
            RecordingResult(),
            RecordingResult([_event_row(event)]),
            RecordingResult(),
            RecordingResult([_event_row(event)]),
        ]
    )
    repository = SqlAlchemyEventRepository(cast(AsyncSession, transaction))

    assert await repository.append(event) == 41
    assert await repository.append(event) == 41
    generated_timestamp = transaction.statements[0].compile().params["created_at"]
    assert generated_timestamp.tzinfo is None


@pytest.mark.asyncio
async def test_event_key_reuse_with_conflicting_payload_is_rejected() -> None:
    """A deterministic key cannot silently alias two semantic events."""
    existing = _event()
    replay = _event(summary="different summary")
    transaction = RecordingSession(
        scripted=[RecordingResult(), RecordingResult([_event_row(existing)])]
    )
    repository = SqlAlchemyEventRepository(cast(AsyncSession, transaction))

    with pytest.raises(EventKeyConflict):
        await repository.append(replay)


@pytest.mark.asyncio
async def test_offset_event_timestamp_is_canonicalized_before_replay_comparison() -> (
    None
):
    """Equivalent offset instants share a cursor while a different instant conflicts."""
    source_time = datetime(2026, 8, 5, 10, 0, tzinfo=timezone(timedelta(hours=8)))
    equivalent_utc = datetime(2026, 8, 5, 2, 0, tzinfo=UTC)
    canonical_row = _event_row(_event(created_at=equivalent_utc))
    canonical_row["created_at"] = equivalent_utc.replace(tzinfo=None)
    transaction = RecordingSession(
        scripted=[
            RecordingResult(),
            RecordingResult([canonical_row]),
            RecordingResult(),
            RecordingResult([canonical_row]),
            RecordingResult(),
            RecordingResult([canonical_row]),
        ]
    )
    repository = SqlAlchemyEventRepository(cast(AsyncSession, transaction))

    assert await repository.append(_event(created_at=source_time)) == 41
    assert await repository.append(_event(created_at=equivalent_utc)) == 41
    with pytest.raises(EventKeyConflict):
        await repository.append(
            _event(created_at=equivalent_utc + timedelta(seconds=1))
        )
    bound_timestamp = transaction.statements[0].compile().params["created_at"]
    assert bound_timestamp == equivalent_utc.replace(tzinfo=None)


@pytest.mark.asyncio
async def test_only_active_slot_duplicate_is_translated_to_active_run_conflict() -> (
    None
):
    """Unrelated integrity failures remain visible as operational defects."""
    active_duplicate = IntegrityError(
        None, None, Exception(1062, "Duplicate entry for key 'uq_runs_active_slot'")
    )
    other_integrity = IntegrityError(
        None, None, Exception(1062, "Duplicate entry for key 'PRIMARY'")
    )

    active_session = RecordingSession(scripted=[active_duplicate])
    with pytest.raises(ActiveRunConflict):
        await SqlAlchemyRunRepository().create_queued(
            UserScope(user_id="user-1"),
            "thread-1",
            SNAPSHOT,
            transaction=cast(AsyncSession, active_session),
        )

    other_session = RecordingSession(scripted=[other_integrity])
    with pytest.raises(IntegrityError):
        await SqlAlchemyRunRepository().create_queued(
            UserScope(user_id="user-1"),
            "thread-1",
            SNAPSHOT,
            transaction=cast(AsyncSession, other_session),
        )
