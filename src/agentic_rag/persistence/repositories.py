"""Repository ports and SQLAlchemy Core adapters for durable runtime state."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any, Protocol, cast, runtime_checkable

from sqlalchemy import (
    JSON,
    BigInteger,
    CheckConstraint,
    Column,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    MetaData,
    String,
    Table,
    Text,
    UniqueConstraint,
    and_,
    case,
    func,
    insert,
    or_,
    select,
    update,
)
from sqlalchemy.engine import CursorResult
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.dialects.mysql import insert as mysql_insert

from agentic_rag.domain.models import (
    DocumentStatus,
    DocumentVersionStatus,
    JobStatus,
    RunStatus,
    UserScope,
)
from agentic_rag.runtime.ids import new_id
from agentic_rag.runtime.models import RuntimeConfigSnapshot


metadata = MetaData()

documents = Table(
    "documents",
    metadata,
    Column("id", String(36), primary_key=True),
    Column("user_id", String(255), nullable=False),
    Column("source_type", String(32), nullable=False),
    Column("filename", String(512), nullable=False),
    Column("mime_type", String(128), nullable=False),
    Column("content_hash", String(64), nullable=False),
    Column("status", String(32), nullable=False),
    Column(
        "active_version_id",
        String(36),
        ForeignKey(
            "document_versions.id",
            name="fk_documents_active_version",
            use_alter=True,
            ondelete="SET NULL",
        ),
        nullable=True,
    ),
    Column("source_trust", String(32), nullable=False, default="untrusted"),
    Column("created_at", DateTime(timezone=True), nullable=False),
    Column("updated_at", DateTime(timezone=True), nullable=False),
    CheckConstraint(
        "status IN ('processing','active','failed','deleted')",
        name="ck_documents_status",
    ),
    CheckConstraint(
        "source_trust IN ('untrusted','trusted_curated')",
        name="ck_documents_source_trust",
    ),
    Index("ix_documents_user_status", "user_id", "status"),
    Index("ix_documents_user_content_hash", "user_id", "content_hash"),
)

document_versions = Table(
    "document_versions",
    metadata,
    Column("id", String(36), primary_key=True),
    Column(
        "document_id",
        String(36),
        ForeignKey("documents.id", name="fk_versions_document", ondelete="CASCADE"),
        nullable=False,
    ),
    Column("version_no", Integer, nullable=False),
    Column("parser_version", String(128), nullable=False),
    Column("pipeline_version", String(128), nullable=False),
    Column("canonical_ast_path", String(1024), nullable=True),
    Column("canonical_ast_hash", String(64), nullable=True),
    Column("manifest_path", String(1024), nullable=True),
    Column("manifest_hash", String(64), nullable=True),
    Column("parent_count", Integer, nullable=False, default=0),
    Column("child_count", Integer, nullable=False, default=0),
    Column("embedding_version", String(128), nullable=False),
    Column("index_generation", String(128), nullable=False),
    Column("status", String(32), nullable=False),
    Column("created_at", DateTime(timezone=True), nullable=False),
    CheckConstraint(
        "status IN ('uploaded','building','active','quarantined','failed','inactive')",
        name="ck_document_versions_status",
    ),
    CheckConstraint(
        "((manifest_path IS NULL AND manifest_hash IS NULL) OR "
        "(manifest_path IS NOT NULL AND manifest_hash IS NOT NULL))",
        name="ck_document_versions_manifest_pair",
    ),
    UniqueConstraint("document_id", "version_no", name="uq_versions_document_number"),
    Index("ix_versions_document_status", "document_id", "status"),
)

parent_chunks = Table(
    "parent_chunks",
    metadata,
    Column("id", String(64), primary_key=True),
    Column("user_id", String(255), nullable=False),
    Column(
        "document_id",
        String(36),
        ForeignKey("documents.id", name="fk_parents_document", ondelete="CASCADE"),
        nullable=False,
    ),
    Column(
        "document_version_id",
        String(36),
        ForeignKey(
            "document_versions.id", name="fk_parents_version", ondelete="CASCADE"
        ),
        nullable=False,
    ),
    Column("ordinal", Integer, nullable=False),
    Column("heading_path", JSON, nullable=False),
    Column("content_type", String(64), nullable=False),
    Column("content", Text, nullable=False),
    Column("page_from", Integer, nullable=True),
    Column("page_to", Integer, nullable=True),
    Column("ast_locator", String(512), nullable=False),
    Column("content_hash", String(64), nullable=False),
    Column("status", String(16), nullable=False),
    CheckConstraint("status IN ('active','inactive')", name="ck_parent_chunks_status"),
    UniqueConstraint(
        "document_version_id", "ordinal", name="uq_parents_version_ordinal"
    ),
    Index("ix_parents_user_status", "user_id", "status"),
    Index("ix_parents_user_document", "user_id", "document_id"),
)

ingestion_jobs = Table(
    "ingestion_jobs",
    metadata,
    Column("id", String(36), primary_key=True),
    Column("user_id", String(255), nullable=False),
    Column(
        "document_id",
        String(36),
        ForeignKey("documents.id", name="fk_jobs_document", ondelete="CASCADE"),
        nullable=False,
    ),
    Column(
        "document_version_id",
        String(36),
        ForeignKey("document_versions.id", name="fk_jobs_version", ondelete="CASCADE"),
        nullable=False,
    ),
    Column("status", String(32), nullable=False),
    Column("lease_owner", String(255), nullable=True),
    Column("lease_expires_at", DateTime(timezone=True), nullable=True),
    Column("heartbeat_at", DateTime(timezone=True), nullable=True),
    Column("error_code", String(128), nullable=True),
    Column("attempt_count", Integer, nullable=False, default=0),
    Column("last_error_detail_ref", String(1024), nullable=True),
    Column("created_at", DateTime(timezone=True), nullable=False),
    Column("updated_at", DateTime(timezone=True), nullable=False),
    CheckConstraint(
        "status IN ('queued','running','completed','quarantined','failed')",
        name="ck_ingestion_jobs_status",
    ),
    Index("ix_jobs_status_lease", "status", "lease_expires_at"),
    Index("ix_jobs_user_created", "user_id", "created_at"),
)

agent_runs = Table(
    "agent_runs",
    metadata,
    Column("id", String(36), primary_key=True),
    Column("user_id", String(255), nullable=False),
    Column("thread_id", String(255), nullable=False),
    Column("checkpoint_thread_id", String(255), nullable=False),
    Column("status", String(32), nullable=False),
    Column("active_slot", Integer, nullable=True),
    Column("lease_owner", String(255), nullable=True),
    Column("lease_expires_at", DateTime(timezone=True), nullable=True),
    Column("heartbeat_at", DateTime(timezone=True), nullable=True),
    Column("attempt_count", Integer, nullable=False, default=0),
    Column("route", String(64), nullable=True),
    Column("runtime_config_snapshot_id", String(64), nullable=False),
    Column("runtime_config_snapshot", JSON, nullable=False),
    Column("result_ref", String(1024), nullable=True),
    Column("error_code", String(128), nullable=True),
    Column("termination_reason", String(255), nullable=True),
    Column("created_at", DateTime(timezone=True), nullable=False),
    Column("started_at", DateTime(timezone=True), nullable=True),
    Column("finished_at", DateTime(timezone=True), nullable=True),
    CheckConstraint(
        "status IN ('queued','running','cancel_requested','cancelled','completed','failed')",
        name="ck_agent_runs_status",
    ),
    CheckConstraint(
        "((status IN ('queued','running','cancel_requested') "
        "AND active_slot IS NOT NULL AND active_slot = 1) "
        "OR (status IN ('cancelled','completed','failed') AND active_slot IS NULL))",
        name="ck_active_slot",
    ),
    UniqueConstraint("user_id", "thread_id", "active_slot", name="uq_runs_active_slot"),
    Index("ix_runs_status_lease", "status", "lease_expires_at"),
    Index("ix_runs_user_thread_created", "user_id", "thread_id", "created_at"),
)

task_outbox = Table(
    "task_outbox",
    metadata,
    Column("id", String(36), primary_key=True),
    Column("aggregate_type", String(32), nullable=False),
    Column("aggregate_id", String(36), nullable=False),
    Column("stream_name", String(255), nullable=False),
    Column("status", String(16), nullable=False),
    Column("attempt_count", Integer, nullable=False, default=0),
    Column("next_attempt_at", DateTime(timezone=True), nullable=False),
    Column("created_at", DateTime(timezone=True), nullable=False),
    Column("dispatched_at", DateTime(timezone=True), nullable=True),
    CheckConstraint(
        "aggregate_type IN ('query_run','ingestion_job')",
        name="ck_outbox_aggregate_type",
    ),
    CheckConstraint("status IN ('pending','dispatched')", name="ck_outbox_status"),
    UniqueConstraint("aggregate_type", "aggregate_id", name="uq_outbox_aggregate"),
    Index("ix_outbox_pending", "status", "next_attempt_at", "id"),
)

messages = Table(
    "messages",
    metadata,
    Column("id", String(36), primary_key=True),
    Column(
        "run_id",
        String(36),
        ForeignKey("agent_runs.id", name="fk_messages_run", ondelete="CASCADE"),
        nullable=False,
    ),
    Column("user_id", String(255), nullable=False),
    Column("thread_id", String(255), nullable=False),
    Column("role", String(32), nullable=False),
    Column("content", Text, nullable=True),
    Column("payload_ref", String(1024), nullable=True),
    Column("created_at", DateTime(timezone=True), nullable=False),
    CheckConstraint("role IN ('user','assistant','tool')", name="ck_messages_role"),
    Index("ix_messages_user_thread_created", "user_id", "thread_id", "created_at"),
)

agent_events = Table(
    "agent_events",
    metadata,
    Column("id", BigInteger, primary_key=True, autoincrement=True),
    Column("event_key", String(64), nullable=False),
    Column("trace_id", String(64), nullable=False),
    Column(
        "run_id",
        String(36),
        ForeignKey("agent_runs.id", name="fk_events_run", ondelete="CASCADE"),
        nullable=False,
    ),
    Column("user_id", String(255), nullable=False),
    Column("node_name", String(128), nullable=True),
    Column("event_type", String(64), nullable=False),
    Column("summary", String(1024), nullable=False),
    Column("payload_ref", String(1024), nullable=True),
    Column("runtime_config_snapshot_id", String(64), nullable=False),
    Column("created_at", DateTime(timezone=True), nullable=False),
    UniqueConstraint("event_key", name="uq_events_event_key"),
    Index("ix_events_user_run_id", "user_id", "run_id", "id"),
)

memory_tombstones = Table(
    "memory_tombstones",
    metadata,
    Column("id", String(36), primary_key=True),
    Column("user_id", String(255), nullable=False),
    Column("memory_id", String(255), nullable=False),
    Column("requested_at", DateTime(timezone=True), nullable=False),
    Column("status", String(16), nullable=False),
    Column("attempt_count", Integer, nullable=False, default=0),
    Column("completed_at", DateTime(timezone=True), nullable=True),
    Column("last_error_detail_ref", String(1024), nullable=True),
    CheckConstraint(
        "status IN ('pending','completed','failed')", name="ck_tombstones_status"
    ),
    UniqueConstraint("user_id", "memory_id", name="uq_tombstones_user_memory"),
    Index("ix_tombstones_status_requested", "status", "requested_at"),
)


ACTIVE_RUN_STATUSES = {
    RunStatus.QUEUED,
    RunStatus.RUNNING,
    RunStatus.CANCEL_REQUESTED,
}
TERMINAL_RUN_STATUSES = {
    RunStatus.CANCELLED,
    RunStatus.COMPLETED,
    RunStatus.FAILED,
}
OUTBOX_CLAIM_LEASE = timedelta(seconds=30)


def _now() -> datetime:
    return datetime.now(UTC)


def _mysql_datetime(value: datetime) -> datetime:
    """Normalize an explicit timestamp to MySQL DATETIME's UTC-naive form."""
    if value.tzinfo is None:
        return value
    return value.astimezone(UTC).replace(tzinfo=None)


def active_slot_for_status(status: RunStatus) -> int | None:
    """Map active run states to the unique slot used by MySQL."""
    return 1 if status in ACTIVE_RUN_STATUSES else None


def _is_mysql_duplicate_for(error: IntegrityError, constraint_name: str) -> bool:
    args = getattr(error.orig, "args", ())
    if not args:
        return False
    return args[0] == 1062 and constraint_name in str(error.orig)


class ActiveRunConflict(RuntimeError):
    """Raised when a user/thread already has a non-terminal run."""


class LeaseLost(RuntimeError):
    """Raised when a worker no longer owns the current live Run lease."""


class EventKeyConflict(RuntimeError):
    """Raised when one deterministic event key is reused for different data."""


@dataclass(frozen=True, slots=True)
class QueryRun:
    id: str
    user_id: str
    thread_id: str
    checkpoint_thread_id: str
    status: RunStatus
    active_slot: int | None
    runtime_config_snapshot_id: str
    runtime_config_snapshot: dict[str, Any]
    lease_owner: str | None = None
    lease_expires_at: datetime | None = None
    claim_generation: int = 0
    result_ref: str | None = None
    error_code: str | None = None


@dataclass(frozen=True, slots=True)
class IngestionJob:
    id: str
    user_id: str
    document_id: str
    document_version_id: str
    status: JobStatus


@dataclass(frozen=True, slots=True)
class Document:
    id: str
    user_id: str
    status: DocumentStatus
    active_version_id: str | None


@dataclass(frozen=True, slots=True)
class DocumentVersion:
    id: str
    document_id: str
    version_no: int
    status: DocumentVersionStatus


@dataclass(frozen=True, slots=True)
class ParentChunk:
    id: str
    user_id: str
    document_id: str
    document_version_id: str
    ordinal: int
    content: str
    status: str


@dataclass(frozen=True, slots=True)
class AgentEvent:
    event_key: str
    trace_id: str
    run_id: str
    user_id: str
    event_type: str
    summary: str
    runtime_config_snapshot_id: str
    node_name: str | None = None
    payload_ref: str | None = None
    created_at: datetime | None = None
    id: int | None = None


@dataclass(frozen=True, slots=True)
class OutboxRecord:
    id: str
    aggregate_type: str
    aggregate_id: str
    stream_name: str
    status: str
    attempt_count: int
    next_attempt_at: datetime
    created_at: datetime


@dataclass(frozen=True, slots=True)
class MemoryTombstone:
    id: str
    user_id: str
    memory_id: str
    status: str
    attempt_count: int


@runtime_checkable
class RunRepository(Protocol):
    async def create_queued(
        self,
        scope: UserScope,
        thread_id: str,
        snapshot: RuntimeConfigSnapshot,
        *,
        transaction: AsyncSession | None = None,
    ) -> QueryRun: ...

    async def get(self, run_id: str, scope: UserScope) -> QueryRun | None: ...

    async def claim(
        self, run_id: str, owner: str, lease_seconds: int
    ) -> QueryRun | None: ...

    async def heartbeat(
        self,
        run_id: str,
        owner: str,
        lease_seconds: int,
        *,
        claim_generation: int,
    ) -> None: ...

    async def request_cancel(self, run_id: str, scope: UserScope) -> RunStatus: ...

    async def finish(
        self,
        run_id: str,
        status: RunStatus,
        result_ref: str | None,
        error_code: str | None,
        *,
        owner: str,
        claim_generation: int,
    ) -> None: ...


@runtime_checkable
class IngestionJobRepository(Protocol):
    async def create(
        self,
        scope: UserScope,
        document_id: str,
        document_version_id: str,
        *,
        status: JobStatus,
        transaction: AsyncSession | None = None,
    ) -> IngestionJob: ...

    async def create_queued(
        self,
        scope: UserScope,
        document_id: str,
        document_version_id: str,
        *,
        transaction: AsyncSession | None = None,
    ) -> IngestionJob: ...

    async def get(
        self,
        job_id: str,
        scope: UserScope,
        *,
        transaction: AsyncSession | None = None,
    ) -> IngestionJob | None: ...


@runtime_checkable
class DocumentRepository(Protocol):
    async def create(
        self,
        scope: UserScope,
        *,
        source_type: str,
        filename: str,
        mime_type: str,
        content_hash: str,
        parser_version: str,
        pipeline_version: str,
        embedding_version: str,
        index_generation: str,
        document_id: str | None = None,
        document_version_id: str | None = None,
        version_status: DocumentVersionStatus = DocumentVersionStatus.UPLOADED,
        transaction: AsyncSession | None = None,
    ) -> tuple[Document, DocumentVersion]: ...

    async def get(self, document_id: str, scope: UserScope) -> Document | None: ...

    async def soft_delete(
        self,
        document_id: str,
        scope: UserScope,
        *,
        transaction: AsyncSession | None = None,
    ) -> bool: ...


@runtime_checkable
class ParentRepository(Protocol):
    async def get_many(
        self, parent_ids: list[str], scope: UserScope
    ) -> list[ParentChunk]: ...


@runtime_checkable
class EventRepository(Protocol):
    async def append(self, event: AgentEvent) -> int: ...

    async def list_after(
        self, run_id: str, scope: UserScope, after_id: int, limit: int
    ) -> list[AgentEvent]: ...


@runtime_checkable
class OutboxRepository(Protocol):
    async def list_pending(self, limit: int) -> list[OutboxRecord]: ...

    async def claim_pending(self, limit: int) -> list[OutboxRecord]: ...

    async def mark_dispatched(self, outbox_id: str) -> None: ...

    async def schedule_retry(self, outbox_id: str) -> None: ...


@runtime_checkable
class MemoryTombstoneRepository(Protocol):
    async def request(self, scope: UserScope, memory_id: str) -> MemoryTombstone: ...

    async def is_deleted(self, scope: UserScope, memory_id: str) -> bool: ...


class _SqlAlchemyRepository:
    def __init__(self, session: AsyncSession | None = None) -> None:
        self._bound_session = session

    def _session(self, transaction: AsyncSession | None = None) -> AsyncSession:
        session = transaction or self._bound_session
        if session is None:
            raise ValueError("an AsyncSession transaction is required")
        return session


def _run_from_row(row: dict[str, Any]) -> QueryRun:
    return QueryRun(
        id=row["id"],
        user_id=row["user_id"],
        thread_id=row["thread_id"],
        checkpoint_thread_id=row["checkpoint_thread_id"],
        status=RunStatus(row["status"]),
        active_slot=row["active_slot"],
        runtime_config_snapshot_id=row["runtime_config_snapshot_id"],
        runtime_config_snapshot=cast(dict[str, Any], row["runtime_config_snapshot"]),
        lease_owner=row["lease_owner"],
        lease_expires_at=row["lease_expires_at"],
        claim_generation=row["attempt_count"],
        result_ref=row["result_ref"],
        error_code=row["error_code"],
    )


class SqlAlchemyRunRepository(_SqlAlchemyRepository):
    async def create_queued(
        self,
        scope: UserScope,
        thread_id: str,
        snapshot: RuntimeConfigSnapshot,
        *,
        transaction: AsyncSession | None = None,
    ) -> QueryRun:
        session = self._session(transaction)
        now = _now()
        run = QueryRun(
            id=new_id(),
            user_id=scope.user_id,
            thread_id=thread_id,
            checkpoint_thread_id=f"query:{scope.user_id}:{thread_id}",
            status=RunStatus.QUEUED,
            active_slot=1,
            runtime_config_snapshot_id=snapshot.snapshot_id,
            runtime_config_snapshot=snapshot.model_dump(mode="json"),
        )
        try:
            await session.execute(
                insert(agent_runs).values(
                    id=run.id,
                    user_id=run.user_id,
                    thread_id=run.thread_id,
                    checkpoint_thread_id=run.checkpoint_thread_id,
                    status=run.status.value,
                    active_slot=run.active_slot,
                    attempt_count=0,
                    runtime_config_snapshot_id=run.runtime_config_snapshot_id,
                    runtime_config_snapshot=run.runtime_config_snapshot,
                    created_at=now,
                )
            )
        except IntegrityError as error:
            if not _is_mysql_duplicate_for(error, "uq_runs_active_slot"):
                raise
            raise ActiveRunConflict(
                f"active run already exists for user={scope.user_id!r}, thread={thread_id!r}"
            ) from error
        await session.execute(
            insert(task_outbox).values(
                id=new_id(),
                aggregate_type="query_run",
                aggregate_id=run.id,
                stream_name="agenticrag:jobs:query",
                status="pending",
                attempt_count=0,
                next_attempt_at=now,
                created_at=now,
            )
        )
        return run

    async def get(self, run_id: str, scope: UserScope) -> QueryRun | None:
        session = self._session()
        row = (
            (
                await session.execute(
                    select(agent_runs).where(
                        agent_runs.c.id == run_id,
                        agent_runs.c.user_id == scope.user_id,
                    )
                )
            )
            .mappings()
            .one_or_none()
        )
        return _run_from_row(dict(row)) if row else None

    async def claim(
        self, run_id: str, owner: str, lease_seconds: int
    ) -> QueryRun | None:
        if lease_seconds <= 0:
            raise ValueError("lease_seconds must be positive")
        session = self._session()
        now = _now()
        result = cast(
            CursorResult[Any],
            await session.execute(
                update(agent_runs)
                .where(
                    agent_runs.c.id == run_id,
                    or_(
                        agent_runs.c.status == RunStatus.QUEUED.value,
                        and_(
                            agent_runs.c.status.in_(
                                [
                                    RunStatus.RUNNING.value,
                                    RunStatus.CANCEL_REQUESTED.value,
                                ]
                            ),
                            agent_runs.c.lease_expires_at <= now,
                        ),
                    ),
                )
                .values(
                    status=case(
                        (
                            agent_runs.c.status == RunStatus.CANCEL_REQUESTED.value,
                            RunStatus.CANCEL_REQUESTED.value,
                        ),
                        else_=RunStatus.RUNNING.value,
                    ),
                    lease_owner=owner,
                    lease_expires_at=now + timedelta(seconds=lease_seconds),
                    heartbeat_at=now,
                    attempt_count=agent_runs.c.attempt_count + 1,
                    started_at=func.coalesce(agent_runs.c.started_at, now),
                )
            ),
        )
        if result.rowcount != 1:
            return None
        row = (
            (await session.execute(select(agent_runs).where(agent_runs.c.id == run_id)))
            .mappings()
            .one_or_none()
        )
        if not row:
            return None
        return _run_from_row(dict(row))

    async def heartbeat(
        self,
        run_id: str,
        owner: str,
        lease_seconds: int,
        *,
        claim_generation: int,
    ) -> None:
        if lease_seconds <= 0:
            raise ValueError("lease_seconds must be positive")
        now = _now()
        result = cast(
            CursorResult[Any],
            await self._session().execute(
                update(agent_runs)
                .where(
                    agent_runs.c.id == run_id,
                    agent_runs.c.lease_owner == owner,
                    agent_runs.c.attempt_count == claim_generation,
                    agent_runs.c.lease_expires_at > now,
                    agent_runs.c.status.in_(
                        [RunStatus.RUNNING.value, RunStatus.CANCEL_REQUESTED.value]
                    ),
                )
                .values(
                    heartbeat_at=now,
                    lease_expires_at=now + timedelta(seconds=lease_seconds),
                )
            ),
        )
        if result.rowcount != 1:
            raise LeaseLost(run_id)

    async def request_cancel(self, run_id: str, scope: UserScope) -> RunStatus:
        session = self._session()
        now = _now()
        queued_result = cast(
            CursorResult[Any],
            await session.execute(
                update(agent_runs)
                .where(
                    agent_runs.c.id == run_id,
                    agent_runs.c.user_id == scope.user_id,
                    agent_runs.c.status == RunStatus.QUEUED.value,
                )
                .values(
                    status=RunStatus.CANCELLED.value,
                    active_slot=None,
                    finished_at=now,
                    lease_owner=None,
                    lease_expires_at=None,
                )
            ),
        )
        if queued_result.rowcount == 1:
            return RunStatus.CANCELLED
        running_result = cast(
            CursorResult[Any],
            await session.execute(
                update(agent_runs)
                .where(
                    agent_runs.c.id == run_id,
                    agent_runs.c.user_id == scope.user_id,
                    agent_runs.c.status == RunStatus.RUNNING.value,
                )
                .values(status=RunStatus.CANCEL_REQUESTED.value)
            ),
        )
        if running_result.rowcount == 1:
            return RunStatus.CANCEL_REQUESTED
        row = (
            await session.execute(
                select(agent_runs.c.status).where(
                    agent_runs.c.id == run_id,
                    agent_runs.c.user_id == scope.user_id,
                )
            )
        ).one_or_none()
        if row is None:
            raise KeyError(run_id)
        return RunStatus(row[0])

    async def finish(
        self,
        run_id: str,
        status: RunStatus,
        result_ref: str | None,
        error_code: str | None,
        *,
        owner: str,
        claim_generation: int,
    ) -> None:
        if status not in TERMINAL_RUN_STATUSES:
            raise ValueError("finish requires a terminal RunStatus")
        now = _now()
        allowed_current_statuses = [RunStatus.RUNNING.value]
        if status is RunStatus.CANCELLED:
            allowed_current_statuses.append(RunStatus.CANCEL_REQUESTED.value)
        result = cast(
            CursorResult[Any],
            await self._session().execute(
                update(agent_runs)
                .where(
                    agent_runs.c.id == run_id,
                    agent_runs.c.status.in_(allowed_current_statuses),
                    agent_runs.c.lease_owner == owner,
                    agent_runs.c.attempt_count == claim_generation,
                    agent_runs.c.lease_expires_at > now,
                )
                .values(
                    status=status.value,
                    active_slot=None,
                    result_ref=result_ref,
                    error_code=error_code,
                    finished_at=now,
                    lease_owner=None,
                    lease_expires_at=None,
                )
            ),
        )
        if result.rowcount != 1:
            raise LeaseLost(run_id)

    async def mark_completed(
        self,
        run_id: str,
        result_ref: str | None,
        *,
        owner: str,
        claim_generation: int,
    ) -> None:
        await self.finish(
            run_id,
            RunStatus.COMPLETED,
            result_ref,
            None,
            owner=owner,
            claim_generation=claim_generation,
        )


class SqlAlchemyIngestionJobRepository(_SqlAlchemyRepository):
    async def create(
        self,
        scope: UserScope,
        document_id: str,
        document_version_id: str,
        *,
        status: JobStatus,
        transaction: AsyncSession | None = None,
    ) -> IngestionJob:
        session = self._session(transaction)
        now = _now()
        job = IngestionJob(
            id=new_id(),
            user_id=scope.user_id,
            document_id=document_id,
            document_version_id=document_version_id,
            status=status,
        )
        await session.execute(
            insert(ingestion_jobs).values(
                id=job.id,
                user_id=job.user_id,
                document_id=job.document_id,
                document_version_id=job.document_version_id,
                status=job.status.value,
                attempt_count=0,
                created_at=now,
                updated_at=now,
            )
        )
        await session.execute(
            insert(task_outbox).values(
                id=new_id(),
                aggregate_type="ingestion_job",
                aggregate_id=job.id,
                stream_name="agenticrag:jobs:ingestion",
                status="pending",
                attempt_count=0,
                next_attempt_at=now,
                created_at=now,
            )
        )
        return job

    async def create_queued(
        self,
        scope: UserScope,
        document_id: str,
        document_version_id: str,
        *,
        transaction: AsyncSession | None = None,
    ) -> IngestionJob:
        return await self.create(
            scope,
            document_id,
            document_version_id,
            status=JobStatus.QUEUED,
            transaction=transaction,
        )

    async def get(
        self,
        job_id: str,
        scope: UserScope,
        *,
        transaction: AsyncSession | None = None,
    ) -> IngestionJob | None:
        row = (
            (
                await self._session(transaction).execute(
                    select(ingestion_jobs).where(
                        ingestion_jobs.c.id == job_id,
                        ingestion_jobs.c.user_id == scope.user_id,
                    )
                )
            )
            .mappings()
            .one_or_none()
        )
        if not row:
            return None
        return IngestionJob(
            id=row["id"],
            user_id=row["user_id"],
            document_id=row["document_id"],
            document_version_id=row["document_version_id"],
            status=JobStatus(row["status"]),
        )


class SqlAlchemyDocumentRepository(_SqlAlchemyRepository):
    async def create(
        self,
        scope: UserScope,
        *,
        source_type: str,
        filename: str,
        mime_type: str,
        content_hash: str,
        parser_version: str,
        pipeline_version: str,
        embedding_version: str,
        index_generation: str,
        document_id: str | None = None,
        document_version_id: str | None = None,
        version_status: DocumentVersionStatus = DocumentVersionStatus.UPLOADED,
        transaction: AsyncSession | None = None,
    ) -> tuple[Document, DocumentVersion]:
        session = self._session(transaction)
        now = _now()
        document = Document(
            id=document_id or new_id(),
            user_id=scope.user_id,
            status=DocumentStatus.PROCESSING,
            active_version_id=None,
        )
        version = DocumentVersion(
            id=document_version_id or new_id(),
            document_id=document.id,
            version_no=1,
            status=version_status,
        )
        await session.execute(
            insert(documents).values(
                id=document.id,
                user_id=document.user_id,
                source_type=source_type,
                filename=filename,
                mime_type=mime_type,
                content_hash=content_hash,
                status=document.status.value,
                source_trust="untrusted",
                created_at=now,
                updated_at=now,
            )
        )
        await session.execute(
            insert(document_versions).values(
                id=version.id,
                document_id=document.id,
                version_no=version.version_no,
                parser_version=parser_version,
                pipeline_version=pipeline_version,
                parent_count=0,
                child_count=0,
                embedding_version=embedding_version,
                index_generation=index_generation,
                status=version.status.value,
                created_at=now,
            )
        )
        return document, version

    async def soft_delete(
        self,
        document_id: str,
        scope: UserScope,
        *,
        transaction: AsyncSession | None = None,
    ) -> bool:
        result = await self._session(transaction).execute(
            update(documents)
            .where(
                documents.c.id == document_id,
                documents.c.user_id == scope.user_id,
                documents.c.status != DocumentStatus.DELETED.value,
            )
            .values(
                status=DocumentStatus.DELETED.value,
                active_version_id=None,
                updated_at=_now(),
            )
        )
        return cast(CursorResult[Any], result).rowcount == 1

    async def get(self, document_id: str, scope: UserScope) -> Document | None:
        row = (
            (
                await self._session().execute(
                    select(documents).where(
                        documents.c.id == document_id,
                        documents.c.user_id == scope.user_id,
                    )
                )
            )
            .mappings()
            .one_or_none()
        )
        if not row:
            return None
        return Document(
            id=row["id"],
            user_id=row["user_id"],
            status=DocumentStatus(row["status"]),
            active_version_id=row["active_version_id"],
        )


class SqlAlchemyParentRepository(_SqlAlchemyRepository):
    async def get_many(
        self, parent_ids: list[str], scope: UserScope
    ) -> list[ParentChunk]:
        if not parent_ids:
            return []
        rows = (
            (
                await self._session().execute(
                    select(parent_chunks).where(
                        parent_chunks.c.id.in_(parent_ids),
                        parent_chunks.c.user_id == scope.user_id,
                        parent_chunks.c.status == "active",
                    )
                )
            )
            .mappings()
            .all()
        )
        by_id = {
            row["id"]: ParentChunk(
                id=row["id"],
                user_id=row["user_id"],
                document_id=row["document_id"],
                document_version_id=row["document_version_id"],
                ordinal=row["ordinal"],
                content=row["content"],
                status=row["status"],
            )
            for row in rows
        }
        return [by_id[parent_id] for parent_id in parent_ids if parent_id in by_id]


class SqlAlchemyEventRepository(_SqlAlchemyRepository):
    async def append(self, event: AgentEvent) -> int:
        session = self._session()
        statement = mysql_insert(agent_events).values(
            event_key=event.event_key,
            trace_id=event.trace_id,
            run_id=event.run_id,
            user_id=event.user_id,
            node_name=event.node_name,
            event_type=event.event_type,
            summary=event.summary,
            payload_ref=event.payload_ref,
            runtime_config_snapshot_id=event.runtime_config_snapshot_id,
            created_at=_mysql_datetime(event.created_at or _now()),
        )
        await session.execute(statement.on_duplicate_key_update(id=agent_events.c.id))
        row = (
            (
                await session.execute(
                    select(agent_events).where(
                        agent_events.c.event_key == event.event_key
                    )
                )
            )
            .mappings()
            .one()
        )
        expected = (
            event.trace_id,
            event.run_id,
            event.user_id,
            event.node_name,
            event.event_type,
            event.summary,
            event.payload_ref,
            event.runtime_config_snapshot_id,
        )
        actual = (
            row["trace_id"],
            row["run_id"],
            row["user_id"],
            row["node_name"],
            row["event_type"],
            row["summary"],
            row["payload_ref"],
            row["runtime_config_snapshot_id"],
        )
        timestamp_conflicts = event.created_at is not None and _mysql_datetime(
            row["created_at"]
        ) != _mysql_datetime(event.created_at)
        if actual != expected or timestamp_conflicts:
            raise EventKeyConflict(event.event_key)
        return cast(int, row["id"])

    async def list_after(
        self, run_id: str, scope: UserScope, after_id: int, limit: int
    ) -> list[AgentEvent]:
        if limit <= 0:
            return []
        rows = (
            (
                await self._session().execute(
                    select(agent_events)
                    .where(
                        agent_events.c.run_id == run_id,
                        agent_events.c.user_id == scope.user_id,
                        agent_events.c.id > after_id,
                    )
                    .order_by(agent_events.c.id)
                    .limit(limit)
                )
            )
            .mappings()
            .all()
        )
        return [AgentEvent(**dict(row)) for row in rows]


class SqlAlchemyOutboxRepository(_SqlAlchemyRepository):
    async def list_pending(self, limit: int) -> list[OutboxRecord]:
        if limit <= 0:
            return []
        rows = (
            (
                await self._session().execute(
                    select(task_outbox)
                    .where(
                        task_outbox.c.status == "pending",
                        task_outbox.c.next_attempt_at <= _now(),
                    )
                    .order_by(task_outbox.c.next_attempt_at, task_outbox.c.id)
                    .limit(limit)
                )
            )
            .mappings()
            .all()
        )
        return _outbox_records(rows)

    async def claim_pending(self, limit: int) -> list[OutboxRecord]:
        if limit <= 0:
            return []
        now = _now()
        session = self._session()
        rows = (
            (
                await session.execute(
                    select(task_outbox)
                    .where(
                        task_outbox.c.status == "pending",
                        task_outbox.c.next_attempt_at <= now,
                    )
                    .order_by(task_outbox.c.next_attempt_at, task_outbox.c.id)
                    .limit(limit)
                    .with_for_update(skip_locked=True)
                )
            )
            .mappings()
            .all()
        )
        if rows:
            await session.execute(
                update(task_outbox)
                .where(
                    task_outbox.c.id.in_([row["id"] for row in rows]),
                    task_outbox.c.status == "pending",
                )
                .values(next_attempt_at=now + OUTBOX_CLAIM_LEASE)
            )
        return _outbox_records(rows)

    async def mark_dispatched(self, outbox_id: str) -> None:
        await self._session().execute(
            update(task_outbox)
            .where(task_outbox.c.id == outbox_id, task_outbox.c.status == "pending")
            .values(status="dispatched", dispatched_at=_now())
        )

    async def schedule_retry(self, outbox_id: str) -> None:
        await self._session().execute(
            update(task_outbox)
            .where(task_outbox.c.id == outbox_id, task_outbox.c.status == "pending")
            .values(
                attempt_count=task_outbox.c.attempt_count + 1,
                next_attempt_at=_now() + timedelta(seconds=5),
            )
        )


def _outbox_records(rows: Sequence[Any]) -> list[OutboxRecord]:
    return [
        OutboxRecord(
            id=row["id"],
            aggregate_type=row["aggregate_type"],
            aggregate_id=row["aggregate_id"],
            stream_name=row["stream_name"],
            status=row["status"],
            attempt_count=row["attempt_count"],
            next_attempt_at=row["next_attempt_at"],
            created_at=row["created_at"],
        )
        for row in rows
    ]


class SqlAlchemyMemoryTombstoneRepository(_SqlAlchemyRepository):
    async def request(self, scope: UserScope, memory_id: str) -> MemoryTombstone:
        session = self._session()
        existing = (
            (
                await session.execute(
                    select(memory_tombstones).where(
                        memory_tombstones.c.user_id == scope.user_id,
                        memory_tombstones.c.memory_id == memory_id,
                    )
                )
            )
            .mappings()
            .one_or_none()
        )
        if existing:
            return MemoryTombstone(
                id=existing["id"],
                user_id=existing["user_id"],
                memory_id=existing["memory_id"],
                status=existing["status"],
                attempt_count=existing["attempt_count"],
            )
        tombstone = MemoryTombstone(
            id=new_id(),
            user_id=scope.user_id,
            memory_id=memory_id,
            status="pending",
            attempt_count=0,
        )
        await session.execute(
            insert(memory_tombstones).values(
                id=tombstone.id,
                user_id=tombstone.user_id,
                memory_id=tombstone.memory_id,
                requested_at=_now(),
                status=tombstone.status,
                attempt_count=tombstone.attempt_count,
            )
        )
        return tombstone

    async def is_deleted(self, scope: UserScope, memory_id: str) -> bool:
        row = (
            await self._session().execute(
                select(memory_tombstones.c.id).where(
                    memory_tombstones.c.user_id == scope.user_id,
                    memory_tombstones.c.memory_id == memory_id,
                )
            )
        ).one_or_none()
        return row is not None
