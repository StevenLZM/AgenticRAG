"""SQL-backed publication and reconciliation lifecycle adapters."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import UTC, datetime, timedelta
from typing import Any, Literal, cast

from sqlalchemy import and_, case, exists, or_, select, update
from sqlalchemy.engine import CursorResult, RowMapping
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from agentic_rag.domain.models import DocumentStatus, DocumentVersionStatus, JobStatus
from agentic_rag.ingestion.indexer import StagingContext
from agentic_rag.ingestion.publisher import (
    PublicationError,
    PublicationObsoleteError,
    PublicationRepository,
    PublicationTarget,
)
from agentic_rag.ingestion.reconciler import (
    DeletedDocument,
    PointerMismatch,
    ReconciliationRepository,
)
from agentic_rag.persistence.repositories import (
    OutboxRecord,
    document_versions,
    documents,
    ingestion_jobs,
    task_outbox,
)


class PublicationConflict(PublicationError):
    """The document pointer changed during a publication attempt."""


class SqlAlchemyPublicationRepository(PublicationRepository):
    """Load trusted publication state and commit the final pointer atomically."""

    def __init__(self, session_factory: async_sessionmaker[AsyncSession]) -> None:
        self._session_factory = session_factory

    async def is_writable(self, context: StagingContext) -> bool:
        async with self._session_factory() as session:
            writable = await session.scalar(
                select(documents.c.id)
                .select_from(
                    documents.join(
                        document_versions,
                        document_versions.c.document_id == documents.c.id,
                    )
                )
                .where(
                    documents.c.id == context.document_id,
                    documents.c.user_id == context.user_id,
                    documents.c.status != DocumentStatus.DELETED.value,
                    documents.c.deletion_status.is_(None),
                    document_versions.c.id == context.document_version_id,
                    document_versions.c.status.in_(
                        (
                            DocumentVersionStatus.BUILDING.value,
                            DocumentVersionStatus.ACTIVE.value,
                            DocumentVersionStatus.INACTIVE.value,
                        )
                    ),
                )
                .limit(1)
            )
        return writable is not None

    async def get_target(self, version_id: str) -> PublicationTarget | None:
        async with self._session_factory() as session:
            row = (
                (
                    await session.execute(
                        select(
                            document_versions,
                            documents.c.user_id.label("user_id"),
                            documents.c.status.label("document_status"),
                            documents.c.active_version_id,
                        )
                        .select_from(
                            document_versions.join(
                                documents,
                                document_versions.c.document_id == documents.c.id,
                            )
                        )
                        .where(document_versions.c.id == version_id)
                    )
                )
                .mappings()
                .one_or_none()
            )
            if row is None or row["document_status"] == DocumentStatus.DELETED.value:
                return None
            active_version_id = cast(str | None, row["active_version_id"])
            status_is_publishable = row["status"] in {
                DocumentVersionStatus.BUILDING.value,
                DocumentVersionStatus.ACTIVE.value,
            } or (
                row["status"] == DocumentVersionStatus.INACTIVE.value
                and active_version_id == version_id
            )
            if not status_is_publishable:
                return None
            required = (
                row["manifest_path"],
                row["manifest_hash"],
                row["canonical_ast_path"],
                row["canonical_ast_hash"],
                row["parent_count"],
                row["child_count"],
            )
            if any(value is None for value in required):
                return None
            context = _context(row)
            previous_context = None
            if active_version_id is not None and active_version_id != version_id:
                previous = (
                    (
                        await session.execute(
                            select(
                                document_versions,
                                documents.c.user_id.label("user_id"),
                            )
                            .select_from(
                                document_versions.join(
                                    documents,
                                    document_versions.c.document_id == documents.c.id,
                                )
                            )
                            .where(
                                document_versions.c.id == active_version_id,
                                document_versions.c.document_id == row["document_id"],
                            )
                        )
                    )
                    .mappings()
                    .one_or_none()
                )
                if previous is None:
                    return None
                if cast(int, previous["version_no"]) > context.version_no:
                    raise PublicationObsoleteError(context, active_version_id)
                previous_context = _context(previous)
            return PublicationTarget(
                context=context,
                active_version_id=active_version_id,
                previous_context=previous_context,
                manifest_uri=cast(str, row["manifest_path"]),
                manifest_hash=cast(str, row["manifest_hash"]),
                canonical_ast_uri=cast(str, row["canonical_ast_path"]),
                canonical_ast_sha256=cast(str, row["canonical_ast_hash"]),
                parent_count=cast(int, row["parent_count"]),
                child_count=cast(int, row["child_count"]),
            )

    async def finalize(self, target: PublicationTarget) -> None:
        context = target.context
        async with self._session_factory.begin() as session:
            document = (
                (
                    await session.execute(
                        select(documents)
                        .where(
                            documents.c.id == context.document_id,
                            documents.c.user_id == context.user_id,
                        )
                        .with_for_update()
                    )
                )
                .mappings()
                .one_or_none()
            )
            if document is None or document["status"] == DocumentStatus.DELETED.value:
                raise PublicationConflict("document was deleted during publication")
            current_pointer = cast(str | None, document["active_version_id"])
            if current_pointer not in {
                target.active_version_id,
                context.document_version_id,
            }:
                winner = await session.execute(
                    select(document_versions.c.id, document_versions.c.version_no).where(
                        document_versions.c.id == current_pointer,
                        document_versions.c.document_id == context.document_id,
                    )
                )
                winner_row = winner.one_or_none()
                if winner_row is not None and winner_row.version_no > context.version_no:
                    raise PublicationObsoleteError(context, cast(str, winner_row.id))
                raise PublicationConflict("active version changed during publication")

            newer_active = await session.scalar(
                select(document_versions.c.id)
                .where(
                    document_versions.c.document_id == context.document_id,
                    document_versions.c.version_no > context.version_no,
                    document_versions.c.status == DocumentVersionStatus.ACTIVE.value,
                )
                .limit(1)
            )
            if newer_active is not None:
                raise PublicationObsoleteError(context, cast(str, newer_active))

            current_version_status = await session.scalar(
                select(document_versions.c.status).where(
                    document_versions.c.id == context.document_version_id,
                    document_versions.c.document_id == context.document_id,
                )
            )
            if (
                current_pointer == context.document_version_id
                and current_version_status == DocumentVersionStatus.ACTIVE.value
            ):
                return

            await session.execute(
                update(document_versions)
                .where(
                    document_versions.c.document_id == context.document_id,
                    document_versions.c.id != context.document_version_id,
                    document_versions.c.status == DocumentVersionStatus.ACTIVE.value,
                )
                .values(status=DocumentVersionStatus.INACTIVE.value)
            )
            result = cast(
                CursorResult[Any],
                await session.execute(
                    update(document_versions)
                    .where(
                        document_versions.c.id == context.document_version_id,
                        document_versions.c.document_id == context.document_id,
                        document_versions.c.status.in_(
                            (
                                DocumentVersionStatus.BUILDING.value,
                                DocumentVersionStatus.ACTIVE.value,
                                DocumentVersionStatus.INACTIVE.value,
                            )
                        ),
                    )
                    .values(status=DocumentVersionStatus.ACTIVE.value)
                ),
            )
            if result.rowcount != 1:
                raise PublicationConflict("version changed during publication")
            await session.execute(
                update(documents)
                .where(
                    documents.c.id == context.document_id,
                    documents.c.user_id == context.user_id,
                )
                .values(
                    active_version_id=context.document_version_id,
                    status=DocumentStatus.ACTIVE.value,
                    updated_at=_now(),
                )
            )

    async def quarantine(self, version_id: str) -> None:
        async with self._session_factory.begin() as session:
            await session.execute(
                update(document_versions)
                .where(
                    document_versions.c.id == version_id,
                    document_versions.c.status.in_(
                        (
                            DocumentVersionStatus.UPLOADED.value,
                            DocumentVersionStatus.BUILDING.value,
                        )
                    ),
                )
                .values(status=DocumentVersionStatus.QUARANTINED.value)
            )


class SqlAlchemyReconciliationRepository(ReconciliationRepository):
    """Bounded MySQL scans for the five Task-5 drift classes."""

    def __init__(
        self,
        session_factory: async_sessionmaker[AsyncSession],
        *,
        stale_after: timedelta = timedelta(minutes=15),
    ) -> None:
        if stale_after <= timedelta(0):
            raise ValueError("stale_after must be positive")
        self._session_factory = session_factory
        self._stale_after = stale_after

    async def claim_pending_outbox(self, limit: int) -> tuple[OutboxRecord, ...]:
        now = _now()
        async with self._session_factory.begin() as session:
            rows = (
                (
                    await session.execute(
                        select(task_outbox)
                        .where(
                            task_outbox.c.status == "pending",
                            task_outbox.c.aggregate_type == "ingestion_job",
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
                    .values(next_attempt_at=now + timedelta(seconds=30))
                )
        return tuple(_outbox(row) for row in rows)

    async def reclaim_expired_jobs(self, limit: int) -> tuple[str, ...]:
        now = _now()
        async with self._session_factory.begin() as session:
            rows = (
                (
                    await session.execute(
                        select(ingestion_jobs.c.id)
                        .where(
                            ingestion_jobs.c.status == JobStatus.RUNNING.value,
                            or_(
                                ingestion_jobs.c.lease_expires_at.is_(None),
                                ingestion_jobs.c.lease_expires_at <= now,
                            ),
                        )
                        .order_by(ingestion_jobs.c.lease_expires_at, ingestion_jobs.c.id)
                        .limit(limit)
                        .with_for_update(skip_locked=True)
                    )
                )
                .scalars()
                .all()
            )
            job_ids = tuple(cast(str, value) for value in rows)
            if job_ids:
                await session.execute(
                    update(ingestion_jobs)
                    .where(
                        ingestion_jobs.c.id.in_(job_ids),
                        ingestion_jobs.c.status == JobStatus.RUNNING.value,
                        or_(
                            ingestion_jobs.c.lease_expires_at.is_(None),
                            ingestion_jobs.c.lease_expires_at <= now,
                        ),
                    )
                    .values(
                        status=JobStatus.QUEUED.value,
                        lease_owner=None,
                        lease_expires_at=None,
                        heartbeat_at=None,
                        attempt_count=ingestion_jobs.c.attempt_count + 1,
                        updated_at=now,
                    )
                )
                await session.execute(
                    update(task_outbox)
                    .where(
                        task_outbox.c.aggregate_type == "ingestion_job",
                        task_outbox.c.aggregate_id.in_(job_ids),
                    )
                    .values(
                        status="pending",
                        attempt_count=task_outbox.c.attempt_count + 1,
                        next_attempt_at=now,
                        dispatched_at=None,
                    )
                )
            return job_ids

    async def list_stale_building_versions(self, limit: int) -> tuple[str, ...]:
        cutoff = _now() - self._stale_after
        live_job = exists(
            select(ingestion_jobs.c.id).where(
                ingestion_jobs.c.document_version_id == document_versions.c.id,
                or_(
                    ingestion_jobs.c.status == JobStatus.QUEUED.value,
                    and_(
                        ingestion_jobs.c.status == JobStatus.RUNNING.value,
                        ingestion_jobs.c.lease_expires_at.is_not(None),
                        ingestion_jobs.c.lease_expires_at > _now(),
                    ),
                ),
            )
        )
        async with self._session_factory() as session:
            values = (
                await session.execute(
                    select(document_versions.c.id)
                    .select_from(
                        document_versions.join(
                            documents,
                            document_versions.c.document_id == documents.c.id,
                        )
                    )
                    .where(
                        document_versions.c.status
                        == DocumentVersionStatus.BUILDING.value,
                        document_versions.c.created_at <= cutoff,
                        documents.c.status != DocumentStatus.DELETED.value,
                        ~live_job,
                    )
                    .order_by(document_versions.c.created_at, document_versions.c.id)
                    .limit(limit)
                )
            ).scalars().all()
        return tuple(cast(str, value) for value in values)

    async def list_pointer_mismatches(self, limit: int) -> tuple[PointerMismatch, ...]:
        pointed = document_versions.alias("pointed_version")
        live_job = exists(
            select(ingestion_jobs.c.id).where(
                ingestion_jobs.c.document_version_id == document_versions.c.id,
                or_(
                    ingestion_jobs.c.status == JobStatus.QUEUED.value,
                    and_(
                        ingestion_jobs.c.status == JobStatus.RUNNING.value,
                        ingestion_jobs.c.lease_expires_at.is_not(None),
                        ingestion_jobs.c.lease_expires_at > _now(),
                    ),
                ),
            )
        )
        async with self._session_factory() as session:
            values = (
                await session.execute(
                    select(
                        document_versions,
                        documents.c.user_id.label("user_id"),
                        documents.c.status.label("document_status"),
                        documents.c.active_version_id,
                        pointed.c.version_no.label("pointed_version_no"),
                    )
                    .select_from(
                        document_versions.join(
                            documents,
                            document_versions.c.document_id == documents.c.id,
                        ).outerjoin(
                            pointed,
                            documents.c.active_version_id == pointed.c.id,
                        )
                    )
                    .where(
                        documents.c.status != DocumentStatus.DELETED.value,
                        ~live_job,
                        or_(
                            and_(
                                document_versions.c.status
                                == DocumentVersionStatus.BUILDING.value,
                                document_versions.c.manifest_path.is_not(None),
                                document_versions.c.manifest_hash.is_not(None),
                                or_(
                                    documents.c.active_version_id.is_(None),
                                    document_versions.c.id
                                    != documents.c.active_version_id,
                                ),
                            ),
                            and_(
                                document_versions.c.status
                                == DocumentVersionStatus.ACTIVE.value,
                                or_(
                                    documents.c.active_version_id.is_(None),
                                    document_versions.c.id
                                    != documents.c.active_version_id,
                                ),
                            ),
                            and_(
                                documents.c.active_version_id
                                == document_versions.c.id,
                                pointed.c.status
                                != DocumentVersionStatus.ACTIVE.value,
                                or_(
                                    document_versions.c.status
                                    != DocumentVersionStatus.BUILDING.value,
                                    and_(
                                        document_versions.c.manifest_path.is_not(None),
                                        document_versions.c.manifest_hash.is_not(None),
                                    ),
                                ),
                            ),
                            and_(
                                documents.c.active_version_id
                                == document_versions.c.id,
                                document_versions.c.status
                                == DocumentVersionStatus.ACTIVE.value,
                                documents.c.status
                                != DocumentStatus.ACTIVE.value,
                            ),
                        ),
                    )
                    .order_by(document_versions.c.created_at, document_versions.c.id)
                    .limit(limit)
                )
            ).mappings().all()
        mismatches: list[PointerMismatch] = []
        for row in values:
            version_id = cast(str, row["id"])
            pointer = cast(str | None, row["active_version_id"])
            version_no = cast(int, row["version_no"])
            pointed_version_no = cast(int | None, row["pointed_version_no"])
            status = cast(str, row["status"])
            document_status = cast(str, row["document_status"])
            has_manifest = row["manifest_path"] is not None and row["manifest_hash"] is not None
            action: Literal["publish", "deactivate", "restore"]
            if (
                version_id == pointer
                and status == DocumentVersionStatus.ACTIVE.value
                and document_status != DocumentStatus.ACTIVE.value
            ):
                action = "restore"
            else:
                publish = has_manifest and (
                    (version_id == pointer and status in {DocumentVersionStatus.BUILDING.value, DocumentVersionStatus.INACTIVE.value})
                    or (
                        version_id != pointer
                        and status in {DocumentVersionStatus.BUILDING.value, DocumentVersionStatus.ACTIVE.value}
                        and (pointed_version_no is None or version_no > pointed_version_no)
                    )
                )
                action = "publish" if publish else "deactivate"
            mismatches.append(
                PointerMismatch(
                    context=_context(row),
                    action=action,
                )
            )
        return tuple(mismatches)

    async def restore_active_document(self, version_id: str) -> bool:
        async with self._session_factory.begin() as session:
            row = (
                (
                    await session.execute(
                        select(
                            documents.c.id.label("document_id"),
                            documents.c.status.label("document_status"),
                            document_versions.c.status.label("version_status"),
                        )
                        .select_from(
                            document_versions.join(
                                documents,
                                document_versions.c.document_id == documents.c.id,
                            )
                        )
                        .where(
                            document_versions.c.id == version_id,
                            documents.c.active_version_id == version_id,
                        )
                        .with_for_update()
                    )
                )
                .mappings()
                .one_or_none()
            )
            if (
                row is None
                or row["version_status"] != DocumentVersionStatus.ACTIVE.value
                or row["document_status"] == DocumentStatus.DELETED.value
            ):
                return False
            result = cast(
                CursorResult[Any],
                await session.execute(
                    update(documents)
                    .where(
                        documents.c.id == row["document_id"],
                        documents.c.active_version_id == version_id,
                        documents.c.status != DocumentStatus.DELETED.value,
                    )
                    .values(status=DocumentStatus.ACTIVE.value, updated_at=_now())
                ),
            )
            return result.rowcount == 1

    async def resolve_deactivated_version(self, version_id: str) -> None:
        async with self._session_factory.begin() as session:
            version = (
                (
                    await session.execute(
                        select(document_versions)
                        .where(document_versions.c.id == version_id)
                        .with_for_update()
                    )
                )
                .mappings()
                .one_or_none()
            )
            if version is None:
                return
            await session.execute(
                update(document_versions)
                .where(
                    document_versions.c.id == version_id,
                    document_versions.c.status.in_(
                        (
                            DocumentVersionStatus.BUILDING.value,
                            DocumentVersionStatus.ACTIVE.value,
                            DocumentVersionStatus.INACTIVE.value,
                        )
                    ),
                )
                .values(status=DocumentVersionStatus.INACTIVE.value)
            )
            await session.execute(
                update(documents)
                .where(
                    documents.c.id == version["document_id"],
                    or_(
                        documents.c.active_version_id == version_id,
                        and_(
                            documents.c.active_version_id.is_(None),
                            documents.c.status == DocumentStatus.ACTIVE.value,
                        ),
                    ),
                )
                .values(
                    active_version_id=None,
                    status=DocumentStatus.FAILED.value,
                    updated_at=_now(),
                )
            )

    async def list_deleted_documents(self, limit: int) -> tuple[DeletedDocument, ...]:
        cutoff = _now() - self._stale_after
        async with self._session_factory() as session:
            document_rows = (
                (
                    await session.execute(
                        select(documents.c.id, documents.c.user_id)
                        .where(
                            documents.c.status == DocumentStatus.DELETED.value,
                            or_(
                                and_(
                                    documents.c.deletion_status == "fenced",
                                    documents.c.deletion_fenced_at.is_not(None),
                                    documents.c.deletion_fenced_at <= cutoff,
                                ),
                                documents.c.deletion_status == "completed",
                            ),
                        )
                        .order_by(
                            case(
                                (documents.c.deletion_status == "fenced", 0),
                                else_=1,
                            ),
                            documents.c.updated_at,
                            documents.c.id,
                        )
                        .limit(limit)
                    )
                )
                .mappings()
                .all()
            )
            document_ids = tuple(cast(str, row["id"]) for row in document_rows)
            if not document_ids:
                return ()
            rows = (
                (
                    await session.execute(
                        select(
                            document_versions,
                            documents.c.user_id.label("user_id"),
                        )
                        .select_from(
                            document_versions.join(
                                documents,
                                document_versions.c.document_id == documents.c.id,
                            )
                        )
                        .where(document_versions.c.document_id.in_(document_ids))
                        .order_by(
                            document_versions.c.document_id,
                            document_versions.c.version_no,
                        )
                    )
                )
                .mappings()
                .all()
            )
        grouped: dict[str, list[StagingContext]] = {value: [] for value in document_ids}
        user_by_document = {
            cast(str, row["id"]): cast(str, row["user_id"])
            for row in document_rows
        }
        for row in rows:
            grouped[cast(str, row["document_id"])].append(_context(row))
        return tuple(
            DeletedDocument(
                user_id=user_by_document[value],
                document_id=value,
                versions=tuple(grouped[value]),
            )
            for value in document_ids
        )

    async def fence_pending_deletions(self, limit: int) -> tuple[str, ...]:
        now = _now()
        async with self._session_factory.begin() as session:
            values = (
                await session.execute(
                    select(documents.c.id)
                    .where(
                        documents.c.status == DocumentStatus.DELETED.value,
                        documents.c.deletion_status == "pending",
                    )
                    .order_by(documents.c.updated_at, documents.c.id)
                    .limit(limit)
                    .with_for_update(skip_locked=True)
                )
            ).scalars().all()
            document_ids = tuple(cast(str, value) for value in values)
            if not document_ids:
                return ()
            await session.execute(
                update(ingestion_jobs)
                .where(
                    ingestion_jobs.c.document_id.in_(document_ids),
                    ingestion_jobs.c.status.in_(
                        (JobStatus.QUEUED.value, JobStatus.RUNNING.value)
                    ),
                )
                .values(
                    status=JobStatus.FAILED.value,
                    lease_owner=None,
                    lease_expires_at=None,
                    heartbeat_at=None,
                    error_code="document_deleted",
                    updated_at=now,
                )
            )
            await session.execute(
                update(documents)
                .where(
                    documents.c.id.in_(document_ids),
                    documents.c.status == DocumentStatus.DELETED.value,
                    documents.c.deletion_status == "pending",
                )
                .values(deletion_status="fenced", deletion_fenced_at=now, updated_at=now)
            )
            return document_ids

    async def quarantine(self, version_id: str) -> None:
        async with self._session_factory.begin() as session:
            await session.execute(
                update(document_versions)
                .where(
                    document_versions.c.id == version_id,
                    document_versions.c.status
                    == DocumentVersionStatus.BUILDING.value,
                )
                .values(status=DocumentVersionStatus.QUARANTINED.value)
            )

    async def mark_deletion_reconciled(self, document_id: str) -> bool:
        async with self._session_factory.begin() as session:
            await session.execute(
                update(document_versions)
                .where(document_versions.c.document_id == document_id)
                .values(status=DocumentVersionStatus.INACTIVE.value)
            )
            completed = cast(
                CursorResult[Any],
                await session.execute(
                    update(documents)
                    .where(
                        documents.c.id == document_id,
                        documents.c.status == DocumentStatus.DELETED.value,
                        documents.c.deletion_status == "fenced",
                    )
                    .values(deletion_status="completed", updated_at=_now())
                ),
            )
            if completed.rowcount == 1:
                return True
            await session.execute(
                update(documents)
                .where(
                    documents.c.id == document_id,
                    documents.c.status == DocumentStatus.DELETED.value,
                    documents.c.deletion_status == "completed",
                )
                .values(updated_at=_now())
            )
            return False


def _context(row: Mapping[str, Any] | RowMapping) -> StagingContext:
    return StagingContext(
        user_id=cast(str, row["user_id"]),
        document_id=cast(str, row["document_id"]),
        document_version_id=cast(str, row["id"]),
        version_no=cast(int, row["version_no"]),
        pipeline_version=cast(str, row["pipeline_version"]),
        embedding_version=cast(str, row["embedding_version"]),
        index_generation=cast(str, row["index_generation"]),
    )


def _outbox(row: Mapping[str, Any] | RowMapping) -> OutboxRecord:
    return OutboxRecord(
        id=cast(str, row["id"]),
        aggregate_type=cast(str, row["aggregate_type"]),
        aggregate_id=cast(str, row["aggregate_id"]),
        stream_name=cast(str, row["stream_name"]),
        status=cast(str, row["status"]),
        attempt_count=cast(int, row["attempt_count"]),
        next_attempt_at=cast(datetime, row["next_attempt_at"]),
        created_at=cast(datetime, row["created_at"]),
    )


def _now() -> datetime:
    return datetime.now(UTC)
