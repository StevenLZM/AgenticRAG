"""Periodic repair of ingestion delivery, publication, and deletion drift."""

from __future__ import annotations

import asyncio
from typing import Literal, Protocol

from pydantic import BaseModel, ConfigDict

from agentic_rag.ingestion.indexer import StagingContext
from agentic_rag.ingestion.publisher import (
    PublicationIntegrityError,
    PublicationObsoleteError,
    VersionLifecycleStore,
    VersionPublisher,
)
from agentic_rag.persistence.repositories import OutboxRecord


class ReconcileReport(BaseModel):
    """Explicit durable IDs repaired or quarantined during one bounded pass."""

    model_config = ConfigDict(frozen=True)

    redispatched_jobs: tuple[str, ...] = ()
    reclaimed_jobs: tuple[str, ...] = ()
    repaired_versions: tuple[str, ...] = ()
    quarantined_versions: tuple[str, ...] = ()
    reconciled_deletions: tuple[str, ...] = ()


class DeletedDocument(BaseModel):
    """Trusted version scopes belonging to one already-soft-deleted document."""

    model_config = ConfigDict(frozen=True)

    user_id: str
    document_id: str
    versions: tuple[StagingContext, ...]


class PointerMismatch(BaseModel):
    """A classified SQL pointer/status drift with one safe repair action."""

    model_config = ConfigDict(frozen=True)

    context: StagingContext
    action: Literal["publish", "deactivate", "restore"]


class ReconciliationRepository(Protocol):
    async def claim_pending_outbox(self, limit: int) -> tuple[OutboxRecord, ...]: ...

    async def reclaim_expired_jobs(self, limit: int) -> tuple[str, ...]: ...

    async def list_stale_building_versions(self, limit: int) -> tuple[str, ...]: ...

    async def list_pointer_mismatches(self, limit: int) -> tuple[PointerMismatch, ...]: ...

    async def resolve_deactivated_version(self, version_id: str) -> None: ...

    async def restore_active_document(self, version_id: str) -> bool: ...

    async def fence_pending_deletions(self, limit: int) -> tuple[str, ...]: ...

    async def list_deleted_documents(self, limit: int) -> tuple[DeletedDocument, ...]: ...

    async def quarantine(self, version_id: str) -> None: ...

    async def mark_deletion_reconciled(self, document_id: str) -> bool: ...


class OutboxRedispatcher(Protocol):
    async def redispatch(self, row: OutboxRecord) -> None: ...


class DocumentArtifactStore(Protocol):
    def delete_document_scope(self, user_id: str, document_id: str) -> None: ...


class IngestionReconciler:
    """Converge bounded lifecycle drift without relying on the ingestion graph."""

    def __init__(
        self,
        *,
        repository: ReconciliationRepository,
        publisher: VersionPublisher,
        dispatcher: OutboxRedispatcher,
        parent_store: VersionLifecycleStore,
        child_store: VersionLifecycleStore,
        artifacts: DocumentArtifactStore,
        scan_limit: int = 100,
    ) -> None:
        if scan_limit <= 0:
            raise ValueError("scan_limit must be positive")
        self._repository = repository
        self._publisher = publisher
        self._dispatcher = dispatcher
        self._parent_store = parent_store
        self._child_store = child_store
        self._artifacts = artifacts
        self._scan_limit = scan_limit

    async def run_once(self) -> ReconcileReport:
        redispatched: list[str] = []
        for row in await self._repository.claim_pending_outbox(self._scan_limit):
            if row.aggregate_type != "ingestion_job":
                continue
            try:
                await self._dispatcher.redispatch(row)
            except Exception:
                continue
            else:
                redispatched.append(row.aggregate_id)

        reclaimed = list(
            await self._repository.reclaim_expired_jobs(self._scan_limit)
        )
        stale = await self._repository.list_stale_building_versions(self._scan_limit)
        mismatches = await self._repository.list_pointer_mismatches(self._scan_limit)
        publish_candidates = _unique(
            (*stale, *(item.context.document_version_id for item in mismatches if item.action == "publish"))
        )
        repaired: list[str] = []
        quarantined: list[str] = []
        deactivated: set[str] = set()
        for version_id in publish_candidates:
            try:
                await self._publisher.publish(version_id)
            except PublicationObsoleteError as error:
                try:
                    await self._child_store.deactivate(error.context)
                    await self._parent_store.deactivate(error.context)
                    await self._repository.resolve_deactivated_version(version_id)
                except Exception:
                    continue
                deactivated.add(version_id)
                repaired.append(version_id)
            except PublicationIntegrityError:
                try:
                    await self._repository.quarantine(version_id)
                except Exception:
                    continue
                else:
                    quarantined.append(version_id)
            except Exception:
                continue
            else:
                repaired.append(version_id)

        for mismatch in mismatches:
            if mismatch.action == "restore":
                version_id = mismatch.context.document_version_id
                try:
                    restored = await self._repository.restore_active_document(version_id)
                except Exception:
                    continue
                if restored:
                    repaired.append(version_id)
                continue
            if mismatch.action != "deactivate":
                continue
            version_id = mismatch.context.document_version_id
            if version_id in deactivated:
                continue
            try:
                await self._child_store.deactivate(mismatch.context)
                await self._parent_store.deactivate(mismatch.context)
                await self._repository.resolve_deactivated_version(version_id)
            except Exception:
                continue
            else:
                repaired.append(version_id)

        await self._repository.fence_pending_deletions(self._scan_limit)
        deletions: list[str] = []
        for deleted in await self._repository.list_deleted_documents(self._scan_limit):
            try:
                for context in deleted.versions:
                    await self._child_store.deactivate(context)
                    await self._child_store.delete(context)
                    await self._parent_store.deactivate(context)
                    await self._parent_store.delete(context)
                await asyncio.to_thread(
                    self._artifacts.delete_document_scope,
                    deleted.user_id,
                    deleted.document_id,
                )
                first_completion = await self._repository.mark_deletion_reconciled(
                    deleted.document_id
                )
            except Exception:
                continue
            if first_completion:
                deletions.append(deleted.document_id)

        return ReconcileReport(
            redispatched_jobs=tuple(redispatched),
            reclaimed_jobs=tuple(reclaimed),
            repaired_versions=tuple(repaired),
            quarantined_versions=tuple(quarantined),
            reconciled_deletions=tuple(deletions),
        )


def _unique(values: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(values))
