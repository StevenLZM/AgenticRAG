"""Offline fake-store lifecycle unit tests for publication and reconciliation."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import UTC, datetime

import pytest
from redis.exceptions import RedisError

from agentic_rag.ingestion.indexer import StagingContext
from agentic_rag.ingestion.manifest import VersionManifest
from agentic_rag.ingestion.publisher import (
    PublicationTarget,
    VersionPublisher,
)
from agentic_rag.ingestion.reconciler import (
    DeletedDocument,
    IngestionReconciler,
    PointerMismatch,
)
from agentic_rag.persistence.artifacts import ArtifactRef
from agentic_rag.persistence.repositories import OutboxRecord


def _context(version_id: str, version_no: int) -> StagingContext:
    return StagingContext(
        user_id="user-1",
        document_id="document-1",
        document_version_id=version_id,
        version_no=version_no,
        pipeline_version="pipeline-v1",
        embedding_version="text-embedding-v3",
        index_generation="index-v2",
    )


@dataclass
class _Artifacts:
    payloads: dict[str, bytes]

    def verify(self, ref: ArtifactRef) -> bool:
        value = self.payloads.get(ref.uri)
        return value is not None and len(value) == ref.size_bytes and hashlib.sha256(value).hexdigest() == ref.sha256

    def read_json(self, ref: ArtifactRef) -> object:
        if not self.verify(ref):
            raise ValueError("bad artifact")
        return json.loads(self.payloads[ref.uri])

    def verify_hash(self, uri: str, sha256: str) -> bool:
        value = self.payloads.get(uri)
        return value is not None and hashlib.sha256(value).hexdigest() == sha256


@dataclass
class _LifecycleStore:
    label: str
    events: list[str]
    active: set[str]
    totals: dict[str, int]
    fail_once_after_event: str | None = None

    async def count_total(self, context: StagingContext) -> int:
        return self.totals.get(context.document_version_id, 0)

    async def activate(self, context: StagingContext) -> None:
        self.active.add(context.document_version_id)
        await self._record(f"{self.label}.activate:{context.document_version_id}")

    async def deactivate(self, context: StagingContext) -> None:
        self.active.discard(context.document_version_id)
        await self._record(f"{self.label}.deactivate:{context.document_version_id}")

    async def delete(self, context: StagingContext) -> None:
        self.active.discard(context.document_version_id)
        self.totals.pop(context.document_version_id, None)
        await self._record(f"{self.label}.delete:{context.document_version_id}")

    async def _record(self, event: str) -> None:
        self.events.append(event)
        if self.fail_once_after_event == event:
            self.fail_once_after_event = None
            raise RuntimeError(f"injected failure after {event}")


@dataclass
class _PublicationRepository:
    target: PublicationTarget
    events: list[str]
    active_version_id: str | None = "version-1"
    quarantined: list[str] = field(default_factory=list)
    fail_once_after_finalize: bool = False

    async def get_target(self, version_id: str) -> PublicationTarget | None:
        if version_id != self.target.context.document_version_id:
            return None
        return self.target.model_copy(update={"active_version_id": self.active_version_id})

    async def finalize(self, target: PublicationTarget) -> None:
        self.events.append(f"finalize:{target.context.document_version_id}")
        self.active_version_id = target.context.document_version_id
        if self.fail_once_after_finalize:
            self.fail_once_after_finalize = False
            raise RuntimeError("injected failure after finalize")

    async def quarantine(self, version_id: str) -> None:
        self.quarantined.append(version_id)


@dataclass
class _ReconciliationRepository:
    pending: tuple[OutboxRecord, ...] = ()
    expired: tuple[str, ...] = ()
    stale: tuple[str, ...] = ()
    mismatches: tuple[PointerMismatch, ...] = ()
    deleted: tuple[DeletedDocument, ...] = ()
    quarantined: list[str] = field(default_factory=list)
    deletion_marks: list[str] = field(default_factory=list)
    resolved_mismatches: list[str] = field(default_factory=list)

    async def claim_pending_outbox(self, limit: int) -> tuple[OutboxRecord, ...]:
        return self.pending[:limit]

    async def reclaim_expired_jobs(self, limit: int) -> tuple[str, ...]:
        return self.expired[:limit]

    async def list_stale_building_versions(self, limit: int) -> tuple[str, ...]:
        return self.stale[:limit]

    async def list_pointer_mismatches(self, limit: int) -> tuple[PointerMismatch, ...]:
        return self.mismatches[:limit]

    async def resolve_deactivated_version(self, version_id: str) -> None:
        self.resolved_mismatches.append(version_id)

    async def list_deleted_documents(self, limit: int) -> tuple[DeletedDocument, ...]:
        return self.deleted[:limit]

    async def quarantine(self, version_id: str) -> None:
        self.quarantined.append(version_id)

    async def mark_deletion_reconciled(self, document_id: str) -> None:
        self.deletion_marks.append(document_id)
        self.deleted = tuple(
            value for value in self.deleted if value.document_id != document_id
        )


@dataclass
class _Dispatcher:
    dispatched: list[str] = field(default_factory=list)
    fail: bool = False

    async def redispatch(self, row: OutboxRecord) -> None:
        if self.fail:
            raise RedisError("offline")
        self.dispatched.append(row.aggregate_id)


def _system(*, corrupt_manifest: bool = False, corrupt_canonical: bool = False) -> tuple[
    VersionPublisher,
    _PublicationRepository,
    _LifecycleStore,
    _LifecycleStore,
    list[str],
]:
    events: list[str] = []
    canonical_payload = b"canonical"
    canonical_hash = hashlib.sha256(canonical_payload).hexdigest()
    manifest = VersionManifest(
        canonical_ast_sha256=canonical_hash,
        parent_count=2,
        child_count=3,
        embedding_model="text-embedding-v3",
        embedding_dimensions=1024,
        index_generation="index-v2",
    )
    encoded = json.dumps(
        manifest.payload(), ensure_ascii=False, separators=(",", ":"), sort_keys=True
    ).encode()
    manifest_hash = "f" * 64 if corrupt_manifest else manifest.manifest_hash
    target = PublicationTarget(
        context=_context("version-2", 2),
        active_version_id="version-1",
        previous_context=_context("version-1", 1),
        manifest_uri=(
            "artifact://documents/user-1/document-1/version-2/"
            f"manifests/index-v2/{manifest_hash}.json"
        ),
        manifest_hash=manifest_hash,
        canonical_ast_uri=(
            "artifact://documents/user-1/document-1/version-2/canonical/"
            "docling-v1/pipeline-v1/1.json"
        ),
        canonical_ast_sha256=canonical_hash,
        parent_count=2,
        child_count=3,
    )
    repository = _PublicationRepository(target=target, events=events)
    parents = _LifecycleStore(
        "parents", events, {"version-1"}, {"version-1": 2, "version-2": 2}
    )
    children = _LifecycleStore(
        "children", events, {"version-1"}, {"version-1": 3, "version-2": 3}
    )
    payloads = {
        target.manifest_uri: encoded,
        target.canonical_ast_uri: canonical_payload,
    }
    if corrupt_canonical:
        payloads.pop(target.canonical_ast_uri)
    publisher = VersionPublisher(
        repository=repository,
        parent_store=parents,
        child_store=children,
        artifacts=_Artifacts(payloads),
    )
    return publisher, repository, parents, children, events


async def test_publish_uses_fixed_idempotent_activation_order() -> None:
    publisher, repository, parents, children, events = _system()

    await publisher.publish("version-2")
    await publisher.publish("version-2")

    assert events[:5] == [
        "parents.activate:version-2",
        "children.activate:version-2",
        "children.deactivate:version-1",
        "parents.deactivate:version-1",
        "finalize:version-2",
    ]
    assert repository.active_version_id == "version-2"
    assert parents.active == {"version-2"}
    assert children.active == {"version-2"}


@pytest.mark.parametrize(
    ("failure_owner", "failure_event"),
    [
        ("parents", "parents.activate:version-2"),
        ("children", "children.activate:version-2"),
        ("children", "children.deactivate:version-1"),
        ("parents", "parents.deactivate:version-1"),
        ("repository", "finalize:version-2"),
    ],
)
async def test_reconciler_finishes_every_interrupted_activation_without_dual_exposure(
    failure_owner: str, failure_event: str
) -> None:
    publisher, repository, parents, children, _ = _system()
    if failure_owner == "repository":
        repository.fail_once_after_finalize = True
    elif failure_owner == "parents":
        parents.fail_once_after_event = failure_event
    else:
        children.fail_once_after_event = failure_event

    with pytest.raises(RuntimeError, match="injected failure"):
        await publisher.publish("version-2")

    # Durable pointer remains the retrieval gate while physical activation is partial.
    assert {repository.active_version_id} in ({"version-1"}, {"version-2"})

    state = _ReconciliationRepository(
        mismatches=(
            PointerMismatch(context=_context("version-2", 2), action="publish"),
        )
    )
    report = await IngestionReconciler(
        repository=state,
        publisher=publisher,
        dispatcher=_Dispatcher(),
        parent_store=parents,
        child_store=children,
    ).run_once()

    assert report.repaired_versions == ("version-2",)
    assert repository.active_version_id == "version-2"
    assert parents.active == {"version-2"}
    assert children.active == {"version-2"}


async def test_bad_manifest_is_quarantined_instead_of_published() -> None:
    publisher, repository, parents, children, _ = _system(corrupt_manifest=True)
    state = _ReconciliationRepository(stale=("version-2",))

    report = await IngestionReconciler(
        repository=state,
        publisher=publisher,
        dispatcher=_Dispatcher(),
        parent_store=parents,
        child_store=children,
    ).run_once()

    assert report.quarantined_versions == ("version-2",)
    assert state.quarantined == ["version-2"]
    assert repository.active_version_id == "version-1"


async def test_missing_canonical_artifact_is_quarantined() -> None:
    publisher, repository, parents, children, _ = _system(corrupt_canonical=True)
    state = _ReconciliationRepository(stale=("version-2",))

    report = await IngestionReconciler(
        repository=state,
        publisher=publisher,
        dispatcher=_Dispatcher(),
        parent_store=parents,
        child_store=children,
    ).run_once()

    assert report.quarantined_versions == ("version-2",)
    assert repository.active_version_id == "version-1"


async def test_obsolete_active_version_is_physically_and_durably_deactivated() -> None:
    publisher, _, parents, children, _ = _system()
    parents.active.add("version-2")
    children.active.add("version-2")
    mismatch = PointerMismatch(context=_context("version-2", 2), action="deactivate")
    state = _ReconciliationRepository(mismatches=(mismatch,))

    report = await IngestionReconciler(
        repository=state,
        publisher=publisher,
        dispatcher=_Dispatcher(),
        parent_store=parents,
        child_store=children,
    ).run_once()

    assert report.repaired_versions == ("version-2",)
    assert "version-2" not in parents.active
    assert "version-2" not in children.active
    assert state.resolved_mismatches == ["version-2"]


async def test_reconciler_reports_redispatch_reclaim_and_idempotent_deletion() -> None:
    publisher, _, parents, children, _ = _system()
    now = datetime.now(UTC)
    outbox = OutboxRecord(
        id="outbox-1",
        aggregate_type="ingestion_job",
        aggregate_id="job-1",
        stream_name="agenticrag:jobs:ingestion",
        status="pending",
        attempt_count=0,
        next_attempt_at=now,
        created_at=now,
    )
    deleted = DeletedDocument(
        document_id="document-1",
        versions=(_context("version-1", 1), _context("version-2", 2)),
    )
    state = _ReconciliationRepository(
        pending=(outbox,), expired=("job-2",), deleted=(deleted,)
    )
    dispatcher = _Dispatcher()
    reconciler = IngestionReconciler(
        repository=state,
        publisher=publisher,
        dispatcher=dispatcher,
        parent_store=parents,
        child_store=children,
    )

    report = await reconciler.run_once()
    second = await reconciler.run_once()

    assert report.redispatched_jobs == ("job-1",)
    assert report.reclaimed_jobs == ("job-2",)
    assert report.reconciled_deletions == ("document-1",)
    assert second.reconciled_deletions == ()
    assert parents.active == set()
    assert children.active == set()
    assert state.deletion_marks == ["document-1"]


async def test_redis_outage_does_not_block_other_reconciliation_classes() -> None:
    publisher, _, parents, children, _ = _system()
    now = datetime.now(UTC)
    outbox = OutboxRecord(
        id="outbox-1",
        aggregate_type="ingestion_job",
        aggregate_id="job-1",
        stream_name="jobs",
        status="pending",
        attempt_count=0,
        next_attempt_at=now,
        created_at=now,
    )
    state = _ReconciliationRepository(pending=(outbox,), expired=("job-2",))

    report = await IngestionReconciler(
        repository=state,
        publisher=publisher,
        dispatcher=_Dispatcher(fail=True),
        parent_store=parents,
        child_store=children,
    ).run_once()

    assert report.redispatched_jobs == ()
    assert report.reclaimed_jobs == ("job-2",)
