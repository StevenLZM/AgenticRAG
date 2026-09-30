"""Offline fake-store lifecycle unit tests for publication and reconciliation."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, cast

import pytest
from redis.exceptions import RedisError

from agentic_rag.ingestion.indexer import StagingContext
from agentic_rag.ingestion.manifest import VersionManifest
from agentic_rag.ingestion.publisher import (
    PublicationIntegrityError,
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
    deleted_scopes: list[tuple[str, str]] = field(default_factory=list)
    fail_scopes: set[tuple[str, str]] = field(default_factory=set)

    def verify(self, ref: ArtifactRef) -> bool:
        value = self.payloads.get(ref.uri)
        return (
            value is not None
            and len(value) == ref.size_bytes
            and hashlib.sha256(value).hexdigest() == ref.sha256
        )

    def read_json(self, ref: ArtifactRef) -> object:
        if not self.verify(ref):
            raise ValueError("bad artifact")
        return json.loads(self.payloads[ref.uri])

    def verify_hash(self, uri: str, sha256: str) -> bool:
        value = self.payloads.get(uri)
        return value is not None and hashlib.sha256(value).hexdigest() == sha256

    def delete_document_scope(self, user_id: str, document_id: str) -> None:
        scope = (user_id, document_id)
        if scope in self.fail_scopes:
            raise OSError("artifact cleanup failed")
        prefix = f"artifact://documents/{user_id}/{document_id}/"
        self.payloads = {
            uri: value
            for uri, value in self.payloads.items()
            if not uri.startswith(prefix)
        }
        self.deleted_scopes.append(scope)


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
    writable_checks: int = 0
    writable_until_check: int | None = None

    async def get_target(self, version_id: str) -> PublicationTarget | None:
        if version_id != self.target.context.document_version_id:
            return None
        return self.target.model_copy(
            update={"active_version_id": self.active_version_id}
        )

    async def finalize(self, target: PublicationTarget) -> None:
        self.events.append(f"finalize:{target.context.document_version_id}")
        self.active_version_id = target.context.document_version_id
        if self.fail_once_after_finalize:
            self.fail_once_after_finalize = False
            raise RuntimeError("injected failure after finalize")

    async def is_writable(self, context: StagingContext) -> bool:
        self.writable_checks += 1
        return (
            self.writable_until_check is None
            or self.writable_checks <= self.writable_until_check
        )

    async def quarantine(self, version_id: str) -> None:
        self.quarantined.append(version_id)


@dataclass
class _ReconciliationRepository:
    pending: tuple[OutboxRecord, ...] = ()
    expired: tuple[str, ...] = ()
    stale: tuple[str, ...] = ()
    mismatches: tuple[PointerMismatch, ...] = ()
    deleted: tuple[DeletedDocument, ...] = ()
    pending_deletions: tuple[DeletedDocument, ...] = ()
    fenced_deletions: tuple[DeletedDocument, ...] = ()
    completed_deletions: tuple[DeletedDocument, ...] = ()
    deletion_fence_ready: bool = False
    quarantined: list[str] = field(default_factory=list)
    deletion_marks: list[str] = field(default_factory=list)
    resolved_mismatches: list[str] = field(default_factory=list)
    restored_documents: list[str] = field(default_factory=list)

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

    async def restore_active_document(self, version_id: str) -> bool:
        self.restored_documents.append(version_id)
        return True

    async def resolve_published_job(self, version_id: str) -> None:
        return None

    async def fence_pending_deletions(self, limit: int) -> tuple[str, ...]:
        selected = self.pending_deletions[:limit]
        self.pending_deletions = self.pending_deletions[len(selected) :]
        self.fenced_deletions = (*self.fenced_deletions, *selected)
        return tuple(value.document_id for value in selected)

    async def list_deleted_documents(self, limit: int) -> tuple[DeletedDocument, ...]:
        ready = (
            (*self.deleted, *self.fenced_deletions)
            if self.deletion_fence_ready
            else self.deleted
        )
        return (*ready, *self.completed_deletions)[:limit]

    async def quarantine(self, version_id: str) -> None:
        self.quarantined.append(version_id)

    async def mark_deletion_reconciled(self, document_id: str) -> bool:
        first_completion = any(
            value.document_id == document_id
            for value in (*self.deleted, *self.fenced_deletions)
        )
        completed = next(
            (
                value
                for value in (*self.deleted, *self.fenced_deletions)
                if value.document_id == document_id
            ),
            None,
        )
        if first_completion:
            self.deletion_marks.append(document_id)
        self.deleted = tuple(
            value for value in self.deleted if value.document_id != document_id
        )
        self.fenced_deletions = tuple(
            value for value in self.fenced_deletions if value.document_id != document_id
        )
        if completed is not None:
            self.completed_deletions = (*self.completed_deletions, completed)
        return first_completion


@dataclass
class _Dispatcher:
    dispatched: list[str] = field(default_factory=list)
    fail: bool = False
    fail_after_publish: bool = False

    async def redispatch(self, row: OutboxRecord) -> None:
        if self.fail:
            raise RedisError("offline")
        self.dispatched.append(row.aggregate_id)
        if self.fail_after_publish:
            raise RuntimeError("SQL mark failed after Redis publish")


@dataclass
class _AliasStore:
    switched_generations: list[str] = field(default_factory=list)

    async def switch_active_alias(self, index_generation: str) -> bool:
        self.switched_generations.append(index_generation)
        return True


@dataclass
class _SelectivePublisher:
    failures: set[str]
    published: list[str] = field(default_factory=list)

    async def publish(self, version_id: str) -> None:
        if version_id in self.failures:
            raise RuntimeError(f"transient {version_id}")
        self.published.append(version_id)


class _FirstDeactivateFails(_LifecycleStore):
    async def deactivate(self, context: StagingContext) -> None:
        if context.document_version_id == "version-2":
            raise RuntimeError("transient deactivation")
        await super().deactivate(context)


def _system(
    *,
    corrupt_manifest: bool = False,
    corrupt_canonical: bool = False,
    alias_store: object | None = None,
) -> tuple[
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
        alias_store=cast(Any, alias_store),
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


async def test_publish_switches_active_alias_after_durable_publication() -> None:
    alias_store = _AliasStore()
    publisher, repository, parents, children, events = _system(
        alias_store=alias_store
    )

    await publisher.publish("version-2")

    assert repository.active_version_id == "version-2"
    assert alias_store.switched_generations == ["index-v2"]
    assert events[-1] == "finalize:version-2"
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
        artifacts=_Artifacts({}),
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
        artifacts=_Artifacts({}),
    ).run_once()

    assert report.quarantined_versions == ("version-2",)
    assert state.quarantined == ["version-2"]
    assert repository.active_version_id == "version-1"


async def test_publisher_rechecks_document_writability_before_each_physical_step() -> (
    None
):
    publisher, repository, parents, children, _ = _system()
    repository.writable_until_check = 1

    with pytest.raises(PublicationIntegrityError, match="writable"):
        await publisher.publish("version-2")

    assert parents.active == {"version-1", "version-2"}
    assert children.active == {"version-1"}
    assert repository.active_version_id == "version-1"


async def test_publisher_checks_lease_before_every_physical_mutation() -> None:
    publisher, repository, parents, children, events = _system()
    checks = 0

    async def lease_fence() -> None:
        nonlocal checks
        checks += 1
        if checks == 2:
            raise RuntimeError("lease lost before child activation")

    with pytest.raises(RuntimeError, match="lease lost"):
        await publisher.publish("version-2", before_side_effect=lease_fence)

    assert events == ["parents.activate:version-2"]
    assert parents.active == {"version-1", "version-2"}
    assert children.active == {"version-1"}
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
        artifacts=_Artifacts({}),
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
        artifacts=_Artifacts({}),
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
        user_id="user-1",
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
        artifacts=_Artifacts({}),
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
        artifacts=_Artifacts({}),
    ).run_once()

    assert report.redispatched_jobs == ()
    assert report.reclaimed_jobs == ("job-2",)


async def test_outbox_sql_mark_failure_does_not_block_later_deletion() -> None:
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
    deleted = DeletedDocument(
        user_id="user-1",
        document_id="document-1",
        versions=(_context("version-1", 1),),
    )
    state = _ReconciliationRepository(pending=(outbox,), deleted=(deleted,))
    dispatcher = _Dispatcher(fail_after_publish=True)

    report = await IngestionReconciler(
        repository=state,
        publisher=publisher,
        dispatcher=dispatcher,
        parent_store=parents,
        child_store=children,
        artifacts=_Artifacts({}),
    ).run_once()

    assert dispatcher.dispatched == ["job-1"]
    assert report.redispatched_jobs == ()
    assert report.reconciled_deletions == ("document-1",)


async def test_inactive_only_deleted_document_removes_all_physical_data_and_artifacts() -> (
    None
):
    publisher, _, parents, children, _ = _system()
    artifacts = _Artifacts(
        {
            "artifact://documents/user-1/document-1/version-1/source/upload.txt": b"x",
            "artifact://documents/user-1/document-1/version-1/fragments/1.json": b"y",
        }
    )
    deleted = DeletedDocument(
        user_id="user-1",
        document_id="document-1",
        versions=(_context("version-1", 1), _context("version-2", 2)),
    )
    state = _ReconciliationRepository(deleted=(deleted,))

    report = await IngestionReconciler(
        repository=state,
        publisher=publisher,
        dispatcher=_Dispatcher(),
        parent_store=parents,
        child_store=children,
        artifacts=artifacts,
    ).run_once()

    assert report.reconciled_deletions == ("document-1",)
    assert parents.totals == {}
    assert children.totals == {}
    assert artifacts.payloads == {}
    assert artifacts.deleted_scopes == [("user-1", "document-1")]


async def test_failed_first_deletion_does_not_block_unrelated_document() -> None:
    publisher, _, parents, children, _ = _system()
    artifacts = _Artifacts({}, fail_scopes={("user-1", "document-1")})
    first = DeletedDocument(
        user_id="user-1",
        document_id="document-1",
        versions=(_context("version-1", 1),),
    )
    second_context = _context("version-3", 3).model_copy(
        update={"document_id": "document-2"}
    )
    parents.totals["version-3"] = 1
    children.totals["version-3"] = 1
    second = DeletedDocument(
        user_id="user-1",
        document_id="document-2",
        versions=(second_context,),
    )
    state = _ReconciliationRepository(deleted=(first, second))

    report = await IngestionReconciler(
        repository=state,
        publisher=publisher,
        dispatcher=_Dispatcher(),
        parent_store=parents,
        child_store=children,
        artifacts=artifacts,
    ).run_once()

    assert report.reconciled_deletions == ("document-2",)
    assert state.deletion_marks == ["document-2"]


async def test_deletion_fence_precedes_cleanup_of_late_inflight_writes() -> None:
    publisher, _, parents, children, _ = _system()
    context = _context("version-late", 1)
    deleted = DeletedDocument(
        user_id=context.user_id,
        document_id=context.document_id,
        versions=(context,),
    )
    state = _ReconciliationRepository(pending_deletions=(deleted,))
    artifacts = _Artifacts({})
    reconciler = IngestionReconciler(
        repository=state,
        publisher=publisher,
        dispatcher=_Dispatcher(),
        parent_store=parents,
        child_store=children,
        artifacts=artifacts,
    )

    first = await reconciler.run_once()

    assert first.reconciled_deletions == ()
    assert state.fenced_deletions == (deleted,)
    assert state.deletion_marks == []

    # A publisher already in flight finishes physical writes after the durable fence.
    parents.active.add(context.document_version_id)
    children.active.add(context.document_version_id)
    parents.totals[context.document_version_id] = 1
    children.totals[context.document_version_id] = 1
    late_uri = (
        f"artifact://documents/{context.user_id}/{context.document_id}/"
        f"{context.document_version_id}/late.bin"
    )
    artifacts.payloads[late_uri] = b"late"

    state.deletion_fence_ready = True
    second = await reconciler.run_once()

    assert second.reconciled_deletions == (context.document_id,)
    assert context.document_version_id not in parents.active
    assert context.document_version_id not in children.active
    assert context.document_version_id not in parents.totals
    assert context.document_version_id not in children.totals
    assert late_uri not in artifacts.payloads
    assert state.deletion_marks == [context.document_id]


async def test_completed_deletion_resweeps_late_physical_writes_without_rereporting() -> (
    None
):
    publisher, _, parents, children, _ = _system()
    context = _context("version-late", 1)
    deleted = DeletedDocument(
        user_id=context.user_id,
        document_id=context.document_id,
        versions=(context,),
    )
    state = _ReconciliationRepository(deleted=(deleted,))
    artifacts = _Artifacts({})
    reconciler = IngestionReconciler(
        repository=state,
        publisher=publisher,
        dispatcher=_Dispatcher(),
        parent_store=parents,
        child_store=children,
        artifacts=artifacts,
    )

    first = await reconciler.run_once()
    assert first.reconciled_deletions == (context.document_id,)

    # A writer that crossed the deletion fence recreates data after completion.
    parents.active.add(context.document_version_id)
    children.active.add(context.document_version_id)
    parents.totals[context.document_version_id] = 1
    children.totals[context.document_version_id] = 1
    late_uri = (
        f"artifact://documents/{context.user_id}/{context.document_id}/"
        f"{context.document_version_id}/late-after-completion.bin"
    )
    artifacts.payloads[late_uri] = b"late"

    second = await reconciler.run_once()

    assert second.reconciled_deletions == ()
    assert context.document_version_id not in parents.totals
    assert context.document_version_id not in children.totals
    assert late_uri not in artifacts.payloads
    assert state.deletion_marks == [context.document_id]


async def test_failed_first_publication_does_not_starve_later_candidate() -> None:
    _, _, parents, children, _ = _system()
    selective = _SelectivePublisher(failures={"version-bad"})
    state = _ReconciliationRepository(stale=("version-bad", "version-good"))

    report = await IngestionReconciler(
        repository=state,
        publisher=cast(Any, selective),
        dispatcher=_Dispatcher(),
        parent_store=parents,
        child_store=children,
        artifacts=_Artifacts({}),
    ).run_once()

    assert report.repaired_versions == ("version-good",)
    assert selective.published == ["version-good"]


async def test_failed_first_deactivation_does_not_starve_later_mismatch() -> None:
    publisher, _, parents, children, events = _system()
    context_3 = _context("version-3", 3)
    parents.totals["version-3"] = 1
    children.totals["version-3"] = 1
    failing_children = _FirstDeactivateFails(
        "children", events, {"version-2", "version-3"}, children.totals
    )
    state = _ReconciliationRepository(
        mismatches=(
            PointerMismatch(context=_context("version-2", 2), action="deactivate"),
            PointerMismatch(context=context_3, action="deactivate"),
        )
    )

    report = await IngestionReconciler(
        repository=state,
        publisher=publisher,
        dispatcher=_Dispatcher(),
        parent_store=parents,
        child_store=failing_children,
        artifacts=_Artifacts({}),
    ).run_once()

    assert report.repaired_versions == ("version-3",)
    assert state.resolved_mismatches == ["version-3"]


async def test_obsolete_publication_is_physically_deactivated_before_terminal_state() -> (
    None
):
    from agentic_rag.ingestion.publisher import PublicationObsoleteError

    _, _, parents, children, _ = _system()
    parents.active.add("version-2")
    children.active.add("version-2")

    class _ObsoletePublisher:
        async def publish(self, version_id: str) -> None:
            raise PublicationObsoleteError(_context(version_id, 2), "version-3")

    state = _ReconciliationRepository(
        mismatches=(
            PointerMismatch(context=_context("version-2", 2), action="publish"),
        )
    )
    report = await IngestionReconciler(
        repository=state,
        publisher=cast(Any, _ObsoletePublisher()),
        dispatcher=_Dispatcher(),
        parent_store=parents,
        child_store=children,
        artifacts=_Artifacts({}),
    ).run_once()

    assert report.repaired_versions == ("version-2",)
    assert "version-2" not in parents.active
    assert "version-2" not in children.active
    assert state.resolved_mismatches == ["version-2"]
