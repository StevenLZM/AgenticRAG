"""Worker-level terminal publication failure convergence regressions."""

from __future__ import annotations

import hashlib
import json
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
from sqlalchemy import insert, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from agentic_rag.domain.models import JobStatus
from agentic_rag.ingestion.indexer import StagingContext
from agentic_rag.ingestion.manifest import VersionManifest
from agentic_rag.ingestion.publisher import VersionPublisher
from agentic_rag.ingestion.reconciler import IngestionReconciler
from agentic_rag.ingestion.state import IngestionJobClaim
from agentic_rag.ingestion.worker import IngestionWorker, SqlAlchemyIngestionJobStore
from agentic_rag.persistence.artifacts import LocalArtifactStore
from agentic_rag.persistence.lifecycle import (
    SqlAlchemyPublicationRepository,
    SqlAlchemyReconciliationRepository,
)
from agentic_rag.persistence.redis_queue import StreamMessage
from agentic_rag.persistence.repositories import (
    document_versions,
    documents,
    ingestion_jobs,
    metadata,
)


class _LifecycleStore:
    def __init__(self, label: str, boundary: str) -> None:
        self.label = label
        self.boundary = boundary
        self.failed = False
        self.active = {"version-1"}
        self.totals = {"version-1": 1, "version-2": 1}

    async def count_total(self, context: StagingContext) -> int:
        return self.totals[context.document_version_id]

    async def activate(self, context: StagingContext) -> None:
        self.active.add(context.document_version_id)
        self._fail(f"new_{self.label}")

    async def deactivate(self, context: StagingContext) -> None:
        self.active.discard(context.document_version_id)
        self._fail(f"old_{self.label}")

    async def delete(self, context: StagingContext) -> None:
        self.active.discard(context.document_version_id)
        self.totals.pop(context.document_version_id, None)

    def _fail(self, boundary: str) -> None:
        if not self.failed and self.boundary == boundary:
            self.failed = True
            raise RuntimeError(f"injected after {boundary}")


class _Broker:
    def __init__(self) -> None:
        self.acked: list[str] = []

    async def publish(self, *args: Any, **kwargs: Any) -> str:
        return "published"

    async def consume(self, *args: Any, **kwargs: Any) -> list[StreamMessage]:
        return []

    async def reclaim(self, *args: Any, **kwargs: Any) -> list[StreamMessage]:
        return []

    async def ack(self, stream: str, group: str, message_id: str) -> None:
        self.acked.append(message_id)

    async def dead_letter(self, *args: Any, **kwargs: Any) -> None:
        return None


class _PublishingGraph:
    def __init__(
        self,
        publisher: VersionPublisher,
        jobs: SqlAlchemyIngestionJobStore,
    ) -> None:
        self._publisher = publisher
        self._jobs = jobs

    async def run(self, claim: IngestionJobClaim) -> dict[str, object]:
        await self._publisher.publish(claim.document_version_id)
        await self._jobs.complete(claim)
        return {"terminal_status": "completed"}


class _NoopDispatcher:
    async def redispatch(self, row: object) -> None:
        raise AssertionError(f"unexpected outbox row: {row}")


@pytest.fixture
async def recovery_system(
    tmp_path: Path,
) -> AsyncIterator[
    tuple[
        async_sessionmaker[AsyncSession],
        LocalArtifactStore,
        SqlAlchemyIngestionJobStore,
    ]
]:
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'recovery.sqlite'}")
    async with engine.begin() as connection:
        await connection.run_sync(metadata.create_all)
    factory = async_sessionmaker(engine, expire_on_commit=False)
    artifacts = LocalArtifactStore(tmp_path / "artifacts")
    now = datetime.now(UTC)
    canonical = artifacts.put_bytes(
        "documents/user-1/document-1/version-2/canonical/docling-v1/ingestion-v1/1.json",
        b'{"schema":"canonical"}',
    )
    manifest = VersionManifest(
        canonical_ast_sha256=canonical.sha256,
        parent_count=1,
        child_count=1,
        embedding_model="text-embedding-v3",
        embedding_dimensions=1024,
        index_generation="index-v2",
    )
    encoded = json.dumps(
        manifest.payload(),
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode()
    manifest_ref = artifacts.put_bytes(
        "documents/user-1/document-1/version-2/manifests/index-v2/"
        f"{hashlib.sha256(encoded).hexdigest()}.json",
        encoded,
    )
    async with factory.begin() as session:
        await session.execute(
            insert(documents).values(
                id="document-1",
                user_id="user-1",
                source_type="pdf",
                filename="report.pdf",
                mime_type="application/pdf",
                content_hash="a" * 64,
                status="active",
                active_version_id="version-1",
                source_trust="untrusted",
                created_at=now,
                updated_at=now,
            )
        )
        for version_id, number, status in (
            ("version-1", 1, "active"),
            ("version-2", 2, "building"),
        ):
            await session.execute(
                insert(document_versions).values(
                    id=version_id,
                    document_id="document-1",
                    version_no=number,
                    parser_version="docling-v1",
                    pipeline_version="ingestion-v1",
                    canonical_ast_path=(canonical.uri if number == 2 else None),
                    canonical_ast_hash=(canonical.sha256 if number == 2 else None),
                    manifest_path=(manifest_ref.uri if number == 2 else None),
                    manifest_hash=(manifest_ref.sha256 if number == 2 else None),
                    parent_count=1,
                    child_count=1,
                    embedding_version="text-embedding-v3",
                    index_generation="index-v2",
                    status=status,
                    created_at=now,
                )
            )
        await session.execute(
            insert(ingestion_jobs).values(
                id="job-1",
                user_id="user-1",
                document_id="document-1",
                document_version_id="version-2",
                status="queued",
                attempt_count=2,
                created_at=now,
                updated_at=now,
            )
        )
    try:
        yield factory, artifacts, SqlAlchemyIngestionJobStore(factory)
    finally:
        await engine.dispose()


@pytest.mark.parametrize(
    "failure_boundary", ("new_parent", "new_child", "old_child", "old_parent")
)
@pytest.mark.asyncio
async def test_terminal_publication_boundary_failure_remains_repairable(
    recovery_system: tuple[
        async_sessionmaker[AsyncSession],
        LocalArtifactStore,
        SqlAlchemyIngestionJobStore,
    ],
    failure_boundary: str,
) -> None:
    factory, artifacts, jobs = recovery_system
    parents = _LifecycleStore("parent", failure_boundary)
    children = _LifecycleStore("child", failure_boundary)
    publisher = VersionPublisher(
        repository=SqlAlchemyPublicationRepository(factory),
        parent_store=parents,
        child_store=children,
        artifacts=artifacts,
    )
    broker = _Broker()
    worker = IngestionWorker(
        jobs=jobs,
        broker=broker,
        graph=_PublishingGraph(publisher, jobs),
        reconciler=None,
        worker_id="worker-1",
        heartbeat_interval_seconds=0.01,
    )
    message = StreamMessage(
        id="1-0", aggregate_id="job-1", enqueued_at=datetime.now(UTC)
    )

    await worker.process_message(message)

    async with factory() as session:
        candidate_status = await session.scalar(
            select(document_versions.c.status).where(
                document_versions.c.id == "version-2"
            )
        )
    assert candidate_status == "building"
    assert await jobs.get_status("job-1") is JobStatus.FAILED

    report = await IngestionReconciler(
        repository=SqlAlchemyReconciliationRepository(factory),
        publisher=publisher,
        dispatcher=_NoopDispatcher(),  # type: ignore[arg-type]
        parent_store=parents,
        child_store=children,
        artifacts=artifacts,
    ).run_once()

    assert report.repaired_versions == ("version-2",)
    assert parents.active == {"version-2"}
    assert children.active == {"version-2"}
    assert await jobs.get_status("job-1") is JobStatus.COMPLETED
    async with factory() as session:
        document = (
            await session.execute(
                select(documents.c.active_version_id, documents.c.status).where(
                    documents.c.id == "document-1"
                )
            )
        ).one()
        version_rows = (
            await session.execute(
                select(document_versions.c.id, document_versions.c.status)
            )
        ).all()
        versions: dict[str, str] = {
            str(row.id): str(row.status) for row in version_rows
        }
    assert document.active_version_id == "version-2"
    assert document.status == "active"
    assert versions == {"version-1": "inactive", "version-2": "active"}
    assert broker.acked == ["1-0"]
