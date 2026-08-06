"""Opt-in publication integration against explicitly disposable MySQL and ES.

The shared fixture requires both ``AGENTIC_RAG_TEST_MYSQL_DSN`` and
``AGENTIC_RAG_TEST_ELASTICSEARCH_URL`` and never guesses developer services.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from uuid import uuid4

import pytest
from elasticsearch import AsyncElasticsearch
from sqlalchemy import delete, insert, select, update
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from agentic_rag.domain.models import DocumentStatus, DocumentVersionStatus, UserScope
from agentic_rag.ingestion.indexer import IndexWriter, StagingContext
from agentic_rag.ingestion.publisher import VersionPublisher
from agentic_rag.ingestion.reconciler import IngestionReconciler
from agentic_rag.persistence.artifacts import LocalArtifactStore
from agentic_rag.persistence.elasticsearch import ElasticsearchChildIndexStore
from agentic_rag.persistence.lifecycle import (
    SqlAlchemyPublicationRepository,
    SqlAlchemyReconciliationRepository,
)
from agentic_rag.persistence.repositories import (
    OutboxRecord,
    SqlAlchemyParentRepository,
    document_versions,
    documents,
    parent_chunks,
)
from agentic_rag.persistence.staging import SqlAlchemyParentStagingStore
from tests.integration.ingestion.test_staging_index import (
    FixedEmbedding,
    _chunks,
    infrastructure,
)


pytestmark = pytest.mark.integration

# Re-exporting the decorated fixture lets this module share the same explicit-DSN
# gate without silently inventing a second infrastructure contract.
infrastructure = infrastructure


class _FailOnceAtBoundary:
    def __init__(self, delegate: Any, *, label: str, boundary: str) -> None:
        self._delegate = delegate
        self._label = label
        self._boundary = boundary
        self._failed = False

    async def count_total(self, context: StagingContext) -> int:
        return await self._delegate.count_total(context)

    async def activate(self, context: StagingContext) -> None:
        await self._delegate.activate(context)
        self._fail(f"new_{self._label}")

    async def deactivate(self, context: StagingContext) -> None:
        await self._delegate.deactivate(context)
        self._fail(f"old_{self._label}")

    async def delete(self, context: StagingContext) -> None:
        await self._delegate.delete(context)

    def _fail(self, boundary: str) -> None:
        if not self._failed and self._boundary == boundary:
            self._failed = True
            raise RuntimeError(f"injected after {boundary}")


class _FailOnceAfterFinalize:
    def __init__(self, delegate: SqlAlchemyPublicationRepository) -> None:
        self._delegate = delegate
        self._failed = False

    async def get_target(self, version_id: str) -> Any:
        return await self._delegate.get_target(version_id)

    async def is_writable(self, context: StagingContext) -> bool:
        return await self._delegate.is_writable(context)

    async def finalize(self, target: Any) -> None:
        await self._delegate.finalize(target)
        if not self._failed:
            self._failed = True
            raise RuntimeError("injected after finalize")

    async def quarantine(self, version_id: str) -> None:
        await self._delegate.quarantine(version_id)


class _BarrierPublicationRepository:
    def __init__(
        self,
        delegate: SqlAlchemyPublicationRepository,
        barrier: asyncio.Barrier,
    ) -> None:
        self._delegate = delegate
        self._barrier = barrier

    async def get_target(self, version_id: str) -> Any:
        return await self._delegate.get_target(version_id)

    async def is_writable(self, context: StagingContext) -> bool:
        return await self._delegate.is_writable(context)

    async def finalize(self, target: Any) -> None:
        await self._barrier.wait()
        await self._delegate.finalize(target)

    async def quarantine(self, version_id: str) -> None:
        await self._delegate.quarantine(version_id)


class _NoopDispatcher:
    async def redispatch(self, row: OutboxRecord) -> None:
        raise AssertionError(f"unexpected outbox row {row.id}")


@pytest.mark.parametrize(
    "failure_boundary",
    ("new_parent", "new_child", "old_child", "old_parent", "finalize"),
)
async def test_real_stores_reconcile_interrupted_replacement_and_deletion(
    infrastructure: tuple[async_sessionmaker[AsyncSession], AsyncElasticsearch],
    tmp_path: Path,
    failure_boundary: str,
) -> None:
    factory, elasticsearch = infrastructure
    unique = uuid4().hex
    context = StagingContext(
        user_id=f"publisher-user-{unique}",
        document_id=str(uuid4()),
        document_version_id=str(uuid4()),
        version_no=1,
        pipeline_version="ingestion-v1",
        embedding_version="text-embedding-v3",
        index_generation=f"test-{unique}",
    )
    chunks = _chunks(context)
    artifacts = LocalArtifactStore(tmp_path / "artifacts")
    canonical = artifacts.put_bytes(
        f"documents/{context.user_id}/{context.document_id}/"
        f"{context.document_version_id}/canonical/docling-v1/"
        f"{context.pipeline_version}/1.json",
        b'{"schema":"canonical"}',
    )
    parent_store = SqlAlchemyParentStagingStore(factory)
    child_store = ElasticsearchChildIndexStore(elasticsearch)

    async with factory.begin() as transaction:
        await transaction.execute(
            insert(documents).values(
                id=context.document_id,
                user_id=context.user_id,
                source_type="text",
                filename="publication.txt",
                mime_type="text/plain",
                content_hash="a" * 64,
                status=DocumentStatus.PROCESSING.value,
                source_trust="untrusted",
            )
        )
        await transaction.execute(
            insert(document_versions).values(
                id=context.document_version_id,
                document_id=context.document_id,
                version_no=1,
                parser_version="docling-v1",
                pipeline_version=context.pipeline_version,
                parent_count=0,
                child_count=0,
                embedding_version=context.embedding_version,
                index_generation=context.index_generation,
                status=DocumentVersionStatus.UPLOADED.value,
            )
        )

    publisher = VersionPublisher(
        repository=SqlAlchemyPublicationRepository(factory),
        parent_store=parent_store,
        child_store=child_store,
        artifacts=artifacts,
    )
    try:
        await IndexWriter(
            embedding=FixedEmbedding(),
            parent_store=parent_store,
            child_store=child_store,
            artifacts=artifacts,
        ).stage(chunks, context=context, canonical_ast=canonical)

        await publisher.publish(context.document_version_id)
        await publisher.publish(context.document_version_id)

        async with factory() as session:
            document = (
                await session.execute(
                    select(documents).where(documents.c.id == context.document_id)
                )
            ).mappings().one()
            version_status = await session.scalar(
                select(document_versions.c.status).where(
                    document_versions.c.id == context.document_version_id
                )
            )
            parent_statuses = tuple(
                (
                    await session.execute(
                        select(parent_chunks.c.status).where(
                            parent_chunks.c.document_version_id
                            == context.document_version_id
                        )
                    )
                ).scalars()
            )

        assert document["active_version_id"] == context.document_version_id
        assert document["status"] == DocumentStatus.ACTIVE.value
        assert version_status == DocumentVersionStatus.ACTIVE.value
        assert parent_statuses == ("active",)
        assert await child_store.count_active(context) == 1
        assert await child_store.count_total(context) == 1

        replacement = context.model_copy(
            update={"document_version_id": str(uuid4()), "version_no": 2}
        )
        replacement_chunks = _chunks(replacement)
        replacement_canonical = artifacts.put_bytes(
            f"documents/{replacement.user_id}/{replacement.document_id}/"
            f"{replacement.document_version_id}/canonical/docling-v1/"
            f"{replacement.pipeline_version}/1.json",
            b'{"schema":"canonical-v2"}',
        )
        async with factory.begin() as transaction:
            await transaction.execute(
                insert(document_versions).values(
                    id=replacement.document_version_id,
                    document_id=replacement.document_id,
                    version_no=replacement.version_no,
                    parser_version="docling-v1",
                    pipeline_version=replacement.pipeline_version,
                    parent_count=0,
                    child_count=0,
                    embedding_version=replacement.embedding_version,
                    index_generation=replacement.index_generation,
                    status=DocumentVersionStatus.UPLOADED.value,
                )
            )
        await IndexWriter(
            embedding=FixedEmbedding(),
            parent_store=parent_store,
            child_store=child_store,
            artifacts=artifacts,
        ).stage(
            replacement_chunks,
            context=replacement,
            canonical_ast=replacement_canonical,
        )

        base_repository = SqlAlchemyPublicationRepository(factory)
        interrupted_publisher = VersionPublisher(
            repository=(
                _FailOnceAfterFinalize(base_repository)
                if failure_boundary == "finalize"
                else base_repository
            ),
            parent_store=_FailOnceAtBoundary(
                parent_store, label="parent", boundary=failure_boundary
            ),
            child_store=_FailOnceAtBoundary(
                child_store, label="child", boundary=failure_boundary
            ),
            artifacts=artifacts,
        )
        with pytest.raises(RuntimeError, match="injected"):
            await interrupted_publisher.publish(replacement.document_version_id)

        async with factory() as session:
            visible = await SqlAlchemyParentRepository(session).get_many(
                [chunks[0].id, replacement_chunks[0].id],
                UserScope(user_id=context.user_id),
            )
        expected_visible = (
            [replacement_chunks[0].id]
            if failure_boundary == "finalize"
            else [chunks[0].id]
        )
        assert [parent.id for parent in visible] == expected_visible

        await _assert_child_search_pointer_gates_parent_hydration(
            elasticsearch,
            child_store,
            factory,
            context,
        )

        reconciler = IngestionReconciler(
            repository=SqlAlchemyReconciliationRepository(factory),
            publisher=publisher,
            dispatcher=_NoopDispatcher(),
            parent_store=parent_store,
            child_store=child_store,
            artifacts=artifacts,
        )
        repair = await reconciler.run_once()
        if failure_boundary == "finalize":
            assert repair.repaired_versions == ()
        else:
            assert repair.repaired_versions == (replacement.document_version_id,)
        assert await child_store.count_active(context) == 0
        assert await child_store.count_active(replacement) == 1
        await _assert_child_search_pointer_gates_parent_hydration(
            elasticsearch,
            child_store,
            factory,
            context,
        )

        async with factory.begin() as transaction:
            await transaction.execute(
                update(documents)
                .where(documents.c.id == context.document_id)
                .values(
                    status=DocumentStatus.DELETED.value,
                    active_version_id=None,
                    deletion_status="pending",
                )
            )
        fence = await reconciler.run_once()
        assert fence.reconciled_deletions == ()
        async with factory.begin() as transaction:
            await transaction.execute(
                update(documents)
                .where(documents.c.id == context.document_id)
                .values(deletion_fenced_at=datetime.now(UTC) - timedelta(hours=1))
            )
        deletion = await reconciler.run_once()
        assert deletion.reconciled_deletions == (context.document_id,)
        assert await parent_store.count_total(context) == 0
        assert await parent_store.count_total(replacement) == 0
        assert await child_store.count_total(context) == 0
        assert await child_store.count_total(replacement) == 0
    finally:
        await elasticsearch.indices.delete(
            index=child_store.index_name(context.index_generation),
            ignore_unavailable=True,
        )
        async with factory.begin() as transaction:
            await transaction.execute(
                delete(documents).where(documents.c.id == context.document_id)
            )


async def _assert_child_search_pointer_gates_parent_hydration(
    elasticsearch: AsyncElasticsearch,
    child_store: ElasticsearchChildIndexStore,
    factory: async_sessionmaker[AsyncSession],
    context: StagingContext,
) -> None:
    response = await elasticsearch.search(
        index=child_store.index_name(context.index_generation),
        query={
            "bool": {
                "filter": [
                    {"term": {"user_id": context.user_id}},
                    {"term": {"document_id": context.document_id}},
                    {"term": {"search_type": "document"}},
                    {"term": {"is_active": True}},
                ]
            }
        },
        size=10,
    )
    parent_ids = [hit["_source"]["parent_id"] for hit in response["hits"]["hits"]]
    async with factory() as session:
        document_pointer = await session.scalar(
            select(documents.c.active_version_id).where(
                documents.c.id == context.document_id
            )
        )
        hydrated = await SqlAlchemyParentRepository(session).get_many(
            parent_ids,
            UserScope(user_id=context.user_id),
        )
    assert all(parent.document_version_id == document_pointer for parent in hydrated)


async def test_real_concurrent_finalizers_converge_to_newest_winner(
    infrastructure: tuple[async_sessionmaker[AsyncSession], AsyncElasticsearch],
    tmp_path: Path,
) -> None:
    factory, elasticsearch = infrastructure
    unique = uuid4().hex
    base = StagingContext(
        user_id=f"concurrent-user-{unique}",
        document_id=str(uuid4()),
        document_version_id=str(uuid4()),
        version_no=1,
        pipeline_version="ingestion-v1",
        embedding_version="text-embedding-v3",
        index_generation=f"test-{unique}",
    )
    contexts = [
        base,
        base.model_copy(update={"document_version_id": str(uuid4()), "version_no": 2}),
        base.model_copy(update={"document_version_id": str(uuid4()), "version_no": 3}),
    ]
    artifacts = LocalArtifactStore(tmp_path / "artifacts")
    parent_store = SqlAlchemyParentStagingStore(factory)
    child_store = ElasticsearchChildIndexStore(elasticsearch)

    async with factory.begin() as transaction:
        await transaction.execute(
            insert(documents).values(
                id=base.document_id,
                user_id=base.user_id,
                source_type="text",
                filename="concurrent.txt",
                mime_type="text/plain",
                content_hash="d" * 64,
                status=DocumentStatus.PROCESSING.value,
                source_trust="untrusted",
            )
        )
        for context in contexts:
            await transaction.execute(
                insert(document_versions).values(
                    id=context.document_version_id,
                    document_id=context.document_id,
                    version_no=context.version_no,
                    parser_version="docling-v1",
                    pipeline_version=context.pipeline_version,
                    parent_count=0,
                    child_count=0,
                    embedding_version=context.embedding_version,
                    index_generation=context.index_generation,
                    status=DocumentVersionStatus.UPLOADED.value,
                )
            )

    try:
        writer = IndexWriter(
            embedding=FixedEmbedding(),
            parent_store=parent_store,
            child_store=child_store,
            artifacts=artifacts,
        )
        for context in contexts:
            version_chunks = _chunks(context)
            canonical = artifacts.put_bytes(
                f"documents/{context.user_id}/{context.document_id}/"
                f"{context.document_version_id}/canonical/docling-v1/"
                f"{context.pipeline_version}/1.json",
                f'{{"version":{context.version_no}}}'.encode(),
            )
            await writer.stage(
                version_chunks,
                context=context,
                canonical_ast=canonical,
            )

        clean_repository = SqlAlchemyPublicationRepository(factory)
        clean_publisher = VersionPublisher(
            repository=clean_repository,
            parent_store=parent_store,
            child_store=child_store,
            artifacts=artifacts,
        )

        barrier = asyncio.Barrier(2)
        contenders = [
            VersionPublisher(
                repository=_BarrierPublicationRepository(clean_repository, barrier),
                parent_store=parent_store,
                child_store=child_store,
                artifacts=artifacts,
            )
            for _ in range(2)
        ]
        outcomes = await asyncio.wait_for(
            asyncio.gather(
                contenders[0].publish(contexts[1].document_version_id),
                contenders[1].publish(contexts[2].document_version_id),
                return_exceptions=True,
            ),
            timeout=20,
        )
        assert any(isinstance(outcome, Exception) for outcome in outcomes)

        reconciler = IngestionReconciler(
            repository=SqlAlchemyReconciliationRepository(factory),
            publisher=clean_publisher,
            dispatcher=_NoopDispatcher(),
            parent_store=parent_store,
            child_store=child_store,
            artifacts=artifacts,
        )
        await reconciler.run_once()

        newest = contexts[2]
        async with factory() as session:
            pointer = await session.scalar(
                select(documents.c.active_version_id).where(
                    documents.c.id == base.document_id
                )
            )
            status_rows = (
                await session.execute(
                    select(document_versions.c.id, document_versions.c.status).where(
                        document_versions.c.document_id == base.document_id
                    )
                )
            ).all()
            statuses: dict[str, str] = {
                str(row.id): str(row.status) for row in status_rows
            }
        assert pointer == newest.document_version_id
        assert statuses[newest.document_version_id] == DocumentVersionStatus.ACTIVE.value
        assert statuses[contexts[1].document_version_id] == DocumentVersionStatus.INACTIVE.value
        assert await child_store.count_active(base) == 0
        assert await child_store.count_active(contexts[1]) == 0
        assert await child_store.count_active(newest) == 1
        await _assert_child_search_pointer_gates_parent_hydration(
            elasticsearch,
            child_store,
            factory,
            base,
        )
    finally:
        await elasticsearch.indices.delete(
            index=child_store.index_name(base.index_generation),
            ignore_unavailable=True,
        )
        async with factory.begin() as transaction:
            await transaction.execute(
                delete(documents).where(documents.c.id == base.document_id)
            )
