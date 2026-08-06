"""Opt-in publication integration against explicitly disposable MySQL and ES.

The shared fixture requires both ``AGENTIC_RAG_TEST_MYSQL_DSN`` and
``AGENTIC_RAG_TEST_ELASTICSEARCH_URL`` and never guesses developer services.
"""

from __future__ import annotations

from pathlib import Path
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


class _FailOnceAfterChildActivation:
    def __init__(self, delegate: ElasticsearchChildIndexStore) -> None:
        self._delegate = delegate
        self._failed = False

    async def count_total(self, context: StagingContext) -> int:
        return await self._delegate.count_total(context)

    async def activate(self, context: StagingContext) -> None:
        await self._delegate.activate(context)
        if not self._failed:
            self._failed = True
            raise RuntimeError("injected after new Child activation")

    async def deactivate(self, context: StagingContext) -> None:
        await self._delegate.deactivate(context)

    async def delete(self, context: StagingContext) -> None:
        await self._delegate.delete(context)


class _NoopDispatcher:
    async def redispatch(self, row: OutboxRecord) -> None:
        raise AssertionError(f"unexpected outbox row {row.id}")


async def test_real_stores_reconcile_interrupted_replacement_and_deletion(
    infrastructure: tuple[async_sessionmaker[AsyncSession], AsyncElasticsearch],
    tmp_path: Path,
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

        interrupted_publisher = VersionPublisher(
            repository=SqlAlchemyPublicationRepository(factory),
            parent_store=parent_store,
            child_store=_FailOnceAfterChildActivation(child_store),
            artifacts=artifacts,
        )
        with pytest.raises(RuntimeError, match="injected"):
            await interrupted_publisher.publish(replacement.document_version_id)

        async with factory() as session:
            visible = await SqlAlchemyParentRepository(session).get_many(
                [chunks[0].id, replacement_chunks[0].id],
                UserScope(user_id=context.user_id),
            )
        assert [parent.id for parent in visible] == [chunks[0].id]

        reconciler = IngestionReconciler(
            repository=SqlAlchemyReconciliationRepository(factory),
            publisher=publisher,
            dispatcher=_NoopDispatcher(),
            parent_store=parent_store,
            child_store=child_store,
        )
        repair = await reconciler.run_once()
        assert repair.repaired_versions == (replacement.document_version_id,)
        assert await child_store.count_active(context) == 0
        assert await child_store.count_active(replacement) == 1

        async with factory.begin() as transaction:
            await transaction.execute(
                update(documents)
                .where(documents.c.id == context.document_id)
                .values(
                    status=DocumentStatus.DELETED.value,
                    active_version_id=None,
                )
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
