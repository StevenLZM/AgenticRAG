"""Opt-in staging integration against explicitly disposable MySQL and ES.

Set both ``AGENTIC_RAG_TEST_MYSQL_DSN`` and
``AGENTIC_RAG_TEST_ELASTICSEARCH_URL``. No developer database or Elasticsearch
endpoint is guessed.
"""

from __future__ import annotations

import asyncio
import os
from collections.abc import AsyncIterator, Sequence
from pathlib import Path
from uuid import uuid4

import pytest
from alembic import command
from alembic.config import Config
from elasticsearch import AsyncElasticsearch
from elasticsearch.exceptions import ApiError
from elastic_transport import TransportError
from sqlalchemy import delete, insert, select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker

from agentic_rag.domain.models import DocumentVersionStatus
from agentic_rag.ingestion.chunker import AstLocator, AstSpan, ChildChunk, ParentChunk
from agentic_rag.ingestion.indexer import IndexWriter, StagingContext
from agentic_rag.persistence.artifacts import LocalArtifactStore
from agentic_rag.persistence.elasticsearch import ElasticsearchChildIndexStore
from agentic_rag.persistence.mysql import create_mysql_engine, create_session_factory
from agentic_rag.persistence.repositories import (
    document_versions,
    documents,
    parent_chunks,
)
from agentic_rag.persistence.staging import SqlAlchemyParentStagingStore


pytestmark = pytest.mark.integration


class FixedEmbedding:
    async def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return [[float(index % 7) / 7 for index in range(1024)] for _ in texts]

    async def embed_query(self, text: str) -> list[float]:
        return [0.0] * 1024


@pytest.fixture
async def infrastructure() -> AsyncIterator[
    tuple[async_sessionmaker[AsyncSession], AsyncElasticsearch]
]:
    mysql_dsn = os.getenv("AGENTIC_RAG_TEST_MYSQL_DSN")
    elasticsearch_url = os.getenv("AGENTIC_RAG_TEST_ELASTICSEARCH_URL")
    if not mysql_dsn or not elasticsearch_url:
        pytest.skip(
            "set AGENTIC_RAG_TEST_MYSQL_DSN and "
            "AGENTIC_RAG_TEST_ELASTICSEARCH_URL to disposable services"
        )

    engine: AsyncEngine = create_mysql_engine(mysql_dsn, pool_pre_ping=True)
    elasticsearch = AsyncElasticsearch(elasticsearch_url)
    try:
        async with engine.connect() as connection:
            await connection.execute(select(1))
        await elasticsearch.info()
    except (OSError, SQLAlchemyError, ApiError, TransportError) as error:
        await engine.dispose()
        await elasticsearch.close()
        pytest.skip(f"disposable staging services unavailable: {type(error).__name__}")
    migration = Config("alembic.ini")
    migration.set_main_option("sqlalchemy.url", mysql_dsn)
    await asyncio.to_thread(command.upgrade, migration, "head")
    try:
        yield create_session_factory(engine), elasticsearch
    finally:
        await engine.dispose()
        await elasticsearch.close()


def _chunks(context: StagingContext) -> list[ParentChunk]:
    locator = AstLocator(
        spans=(
            AstSpan(
                canonical_path="#/text_blocks/0",
                block_id="block-integration",
                page_from=1,
                page_to=1,
                char_from=0,
                char_to=11,
                parent_char_from=0,
                parent_char_to=11,
            ),
        ),
        segment_ordinal=0,
        parent_char_from=0,
        parent_char_to=11,
    )
    child = ChildChunk(
        id=f"child-{uuid4().hex}",
        parent_id=f"parent-{uuid4().hex}",
        parent_ordinal=0,
        document_id=context.document_id,
        document_version_id=context.document_version_id,
        user_id=context.user_id,
        ordinal=0,
        heading_path=("Integration",),
        heading_ast_locators=("#/text_blocks/heading",),
        content_type="paragraph",
        content="hello index",
        contextualized_content="Integration\nhello index",
        token_count=3,
        page_from=1,
        page_to=1,
        ast_locator=locator,
        content_hash="b" * 64,
    )
    parent = ParentChunk(
        id=child.parent_id,
        document_id=context.document_id,
        document_version_id=context.document_version_id,
        user_id=context.user_id,
        ordinal=0,
        heading_path=("Integration",),
        heading_ast_locators=("#/text_blocks/heading",),
        content_type="paragraph",
        content="hello index",
        token_count=2,
        page_from=1,
        page_to=1,
        ast_locator=locator,
        content_hash="c" * 64,
        children=(child,),
    )
    return [parent]


@pytest.mark.asyncio
async def test_staging_writes_inactive_parent_child_and_manifest_last(
    infrastructure: tuple[async_sessionmaker[AsyncSession], AsyncElasticsearch],
    tmp_path: Path,
) -> None:
    factory, elasticsearch = infrastructure
    unique = uuid4().hex
    context = StagingContext(
        user_id=f"staging-user-{unique}",
        document_id=str(uuid4()),
        document_version_id=str(uuid4()),
        index_generation=f"test-{unique}",
    )
    chunks = _chunks(context)
    artifacts = LocalArtifactStore(tmp_path / "artifacts")
    canonical = artifacts.put_bytes("canonical.json", b'{"schema":"canonical"}')
    child_store = ElasticsearchChildIndexStore(elasticsearch)
    parent_store = SqlAlchemyParentStagingStore(factory)

    async with factory.begin() as transaction:
        await transaction.execute(
            insert(documents).values(
                id=context.document_id,
                user_id=context.user_id,
                source_type="text",
                filename="integration.txt",
                mime_type="text/plain",
                content_hash="a" * 64,
                status="processing",
                source_trust="untrusted",
            )
        )
        await transaction.execute(
            insert(document_versions).values(
                id=context.document_version_id,
                document_id=context.document_id,
                version_no=1,
                parser_version="docling-v1",
                pipeline_version="ingestion-v1",
                parent_count=0,
                child_count=0,
                embedding_version="text-embedding-v3",
                index_generation=context.index_generation,
                status=DocumentVersionStatus.UPLOADED.value,
            )
        )

    writer = IndexWriter(
        embedding=FixedEmbedding(),
        parent_store=parent_store,
        child_store=child_store,
        artifacts=artifacts,
    )
    try:
        manifest = await writer.stage(
            chunks, context=context, canonical_ast=canonical
        )

        async with factory() as session:
            parent_row = (
                await session.execute(
                    select(parent_chunks).where(
                        parent_chunks.c.document_version_id
                        == context.document_version_id
                    )
                )
            ).mappings().one()
            version_row = (
                await session.execute(
                    select(document_versions).where(
                        document_versions.c.id == context.document_version_id
                    )
                )
            ).mappings().one()

        assert parent_row["status"] == "inactive"
        assert version_row["status"] == DocumentVersionStatus.BUILDING.value
        assert version_row["parent_count"] == 1
        assert version_row["child_count"] == 1
        assert version_row["manifest_hash"] == manifest.manifest_hash
        assert version_row["manifest_path"].startswith("artifact://")
        assert await child_store.count(context) == 1
        assert await child_store.count_active(context) == 0
    finally:
        await elasticsearch.indices.delete(
            index=child_store.index_name(context.index_generation),
            ignore_unavailable=True,
        )
        async with factory.begin() as transaction:
            await transaction.execute(
                delete(documents).where(documents.c.id == context.document_id)
            )
