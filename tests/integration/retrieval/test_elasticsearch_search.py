"""Opt-in Elasticsearch integration tests for hybrid Child recall.

Set ``AGENTIC_RAG_TEST_ELASTICSEARCH_URL`` to a disposable Elasticsearch
endpoint.  The fixture never guesses a developer endpoint.
"""

from __future__ import annotations

import os
from collections.abc import AsyncIterator
from dataclasses import dataclass
from uuid import uuid4

import pytest
from elasticsearch import AsyncElasticsearch

from agentic_rag.ingestion.chunker import AstLocator, AstSpan, ChildChunk
from agentic_rag.ingestion.indexer import EmbeddedChild, StagingContext
from agentic_rag.persistence.elasticsearch import ElasticsearchChildIndexStore
from agentic_rag.retrieval.adapters.elasticsearch import (
    ElasticsearchBm25Index,
    ElasticsearchVectorIndex,
)
from agentic_rag.retrieval.models import SearchFilter

pytestmark = pytest.mark.integration


@dataclass(frozen=True)
class _IndexFixture:
    vector: ElasticsearchVectorIndex
    lexical: ElasticsearchBm25Index
    query_vector: tuple[float, ...]
    index_generation: str

    def filter(self, user_id: str) -> SearchFilter:
        return SearchFilter(user_id=user_id, index_generation=self.index_generation)


@pytest.fixture
async def index_fixture() -> AsyncIterator[_IndexFixture]:
    elasticsearch_url = os.getenv("AGENTIC_RAG_TEST_ELASTICSEARCH_URL")
    if not elasticsearch_url:
        pytest.skip("set AGENTIC_RAG_TEST_ELASTICSEARCH_URL to a disposable service")

    elasticsearch = AsyncElasticsearch(elasticsearch_url)
    generation = f"search-{uuid4().hex}"
    index = ElasticsearchChildIndexStore.index_name(generation)
    vector = tuple(0.25 for _ in range(1024))
    store = ElasticsearchChildIndexStore(elasticsearch)
    contexts = tuple(
        StagingContext(
            user_id=user_id,
            document_id=f"document-{user_id}-{uuid4().hex}",
            document_version_id=f"version-{user_id}-{uuid4().hex}",
            version_no=1,
            pipeline_version="ingestion-v1",
            embedding_version="text-embedding-v3",
            index_generation=generation,
        )
        for user_id in ("u1", "u2", "u1-inactive")
    )
    try:
        await elasticsearch.info()
        for context in contexts:
            await store.stage(
                context,
                [_embedded_child(context, vector, suffix=context.user_id)],
            )
        await store.activate(contexts[0])
        await store.activate(contexts[1])
        await elasticsearch.indices.refresh(index=index)
        yield _IndexFixture(
            vector=ElasticsearchVectorIndex(
                elasticsearch,
                index=index,
                index_generation=generation,
            ),
            lexical=ElasticsearchBm25Index(
                elasticsearch,
                index=index,
                index_generation=generation,
            ),
            query_vector=vector,
            index_generation=generation,
        )
    finally:
        await elasticsearch.indices.delete(index=index, ignore_unavailable=True)
        await elasticsearch.close()


def _embedded_child(
    context: StagingContext,
    vector: tuple[float, ...],
    *,
    suffix: str,
) -> EmbeddedChild:
    span = AstSpan(
        canonical_path="#/text_blocks/0",
        block_id=f"block-{suffix}",
        page_from=1,
        page_to=1,
        char_from=0,
        char_to=18,
        parent_char_from=0,
        parent_char_to=18,
    )
    child = ChildChunk(
        id=f"child-{suffix}-{uuid4().hex}",
        parent_id=f"parent-{suffix}-{uuid4().hex}",
        parent_ordinal=0,
        document_id=context.document_id,
        document_version_id=context.document_version_id,
        user_id=context.user_id,
        ordinal=0,
        heading_path=("Contracts",),
        heading_ast_locators=("#/text_blocks/0",),
        content_type="paragraph",
        content="termination clause",
        contextualized_content="Contracts termination clause",
        token_count=3,
        page_from=1,
        page_to=1,
        ast_locator=AstLocator(
            spans=(span,),
            segment_ordinal=0,
            parent_char_from=0,
            parent_char_to=18,
        ),
        content_hash="a" * 64,
    )
    return EmbeddedChild(
        chunk=child,
        embedding=vector,
        user_id=context.user_id,
        document_id=context.document_id,
        document_version_id=context.document_version_id,
        index_generation=context.index_generation,
    )


@pytest.mark.asyncio
async def test_dense_and_bm25_apply_identical_user_filter(
    index_fixture: _IndexFixture,
) -> None:
    dense = await index_fixture.vector.search(
        index_fixture.query_vector, index_fixture.filter("u1"), 40
    )
    lexical = await index_fixture.lexical.search(
        "termination clause", index_fixture.filter("u1"), 40
    )

    assert dense and lexical
    assert {hit.user_id for hit in dense + lexical} == {"u1"}
    assert {hit.lane for hit in dense} == {"dense"}
    assert {hit.lane for hit in lexical} == {"bm25"}
