"""Boundary contract for ES generation mappings and server-owned metadata."""

from __future__ import annotations

from typing import Any, cast

import pytest

from agentic_rag.ingestion.chunker import AstLocator, AstSpan, ChildChunk
from agentic_rag.ingestion.indexer import EmbeddedChild, StagingContext
from agentic_rag.persistence.elasticsearch import ElasticsearchChildIndexStore


class _Response:
    def __init__(self, body: dict[str, Any]) -> None:
        self.body = body


class _Indices:
    def __init__(self) -> None:
        self.mappings: dict[str, Any] | None = None

    async def create(self, *, index: str, mappings: dict[str, Any]) -> _Response:
        self.mappings = mappings
        return _Response({"acknowledged": True})


class _Client:
    def __init__(self) -> None:
        self.indices = _Indices()
        self.operations: list[dict[str, Any]] = []

    async def bulk(
        self,
        *,
        operations: list[dict[str, Any]],
        refresh: str,
    ) -> _Response:
        self.operations = operations
        return _Response(
            {"errors": False, "items": [{"index": {"_id": "child-1", "status": 201}}]}
        )


def _embedded_child() -> EmbeddedChild:
    locator = AstLocator(
        spans=(
            AstSpan(
                canonical_path="#/text_blocks/0",
                block_id="block-1",
                page_from=1,
                page_to=1,
                char_from=0,
                char_to=5,
                parent_char_from=0,
                parent_char_to=5,
            ),
        ),
        segment_ordinal=0,
        parent_char_from=0,
        parent_char_to=5,
    )
    child = ChildChunk(
        id="child-1",
        parent_id="parent-1",
        parent_ordinal=0,
        document_id="document-1",
        document_version_id="version-1",
        user_id="user-1",
        ordinal=0,
        heading_path=("Heading",),
        heading_ast_locators=("#/text_blocks/h",),
        content_type="paragraph",
        content="hello",
        contextualized_content="Heading\nhello",
        token_count=2,
        page_from=1,
        page_to=1,
        ast_locator=locator,
        content_hash="a" * 64,
    )
    return EmbeddedChild(
        chunk=child,
        embedding=tuple(0.1 for _ in range(1024)),
        user_id="user-1",
        document_id="document-1",
        document_version_id="version-1",
        index_generation="index-v1",
    )


@pytest.mark.asyncio
async def test_es_mapping_and_payload_include_durable_version_metadata() -> None:
    client = _Client()
    store = ElasticsearchChildIndexStore(cast(Any, client))
    context = StagingContext(
        user_id="user-1",
        document_id="document-1",
        document_version_id="version-1",
        version_no=7,
        pipeline_version="ingestion-v1",
        embedding_version="text-embedding-v3",
        index_generation="index-v1",
    )

    assert await store.stage(context, [_embedded_child()]) == 1

    assert client.indices.mappings is not None
    properties = client.indices.mappings["properties"]
    assert properties["version_no"] == {"type": "integer"}
    assert properties["pipeline_version"] == {"type": "keyword"}
    assert properties["embedding_version"] == {"type": "keyword"}
    source = client.operations[1]
    assert source["version_no"] == 7
    assert source["pipeline_version"] == "ingestion-v1"
    assert source["embedding_version"] == "text-embedding-v3"


@pytest.mark.parametrize("generation", ["INDEX-V1", " index-v1 "])
def test_es_index_name_does_not_alias_noncanonical_generations(
    generation: str,
) -> None:
    with pytest.raises(ValueError):
        ElasticsearchChildIndexStore.index_name(generation)
