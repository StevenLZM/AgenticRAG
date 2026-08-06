"""Boundary contract for ES generation mappings and server-owned metadata."""

from __future__ import annotations

from typing import Any, cast

import pytest
from elastic_transport import ApiResponseMeta, HttpHeaders, NodeConfig
from elasticsearch.exceptions import BadRequestError

from agentic_rag.ingestion.chunker import AstLocator, AstSpan, ChildChunk
from agentic_rag.ingestion.indexer import EmbeddedChild, StagingContext
from agentic_rag.persistence.elasticsearch import (
    ChildIndexWriteError,
    ElasticsearchChildIndexStore,
)


class _Response:
    def __init__(self, body: dict[str, Any]) -> None:
        self.body = body


class _Indices:
    def __init__(self, *, existing_mapping: dict[str, Any] | None = None) -> None:
        self.mappings = existing_mapping
        self.create_calls = 0
        self.get_mapping_calls = 0

    async def create(self, *, index: str, mappings: dict[str, Any]) -> _Response:
        self.create_calls += 1
        if self.mappings is not None:
            raise _already_exists(index)
        self.mappings = mappings
        return _Response({"acknowledged": True})

    async def get_mapping(self, *, index: str) -> _Response:
        self.get_mapping_calls += 1
        assert self.mappings is not None
        return _Response({index: {"mappings": self.mappings}})


class _Client:
    def __init__(self, *, existing_mapping: dict[str, Any] | None = None) -> None:
        self.indices = _Indices(existing_mapping=existing_mapping)
        self.operations: list[dict[str, Any]] = []
        self.bulk_calls = 0

    async def bulk(
        self,
        *,
        operations: list[dict[str, Any]],
        refresh: str,
    ) -> _Response:
        self.bulk_calls += 1
        self.operations = operations
        return _Response(
            {"errors": False, "items": [{"index": {"_id": "child-1", "status": 201}}]}
        )


def _embedded_child(*, index_generation: str = "index-v2") -> EmbeddedChild:
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
        index_generation=index_generation,
    )


def _context(*, index_generation: str = "index-v2") -> StagingContext:
    return StagingContext(
        user_id="user-1",
        document_id="document-1",
        document_version_id="version-1",
        version_no=7,
        pipeline_version="ingestion-v1",
        embedding_version="text-embedding-v3",
        index_generation=index_generation,
    )


def _already_exists(index: str) -> BadRequestError:
    return BadRequestError(
        "resource already exists",
        ApiResponseMeta(
            status=400,
            http_version="1.1",
            headers=HttpHeaders(),
            duration=0.0,
            node=NodeConfig("http", "localhost", 9200),
        ),
        {
            "error": {
                "type": "resource_already_exists_exception",
                "index": index,
            },
            "status": 400,
        },
    )


def _base_v1_strict_mapping() -> dict[str, Any]:
    return {
        "dynamic": "strict",
        "properties": {
            "id": {"type": "keyword"},
            "parent_id": {"type": "keyword"},
            "parent_ordinal": {"type": "integer"},
            "user_id": {"type": "keyword"},
            "document_id": {"type": "keyword"},
            "document_version_id": {"type": "keyword"},
            "ordinal": {"type": "integer"},
            "heading_path": {"type": "keyword"},
            "heading_ast_locators": {"type": "keyword"},
            "content_type": {"type": "keyword"},
            "content": {"type": "text"},
            "contextualized_content": {"type": "text"},
            "token_count": {"type": "integer"},
            "page_from": {"type": "integer"},
            "page_to": {"type": "integer"},
            "ast_locator": {"type": "object", "enabled": False},
            "content_hash": {"type": "keyword"},
            "embedding": {
                "type": "dense_vector",
                "dims": 1024,
                "index": True,
                "similarity": "cosine",
            },
            "index_generation": {"type": "keyword"},
            "search_type": {"type": "keyword"},
            "is_active": {"type": "boolean"},
        },
    }


@pytest.mark.asyncio
async def test_es_mapping_and_payload_include_durable_version_metadata() -> None:
    client = _Client()
    store = ElasticsearchChildIndexStore(cast(Any, client))
    context = _context()

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


@pytest.mark.asyncio
async def test_legacy_index_v1_fails_closed_before_bulk() -> None:
    client = _Client(existing_mapping=_base_v1_strict_mapping())
    store = ElasticsearchChildIndexStore(cast(Any, client))
    context = _context(index_generation="index-v1")

    with pytest.raises(ChildIndexWriteError, match="index-v1.*index-v2"):
        await store.stage(
            context,
            [_embedded_child(index_generation="index-v1")],
        )

    assert client.bulk_calls == 0


@pytest.mark.asyncio
async def test_base_strict_mapping_cannot_masquerade_as_current_generation() -> None:
    client = _Client(existing_mapping=_base_v1_strict_mapping())
    store = ElasticsearchChildIndexStore(cast(Any, client))

    with pytest.raises(ChildIndexWriteError, match="incompatible strict mapping"):
        await store.stage(_context(), [_embedded_child()])

    assert client.indices.get_mapping_calls == 1
    assert client.bulk_calls == 0


@pytest.mark.asyncio
async def test_compatible_existing_index_is_an_idempotent_retry_boundary() -> None:
    client = _Client()
    store = ElasticsearchChildIndexStore(cast(Any, client))

    assert await store.stage(_context(), [_embedded_child()]) == 1
    assert await store.stage(_context(), [_embedded_child()]) == 1

    assert client.indices.create_calls == 2
    assert client.indices.get_mapping_calls == 1
    assert client.bulk_calls == 2


@pytest.mark.parametrize("generation", ["INDEX-V1", " index-v1 "])
def test_es_index_name_does_not_alias_noncanonical_generations(
    generation: str,
) -> None:
    with pytest.raises(ValueError):
        ElasticsearchChildIndexStore.index_name(generation)
