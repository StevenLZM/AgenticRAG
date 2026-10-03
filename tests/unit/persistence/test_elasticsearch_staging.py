"""Boundary contract for ES generation mappings and server-owned metadata."""

from __future__ import annotations

from typing import Any, cast

import pytest
from elastic_transport import ApiResponseMeta, HttpHeaders, NodeConfig
from elasticsearch.exceptions import BadRequestError, NotFoundError

from agentic_rag.ingestion.chunker import AstLocator, AstSpan, ChildChunk
from agentic_rag.ingestion.indexer import EmbeddedChild, StagingContext
from agentic_rag.persistence.elasticsearch import (
    ACTIVE_CHILD_INDEX_ALIAS,
    ChildIndexMappingError,
    ChildIndexWriteError,
    ElasticsearchChildIndexStore,
)


class _Response:
    def __init__(self, body: dict[str, Any]) -> None:
        self.body = body


class _Indices:
    def __init__(self, *, existing_mapping: dict[str, Any] | None = None) -> None:
        self.mappings = existing_mapping
        self.indices: set[str] = {"agenticrag-children-index-v2"} if existing_mapping else set()
        self.aliases: dict[str, set[str]] = {}
        self.alias_updates: list[list[dict[str, Any]]] = []
        self.put_alias_calls: list[dict[str, str]] = []
        self.create_calls = 0
        self.get_mapping_calls = 0

    async def create(self, *, index: str, mappings: dict[str, Any]) -> _Response:
        self.create_calls += 1
        if self.mappings is not None:
            raise _already_exists(index)
        self.mappings = mappings
        self.indices.add(index)
        return _Response({"acknowledged": True})

    async def exists(self, *, index: str) -> bool:
        return index in self.indices

    async def get_mapping(self, *, index: str) -> _Response:
        self.get_mapping_calls += 1
        assert self.mappings is not None
        return _Response({index: {"mappings": self.mappings}})

    async def get_alias(self, *, name: str) -> _Response:
        targets = self.aliases.get(name)
        if not targets:
            raise _not_found(name)
        return _Response({index: {"aliases": {name: {}}} for index in targets})

    async def put_alias(self, *, index: str, name: str) -> _Response:
        self.put_alias_calls.append({"index": index, "name": name})
        self.aliases.setdefault(name, set()).add(index)
        return _Response({"acknowledged": True})

    async def update_aliases(self, *, actions: list[dict[str, Any]]) -> _Response:
        self.alias_updates.append(actions)
        for action in actions:
            if "remove" in action:
                payload = action["remove"]
                self.aliases.get(payload["alias"], set()).discard(payload["index"])
            if "add" in action:
                payload = action["add"]
                self.aliases.setdefault(payload["alias"], set()).add(payload["index"])
        return _Response({"acknowledged": True})


class _Client:
    def __init__(
        self,
        *,
        existing_mapping: dict[str, Any] | None = None,
        lifecycle_response: dict[str, Any] | None = None,
    ) -> None:
        self.indices = _Indices(existing_mapping=existing_mapping)
        self.operations: list[dict[str, Any]] = []
        self.bulk_calls = 0
        self.lifecycle_response = lifecycle_response or {
            "total": 1,
            "version_conflicts": 0,
            "failures": [],
        }

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

    async def update_by_query(self, **kwargs: Any) -> _Response:
        return _Response(self.lifecycle_response)

    async def delete_by_query(self, **kwargs: Any) -> _Response:
        return _Response(self.lifecycle_response)


@pytest.mark.asyncio
async def test_ik_staging_preserves_standard_field_and_indexes_chinese_subfield() -> None:
    client = _Client()
    store = ElasticsearchChildIndexStore(cast(Any, client), lexical_analysis="ik")

    await store.stage(_context(index_generation="index-v4"), [_embedded_child(index_generation="index-v4")])

    assert client.indices.mappings["properties"]["contextualized_content"] == {
        "type": "text",
        "fields": {"zh": {"type": "text", "analyzer": "ik_max_word", "search_analyzer": "ik_smart"}},
    }
    assert client.indices.mappings["_meta"]["lexical_analysis"] == "ik-v1"
    assert client.operations[1]["contextualized_content"] == "Heading\nhello"
    assert client.operations[1]["embedding"] == [0.1] * 1024


@pytest.mark.asyncio
async def test_ik_cannot_silently_reuse_a_standard_index() -> None:
    client = _Client()
    await ElasticsearchChildIndexStore(cast(Any, client)).stage(_context(), [_embedded_child()])
    store = ElasticsearchChildIndexStore(cast(Any, client), lexical_analysis="ik")
    with pytest.raises(ChildIndexMappingError, match="incompatible"):
        await store.stage(_context(), [_embedded_child()])
    assert client.bulk_calls == 1


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


def _not_found(alias: str) -> NotFoundError:
    return NotFoundError(
        "alias not found",
        ApiResponseMeta(
            status=404,
            http_version="1.1",
            headers=HttpHeaders(),
            duration=0.0,
            node=NodeConfig("http", "localhost", 9200),
        ),
        {"error": {"type": "alias_not_found_exception", "reason": alias}, "status": 404},
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

    with pytest.raises(ChildIndexMappingError, match="index-v1.*index-v3"):
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


@pytest.mark.asyncio
async def test_missing_active_alias_is_created_for_existing_index() -> None:
    client = _Client()
    store = ElasticsearchChildIndexStore(cast(Any, client))

    await store.stage(_context(), [_embedded_child()])

    assert await store.ensure_active_alias("index-v2") is True
    assert client.indices.aliases == {
        ACTIVE_CHILD_INDEX_ALIAS: {"agenticrag-children-index-v2"}
    }
    assert client.indices.put_alias_calls == [
        {"index": "agenticrag-children-index-v2", "name": ACTIVE_CHILD_INDEX_ALIAS}
    ]


@pytest.mark.asyncio
async def test_missing_physical_index_keeps_alias_pending() -> None:
    client = _Client()
    store = ElasticsearchChildIndexStore(cast(Any, client))

    assert await store.ensure_active_alias("index-v2") is False
    assert client.indices.put_alias_calls == []


@pytest.mark.asyncio
async def test_existing_active_alias_is_not_silently_switched() -> None:
    client = _Client()
    store = ElasticsearchChildIndexStore(cast(Any, client))

    await store.stage(_context(), [_embedded_child()])
    client.indices.aliases[ACTIVE_CHILD_INDEX_ALIAS] = {"agenticrag-children-index-v1"}

    with pytest.raises(ChildIndexWriteError, match="active alias"):
        await store.ensure_active_alias("index-v2")

    assert client.indices.alias_updates == []


@pytest.mark.asyncio
async def test_existing_active_alias_is_idempotent() -> None:
    client = _Client()
    store = ElasticsearchChildIndexStore(cast(Any, client))

    await store.stage(_context(), [_embedded_child()])
    client.indices.aliases[ACTIVE_CHILD_INDEX_ALIAS] = {
        "agenticrag-children-index-v2"
    }

    assert await store.ensure_active_alias("index-v2") is False
    assert await store.switch_active_alias("index-v2") is False
    assert client.indices.alias_updates == []


@pytest.mark.asyncio
async def test_switch_active_alias_uses_one_atomic_update() -> None:
    client = _Client()
    store = ElasticsearchChildIndexStore(cast(Any, client))

    await store.stage(_context(), [_embedded_child()])
    client.indices.aliases[ACTIVE_CHILD_INDEX_ALIAS] = {"agenticrag-children-index-v1"}

    await store.switch_active_alias("index-v2")

    assert client.indices.aliases == {
        ACTIVE_CHILD_INDEX_ALIAS: {"agenticrag-children-index-v2"}
    }
    assert client.indices.alias_updates == [
        [
            {
                "remove": {
                    "index": "agenticrag-children-index-v1",
                    "alias": ACTIVE_CHILD_INDEX_ALIAS,
                }
            },
            {
                "add": {
                    "index": "agenticrag-children-index-v2",
                    "alias": ACTIVE_CHILD_INDEX_ALIAS,
                }
            },
        ]
    ]


@pytest.mark.parametrize("generation", ["INDEX-V1", " index-v1 "])
def test_es_index_name_does_not_alias_noncanonical_generations(
    generation: str,
) -> None:
    with pytest.raises(ValueError):
        ElasticsearchChildIndexStore.index_name(generation)


@pytest.mark.asyncio
@pytest.mark.parametrize("operation", ["activate", "deactivate", "delete"])
async def test_lifecycle_write_fails_closed_on_partial_es_conflicts(
    operation: str,
) -> None:
    client = _Client(
        lifecycle_response={"total": 1, "version_conflicts": 1, "failures": []}
    )
    store = ElasticsearchChildIndexStore(cast(Any, client))

    with pytest.raises(ChildIndexWriteError, match="did not fully apply"):
        await getattr(store, operation)(_context())
