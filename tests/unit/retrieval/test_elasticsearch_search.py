"""Unit coverage for Elasticsearch hybrid recall request boundaries."""

from __future__ import annotations

from typing import Any, cast

import pytest

from agentic_rag.retrieval.adapters.elasticsearch import (
    ACTIVE_CHILD_INDEX_ALIAS,
    ElasticsearchBm25Index,
    ElasticsearchVectorIndex,
    IndexGenerationMismatchError,
    QueryVectorDimensionError,
    UnsupportedDateRangeError,
)
from agentic_rag.retrieval.models import DateRange, SearchFilter


class _Response:
    def __init__(self, body: dict[str, Any]) -> None:
        self.body = body


class _Client:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    async def search(self, **kwargs: Any) -> _Response:
        self.calls.append(kwargs)
        return _Response(
            {
                "hits": {
                    "hits": [
                        {
                            "_id": "fallback-id",
                            "_score": 2.5,
                            "_source": {
                                "id": "child-1",
                                "parent_id": "parent-1",
                                "user_id": "u1",
                                "document_id": "document-1",
                                "document_version_id": "version-1",
                                "content": "termination clause",
                                "ast_locator": {"segment_ordinal": 0},
                            },
                        }
                    ]
                }
            }
        )


def _filter(**overrides: Any) -> SearchFilter:
    values = {"user_id": "u1", "index_generation": "index-v2"}
    values.update(overrides)
    return SearchFilter(**values)


@pytest.mark.asyncio
async def test_ik_bm25_prefers_chinese_field_without_adding_a_recall_lane() -> None:
    client = _Client()
    lexical = ElasticsearchBm25Index(cast(Any, client), index_generation="index-v4", lexical_analysis="ik")
    hits = await lexical.search("京东工作经历", _filter(index_generation="index-v4"), 7)
    assert client.calls[0]["query"]["bool"]["must"] == {
        "multi_match": {"query": "京东工作经历", "fields": ["contextualized_content.zh^2", "contextualized_content"],
                        "type": "best_fields", "tie_breaker": 0.0},
    }
    assert client.calls[0]["query"]["bool"]["filter"] == [
        {"term": {"user_id": "u1"}}, {"term": {"is_active": True}},
        {"term": {"index_generation": "index-v4"}},
    ]
    assert client.calls[0]["size"] == 7
    assert len(client.calls) == 1
    assert [hit.lane for hit in hits] == ["bm25"]


@pytest.mark.asyncio
async def test_standard_profile_keeps_legacy_bm25_query() -> None:
    client = _Client()
    lexical = ElasticsearchBm25Index(cast(Any, client), index_generation="index-v3", lexical_analysis="standard")
    await lexical.search("京东", _filter(index_generation="index-v3"), 7)
    assert client.calls[0]["query"]["bool"]["must"] == {"match": {"contextualized_content": "京东"}}


@pytest.mark.asyncio
async def test_dense_and_bm25_serialize_the_same_server_owned_filters() -> None:
    client = _Client()
    filter = _filter(
        search_type="document",
        document_ids=("document-1", "document-2"),
        content_types=("paragraph",),
    )
    dense = ElasticsearchVectorIndex(
        cast(Any, client), index="children-active", index_generation="index-v2"
    )
    lexical = ElasticsearchBm25Index(
        cast(Any, client), index="children-active", index_generation="index-v2"
    )

    dense_hits = await dense.search([0.1] * 1024, filter, 3)
    lexical_hits = await lexical.search("termination clause", filter, 3)

    assert client.calls[0]["knn"]["filter"] == client.calls[1]["query"]["bool"][
        "filter"
    ]
    assert client.calls[0]["knn"]["filter"] == [
        {"term": {"user_id": "u1"}},
        {"term": {"is_active": True}},
        {"term": {"index_generation": "index-v2"}},
        {"term": {"search_type": "document"}},
        {"terms": {"document_id": ["document-1", "document-2"]}},
        {"terms": {"content_type": ["paragraph"]}},
    ]
    assert client.calls[0]["source_includes"] == client.calls[1]["source_includes"]
    assert dense_hits[0].lane == "dense"
    assert lexical_hits[0].lane == "bm25"
    assert dense_hits[0].lane_rank == lexical_hits[0].lane_rank == 1
    assert dense_hits[0].ast_locator == '{"segment_ordinal":0}'


@pytest.mark.asyncio
@pytest.mark.parametrize("adapter", [ElasticsearchVectorIndex, ElasticsearchBm25Index])
async def test_rejects_date_range_before_sending_an_es_request(
    adapter: type[ElasticsearchVectorIndex | ElasticsearchBm25Index],
) -> None:
    client = _Client()
    index = adapter(cast(Any, client), index_generation="index-v2")

    with pytest.raises(UnsupportedDateRangeError, match="date_range"):
        if isinstance(index, ElasticsearchVectorIndex):
            await index.search(
                [0.1] * 1024,
                _filter(date_range=DateRange(start="2026-01-01")),
                3,
            )
        else:
            await index.search(
                "termination clause",
                _filter(date_range=DateRange(start="2026-01-01")),
                3,
            )

    assert client.calls == []


@pytest.mark.asyncio
async def test_dense_and_bm25_default_to_the_active_child_alias() -> None:
    client = _Client()
    dense = ElasticsearchVectorIndex(cast(Any, client), index_generation="index-v2")
    lexical = ElasticsearchBm25Index(cast(Any, client), index_generation="index-v2")

    await dense.search([0.1] * 1024, _filter(), 3)
    await lexical.search("termination clause", _filter(), 3)

    assert [call["index"] for call in client.calls] == [
        ACTIVE_CHILD_INDEX_ALIAS,
        ACTIVE_CHILD_INDEX_ALIAS,
    ]


@pytest.mark.asyncio
async def test_dense_rejects_a_non_1024_dimension_vector_before_search() -> None:
    client = _Client()
    dense = ElasticsearchVectorIndex(
        cast(Any, client), index="children-active", index_generation="index-v2"
    )

    with pytest.raises(QueryVectorDimensionError, match="exactly 1024"):
        await dense.search([0.1] * 1023, _filter(), 3)

    assert client.calls == []


@pytest.mark.asyncio
@pytest.mark.parametrize("adapter", [ElasticsearchVectorIndex, ElasticsearchBm25Index])
async def test_rejects_a_filter_for_another_index_generation_before_search(
    adapter: type[ElasticsearchVectorIndex | ElasticsearchBm25Index],
) -> None:
    client = _Client()
    index = adapter(cast(Any, client), index="children-active", index_generation="index-v2")

    with pytest.raises(IndexGenerationMismatchError, match="index-v1.*index-v2"):
        if isinstance(index, ElasticsearchVectorIndex):
            await index.search([0.1] * 1024, _filter(index_generation="index-v1"), 3)
        else:
            await index.search("termination clause", _filter(index_generation="index-v1"), 3)

    assert client.calls == []
