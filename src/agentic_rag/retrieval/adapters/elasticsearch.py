"""Elasticsearch implementations of dense and BM25 Child recall.

The two retrieval lanes deliberately share filter serialization.  This keeps
tenant and publication scope outside agent control and prevents one lane from
observing records the other lane cannot observe.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any, Literal, cast

from elasticsearch import AsyncElasticsearch

from agentic_rag.ingestion.indexer import EMBEDDING_DIMENSIONS
from agentic_rag.models.indexing import (
    ACTIVE_CHILD_INDEX_ALIAS,
    validate_index_generation,
)
from agentic_rag.retrieval.models import ChildHit, SearchFilter


class QueryVectorDimensionError(ValueError):
    """Raised when a dense query cannot target the Child vector mapping."""


class IndexGenerationMismatchError(ValueError):
    """Raised before ES I/O when a request targets another index generation."""


class UnsupportedDateRangeError(ValueError):
    """Raised until Child index mappings store a queryable document date."""


_HIT_SOURCE_FIELDS = (
    "id",
    "parent_id",
    "user_id",
    "document_id",
    "document_version_id",
    "content",
    "ast_locator",
)


def serialize_filter(filter: SearchFilter) -> list[dict[str, Any]]:
    """Serialize all supported server-owned and agent-safe selectors for ES."""
    clauses: list[dict[str, Any]] = [
        {"term": {"user_id": filter.user_id}},
        {"term": {"is_active": True}},
        {"term": {"index_generation": filter.index_generation}},
    ]
    if filter.search_type is not None:
        clauses.append({"term": {"search_type": filter.search_type}})
    if filter.document_ids:
        clauses.append({"terms": {"document_id": list(filter.document_ids)}})
    if filter.content_types:
        clauses.append({"terms": {"content_type": list(filter.content_types)}})
    return clauses


class _ElasticsearchChildSearch:
    """Shared ES index boundary and hit decoding for the two recall lanes."""

    def __init__(
        self,
        client: AsyncElasticsearch,
        *,
        index_generation: str,
        index: str | None = None,
    ) -> None:
        self._client = client
        self._index_generation = validate_index_generation(index_generation)
        self._index = index or ACTIVE_CHILD_INDEX_ALIAS

    def _validate_filter(self, filter: SearchFilter) -> list[dict[str, Any]]:
        if filter.index_generation != self._index_generation:
            raise IndexGenerationMismatchError(
                "SearchFilter index generation "
                f"{filter.index_generation!r} does not match adapter generation "
                f"{self._index_generation!r}"
            )
        if filter.date_range is not None:
            raise UnsupportedDateRangeError(
                "date_range is unsupported until Child index mappings include "
                "a populated document-date field"
            )
        return serialize_filter(filter)

    @staticmethod
    def _validate_top_k(top_k: int) -> None:
        if top_k <= 0:
            raise ValueError("top_k must be positive")

    @staticmethod
    def _hits(
        response: Any,
        *,
        lane: Literal["dense", "bm25"],
    ) -> list[ChildHit]:
        body = cast(Mapping[str, Any], response.body)
        hits_section = cast(Mapping[str, Any], body.get("hits", {}))
        raw_hits = cast(Sequence[Mapping[str, Any]], hits_section.get("hits", ()))
        hits: list[ChildHit] = []
        for rank, raw_hit in enumerate(raw_hits, start=1):
            source = cast(Mapping[str, Any], raw_hit.get("_source", {}))
            locator = source.get("ast_locator", "")
            hits.append(
                ChildHit(
                    child_id=str(source.get("id", raw_hit.get("_id", ""))),
                    parent_id=str(source["parent_id"]),
                    user_id=str(source["user_id"]),
                    document_id=str(source["document_id"]),
                    document_version_id=str(source["document_version_id"]),
                    content=str(source["content"]),
                    ast_locator=(
                        locator
                        if isinstance(locator, str)
                        else json.dumps(locator, separators=(",", ":"), sort_keys=True)
                    ),
                    lane=lane,
                    lane_rank=rank,
                    retrieval_score=float(raw_hit.get("_score") or 0.0),
                )
            )
        return hits


class ElasticsearchVectorIndex(_ElasticsearchChildSearch):
    """KNN Child retrieval over the active Elasticsearch Child alias."""

    async def search(
        self,
        query_vector: Sequence[float],
        filter: SearchFilter,
        top_k: int,
    ) -> list[ChildHit]:
        if len(query_vector) != EMBEDDING_DIMENSIONS:
            raise QueryVectorDimensionError(
                f"dense query vectors must have exactly {EMBEDDING_DIMENSIONS} dimensions"
            )
        self._validate_top_k(top_k)
        clauses = self._validate_filter(filter)
        response = await self._client.search(
            index=self._index,
            knn={
                "field": "embedding",
                "query_vector": list(query_vector),
                "k": top_k,
                "num_candidates": max(top_k, min(top_k * 4, 10_000)),
                "filter": clauses,
            },
            size=top_k,
            source_includes=list(_HIT_SOURCE_FIELDS),
        )
        return self._hits(response, lane="dense")


class ElasticsearchBm25Index(_ElasticsearchChildSearch):
    """BM25 Child retrieval over contextualized Child content."""

    async def search(
        self,
        query_text: str,
        filter: SearchFilter,
        top_k: int,
    ) -> list[ChildHit]:
        self._validate_top_k(top_k)
        clauses = self._validate_filter(filter)
        response = await self._client.search(
            index=self._index,
            query={
                "bool": {
                    "filter": clauses,
                    "must": {"match": {"contextualized_content": query_text}},
                }
            },
            size=top_k,
            source_includes=list(_HIT_SOURCE_FIELDS),
        )
        return self._hits(response, lane="bm25")
