"""Elasticsearch 8 adapter for inactive Child-vector staging."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Any, cast

from elasticsearch import AsyncElasticsearch
from elasticsearch.exceptions import BadRequestError

from agentic_rag.ingestion.indexer import EmbeddedChild, StagingContext


_INDEX_GENERATION = re.compile(r"[a-z0-9][a-z0-9_-]{0,63}")


class ChildIndexWriteError(RuntimeError):
    """Raised when an ES bulk batch does not fully stage."""


class ElasticsearchChildIndexStore:
    """Keep ES-specific layout and bulk semantics behind one replaceable adapter.

    Each bulk request is an explicit retry boundary. Documents use deterministic
    Child IDs and the ``index`` operation, so retrying a failed version converges
    without duplicates. No record becomes active in this adapter.
    """

    def __init__(
        self,
        client: AsyncElasticsearch,
        *,
        bulk_batch_size: int = 500,
    ) -> None:
        if bulk_batch_size <= 0:
            raise ValueError("bulk_batch_size must be positive")
        self._client = client
        self._bulk_batch_size = bulk_batch_size

    @staticmethod
    def index_name(index_generation: str) -> str:
        normalized = index_generation.strip().lower()
        if _INDEX_GENERATION.fullmatch(normalized) is None:
            raise ValueError(
                "index_generation must be a lowercase ES-safe version label"
            )
        return f"agenticrag-children-{normalized}"

    async def stage(
        self, context: StagingContext, children: Sequence[EmbeddedChild]
    ) -> int:
        await self._ensure_index(context.index_generation)
        index = self.index_name(context.index_generation)
        staged = 0
        for start in range(0, len(children), self._bulk_batch_size):
            batch = children[start : start + self._bulk_batch_size]
            operations: list[Mapping[str, Any]] = []
            for child in batch:
                self._validate_metadata(context, child)
                operations.append({"index": {"_index": index, "_id": child.chunk.id}})
                operations.append(self._document(context, child))
            response = await self._client.bulk(
                operations=operations,
                refresh="wait_for",
            )
            body = cast(Mapping[str, Any], response.body)
            if body.get("errors"):
                failures = _bulk_failures(body)
                raise ChildIndexWriteError(
                    "Elasticsearch Child staging batch failed: " + ", ".join(failures)
                )
            items = cast(Sequence[Mapping[str, Any]], body.get("items", ()))
            if len(items) != len(batch):
                raise ChildIndexWriteError(
                    "Elasticsearch bulk response count does not match request"
                )
            staged += len(items)
        return staged

    async def count(self, context: StagingContext) -> int:
        return await self._count(context, is_active=False)

    async def count_active(self, context: StagingContext) -> int:
        """Return active records for publication/reconciliation validation."""
        return await self._count(context, is_active=True)

    async def _count(self, context: StagingContext, *, is_active: bool) -> int:
        response = await self._client.count(
            index=self.index_name(context.index_generation),
            query={
                "bool": {
                    "filter": [
                        {"term": {"user_id": context.user_id}},
                        {"term": {"document_id": context.document_id}},
                        {
                            "term": {
                                "document_version_id": context.document_version_id
                            }
                        },
                        {"term": {"search_type": "document"}},
                        {"term": {"is_active": is_active}},
                    ]
                }
            },
        )
        return int(cast(Mapping[str, Any], response.body)["count"])

    async def _ensure_index(self, index_generation: str) -> None:
        index = self.index_name(index_generation)
        try:
            await self._client.indices.create(
                index=index,
                mappings={
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
                },
            )
        except BadRequestError as error:
            body = error.body if isinstance(error.body, Mapping) else {}
            detail = body.get("error", {})
            error_type = detail.get("type") if isinstance(detail, Mapping) else None
            if error_type != "resource_already_exists_exception":
                raise

    @staticmethod
    def _validate_metadata(
        context: StagingContext, child: EmbeddedChild
    ) -> None:
        actual = (
            child.user_id,
            child.document_id,
            child.document_version_id,
            child.index_generation,
            child.search_type,
            child.is_active,
        )
        expected = (
            context.user_id,
            context.document_id,
            context.document_version_id,
            context.index_generation,
            "document",
            False,
        )
        if actual != expected:
            raise ChildIndexWriteError(
                f"Child {child.chunk.id!r} metadata is outside staging context"
            )

    @staticmethod
    def _document(
        context: StagingContext, embedded: EmbeddedChild
    ) -> Mapping[str, Any]:
        child = embedded.chunk
        return {
            "id": child.id,
            "parent_id": child.parent_id,
            "parent_ordinal": child.parent_ordinal,
            "user_id": context.user_id,
            "document_id": context.document_id,
            "document_version_id": context.document_version_id,
            "ordinal": child.ordinal,
            "heading_path": list(child.heading_path),
            "heading_ast_locators": list(child.heading_ast_locators),
            "content_type": child.content_type,
            "content": child.content,
            "contextualized_content": child.contextualized_content,
            "token_count": child.token_count,
            "page_from": child.page_from,
            "page_to": child.page_to,
            "ast_locator": child.ast_locator.model_dump(mode="json"),
            "content_hash": child.content_hash,
            "embedding": list(embedded.embedding),
            "index_generation": context.index_generation,
            "search_type": "document",
            "is_active": False,
        }


def _bulk_failures(body: Mapping[str, Any]) -> list[str]:
    failures: list[str] = []
    for item in cast(Sequence[Mapping[str, Any]], body.get("items", ())):
        result = item.get("index", {})
        if not isinstance(result, Mapping) or "error" not in result:
            continue
        failures.append(f"{result.get('_id', '<unknown>')}:{result.get('status', '?')}")
    return failures or ["unknown bulk failure"]
