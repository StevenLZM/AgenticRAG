"""Elasticsearch 8 adapter for inactive Child-vector staging."""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping, Sequence
import logging
from typing import Any, cast

from elasticsearch import AsyncElasticsearch
from elasticsearch.exceptions import BadRequestError, NotFoundError

from agentic_rag.ingestion.indexer import (
    EMBEDDING_DIMENSIONS,
    EMBEDDING_MODEL,
    EmbeddedChild,
    StagingContext,
)
from agentic_rag.models.indexing import (
    ACTIVE_CHILD_INDEX_ALIAS,
    DEFAULT_INDEX_GENERATION,
    LEGACY_INDEX_GENERATIONS,
    validate_index_generation,
)


class ChildIndexWriteError(RuntimeError):
    """Raised when an ES bulk batch does not fully stage."""


class ChildIndexMappingError(ChildIndexWriteError):
    """Raised before bulk when a generation index has an incompatible schema."""


logger = logging.getLogger(__name__)


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
        validate_index_generation(index_generation)
        return f"agenticrag-children-{index_generation}"

    async def ensure_active_alias(self, index_generation: str) -> bool:
        """Repair a missing active alias without changing an existing generation.

        Startup recovery may create the alias when the configured physical index is
        already present.  An alias pointing at another generation is an operator or
        rollout mismatch and therefore fails closed instead of being silently moved.
        """
        index = self.index_name(index_generation)
        if not bool(await self._client.indices.exists(index=index)):
            logger.warning(
                "elasticsearch_active_alias_pending",
                extra={
                    "component": "elasticsearch",
                    "alias": ACTIVE_CHILD_INDEX_ALIAS,
                    "index": index,
                    "outcome": "pending",
                    "retryable": True,
                },
            )
            return False

        await self._ensure_index(index_generation)
        targets = await self._read_active_alias_targets()
        if not targets:
            await self._client.indices.put_alias(
                index=index,
                name=ACTIVE_CHILD_INDEX_ALIAS,
            )
            await self._verify_active_alias(index)
            logger.warning(
                "elasticsearch_active_alias_repaired",
                extra={
                    "component": "elasticsearch",
                    "alias": ACTIVE_CHILD_INDEX_ALIAS,
                    "index": index,
                    "outcome": "recovered",
                    "retryable": False,
                },
            )
            return True
        if targets != (index,):
            raise ChildIndexWriteError(
                "active alias points at an unexpected index generation: "
                f"expected {index!r}, found {', '.join(targets)}"
            )
        return False

    async def switch_active_alias(self, index_generation: str) -> bool:
        """Atomically point the active alias at a validated physical generation."""
        index = self.index_name(index_generation)
        if not bool(await self._client.indices.exists(index=index)):
            raise ChildIndexWriteError(
                f"cannot switch active alias to missing index {index!r}"
            )
        await self._ensure_index(index_generation)
        targets = await self._read_active_alias_targets()
        if targets == (index,):
            return False

        actions: list[Mapping[str, Any]] = [
            {
                "remove": {
                    "index": target,
                    "alias": ACTIVE_CHILD_INDEX_ALIAS,
                }
            }
            for target in targets
        ]
        actions.append(
            {
                "add": {
                    "index": index,
                    "alias": ACTIVE_CHILD_INDEX_ALIAS,
                }
            }
        )
        response = await self._client.indices.update_aliases(actions=actions)
        body = cast(Mapping[str, Any], response.body)
        if body.get("acknowledged") is False:
            raise ChildIndexWriteError("Elasticsearch active alias switch was not acknowledged")
        await self._verify_active_alias(index)
        logger.info(
            "elasticsearch_active_alias_switched",
            extra={
                "component": "elasticsearch",
                "alias": ACTIVE_CHILD_INDEX_ALIAS,
                "index": index,
                "outcome": "switched",
                "retryable": False,
            },
        )
        return True

    async def stage(
        self,
        context: StagingContext,
        children: Sequence[EmbeddedChild],
        *,
        before_side_effect: Callable[[], Awaitable[None]] | None = None,
    ) -> int:
        await self._ensure_index(context.index_generation)
        index = self.index_name(context.index_generation)
        staged = 0
        for start in range(0, len(children), self._bulk_batch_size):
            batch = children[start : start + self._bulk_batch_size]
            if before_side_effect is not None:
                await before_side_effect()
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

    async def count_total(self, context: StagingContext) -> int:
        response = await self._client.count(
            index=self.index_name(context.index_generation),
            query={"bool": {"filter": self._scope_filters(context)}},
        )
        return int(cast(Mapping[str, Any], response.body)["count"])

    async def activate(self, context: StagingContext) -> None:
        await self._set_active(context, is_active=True)

    async def deactivate(self, context: StagingContext) -> None:
        await self._set_active(context, is_active=False)

    async def delete(self, context: StagingContext) -> None:
        response = await self._client.delete_by_query(
            index=self.index_name(context.index_generation),
            query={"bool": {"filter": self._scope_filters(context)}},
            allow_no_indices=True,
            conflicts="abort",
            ignore_unavailable=True,
            refresh=True,
        )
        self._require_complete_lifecycle(
            cast(Mapping[str, Any], response.body), operation="delete"
        )

    async def _set_active(self, context: StagingContext, *, is_active: bool) -> None:
        response = await self._client.update_by_query(
            index=self.index_name(context.index_generation),
            query={"bool": {"filter": self._scope_filters(context)}},
            script={
                "lang": "painless",
                "source": "ctx._source.is_active = params.is_active",
                "params": {"is_active": is_active},
            },
            allow_no_indices=True,
            conflicts="abort",
            ignore_unavailable=True,
            refresh=True,
        )
        self._require_complete_lifecycle(
            cast(Mapping[str, Any], response.body), operation="activation"
        )

    @staticmethod
    def _require_complete_lifecycle(body: Mapping[str, Any], *, operation: str) -> None:
        failures = body.get("failures", ())
        if body.get("timed_out") or int(body.get("version_conflicts", 0)) or failures:
            raise ChildIndexWriteError(
                f"Elasticsearch Child {operation} did not fully apply"
            )

    async def _count(self, context: StagingContext, *, is_active: bool) -> int:
        response = await self._client.count(
            index=self.index_name(context.index_generation),
            query={
                "bool": {
                    "filter": [
                        *self._scope_filters(context),
                        {"term": {"is_active": is_active}},
                    ]
                }
            },
        )
        return int(cast(Mapping[str, Any], response.body)["count"])

    @staticmethod
    def _scope_filters(context: StagingContext) -> list[Mapping[str, Any]]:
        return [
            {"term": {"user_id": context.user_id}},
            {"term": {"document_id": context.document_id}},
            {"term": {"document_version_id": context.document_version_id}},
            {"term": {"search_type": "document"}},
        ]

    async def _ensure_index(self, index_generation: str) -> None:
        index = self.index_name(index_generation)
        if index_generation in LEGACY_INDEX_GENERATIONS:
            raise ChildIndexMappingError(
                f"Index Generation {index_generation!r} uses the legacy Child schema; "
                f"stage new versions with {DEFAULT_INDEX_GENERATION!r}"
            )
        expected_mapping = _index_mapping(index_generation)
        try:
            await self._client.indices.create(
                index=index,
                mappings=expected_mapping,
            )
        except BadRequestError as error:
            body = error.body if isinstance(error.body, Mapping) else {}
            detail = body.get("error", {})
            error_type = detail.get("type") if isinstance(detail, Mapping) else None
            if error_type != "resource_already_exists_exception":
                raise
            response = await self._client.indices.get_mapping(index=index)
            mapping_body = cast(Mapping[str, Any], response.body)
            index_body = mapping_body.get(index)
            actual_mapping = (
                index_body.get("mappings") if isinstance(index_body, Mapping) else None
            )
            if not isinstance(actual_mapping, Mapping):
                raise ChildIndexMappingError(
                    f"existing index {index!r} did not return a readable strict mapping"
                )
            differences = _mapping_differences(actual_mapping, expected_mapping)
            if differences:
                raise ChildIndexMappingError(
                    f"existing index {index!r} has an incompatible strict mapping: "
                    + "; ".join(differences)
                )

    async def _read_active_alias_targets(self) -> tuple[str, ...]:
        try:
            response = await self._client.indices.get_alias(
                name=ACTIVE_CHILD_INDEX_ALIAS
            )
        except NotFoundError:
            return ()
        body = cast(Mapping[str, Any], response.body)
        return tuple(sorted(str(index) for index in body))

    async def _verify_active_alias(self, expected_index: str) -> None:
        targets = await self._read_active_alias_targets()
        if targets != (expected_index,):
            raise ChildIndexWriteError(
                "Elasticsearch active alias verification failed: "
                f"expected {expected_index!r}, found {', '.join(targets) or '<none>'}"
            )

    @staticmethod
    def _validate_metadata(context: StagingContext, child: EmbeddedChild) -> None:
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
            "version_no": context.version_no,
            "pipeline_version": context.pipeline_version,
            "embedding_version": context.embedding_version,
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


def _index_mapping(index_generation: str) -> dict[str, Any]:
    return {
        "dynamic": "strict",
        "_meta": {
            "index_generation": index_generation,
            "embedding_model": EMBEDDING_MODEL,
            "embedding_dimensions": EMBEDDING_DIMENSIONS,
            "schema_version": 2,
        },
        "properties": {
            "id": {"type": "keyword"},
            "parent_id": {"type": "keyword"},
            "parent_ordinal": {"type": "integer"},
            "user_id": {"type": "keyword"},
            "document_id": {"type": "keyword"},
            "document_version_id": {"type": "keyword"},
            "version_no": {"type": "integer"},
            "pipeline_version": {"type": "keyword"},
            "embedding_version": {"type": "keyword"},
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
                "dims": EMBEDDING_DIMENSIONS,
                "index": True,
                "similarity": "cosine",
            },
            "index_generation": {"type": "keyword"},
            "search_type": {"type": "keyword"},
            "is_active": {"type": "boolean"},
        },
    }


def _mapping_differences(
    actual: Mapping[str, Any], expected: Mapping[str, Any]
) -> list[str]:
    differences: list[str] = []
    if actual.get("dynamic") != expected["dynamic"]:
        differences.append("dynamic must be strict")
    if actual.get("_meta") != expected["_meta"]:
        differences.append("schema _meta does not match this generation")

    actual_properties = actual.get("properties")
    expected_properties = cast(Mapping[str, Any], expected["properties"])
    if not isinstance(actual_properties, Mapping):
        differences.append("properties are absent")
        return differences
    missing = sorted(set(expected_properties) - set(actual_properties))
    unexpected = sorted(set(actual_properties) - set(expected_properties))
    changed = sorted(
        field
        for field in set(actual_properties) & set(expected_properties)
        if not _mapping_field_matches(
            actual_properties[field], expected_properties[field]
        )
    )
    if missing:
        differences.append("missing properties: " + ", ".join(missing))
    if unexpected:
        differences.append("unexpected properties: " + ", ".join(unexpected))
    if changed:
        differences.append("changed properties: " + ", ".join(changed))
    return differences


def _mapping_field_matches(actual: object, expected: object) -> bool:
    """Compare reviewed mapping keys while tolerating ES server defaults.

    Elasticsearch adds an ``index_options`` object to dense-vector mappings
    when the HNSW index is created.  It is a server-owned representation of
    the requested ``index``/``similarity`` contract, not a schema change.  All
    application-owned dense-vector keys remain strict below.
    """
    if not isinstance(actual, Mapping) or not isinstance(expected, Mapping):
        return actual == expected
    if expected.get("type") != "dense_vector":
        return actual == expected
    # ``index_options`` is server-owned for HNSW vectors; every other field,
    # including ``element_type`` when configured, remains application-owned.
    normalized_actual = {
        key: value for key, value in actual.items() if key != "index_options"
    }
    return normalized_actual == dict(expected)
