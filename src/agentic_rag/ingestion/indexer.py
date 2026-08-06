"""Embedding validation and cross-store invisible staging orchestration."""

from __future__ import annotations

import asyncio
import math
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from typing import Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field, field_validator

from agentic_rag.ingestion.chunker import ChildChunk, ParentChunk
from agentic_rag.ingestion.manifest import VersionManifest
from agentic_rag.models.embeddings import EmbeddingPort
from agentic_rag.models.indexing import validate_index_generation
from agentic_rag.persistence.artifacts import ArtifactRef, ArtifactStore


EMBEDDING_MODEL: Literal["text-embedding-v3"] = "text-embedding-v3"
EMBEDDING_DIMENSIONS: Literal[1024] = 1024


class EmbeddingValidationError(ValueError):
    """Base class for fail-closed embedding response validation."""


class EmbeddingCountError(EmbeddingValidationError):
    """Raised when a provider returns a different number of vectors."""


class EmbeddingDimensionError(EmbeddingValidationError):
    """Raised when any vector differs from the configured target dimension."""


class EmbeddingValueError(EmbeddingValidationError):
    """Raised when a vector contains a non-numeric or non-finite element."""


class StagingContext(BaseModel):
    """Trusted identity supplied from the durable ingestion Job, never source text."""

    model_config = ConfigDict(frozen=True)

    user_id: str
    document_id: str
    document_version_id: str
    version_no: int = Field(gt=0)
    pipeline_version: str
    embedding_version: str
    index_generation: str

    @field_validator(
        "user_id",
        "document_id",
        "document_version_id",
        "pipeline_version",
        "embedding_version",
    )
    @classmethod
    def _not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("staging identity values must not be blank")
        return value

    @field_validator("index_generation")
    @classmethod
    def _canonical_index_generation(cls, value: str) -> str:
        return validate_index_generation(value)


@dataclass(frozen=True, slots=True)
class EmbeddedChild:
    """Child chunk plus trusted filtering metadata and validated vector."""

    chunk: ChildChunk
    embedding: tuple[float, ...]
    user_id: str
    document_id: str
    document_version_id: str
    index_generation: str
    search_type: str = "document"
    is_active: bool = False


class ParentStagingStore(Protocol):
    """MySQL-side staging boundary."""

    async def assert_writable(self, context: StagingContext) -> None: ...

    async def stage(
        self, context: StagingContext, parents: Sequence[ParentChunk]
    ) -> int: ...

    async def count(self, context: StagingContext) -> int: ...

    async def attach_manifest(
        self,
        context: StagingContext,
        *,
        canonical_ast_uri: str,
        canonical_ast_sha256: str,
        manifest_uri: str,
        manifest_hash: str,
        parent_count: int,
        child_count: int,
    ) -> None: ...


class ChildIndexStore(Protocol):
    """Replaceable vector-index staging boundary (ES now, Milvus later)."""

    async def stage(
        self,
        context: StagingContext,
        children: Sequence[EmbeddedChild],
        *,
        before_side_effect: Callable[[], Awaitable[None]] | None = None,
    ) -> int: ...

    async def count(self, context: StagingContext) -> int: ...


class StagingError(RuntimeError):
    """Base class for staging failures that must not publish a version."""


class StagingIdentityError(StagingError):
    """Raised when chunk metadata disagrees with the durable Job context."""


class StagingCountError(StagingError):
    """Raised when either store does not contain the exact expected records."""


class IndexWriter:
    """Stage one document version without making it searchable.

    Parent and Child writes use deterministic IDs and remain inactive. A retry may
    safely re-enter either adapter after a partial cross-store failure. The
    Manifest is persisted and attached only after exact counts from both stores.
    """

    def __init__(
        self,
        *,
        embedding: EmbeddingPort,
        parent_store: ParentStagingStore,
        child_store: ChildIndexStore,
        artifacts: ArtifactStore,
        embedding_batch_size: int = 64,
        embedding_dimensions: int = EMBEDDING_DIMENSIONS,
    ) -> None:
        if embedding_batch_size <= 0:
            raise ValueError("embedding_batch_size must be positive")
        if embedding_dimensions != EMBEDDING_DIMENSIONS:
            raise ValueError(
                f"this index generation requires {EMBEDDING_DIMENSIONS} dimensions"
            )
        self._embedding = embedding
        self._parent_store = parent_store
        self._child_store = child_store
        self._artifacts = artifacts
        self._embedding_batch_size = embedding_batch_size
        self._embedding_dimensions = embedding_dimensions

    async def stage(
        self,
        chunks: Sequence[ParentChunk],
        *,
        context: StagingContext,
        canonical_ast: ArtifactRef,
        before_side_effect: Callable[[], Awaitable[None]] | None = None,
    ) -> VersionManifest:
        parents = tuple(chunks)
        children = self._validate_and_flatten(parents, context)
        if not await asyncio.to_thread(self._artifacts.verify, canonical_ast):
            raise StagingError("Canonical AST Artifact failed integrity verification")

        embedded_children = await self._embed(
            children, context, before_side_effect=before_side_effect
        )

        await _fence(before_side_effect)
        await self._parent_store.assert_writable(context)
        staged_parents = await self._parent_store.stage(context, parents)
        if staged_parents != len(parents):
            raise StagingCountError(
                f"Parent stage reported {staged_parents}; expected {len(parents)}"
            )
        await _fence(before_side_effect)
        await self._parent_store.assert_writable(context)
        staged_children = await self._child_store.stage(
            context,
            embedded_children,
            before_side_effect=before_side_effect,
        )
        if staged_children != len(children):
            raise StagingCountError(
                f"Child stage reported {staged_children}; expected {len(children)}"
            )

        await _fence(before_side_effect)
        parent_count = await self._parent_store.count(context)
        await _fence(before_side_effect)
        child_count = await self._child_store.count(context)
        if parent_count != len(parents):
            raise StagingCountError(
                f"Parent store contains {parent_count}; expected {len(parents)}"
            )
        if child_count != len(children):
            raise StagingCountError(
                f"Child store contains {child_count}; expected {len(children)}"
            )

        manifest = VersionManifest(
            canonical_ast_sha256=canonical_ast.sha256,
            parent_count=parent_count,
            child_count=child_count,
            embedding_model=EMBEDDING_MODEL,
            embedding_dimensions=EMBEDDING_DIMENSIONS,
            index_generation=context.index_generation,
        )
        await _fence(before_side_effect)
        await self._parent_store.assert_writable(context)
        manifest_ref = await asyncio.to_thread(
            self._artifacts.put_json,
            self._manifest_path(context, manifest),
            manifest.payload(),
        )
        if manifest_ref.sha256 != manifest.manifest_hash:
            await asyncio.to_thread(self._artifacts.delete, manifest_ref)
            raise StagingError("Manifest Artifact hash is not deterministic")
        # Keep a valid deterministic Artifact when attachment fails. It may already
        # be referenced by an earlier successful retry; deleting it would corrupt
        # that version. A later retry overwrites the same bytes and re-attaches it.
        await _fence(before_side_effect)
        await self._parent_store.assert_writable(context)
        await self._parent_store.attach_manifest(
            context,
            canonical_ast_uri=canonical_ast.uri,
            canonical_ast_sha256=canonical_ast.sha256,
            manifest_uri=manifest_ref.uri,
            manifest_hash=manifest_ref.sha256,
            parent_count=parent_count,
            child_count=child_count,
        )
        return manifest

    def _validate_and_flatten(
        self, parents: tuple[ParentChunk, ...], context: StagingContext
    ) -> tuple[ChildChunk, ...]:
        if not parents:
            raise StagingIdentityError("staging requires at least one Parent chunk")
        children: list[ChildChunk] = []
        parent_ids: set[str] = set()
        child_ids: set[str] = set()
        for parent in parents:
            self._validate_parent(parent, context)
            if parent.id in parent_ids:
                raise StagingIdentityError(f"duplicate Parent id {parent.id!r}")
            parent_ids.add(parent.id)
            if not parent.children:
                raise StagingIdentityError(
                    f"Parent {parent.id!r} requires at least one Child"
                )
            for child in parent.children:
                self._validate_child(child, parent, context)
                if child.id in child_ids:
                    raise StagingIdentityError(f"duplicate Child id {child.id!r}")
                child_ids.add(child.id)
                children.append(child)
        return tuple(children)

    @staticmethod
    def _validate_parent(parent: ParentChunk, context: StagingContext) -> None:
        actual = (parent.user_id, parent.document_id, parent.document_version_id)
        expected = (context.user_id, context.document_id, context.document_version_id)
        if actual != expected:
            raise StagingIdentityError(
                f"Parent {parent.id!r} identity does not match durable Job context"
            )

    @staticmethod
    def _validate_child(
        child: ChildChunk, parent: ParentChunk, context: StagingContext
    ) -> None:
        actual = (child.user_id, child.document_id, child.document_version_id)
        expected = (context.user_id, context.document_id, context.document_version_id)
        if actual != expected or child.parent_id != parent.id:
            raise StagingIdentityError(
                f"Child {child.id!r} identity does not match Parent/Job context"
            )

    async def _embed(
        self,
        children: tuple[ChildChunk, ...],
        context: StagingContext,
        *,
        before_side_effect: Callable[[], Awaitable[None]] | None = None,
    ) -> tuple[EmbeddedChild, ...]:
        staged: list[EmbeddedChild] = []
        for start in range(0, len(children), self._embedding_batch_size):
            batch = children[start : start + self._embedding_batch_size]
            await _fence(before_side_effect)
            vectors = await self._embedding.embed_documents(
                tuple(child.contextualized_content for child in batch)
            )
            if len(vectors) != len(batch):
                raise EmbeddingCountError(
                    f"embedding provider returned {len(vectors)} vectors for "
                    f"{len(batch)} documents"
                )
            for child, raw_vector in zip(batch, vectors, strict=True):
                vector = self._validate_vector(child.id, raw_vector)
                staged.append(
                    EmbeddedChild(
                        chunk=child,
                        embedding=vector,
                        user_id=context.user_id,
                        document_id=context.document_id,
                        document_version_id=context.document_version_id,
                        index_generation=context.index_generation,
                    )
                )
        return tuple(staged)

    def _validate_vector(
        self, child_id: str, raw_vector: Sequence[float]
    ) -> tuple[float, ...]:
        if len(raw_vector) != self._embedding_dimensions:
            raise EmbeddingDimensionError(
                f"embedding for {child_id!r} has {len(raw_vector)} dimensions; "
                f"expected {self._embedding_dimensions}"
            )
        vector: list[float] = []
        for index, value in enumerate(raw_vector):
            if isinstance(value, bool):
                raise EmbeddingValueError(
                    f"embedding for {child_id!r} contains a non-numeric value at {index}"
                )
            try:
                converted = float(value)
            except (TypeError, ValueError) as error:
                raise EmbeddingValueError(
                    f"embedding for {child_id!r} contains a non-numeric value at {index}"
                ) from error
            if not math.isfinite(converted):
                raise EmbeddingValueError(
                    f"embedding for {child_id!r} contains a non-finite value at {index}"
                )
            vector.append(converted)
        return tuple(vector)

    @staticmethod
    def _manifest_path(context: StagingContext, manifest: VersionManifest) -> str:
        return (
            f"documents/{context.user_id}/{context.document_id}/"
            f"{context.document_version_id}/manifests/{context.index_generation}/"
            f"{manifest.manifest_hash}.json"
        )


async def _fence(callback: Callable[[], Awaitable[None]] | None) -> None:
    if callback is not None:
        await callback()
