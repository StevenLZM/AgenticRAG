"""Manifest-gated, idempotent document-version publication."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Awaitable, Callable
from typing import Protocol

from pydantic import BaseModel, ConfigDict, Field, model_validator

from agentic_rag.ingestion.indexer import (
    EMBEDDING_DIMENSIONS,
    EMBEDDING_MODEL,
    StagingContext,
)
from agentic_rag.ingestion.manifest import VersionManifest
from agentic_rag.models.indexing import LEGACY_INDEX_GENERATIONS
from agentic_rag.persistence.artifacts import ArtifactRef


class PublicationError(RuntimeError):
    """Base class for publication failures."""


class PublicationIntegrityError(PublicationError):
    """A durable version cannot be trusted enough to publish."""


class PublicationObsoleteError(PublicationError):
    """A newer durable winner superseded this candidate."""

    def __init__(self, context: StagingContext, winner_version_id: str) -> None:
        super().__init__(
            f"version {context.document_version_id!r} lost to {winner_version_id!r}"
        )
        self.context = context
        self.winner_version_id = winner_version_id


class PublicationTarget(BaseModel):
    """Trusted durable state needed to publish exactly one version."""

    model_config = ConfigDict(frozen=True)

    context: StagingContext
    active_version_id: str | None = None
    previous_context: StagingContext | None = None
    manifest_uri: str
    manifest_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    canonical_ast_uri: str
    canonical_ast_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    parent_count: int = Field(gt=0)
    child_count: int = Field(gt=0)

    @model_validator(mode="after")
    def _previous_version_matches_pointer(self) -> PublicationTarget:
        previous_id = (
            self.previous_context.document_version_id if self.previous_context else None
        )
        if self.active_version_id == self.context.document_version_id:
            return self
        if previous_id != self.active_version_id:
            raise ValueError("previous context must identify the active version")
        if self.previous_context is not None and (
            self.previous_context.user_id != self.context.user_id
            or self.previous_context.document_id != self.context.document_id
        ):
            raise ValueError(
                "previous version must remain in the same user/document scope"
            )
        return self


class PublicationRepository(Protocol):
    """MySQL publication boundary; ``finalize`` is one atomic transaction."""

    async def get_target(self, version_id: str) -> PublicationTarget | None: ...

    async def is_writable(self, context: StagingContext) -> bool: ...

    async def finalize(self, target: PublicationTarget) -> None: ...

    async def quarantine(self, version_id: str) -> None: ...


class VersionLifecycleStore(Protocol):
    """Replaceable physical-index lifecycle operations for one trusted version."""

    async def count_total(self, context: StagingContext) -> int: ...

    async def activate(self, context: StagingContext) -> None: ...

    async def deactivate(self, context: StagingContext) -> None: ...

    async def delete(self, context: StagingContext) -> None: ...


class ActiveIndexAliasStore(Protocol):
    """Atomically publish the validated physical index generation."""

    async def switch_active_alias(self, index_generation: str) -> bool: ...


class ManifestArtifactReader(Protocol):
    """Minimal immutable Artifact boundary required by publication."""

    def verify(self, ref: ArtifactRef) -> bool: ...

    def read_json(self, ref: ArtifactRef) -> object: ...

    def verify_hash(self, uri: str, sha256: str) -> bool: ...


class VersionPublisher:
    """Publish a fully staged version with a retry-safe fixed write sequence."""

    def __init__(
        self,
        *,
        repository: PublicationRepository,
        parent_store: VersionLifecycleStore,
        child_store: VersionLifecycleStore,
        artifacts: ManifestArtifactReader,
        alias_store: ActiveIndexAliasStore | None = None,
    ) -> None:
        self._repository = repository
        self._parent_store = parent_store
        self._child_store = child_store
        self._artifacts = artifacts
        self._alias_store = alias_store

    async def publish(
        self,
        version_id: str,
        *,
        before_side_effect: Callable[[], Awaitable[None]] | None = None,
    ) -> None:
        target = await self._repository.get_target(version_id)
        if target is None:
            raise PublicationIntegrityError(
                f"version {version_id!r} is not publishable"
            )

        await self._verify(target)

        # This order is a durable contract. Each operation is idempotent and scoped by
        # the trusted user/document/version identity in ``StagingContext``.
        await _fence(before_side_effect)
        await self._require_writable(target.context)
        await self._parent_store.activate(target.context)
        await _fence(before_side_effect)
        await self._require_writable(target.context)
        await self._child_store.activate(target.context)
        if target.previous_context is not None:
            await _fence(before_side_effect)
            await self._require_writable(target.context)
            await self._child_store.deactivate(target.previous_context)
            await _fence(before_side_effect)
            await self._require_writable(target.context)
            await self._parent_store.deactivate(target.previous_context)
        await _fence(before_side_effect)
        await self._require_writable(target.context)
        await self._repository.finalize(target)
        if self._alias_store is not None:
            await _fence(before_side_effect)
            await self._alias_store.switch_active_alias(
                target.context.index_generation
            )

    async def _require_writable(self, context: StagingContext) -> None:
        if not await self._repository.is_writable(context):
            raise PublicationIntegrityError(
                "document is no longer writable during publication"
            )

    async def _verify(self, target: PublicationTarget) -> None:
        if (
            target.context.index_generation in LEGACY_INDEX_GENERATIONS
            or target.context.embedding_version != EMBEDDING_MODEL
        ):
            raise PublicationIntegrityError(
                "version does not use the current strict Child index contract"
            )
        manifest = VersionManifest(
            canonical_ast_sha256=target.canonical_ast_sha256,
            parent_count=target.parent_count,
            child_count=target.child_count,
            embedding_model=EMBEDDING_MODEL,
            embedding_dimensions=EMBEDDING_DIMENSIONS,
            index_generation=target.context.index_generation,
        )
        encoded = json.dumps(
            manifest.payload(),
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        manifest_ref = ArtifactRef(
            uri=target.manifest_uri,
            sha256=target.manifest_hash,
            size_bytes=len(encoded),
        )
        expected_uri = (
            f"artifact://documents/{target.context.user_id}/"
            f"{target.context.document_id}/{target.context.document_version_id}/"
            f"manifests/{target.context.index_generation}/{target.manifest_hash}.json"
        )
        if target.manifest_uri != expected_uri:
            raise PublicationIntegrityError(
                "Manifest Artifact URI is outside the trusted version scope"
            )
        if target.manifest_hash != manifest.manifest_hash:
            raise PublicationIntegrityError(
                "durable Manifest hash does not match counts"
            )
        if not await asyncio.to_thread(self._artifacts.verify, manifest_ref):
            raise PublicationIntegrityError(
                "Manifest Artifact failed integrity verification"
            )
        try:
            payload = await asyncio.to_thread(self._artifacts.read_json, manifest_ref)
        except (OSError, TypeError, ValueError) as error:
            raise PublicationIntegrityError(
                "Manifest Artifact is unreadable"
            ) from error
        if payload != manifest.payload():
            raise PublicationIntegrityError(
                "Manifest Artifact payload does not match version"
            )
        canonical_prefix = (
            f"artifact://documents/{target.context.user_id}/"
            f"{target.context.document_id}/{target.context.document_version_id}/canonical/"
        )
        if not target.canonical_ast_uri.startswith(canonical_prefix):
            raise PublicationIntegrityError(
                "Canonical AST Artifact URI is outside the trusted version scope"
            )
        if not await asyncio.to_thread(
            self._artifacts.verify_hash,
            target.canonical_ast_uri,
            target.canonical_ast_sha256,
        ):
            raise PublicationIntegrityError(
                "Canonical AST Artifact failed integrity verification"
            )

        parent_count = await self._parent_store.count_total(target.context)
        child_count = await self._child_store.count_total(target.context)
        if parent_count != target.parent_count or child_count != target.child_count:
            raise PublicationIntegrityError(
                "staged Parent/Child counts do not match the durable Manifest"
            )


async def _fence(callback: Callable[[], Awaitable[None]] | None) -> None:
    if callback is not None:
        await callback()
