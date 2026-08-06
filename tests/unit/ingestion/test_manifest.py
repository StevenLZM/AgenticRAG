"""Behavioral contracts for embedding validation and invisible staging."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest

from agentic_rag.ingestion.chunker import (
    AstLocator,
    AstSpan,
    ChildChunk,
    ParentChunk,
)
from agentic_rag.ingestion.indexer import (
    ChildIndexStore,
    EmbeddedChild,
    EmbeddingCountError,
    EmbeddingDimensionError,
    IndexWriter,
    ParentStagingStore,
    StagingContext,
    StagingCountError,
)
from agentic_rag.ingestion.manifest import VersionManifest
from agentic_rag.persistence.artifacts import ArtifactRef, LocalArtifactStore


CANONICAL_SHA256 = "a" * 64


def _locator() -> AstLocator:
    return AstLocator(
        spans=(
            AstSpan(
                canonical_path="#/text_blocks/0",
                block_id="block-1",
                page_from=1,
                page_to=1,
                char_from=0,
                char_to=12,
                parent_char_from=0,
                parent_char_to=12,
            ),
        ),
        segment_ordinal=0,
        parent_char_from=0,
        parent_char_to=12,
    )


def _chunks() -> list[ParentChunk]:
    locator = _locator()
    child = ChildChunk(
        id="child-1",
        parent_id="parent-1",
        parent_ordinal=0,
        document_id="document-1",
        document_version_id="version-1",
        user_id="user-1",
        ordinal=0,
        heading_path=("Overview",),
        heading_ast_locators=("#/text_blocks/heading-1",),
        content_type="paragraph",
        content="hello world!",
        contextualized_content="Overview\nhello world!",
        token_count=4,
        page_from=1,
        page_to=1,
        ast_locator=locator,
        content_hash="b" * 64,
    )
    return [
        ParentChunk(
            id="parent-1",
            document_id="document-1",
            document_version_id="version-1",
            user_id="user-1",
            ordinal=0,
            heading_path=("Overview",),
            heading_ast_locators=("#/text_blocks/heading-1",),
            content_type="paragraph",
            content="hello world!",
            token_count=3,
            page_from=1,
            page_to=1,
            ast_locator=locator,
            content_hash="c" * 64,
            children=(child,),
        )
    ]


class FakeEmbedding:
    def __init__(self, responses: Sequence[Sequence[Sequence[float]]]) -> None:
        self._responses = [list(map(list, response)) for response in responses]
        self.calls: list[tuple[str, ...]] = []

    async def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        self.calls.append(tuple(texts))
        return self._responses.pop(0)

    async def embed_query(self, text: str) -> list[float]:
        raise AssertionError("query embedding is outside ingestion staging")


class RecordingParentStore(ParentStagingStore):
    def __init__(self, events: list[str], *, count: int = 1) -> None:
        self.events = events
        self._count = count
        self.staged: tuple[ParentChunk, ...] = ()
        self.attachment: dict[str, Any] | None = None
        self.attach_error: BaseException | None = None
        self.writable = True
        self.become_nonwritable_after_stage = False

    async def assert_writable(self, context: StagingContext) -> None:
        if not self.writable:
            raise RuntimeError("document is deleted")

    async def stage(
        self, context: StagingContext, parents: Sequence[ParentChunk]
    ) -> int:
        self.events.append("parents.stage")
        self.staged = tuple(parents)
        if self.become_nonwritable_after_stage:
            self.writable = False
        return len(parents)

    async def count(self, context: StagingContext) -> int:
        self.events.append("parents.count")
        return self._count

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
    ) -> None:
        self.events.append("parents.attach_manifest")
        if self.attach_error is not None:
            raise self.attach_error
        self.attachment = {
            "context": context,
            "canonical_ast_uri": canonical_ast_uri,
            "canonical_ast_sha256": canonical_ast_sha256,
            "manifest_uri": manifest_uri,
            "manifest_hash": manifest_hash,
            "parent_count": parent_count,
            "child_count": child_count,
        }


class RecordingChildStore(ChildIndexStore):
    def __init__(self, events: list[str], *, count: int = 1) -> None:
        self.events = events
        self._count = count
        self.staged: tuple[EmbeddedChild, ...] = ()

    async def stage(
        self,
        context: StagingContext,
        children: Sequence[EmbeddedChild],
        *,
        before_side_effect: Any = None,
    ) -> int:
        self.events.append("children.stage")
        self.staged = tuple(children)
        return len(children)

    async def count(self, context: StagingContext) -> int:
        self.events.append("children.count")
        return self._count


def _context() -> StagingContext:
    return StagingContext(
        user_id="user-1",
        document_id="document-1",
        document_version_id="version-1",
        version_no=1,
        pipeline_version="ingestion-v1",
        embedding_version="text-embedding-v3",
        index_generation="index-v1",
    )


def _writer(
    tmp_path: Path,
    embedding: FakeEmbedding,
    parents: RecordingParentStore,
    children: RecordingChildStore,
    *,
    embedding_batch_size: int = 64,
) -> tuple[IndexWriter, LocalArtifactStore]:
    artifacts = LocalArtifactStore(tmp_path / "artifacts")
    return (
        IndexWriter(
            embedding=embedding,
            parent_store=parents,
            child_store=children,
            artifacts=artifacts,
            embedding_batch_size=embedding_batch_size,
        ),
        artifacts,
    )


def test_manifest_hash_is_deterministic_and_changes_with_counts() -> None:
    first = VersionManifest(
        canonical_ast_sha256=CANONICAL_SHA256,
        parent_count=1,
        child_count=2,
        embedding_model="text-embedding-v3",
        embedding_dimensions=1024,
        index_generation="index-v1",
    )
    second = VersionManifest(
        index_generation="index-v1",
        embedding_dimensions=1024,
        embedding_model="text-embedding-v3",
        child_count=2,
        parent_count=1,
        canonical_ast_sha256=CANONICAL_SHA256,
    )
    changed = first.model_copy(update={"child_count": 3})

    assert first.manifest_hash == second.manifest_hash
    assert changed.manifest_hash != first.manifest_hash
    assert len(first.manifest_hash) == 64


@pytest.mark.parametrize("generation", ["INDEX-V1", " index-v1 ", "index/v1"])
def test_staging_context_rejects_noncanonical_index_generation(
    generation: str,
) -> None:
    with pytest.raises(ValueError):
        StagingContext(
            user_id="user-1",
            document_id="document-1",
            document_version_id="version-1",
            version_no=1,
            pipeline_version="ingestion-v1",
            embedding_version="text-embedding-v3",
            index_generation=generation,
        )


@pytest.mark.asyncio
async def test_indexer_rejects_each_wrong_embedding_dimension_before_writes(
    tmp_path: Path,
) -> None:
    events: list[str] = []
    embedding = FakeEmbedding([[[0.0] * 1536]])
    parents = RecordingParentStore(events)
    children = RecordingChildStore(events)
    writer, artifacts = _writer(tmp_path, embedding, parents, children)
    canonical = artifacts.put_bytes("canonical.json", b"canonical")

    with pytest.raises(EmbeddingDimensionError, match="child-1"):
        await writer.stage(_chunks(), context=_context(), canonical_ast=canonical)

    assert events == []
    assert parents.attachment is None


@pytest.mark.asyncio
async def test_indexer_rejects_embedding_response_count_before_writes(
    tmp_path: Path,
) -> None:
    events: list[str] = []
    embedding = FakeEmbedding([[]])
    parents = RecordingParentStore(events)
    children = RecordingChildStore(events)
    writer, artifacts = _writer(tmp_path, embedding, parents, children)
    canonical = artifacts.put_bytes("canonical.json", b"canonical")

    with pytest.raises(EmbeddingCountError):
        await writer.stage(_chunks(), context=_context(), canonical_ast=canonical)

    assert events == []


@pytest.mark.asyncio
async def test_manifest_is_written_only_after_both_store_counts_match(
    tmp_path: Path,
) -> None:
    events: list[str] = []
    embedding = FakeEmbedding([[[0.0] * 1024]])
    parents = RecordingParentStore(events)
    children = RecordingChildStore(events, count=0)
    writer, artifacts = _writer(tmp_path, embedding, parents, children)
    canonical = artifacts.put_bytes("canonical.json", b"canonical")

    with pytest.raises(StagingCountError, match="Child"):
        await writer.stage(_chunks(), context=_context(), canonical_ast=canonical)

    assert events == [
        "parents.stage",
        "children.stage",
        "parents.count",
        "children.count",
    ]
    assert parents.attachment is None


@pytest.mark.asyncio
async def test_staging_rechecks_document_writability_before_child_write(
    tmp_path: Path,
) -> None:
    events: list[str] = []
    embedding = FakeEmbedding([[[0.0] * 1024]])
    parents = RecordingParentStore(events)
    parents.become_nonwritable_after_stage = True
    children = RecordingChildStore(events)
    writer, artifacts = _writer(tmp_path, embedding, parents, children)
    canonical = artifacts.put_bytes("canonical.json", b"canonical")

    with pytest.raises(RuntimeError, match="deleted"):
        await writer.stage(_chunks(), context=_context(), canonical_ast=canonical)

    assert events == ["parents.stage"]
    assert children.staged == ()


@pytest.mark.asyncio
async def test_staging_checks_lease_at_each_cross_store_side_effect(
    tmp_path: Path,
) -> None:
    events: list[str] = []
    embedding = FakeEmbedding([[[0.0] * 1024]])
    parents = RecordingParentStore(events)
    children = RecordingChildStore(events)
    writer, artifacts = _writer(tmp_path, embedding, parents, children)
    canonical = artifacts.put_bytes("canonical.json", b"canonical")
    checks = 0

    async def lease_fence() -> None:
        nonlocal checks
        checks += 1
        # embedding, parent stage, then child-stage boundary
        if checks == 3:
            raise RuntimeError("lease lost before child stage")

    with pytest.raises(RuntimeError, match="lease lost"):
        await writer.stage(
            _chunks(),
            context=_context(),
            canonical_ast=canonical,
            before_side_effect=lease_fence,
        )

    assert events == ["parents.stage"]
    assert children.staged == ()
    assert parents.attachment is None


@pytest.mark.asyncio
async def test_valid_staging_is_invisible_and_attaches_deterministic_manifest(
    tmp_path: Path,
) -> None:
    events: list[str] = []
    embedding = FakeEmbedding([[[0.25] * 1024]])
    parents = RecordingParentStore(events)
    children = RecordingChildStore(events)
    writer, artifacts = _writer(tmp_path, embedding, parents, children)
    canonical = artifacts.put_bytes("canonical.json", b"canonical")

    manifest = await writer.stage(
        _chunks(), context=_context(), canonical_ast=canonical
    )

    assert events == [
        "parents.stage",
        "children.stage",
        "parents.count",
        "children.count",
        "parents.attach_manifest",
    ]
    assert children.staged[0].is_active is False
    assert children.staged[0].user_id == "user-1"
    assert children.staged[0].search_type == "document"
    assert len(children.staged[0].embedding) == 1024
    assert manifest.parent_count == 1
    assert manifest.child_count == 1
    assert parents.attachment is not None
    assert parents.attachment["manifest_hash"] == manifest.manifest_hash
    assert parents.attachment["canonical_ast_sha256"] == canonical.sha256
    manifest_ref = parents.attachment["manifest_uri"]
    assert isinstance(manifest_ref, str)
    assert manifest_ref.startswith("artifact://documents/user-1/document-1/version-1/")


@pytest.mark.asyncio
async def test_indexer_batches_embeddings_at_configured_retry_boundary(
    tmp_path: Path,
) -> None:
    parent = _chunks()[0]
    second_child = parent.children[0].model_copy(
        update={"id": "child-2", "ordinal": 1, "content_hash": "d" * 64}
    )
    chunks = [parent.model_copy(update={"children": (*parent.children, second_child)})]
    events: list[str] = []
    embedding = FakeEmbedding([[[0.0] * 1024], [[1.0] * 1024]])
    parents = RecordingParentStore(events)
    children = RecordingChildStore(events, count=2)
    writer, artifacts = _writer(
        tmp_path, embedding, parents, children, embedding_batch_size=1
    )
    canonical = artifacts.put_bytes("canonical.json", b"canonical")

    manifest = await writer.stage(chunks, context=_context(), canonical_ast=canonical)

    assert embedding.calls == [
        ("Overview\nhello world!",),
        ("Overview\nhello world!",),
    ]
    assert manifest.child_count == 2


@pytest.mark.asyncio
async def test_retry_attachment_failure_keeps_previously_referenced_manifest(
    tmp_path: Path,
) -> None:
    events: list[str] = []
    embedding = FakeEmbedding([[[0.0] * 1024], [[0.0] * 1024]])
    parents = RecordingParentStore(events)
    children = RecordingChildStore(events)
    writer, artifacts = _writer(tmp_path, embedding, parents, children)
    canonical = artifacts.put_bytes("canonical.json", b"canonical")
    manifest = await writer.stage(
        _chunks(), context=_context(), canonical_ast=canonical
    )
    assert parents.attachment is not None
    encoded_manifest = json.dumps(
        manifest.payload(),
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    referenced = ArtifactRef(
        uri=parents.attachment["manifest_uri"],
        sha256=parents.attachment["manifest_hash"],
        size_bytes=len(encoded_manifest),
    )
    parents.attach_error = RuntimeError("temporary database failure")

    with pytest.raises(RuntimeError, match="temporary database failure"):
        await writer.stage(_chunks(), context=_context(), canonical_ast=canonical)

    assert artifacts.verify(referenced) is True
