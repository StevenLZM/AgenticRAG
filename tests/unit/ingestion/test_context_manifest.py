"""Manifest v2 binds exact stored representation identities; v1 stays byte stable."""

import hashlib
import json
import pytest
from agentic_rag.ingestion.manifest import VersionManifest
from agentic_rag.ingestion.indexer import IndexWriter
from tests.unit.ingestion.test_manifest import (
    FakeEmbedding,
    RecordingParentStore,
    RecordingChildStore,
    _context,
)
from tests.unit.ingestion.test_context_service import setup, config, Recorded, fence
from agentic_rag.ingestion.contextualization.service import ContextualizationService
from agentic_rag.ingestion.contextualization.store import MemoryContextBuildStore


@pytest.mark.asyncio
async def test_staging_context_manifest_binds_stored_artifacts_and_child_digest(
    tmp_path,
):
    artifacts, ast, ref, parent, claim = setup(tmp_path, ("本制度适用于星河公司。",))
    result = await ContextualizationService(
        artifacts=artifacts,
        store=MemoryContextBuildStore(),
        model=Recorded(),
        count_tokens=len,
        count_model_tokens=lambda text, model_id: len(text),
    ).enrich(
        ast,
        [parent],
        canonical_ref=ref,
        build=config(),
        claim=claim,
        before_side_effect=fence,
    )
    events = []
    embedding = FakeEmbedding([[[0.0] * 1024]])
    writer = IndexWriter(
        embedding=embedding,
        parent_store=RecordingParentStore(events),
        child_store=RecordingChildStore(events),
        artifacts=artifacts,
    )
    manifest = await writer.stage(
        result.parents,
        context=_context().model_copy(update={"index_generation": "context-test-v1"}),
        canonical_ast=ref,
        context_artifact_ref=result.context_artifact_ref,
        enriched_chunks_ref=result.enriched_chunks_ref,
    )
    assert manifest.payload()["schema_version"] == 2
    assert manifest.enriched_chunks_ref == result.enriched_chunks_ref
    assert embedding.calls == [(result.parents[0].children[0].contextualized_content,)]
    with pytest.raises(ValueError, match="artifact|context|enriched"):
        await writer.stage(
            result.parents,
            context=_context().model_copy(
                update={"index_generation": "context-test-v1"}
            ),
            canonical_ast=ref,
            context_artifact_ref=result.context_artifact_ref.model_copy(
                update={"sha256": "f" * 64}
            ),
            enriched_chunks_ref=result.enriched_chunks_ref,
        )


def test_old_manifest_hash_and_payload_unchanged():
    payload = dict(
        canonical_ast_sha256="a" * 64,
        parent_count=1,
        child_count=1,
        embedding_model="text-embedding-v3",
        embedding_dimensions=1024,
        index_generation="index-v2",
    )
    manifest = VersionManifest(**payload)
    assert manifest.payload() == payload
    assert (
        manifest.manifest_hash
        == hashlib.sha256(
            json.dumps(
                payload, ensure_ascii=False, separators=(",", ":"), sort_keys=True
            ).encode()
        ).hexdigest()
    )


@pytest.mark.asyncio
async def test_publisher_rejects_same_count_wrong_child_hash(tmp_path):
    from agentic_rag.ingestion.publisher import (
        VersionPublisher,
        PublicationTarget,
        PublicationIntegrityError,
    )
    from tests.unit.ingestion.test_publisher_reconciler_unit import (
        _PublicationRepository,
        _LifecycleStore,
    )

    artifacts, ast, ref, parent, claim = setup(tmp_path, ("本制度适用于星河公司。",))
    canonical = artifacts.put_json(
        "documents/user-1/document-1/version-1/canonical/ast.json",
        ast.model_dump(mode="json"),
    )
    result = await ContextualizationService(
        artifacts=artifacts,
        store=MemoryContextBuildStore(),
        model=Recorded(),
        count_tokens=len,
        count_model_tokens=lambda text, model_id: len(text),
    ).enrich(
        ast,
        [parent],
        canonical_ref=ref,
        build=config(),
        claim=claim,
        before_side_effect=fence,
    )
    events = []
    parents = RecordingParentStore(events)
    writer = IndexWriter(
        embedding=FakeEmbedding([[[0.0] * 1024]]),
        parent_store=parents,
        child_store=RecordingChildStore(events),
        artifacts=artifacts,
    )
    manifest = await writer.stage(
        result.parents,
        context=_context().model_copy(update={"index_generation": "context-test-v1"}),
        canonical_ast=canonical,
        context_artifact_ref=result.context_artifact_ref,
        enriched_chunks_ref=result.enriched_chunks_ref,
    )
    target = PublicationTarget(
        context=_context().model_copy(update={"index_generation": "context-test-v1"}),
        canonical_ast_uri=canonical.uri,
        canonical_ast_sha256=canonical.sha256,
        parent_count=1,
        child_count=1,
        manifest_uri=parents.attachment["manifest_uri"],
        manifest_hash=manifest.manifest_hash,
    )
    repo = _PublicationRepository(target, events, active_version_id=None)

    class Children(_LifecycleStore):
        actual_hash = "f" * 64

        async def child_identities(self, context):
            return ((result.parents[0].children[0].id, self.actual_hash),)

    children = Children("children", events, set(), {"version-1": 1})
    publisher = VersionPublisher(
        repository=repo,
        parent_store=_LifecycleStore("parents", events, set(), {"version-1": 1}),
        child_store=children,
        artifacts=artifacts,
    )
    with pytest.raises(PublicationIntegrityError, match="identit|digest"):
        await publisher.publish("version-1")
    assert not children.active
    children.actual_hash = result.parents[0].children[0].content_hash
    await publisher.publish("version-1")
    assert children.active == {"version-1"}
