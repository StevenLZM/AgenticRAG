"""Enabled graph executes real context service and staging, preserving raw refs."""

import pytest
from langgraph.checkpoint.memory import InMemorySaver
from agentic_rag.ingestion.graph import IngestionGraph, DefaultIngestionPipeline
from agentic_rag.ingestion.state import ArtifactPointer, IngestionRuntime
from agentic_rag.ingestion.indexer import IndexWriter
from agentic_rag.ingestion.contextualization.service import ContextualizationService
from agentic_rag.ingestion.contextualization.store import MemoryContextBuildStore
from tests.unit.ingestion.test_context_service import setup, Recorded, config
from tests.unit.ingestion.test_manifest import (
    FakeEmbedding,
    RecordingParentStore,
    RecordingChildStore,
)


@pytest.mark.asyncio
async def test_enabled_graph_real_service_staging_and_reference_only_state(tmp_path):
    artifacts, ast, ref, parent, claim = setup(tmp_path, ("本制度适用于星河公司。",))
    raw = artifacts.put_json(
        "raw-parents.json", {"parents": [parent.model_dump(mode="json")]}
    )
    events = []
    model = Recorded()
    embedding = FakeEmbedding([[[0.0] * 1024]])

    class Jobs:
        async def assert_lease(self, c):
            pass

        async def complete(self, c):
            pass

    class Pipeline(DefaultIngestionPipeline):
        async def load_job(self, c):
            return IngestionRuntime(
                job_id=c.job_id,
                user_id=c.user_id,
                document_id=c.document_id,
                document_version_id=c.document_version_id,
                version_no=1,
                filename="x.txt",
                mime_type="text/plain",
                source_type="text",
                content_hash="a" * 64,
                parser_version="p1",
                pipeline_version="context-v1",
                embedding_version="text-embedding-v3",
                index_generation="index-v2",
            )

        async def upload_safety_gate(self, r):
            return ArtifactPointer(**ref.model_dump())

        async def parse_fragments(self, *args):
            return [ArtifactPointer(**ref.model_dump())]

        async def assemble_canonical(self, *args):
            return ArtifactPointer(**ref.model_dump())

        async def content_safety_gate(self, *args):
            return True, []

        async def chunk(self, *args):
            return ArtifactPointer(**raw.__dict__)

        async def publish(self, *args):
            events.append("publish")

    service = ContextualizationService(
        artifacts=artifacts,
        store=MemoryContextBuildStore(),
        model=model,
        count_tokens=len,
        count_model_tokens=lambda text, model_id: len(text),
    )
    writer = IndexWriter(
        embedding=embedding,
        parent_store=RecordingParentStore(events),
        child_store=RecordingChildStore(events),
        artifacts=artifacts,
    )
    pipeline = Pipeline(
        jobs=Jobs(),
        artifacts=artifacts,
        upload_scanner=None,
        parser=None,
        assembler=None,
        content_scanner=None,
        chunker=None,
        index_writer=writer,
        publisher=None,
        contextualizer=service,
        context_build=config(),
    )
    graph = IngestionGraph(
        pipeline=pipeline, checkpointer=InMemorySaver(), contextualization_enabled=True
    )
    state = await graph.run(claim)
    assert state["terminal_status"] == "completed"
    assert (
        state["completed_nodes"].index("contextualize")
        == state["completed_nodes"].index("chunk") + 1
    )
    assert state["chunks_ref"]["uri"] == raw.uri
    assert state["enriched_chunks_ref"]["uri"] != raw.uri
    assert "company: 星河公司" in embedding.calls[0][0]
    assert "publish" in events
    assert artifacts.read_json(raw) == {"parents": [parent.model_dump(mode="json")]}


@pytest.mark.asyncio
async def test_old_checkpoint_cannot_resume_under_new_context_configuration():
    from agentic_rag.ingestion.state import IngestionJobClaim
    from agentic_rag.ingestion.worker import DeterministicIngestionError

    class Pipeline:
        context_config_hash = "frozen"

    saver = InMemorySaver()
    graph = IngestionGraph(
        pipeline=Pipeline(), checkpointer=saver, contextualization_enabled=True
    )
    claim = IngestionJobClaim(
        job_id="old",
        user_id="u",
        document_id="d",
        document_version_id="v",
        owner="w",
        claim_generation=2,
    )
    await graph.compiled.aupdate_state(
        graph._config("old"),
        {"job_id": "old", "completed_nodes": ["chunk"]},
        as_node="chunk",
    )
    with pytest.raises(DeterministicIngestionError, match="original context"):
        await graph.run(claim)
