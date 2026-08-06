from __future__ import annotations

import json
import hashlib
from pathlib import Path

import aiosqlite
import pytest
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from agentic_rag.ingestion.graph import IngestionGraph
from agentic_rag.ingestion.state import (
    ArtifactPointer,
    IngestionJobClaim,
    IngestionRuntime,
)


pytestmark = pytest.mark.e2e


class RecoverablePipeline:
    def __init__(self) -> None:
        self.fail_stage_once = True
        self.staged_parent_ids: set[str] = set()
        self.staged_child_ids: set[str] = set()
        self.calls: list[str] = []
        self.completed = False
        self.safe = True
        self.quarantined = False

    async def assert_lease(self, claim: IngestionJobClaim) -> None:
        assert claim.user_id == "user-1"
        assert claim.document_id == "document-1"
        assert claim.document_version_id == "version-1"
        self.calls.append(f"lease:{claim.claim_generation}")

    async def load_job(self, claim: IngestionJobClaim) -> IngestionRuntime:
        return IngestionRuntime(
            job_id=claim.job_id,
            user_id=claim.user_id,
            document_id=claim.document_id,
            document_version_id=claim.document_version_id,
            version_no=1,
            filename="sample.pdf",
            mime_type="application/pdf",
            source_type="pdf",
            content_hash="a" * 64,
            parser_version="docling-v1",
            pipeline_version="ingestion-v1",
            embedding_version="text-embedding-v3",
            index_generation="index-v2",
        )

    async def upload_safety_gate(self, runtime: IngestionRuntime) -> ArtifactPointer:
        self.calls.append("upload_safety_gate")
        return _ref("source")

    async def parse_fragments(
        self, runtime: IngestionRuntime, source: ArtifactPointer
    ) -> list[ArtifactPointer]:
        self.calls.append("parse_fragments")
        return [_ref("fragment")]

    async def assemble_canonical(
        self, runtime: IngestionRuntime, fragments: list[ArtifactPointer]
    ) -> ArtifactPointer:
        self.calls.append("assemble_canonical")
        return _ref("canonical")

    async def content_safety_gate(
        self, runtime: IngestionRuntime, canonical: ArtifactPointer
    ) -> tuple[bool, list[str]]:
        self.calls.append("content_safety_gate")
        return self.safe, ([] if self.safe else ["instruction_like_content"])

    async def validate_canonical(self, canonical: ArtifactPointer) -> None:
        self.calls.append("validate_canonical")

    async def chunk(
        self, runtime: IngestionRuntime, canonical: ArtifactPointer
    ) -> ArtifactPointer:
        self.calls.append("chunk")
        return _ref("chunks")

    async def embed_and_stage(
        self,
        runtime: IngestionRuntime,
        canonical: ArtifactPointer,
        chunks: ArtifactPointer,
    ) -> str:
        self.calls.append("embed_and_stage")
        self.staged_parent_ids.add("parent-deterministic")
        self.staged_child_ids.add("child-deterministic")
        if self.fail_stage_once:
            self.fail_stage_once = False
            raise RuntimeError("crash after stage_index")
        return "b" * 64

    async def publish(self, runtime: IngestionRuntime) -> None:
        self.calls.append("publish")

    async def finalize(self, claim: IngestionJobClaim) -> None:
        self.calls.append("finalize")
        self.completed = True

    async def quarantine(self, claim: IngestionJobClaim, reasons: list[str]) -> None:
        self.quarantined = True


def _ref(name: str) -> ArtifactPointer:
    return ArtifactPointer(
        uri=f"artifact://{name}.json",
        sha256=hashlib.sha256(name.encode()).hexdigest(),
        size_bytes=10,
    )


def _claim(generation: int) -> IngestionJobClaim:
    return IngestionJobClaim(
        job_id="job-1",
        user_id="user-1",
        document_id="document-1",
        document_version_id="version-1",
        owner=f"worker-{generation}",
        claim_generation=generation,
    )


@pytest.mark.asyncio
async def test_graph_resumes_failed_stage_without_duplicate_ids(
    tmp_path: Path,
) -> None:
    pipeline = RecoverablePipeline()
    async with aiosqlite.connect(tmp_path / "ingestion.sqlite") as connection:
        checkpointer = AsyncSqliteSaver(connection)
        await checkpointer.setup()
        graph = IngestionGraph(pipeline=pipeline, checkpointer=checkpointer)

        with pytest.raises(RuntimeError, match="crash after stage_index"):
            await graph.run(_claim(0))
        state = await graph.get_state("job-1")
        json.dumps(state)

        result = await graph.run(_claim(1))

    assert result["terminal_status"] == "completed"
    assert pipeline.completed is True
    assert pipeline.staged_parent_ids == {"parent-deterministic"}
    assert pipeline.staged_child_ids == {"child-deterministic"}
    assert pipeline.calls.count("parse_fragments") == 1
    assert pipeline.calls.count("embed_and_stage") == 2
    assert "lease:1" in pipeline.calls


@pytest.mark.asyncio
async def test_graph_uses_fixed_node_order_and_thread_identity(tmp_path: Path) -> None:
    pipeline = RecoverablePipeline()
    pipeline.fail_stage_once = False
    async with aiosqlite.connect(tmp_path / "ingestion.sqlite") as connection:
        checkpointer = AsyncSqliteSaver(connection)
        await checkpointer.setup()
        graph = IngestionGraph(pipeline=pipeline, checkpointer=checkpointer)
        await graph.run(_claim(0))
        snapshot = await graph.compiled.aget_state(
            {"configurable": {"thread_id": "ingestion:job-1"}}
        )

    assert snapshot.values["completed_nodes"] == [
        "load_job",
        "upload_safety_gate",
        "parse_fragments",
        "assemble_canonical",
        "content_safety_gate",
        "validate_canonical",
        "chunk",
        "embed_and_stage",
        "publish",
        "finalize",
    ]


@pytest.mark.asyncio
async def test_content_quarantine_is_terminal_and_never_reaches_chunking(
    tmp_path: Path,
) -> None:
    pipeline = RecoverablePipeline()
    pipeline.safe = False
    async with aiosqlite.connect(tmp_path / "ingestion.sqlite") as connection:
        checkpointer = AsyncSqliteSaver(connection)
        await checkpointer.setup()
        result = await IngestionGraph(pipeline=pipeline, checkpointer=checkpointer).run(
            _claim(0)
        )

    assert result["terminal_status"] == "quarantined"
    assert pipeline.quarantined is True
    assert "chunk" not in pipeline.calls
    assert "embed_and_stage" not in pipeline.calls
    assert "publish" not in pipeline.calls
