"""Fixed, checkpointed LangGraph for recoverable document ingestion."""

from __future__ import annotations

import asyncio
from typing import Any, Protocol, cast

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from agentic_rag.ingestion.state import (
    ArtifactPointer,
    IngestionJobClaim,
    IngestionRuntime,
    IngestionState,
)
from agentic_rag.ingestion.assembler import (
    CanonicalAst,
    DocumentEnvelope,
    FragmentAst,
    GlobalAssembler,
)
from agentic_rag.ingestion.chunker import ChunkingPipeline, ParentChunk
from agentic_rag.ingestion.indexer import IndexWriter, StagingContext
from agentic_rag.ingestion.parser import DocumentParser
from agentic_rag.ingestion.publisher import VersionPublisher
from agentic_rag.ingestion.worker import (
    DeterministicIngestionError,
    SqlAlchemyIngestionJobStore,
)
from agentic_rag.persistence.artifacts import ArtifactRef, ArtifactStore
from agentic_rag.safety.content import ContentSafetyScanner
from agentic_rag.safety.uploads import UploadSafetyScanner, UploadSafetyStatus


class IngestionPipeline(Protocol):
    """Process-owned atomic operations used by graph orchestration nodes."""

    async def assert_lease(self, claim: IngestionJobClaim) -> None: ...

    async def load_job(self, claim: IngestionJobClaim) -> IngestionRuntime: ...

    async def upload_safety_gate(
        self, runtime: IngestionRuntime
    ) -> ArtifactPointer: ...

    async def parse_fragments(
        self,
        claim: IngestionJobClaim,
        runtime: IngestionRuntime,
        source: ArtifactPointer,
    ) -> list[ArtifactPointer]: ...

    async def assemble_canonical(
        self,
        claim: IngestionJobClaim,
        runtime: IngestionRuntime,
        fragments: list[ArtifactPointer],
    ) -> ArtifactPointer: ...

    async def content_safety_gate(
        self, runtime: IngestionRuntime, canonical: ArtifactPointer
    ) -> tuple[bool, list[str]]: ...

    async def validate_canonical(self, canonical: ArtifactPointer) -> None: ...

    async def chunk(
        self,
        claim: IngestionJobClaim,
        runtime: IngestionRuntime,
        canonical: ArtifactPointer,
    ) -> ArtifactPointer: ...

    async def embed_and_stage(
        self,
        claim: IngestionJobClaim,
        runtime: IngestionRuntime,
        canonical: ArtifactPointer,
        chunks: ArtifactPointer,
    ) -> str: ...

    async def publish(
        self, claim: IngestionJobClaim, runtime: IngestionRuntime
    ) -> None: ...

    async def finalize(self, claim: IngestionJobClaim) -> None: ...

    async def quarantine(
        self, claim: IngestionJobClaim, reasons: list[str]
    ) -> None: ...


class IngestionGraph:
    """Compile and invoke the mandatory ingestion nodes with stable checkpoints."""

    def __init__(
        self,
        *,
        pipeline: IngestionPipeline,
        checkpointer: BaseCheckpointSaver[str],
    ) -> None:
        self._pipeline = pipeline
        builder = StateGraph(IngestionState)
        builder.add_node("load_job", self._load_job)
        builder.add_node("upload_safety_gate", self._upload_safety_gate)
        builder.add_node("parse_fragments", self._parse_fragments)
        builder.add_node("assemble_canonical", self._assemble_canonical)
        builder.add_node("content_safety_gate", self._content_safety_gate)
        builder.add_node("validate_canonical", self._validate_canonical)
        builder.add_node("chunk", self._chunk)
        builder.add_node("embed_and_stage", self._embed_and_stage)
        builder.add_node("publish", self._publish)
        builder.add_node("finalize", self._finalize)
        builder.add_edge(START, "load_job")
        builder.add_edge("load_job", "upload_safety_gate")
        builder.add_edge("upload_safety_gate", "parse_fragments")
        builder.add_edge("parse_fragments", "assemble_canonical")
        builder.add_edge("assemble_canonical", "content_safety_gate")
        builder.add_conditional_edges(
            "content_safety_gate",
            self._after_content_safety,
            {"continue": "validate_canonical", "terminal": "finalize"},
        )
        builder.add_edge("validate_canonical", "chunk")
        builder.add_edge("chunk", "embed_and_stage")
        builder.add_edge("embed_and_stage", "publish")
        builder.add_edge("publish", "finalize")
        builder.add_edge("finalize", END)
        self.compiled: CompiledStateGraph[
            IngestionState, None, IngestionState, IngestionState
        ] = builder.compile(checkpointer=checkpointer, name="IngestionGraph")

    async def run(self, claim: IngestionJobClaim) -> dict[str, object]:
        config = self._config(claim.job_id)
        snapshot = await self.compiled.aget_state(config)
        if snapshot.next:
            await self.compiled.aupdate_state(
                config,
                {
                    "owner": claim.owner,
                    "claim_generation": claim.claim_generation,
                },
            )
            result = await self.compiled.ainvoke(None, config)
        else:
            initial: IngestionState = {
                "job_id": claim.job_id,
                "user_id": claim.user_id,
                "document_id": claim.document_id,
                "document_version_id": claim.document_version_id,
                "owner": claim.owner,
                "claim_generation": claim.claim_generation,
                "completed_nodes": [],
                "terminal_status": None,
                "quarantine_reasons": [],
            }
            result = await self.compiled.ainvoke(
                initial,
                config,
            )
        return cast(dict[str, object], result)

    async def get_state(self, job_id: str) -> dict[str, Any]:
        snapshot = await self.compiled.aget_state(self._config(job_id))
        return dict(snapshot.values)

    @staticmethod
    def _config(job_id: str) -> RunnableConfig:
        return {"configurable": {"thread_id": f"ingestion:{job_id}"}}

    @staticmethod
    def _claim(state: IngestionState) -> IngestionJobClaim:
        runtime = state.get("runtime", {})
        return IngestionJobClaim(
            job_id=state["job_id"],
            user_id=str(runtime.get("user_id", state["user_id"])),
            document_id=str(runtime.get("document_id", state["document_id"])),
            document_version_id=str(
                runtime.get("document_version_id", state["document_version_id"])
            ),
            owner=state["owner"],
            claim_generation=state["claim_generation"],
        )

    @staticmethod
    def _done(state: IngestionState, node: str) -> list[str]:
        return [*state.get("completed_nodes", []), node]

    async def _load_job(self, state: IngestionState) -> dict[str, Any]:
        claim = self._claim(state)
        await self._pipeline.assert_lease(claim)
        runtime = await self._pipeline.load_job(claim)
        return {
            "runtime": runtime.model_dump(mode="json"),
            "ingestion_key": runtime.ingestion_key,
            "completed_nodes": self._done(state, "load_job"),
        }

    async def _upload_safety_gate(self, state: IngestionState) -> dict[str, Any]:
        await self._pipeline.assert_lease(self._claim(state))
        source = await self._pipeline.upload_safety_gate(self._runtime(state))
        return {
            "source_ref": source.model_dump(mode="json"),
            "completed_nodes": self._done(state, "upload_safety_gate"),
        }

    async def _parse_fragments(self, state: IngestionState) -> dict[str, Any]:
        await self._pipeline.assert_lease(self._claim(state))
        refs = await self._pipeline.parse_fragments(
            self._claim(state),
            self._runtime(state),
            ArtifactPointer.model_validate(state["source_ref"]),
        )
        return {
            "fragment_refs": [ref.model_dump(mode="json") for ref in refs],
            "completed_nodes": self._done(state, "parse_fragments"),
        }

    async def _assemble_canonical(self, state: IngestionState) -> dict[str, Any]:
        await self._pipeline.assert_lease(self._claim(state))
        canonical = await self._pipeline.assemble_canonical(
            self._claim(state),
            self._runtime(state),
            [ArtifactPointer.model_validate(value) for value in state["fragment_refs"]],
        )
        return {
            "canonical_ref": canonical.model_dump(mode="json"),
            "completed_nodes": self._done(state, "assemble_canonical"),
        }

    async def _content_safety_gate(self, state: IngestionState) -> dict[str, Any]:
        claim = self._claim(state)
        await self._pipeline.assert_lease(claim)
        accepted, reasons = await self._pipeline.content_safety_gate(
            self._runtime(state), ArtifactPointer.model_validate(state["canonical_ref"])
        )
        update: dict[str, Any] = {
            "completed_nodes": self._done(state, "content_safety_gate")
        }
        if not accepted:
            await self._pipeline.quarantine(claim, reasons)
            update.update(
                terminal_status="quarantined", quarantine_reasons=list(reasons)
            )
        return update

    @staticmethod
    def _after_content_safety(state: IngestionState) -> str:
        return "terminal" if state.get("terminal_status") else "continue"

    async def _validate_canonical(self, state: IngestionState) -> dict[str, Any]:
        await self._pipeline.assert_lease(self._claim(state))
        await self._pipeline.validate_canonical(
            ArtifactPointer.model_validate(state["canonical_ref"])
        )
        return {"completed_nodes": self._done(state, "validate_canonical")}

    async def _chunk(self, state: IngestionState) -> dict[str, Any]:
        await self._pipeline.assert_lease(self._claim(state))
        chunks = await self._pipeline.chunk(
            self._claim(state),
            self._runtime(state),
            ArtifactPointer.model_validate(state["canonical_ref"]),
        )
        return {
            "chunks_ref": chunks.model_dump(mode="json"),
            "completed_nodes": self._done(state, "chunk"),
        }

    async def _embed_and_stage(self, state: IngestionState) -> dict[str, Any]:
        await self._pipeline.assert_lease(self._claim(state))
        manifest_hash = await self._pipeline.embed_and_stage(
            self._claim(state),
            self._runtime(state),
            ArtifactPointer.model_validate(state["canonical_ref"]),
            ArtifactPointer.model_validate(state["chunks_ref"]),
        )
        return {
            "manifest_hash": manifest_hash,
            "completed_nodes": self._done(state, "embed_and_stage"),
        }

    async def _publish(self, state: IngestionState) -> dict[str, Any]:
        await self._pipeline.assert_lease(self._claim(state))
        await self._pipeline.publish(self._claim(state), self._runtime(state))
        return {"completed_nodes": self._done(state, "publish")}

    async def _finalize(self, state: IngestionState) -> dict[str, Any]:
        if state.get("terminal_status") != "quarantined":
            claim = self._claim(state)
            await self._pipeline.assert_lease(claim)
            await self._pipeline.finalize(claim)
            terminal_status = "completed"
        else:
            terminal_status = "quarantined"
        return {
            "terminal_status": terminal_status,
            "completed_nodes": self._done(state, "finalize"),
        }

    @staticmethod
    def _runtime(state: IngestionState) -> IngestionRuntime:
        return IngestionRuntime.model_validate(state["runtime"])


class DefaultIngestionPipeline:
    """Bind fixed graph nodes to the Phase 2 parser, assembler and index services."""

    def __init__(
        self,
        *,
        jobs: SqlAlchemyIngestionJobStore,
        artifacts: ArtifactStore,
        upload_scanner: UploadSafetyScanner,
        parser: DocumentParser,
        assembler: GlobalAssembler,
        content_scanner: ContentSafetyScanner,
        chunker: ChunkingPipeline,
        index_writer: IndexWriter,
        publisher: VersionPublisher,
    ) -> None:
        self._jobs = jobs
        self._artifacts = artifacts
        self._upload_scanner = upload_scanner
        self._parser = parser
        self._assembler = assembler
        self._content_scanner = content_scanner
        self._chunker = chunker
        self._index_writer = index_writer
        self._publisher = publisher

    async def assert_lease(self, claim: IngestionJobClaim) -> None:
        await self._jobs.assert_lease(claim)

    async def load_job(self, claim: IngestionJobClaim) -> IngestionRuntime:
        return await self._jobs.load_job(claim)

    async def upload_safety_gate(self, runtime: IngestionRuntime) -> ArtifactPointer:
        uri = (
            f"artifact://documents/{runtime.user_id}/{runtime.document_id}/"
            f"{runtime.document_version_id}/source/{runtime.filename}"
        )
        ref = await asyncio.to_thread(self._artifacts.describe, uri)
        if ref.sha256 != runtime.content_hash:
            raise DeterministicIngestionError("source Artifact hash changed")
        content = await asyncio.to_thread(self._artifacts.read_bytes, ref)
        decision = await asyncio.to_thread(
            self._upload_scanner.scan,
            runtime.filename,
            runtime.mime_type,
            content,
        )
        if decision.status is UploadSafetyStatus.REJECTED:
            raise DeterministicIngestionError("source failed upload safety recheck")
        if (
            decision.status is UploadSafetyStatus.QUARANTINED
            and runtime.source_trust != "trusted_curated"
        ):
            raise DeterministicIngestionError("unreviewed source is quarantined")
        return _pointer(ref)

    async def parse_fragments(
        self,
        claim: IngestionJobClaim,
        runtime: IngestionRuntime,
        source: ArtifactPointer,
    ) -> list[ArtifactPointer]:
        envelope = DocumentEnvelope(
            document_id=runtime.document_id,
            document_version_id=runtime.document_version_id,
            user_id=runtime.user_id,
            source_type=runtime.source_type,
            source_uri=source.uri,
            content_hash=runtime.content_hash,
            parser_version=runtime.parser_version,
            pipeline_version=runtime.pipeline_version,
            created_at=runtime.created_at,
        )
        refs: list[ArtifactPointer] = []
        async for fragment in self._parser.parse_batches(
            _artifact(source),
            envelope,
            before_artifact_write=lambda: self._jobs.assert_lease(claim),
        ):
            if fragment.artifact_ref is None:
                raise DeterministicIngestionError("parser did not persist Fragment AST")
            refs.append(_pointer(fragment.artifact_ref))
        if not refs:
            raise DeterministicIngestionError("parser produced no Fragment AST")
        return refs

    async def assemble_canonical(
        self,
        claim: IngestionJobClaim,
        runtime: IngestionRuntime,
        fragments: list[ArtifactPointer],
    ) -> ArtifactPointer:
        values = []
        for pointer in fragments:
            payload = await asyncio.to_thread(
                self._artifacts.read_json, _artifact(pointer)
            )
            values.append(FragmentAst.model_validate(payload))
        canonical = await asyncio.to_thread(
            self._assembler.assemble, values, persist=False
        )
        await self._jobs.assert_lease(claim)
        ref = await asyncio.to_thread(
            self._artifacts.put_json,
            (
                f"documents/{runtime.user_id}/{runtime.document_id}/"
                f"{runtime.document_version_id}/canonical/{runtime.parser_version}/"
                f"{runtime.pipeline_version}/{canonical.schema_version}.json"
            ),
            canonical.model_dump(mode="json"),
        )
        return _pointer(ref)

    async def content_safety_gate(
        self, runtime: IngestionRuntime, canonical: ArtifactPointer
    ) -> tuple[bool, list[str]]:
        ast = await self._read_canonical(canonical)
        decision = self._content_scanner.scan(ast)
        accepted = (
            decision.status is UploadSafetyStatus.ACCEPTED
            or runtime.source_trust == "trusted_curated"
        )
        return accepted, list(decision.reasons)

    async def validate_canonical(self, canonical: ArtifactPointer) -> None:
        if not await asyncio.to_thread(self._artifacts.verify, _artifact(canonical)):
            raise DeterministicIngestionError(
                "Canonical AST failed integrity verification"
            )
        await self._read_canonical(canonical)

    async def chunk(
        self,
        claim: IngestionJobClaim,
        runtime: IngestionRuntime,
        canonical: ArtifactPointer,
    ) -> ArtifactPointer:
        ast = await self._read_canonical(canonical)
        parents = await asyncio.to_thread(self._chunker.build, ast)
        await self._jobs.assert_lease(claim)
        ref = await asyncio.to_thread(
            self._artifacts.put_json,
            (
                f"documents/{runtime.user_id}/{runtime.document_id}/"
                f"{runtime.document_version_id}/chunks/{runtime.pipeline_version}/parents.json"
            ),
            {"parents": [parent.model_dump(mode="json") for parent in parents]},
        )
        return _pointer(ref)

    async def embed_and_stage(
        self,
        claim: IngestionJobClaim,
        runtime: IngestionRuntime,
        canonical: ArtifactPointer,
        chunks: ArtifactPointer,
    ) -> str:
        payload = await asyncio.to_thread(self._artifacts.read_json, _artifact(chunks))
        if not isinstance(payload, dict) or not isinstance(
            payload.get("parents"), list
        ):
            raise DeterministicIngestionError("chunk Artifact has an invalid shape")
        parents = [ParentChunk.model_validate(value) for value in payload["parents"]]
        manifest = await self._index_writer.stage(
            parents,
            context=StagingContext(
                user_id=runtime.user_id,
                document_id=runtime.document_id,
                document_version_id=runtime.document_version_id,
                version_no=runtime.version_no,
                pipeline_version=runtime.pipeline_version,
                embedding_version=runtime.embedding_version,
                index_generation=runtime.index_generation,
            ),
            canonical_ast=_artifact(canonical),
            before_side_effect=lambda: self._jobs.assert_lease(claim),
        )
        return manifest.manifest_hash

    async def publish(
        self, claim: IngestionJobClaim, runtime: IngestionRuntime
    ) -> None:
        await self._publisher.publish(
            runtime.document_version_id,
            before_side_effect=lambda: self._jobs.assert_lease(claim),
        )

    async def finalize(self, claim: IngestionJobClaim) -> None:
        await self._jobs.complete(claim)

    async def quarantine(self, claim: IngestionJobClaim, reasons: list[str]) -> None:
        await self._jobs.quarantine(claim, reasons)

    async def _read_canonical(self, pointer: ArtifactPointer) -> CanonicalAst:
        payload = await asyncio.to_thread(self._artifacts.read_json, _artifact(pointer))
        return CanonicalAst.model_validate(payload)


def _artifact(pointer: ArtifactPointer) -> ArtifactRef:
    return ArtifactRef(**pointer.model_dump())


def _pointer(ref: ArtifactRef) -> ArtifactPointer:
    return ArtifactPointer(uri=ref.uri, sha256=ref.sha256, size_bytes=ref.size_bytes)
