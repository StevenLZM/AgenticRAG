"""Process harness for explicitly isolated supervisor acceptance environments.

The manifest contains resource names and a settings-file path, never credentials.
All stages use the production executor; auditing does not replace model results.
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
from pathlib import Path
import signal
import sys
import time
from typing import Any, cast

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sqlalchemy.engine import make_url  # noqa: E402
from agentic_rag.config import Settings  # noqa: E402
from agentic_rag.runtime.model_gateway import ModelGateway  # noqa: E402


def settings_for(manifest: Path) -> Settings:
    data = json.loads(manifest.read_text())
    if not data["database"].startswith("agentic_rag_supervisor_test_") or not data["prefix"].startswith("sup-live-"):
        raise ValueError("acceptance requires explicitly isolated resource names")
    original = Settings(**{"_env_file": data["settings_file"]})
    url = make_url(original.mysql_dsn)
    if url.host not in ("localhost", "127.0.0.1", "::1"):
        raise ValueError("acceptance MySQL must be local")
    return original.model_copy(update={
        "mysql_dsn": url.set(database=data["database"]).render_as_string(hide_password=False),
        "redis_url": f"redis://127.0.0.1:{data['redis_port']}/0",
        "supervisor_enabled": True, "mem0_enabled": False, "tool_runtime_enabled": False,
        "amap_mcp_enabled": False, "mcp_servers": (), "default_user_id": data["prefix"],
        "index_generation": data["prefix"], "lexical_analysis": "standard",
        "artifact_root": manifest.parent / "artifacts",
        "query_checkpoint_path": manifest.parent / "legacy-checkpoint.sqlite",
        "ingestion_checkpoint_path": manifest.parent / "ingestion-checkpoint.sqlite",
        "max_concurrent_reranks": 2, "max_concurrent_llm_calls": 8,
        "supervisor_tenant_model_limit": 6,
    })


def audit(path: Path, event: str, **fields: Any) -> None:
    with path.open("a") as output:
        output.write(json.dumps({"time": time.time(), "pid": os.getpid(), "event": event, **fields}, ensure_ascii=False) + "\n")


class AuditedGateway:
    def __init__(self, gateway, log: Path):
        self.gateway, self.log = gateway, log
        create = gateway._client.chat.completions.create

        async def audited_create(**kwargs):
            audit(log, "provider.request", requested_model=kwargs.get("model"),
                  max_output_tokens=kwargs.get("max_tokens"), streaming=bool(kwargs.get("stream")))
            response = await create(**kwargs)
            if not kwargs.get("stream"):
                audit(log, "provider.response", actual_model=response.model,
                      usage=response.usage.model_dump() if response.usage else None,
                      choices=[{"finish_reason": c.finish_reason, "content": c.message.content,
                                "reasoning_length": len(getattr(c.message, "reasoning_content", "") or "")}
                               for c in response.choices])
            return response

        gateway._client.chat.completions.create = audited_create

    async def complete_structured(self, call, schema):
        started = time.time()
        audit(self.log, "model.started", schema=schema.__name__, requested_model=call.requested_model,
              max_output_tokens=call.max_output_tokens, max_provider_attempts=call.max_provider_attempts)
        try:
            result = await self.gateway.complete_structured(call, schema)
        except BaseException as error:
            audit(self.log, "model.failed", schema=schema.__name__, error=type(error).__name__,
                  validation_cause=str(error.__cause__)[:1000] if isinstance(error, ValueError) else None)
            raise
        audit(self.log, "model.completed", schema=schema.__name__, actual_model=result.actual_model,
              input_tokens=result.input_tokens, output_tokens=result.output_tokens, attempts=result.attempts,
              latency_seconds=time.time() - started, value=result.value.model_dump(mode="json"))
        return result

    async def stream(self, call):
        audit(self.log, "provider.stream.started", requested_model=call.requested_model, max_output_tokens=call.max_output_tokens)
        first = True
        async for event in self.gateway.stream(call):
            if first and event.kind == "delta":
                audit(self.log, "provider.first_delta", actual_model=event.actual_model)
                first = False
            if event.kind == "completed":
                audit(self.log, "provider.stream.completed", actual_model=event.actual_model,
                      input_tokens=event.input_tokens, output_tokens=event.output_tokens)
            yield event


async def seed(manifest: Path, *, supplement: bool = False) -> list[dict[str, str]]:
    from openai import AsyncOpenAI
    from agentic_rag.bootstrap import build_container
    from agentic_rag.domain.models import DocumentVersionStatus, UserScope
    from agentic_rag.ingestion.chunker import AstLocator, AstSpan, ChildChunk, ParentChunk
    from agentic_rag.ingestion.indexer import IndexWriter, StagingContext
    from agentic_rag.ingestion.publisher import VersionPublisher
    from agentic_rag.persistence.elasticsearch import ElasticsearchChildIndexStore
    from agentic_rag.persistence.staging import SqlAlchemyParentStagingStore
    from agentic_rag.persistence.repositories import SqlAlchemyDocumentRepository
    from agentic_rag.persistence.lifecycle import SqlAlchemyPublicationRepository
    from agentic_rag.persistence.artifacts import LocalArtifactStore
    from agentic_rag.runtime.ids import new_id
    from agentic_rag.runtime.query_composition import _OpenAIEmbedding

    settings = settings_for(manifest)
    if settings.qwen_api_key is None:
        raise ValueError("acceptance requires configured Qwen credentials")
    container = build_container(settings)
    assert isinstance(container.artifacts, LocalArtifactStore)
    client = AsyncOpenAI(api_key=settings.qwen_api_key.get_secret_value(), base_url=settings.qwen_embedding_base_url, max_retries=0, timeout=30)
    embedding = _OpenAIEmbedding(client, settings.embedding_model, settings.embedding_dimensions)
    docs = [("远海制造2024年度补充披露", "远海制造2024年度补充披露：2024年合并营业收入为200亿元人民币，2025年合并营业收入为260亿元人民币。两年均为合并口径，同一业务范围。")]
    if not supplement:
        docs = [("凌星科技2025年报", "凌星科技2025年年度报告：2024年合并营业收入为100亿元人民币，2025年合并营业收入为120亿元人民币。两年均为合并口径，同一业务范围。"),
                ("远海制造2025年报", "远海制造2025年年度报告：2025年合并营业收入为260亿元人民币。该段落没有披露2024年的可比数据，2024年合并口径数据另见年度补充披露。")]
    output = []
    try:
        scope = UserScope(user_id=settings.default_user_id)
        for title, content in docs:
            async with container.repositories.session_factory.begin() as db:
                document, version = await SqlAlchemyDocumentRepository().create(scope, source_type="text", filename=title + ".txt", mime_type="text/plain",
                    content_hash=hashlib.sha256(content.encode()).hexdigest(), parser_version="acceptance-seed-v1", pipeline_version="acceptance-seed-v1",
                    embedding_version=settings.embedding_model, index_generation=settings.index_generation,
                    version_status=DocumentVersionStatus.UPLOADED, transaction=db)
            parent_id = new_id()
            span = AstSpan(canonical_path="#/text_blocks/0", block_id="block-1", page_from=1, page_to=1,
                           char_from=0, char_to=len(content), parent_char_from=0, parent_char_to=len(content))
            locator = AstLocator(spans=(span,), segment_ordinal=0, parent_char_from=0, parent_char_to=len(content))
            common: dict[str, Any] = dict(document_id=document.id, document_version_id=version.id, user_id=scope.user_id, ordinal=0,
                          heading_path=(title,), heading_ast_locators=(), content_type="text", content=content, token_count=len(content),
                          page_from=1, page_to=1, ast_locator=locator, content_hash=hashlib.sha256(content.encode()).hexdigest())
            child = ChildChunk(id=new_id(), parent_id=parent_id, parent_ordinal=0, contextualized_content=content, **common)
            parent = ParentChunk(id=parent_id, children=(child,), **common)
            context = StagingContext(user_id=scope.user_id, document_id=document.id, document_version_id=version.id, version_no=version.version_no,
                pipeline_version="acceptance-seed-v1", embedding_version=settings.embedding_model, index_generation=settings.index_generation)
            canonical = container.artifacts.put_json(f"documents/{scope.user_id}/{document.id}/{version.id}/canonical/acceptance.json", {"text_blocks": [{"block_id": "block-1", "text": content}]})
            parents = SqlAlchemyParentStagingStore(container.repositories.session_factory, embedding_model=settings.embedding_model)
            children = ElasticsearchChildIndexStore(container.elasticsearch, embedding_model=settings.embedding_model)
            await IndexWriter(embedding=embedding, embedding_model=settings.embedding_model, parent_store=parents, child_store=children, artifacts=container.artifacts).stage((parent,), context=context, canonical_ast=canonical)
            await VersionPublisher(repository=SqlAlchemyPublicationRepository(container.repositories.session_factory), parent_store=parents,
                                   child_store=children, artifacts=container.artifacts, embedding_model=settings.embedding_model).publish(version.id)
            output.append({"document_id": document.id, "version_id": version.id, "title": title, "content": content})
        await container.elasticsearch.indices.refresh(index=f"agenticrag-children-{settings.index_generation}")
        name = "supplement.json" if supplement else "corpus.json"
        (manifest.parent / name).write_text(json.dumps(output, ensure_ascii=False, indent=2))
        return output
    finally:
        await client.close()
        await container.close()


async def run_process(manifest: Path, role: str) -> None:
    from agentic_rag.bootstrap import build_container
    from agentic_rag.orchestration.composition import build_executor
    from agentic_rag.orchestration.service import CoordinationService, check_schema
    from agentic_rag.orchestration.store import CoordinationStore
    from agentic_rag.orchestration.worker import AgentWorker
    settings = settings_for(manifest)
    container = build_container(settings)
    clients: list[Any] = []
    log = manifest.parent / f"{role}-{os.getpid()}.events.jsonl"
    stop = asyncio.Event()
    for sig in (signal.SIGTERM, signal.SIGINT):
        asyncio.get_running_loop().add_signal_handler(sig, stop.set)
    try:
        await check_schema(container.mysql_engine)
        if role == "api":
            import uvicorn
            from agentic_rag.api.app import create_app
            port = json.loads(manifest.read_text())["api_port"]
            server = uvicorn.Server(uvicorn.Config(create_app(settings, container=container), host="127.0.0.1", port=port, access_log=False, log_level="warning"))
            audit(log, "ready", role=role)
            await server.serve()
        elif role == "coordinator":
            service = CoordinationService(CoordinationStore(container.repositories.session_factory, max_active_runs=settings.max_concurrent_query_runs), container.broker, worker_id=f"acceptance:{os.getpid()}")
            audit(log, "ready", role=role)
            await service.run_forever(stop)
        else:
            executor, clients = await build_executor(container, settings, role, retrieval_index=f"agenticrag-children-{settings.index_generation}")
            # This test-only proxy forwards the two gateway methods used by
            # StageExecutor, retaining the production gateway for both calls.
            executor.gateway = cast(ModelGateway, AuditedGateway(executor.gateway, log))
            async def execute(claim):
                audit(log, "stage.started", work_id=claim.work_id, run_id=claim.run_id, stage=claim.stage, binding=claim.payload)
                try:
                    result = await executor(claim)
                except BaseException as error:
                    audit(log, "stage.failed", stage=claim.stage, error=type(error).__name__)
                    raise
                audit(log, "stage.executed", work_id=claim.work_id, run_id=claim.run_id, stage=claim.stage)
                return result
            worker = AgentWorker(executor.store, execute, worker_id=f"acceptance:{os.getpid()}", role=role, broker=container.broker)
            audit(log, "ready", role=role)
            await worker.run_forever(stop)
    finally:
        for client in clients:
            await client.close()
        await container.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--role", required=True, choices=["api", "coordinator", "supervisor", "retrieval", "analysis", "evidence", "answer", "all", "seed", "supplement"])
    args = parser.parse_args()
    asyncio.run(seed(args.manifest, supplement=args.role == "supplement") if args.role in ("seed", "supplement") else run_process(args.manifest, args.role))
