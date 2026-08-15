"""Run one opt-in, production-composed Query API acceptance case.

This command is intentionally separate from the hermetic E2E fixtures.  When
``AGENTIC_RAG_RUN_REAL_QUERY_PROVIDER_E2E=1`` is set it composes the real
QueryGraph/Worker, DeepSeek, Qwen, reranker, Mem0, MySQL, Redis and
Elasticsearch boundaries in an isolated user/index namespace.  It writes the
same summary format consumed by ``verify_acceptance.py`` and fails closed on
missing credentials, provider failures, audit failures or cleanup failures.
"""

# The source-layout bootstrap below intentionally precedes application imports.
# Keep the script runnable directly without an editable install.
# ruff: noqa: E402

from __future__ import annotations

import argparse
import asyncio
import hashlib
import os
import shutil
import sys
import tempfile
from collections.abc import Sequence
from contextlib import suppress
from pathlib import Path
from typing import Any, cast
from uuid import uuid4

import httpx
from sqlalchemy import delete, select

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agentic_rag.api.app import create_app
from agentic_rag.bootstrap import AppContainer, build_container
from agentic_rag.config import Settings
from agentic_rag.domain.models import DocumentVersionStatus, UserScope
from agentic_rag.ingestion.chunker import AstLocator, AstSpan, ChildChunk, ParentChunk
from agentic_rag.ingestion.indexer import EMBEDDING_MODEL, IndexWriter, StagingContext
from agentic_rag.ingestion.publisher import VersionPublisher
from agentic_rag.persistence.elasticsearch import ElasticsearchChildIndexStore
from agentic_rag.persistence.lifecycle import SqlAlchemyPublicationRepository
from agentic_rag.persistence.repositories import (
    SqlAlchemyDocumentRepository,
    agent_events,
    agent_runs,
    documents,
    parent_chunks,
    task_outbox,
)
from agentic_rag.persistence.staging import SqlAlchemyParentStagingStore
from agentic_rag.query.graph import QueryGraphDependencies
from agentic_rag.runtime.query_composition import (
    build_query_dependencies,
    build_query_snapshot,
    close_query_dependencies,
)
from agentic_rag.runtime.query_worker import QueryWorker, build_graph_factory
from agentic_rag.runtime.run_manager import TransactionalRunRepository
from evals.clients import HttpQueryClient
from evals.models import EvaluationCase
from evals.report import write_summary
from evals.run import EvalRunner
from scripts.run_query_worker import TransactionalQueryOutboxAdapter
from scripts.verify_acceptance import verify_acceptance
from scripts.backup_local import run_backup_restore_drill
from scripts.run_recovery_drill import run_recovery_drill
from agentic_rag.persistence.outbox import OutboxDispatcher


class _FixedEmbedding:
    async def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return [[1.0] + [0.0] * 1023 for _ in texts]

    async def embed_query(self, text: str) -> list[float]:
        del text
        return [1.0] + [0.0] * 1023


def _locator() -> AstLocator:
    span = AstSpan(
        canonical_path="#/text_blocks/0",
        block_id="real-provider-block",
        page_from=1,
        page_to=1,
        char_from=0,
        char_to=128,
        parent_char_from=0,
        parent_char_to=128,
    )
    return AstLocator(
        spans=(span,),
        segment_ordinal=0,
        parent_char_from=0,
        parent_char_to=128,
    )


async def _seed_document(
    container: AppContainer,
    settings: Settings,
    scope: UserScope,
) -> str:
    """Seed one real Parent/Child pair through the production indexer ports."""
    factory = container.repositories.session_factory
    async with factory.begin() as session:
        document, version = await SqlAlchemyDocumentRepository().create(
            scope,
            source_type="text",
            filename="real-query-acceptance.txt",
            mime_type="text/plain",
            content_hash=hashlib.sha256(b"real-query-acceptance").hexdigest(),
            parser_version="acceptance-parser-v1",
            pipeline_version="acceptance-pipeline-v1",
            embedding_version=EMBEDDING_MODEL,
            index_generation=settings.index_generation,
            version_status=DocumentVersionStatus.UPLOADED,
            transaction=session,
        )

    content = "The acceptance document requires thirty days notice before cancellation."
    locator = _locator()
    parent_id = f"acceptance-parent-{uuid4().hex}"
    child = ChildChunk(
        id=f"acceptance-child-{uuid4().hex}",
        parent_id=parent_id,
        parent_ordinal=0,
        document_id=document.id,
        document_version_id=version.id,
        user_id=scope.user_id,
        ordinal=0,
        heading_path=("Notice",),
        heading_ast_locators=(),
        content_type="text",
        content=content,
        contextualized_content=content,
        token_count=12,
        page_from=1,
        page_to=1,
        ast_locator=locator,
        content_hash=hashlib.sha256(content.encode()).hexdigest(),
    )
    parent = ParentChunk(
        id=parent_id,
        document_id=document.id,
        document_version_id=version.id,
        user_id=scope.user_id,
        ordinal=0,
        heading_path=("Notice",),
        heading_ast_locators=(),
        content_type="text",
        content=content,
        token_count=12,
        page_from=1,
        page_to=1,
        ast_locator=locator,
        content_hash=hashlib.sha256(content.encode()).hexdigest(),
        children=(child,),
    )
    context = StagingContext(
        user_id=scope.user_id,
        document_id=document.id,
        document_version_id=version.id,
        version_no=version.version_no,
        pipeline_version="acceptance-pipeline-v1",
        embedding_version=EMBEDDING_MODEL,
        index_generation=settings.index_generation,
    )
    canonical = container.artifacts.put_json(
        f"documents/{scope.user_id}/{document.id}/{version.id}/canonical/source.json",
        {"text": content},
    )
    await IndexWriter(
        embedding=_FixedEmbedding(),
        parent_store=SqlAlchemyParentStagingStore(factory),
        child_store=ElasticsearchChildIndexStore(container.elasticsearch),
        artifacts=cast(Any, container.artifacts),
    ).stage((parent,), context=context, canonical_ast=canonical)
    await VersionPublisher(
        repository=SqlAlchemyPublicationRepository(factory),
        parent_store=SqlAlchemyParentStagingStore(factory),
        child_store=ElasticsearchChildIndexStore(container.elasticsearch),
        artifacts=cast(Any, container.artifacts),
    ).publish(version.id)
    return parent_id


async def _cleanup(container: AppContainer, settings: Settings) -> None:
    """Delete only this run's generated MySQL rows and index generation."""
    with suppress(Exception):
        await container.elasticsearch.indices.delete(
            index=f"agenticrag-children-{settings.index_generation}",
            ignore_unavailable=True,
        )
    with suppress(Exception):
        async with container.repositories.session_factory.begin() as session:
            run_ids = select(agent_runs.c.id).where(
                agent_runs.c.user_id == settings.default_user_id
            )
            # Never clear the shared outbox: the acceptance run owns only its
            # query aggregate rows. Ingestion and other tenants must remain
            # untouched when cleanup is retried.
            await session.execute(
                delete(task_outbox).where(
                    task_outbox.c.aggregate_type == "query_run",
                    task_outbox.c.aggregate_id.in_(run_ids),
                )
            )
            await session.execute(
                delete(agent_events).where(agent_events.c.user_id == settings.default_user_id)
            )
            await session.execute(
                delete(agent_runs).where(agent_runs.c.user_id == settings.default_user_id)
            )
            await session.execute(
                delete(parent_chunks).where(parent_chunks.c.user_id == settings.default_user_id)
            )
            await session.execute(
                delete(documents).where(documents.c.user_id == settings.default_user_id)
            )


async def run(output: Path) -> dict[str, object]:
    if os.environ.get("AGENTIC_RAG_RUN_REAL_QUERY_PROVIDER_E2E") != "1":
        raise RuntimeError("set AGENTIC_RAG_RUN_REAL_QUERY_PROVIDER_E2E=1 to permit live provider calls")

    base = Settings()
    root = Path(tempfile.mkdtemp(prefix="agentic-rag-real-query-"))
    suffix = uuid4().hex[:12]
    settings = base.model_copy(
        update={
            "default_user_id": f"real-query-{suffix}",
            "index_generation": f"real-query-{suffix}",
            "artifact_root": root / "artifacts",
            "query_checkpoint_path": root / "query.sqlite",
            "ingestion_checkpoint_path": root / "ingestion.sqlite",
            "mem0_collection": f"agent_memories_{suffix}",
            "mem0_history_db_path": root / "mem0" / "history.db",
            "query_run_timeout_seconds": 180,
        }
    )
    container = build_container(settings)
    dependencies: QueryGraphDependencies | None = None
    checkpoint_context: Any = None
    stop = asyncio.Event()
    worker_task: asyncio.Task[None] | None = None
    app_lifespan: Any = None
    http_client: httpx.AsyncClient | None = None
    try:
        await container.redis.ping()
        await container.elasticsearch.info()
        memory_service = container.memory_service
        if not settings.mem0_enabled or memory_service is None or not bool(
            getattr(memory_service, "available", False)
        ):
            raise RuntimeError("real API acceptance requires an available Mem0 provider")
        parent_id = await _seed_document(
            container, settings, UserScope(user_id=settings.default_user_id)
        )
        dependencies = await build_query_dependencies(
            container,
            settings,
            child_index=f"agenticrag-children-{settings.index_generation}",
        )
        checkpoint_context = container.checkpoints.open_query()
        checkpointer = await checkpoint_context.__aenter__()
        worker = QueryWorker(
            runs=TransactionalRunRepository(container.repositories.session_factory),
            broker=container.broker,
            graph_factory=build_graph_factory(dependencies, checkpointer),
            worker_id=f"real-query-{suffix}",
            trace_recorder=dependencies.trace_recorder,
            event_emitter=dependencies.event_emitter,
            block_ms=50,
            heartbeat_interval_seconds=0.5,
            lease_seconds=10,
            run_timeout_seconds=180,
            outbox_dispatcher=OutboxDispatcher(
                TransactionalQueryOutboxAdapter(container.repositories.session_factory),
                container.broker,
                aggregate_type="query_run",
            ),
            outbox_interval_seconds=0.1,
        )
        worker_task = asyncio.create_task(worker.run_forever(stop_event=stop))

        app = create_app(settings, container=container)
        app_lifespan = app.router.lifespan_context(app)
        await app_lifespan.__aenter__()
        http_client = httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://real-query"
        )
        snapshot = build_query_snapshot(settings)
        case = EvaluationCase.model_validate(
            {
                "case_id": f"real-api-{suffix}",
                "user_id": settings.default_user_id,
                "question": "How many days notice does the acceptance document require before cancellation?",
                "reference_answer": "The acceptance document requires thirty days notice before cancellation.",
                "reference_parent_ids": [parent_id],
                "expected_route": "fast_rag",
                "tags": ["real-provider", "api", "mem0"],
                "runtime_config_snapshot_id": snapshot.snapshot_id,
            }
        )
        client = HttpQueryClient("http://real-query", http_client=http_client, timeout_seconds=180)
        recovery_ok = (await asyncio.to_thread(run_recovery_drill)).gate_passed
        backup_ok = await asyncio.to_thread(run_backup_restore_drill)
        summary = await EvalRunner(
            client,
            output_dir=output,
            evaluation_mode="api",
            client_provenance=HttpQueryClient.provenance,
        ).run([case], recovery_drill_passed=recovery_ok, backup_restore_passed=backup_ok)
        summary["memory_provider"] = {
            "enabled": settings.mem0_enabled,
            "available": bool(getattr(memory_service, "available", False)),
            "degraded": bool(getattr(memory_service, "degraded", True)),
        }
        write_summary(output / "summary.json", summary)
        if verify_acceptance(summary) != 0:
            raise RuntimeError("real API acceptance gates failed")
        return dict(summary)
    finally:
        stop.set()
        if worker_task is not None:
            with suppress(asyncio.CancelledError):
                await worker_task
        if app_lifespan is not None:
            with suppress(Exception):
                await app_lifespan.__aexit__(None, None, None)
        if http_client is not None:
            with suppress(Exception):
                await http_client.aclose()
        if checkpoint_context is not None:
            with suppress(Exception):
                await checkpoint_context.__aexit__(None, None, None)
        if dependencies is not None:
            with suppress(Exception):
                await close_query_dependencies(dependencies)
        await _cleanup(container, settings)
        await container.close()
        shutil.rmtree(root, ignore_errors=True)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        summary = asyncio.run(run(args.output))
    except Exception as error:
        print(f"REAL QUERY ACCEPTANCE FAILED: {type(error).__name__}: {error}", file=sys.stderr)
        return 1
    print(f"REAL QUERY ACCEPTANCE PASSED: {args.output / 'summary.json'}")
    print(f"snapshot={summary.get('runtime_config_snapshot_id', '<runner-summary>')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
