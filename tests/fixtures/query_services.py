"""Disposable local-service fixture for the real Query runtime path.

The fixture deliberately injects deterministic model, embedding and reranker
ports, while MySQL, Redis, Elasticsearch, the API routes, Outbox dispatcher,
SQLite checkpoint and Query Worker remain the production boundaries.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
from collections.abc import AsyncIterator, Sequence
from contextlib import suppress
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast
from urllib.parse import urlparse
from uuid import uuid4

import httpx
import pytest
from alembic import command
from alembic.config import Config
from fastapi import FastAPI
from sqlalchemy import delete, select
from sqlalchemy.engine import make_url
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from agentic_rag.api.app import create_app
from agentic_rag.bootstrap import AppContainer, build_container
from agentic_rag.config import Settings
from agentic_rag.domain.models import DocumentVersionStatus, RunStatus, UserScope
from agentic_rag.ingestion.chunker import AstLocator, AstSpan, ChildChunk, ParentChunk
from agentic_rag.ingestion.indexer import EMBEDDING_MODEL, IndexWriter, StagingContext
from agentic_rag.ingestion.publisher import VersionPublisher
from agentic_rag.memory.models import MemoryContext, MemoryRecord, PublicMessage
from agentic_rag.persistence.elasticsearch import ElasticsearchChildIndexStore
from agentic_rag.persistence.repositories import (
    OutboxRecord,
    SqlAlchemyDocumentRepository,
    SqlAlchemyOutboxRepository,
    SqlAlchemyParentRepository,
    agent_events,
    agent_runs,
    documents,
    parent_chunks,
    task_outbox,
)
from agentic_rag.persistence.staging import SqlAlchemyParentStagingStore
from agentic_rag.query.audit import (
    CitationValidator,
    EvidenceGrader,
    FaithfulnessAuditor,
    ParentRepositoryAuthorizationResolver,
)
from agentic_rag.query.evidence_builder import EvidenceBuilder
from agentic_rag.query.generation import AnswerGenerator
from agentic_rag.query.graph import QueryGraphDependencies
from agentic_rag.query.research_loop import ResearchAgentLoop, ResearchLoopDependencies
from agentic_rag.retrieval.adapters.elasticsearch import (
    ElasticsearchBm25Index,
    ElasticsearchVectorIndex,
)
from agentic_rag.retrieval.graph import RetrievalDependencies, RetrievalService
from agentic_rag.retrieval.parents import ParentFetcher
from agentic_rag.retrieval.reranker import Reranker
from agentic_rag.runtime.model_gateway import ModelGateway
from agentic_rag.runtime.query_composition import build_query_snapshot
from agentic_rag.runtime.query_worker import (
    QUERY_STREAM,
    QueryWorker,
    build_graph_factory,
)
from agentic_rag.observability.logging import AgentEventEmitter
from agentic_rag.observability.tracing import TraceRecorder
from agentic_rag.persistence.lifecycle import SqlAlchemyPublicationRepository
from agentic_rag.persistence.outbox import OutboxDispatcher


class _FixedEmbedding:
    async def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        del texts
        return [[1.0] + [0.0] * 1023]

    async def embed_query(self, text: str) -> list[float]:
        del text
        return [1.0] + [0.0] * 1023


class _FixedCrossEncoder:
    def predict(self, pairs: Sequence[tuple[str, str]]) -> Sequence[float]:
        return [1.0 for _ in pairs]


class _DeterministicResponses:
    async def create(self, **kwargs: object) -> dict[str, object]:
        messages = kwargs.get("input", ())
        system = ""
        user_payload: dict[str, object] = {}
        if isinstance(messages, Sequence) and messages:
            first = messages[0]
            if isinstance(first, dict):
                system = str(first.get("content", ""))
            if len(messages) > 1 and isinstance(messages[1], dict):
                try:
                    parsed = json.loads(str(messages[1].get("content", "{}")))
                except (TypeError, ValueError):
                    parsed = {}
                if isinstance(parsed, dict):
                    user_payload = parsed
        if "Classify the query" in system:
            value: object = {
                "route": "fast_rag",
                "normalized_query": str(user_payload.get("question", "query")),
                "reason_code": "deterministic_e2e",
            }
        elif "Determine whether the available evidence" in system:
            value = {"decision": "sufficient", "gaps": []}
        elif "Generate a concise" in system:
            manifest = user_payload.get("evidence_manifest", {})
            evidence_id = next(iter(manifest), "") if isinstance(manifest, dict) else ""
            value = {
                "segments": [
                    {
                        "kind": "content",
                        "text": "The seeded production document requires thirty days notice.",
                        "evidence_ids": [evidence_id],
                    }
                ]
            }
        elif "Audit whether each factual answer claim" in system:
            value = {"passed": True, "unsupported_claim_ids": [], "reasons": []}
        else:
            value = {"passed": True, "unsupported_claim_ids": [], "reasons": []}
        return {
            "output_text": json.dumps(value, separators=(",", ":")),
            "model": str(kwargs.get("model", "deterministic")),
            "usage": {"input_tokens": 11, "output_tokens": 7},
        }


class _DeterministicClient:
    responses = _DeterministicResponses()


class _Memory:
    async def load_context(self, scope: UserScope, query: str, limit: int = 10) -> MemoryContext:
        del scope, query, limit
        return MemoryContext()

    async def extract_and_store(
        self, scope: UserScope, run_id: str, messages: Sequence[PublicMessage]
    ) -> None:
        del scope, run_id, messages

    async def list(self, scope: UserScope) -> list[MemoryRecord]:
        del scope
        return []

    async def delete(self, scope: UserScope, memory_id: str) -> None:
        del scope, memory_id

    async def reconcile_deletions(self) -> None:
        return None


class _Outbox:
    def __init__(self, factory: async_sessionmaker[AsyncSession]) -> None:
        self._factory = factory

    async def list_pending(
        self, limit: int, *, aggregate_type: str | None = None
    ) -> list[OutboxRecord]:
        async with self._factory() as session:
            return await SqlAlchemyOutboxRepository(session).list_pending(
                limit, aggregate_type=aggregate_type
            )

    async def claim_pending(
        self, limit: int, *, aggregate_type: str | None = None
    ) -> list[OutboxRecord]:
        async with self._factory.begin() as session:
            return await SqlAlchemyOutboxRepository(session).claim_pending(
                limit, aggregate_type=aggregate_type
            )

    async def mark_dispatched(self, outbox_id: str) -> None:
        async with self._factory.begin() as session:
            await SqlAlchemyOutboxRepository(session).mark_dispatched(outbox_id)

    async def schedule_retry(self, outbox_id: str) -> None:
        async with self._factory.begin() as session:
            await SqlAlchemyOutboxRepository(session).schedule_retry(outbox_id)


def _require_local_services(tmp_path: Path) -> Settings:
    if os.getenv("AGENTIC_RAG_RUN_REAL_QUERY_E2E") != "1":
        pytest.skip("set AGENTIC_RAG_RUN_REAL_QUERY_E2E=1 for real Query E2E")
    mysql_dsn = os.getenv("AGENTIC_RAG_TEST_MYSQL_DSN")
    redis_url = os.getenv("AGENTIC_RAG_TEST_REDIS_DSN")
    elasticsearch_url = os.getenv("AGENTIC_RAG_TEST_ELASTICSEARCH_URL")
    if not mysql_dsn or not redis_url or not elasticsearch_url:
        pytest.skip(
            "set AGENTIC_RAG_TEST_MYSQL_DSN, AGENTIC_RAG_TEST_REDIS_DSN and "
            "AGENTIC_RAG_TEST_ELASTICSEARCH_URL for real Query E2E"
        )
    if make_url(mysql_dsn).host not in {"localhost", "127.0.0.1", "::1"}:
        pytest.skip("real Query E2E requires a local MySQL DSN")
    parsed_redis = urlparse(redis_url)
    parsed_es = urlparse(elasticsearch_url)
    if parsed_redis.hostname not in {"localhost", "127.0.0.1", "::1"}:
        pytest.skip("real Query E2E requires a local Redis URL")
    if parsed_es.hostname not in {"localhost", "127.0.0.1", "::1"}:
        pytest.skip("real Query E2E requires a local Elasticsearch URL")
    try:
        base = Settings()  # type: ignore[call-arg]
    except Exception as error:
        pytest.fail(f"configured real Query E2E services require valid .env.local: {error}")
    return base.model_copy(
        update={
            "mysql_dsn": mysql_dsn,
            "redis_url": redis_url,
            "elasticsearch_url": elasticsearch_url,
            "default_user_id": f"query-e2e-{uuid4().hex}",
            "index_generation": f"query-e2e-{uuid4().hex}",
            "artifact_root": tmp_path / "artifacts",
            "query_checkpoint_path": tmp_path / "query.sqlite",
            "ingestion_checkpoint_path": tmp_path / "ingestion.sqlite",
            "query_run_timeout_seconds": 60,
        }
    )


def _locator() -> AstLocator:
    span = AstSpan(
        canonical_path="#/text_blocks/0",
        block_id="block-1",
        page_from=1,
        page_to=1,
        char_from=0,
        char_to=64,
        parent_char_from=0,
        parent_char_to=64,
    )
    return AstLocator(spans=(span,), segment_ordinal=0, parent_char_from=0, parent_char_to=64)


async def _seed_document(container: AppContainer, settings: Settings, scope: UserScope) -> None:
    factory = container.repositories.session_factory
    document_repository = SqlAlchemyDocumentRepository()
    async with factory.begin() as session:
        document, version = await document_repository.create(
            scope,
            source_type="text",
            filename="query-e2e.txt",
            mime_type="text/plain",
            content_hash=hashlib.sha256(b"query-e2e").hexdigest(),
            parser_version="e2e-parser-v1",
            pipeline_version="e2e-pipeline-v1",
            embedding_version=EMBEDDING_MODEL,
            index_generation=settings.index_generation,
            version_status=DocumentVersionStatus.UPLOADED,
            transaction=session,
        )
    parent_id = f"query-e2e-parent-{uuid4().hex}"
    content = "The seeded production document requires thirty days notice."
    locator = _locator()
    child = ChildChunk(
        id=f"query-e2e-child-{uuid4().hex}",
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
        token_count=10,
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
        token_count=10,
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
        pipeline_version="e2e-pipeline-v1",
        embedding_version=EMBEDDING_MODEL,
        index_generation=settings.index_generation,
    )
    canonical = container.artifacts.put_json(
        f"documents/{scope.user_id}/{document.id}/{version.id}/canonical/source.json",
        {"text": content},
    )
    parents = SqlAlchemyParentStagingStore(factory)
    children = ElasticsearchChildIndexStore(container.elasticsearch)
    writer = IndexWriter(
        embedding=_FixedEmbedding(),
        parent_store=parents,
        child_store=children,
        artifacts=cast(Any, container.artifacts),
    )
    await writer.stage((parent,), context=context, canonical_ast=canonical)
    await VersionPublisher(
        repository=SqlAlchemyPublicationRepository(factory),
        parent_store=parents,
        child_store=children,
        artifacts=cast(Any, container.artifacts),
    ).publish(version.id)


def _dependencies(container: AppContainer, settings: Settings) -> QueryGraphDependencies:
    snapshot = build_query_snapshot(settings)
    gateway = ModelGateway(_DeterministicClient(), max_retries=0)
    parents = _SessionParents(container.repositories.session_factory)
    retrieval = RetrievalService(
        RetrievalDependencies(
            embedding=_FixedEmbedding(),
            vector=ElasticsearchVectorIndex(
                container.elasticsearch,
                index_generation=settings.index_generation,
                index=f"agenticrag-children-{settings.index_generation}",
            ),
            lexical=ElasticsearchBm25Index(
                container.elasticsearch,
                index_generation=settings.index_generation,
                index=f"agenticrag-children-{settings.index_generation}",
            ),
            reranker=Reranker(_FixedCrossEncoder(), model_version="e2e-reranker"),
            parent_fetcher=ParentFetcher(parents),
        )
    )
    evidence_builder = EvidenceBuilder()
    return QueryGraphDependencies(
        memory=_Memory(),
        gateway=gateway,
        retrieval=retrieval,
        evidence_builder=evidence_builder,
        evidence_grader=cast(Any, EvidenceGrader(gateway)),
        research_loop=ResearchAgentLoop(
            ResearchLoopDependencies(
                gateway=gateway, retrieval=retrieval, evidence_builder=evidence_builder
            )
        ),
        generator=AnswerGenerator(gateway),
        faithfulness_auditor=FaithfulnessAuditor(gateway),
        citation_validator=CitationValidator(),
        authorization_resolver=ParentRepositoryAuthorizationResolver(parents),
        event_repository=cast(Any, container.event_repository),
        trace_recorder=TraceRecorder(runtime_config_snapshot_id=snapshot.snapshot_id),
        event_emitter=AgentEventEmitter(
            cast(Any, container.event_repository),
            container.artifacts,
            runtime_config_snapshot_id=snapshot.snapshot_id,
        ),
        owned_resources=(),
    )


class _SessionParents(SqlAlchemyParentRepository):
    def __init__(self, factory: async_sessionmaker[AsyncSession]) -> None:
        super().__init__(None)
        self._factory = factory

    async def get_many(self, parent_ids: list[str], scope: UserScope) -> list[Any]:
        async with self._factory() as session:
            return await SqlAlchemyParentRepository(session).get_many(parent_ids, scope)


class RealQueryFixture:
    def __init__(
        self,
        *,
        settings: Settings,
        container: AppContainer,
        app: FastAPI,
        client: httpx.AsyncClient,
        app_lifespan: Any,
    ) -> None:
        self.settings = settings
        self.container = container
        self.app = app
        self.client = client
        self._app_lifespan = app_lifespan
        self._worker: QueryWorker | None = None
        self._worker_task: asyncio.Task[None] | None = None
        self._stop = asyncio.Event()
        self._checkpoint_context: Any = None
        self._dependencies: QueryGraphDependencies | None = None

    async def start_worker(self) -> None:
        self._dependencies = _dependencies(self.container, self.settings)
        self._checkpoint_context = self.container.checkpoints.open_query()
        checkpointer = await self._checkpoint_context.__aenter__()
        self._worker = QueryWorker(
            runs=self.container.run_manager.runs,  # type: ignore[union-attr]
            broker=self.container.broker,
            graph_factory=build_graph_factory(self._dependencies, checkpointer),
            worker_id=f"query-e2e-{uuid4().hex}",
            trace_recorder=self._dependencies.trace_recorder,
            event_emitter=self._dependencies.event_emitter,
            block_ms=50,
            heartbeat_interval_seconds=0.2,
            lease_seconds=2,
            run_timeout_seconds=30,
            outbox_dispatcher=OutboxDispatcher(
                _Outbox(self.container.repositories.session_factory),
                self.container.broker,
                aggregate_type="query_run",
            ),
            outbox_interval_seconds=0.05,
        )
        self._stop = asyncio.Event()
        self._worker_task = asyncio.create_task(self._worker.run_forever(stop_event=self._stop))

    async def stop_worker(self) -> None:
        self._stop.set()
        if self._worker_task is not None:
            with suppress(asyncio.CancelledError):
                await self._worker_task
        self._worker_task = None
        if self._checkpoint_context is not None:
            await self._checkpoint_context.__aexit__(None, None, None)
            self._checkpoint_context = None

    async def restart_worker(self) -> None:
        await self.stop_worker()
        await self.start_worker()

    async def create_query(self, query: str, *, thread_id: str | None = None) -> httpx.Response:
        payload: dict[str, object] = {"query": query, "wait_seconds": 0}
        if thread_id is not None:
            payload["thread_id"] = thread_id
        return await self.client.post("/v1/query-runs", json=payload)

    async def wait_for_terminal(
        self, run_id: str, *, restart_worker: bool = False, timeout: float = 30.0
    ) -> dict[str, object]:
        if restart_worker:
            await self.restart_worker()
        deadline = asyncio.get_running_loop().time() + timeout
        while asyncio.get_running_loop().time() < deadline:
            response = await self.client.get(f"/v1/query-runs/{run_id}")
            payload = response.json()
            if payload.get("status") in {status.value for status in (RunStatus.COMPLETED, RunStatus.FAILED, RunStatus.CANCELLED)}:
                return payload
            await asyncio.sleep(0.1)
        raise AssertionError(f"query run {run_id} did not reach a terminal state")

    async def inject_duplicate_delivery(self, run_id: str) -> None:
        await self.container.broker.publish(
            QUERY_STREAM, run_id, datetime.now(UTC), dedupe_key=None
        )

    async def events_contain(self, run_id: str, event_type: str) -> bool:
        events = await self._all_events(run_id)
        return any(event.event_type == event_type for event in events)

    async def sse_body(self, run_id: str) -> str:
        async with self.client.stream(
            "GET", f"/v1/query-runs/{run_id}/events"
        ) as response:
            return "\n".join([line async for line in response.aiter_lines()])

    async def _all_events(self, run_id: str) -> list[Any]:
        scope = UserScope(user_id=self.settings.default_user_id)
        return await cast(Any, self.container.event_repository).list_after(run_id, scope, 0, 200)

    async def terminal_event_count(self, run_id: str) -> int:
        events = await self._all_events(run_id)
        return sum(event.event_type in {"RUN_COMPLETED", "RUN_FAILED", "RUN_CANCELLED"} for event in events)

    async def close(self) -> None:
        await self.stop_worker()
        await self.client.aclose()
        await self._app_lifespan.__aexit__(None, None, None)
        index_name = f"agenticrag-children-{self.settings.index_generation}"
        with suppress(Exception):
            await self.container.elasticsearch.indices.delete(index=index_name, ignore_unavailable=True)
        with suppress(Exception):
            async with self.container.repositories.session_factory.begin() as session:
                await session.execute(delete(task_outbox))
                await session.execute(delete(agent_events).where(agent_events.c.user_id == self.settings.default_user_id))
                await session.execute(delete(agent_runs).where(agent_runs.c.user_id == self.settings.default_user_id))
                await session.execute(delete(parent_chunks).where(parent_chunks.c.user_id == self.settings.default_user_id))
                await session.execute(delete(documents).where(documents.c.user_id == self.settings.default_user_id))
        await self.container.close()


@pytest.fixture
async def real_query_fixture(tmp_path: Path) -> AsyncIterator[RealQueryFixture]:
    settings = _require_local_services(tmp_path)
    migration = Config(str(Path("alembic.ini").resolve()))
    migration.set_main_option("sqlalchemy.url", settings.mysql_dsn)
    await asyncio.to_thread(command.upgrade, migration, "head")
    container = build_container(settings)
    await container.redis.ping()
    await container.elasticsearch.info()
    async with container.mysql_engine.connect() as connection:
        await connection.execute(select(1))
    scope = UserScope(user_id=settings.default_user_id)
    await _seed_document(container, settings, scope)
    app = create_app(settings)
    app_lifespan = app.router.lifespan_context(app)
    await app_lifespan.__aenter__()
    client = httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    )
    fixture = RealQueryFixture(
        settings=settings,
        container=container,
        app=app,
        client=client,
        app_lifespan=app_lifespan,
    )
    await fixture.start_worker()
    try:
        yield fixture
    finally:
        await fixture.close()
