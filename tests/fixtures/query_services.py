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
from collections.abc import AsyncIterator, Awaitable, Mapping, Sequence
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
from agentic_rag.observability.logging import emit_degradation, event_emission_scope
from agentic_rag.persistence.elasticsearch import ElasticsearchChildIndexStore
from agentic_rag.persistence.repositories import (
    SqlAlchemyDocumentRepository,
    SqlAlchemyParentRepository,
    agent_events,
    agent_runs,
    documents,
    memory_tombstones,
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
from agentic_rag.runtime.models import RuntimeConfigSnapshot
from agentic_rag.runtime.query_composition import (
    build_query_dependencies,
    build_query_snapshot,
    close_query_dependencies,
)
from agentic_rag.runtime.query_worker import (
    QUERY_STREAM,
    QueryWorker,
    build_graph_factory,
)
from agentic_rag.runtime.run_manager import RunManager, TransactionalRunRepository
from agentic_rag.testing.isolated_query_broker import IsolatedQueryBroker
from agentic_rag.testing.real_provider_config import (
    explicit_provider_configuration_issue,
    provider_configuration_issue,
    provider_environment_from_process,
)
from agentic_rag.observability.logging import AgentEventEmitter
from agentic_rag.observability.tracing import TraceRecorder
from agentic_rag.persistence.lifecycle import SqlAlchemyPublicationRepository
from agentic_rag.persistence.outbox import OutboxDispatcher
from scripts.real_acceptance_evidence import (
    create_isolated_mysql_database,
    drop_isolated_mysql_database,
    isolated_mysql_dsn,
    require_service_backup_admin_dsn,
    service_backup_configuration_issue,
)
from scripts.run_query_worker import TransactionalQueryOutboxAdapter


class FixtureTeardownError(RuntimeError):
    """A disposable real-service fixture could not release all owned state."""


async def _record_fixture_teardown_error(
    failures: list[tuple[str, BaseException]], boundary: str, operation: Awaitable[object]
) -> None:
    try:
        await operation
    except BaseException as error:
        failures.append((boundary, error))


def _raise_fixture_teardown_failures(
    failures: Sequence[tuple[str, BaseException]],
) -> None:
    if not failures:
        return
    for _, error in failures:
        if not isinstance(error, Exception):
            raise error
    details = ", ".join(f"{boundary}: {type(error).__name__}" for boundary, error in failures)
    raise FixtureTeardownError(f"real Query fixture teardown failed ({details})") from failures[0][1]


async def _strict_container_close(container: AppContainer) -> None:
    await container.close(raise_on_error=True)


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


def _contract_only_fixture_summary(snapshot_id: str) -> dict[str, object]:
    """Describe seeded/deterministic fixtures without advertising quality evidence."""
    return {
        "fixture_kind": "contract_only",
        "evaluation_mode": "contract",
        "client_provenance": "contract_fixture",
        "quality_measurement": False,
        "runtime_config_snapshot_id": snapshot_id,
        "real_query_count": 0,
        "requested_cases": 0,
        "completed_cases": 0,
    }


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
        pytest.fail("configured real Query E2E requires a local MySQL DSN")
    parsed_redis = urlparse(redis_url)
    parsed_es = urlparse(elasticsearch_url)
    if parsed_redis.hostname not in {"localhost", "127.0.0.1", "::1"}:
        pytest.fail("configured real Query E2E requires a local Redis URL")
    if parsed_es.hostname not in {"localhost", "127.0.0.1", "::1"}:
        pytest.fail("configured real Query E2E requires a local Elasticsearch URL")
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


def _require_real_provider_services(tmp_path: Path) -> Settings:
    """Build one isolated settings object after explicit live-service opt-in."""
    if os.getenv("AGENTIC_RAG_RUN_REAL_QUERY_PROVIDER_E2E") != "1":
        pytest.skip(
            "set AGENTIC_RAG_RUN_REAL_QUERY_PROVIDER_E2E=1 for real Query provider E2E"
        )
    mysql_dsn = os.getenv("AGENTIC_RAG_TEST_MYSQL_DSN", "").strip()
    redis_url = os.getenv("AGENTIC_RAG_TEST_REDIS_DSN", "").strip()
    elasticsearch_url = os.getenv("AGENTIC_RAG_TEST_ELASTICSEARCH_URL", "").strip()
    missing = [
        name
        for name, value in (
            ("AGENTIC_RAG_TEST_MYSQL_DSN", mysql_dsn),
            ("AGENTIC_RAG_TEST_REDIS_DSN", redis_url),
            ("AGENTIC_RAG_TEST_ELASTICSEARCH_URL", elasticsearch_url),
        )
        if not value
    ]
    if missing:
        pytest.skip("missing explicit real Query provider E2E settings: " + ", ".join(missing))
    if make_url(mysql_dsn).host not in {"localhost", "127.0.0.1", "::1"}:
        pytest.fail("configured real Query provider E2E requires a local MySQL DSN")
    parsed_redis = urlparse(redis_url)
    parsed_es = urlparse(elasticsearch_url)
    if parsed_redis.hostname not in {"localhost", "127.0.0.1", "::1"}:
        pytest.fail("configured real Query provider E2E requires a local Redis URL")
    if parsed_es.hostname not in {"localhost", "127.0.0.1", "::1"}:
        pytest.fail("configured real Query provider E2E requires a local Elasticsearch URL")
    try:
        base = Settings(  # type: ignore[call-arg]
            mysql_dsn=mysql_dsn,
            redis_url=redis_url,
            elasticsearch_url=elasticsearch_url,
        )
    except Exception as error:
        explicit_issue = explicit_provider_configuration_issue(
            provider_environment_from_process()
        )
        if explicit_issue is not None:
            _, fields = explicit_issue
            pytest.fail(
                "invalid DeepSeek/Qwen/Mem0 configuration for real Query provider E2E: "
                + ", ".join(fields)
            )
        if _missing_provider_settings(error):
            pytest.skip(
                "configured DeepSeek/Qwen/Mem0 variables are required for real Query "
                "provider E2E"
            )
        pytest.fail(
            "configured real Query provider E2E settings are invalid: "
            f"{type(error).__name__}"
        )
    issue = provider_configuration_issue(base)
    if issue is not None:
        kind, fields = issue
        message = (
            f"{kind} DeepSeek/Qwen/Mem0 configuration for real Query provider E2E: "
            + ", ".join(fields)
        )
        if kind == "missing":
            pytest.skip(message)
        pytest.fail(message)
    backup_issue = service_backup_configuration_issue()
    if backup_issue is not None:
        if (
            "set AGENTIC_RAG_RUN_REAL_BACKUP_RESTORE" in backup_issue
            or "missing AGENTIC_RAG_TEST_MYSQL_ADMIN_DSN" in backup_issue
        ):
            pytest.skip("real Query provider E2E backup/restore unavailable: " + backup_issue)
        pytest.fail("invalid real Query provider E2E backup/restore configuration: " + backup_issue)

    suffix = uuid4().hex
    root = tmp_path / f"real-query-{suffix}"
    return base.model_copy(
        update={
            "mysql_dsn": mysql_dsn,
            "redis_url": redis_url,
            "elasticsearch_url": elasticsearch_url,
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


def _missing_provider_settings(error: Exception) -> bool:
    """Skip absent credentials, but fail malformed configured provider settings."""
    errors = getattr(error, "errors", None)
    details = errors() if callable(errors) else ()
    provider_fields = {
        "deepseek_base_url",
        "deepseek_api_key",
        "qwen_embedding_base_url",
        "qwen_api_key",
    }
    return bool(details) and all(
        isinstance(detail, Mapping)
        and detail.get("type") == "missing"
        and isinstance(detail.get("loc"), tuple)
        and detail["loc"][-1] in provider_fields
        for detail in details
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


async def _seed_document(container: AppContainer, settings: Settings, scope: UserScope) -> str:
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
    return parent_id


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


async def _cleanup_local_fixture_mysql(
    container: AppContainer,
    settings: Settings,
    broker: IsolatedQueryBroker,
) -> None:
    """Delete only rows owned by this fixture user and private stream."""
    async with container.repositories.session_factory.begin() as session:
        run_ids = list(
            (
                await session.execute(
                    select(agent_runs.c.id).where(
                        agent_runs.c.user_id == settings.default_user_id
                    )
                )
            ).scalars()
        )
        if run_ids:
            await session.execute(
                delete(task_outbox).where(
                    task_outbox.c.aggregate_type == "query_run",
                    task_outbox.c.aggregate_id.in_(run_ids),
                    task_outbox.c.stream_name == broker.query_stream,
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


async def _cleanup_local_fixture_redis(
    container: AppContainer, broker: IsolatedQueryBroker
) -> None:
    await container.redis.delete(*broker.cleanup_keys)


class RealQueryFixture:
    def __init__(
        self,
        *,
        settings: Settings,
        container: AppContainer,
        app: FastAPI,
        client: httpx.AsyncClient,
        app_lifespan: Any,
        broker: IsolatedQueryBroker,
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
        self._isolated_broker = broker

    async def start_worker(self) -> None:
        self._dependencies = _dependencies(self.container, self.settings)
        self._checkpoint_context = self.container.checkpoints.open_query()
        checkpointer = await self._checkpoint_context.__aenter__()
        self._worker = QueryWorker(
            runs=TransactionalRunRepository(self.container.repositories.session_factory),
            broker=self._isolated_broker,
            graph_factory=build_graph_factory(self._dependencies, checkpointer),
            worker_id=f"query-e2e-{uuid4().hex}",
            trace_recorder=self._dependencies.trace_recorder,
            event_emitter=self._dependencies.event_emitter,
            block_ms=50,
            heartbeat_interval_seconds=0.2,
            lease_seconds=2,
            run_timeout_seconds=30,
            outbox_dispatcher=OutboxDispatcher(
                TransactionalQueryOutboxAdapter(
                    self.container.repositories.session_factory,
                    user_id=self.settings.default_user_id,
                    stream_name=self._isolated_broker.query_stream,
                ),
                self._isolated_broker,
                aggregate_type="query_run",
            ),
            outbox_interval_seconds=0.05,
        )
        self._stop = asyncio.Event()
        self._worker_task = asyncio.create_task(self._worker.run_forever(stop_event=self._stop))

    async def stop_worker(self) -> None:
        self._stop.set()
        worker_task = self._worker_task
        self._worker_task = None
        if worker_task is not None:
            await worker_task

    async def _close_checkpoint(self) -> None:
        checkpoint_context = self._checkpoint_context
        self._checkpoint_context = None
        if checkpoint_context is not None:
            await checkpoint_context.__aexit__(None, None, None)

    async def restart_worker(self) -> None:
        await self._stop_worker_for_restart()
        await self.start_worker()

    async def _stop_worker_for_restart(self) -> None:
        failures: list[tuple[str, BaseException]] = []
        await _record_fixture_teardown_error(failures, "Query Worker", self.stop_worker())
        await _record_fixture_teardown_error(
            failures, "Query checkpoint", self._close_checkpoint()
        )
        if self._dependencies is not None:
            await _record_fixture_teardown_error(
                failures,
                "Query dependencies",
                close_query_dependencies(self._dependencies),
            )
            self._dependencies = None
        _raise_fixture_teardown_failures(failures)

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
        await self._isolated_broker.publish(
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
        failures: list[tuple[str, BaseException]] = []
        await _record_fixture_teardown_error(failures, "Query Worker", self.stop_worker())
        await _record_fixture_teardown_error(
            failures, "Query checkpoint", self._close_checkpoint()
        )
        await _record_fixture_teardown_error(failures, "Query API client", self.client.aclose())
        await _record_fixture_teardown_error(
            failures, "Query API lifespan", self._app_lifespan.__aexit__(None, None, None)
        )
        await _record_fixture_teardown_error(
            failures,
            "fixture Elasticsearch index",
            self.container.elasticsearch.indices.delete(
                index=f"agenticrag-children-{self.settings.index_generation}",
                ignore_unavailable=True,
            ),
        )
        await _record_fixture_teardown_error(
            failures,
            "fixture MySQL state",
            _cleanup_local_fixture_mysql(
                self.container, self.settings, self._isolated_broker
            ),
        )
        await _record_fixture_teardown_error(
            failures,
            "fixture Redis state",
            _cleanup_local_fixture_redis(self.container, self._isolated_broker),
        )
        await _record_fixture_teardown_error(
            failures, "fixture container", _strict_container_close(self.container)
        )
        _raise_fixture_teardown_failures(failures)


@pytest.fixture
async def real_query_fixture(tmp_path: Path) -> AsyncIterator[RealQueryFixture]:
    settings = _require_local_services(tmp_path)
    migration = Config(str(Path("alembic.ini").resolve()))
    migration.set_main_option("sqlalchemy.url", settings.mysql_dsn)
    await asyncio.to_thread(command.upgrade, migration, "head")
    container = build_container(settings)
    isolated_broker = IsolatedQueryBroker(
        container.broker,
        namespace=f"query-fixture-{uuid4().hex}",
    )
    container.run_manager = RunManager(
        session_factory=container.repositories.session_factory,
        runs=TransactionalRunRepository(
            container.repositories.session_factory,
            outbox_stream_name=isolated_broker.query_stream,
        ),
    )
    await container.redis.ping()
    await container.elasticsearch.info()
    async with container.mysql_engine.connect() as connection:
        await connection.execute(select(1))
    scope = UserScope(user_id=settings.default_user_id)
    await _seed_document(container, settings, scope)
    app = create_app(settings, container=container)
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
        broker=isolated_broker,
    )
    await fixture.start_worker()
    try:
        yield fixture
    finally:
        await fixture.close()


class RealQueryRuntime(RealQueryFixture):
    """One opt-in live-provider runtime used only for protocol-smoke checks.

    The fixture creates the production container once, then shares it between
    the API and ``QueryWorker``.  Its only synthetic collaborator is the
    deterministic vector used to seed a disposable Elasticsearch generation;
    query-time DeepSeek, Qwen, reranker and Mem0 clients all come from the
    production composition root.  Its direct seed means it must never be
    presented as a business-quality RAG evaluation.
    """

    def __init__(
        self,
        *,
        settings: Settings,
        container: AppContainer,
        app: FastAPI,
        client: httpx.AsyncClient,
        app_lifespan: Any,
        parent_id: str,
        broker: IsolatedQueryBroker,
    ) -> None:
        super().__init__(
            settings=settings,
            container=container,
            app=app,
            client=client,
            app_lifespan=app_lifespan,
            broker=broker,
        )
        self.snapshot: RuntimeConfigSnapshot = container.runtime_snapshot
        self.seeded_question = (
            "How many days notice does the seeded production document require?"
        )
        self._parent_id = parent_id
        self._isolated_broker = broker
        self._evaluation_summary: Mapping[str, object] | None = None
        self.memory_boundary: dict[str, bool] = {"read": False, "write": False}
        self.memory_marker: str | None = None
        self._degradation_run_ids: set[str] = set()

    @property
    def evaluation_summary(self) -> Mapping[str, object]:
        if self._evaluation_summary is None:
            raise RuntimeError("real Query evaluation has not completed")
        return self._evaluation_summary

    @property
    def seeded_parent_id(self) -> str:
        """Server-created Parent ID the live API answer must cite."""
        return self._parent_id

    async def start_worker(self) -> None:
        dependencies = await build_query_dependencies(
            self.container,
            self.settings,
            child_index=f"agenticrag-children-{self.settings.index_generation}",
        )
        self._dependencies = dependencies
        self._checkpoint_context = self.container.checkpoints.open_query()
        checkpointer = await self._checkpoint_context.__aenter__()
        self._worker = QueryWorker(
            runs=TransactionalRunRepository(self.container.repositories.session_factory),
            broker=self._isolated_broker,
            graph_factory=build_graph_factory(dependencies, checkpointer),
            worker_id=f"real-query-e2e-{uuid4().hex}",
            concurrency=dependencies.concurrency,
            trace_recorder=dependencies.trace_recorder,
            event_emitter=dependencies.event_emitter,
            block_ms=50,
            heartbeat_interval_seconds=0.5,
            lease_seconds=10,
            run_timeout_seconds=self.settings.query_run_timeout_seconds,
            outbox_dispatcher=OutboxDispatcher(
                TransactionalQueryOutboxAdapter(
                    self.container.repositories.session_factory,
                    user_id=self.settings.default_user_id,
                    stream_name=self._isolated_broker.query_stream,
                ),
                self._isolated_broker,
                aggregate_type="query_run",
            ),
            outbox_interval_seconds=0.1,
        )
        self._stop = asyncio.Event()
        self._worker_task = asyncio.create_task(self._worker.run_forever(stop_event=self._stop))

    async def inject_duplicate_delivery(self, run_id: str) -> None:
        """Exercise duplicate delivery only in this live fixture's stream."""
        await self._isolated_broker.publish(
            QUERY_STREAM, run_id, datetime.now(UTC), dedupe_key=None
        )

    async def wait_for_terminal(
        self,
        run_id: str,
        *,
        restart_worker: bool = False,
        timeout: float = 180.0,
    ) -> dict[str, object]:
        result = await super().wait_for_terminal(
            run_id,
            restart_worker=restart_worker,
            timeout=timeout,
        )
        if result.get("status") == RunStatus.COMPLETED.value:
            await self._emit_controlled_degradation(run_id)
        return result

    async def read_sse(self, run_id: str) -> list[dict[str, object]]:
        """Read the completed run's reconnectable, public SSE projection."""
        await self.wait_for_terminal(run_id)
        response = await self.client.get(f"/v1/query-runs/{run_id}/events")
        if response.status_code != 200:
            raise AssertionError(f"SSE endpoint returned {response.status_code}")
        events: list[dict[str, object]] = []
        for block in response.text.split("\n\n"):
            for line in block.splitlines():
                if not line.startswith("data: "):
                    continue
                try:
                    payload = json.loads(line.removeprefix("data: "))
                except json.JSONDecodeError as error:
                    raise AssertionError("SSE contained invalid JSON") from error
                if isinstance(payload, Mapping):
                    events.append(dict(payload))
        return events

    async def exercise_memory_boundary(self) -> None:
        """Prove a scoped Mem0 read and write through the production service."""
        memory = self.container.memory_service
        if memory is None or not bool(getattr(memory, "available", False)):
            raise RuntimeError("real Query provider E2E requires an available Mem0 provider")
        scope = UserScope(user_id=self.settings.default_user_id)
        marker = f"memory-boundary-{uuid4().hex}"
        context = await memory.load_context(scope, marker)
        if context.degraded:
            raise RuntimeError("Mem0 read boundary degraded during real provider E2E")
        self.memory_boundary["read"] = True
        await memory.extract_and_store(
            scope,
            f"memory-boundary-{uuid4().hex}",
            [
                PublicMessage(
                    id="memory-boundary-message",
                    role="user",
                    content=(
                        "Please remember this durable acceptance preference marker: "
                        f"{marker}."
                    ),
                )
            ],
        )
        deadline = asyncio.get_running_loop().time() + 30.0
        while asyncio.get_running_loop().time() < deadline:
            records = await memory.list(scope)
            if any(marker in record.text for record in records):
                self.memory_boundary["write"] = True
                self.memory_marker = marker
                return
            await asyncio.sleep(0.25)
        raise RuntimeError("Mem0 write boundary did not return the stored acceptance marker")

    async def run_evaluation(self) -> None:
        """Retain the fixture hook as a protocol-smoke marker, not an evaluation."""
        self._evaluation_summary = _contract_only_fixture_summary(
            self.snapshot.snapshot_id
        )

    async def close(self) -> None:
        """Tear down every real boundary even when the worker already failed."""
        failures: list[tuple[str, BaseException]] = []
        await _record_fixture_teardown_error(failures, "Query Worker", self.stop_worker())
        await _record_fixture_teardown_error(
            failures, "Query checkpoint", self._close_checkpoint()
        )
        if self._dependencies is not None:
            await _record_fixture_teardown_error(
                failures, "Query dependencies", close_query_dependencies(self._dependencies)
            )
            self._dependencies = None
        await _record_fixture_teardown_error(failures, "Query API client", self.client.aclose())
        await _record_fixture_teardown_error(
            failures, "Query API lifespan", self._app_lifespan.__aexit__(None, None, None)
        )
        await _cleanup_real_provider_boundaries(
            failures, self.container, self.settings, self._isolated_broker
        )
        await _record_fixture_teardown_error(
            failures, "fixture container", _strict_container_close(self.container)
        )
        _raise_fixture_teardown_failures(failures)

    async def _emit_controlled_degradation(self, run_id: str) -> None:
        if run_id in self._degradation_run_ids:
            return
        dependencies = self._dependencies
        emitter = dependencies.event_emitter if dependencies is not None else None
        if emitter is None:
            raise RuntimeError("real Query provider E2E requires a durable event emitter")
        async with event_emission_scope(
            emitter,
            run_id,
            "console.acceptance",
            user_id=self.settings.default_user_id,
        ):
            await emit_degradation(
                component="retrieval",
                reason="circuit_open",
                run_id=run_id,
                snapshot_id=self.snapshot.snapshot_id,
                attempt=1,
                retryable=True,
                outcome="degraded",
            )
        self._degradation_run_ids.add(run_id)


async def _cleanup_real_provider_mysql(
    container: AppContainer,
    settings: Settings,
    broker: IsolatedQueryBroker | None,
) -> None:
    """Delete exactly this fixture's user-scoped rows."""
    run_ids: list[str] = []
    async with container.repositories.session_factory.begin() as session:
        run_ids = list(
            (
                await session.execute(
                    select(agent_runs.c.id).where(
                        agent_runs.c.user_id == settings.default_user_id
                    )
                )
            ).scalars()
        )
        if run_ids:
            await session.execute(
                delete(task_outbox).where(
                    task_outbox.c.aggregate_type == "query_run",
                    task_outbox.c.aggregate_id.in_(run_ids),
                )
            )
        await session.execute(
            delete(memory_tombstones).where(
                memory_tombstones.c.user_id == settings.default_user_id
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


async def _cleanup_real_provider_redis(
    container: AppContainer, broker: IsolatedQueryBroker | None
) -> None:
    if broker is not None:
        await container.redis.delete(*broker.cleanup_keys)


async def _cleanup_real_provider_query_index(container: AppContainer, settings: Settings) -> None:
    await container.elasticsearch.indices.delete(
        index=f"agenticrag-children-{settings.index_generation}",
        ignore_unavailable=True,
    )


async def _cleanup_real_provider_mem0_index(container: AppContainer, settings: Settings) -> None:
    await container.elasticsearch.indices.delete(
        index=settings.mem0_collection,
        ignore_unavailable=True,
    )


async def _cleanup_real_provider_boundaries(
    failures: list[tuple[str, BaseException]],
    container: AppContainer,
    settings: Settings,
    broker: IsolatedQueryBroker | None,
) -> None:
    await _record_fixture_teardown_error(
        failures, "fixture query index", _cleanup_real_provider_query_index(container, settings)
    )
    await _record_fixture_teardown_error(
        failures, "fixture Mem0 index", _cleanup_real_provider_mem0_index(container, settings)
    )
    await _record_fixture_teardown_error(
        failures, "fixture MySQL state", _cleanup_real_provider_mysql(container, settings, broker)
    )
    await _record_fixture_teardown_error(
        failures, "fixture Redis state", _cleanup_real_provider_redis(container, broker)
    )


@pytest.fixture
async def real_query_runtime(tmp_path: Path) -> AsyncIterator[RealQueryRuntime]:
    """Opt-in live provider protocol fixture; it produces no quality report."""
    settings = _require_real_provider_services(tmp_path)
    admin_mysql_dsn = require_service_backup_admin_dsn()
    suffix = settings.index_generation.removeprefix("real-query-")
    source_mysql_database = f"agentic_rag_acceptance_source_{suffix}"
    await asyncio.to_thread(
        create_isolated_mysql_database, admin_mysql_dsn, source_mysql_database
    )
    settings = settings.model_copy(
        update={
            "mysql_dsn": isolated_mysql_dsn(
                admin_mysql_dsn, source_mysql_database
            )
        }
    )
    container: AppContainer | None = None
    app_lifespan: Any = None
    client: httpx.AsyncClient | None = None
    runtime: RealQueryRuntime | None = None
    isolated_broker: IsolatedQueryBroker | None = None
    try:
        migration = Config(str(Path("alembic.ini").resolve()))
        migration.set_main_option("sqlalchemy.url", settings.mysql_dsn)
        await asyncio.to_thread(command.upgrade, migration, "head")
        container = build_container(settings)
        isolated_broker = IsolatedQueryBroker(
            container.broker,
            namespace=settings.index_generation,
        )
        container.run_manager = RunManager(
            session_factory=container.repositories.session_factory,
            runs=TransactionalRunRepository(
                container.repositories.session_factory,
                outbox_stream_name=isolated_broker.query_stream,
            ),
        )
        await container.redis.ping()
        await container.elasticsearch.info()
        async with container.mysql_engine.connect() as connection:
            await connection.execute(select(1))
        if container.memory_service is None or not bool(
            getattr(container.memory_service, "available", False)
        ):
            raise RuntimeError("real Query provider E2E requires an available Mem0 provider")
        parent_id = await _seed_document(
            container,
            settings,
            UserScope(user_id=settings.default_user_id),
        )
        app = create_app(settings, container=container)
        app_lifespan = app.router.lifespan_context(app)
        await app_lifespan.__aenter__()
        client = httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://real-query",
        )
        runtime = RealQueryRuntime(
            settings=settings,
            container=container,
            app=app,
            client=client,
            app_lifespan=app_lifespan,
            parent_id=parent_id,
            broker=isolated_broker,
        )
        await runtime.start_worker()
        await runtime.exercise_memory_boundary()
        await runtime.run_evaluation()
        yield runtime
    finally:
        try:
            if runtime is not None:
                await runtime.close()
            else:
                try:
                    if client is not None:
                        await client.aclose()
                finally:
                    try:
                        if app_lifespan is not None:
                            await app_lifespan.__aexit__(None, None, None)
                    finally:
                        if container is not None:
                            failures: list[tuple[str, BaseException]] = []
                            await _cleanup_real_provider_boundaries(
                                failures, container, settings, isolated_broker
                            )
                            await _record_fixture_teardown_error(
                                failures,
                                "fixture container",
                                _strict_container_close(container),
                            )
                            _raise_fixture_teardown_failures(failures)
        finally:
            await asyncio.to_thread(
                drop_isolated_mysql_database,
                admin_mysql_dsn,
                source_mysql_database,
            )
