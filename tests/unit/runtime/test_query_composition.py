"""Production Query dependency composition contracts."""

from __future__ import annotations

from contextlib import asynccontextmanager
from types import SimpleNamespace

import pytest

from agentic_rag.config import Settings
from agentic_rag.domain.models import UserScope
from agentic_rag.query.evidence_builder import EvidenceBuilder
from agentic_rag.query.subagents import SubagentDispatcher
from agentic_rag.query.todos import TodoItem
from agentic_rag.query.tools import ResearchContext
from agentic_rag.retrieval.models import ChildHit, EvidenceBatch, ParentEvidence
from agentic_rag.runtime.concurrency import ConcurrencyManager
from agentic_rag.runtime.models import RuntimeConfigSnapshot
import agentic_rag.runtime.query_composition as query_composition
from agentic_rag.runtime.query_composition import (
    QueryCompositionError,
    build_query_dependencies,
)


SNAPSHOT = RuntimeConfigSnapshot(
    app_version="test",
    graph_version="query-v1",
    prompt_version="prompt-v1",
    main_model_id="main",
    light_model_id="light",
    embedding_model="embedding",
    embedding_dimensions=1024,
    reranker_version="reranker",
    retrieval_config_version="retrieval",
    index_generation="current-index",
    memory_config_version="memory",
)


def _batch_for_scope(scope: UserScope, query: str) -> EvidenceBatch:
    hit = ChildHit(
        child_id="child-1",
        parent_id="parent-1",
        user_id=scope.user_id,
        document_id="doc-1",
        document_version_id="version-1",
        content=query,
        ast_locator="#/text/1",
        lane="dense",
        lane_rank=1,
        score=1.0,
    )
    return EvidenceBatch(
        query=query,
        parents=(
            ParentEvidence(
                parent_id="parent-1",
                document_id="doc-1",
                document_version_id="version-1",
                content=query,
                child_hits=(hit,),
                rerank_score=1.0,
            ),
        ),
    )


class RecordingRetrieval:
    def __init__(self) -> None:
        self.requests: list[tuple[object, UserScope, RuntimeConfigSnapshot]] = []

    async def retrieve(
        self,
        request: object,
        scope: UserScope,
        snapshot: RuntimeConfigSnapshot,
    ) -> EvidenceBatch:
        self.requests.append((request, scope, snapshot))
        return _batch_for_scope(scope, request.query)  # type: ignore[union-attr]


def test_build_subagent_dispatcher_returns_the_real_dispatcher() -> None:
    dispatcher = query_composition.build_subagent_dispatcher(
        retrieval=RecordingRetrieval(),
        evidence_builder=EvidenceBuilder(),
        snapshot=SNAPSHOT,
        concurrency=ConcurrencyManager(),
    )

    assert isinstance(dispatcher, SubagentDispatcher)


async def test_composed_child_worker_inherits_server_scope_and_current_snapshot() -> None:
    retrieval = RecordingRetrieval()
    dispatcher = query_composition.build_subagent_dispatcher(
        retrieval=retrieval,
        evidence_builder=EvidenceBuilder(),
        snapshot=SNAPSHOT,
        concurrency=ConcurrencyManager(),
    )
    parent_context = ResearchContext(
        scope=UserScope(user_id="server-owned-user"),
        snapshot=SNAPSHOT,
    )

    result = await dispatcher.delegate(
        (
            TodoItem(
                id="todo-1",
                title="Find the notice period",
                owner="supervisor",
            ),
        ),
        parent_context,
    )

    assert result.results[0].evidence.index_generation == "current-index"
    assert retrieval.requests[0][1].user_id == "server-owned-user"
    assert retrieval.requests[0][2] == SNAPSHOT


async def test_composed_child_inherits_evaluation_snapshot_not_process_baseline() -> None:
    from agentic_rag.runtime.models import EvaluationMetadata
    retrieval = RecordingRetrieval()
    dispatcher = query_composition.build_subagent_dispatcher(
        retrieval=retrieval, evidence_builder=EvidenceBuilder(), snapshot=SNAPSHOT,
        concurrency=ConcurrencyManager())
    evaluation = EvaluationMetadata(session_id="e", dataset_sha256="a" * 64, corpus_snapshot_id="b" * 64)
    current = SNAPSHOT.model_copy(update={"evaluation": evaluation})
    await dispatcher.delegate((TodoItem(id="todo", title="Find fact", owner="supervisor"),),
                              ResearchContext(scope=UserScope(user_id="server-owned-user"), snapshot=current))
    assert retrieval.requests[0][2].snapshot_id == current.snapshot_id


async def test_worker_entrypoint_receives_the_composed_concurrency_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The entrypoint must not replace the graph's process-wide budget."""
    import scripts.run_query_worker as worker_entrypoint

    from sqlalchemy.ext.asyncio import create_async_engine
    from agentic_rag.persistence.repositories import metadata

    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as connection:
        await connection.run_sync(metadata.create_all)
    shared_concurrency = ConcurrencyManager()
    dependencies = SimpleNamespace(
        concurrency=shared_concurrency,
        trace_recorder=None,
        event_emitter=None,
    )
    recorded: dict[str, object] = {}

    class RecordingWorker:
        def __init__(self, **kwargs: object) -> None:
            recorded.update(kwargs)

        async def run_forever(self, *, stop_event: object) -> None:
            del stop_event

    class FakeCheckpoints:
        @asynccontextmanager
        async def open_query(self) -> object:
            yield object()

    class FakeContainer:
        mysql_engine = engine
        checkpoints = FakeCheckpoints()
        broker = object()
        repositories = SimpleNamespace(session_factory=object())

        async def close(self) -> None:
            await engine.dispose()

    async def dependencies_factory(*_: object) -> object:
        return dependencies

    async def close_dependencies(*_: object) -> None:
        return None

    async def ensure_active_child_alias(container: object, settings: object) -> None:
        recorded["alias_container"] = container
        recorded["alias_settings"] = settings

    monkeypatch.setattr(worker_entrypoint, "build_container", lambda _: FakeContainer())
    monkeypatch.setattr(worker_entrypoint, "build_graph_factory", lambda *_: object())
    monkeypatch.setattr(worker_entrypoint, "TransactionalRunRepository", lambda _: object())
    monkeypatch.setattr(
        worker_entrypoint, "OutboxDispatcher", lambda *_, **__: object()
    )
    monkeypatch.setattr(worker_entrypoint, "QueryWorker", RecordingWorker)
    monkeypatch.setattr(worker_entrypoint, "close_query_dependencies", close_dependencies)
    monkeypatch.setattr(
        worker_entrypoint,
        "ensure_active_child_alias",
        ensure_active_child_alias,
        raising=False,
    )

    settings = _settings()
    await worker_entrypoint.run(settings, dependencies_factory=dependencies_factory)  # type: ignore[arg-type]

    assert recorded["concurrency"] is shared_concurrency
    assert recorded["alias_settings"] is settings


async def test_query_worker_alias_startup_uses_current_index_generation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import scripts.run_query_worker as worker_entrypoint

    calls: list[tuple[object, str]] = []

    class RecordingAliasStore:
        def __init__(self, client: object) -> None:
            self._client = client

        async def ensure_active_alias(self, index_generation: str) -> bool:
            calls.append((self._client, index_generation))
            return True

    client = object()
    monkeypatch.setattr(
        worker_entrypoint, "ElasticsearchChildIndexStore", RecordingAliasStore
    )

    settings = _settings(index_generation="runtime-index")
    container = SimpleNamespace(elasticsearch=client)
    await worker_entrypoint.ensure_active_child_alias(container, settings)

    assert calls == [(client, "runtime-index")]


def _settings(**overrides: object) -> Settings:
    values: dict[str, object] = {
        "mysql_dsn": "mysql+asyncmy://user:password@127.0.0.1:3306/app",
        "deepseek_base_url": "https://api.deepseek.com",
        "qwen_embedding_base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
    }
    values.update(overrides)
    return Settings(**values)


@pytest.mark.asyncio
async def test_query_composition_fails_closed_without_model_credentials() -> None:
    with pytest.raises(QueryCompositionError, match="credentials"):
        await build_query_dependencies(
            object(),
            _settings(
                deepseek_api_key="replace-with-key",
                qwen_api_key="replace-with-key",
            ),
        )


@pytest.mark.asyncio
async def test_query_composition_reports_missing_reranker_dependency(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def missing_dependency(name: str) -> object:
        if name == "sentence_transformers":
            raise ImportError("simulated missing sentence-transformers")
        return query_composition.import_module(name)

    monkeypatch.setattr(query_composition, "import_module", missing_dependency)
    with pytest.raises(QueryCompositionError, match="reranker"):
        await build_query_dependencies(
            object(),
            _settings(
                deepseek_api_key="deepseek-test-key",
                qwen_api_key="qwen-test-key",
            ),
        )
