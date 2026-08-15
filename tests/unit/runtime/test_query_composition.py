"""Production Query dependency composition contracts."""

from __future__ import annotations

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
