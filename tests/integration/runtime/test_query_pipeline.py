"""Real-service Query API -> Outbox -> Worker -> Graph integration contract."""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace
from typing import TYPE_CHECKING

import pytest

from agentic_rag.config import Settings
from agentic_rag.domain.models import UserScope
from agentic_rag.memory.models import MemoryContext
from agentic_rag.query.state import new_query_state
from agentic_rag.retrieval.models import ChildHit, EvidenceBatch, ParentEvidence
from agentic_rag.runtime.concurrency import ConcurrencyManager
from agentic_rag.runtime.model_gateway import ModelResponse
import agentic_rag.runtime.query_composition as query_composition

pytest_plugins = ("tests.fixtures.query_services",)

if TYPE_CHECKING:
    from tests.fixtures.query_services import RealQueryFixture

pytestmark = pytest.mark.integration


@pytest.mark.e2e


async def test_query_run_reaches_audited_answer_through_real_boundaries(
    real_query_fixture: RealQueryFixture,
) -> None:
    response = await real_query_fixture.create_query(
        "What does the seeded production document require?"
    )

    assert response.status_code == 202
    run_id = response.json()["run_id"]
    terminal = await real_query_fixture.wait_for_terminal(run_id)

    assert terminal["status"] == "completed"
    assert terminal["answer"]["audited"] is True
    assert terminal["answer"]["segments"]
    assert await real_query_fixture.events_contain(run_id, "ANSWER_FINALIZED")
    assert "ANSWER_FINALIZED" in await real_query_fixture.sse_body(run_id)


@pytest.mark.e2e
async def test_duplicate_delivery_and_worker_restart_keep_one_terminal_run(
    real_query_fixture: RealQueryFixture,
) -> None:
    response = await real_query_fixture.create_query(
        "What does the seeded production document require?", thread_id="restart"
    )
    run_id = response.json()["run_id"]
    await real_query_fixture.inject_duplicate_delivery(run_id)
    terminal = await real_query_fixture.wait_for_terminal(run_id, restart_worker=True)

    assert terminal["status"] == "completed"
    assert await real_query_fixture.terminal_event_count(run_id) == 1


class _PipelineMemory:
    async def load_context(
        self, scope: UserScope, query: str, limit: int = 10
    ) -> MemoryContext:
        del scope, query, limit
        return MemoryContext()

    async def extract_and_store(self, *args: object, **kwargs: object) -> None:
        del args, kwargs


class _RecordingRetrieval:
    last: "_RecordingRetrieval | None" = None

    def __init__(self, _: object) -> None:
        self.calls: list[tuple[object, UserScope, object]] = []
        type(self).last = self

    async def retrieve(
        self, request: object, scope: UserScope, snapshot: object
    ) -> EvidenceBatch:
        self.calls.append((request, scope, snapshot))
        query = request.query  # type: ignore[union-attr]
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


class _ScriptedGateway:
    def __init__(self) -> None:
        self._actions = iter(
            (
                {"action": "create_todos", "items": [{"key": "notice", "title": "Find notice period"}]},
                {"action": "delegate_research", "todo_ids": ["todo-1"]},
                {"action": "submit_evidence"},
            )
        )

    async def complete_structured(
        self, call: object, schema: type[object]
    ) -> ModelResponse[object]:
        if getattr(call, "model_role", "") == "light":
            value: object = {
                "route": "research",
                "normalized_query": "Find notice period",
                "reason_code": "scripted_pipeline",
            }
        else:
            value = next(self._actions)
        return ModelResponse(
            value=schema.model_validate(value),
            requested_model="test",
            actual_model="test",
            input_tokens=1,
            output_tokens=1,
            attempts=1,
            latency_ms=1,
        )


class _NoopOpenAIClient:
    def __init__(self, **_: object) -> None:
        return None

    async def close(self) -> None:
        return None


class _CrossEncoder:
    def __init__(self, _: str) -> None:
        return None

    def predict(self, pairs: object) -> list[float]:
        return [1.0 for _ in pairs]  # type: ignore[union-attr]


async def test_production_composition_graph_factory_delegates_research_with_server_scope(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A production-composed graph delegates child retrieval instead of refusing it."""
    gateway = _ScriptedGateway()
    monkeypatch.setattr(query_composition, "AsyncOpenAI", _NoopOpenAIClient)
    monkeypatch.setattr(
        query_composition,
        "import_module",
        lambda _: SimpleNamespace(CrossEncoder=_CrossEncoder),
    )
    monkeypatch.setattr(query_composition, "ModelGateway", lambda _: gateway)
    monkeypatch.setattr(query_composition, "RetrievalService", _RecordingRetrieval)
    container = SimpleNamespace(
        elasticsearch=object(),
        repositories=SimpleNamespace(session_factory=object()),
        artifacts=object(),
        event_repository=object(),
        memory_service=_PipelineMemory(),
    )
    settings = Settings(
        mysql_dsn="mysql+asyncmy://user:password@127.0.0.1:3306/app",
        deepseek_base_url="https://api.deepseek.com",
        qwen_embedding_base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        deepseek_api_key="test-deepseek-key",
        qwen_api_key="test-qwen-key",
    )
    shared_concurrency = ConcurrencyManager()
    dependencies = await query_composition.build_query_dependencies(
        container,
        settings,
        concurrency=shared_concurrency,
    )
    try:
        assert dependencies.concurrency is shared_concurrency
        assert dependencies.research_loop._subagents is not None

        from agentic_rag.runtime.query_worker import build_graph_factory

        graph = build_graph_factory(
            replace(dependencies, event_repository=None, event_emitter=None),
            None,  # type: ignore[arg-type]
        )(checkpoint_thread_id="query:server-user:pipeline")
        state = new_query_state(
            run_id="run-production-subagent",
            question="Find notice period",
            scope=UserScope(user_id="server-user"),
            snapshot=query_composition.build_query_snapshot(settings),
        )

        result = await graph.ainvoke(
            state,
            {"configurable": {"thread_id": "query:server-user:pipeline"}},
        )

        retrieval = _RecordingRetrieval.last
        assert retrieval is not None and len(retrieval.calls) == 1
        assert retrieval.calls[0][1].user_id == "server-user"
        observations = result["research"]["observations"]
        assert all(
            observation.get("error") != "subagents unavailable"
            for observation in observations
        )
    finally:
        await query_composition.close_query_dependencies(dependencies)
