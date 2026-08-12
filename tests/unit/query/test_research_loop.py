"""Observation-loop tests for the bounded research agent."""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field

import pytest

from agentic_rag.domain.models import UserScope
from agentic_rag.query.evidence_builder import EvidenceBuilder
from agentic_rag.query.state import new_query_state
from agentic_rag.retrieval.models import ChildHit, EvidenceBatch, ParentEvidence
from agentic_rag.runtime.model_gateway import ModelResponse
from agentic_rag.runtime.models import RuntimeConfigSnapshot


SNAPSHOT = RuntimeConfigSnapshot(
    app_version="test", graph_version="query-v1", prompt_version="prompt-v1",
    main_model_id="main", light_model_id="light", embedding_model="embedding",
    embedding_dimensions=1024, reranker_version="reranker", retrieval_config_version="retrieval",
    index_generation="index", memory_config_version="memory",
)
SCOPE = UserScope(user_id="user-1")


def _batch() -> EvidenceBatch:
    hit = ChildHit(
        child_id="child-1", parent_id="parent-1", user_id="user-1", document_id="doc-1",
        document_version_id="version-1", content="Notice is thirty days.",
        ast_locator="#/text/1", lane="dense", lane_rank=1, score=1.0,
    )
    return EvidenceBatch(
        query="notice", parents=(ParentEvidence(
            parent_id="parent-1", document_id="doc-1", document_version_id="version-1",
            content="Notice is thirty days.", child_hits=(hit,), rerank_score=1.0,
        ),),
    )


@dataclass
class ScriptedGateway:
    actions: list[dict[str, object]]
    calls: int = 0

    async def complete_structured(self, call: object, schema: type[object]) -> ModelResponse[object]:
        del call
        action = self.actions[self.calls]
        self.calls += 1
        return ModelResponse(
            value=schema.model_validate(action), requested_model="main", actual_model="main",
            input_tokens=1, output_tokens=1, attempts=1, latency_ms=1,
        )


@dataclass
class FakeRetrieval:
    batch: EvidenceBatch = field(default_factory=_batch)
    calls: int = 0

    async def retrieve(self, request: object, scope: object, snapshot: object) -> EvidenceBatch:
        del request, scope, snapshot
        self.calls += 1
        return self.batch


def _state() -> dict[str, object]:
    return new_query_state(run_id="run-1", question="What notice applies?", scope=SCOPE, snapshot=SNAPSHOT)


async def test_agent_reenters_after_retrieval_observation_and_submits_evidence() -> None:
    from agentic_rag.query.research_loop import ResearchAgentLoop, ResearchLoopDependencies

    gateway = ScriptedGateway([
        {"action": "retrieve_evidence", "query": "notice"},
        {"action": "submit_evidence"},
    ])
    loop = ResearchAgentLoop(ResearchLoopDependencies(
        gateway=gateway, retrieval=FakeRetrieval(), evidence_builder=EvidenceBuilder()
    ))

    result = await loop.ainvoke(_state())

    assert gateway.calls == 2
    assert result["research"]["submitted"] is True
    assert result["research"]["observations"][0]["kind"] == "retrieval"
    assert json.loads(json.dumps(result)) == result


async def test_unknown_model_action_fails_closed_without_arbitrary_tool_execution() -> None:
    from agentic_rag.query.research_loop import ResearchAgentLoop, ResearchLoopDependencies

    loop = ResearchAgentLoop(ResearchLoopDependencies(
        gateway=ScriptedGateway([{"action": "search_memory", "query": "secret"}]),
        retrieval=FakeRetrieval(), evidence_builder=EvidenceBuilder(),
    ))

    result = await loop.ainvoke(_state())

    assert result["research"]["cannot_answer"] is True
    assert result["termination_reason"] == "research_action_invalid"


async def test_loop_marks_unfinished_work_blocked_after_four_rounds() -> None:
    from agentic_rag.query.research_loop import ResearchAgentLoop, ResearchLoopDependencies

    loop = ResearchAgentLoop(ResearchLoopDependencies(
        gateway=ScriptedGateway([{"action": "calculator", "expression": "1 + 1"}] * 4),
        retrieval=FakeRetrieval(), evidence_builder=EvidenceBuilder(),
    ))

    result = await loop.ainvoke(_state())

    assert result["termination_reason"] == "research_round_limit"
    assert len(result["research"]["observations"]) == 4


async def test_loop_propagates_cancellation_from_tool() -> None:
    from agentic_rag.query.research_loop import ResearchAgentLoop, ResearchLoopDependencies

    class CancellingRetrieval:
        async def retrieve(self, request: object, scope: object, snapshot: object) -> EvidenceBatch:
            del request, scope, snapshot
            raise asyncio.CancelledError()

    loop = ResearchAgentLoop(ResearchLoopDependencies(
        gateway=ScriptedGateway([{"action": "retrieve_evidence", "query": "notice"}]),
        retrieval=CancellingRetrieval(), evidence_builder=EvidenceBuilder(),
    ))

    with pytest.raises(asyncio.CancelledError):
        await loop.ainvoke(_state())


async def test_context_compacts_only_old_observations_and_completed_todo_details() -> None:
    from agentic_rag.query.context import ContextBuilder

    @dataclass
    class Compactor:
        calls: list[dict[str, object]] = field(default_factory=list)

        async def compact(self, content: dict[str, object]) -> str:
            self.calls.append(content)
            return "older work compacted"

    compactor = Compactor()
    state = _state()
    state["memory_context"] = {"rendered_context": "preference", "degraded": False}
    state["research"] = {
        "todos": [
            {"id": "done", "title": "done detail", "owner": "agent", "status": "completed", "dependencies": [], "evidence_ids": ["e1"]},
            {"id": "open", "title": "still needed", "owner": "agent", "status": "pending", "dependencies": []},
        ],
        "observations": [{"kind": "old", "value": "x" * 300}, {"kind": "latest", "value": "keep"}],
        "gaps": ["missing date"],
    }
    state["packed_context"] = {"manifest": {"e1": {"document_id": "doc-1"}}}

    built = await ContextBuilder(compactor=compactor, max_tokens=100).build(state)

    assert built["question"] == "What notice applies?"
    assert built["latest_observation"] == {"kind": "latest", "value": "keep"}
    assert built["unresolved_todos"][0]["id"] == "open"
    assert built["evidence_manifest"] == {"e1": {"document_id": "doc-1"}}
    assert built["compacted_history"] == "older work compacted"
    assert len(compactor.calls) == 1
