"""Tests for the memory-routed, one-retrieval Fast RAG entry flow."""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field

import pytest

from agentic_rag.domain.models import UserScope
from agentic_rag.memory.models import MemoryContext
from agentic_rag.models.schemas import RouteDecision
from agentic_rag.query.evidence_builder import EvidenceBuilder
from agentic_rag.retrieval.models import ChildHit, EvidenceBatch, ParentEvidence
from agentic_rag.runtime.model_gateway import (
    ModelResponse,
    StructuredOutputValidationError,
)
from agentic_rag.runtime.models import RuntimeConfigSnapshot


SNAPSHOT = RuntimeConfigSnapshot(
    app_version="test",
    graph_version="query-v1",
    prompt_version="prompt-v1",
    main_model_id="main-model",
    light_model_id="light-model",
    embedding_model="text-embedding-v3",
    embedding_dimensions=1024,
    reranker_version="reranker-v1",
    retrieval_config_version="retrieval-v1",
    index_generation="index-v1",
    memory_config_version="memory-v1",
)
SCOPE = UserScope(user_id="user-1")


def _batch() -> EvidenceBatch:
    hit = ChildHit(
        child_id="child-1",
        parent_id="parent-1",
        user_id=SCOPE.user_id,
        document_id="document-1",
        document_version_id="version-1",
        content="The contract requires 30 days notice.",
        ast_locator="#/text_blocks/1",
        lane="dense",
        lane_rank=1,
        score=0.9,
    )
    return EvidenceBatch(
        query="contract notice",
        parents=(
            ParentEvidence(
                parent_id="parent-1",
                document_id="document-1",
                document_version_id="version-1",
                content="The contract requires 30 days notice.",
                child_hits=(hit,),
                rerank_score=0.9,
            ),
        ),
    )


@dataclass
class FakeMemory:
    context: MemoryContext = field(default_factory=MemoryContext)
    calls: list[tuple[UserScope, str]] = field(default_factory=list)

    async def load_context(self, scope: UserScope, query: str, limit: int = 10) -> MemoryContext:
        del limit
        self.calls.append((scope, query))
        return self.context


@dataclass
class FakeGateway:
    decision: RouteDecision | BaseException
    calls: list[object] = field(default_factory=list)

    async def complete_structured(self, call: object, schema: type[object]) -> ModelResponse[object]:
        self.calls.append((call, schema))
        if isinstance(self.decision, BaseException):
            raise self.decision
        return ModelResponse(
            value=self.decision,
            requested_model="light-model",
            actual_model="light-model",
            input_tokens=1,
            output_tokens=1,
            attempts=1,
            latency_ms=1,
        )


@dataclass
class FakeRetrieval:
    batch: EvidenceBatch = field(default_factory=_batch)
    calls: list[tuple[object, UserScope, RuntimeConfigSnapshot]] = field(default_factory=list)

    async def retrieve(
        self, request: object, scope: UserScope, snapshot: RuntimeConfigSnapshot
    ) -> EvidenceBatch:
        self.calls.append((request, scope, snapshot))
        return self.batch


@dataclass
class FakeGrader:
    result: object
    calls: list[object] = field(default_factory=list)

    async def grade(self, question: str, packed_evidence: object, **_: object) -> object:
        self.calls.append((question, packed_evidence))
        return self.result


def _initial_state() -> dict[str, object]:
    from agentic_rag.query.state import new_query_state

    return new_query_state(
        run_id="run-1", question="What notice is required?", scope=SCOPE, snapshot=SNAPSHOT
    )


@pytest.mark.parametrize("non_finite", [float("nan"), float("inf"), float("-inf")])
def test_query_state_rejects_non_finite_message_values(non_finite: float) -> None:
    from agentic_rag.query.state import InvalidQueryState, new_query_state

    with pytest.raises(InvalidQueryState):
        new_query_state(
            run_id="run-1",
            question="What notice is required?",
            scope=SCOPE,
            snapshot=SNAPSHOT,
            messages=[{"role": "user", "score": non_finite}],
        )


async def test_memory_loads_once_before_router_and_fast_path_never_reloads() -> None:
    from agentic_rag.models.schemas import EvidenceGrade
    from agentic_rag.query.fast_rag import FastRagDependencies, run_fast_rag
    from agentic_rag.query.router import MemoryContextLoader, route_query

    memory = FakeMemory()
    gateway = FakeGateway(RouteDecision(route="fast_rag", normalized_query="contract notice", reason_code="simple"))
    retrieval = FakeRetrieval()
    grader = FakeGrader(EvidenceGrade(decision="sufficient"))
    state = _initial_state()

    state.update(await MemoryContextLoader(memory).load(state))
    state.update(await route_query(state, gateway))
    state.update(
        await run_fast_rag(
            state,
            FastRagDependencies(
                retrieval=retrieval,
                evidence_builder=EvidenceBuilder(),
                evidence_grader=grader,
            ),
        )
    )

    assert len(memory.calls) == 1
    assert len(gateway.calls) == 1
    call, _ = gateway.calls[0]
    assert call.model_role == "light"
    assert "# ROLE" in call.messages[0]["content"]
    assert len(retrieval.calls) == 1
    assert state["next_node"] == "generate"
    assert json.loads(json.dumps(state)) == state


async def test_fast_path_retrieves_once_then_escalates_on_insufficient_evidence() -> None:
    from agentic_rag.models.schemas import EvidenceGrade
    from agentic_rag.query.fast_rag import FastRagDependencies, run_fast_rag

    state = _initial_state()
    state["route"] = RouteDecision(
        route="fast_rag", normalized_query="contract notice", reason_code="simple"
    ).model_dump(mode="json")
    retrieval = FakeRetrieval()
    grader = FakeGrader(EvidenceGrade(decision="insufficient", gaps=("second contract",)))

    state.update(
        await run_fast_rag(
            state,
            FastRagDependencies(retrieval, EvidenceBuilder(), grader),
        )
    )

    assert len(retrieval.calls) == 1
    assert state["next_node"] == "research_agent"
    assert state["research"]["gaps"] == ["second contract"]


@pytest.mark.parametrize("decision", ["clarify", "refuse"])
async def test_fast_path_terminates_clarify_or_refuse_without_second_retrieval(
    decision: str,
) -> None:
    from agentic_rag.models.schemas import EvidenceGrade
    from agentic_rag.query.fast_rag import FastRagDependencies, run_fast_rag

    state = _initial_state()
    state["route"] = RouteDecision(
        route="fast_rag", normalized_query="contract notice", reason_code="simple"
    ).model_dump(mode="json")
    retrieval = FakeRetrieval()
    grader = FakeGrader(EvidenceGrade(decision=decision))

    state.update(
        await run_fast_rag(
            state,
            FastRagDependencies(retrieval, EvidenceBuilder(), grader),
        )
    )

    assert len(retrieval.calls) == 1
    assert state["next_node"] == "end"
    assert state["termination_reason"] == decision


async def test_invalid_router_output_fails_closed_to_research() -> None:
    from agentic_rag.query.router import route_query

    state = _initial_state()
    update = await route_query(state, FakeGateway(StructuredOutputValidationError("bad route")))

    assert update["route"]["route"] == "research"
    assert update["next_node"] == "research_agent"
    assert update["errors"][0]["code"] == "router_schema_invalid"


async def test_memory_loader_propagates_cancellation() -> None:
    from agentic_rag.query.router import MemoryContextLoader

    class CancellingMemory:
        async def load_context(self, scope: UserScope, query: str, limit: int = 10) -> MemoryContext:
            del scope, query, limit
            raise asyncio.CancelledError()

    with pytest.raises(asyncio.CancelledError):
        await MemoryContextLoader(CancellingMemory()).load(_initial_state())


async def test_fast_path_degrades_to_research_when_all_retrieval_is_unavailable() -> None:
    from agentic_rag.query.fast_rag import FastRagDependencies, run_fast_rag
    from agentic_rag.retrieval.graph import RetrievalUnavailable

    class UnavailableRetrieval:
        calls = 0

        async def retrieve(self, request: object, scope: object, snapshot: object) -> EvidenceBatch:
            del request, scope, snapshot
            self.calls += 1
            raise RetrievalUnavailable({})

    state = _initial_state()
    state["route"] = RouteDecision(
        route="fast_rag", normalized_query="contract notice", reason_code="simple"
    ).model_dump(mode="json")
    retrieval = UnavailableRetrieval()
    update = await run_fast_rag(
        state,
        FastRagDependencies(retrieval, EvidenceBuilder(), FakeGrader(object())),
    )

    assert retrieval.calls == 1
    assert update["next_node"] == "research_agent"
    assert update["errors"][0]["code"] == "fast_retrieval_unavailable"
