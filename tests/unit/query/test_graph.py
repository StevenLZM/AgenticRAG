"""Macro-graph contracts for the audited query runtime."""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field

import pytest

from agentic_rag.domain.models import UserScope
from agentic_rag.memory.models import MemoryContext, PublicMessage
from agentic_rag.models.schemas import EvidenceGrade, RouteDecision
from agentic_rag.query.evidence_builder import EvidenceBuilder
from agentic_rag.retrieval.models import ChildHit, EvidenceBatch, ParentEvidence
from agentic_rag.runtime.model_gateway import ModelResponse
from agentic_rag.runtime.models import RuntimeConfigSnapshot


SCOPE = UserScope(user_id="user-1")
SNAPSHOT = RuntimeConfigSnapshot(
    app_version="test", graph_version="query-v1", prompt_version="prompt-v1",
    main_model_id="main", light_model_id="light", embedding_model="embedding",
    embedding_dimensions=1024, reranker_version="reranker",
    retrieval_config_version="retrieval", index_generation="index-v1",
    memory_config_version="memory-v1",
)


def _batch() -> EvidenceBatch:
    child = ChildHit(
        child_id="child-1", parent_id="parent-1", user_id=SCOPE.user_id,
        document_id="document-1", document_version_id="version-1",
        content="The contract requires thirty days notice.", ast_locator="#/text/1",
        lane="dense", lane_rank=1, score=1.0,
    )
    return EvidenceBatch(
        query="notice",
        parents=(ParentEvidence(
            parent_id="parent-1", document_id="document-1",
            document_version_id="version-1", content=child.content,
            child_hits=(child,), rerank_score=1.0,
        ),),
    )


@dataclass
class FakeMemory:
    loads: int = 0
    stored: list[tuple[str, list[PublicMessage]]] = field(default_factory=list)

    async def load_context(self, scope: UserScope, query: str, limit: int = 10) -> MemoryContext:
        del scope, query, limit
        self.loads += 1
        return MemoryContext()

    async def extract_and_store(
        self, scope: UserScope, run_id: str, messages: list[PublicMessage]
    ) -> None:
        del scope
        self.stored.append((run_id, messages))


@dataclass
class FakeGateway:
    route: RouteDecision
    answer_values: list[object] = field(default_factory=list)

    async def complete_structured(self, call: object, schema: type[object]) -> ModelResponse[object]:
        role = getattr(call, "model_role", "")
        if role == "light" and "Classify the query" in getattr(call, "messages")[0]["content"]:
            value: object = self.route
        else:
            value = self.answer_values.pop(0)
        return ModelResponse(
            value=value, requested_model="test", actual_model="test", input_tokens=1,
            output_tokens=1, attempts=1, latency_ms=1,
        )


@dataclass
class FakeRetrieval:
    calls: int = 0

    async def retrieve(self, request: object, scope: UserScope, snapshot: RuntimeConfigSnapshot) -> EvidenceBatch:
        del request, scope, snapshot
        self.calls += 1
        return _batch()


@dataclass
class FakeGrader:
    grades: list[EvidenceGrade]
    calls: int = 0

    async def grade(self, question: str, packed_evidence: object, **_: object) -> EvidenceGrade:
        del question, packed_evidence
        self.calls += 1
        return self.grades.pop(0)


@dataclass
class FakeResearchLoop:
    result: dict[str, object]
    calls: int = 0

    async def ainvoke(self, state: object) -> dict[str, object]:
        self.calls += 1
        return {**self.result, "evidence": list(getattr(state, "get")("evidence", self.result.get("evidence", [])))}


@dataclass
class EventLog:
    types: list[str] = field(default_factory=list)
    events: list[object] = field(default_factory=list)

    async def append(self, event: object) -> int:
        self.types.append(getattr(event, "event_type"))
        self.events.append(event)
        return len(self.types)


@dataclass
class RecordingEvidenceBuilder:
    delegate: EvidenceBuilder = field(default_factory=EvidenceBuilder)
    calls: list[tuple[object, object, object, object]] = field(default_factory=list)

    def build(self, batches: object, targets: object, scope: object, snapshot: object) -> object:
        self.calls.append((batches, targets, scope, snapshot))
        return self.delegate.build(batches, targets, scope, snapshot)  # type: ignore[arg-type]


class Resolver:
    async def resolve(self, manifest: object, scope: UserScope, snapshot: RuntimeConfigSnapshot) -> dict[str, object]:
        del scope, snapshot
        return {
            evidence_id: {
                "user_id": SCOPE.user_id, "index_generation": SNAPSHOT.index_generation,
                "is_active": True, "parent_id": entry.parent_id,
                "document_id": entry.document_id,
                "document_version_id": entry.document_version_id,
                "ast_locator": entry.ast_locator,
            }
            for evidence_id, entry in manifest.items()
        }


def _state() -> dict[str, object]:
    from agentic_rag.query.state import new_query_state

    return new_query_state(
        run_id="run-1", question="What notice is required?", scope=SCOPE,
        snapshot=SNAPSHOT, messages=[{"id": "m1", "role": "user", "content": "What notice is required?"}],
    )


def _deps(*, route: str = "fast_rag", grades: list[str] | None = None, research: dict[str, object] | None = None, evidence_builder: object | None = None, trace_recorder: object | None = None) -> tuple[object, FakeMemory, FakeRetrieval, EventLog]:
    from agentic_rag.query.audit import CitationValidator, FaithfulnessAuditor
    from agentic_rag.query.graph import QueryGraphDependencies

    memory = FakeMemory()
    retrieval = FakeRetrieval()
    gateway = FakeGateway(
        RouteDecision(route=route, normalized_query="notice", reason_code="test"),
        answer_values=[
            {"segments": [{"kind": "content", "text": "Thirty days.", "evidence_ids": ["evidence-placeholder"]}]},
            {"passed": True, "unsupported_claim_ids": [], "reasons": []},
        ],
    )
    # The graph supplies the server-derived real evidence id before generation;
    # this test double replaces it from the manifest in its generator wrapper.
    class Generator:
        async def generate(self, question: str, packed: object, snapshot: object, **_: object) -> object:
            from agentic_rag.query.generation import AnswerDraft
            item = packed.items[0]
            return AnswerDraft.model_validate({"segments": [{"kind": "content", "text": "Thirty days.", "evidence_ids": [item.evidence_id]}]})

    event_log = EventLog()
    configured_grades = [EvidenceGrade(decision=value) for value in (grades or ["sufficient"])]
    deps = QueryGraphDependencies(
        memory=memory, gateway=gateway, retrieval=retrieval, evidence_builder=evidence_builder or EvidenceBuilder(),
        evidence_grader=FakeGrader(configured_grades),
        research_loop=FakeResearchLoop(research or {
            "research": {"submitted": True},
            "retrieval_batches": [_batch().model_dump(mode="json")],
            "evidence": [], "next_node": "generate", "termination_reason": None,
        }),
        generator=Generator(), faithfulness_auditor=FaithfulnessAuditor(gateway),
        citation_validator=CitationValidator(), authorization_resolver=Resolver(), event_repository=event_log,
        trace_recorder=trace_recorder,
    )
    return deps, memory, retrieval, event_log


async def test_fast_path_loads_memory_once_and_emits_all_audit_gates_in_order() -> None:
    from agentic_rag.query.graph import build_query_graph

    deps, memory, retrieval, events = _deps()
    result = await build_query_graph(deps).ainvoke(_state())

    assert result["termination_reason"] == "completed"
    assert memory.loads == 1
    assert retrieval.calls == 1
    assert events.types.index("EVIDENCE_GRADED") < events.types.index("FAITHFULNESS_AUDITED") < events.types.index("CITATION_VALIDATED")
    assert json.loads(json.dumps(result)) == result


async def test_real_query_graph_records_lane_spans_when_trace_recorder_is_injected() -> None:
    """Instrumentation must wrap real graph boundaries, not manual span calls."""
    from agentic_rag.observability.tracing import TraceRecorder
    from agentic_rag.query.graph import build_query_graph

    recorder = TraceRecorder(runtime_config_snapshot_id=SNAPSHOT.snapshot_id)
    deps, _memory, _retrieval, _events = _deps(trace_recorder=recorder)
    result = await build_query_graph(deps).ainvoke(_state())

    assert result["termination_reason"] == "completed"
    names = {span.name for span in recorder.events}
    assert {"graph.node.memory_loader", "memory", "llm", "retrieval", "rerank", "audit"} <= names


async def test_insufficient_fast_evidence_escalates_to_research_before_audits() -> None:
    from agentic_rag.query.graph import build_query_graph

    deps, _memory, _retrieval, events = _deps(grades=["insufficient", "sufficient"])
    result = await build_query_graph(deps).ainvoke(_state())

    assert result["termination_reason"] == "completed"
    assert deps.research_loop.calls == 1
    assert events.types.count("EVIDENCE_GRADED") == 2


@pytest.mark.parametrize("decision", ["clarify", "refuse"])
async def test_clarify_and_refuse_are_terminal_without_answer_audits(decision: str) -> None:
    from agentic_rag.query.graph import build_query_graph

    deps, _memory, _retrieval, events = _deps(grades=[decision])
    result = await build_query_graph(deps).ainvoke(_state())

    assert result["termination_reason"] == decision
    assert "FAITHFULNESS_AUDITED" not in events.types
    assert "CITATION_VALIDATED" not in events.types


async def test_graph_propagates_cancellation_from_research_loop() -> None:
    from agentic_rag.query.graph import build_query_graph

    deps, _memory, _retrieval, _events = _deps(route="research")

    async def cancelled(state: object) -> dict[str, object]:
        del state
        raise asyncio.CancelledError()

    deps.research_loop.ainvoke = cancelled  # type: ignore[method-assign]
    from langgraph.errors import NodeCancelledError

    with pytest.raises(NodeCancelledError):
        await build_query_graph(deps).ainvoke(_state())


async def test_research_evidence_builder_uses_verified_batches_and_targets() -> None:
    from agentic_rag.query.graph import build_query_graph

    recording = RecordingEvidenceBuilder()
    deps, _memory, _retrieval, _events = _deps(
        route="research",
        evidence_builder=recording,
        research={
            "research": {"submitted": True},
            "retrieval_batches": [_batch().model_dump(mode="json")],
            "evidence": [],
            "next_node": "generate",
            "termination_reason": None,
        },
    )
    result = await build_query_graph(deps).ainvoke(_state())

    assert result["termination_reason"] == "completed"
    assert len(recording.calls) == 1
    batches, targets, scope, snapshot = recording.calls[0]
    assert len(batches) == 1
    assert targets[0].target_id == "query:run-1"
    assert targets[0].description == "What notice is required?"
    assert scope == SCOPE
    assert snapshot.index_generation == SNAPSHOT.index_generation


async def test_event_replay_has_stable_key_and_no_dynamic_timestamp() -> None:
    from agentic_rag.query.graph import _event

    deps, _memory, _retrieval, events = _deps()
    state = _state()
    await _event(deps, state, "ANSWER_FINALIZED", "completed")
    await _event(deps, state, "ANSWER_FINALIZED", "completed")

    assert len(events.events) == 2
    assert events.events[0].event_key == events.events[1].event_key
    assert events.events[0].created_at is None
    assert events.events[1].created_at is None
