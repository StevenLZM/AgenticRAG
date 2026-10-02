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
from agentic_rag.runtime.model_gateway import ModelGateway, ModelResponse
from agentic_rag.runtime.models import RuntimeConfigSnapshot


SNAPSHOT = RuntimeConfigSnapshot(
    app_version="test", graph_version="query-v1", prompt_version="prompt-v1",
    main_model_id="main", light_model_id="light", embedding_model="embedding",
    embedding_dimensions=1024, reranker_version="reranker", retrieval_config_version="retrieval",
    index_generation="index", memory_config_version="memory", max_research_rounds=2,
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


@dataclass
class RepairingResponses:
    values: list[str]
    calls: int = 0

    async def create(self, **_: object) -> object:
        value = self.values[self.calls]
        self.calls += 1
        return type("Response", (), {"output_text": value, "model": "main"})()


@dataclass
class RepairingClient:
    values: list[str]

    def __post_init__(self) -> None:
        self.responses = RepairingResponses(self.values)


def _state() -> dict[str, object]:
    state = new_query_state(run_id="run-1", question="What notice applies?", scope=SCOPE, snapshot=SNAPSHOT)
    state["research"] = {
        "todos": [{"id": "todo-1", "title": "What notice applies?", "owner": "supervisor",
                   "status": "pending", "blocked_by": []}],
        "observations": [{"kind": "todo_created", "todo_ids": ["todo-1"]}],
    }
    return state


async def _drive(loop, state):
    """Exercise successive checkpoint boundaries as the QueryGraph self-edge does."""
    for _ in range(10):
        update = await loop.ainvoke(state)
        if update["next_node"] != "research_agent":
            return update
        state = {**state, **json.loads(json.dumps(update))}
    raise AssertionError("loop did not terminate")


def _state_without_research_todos() -> dict[str, object]:
    state = _state()
    state["research"] = {"todos": [], "observations": []}
    return state


def _state_with_research_attempt_count(value: int) -> dict[str, object]:
    state = _state()
    state["research_attempt_count"] = value
    return state


def _deps(
    actions: list[dict[str, object]] | None = None,
    *,
    gateway: ScriptedGateway | None = None,
    dispatcher: object | None = None,
) -> object:
    from agentic_rag.query.research_loop import ResearchLoopDependencies

    return ResearchLoopDependencies(
        gateway=gateway or ScriptedGateway(actions or []),
        retrieval=FakeRetrieval(),
        evidence_builder=EvidenceBuilder(),
        subagents=dispatcher,  # type: ignore[arg-type]
    )


def _recording_dispatcher() -> object:
    from agentic_rag.query.evidence_builder import EvidenceCoverageTarget
    from agentic_rag.query.subagents import DelegationResult, SubagentResult

    class RecordingDispatcher:
        async def delegate(self, items: object, context: object, **_: object) -> DelegationResult:
            selected = tuple(items)  # type: ignore[arg-type]
            packed = EvidenceBuilder().build(
                [_batch()],
                [EvidenceCoverageTarget(target_id=selected[0].id, description=selected[0].title)],
                context.scope,
                context.snapshot,
            )
            return DelegationResult(
                results=(SubagentResult(todo_id=selected[0].id, evidence=packed, batch=_batch()),),
                blocked_todo_ids=(),
                child_states=(),
            )

    return RecordingDispatcher()


async def test_empty_research_state_creates_explicit_plan_and_can_delegate() -> None:
    """The model creates the real plan without a duplicate synthetic root."""
    from agentic_rag.query.research_loop import ResearchAgentLoop

    state = _state_without_research_todos()
    state["runtime_config_snapshot"]["max_research_rounds"] = 4
    loop = ResearchAgentLoop(_deps(
        actions=[
            {"action": "create_todos", "items": [{"key": "notice", "title": "Find notice period"}]},
            {"action": "delegate_research", "todo_ids": ["todo-1"]},
            {"action": "submit_evidence"},
        ],
        dispatcher=_recording_dispatcher(),
    ))  # type: ignore[arg-type]

    result = await _drive(loop, state)  # type: ignore[arg-type]

    assert [todo["id"] for todo in result["research"]["todos"]] == ["todo-1"]  # type: ignore[index]
    assert result["research"]["observations"][0]["kind"] == "todo_created"  # type: ignore[index]
    assert result["research"]["observations"][1]["kind"] == "delegate"  # type: ignore[index]


async def test_research_attempt_count_survives_reentry_and_stops_at_snapshot_limit() -> None:
    """A second graph entry cannot give one run a third model action."""
    from agentic_rag.query.research_loop import ResearchAgentLoop

    first = _state_with_research_attempt_count(1)
    initial_gateway = ScriptedGateway(
        [{"action": "calculator", "todo_id": "todo-1", "expression": "1+1"}] * 3
    )
    update = await ResearchAgentLoop(_deps(gateway=initial_gateway)).ainvoke(  # type: ignore[arg-type]
        first
    )

    assert update["research_attempt_count"] == 2
    assert initial_gateway.calls == 1
    resumed = {**first, **update}
    gateway = ScriptedGateway([])
    limited = await ResearchAgentLoop(_deps(gateway=gateway)).ainvoke(resumed)  # type: ignore[arg-type]

    assert limited["termination_reason"] == "research_round_limit"
    assert gateway.calls == 0


async def test_agent_reenters_after_retrieval_observation_and_submits_evidence() -> None:
    from agentic_rag.query.research_loop import ResearchAgentLoop, ResearchLoopDependencies

    gateway = ScriptedGateway([
        {"action": "retrieve_evidence", "todo_id": "todo-1", "query": "notice"},
        {"action": "submit_evidence"},
    ])
    loop = ResearchAgentLoop(ResearchLoopDependencies(
        gateway=gateway, retrieval=FakeRetrieval(), evidence_builder=EvidenceBuilder()
    ))

    result = await _drive(loop, _state())

    assert gateway.calls == 2
    assert result["research"]["submitted"] is True
    assert result["research"]["observations"][0]["kind"] == "todo_created"
    assert result["research"]["observations"][1]["kind"] == "retrieval"
    assert json.loads(json.dumps(result)) == result


async def test_unknown_model_action_fails_closed_without_arbitrary_tool_execution() -> None:
    from agentic_rag.query.research_loop import ResearchAgentLoop, ResearchLoopDependencies

    loop = ResearchAgentLoop(ResearchLoopDependencies(
        gateway=ScriptedGateway([{"action": "search_memory", "query": "secret"}]),
        retrieval=FakeRetrieval(), evidence_builder=EvidenceBuilder(),
    ))

    result = await _drive(loop, _state())

    assert result["research"]["cannot_answer"] is True
    assert result["termination_reason"] == "research_action_invalid"
    assert result["research"]["observations"][-1] == {
        "kind": "cannot_answer",
        "reason": "research_action_invalid",
        "error_code": "research_action_invalid",
        "retryable": False,
        "attempt": 1,
    }


@pytest.mark.parametrize(
    ("failure", "expected_code"),
    [
        (TimeoutError("Authorization: Bearer model-secret provider_response=https://private"), "model_unavailable"),
        (OSError("Authorization: Bearer model-secret provider_response=https://private"), "model_unavailable"),
    ],
)
async def test_gateway_exception_text_never_enters_research_state(
    failure: Exception,
    expected_code: str,
) -> None:
    from agentic_rag.query.research_loop import ResearchAgentLoop, ResearchLoopDependencies

    class FailingGateway:
        async def complete_structured(self, call: object, schema: type[object]) -> object:
            del call, schema
            raise failure

    loop = ResearchAgentLoop(ResearchLoopDependencies(
        gateway=FailingGateway(), retrieval=FakeRetrieval(), evidence_builder=EvidenceBuilder()
    ))

    result = await _drive(loop, _state())
    serialized = json.dumps(result)

    assert expected_code in serialized
    assert "model-secret" not in serialized
    assert "provider_response" not in serialized
    assert "https://private" not in serialized


async def test_retrieval_exception_text_is_replaced_by_controlled_metadata() -> None:
    from agentic_rag.query.research_loop import ResearchAgentLoop, ResearchLoopDependencies

    class FailingRetrieval:
        async def retrieve(self, request: object, scope: object, snapshot: object) -> EvidenceBatch:
            del request, scope, snapshot
            raise OSError("Authorization: Bearer retrieval-secret url=https://private")

    loop = ResearchAgentLoop(ResearchLoopDependencies(
        gateway=ScriptedGateway([
            {"action": "retrieve_evidence", "todo_id": "todo-1", "query": "notice"},
            {"action": "cannot_answer", "reason": "insufficient_verified_evidence"},
        ]),
        retrieval=FailingRetrieval(),
        evidence_builder=EvidenceBuilder(),
    ))

    result = await _drive(loop, _state())
    serialized = json.dumps(result)

    assert result["research"]["observations"][1] == {
        "kind": "retrieval",
        "ok": False,
        "error_code": "retrieval_unavailable",
        "retryable": True,
        "attempt": 1,
    }
    assert "retrieval-secret" not in serialized
    assert "https://private" not in serialized


async def test_gateway_repairs_action_specific_schema_before_loop_executes() -> None:
    from agentic_rag.query.research_loop import ResearchAgentLoop, ResearchLoopDependencies

    client = RepairingClient([
        '{"action":"retrieve_evidence"}',
        '{"action":"retrieve_evidence","todo_id":"todo-1","query":"notice"}',
        '{"action":"submit_evidence"}',
    ])
    gateway = ModelGateway(client, max_retries=0)
    loop = ResearchAgentLoop(ResearchLoopDependencies(
        gateway=gateway, retrieval=FakeRetrieval(), evidence_builder=EvidenceBuilder()
    ))

    result = await _drive(loop, _state())

    assert client.responses.calls == 3
    assert result["research"]["submitted"] is True
    assert result["termination_reason"] is None


async def test_loop_marks_unfinished_work_blocked_after_two_rounds() -> None:
    from agentic_rag.query.research_loop import ResearchAgentLoop, ResearchLoopDependencies

    loop = ResearchAgentLoop(ResearchLoopDependencies(
        gateway=ScriptedGateway([{"action": "calculator", "todo_id": "todo-1", "expression": "1 + 1"}] * 4),
        retrieval=FakeRetrieval(), evidence_builder=EvidenceBuilder(),
    ))

    result = await _drive(loop, _state())

    assert result["termination_reason"] == "research_round_limit"
    assert len(result["research"]["observations"]) == 3


async def test_delegate_with_empty_evidence_blocks_todo_instead_of_completing() -> None:
    from agentic_rag.query.research_loop import ResearchAgentLoop, ResearchLoopDependencies
    from agentic_rag.query.subagents import DelegationResult, SubagentResult

    empty = type(
        "EmptyEvidence",
        (),
        {
            "items": (),
            "manifest": {},
            "rendered_context": "",
            "token_count": 0,
            "index_generation": SNAPSHOT.index_generation,
        },
    )()

    class EmptyDispatcher:
        async def delegate(self, *args: object, **kwargs: object) -> DelegationResult:
            del args, kwargs
            return DelegationResult(
                results=(SubagentResult(todo_id="todo-1", evidence=empty),),
                blocked_todo_ids=(),
                child_states=(),
            )

    state = _state()
    state["research"] = {
        "todos": [{
            "id": "todo-1", "title": "Find notice", "owner": "supervisor",
            "status": "pending", "dependencies": [], "evidence_ids": [],
        }],
        "observations": [],
    }
    loop = ResearchAgentLoop(ResearchLoopDependencies(
        gateway=ScriptedGateway([
            {"action": "delegate_research", "todo_ids": ["todo-1"]},
            {"action": "cannot_answer", "reason": "insufficient_verified_evidence"},
        ]),
        retrieval=FakeRetrieval(), evidence_builder=EvidenceBuilder(), subagents=EmptyDispatcher(),
    ))

    result = await _drive(loop, state)

    assert result["research"]["todos"][0]["status"] == "blocked"
    assert result["research"]["todos"][0]["evidence_ids"] == []


async def test_delegate_cannot_complete_a_blocked_todo_from_subagent_evidence() -> None:
    """Only the reducer may transition Todo state after a delegated result arrives."""
    from agentic_rag.query.subagents import DelegationResult, SubagentResult
    from agentic_rag.query.research_loop import ResearchAgentLoop, ResearchLoopDependencies

    class UnexpectedResultDispatcher:
        async def delegate(self, *args: object, **kwargs: object) -> DelegationResult:
            del args, kwargs
            packed = EvidenceBuilder().build(
                [_batch()],
                [],
                SCOPE,
                SNAPSHOT,
            )
            return DelegationResult(
                results=(SubagentResult(todo_id="todo-1", evidence=packed),),
                blocked_todo_ids=(),
                child_states=(),
            )

    state = _state()
    state["research"] = {
        "todos": [{
            "id": "todo-1", "title": "Find notice", "owner": "supervisor",
            "status": "blocked", "dependencies": [], "evidence_ids": [],
        }],
        "observations": [],
    }
    loop = ResearchAgentLoop(ResearchLoopDependencies(
        gateway=ScriptedGateway([
            {"action": "delegate_research", "todo_ids": ["todo-1"]},
            {"action": "cannot_answer", "reason": "insufficient_verified_evidence"},
        ]),
        retrieval=FakeRetrieval(),
        evidence_builder=EvidenceBuilder(),
        subagents=UnexpectedResultDispatcher(),
    ))

    result = await _drive(loop, state)

    assert result["research"]["todos"][0]["status"] == "blocked"
    assert result["research"]["todos"][0]["evidence_ids"] == []


async def test_delegate_with_all_children_timed_out_keeps_todos_blocked() -> None:
    from agentic_rag.query.research_loop import ResearchAgentLoop, ResearchLoopDependencies
    from agentic_rag.query.subagents import DelegationResult

    class TimeoutDispatcher:
        async def delegate(self, *args: object, **kwargs: object) -> DelegationResult:
            del args, kwargs
            return DelegationResult(results=(), blocked_todo_ids=("todo-1",), child_states=())

    state = _state()
    state["research"] = {
        "todos": [{
            "id": "todo-1", "title": "Find notice", "owner": "supervisor",
            "status": "pending", "dependencies": [], "evidence_ids": [],
        }],
        "observations": [],
    }
    loop = ResearchAgentLoop(ResearchLoopDependencies(
        gateway=ScriptedGateway([
            {"action": "delegate_research", "todo_ids": ["todo-1"]},
            {"action": "cannot_answer", "reason": "insufficient_verified_evidence"},
        ]),
        retrieval=FakeRetrieval(), evidence_builder=EvidenceBuilder(), subagents=TimeoutDispatcher(),
    ))

    result = await _drive(loop, state)

    assert result["research"]["todos"][0]["status"] == "blocked"


async def test_delegate_keeps_raw_child_batches_for_unified_research_pack() -> None:
    from agentic_rag.query.research_loop import ResearchAgentLoop, ResearchLoopDependencies
    from agentic_rag.query.subagents import DelegationResult, SubagentResult

    class Dispatcher:
        async def delegate(self, *args: object, **kwargs: object) -> DelegationResult:
            del args, kwargs
            packed = EvidenceBuilder().build([_batch()], [], SCOPE, SNAPSHOT)
            return DelegationResult(
                results=(SubagentResult(todo_id="todo-1", evidence=packed, batch=_batch()),),
                blocked_todo_ids=(),
                child_states=(),
            )

    state = _state()
    state["research"] = {
        "todos": [{
            "id": "todo-1", "title": "Find notice", "owner": "supervisor",
            "status": "pending", "dependencies": [], "evidence_ids": [],
        }],
        "observations": [],
    }
    loop = ResearchAgentLoop(ResearchLoopDependencies(
        gateway=ScriptedGateway([
            {"action": "delegate_research", "todo_ids": ["todo-1"]},
            {"action": "submit_evidence"},
        ]),
        retrieval=FakeRetrieval(), evidence_builder=EvidenceBuilder(), subagents=Dispatcher(),
    ))

    result = await _drive(loop, state)

    assert result["retrieval_batches"]
    assert result["retrieval_batches"][0]["query"] == "notice"
    assert result["packed_context"]["items"]


async def test_second_supervisor_action_sees_delegated_evidence() -> None:
    """Delegation updates the staged pack before the supervisor is re-entered."""
    from agentic_rag.query.evidence_builder import EvidenceCoverageTarget
    from agentic_rag.query.research_loop import ResearchAgentLoop, ResearchLoopDependencies
    from agentic_rag.query.subagents import DelegationResult, SubagentResult

    @dataclass
    class CapturingGateway(ScriptedGateway):
        prompts: list[str] = field(default_factory=list)

        async def complete_structured(self, call: object, schema: type[object]) -> ModelResponse[object]:
            messages = getattr(call, "messages", ())
            if messages and isinstance(messages[-1], dict):
                self.prompts.append(str(messages[-1].get("content", "")))
            return await super().complete_structured(call, schema)

    class Dispatcher:
        async def delegate(self, *args: object, **kwargs: object) -> DelegationResult:
            del kwargs
            selected = tuple(args[0])
            batch = _batch().model_copy(update={"target_ids": (selected[0].id,)})
            packed = EvidenceBuilder().build(
                [batch],
                [EvidenceCoverageTarget(target_id=selected[0].id, description=selected[0].title)],
                SCOPE,
                SNAPSHOT,
            )
            return DelegationResult(
                results=(SubagentResult(todo_id=selected[0].id, evidence=packed, batch=batch),),
                blocked_todo_ids=(),
                child_states=(),
            )

    state = _state()
    state["research"] = {
        "todos": [{
            "id": "todo-1", "title": "Find notice", "owner": "supervisor",
            "status": "pending", "dependencies": [], "evidence_ids": [],
        }],
        "observations": [],
    }
    gateway = CapturingGateway(actions=[
        {"action": "delegate_research", "todo_ids": ["todo-1"]},
        {"action": "submit_evidence"},
    ])
    agent = ResearchAgentLoop(ResearchLoopDependencies(
        gateway=gateway,
        retrieval=FakeRetrieval(),
        evidence_builder=EvidenceBuilder(),
        subagents=Dispatcher(),
    ))
    result = await _drive(agent, state)

    assert len(gateway.prompts) == 2
    assert "Notice is thirty days." in gateway.prompts[1]
    evidence_id = result["evidence"][0]["evidence_id"]
    assert evidence_id in gateway.prompts[1]
    assert result["retrieval_batches"]


async def test_invalid_checkpointed_pack_terminates_without_an_answer() -> None:
    from agentic_rag.query.research_loop import ResearchAgentLoop

    state = _state()
    state["packed_context"] = {
        "items": [],
        "manifest": {},
        "rendered_context": "",
        "token_count": 0,
        "index_generation": "stale-index",
    }
    gateway = ScriptedGateway(actions=[])

    result = await ResearchAgentLoop(_deps(gateway=gateway)).ainvoke(state)

    assert result["research"]["cannot_answer"] is True
    assert result["termination_reason"] == "cannot_answer"
    assert result["next_node"] == "end"
    assert gateway.calls == 0
    assert result["research"]["observations"][-1]["error_code"] == "research_evidence_invalid"


async def test_over_budget_checkpointed_pack_terminates_without_an_answer() -> None:
    from agentic_rag.query.research_loop import ResearchAgentLoop

    state = _state()
    state["packed_context"] = {
        "items": [],
        "manifest": {},
        "rendered_context": "",
        "token_count": SNAPSHOT.max_evidence_tokens + 1,
        "index_generation": SNAPSHOT.index_generation,
    }

    result = await ResearchAgentLoop(_deps(gateway=ScriptedGateway(actions=[]))).ainvoke(state)

    assert result["research"]["cannot_answer"] is True
    assert result["termination_reason"] == "cannot_answer"


def _large_batch(number: int) -> EvidenceBatch:
    parents = []
    for index in range(6):
        content = (f"Document {number}, clause {index}: notice and obligations. " * 60)[:1200]
        hit = ChildHit(
            child_id=f"child-{number}-{index}", parent_id=f"parent-{number}-{index}",
            user_id=SCOPE.user_id, document_id=f"doc-{number}",
            document_version_id=f"version-{number}", content=content,
            ast_locator=f"#/text/{index}", lane="dense", lane_rank=index + 1, score=1.0,
        )
        parents.append(ParentEvidence(
            parent_id=hit.parent_id, document_id=hit.document_id,
            document_version_id=hit.document_version_id, content=content,
            child_hits=(hit,), rerank_score=1.0,
        ))
    return EvidenceBatch(query=f"document {number}", parents=tuple(parents), target_ids=(f"todo-{number}",))


@pytest.mark.parametrize("mode", ["direct", "delegate_after_direct", "parallel_delegate"])
async def test_legal_evidence_accumulation_is_repacked_before_budget_check(mode: str) -> None:
    from agentic_rag.query.evidence_builder import EvidenceCoverageTarget
    from agentic_rag.query.research_loop import ResearchAgentLoop, ResearchLoopDependencies
    from agentic_rag.query.subagents import DelegationResult, SubagentResult

    batches = [_large_batch(1), _large_batch(2)]
    packs = [EvidenceBuilder().build(
        [batch], [EvidenceCoverageTarget(target_id=f"todo-{i}", description=batch.query)], SCOPE, SNAPSHOT,
    ) for i, batch in enumerate(batches, 1)]
    assert [len(pack.items) for pack in packs] == [6, 6]
    assert all(pack.token_count < 12000 for pack in packs)
    assert sum(pack.token_count for pack in packs) > 12000

    class Retrieval:
        async def retrieve(self, request, scope, snapshot):
            return batches[int(request.query[-1]) - 1]

    class Dispatcher:
        async def delegate(self, selected, context, **kwargs):
            return DelegationResult(results=tuple(
                SubagentResult(todo_id=todo.id, evidence=packs[int(todo.id[-1]) - 1],
                               batch=batches[int(todo.id[-1]) - 1]) for todo in selected
            ), blocked_todo_ids=(), child_states=())

    actions = ([{"action": "delegate_research", "todo_ids": ["todo-1", "todo-2"]}]
               if mode == "parallel_delegate" else [
                   {"action": "retrieve_evidence", "todo_id": "todo-1", "query": "document 1"},
                   {"action": "retrieve_evidence", "todo_id": "todo-2", "query": "document 2"}
                   if mode == "direct" else {"action": "delegate_research", "todo_ids": ["todo-2"]},
               ])
    state = _state()
    state["research"]["todos"].append({
        "id": "todo-2", "title": "Compare obligations", "owner": "supervisor",
        "status": "pending", "blocked_by": [],
    })
    gateway = ScriptedGateway(actions)
    deps = ResearchLoopDependencies(
        gateway=gateway, retrieval=Retrieval(), evidence_builder=EvidenceBuilder(), subagents=Dispatcher(),
    )
    for _ in actions:
        # Deserialize each checkpoint and recreate the loop to cover recovery boundaries.
        result = await ResearchAgentLoop(deps).ainvoke(state)
        assert result["research"]["cannot_answer"] is False
        assert result["next_node"] == "research_agent"
        state = {**state, **json.loads(json.dumps(result))}
    assert len(result["retrieval_batches"]) == 2
    packed = result["packed_context"]
    assert 0 < packed["token_count"] <= 12000
    assert packed["token_count"] == len(packed["rendered_context"])
    assert {item["document_id"] for item in packed["items"]} == {"doc-1", "doc-2"}
    assert set(packed["manifest"]) == {item["evidence_id"] for item in result["evidence"]}
    for todo in result["research"]["todos"]:
        assert set(todo["evidence_ids"]).issubset(packed["manifest"])


@pytest.mark.parametrize("budget", [2000, 12000])
async def test_retrieving_the_same_source_after_cropping_remains_valid(budget: int) -> None:
    from agentic_rag.query.research_loop import ResearchAgentLoop, ResearchLoopDependencies

    batches = []
    for number in (1, 2):
        batch = _large_batch(number)
        parents = []
        for index, parent in enumerate(batch.parents[:1] if budget == 2000 else batch.parents):
            document_id = f"doc-{number}-{index}"
            parents.append(parent.model_copy(update={
                "document_id": document_id,
                "child_hits": tuple(hit.model_copy(update={"document_id": document_id})
                                    for hit in parent.child_hits),
            }))
        batches.append(batch.model_copy(update={"parents": tuple(parents)}))

    class Retrieval:
        async def retrieve(self, request, scope, snapshot):
            return batches[int(request.query[-1]) - 1]

    snapshot = SNAPSHOT.model_copy(update={"max_evidence_tokens": budget, "max_research_rounds": 4})
    state = new_query_state(run_id="crop-run", question="Compare sources", scope=SCOPE, snapshot=snapshot)
    state["research"] = {"todos": [
        {"id": f"todo-{n}", "title": f"Inspect source {n}", "owner": "supervisor",
         "status": "pending", "blocked_by": []} for n in (1, 2, 3)
    ]}
    loop = ResearchAgentLoop(ResearchLoopDependencies(
        gateway=ScriptedGateway([
            {"action": "retrieve_evidence", "todo_id": f"todo-{n}", "query": f"source {n}"}
            for n in (1, 2, 3)
        ]), retrieval=Retrieval(), evidence_builder=EvidenceBuilder(),
    ))
    for _ in range(2):
        result = await loop.ainvoke(state)
        assert result["research"]["cannot_answer"] is False
        state = {**state, **json.loads(json.dumps(result))}
    cropped = next(item for item in result["evidence"] if len(item["content"]) < 1200)
    original = next(parent for batch in batches for parent in batch.parents
                    if parent.parent_id == cropped["parent_id"])
    batches.append(EvidenceBatch(query="repeat source", parents=(original,)))
    result = await loop.ainvoke(state)
    assert result["research"]["cannot_answer"] is False
    assert result["next_node"] == "research_agent"
    assert len(result["retrieval_batches"]) == 3
    assert result["packed_context"]["token_count"] <= budget


@pytest.mark.parametrize("damage", ["content", "document", "heading"])
async def test_repacking_rejects_conflicting_raw_source_versions(damage: str) -> None:
    from agentic_rag.query.research_loop import ResearchAgentLoop, ResearchLoopDependencies

    batch = _batch()
    parent = batch.parents[0]
    field, value = {"content": ("content", "Changed original"), "document": ("document_id", "wrong-doc"),
                    "heading": ("heading_path", ("wrong-heading",))}[damage]
    changed = parent.model_copy(update={field: value})
    changed = changed.model_copy(update={"child_hits": tuple(hit.model_copy(update={
        "content": changed.content, "document_id": changed.document_id,
    }) for hit in changed.child_hits)})
    batches = [batch, batch.model_copy(update={"parents": (changed,)})]

    class Retrieval:
        async def retrieve(self, request, scope, snapshot):
            return batches[int(request.query[-1]) - 1]

    state = _state()
    state["research"]["todos"].append({
        "id": "todo-2", "title": "Recheck", "owner": "supervisor", "status": "pending", "blocked_by": [],
    })
    loop = ResearchAgentLoop(ResearchLoopDependencies(
        gateway=ScriptedGateway([
            {"action": "retrieve_evidence", "todo_id": f"todo-{n}", "query": f"source {n}"} for n in (1, 2)
        ]), retrieval=Retrieval(), evidence_builder=EvidenceBuilder(),
    ))
    result = await loop.ainvoke(state)
    result = await loop.ainvoke({**state, **json.loads(json.dumps(result))})
    assert result["research"]["cannot_answer"] is True
    assert result["research"]["observations"][-1]["error_code"] == "research_evidence_invalid"


async def test_unbacked_checkpoint_cannot_claim_a_different_content_is_just_a_crop() -> None:
    from agentic_rag.query.research_loop import ResearchAgentLoop
    from agentic_rag.safety.context import DataEnvelope

    packed = EvidenceBuilder().build([_batch()], [], SCOPE, SNAPSHOT)
    item = packed.items[0].model_copy(update={"content": "Different unverified content"})
    rendered = DataEnvelope(source_label=f"document:{item.document_id}", evidence_id=item.evidence_id,
                            content=item.content, heading_path=item.heading_path).render()
    state = _state()
    state["packed_context"] = packed.model_copy(update={
        "items": (item,), "rendered_context": rendered, "token_count": len(rendered),
    }).model_dump(mode="json")
    loop = ResearchAgentLoop(_deps(actions=[
        {"action": "retrieve_evidence", "todo_id": "todo-1", "query": "notice"},
    ]))
    result = await loop.ainvoke(state)
    assert result["research"]["cannot_answer"] is True
    assert result["research"]["observations"][-1]["error_code"] == "research_evidence_invalid"


@pytest.mark.parametrize("damage", ["index", "manifest", "rendered", "over_budget", "missing_batch", "locator"])
async def test_repacking_does_not_hide_invalid_delegated_evidence(damage: str) -> None:
    from agentic_rag.query.research_loop import ResearchAgentLoop, ResearchLoopDependencies
    from agentic_rag.query.subagents import DelegationResult, SubagentResult

    batch = _batch()
    packed = EvidenceBuilder().build([batch], [], SCOPE, SNAPSHOT)
    if damage == "index":
        packed = packed.model_copy(update={"index_generation": "stale"})
    elif damage == "manifest":
        packed = packed.model_copy(update={"manifest": {}})
    elif damage == "rendered":
        packed = packed.model_copy(update={"rendered_context": "forged", "token_count": 6})
    elif damage == "over_budget":
        packed = packed.model_copy(update={"token_count": 12001})
    elif damage == "locator":
        item = packed.items[0].model_copy(update={"ast_locator": "#/not-in-raw-batch"})
        packed = packed.model_copy(update={"items": (item,), "manifest": {
            item.evidence_id: packed.manifest[item.evidence_id].model_copy(update={"ast_locator": item.ast_locator}),
        }})

    class Dispatcher:
        async def delegate(self, *args, **kwargs):
            return DelegationResult(results=(SubagentResult(
                todo_id="todo-1", evidence=packed, batch=None if damage == "missing_batch" else batch,
            ),), blocked_todo_ids=(), child_states=())

    loop = ResearchAgentLoop(ResearchLoopDependencies(
        gateway=ScriptedGateway([{"action": "delegate_research", "todo_ids": ["todo-1"]}]),
        retrieval=FakeRetrieval(), evidence_builder=EvidenceBuilder(), subagents=Dispatcher(),
    ))
    result = await loop.ainvoke(_state())
    assert result["research"]["cannot_answer"] is True
    assert result["next_node"] == "end"
    expected = "research_batches_missing" if damage == "missing_batch" else "research_evidence_invalid"
    assert result["research"]["observations"][-1]["error_code"] == expected


async def test_loop_propagates_cancellation_from_tool() -> None:
    from agentic_rag.query.research_loop import ResearchAgentLoop, ResearchLoopDependencies

    class CancellingRetrieval:
        async def retrieve(self, request: object, scope: object, snapshot: object) -> EvidenceBatch:
            del request, scope, snapshot
            raise asyncio.CancelledError()

    loop = ResearchAgentLoop(ResearchLoopDependencies(
        gateway=ScriptedGateway([{"action": "retrieve_evidence", "todo_id": "todo-1", "query": "notice"}]),
        retrieval=CancellingRetrieval(), evidence_builder=EvidenceBuilder(),
    ))

    with pytest.raises(asyncio.CancelledError):
        await _drive(loop, _state())


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
    assert built["transition_contract"] == {
        "available_todo_ids": ["open"],
        "known_evidence_ids": ["e1"],
        "unresolved_todos_remain": True,
        "submit_evidence_valid": False,
        "ready_todo_ids": ["open"],
        "waiting_todo_ids": [],
        "upstream_blocked_todo_ids": [],
        "blocked_todo_ids": [],
        "retryable_todo_ids": [],
        "plan_required": False,
    }
    assert built["compacted_history"] == "older work compacted"
    assert len(compactor.calls) == 1


async def test_context_builder_preserves_bounded_evidence_text_for_research() -> None:
    from agentic_rag.query.context import ContextBuilder

    state = _state()
    state["packed_context"] = {
        "manifest": {"e1": {"document_id": "doc-1"}},
        "rendered_context": "[e1] verified evidence text",
    }

    built = await ContextBuilder().build(state)

    assert built["packed_context"] == "[e1] verified evidence text"
