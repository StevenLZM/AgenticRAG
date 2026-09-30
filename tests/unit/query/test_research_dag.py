"""Real loop and dispatcher behavior with scripted model decisions."""
import json

import pytest
from pydantic import ValidationError

from agentic_rag.query.research_loop import ResearchAgentLoop, ResearchLoopDependencies, _ResearchActionSchema
from agentic_rag.query.todos import TodoItem
from agentic_rag.query.evidence_builder import EvidenceBuilder
from tests.unit.query.test_research_loop import FakeRetrieval, ScriptedGateway, _state_without_research_todos as _state


def state_with(*todos):
    state = _state()
    state["research"] = {"todos": [todo.model_dump(mode="json") for todo in todos]}
    return state


def todo(key, **changes):
    return TodoItem(id=key, title=key, owner="supervisor", **changes)


def loop(actions, retrieval=None):
    return ResearchAgentLoop(ResearchLoopDependencies(
        gateway=ScriptedGateway(actions), retrieval=retrieval or FakeRetrieval(),
        evidence_builder=EvidenceBuilder(),
    ))


async def test_one_action_per_checkpoint_and_dependency_chain():
    agent = loop([
        {"action": "create_todos", "items": [
            {"key": "a", "title": "locate"},
            {"key": "b", "title": "extract", "blocked_by": ["a"]},
        ]},
        {"action": "retrieve_evidence", "todo_id": "todo-1", "query": "notice"},
        {"action": "retrieve_evidence", "todo_id": "todo-2", "query": "dates"},
        {"action": "submit_evidence"},
    ])
    state = _state()
    state["runtime_config_snapshot"]["max_research_rounds"] = 4
    for attempt in range(1, 5):
        update = await agent.ainvoke(state)
        assert update["research_attempt_count"] == attempt
        state.update(json.loads(json.dumps(update)))
        if attempt < 4:
            assert update["next_node"] == "research_agent"
    assert update["research"]["submitted"] is True
    assert [item["status"] for item in update["research"]["todos"]] == ["completed", "completed"]
    assert len(update["retrieval_batches"]) == 2


async def test_waiting_todo_never_reaches_retrieval():
    retrieval = FakeRetrieval()
    agent = loop([{"action": "retrieve_evidence", "todo_id": "b", "query": "dates"}], retrieval)
    update = await agent.ainvoke(state_with(todo("a"), todo("b", blocked_by=("a",))))
    assert retrieval.calls == 0
    assert update["research"]["observations"][-1]["error_code"] == "todo_not_ready"


async def test_calculator_result_completes_only_selected_todo():
    update = await loop([{"action": "calculator", "todo_id": "a", "expression": "2+3"}]).ainvoke(
        state_with(todo("a"), todo("b", blocked_by=("a",)))
    )
    a, b = update["research"]["todos"]
    assert a["status"] == "completed" and a["result_ref"]
    assert b["status"] == "pending"
    assert update["research"]["results"][a["result_ref"]]["value"] == 5


@pytest.mark.parametrize("action", [
    {"action": "update_todos", "updates": [{"todo_id": "a", "status": "completed"}]},
    {"action": "retrieve_evidence", "query": "bypass"},
    {"action": "calculator", "expression": "1+1"},
])
def test_model_cannot_bypass_task_lifecycle(action):
    with pytest.raises(ValidationError):
        _ResearchActionSchema.model_validate(action)


async def test_invalid_checkpoint_fails_before_model_call():
    update = await loop([]).ainvoke(state_with(todo("b", blocked_by=("missing",))))
    assert update["research"]["cannot_answer"] is True
    assert update["research_attempt_count"] == 0


async def test_context_frontier_and_dependency_output():
    from agentic_rag.query.context import ContextBuilder
    state = state_with(todo("a", status="completed", result_ref="calc:a"),
                       todo("b", blocked_by=("a",)), todo("c", blocked_by=("b",)))
    state["research"]["results"] = {"calc:a": {"ok": True, "value": 5}}
    built = await ContextBuilder().build(state)
    contract = built["transition_contract"]
    assert contract["ready_todo_ids"] == ["b"]
    assert contract["waiting_todo_ids"] == ["c"]
    assert contract["submit_evidence_valid"] is False
    assert built["task_results"]["calc:a"]["value"] == 5


async def test_dispatcher_reads_dependency_manifest_and_calculation_per_call():
    from agentic_rag.query.subagents import SubagentDispatcher
    from agentic_rag.query.todos import TodoDependencyInput
    from agentic_rag.query.tools import ResearchContext
    from agentic_rag.runtime.concurrency import ConcurrencyManager
    from tests.unit.query.test_research_loop import _batch, SCOPE, SNAPSHOT
    pack = EvidenceBuilder().build([_batch()], [], SCOPE, SNAPSHOT)
    observed = []
    async def worker(child, tools):
        observed.append(child)
        return pack
    dispatcher = SubagentDispatcher(tools=object(), concurrency=ConcurrencyManager(), worker=worker)
    evidence_id = pack.items[0].evidence_id
    await dispatcher.delegate([todo("b", blocked_by=("a",))], ResearchContext(scope=SCOPE, snapshot=SNAPSHOT),
        resolved_todo_ids=frozenset({"a"}), packed_evidence=pack,
        dependency_inputs={"b": (TodoDependencyInput(todo_id="a", evidence_ids=(evidence_id,), result_ref="r1"),)},
        task_results={"r1": {"ok": True, "value": 5}}, memory_summary="current")
    assert "Notice is thirty days" in observed[0].dependency_context
    assert observed[0].dependency_results["r1"]["value"] == 5
    assert evidence_id in observed[0].evidence_manifest
    assert observed[0].memory_summary == "current"


async def test_graph_self_loops_real_task_chain():
    from dataclasses import replace
    from agentic_rag.query.graph import build_query_graph
    from tests.unit.query.test_graph import _deps
    from agentic_rag.query.router import RouteDecision
    deps, memory, retrieval, events = _deps(route="research", grades=["sufficient"])
    deps.gateway.route = RouteDecision(route="research", normalized_query="notice", reason_code="test")
    agent = loop([
        {"action": "create_todos", "items": [{"key": "a", "title": "notice"}]},
        {"action": "retrieve_evidence", "todo_id": "todo-1", "query": "notice"},
        {"action": "submit_evidence"},
    ], retrieval)
    state = _state()
    state["runtime_config_snapshot"]["max_research_rounds"] = 4
    result = await build_query_graph(replace(deps, research_loop=agent)).ainvoke(state)
    assert result["research_attempt_count"] == 3
    assert result["research"]["submitted"] is True


async def test_real_delegated_chain_uses_supervisor_resolved_query():
    from agentic_rag.runtime.query_composition import build_subagent_dispatcher
    from agentic_rag.runtime.concurrency import ConcurrencyManager
    from tests.unit.query.test_research_loop import SNAPSHOT
    class Recording(FakeRetrieval):
        queries = []
        async def retrieve(self, request, scope, snapshot):
            self.queries.append(request.query)
            return await super().retrieve(request, scope, snapshot)
    retrieval = Recording()
    builder = EvidenceBuilder()
    dispatcher = build_subagent_dispatcher(retrieval=retrieval, evidence_builder=builder,
        snapshot=SNAPSHOT, concurrency=ConcurrencyManager())
    agent = ResearchAgentLoop(ResearchLoopDependencies(
        gateway=ScriptedGateway([
            {"action": "delegate_research", "todo_ids": ["a"]},
            {"action": "delegate_research", "todo_ids": ["b"], "queries": {"b": "thirty day notice exceptions"}},
            {"action": "submit_evidence"},
        ]), retrieval=retrieval, evidence_builder=builder, subagents=dispatcher))
    state = state_with(todo("a"), todo("b", blocked_by=("a",)))
    state["runtime_config_snapshot"]["max_research_rounds"] = 4
    for _ in range(3):
        state.update(await agent.ainvoke(state))
    assert state["research"]["submitted"] is True
    assert retrieval.queries == ["a", "thirty day notice exceptions"]


async def test_grader_gaps_cannot_be_bypassed_by_two_submit_actions():
    agent = loop([
        {"action": "retrieve_evidence", "todo_id": "a", "query": "notice"},
        {"action": "submit_evidence"},
        {"action": "submit_evidence"},
        {"action": "submit_evidence"},
    ])
    state = state_with(todo("a"))
    state["runtime_config_snapshot"]["max_research_rounds"] = 4
    state.update(await agent.ainvoke(state))
    state.update(await agent.ainvoke(state))
    state["research"]["gaps"] = ["missing exceptions"]
    for _ in range(2):
        state.update(await agent.ainvoke(state))
        assert state["research"]["submitted"] is False
        assert state["research"]["needs_replan"] is True


async def test_child_failure_preserves_successful_sibling():
    from agentic_rag.query.subagents import SubagentDispatcher
    from agentic_rag.query.tools import ResearchContext
    from agentic_rag.runtime.concurrency import ConcurrencyManager
    from tests.unit.query.test_research_loop import _batch, SCOPE, SNAPSHOT
    pack = EvidenceBuilder().build([_batch()], [], SCOPE, SNAPSHOT)
    async def worker(child, tools):
        if child.todo_id == "bad":
            raise OSError("private failure")
        return pack
    dispatcher = SubagentDispatcher(tools=object(), concurrency=ConcurrencyManager(), worker=worker)
    result = await dispatcher.delegate([todo("good"), todo("bad")], ResearchContext(scope=SCOPE, snapshot=SNAPSHOT))
    assert [item.todo_id for item in result.results] == ["good"]
    assert result.blocked_todo_ids == ("bad",)
