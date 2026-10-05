from types import SimpleNamespace

import pytest


def test_graph_snapshot_declares_variant_and_disables_memory_without_changing_budget():
    from evals.paired_graph_runtime import arm_snapshot
    from tests.unit.evals.test_collector import SNAPSHOT
    baseline = arm_snapshot(SNAPSHOT, "ik", "a" * 64, "b" * 64)
    candidate = arm_snapshot(SNAPSHOT, "ik_max_word", "a" * 64, "b" * 64)
    assert baseline.evaluation.memory_policy == candidate.evaluation.memory_policy == "disabled"
    left = baseline.model_dump(exclude={"evaluation", "retrieval_config_version"})
    right = candidate.model_dump(exclude={"evaluation", "retrieval_config_version"})
    assert left == right
    assert candidate.snapshot_id != baseline.snapshot_id


async def test_actual_event_payload_and_provider_usage_are_saved_locally(tmp_path):
    from evals.paired_graph_runtime import LocalEvents
    from agentic_rag.observability.logging import AgentEventEmitter
    from agentic_rag.persistence.artifacts import LocalArtifactStore
    artifacts = LocalArtifactStore(tmp_path / "artifacts")
    events = LocalEvents(artifacts)
    events.begin(tmp_path / "events.json")
    emitter = AgentEventEmitter(events, artifacts, runtime_config_snapshot_id="snapshot")
    await emitter.emit(run_id="run", user_id="u", event_type="PROVIDER_REQUEST_STARTED",
                       attributes={"provider_request_id": "real", "operation": "llm"})
    assert events.records[0]["attributes"]["provider_request_id"] == "real"
    assert events.records[0]["run_id"] == "run"
    assert (tmp_path / "events.json").exists()


def test_graph_projection_uses_audited_actual_answer_and_retains_failure():
    from evals.paired_graph_runtime import project_graph_result
    from tests.unit.evals.test_collector import observed
    _, state = observed()
    result = project_graph_result(state, error_code=None)
    assert result["answer"] == "answer"
    assert result["client_provenance"] == "real_query_graph"
    failed = project_graph_result(state, error_code="TimeoutError")
    assert failed["outcome"] == "TimeoutError" and failed["answer"] == ""


async def test_request_guard_persists_before_send_and_enforces_shared_per_query_limit(tmp_path):
    from evals.paired_graph_runtime import ProviderBudget
    from evals.gold_rebinding import _json
    calls = []
    path = tmp_path / "provider-budget.json"
    async def send(**kwargs):
        assert _json(path)[0]["status"] == "running"
        calls.append(kwargs)
        return SimpleNamespace(usage=None)
    resource = SimpleNamespace(create=send)
    budget = ProviderBudget(limit=1)
    budget.wrap(resource, "llm")
    budget.begin(path)
    await resource.create(model="actual-model")
    with pytest.raises(RuntimeError, match="budget"):
        await resource.create(model="actual-model")
    assert len(calls) == 1


async def test_external_tool_boundary_never_connects_and_memory_is_fail_closed(tmp_path):
    from evals.paired_graph_runtime import BlockedExternalAdapter, DisabledMemory
    from agentic_rag.tool_runtime.models import ToolError
    adapter = SimpleNamespace(adapter_id="mcp.maps", config=None)
    blocked = BlockedExternalAdapter(adapter)
    with pytest.raises(ToolError):
        await blocked.list_tools(None)
    with pytest.raises(RuntimeError, match="memory"):
        await DisabledMemory().retrieve(None)


def test_sql_boundary_allows_reads_and_rejects_writes():
    from sqlalchemy import create_engine, text
    from evals.paired_graph_runtime import protect_sql_reads
    engine = create_engine("sqlite://")
    protect_sql_reads(engine)
    with engine.connect() as conn:
        assert conn.execute(text("SELECT 1")).scalar() == 1
        with pytest.raises(ValueError, match="read-only"):
            conn.execute(text("CREATE TABLE accidental_write(id INT)"))
    engine.dispose()


async def test_real_graph_entry_skips_memory_and_saves_actual_streamed_state(tmp_path):
    from dataclasses import replace
    from agentic_rag.observability.logging import AgentEventEmitter
    from agentic_rag.persistence.artifacts import LocalArtifactStore
    from evals.paired_graph_runtime import arm_snapshot, execute_graph, LocalEvents, ProviderBudget, DisabledMemory
    from evals.paired_graph import GraphSearchClient
    from tests.unit.evals.test_paired_bm25 import SearchES
    from tests.unit.query.test_graph import _deps, SNAPSHOT, SCOPE
    snapshot = arm_snapshot(SNAPSHOT, "ik", "a" * 64, "b" * 64)
    deps, memory, retrieval, _ = _deps(route="chat")
    deps.gateway.answer_values[:] = [{"text": "actual chat output"}]
    artifacts = LocalArtifactStore(tmp_path / "artifacts")
    events = LocalEvents(artifacts)
    emitter = AgentEventEmitter(events, artifacts, runtime_config_snapshot_id=SNAPSHOT.snapshot_id)
    deps = replace(deps, memory=DisabledMemory(), event_repository=events, event_emitter=emitter)
    transport = GraphSearchClient(SearchES(), index="children", user_id=SCOPE.user_id, generation="index-v1")
    result = await execute_graph(deps=deps, events=events, budget=ProviderBudget(), transport=transport,
        snapshot=snapshot, arm="ik", question="hello", user_id=SCOPE.user_id, directory=tmp_path / "case")
    assert result["answer"] == "actual chat output"
    assert result["outcome"] == "completed"
    assert result["memory_policy"] == "disabled"
    assert retrieval.calls == 0 and memory.loads == 0 and not memory.stored
    assert (tmp_path / "case" / "state.json").exists()


@pytest.mark.parametrize("reason", ["router_unavailable", "retrieval_unavailable", "evidence_grader_unavailable"])
async def test_controlled_technical_failure_is_not_judged_as_model_answer(tmp_path, reason):
    from agentic_rag.query.chat import run_chat
    from agentic_rag.query.routing_policy import technical_failure
    from evals.paired_graph_runtime import project_graph_result
    from evals.formal_command import score_case
    from evals.gold_v2_models import GoldCaseV2
    from tests.unit.evals.test_collector import observed
    from tests.unit.evals.test_gold_v2_validation import gold_payload
    _, state = observed()
    state.update(route={"route": "chat"}, response_mode="technical_error",
        policy_decision=technical_failure(reason).model_dump(mode="json"), errors=[{"code": reason}])
    state.update(await run_chat(state, None))
    trace = project_graph_result(state, error_code=None)
    assert trace["outcome"] == reason
    assert trace["terminal_status"] == "cannot_answer"
    assert trace["answer_origin"] == "controlled_response"
    assert "服务暂时不可用" in trace["answer"]
    class NoJudge:
        metadata_v2 = {}
        def __getattr__(self, name):
            pytest.fail("outage message must not incur judge calls: " + name)
    row = await score_case(GoldCaseV2.model_validate(gold_payload()), trace, NoJudge(), tmp_path, "binding")
    assert row["failure_class"] == "system_failure"
    assert row["task_success"]["value"] == 0


def test_research_limit_with_retrieved_but_unaudited_evidence_is_a_valid_terminal():
    from evals.paired_graph_runtime import project_graph_result
    from tests.unit.evals.test_collector import observed
    _, state = observed()
    state.update(answer={}, termination_reason="research_round_limit", evidence=[{"parent_id": "p1"}])
    trace = project_graph_result(state, error_code=None)
    assert trace["outcome"] == "research_round_limit"
    assert trace["public_answer"]["evidence_parent_ids"] == []
    assert trace["context_items"][0]["parent_id"] == "p1"
    assert trace["answer"] == ""
