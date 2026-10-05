from types import SimpleNamespace

import pytest

from evals.gold_rebinding import _write


def stored_terminal(tmp_path):
    from tests.unit.evals.test_collector import observed, SNAPSHOT
    _, state = observed()
    state.update(answer={}, termination_reason="research_round_limit", next_node="end", research_attempt_count=6,
                 evidence=[{"parent_id": "p1"}])
    for name, value in {"state": state, "query": {"status": "failed", "error_type": "ValueError"},
        "grades": [], "events": [], "provider-budget": [],
        "history": [{"elapsed_seconds": 120.0, "termination_reason": "research_round_limit", "next_node": "end"}]}.items():
        _write(tmp_path / (name + ".json"), value)
    return SimpleNamespace(question="question", user_id="u1"), SNAPSHOT


def test_recover_completed_terminal_offline_without_changing_source(tmp_path):
    from evals.paired_graph_recovery import recover_terminal_trace
    case, snapshot = stored_terminal(tmp_path)
    before = {p.name: p.read_bytes() for p in tmp_path.glob("*.json")}
    trace = recover_terminal_trace(tmp_path, case=case, snapshot=snapshot)
    assert trace["outcome"] == "research_round_limit"
    assert trace["answer"] == ""
    assert trace["research_rounds"] == 6
    assert trace["total_seconds"] == 120
    assert trace["recovery"]["new_provider_calls"] == 0
    assert {p.name: p.read_bytes() for p in tmp_path.glob("*.json")} == before


@pytest.mark.parametrize("fault", ["question", "unfinished", "snapshot"])
def test_recovery_cannot_turn_unknown_or_foreign_state_into_a_completed_query(tmp_path, fault):
    from evals.paired_graph_recovery import recover_terminal_trace
    from evals.gold_rebinding import _json
    case, snapshot = stored_terminal(tmp_path)
    state = _json(tmp_path / "state.json")
    if fault == "question":
        state["request"]["question"] = "other"
    elif fault == "unfinished":
        state["termination_reason"] = None
    else:
        state["runtime_config_snapshot"]["index_generation"] = "different"
    _write(tmp_path / "state.json", state)
    with pytest.raises(ValueError):
        recover_terminal_trace(tmp_path, case=case, snapshot=snapshot)


def continuation_source(tmp_path):
    from evals.gold_v2_models import canonical_hash
    from evals.paired_graph_command import operation_binding
    from evals.paired_graph_runtime import arm_snapshot
    from evals.gold_rebinding import _json
    from tests.unit.evals.test_collector import SNAPSHOT
    original = {"base_snapshot": SNAPSHOT.model_dump(mode="json"),
                "implementation": {"git_head": "same", "files": {"evals/collector.py": "before"}}}
    current = {**original, "implementation": {"git_head": "same", "files": {"evals/collector.py": "after"}}}
    case = SimpleNamespace(case_id="case", question="question", user_id="u1", snapshot_id="a" * 64)
    directory = tmp_path / "ik" / "cases" / "case"
    stored_terminal(directory)
    snapshot = arm_snapshot(SNAPSHOT, "ik", canonical_hash(original), case.snapshot_id)
    state = _json(directory / "state.json")
    state["runtime_config_snapshot"] = snapshot.model_dump(mode="json")
    state["retrieval_batches"][0]["observation"]["snapshot_id"] = snapshot.snapshot_id
    _write(directory / "state.json", state)
    _write(directory / "query.json", {"status": "failed", "error_type": "ValueError",
        "binding": operation_binding(canonical_hash(original), case.case_id, "ik", "query")})
    _write(tmp_path / "experiment.json", original)
    _write(tmp_path / "phases" / "phase.json", {"status": "verified"})
    return current, case


def test_continuation_is_bound_to_source_and_has_only_thirty_nine_new_queries(tmp_path):
    from evals.paired_graph_recovery import prepare_continuation
    current, case = continuation_source(tmp_path)
    before = {str(p): p.read_bytes() for p in tmp_path.rglob("*.json")}
    metadata, trace = prepare_continuation(tmp_path, current, [case])
    assert metadata["new_query_allowance"] == 39
    assert metadata["replayed_queries"] == 0
    assert trace["outcome"] == "research_round_limit"
    assert {str(p): p.read_bytes() for p in tmp_path.rglob("*.json")} == before


@pytest.mark.parametrize("fault", ["corpus", "production_code", "unverified_interval", "extra_query"])
def test_continuation_rejects_scope_implementation_and_interval_drift(tmp_path, fault):
    from evals.paired_graph_recovery import prepare_continuation
    current, case = continuation_source(tmp_path)
    if fault == "corpus":
        case.snapshot_id = "b" * 64
    elif fault == "production_code":
        current["implementation"]["files"]["src/agentic_rag/query/graph.py"] = "changed"
    elif fault == "unverified_interval":
        _write(tmp_path / "phases" / "phase.json", {"status": "running"})
    else:
        _write(tmp_path / "ik_max_word" / "cases" / "case" / "query.json", {"status": "running"})
    with pytest.raises(ValueError):
        prepare_continuation(tmp_path, current, [case])
