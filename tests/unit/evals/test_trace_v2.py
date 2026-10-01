from copy import deepcopy

import pytest

from evals.trace_v2 import project_case_trace
from tests.unit.evals.test_collector import SNAPSHOT, observed


def test_child_rank_preserved_and_context_is_packed_not_topk():
    run, state = observed()
    state["retrieval_batches"][0]["query"] = "真实检索词"
    trace = state["retrieval_batches"][0]["observation"]
    trace["stages"]["dense"].append({**trace["stages"]["dense"][0], "child_id": "c2"})
    state["research_attempt_count"] = 4
    result = project_case_trace(run, state, snapshot_id=SNAPSHOT.snapshot_id)
    assert result["retrieval_rounds"][0]["stages"]["dense"] == ["c1", "c2"]
    assert result["context_items"][0]["content"] == "actual evidence"
    assert result["research_rounds"] == 4
    assert result["retrieval_calls"] == 1
    assert result["provider_usage"]["status"] == "unknown"
    assert result["ttft_seconds"] is None


def test_foreign_child_rejected_even_if_parent_is_valid():
    run, state = observed()
    state["retrieval_batches"][0]["observation"]["stages"]["dense"][0]["user_id"] = "foreign"
    with pytest.raises(ValueError):
        project_case_trace(run, state, snapshot_id=SNAPSHOT.snapshot_id)


def test_terminal_limit_not_successful_refusal():
    run, state = observed()
    state.update(termination_reason="research_round_limit", answer={})
    run["answer"] = {"status": "research_round_limit", "route": "research"}
    result = project_case_trace(run, state, snapshot_id=SNAPSHOT.snapshot_id)
    assert result["outcome"] == "research_round_limit"
    assert result["answer"] == ""
    assert result["answer_origin"] == "terminal_status"


def test_missing_context_is_unknown_not_reconstructed():
    run, state = observed()
    state.pop("packed_context")
    with pytest.raises(ValueError, match="context"):
        project_case_trace(run, state, snapshot_id=SNAPSHOT.snapshot_id)
    run["status"] = "failed"
    run["answer"] = None
    result = project_case_trace(run, state, snapshot_id=SNAPSHOT.snapshot_id)
    assert result["context_items"] is None


def test_multiple_calls_keep_separate_rankings():
    run, state = observed()
    state["retrieval_batches"].append(deepcopy(state["retrieval_batches"][0]))
    result = project_case_trace(run, state, snapshot_id=SNAPSHOT.snapshot_id)
    assert len(result["retrieval_rounds"]) == 2
    assert result["ranking_scope"] == "per-retrieval"


def test_final_answer_includes_every_display_segment():
    run, state = observed()
    segments = [{"kind": "heading", "text": "2025年标准", "evidence_ids": []},
                {"kind": "content", "text": "470元/人/晚", "evidence_ids": []},
                {"kind": "references", "text": "来源：2026版政策", "evidence_ids": []}]
    state["answer"]["segments"] = segments
    run["answer"]["segments"] = segments
    result = project_case_trace(run, state, snapshot_id=SNAPSHOT.snapshot_id)
    assert result["answer"] == "2025年标准\n470元/人/晚\n来源：2026版政策"


def test_safe_terminal_projection_matches_api_snapshot():
    run, state = observed()
    state.update(termination_reason="clarify", answer={"status": "clarify"})
    run["answer"] = {"status": "clarify", "route": "research"}
    result = project_case_trace(run, state, snapshot_id=SNAPSHOT.snapshot_id)
    assert result["public_answer"]["runtime_config_snapshot_id"] == SNAPSHOT.snapshot_id


def test_provider_failure_is_not_a_correct_evidence_refusal():
    run, state = observed()
    state.update(termination_reason="cannot_answer", answer={"status": "cannot_answer"},
                 research={"observations": [{"kind": "cannot_answer", "reason": "model_unavailable",
                                               "error_code": "model_unavailable"}]})
    run["answer"] = {"status": "cannot_answer", "route": "research"}
    result = project_case_trace(run, state, snapshot_id=SNAPSHOT.snapshot_id)
    assert result["outcome"] == "model_unavailable"
    assert result["terminal_status"] == "cannot_answer"
    assert result["answer"] == ""


def test_degraded_components_preserved_in_actual_retrieval_stages():
    run, state = observed()
    state["retrieval_batches"][0]["degraded_components"] = ["dense", "reranker"]
    result = project_case_trace(run, state, snapshot_id=SNAPSHOT.snapshot_id)
    assert result["retrieval_rounds"][0]["stage_status"]["dense"] == "failed"
    assert result["retrieval_rounds"][0]["stage_status"]["rerank"] == "fallback"
