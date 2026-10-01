import pytest

from evals.agent_metrics_v2 import agent_metrics
from evals.report_v2 import build_formal_summary
from tests.unit.evals.test_formal_eval_command import cases


def row(case, success=1, accuracy=.8):
    return {"case_id": case.case_id, "task_success": {"value": success}, "fields": {},
            "ragas": {"metrics": {"answer_correctness": {"status": "available", "value": accuracy}}},
            "context": {"fact_recall": .5}, "retrieval": {"union": {}}, "total_seconds": 3,
            "route": "research", "outcome": "completed", "research_rounds": 4, "retrieval_calls": 2}


def test_failures_remain_in_denominator_and_unknown_is_null():
    selected = cases()[:3]
    result = build_formal_summary(selected, [row(selected[0])], {selected[1].case_id: {"reason": "TimeoutError"}}, {})
    assert result["requested_cases"] == 3
    assert result["metrics"]["answer_correctness"]["evaluated"] == 1
    assert result["metrics"]["answer_correctness"]["total"] == 3
    assert result["conservative_task_success_rate"] == pytest.approx(1 / 3)
    assert result["metrics"]["context_recall"]["value"] is None
    assert result["pending_cases"] == 1
    assert result["production_release_accepted"] is False


def test_unscored_judge_never_turns_into_zero_or_success():
    selected = cases()[:1]
    value = row(selected[0], success=None)
    value["ragas"]["metrics"]["answer_correctness"] = {"status": "failed", "value": None}
    result = build_formal_summary(selected, [value], {}, {})
    assert result["metrics"]["answer_correctness"]["value"] is None
    assert result["metrics"]["task_success"]["value"] is None
    assert result["scoring_complete"] is False


def test_grounded_scores_and_judge_errors_have_separate_denominators():
    selected = cases()[:2]
    good, bad = row(selected[0]), row(selected[1], success=None)
    good["grounded"] = {"status": "available", "precision": 1, "recall": .5, "f1": 2 / 3}
    bad["grounded"] = {"status": "failed"}
    bad["failure_class"] = "judge_error"
    result = build_formal_summary(selected, [good, bad], {}, {})
    assert result["metrics"]["grounded_factual_precision"]["evaluated"] == 1
    assert result["metrics"]["grounded_factual_precision"]["total"] == 2
    assert result["failure_classes"] == {"judge_error": 1}
    assert result["eligible_for_tuning"] is False


def test_field_only_judge_failure_is_visible_in_scoring_failures():
    selected = cases()[:1]
    value = row(selected[0], success=None)
    value["fields"] = {"status": "failed", "reason": "unit_not_supported_by_quote", "all_critical_fields_pass": None}
    value["grounded"] = {"status": "available", "precision": 1, "recall": 1, "f1": 1}
    result = build_formal_summary(selected, [value], {}, {})
    assert result["scoring_failures"][selected[0].case_id]["critical_fields"]["reason"] == "unit_not_supported_by_quote"


@pytest.mark.parametrize("metrics", [{}, {"answer_correctness": {"status": "available", "value": .8}}])
def test_omitted_expected_metrics_are_not_scoring_complete(metrics):
    selected = cases()[:1]
    value = row(selected[0])
    value["ragas"]["metrics"] = metrics
    assert build_formal_summary(selected, [value], {}, {})["scoring_complete"] is False


def test_wall_budget_pending_separate_from_failure_and_conservative_macro():
    selected = [cases()[0], cases()[6]]
    value = row(selected[0])
    result = build_formal_summary(selected, [value], {selected[1].case_id: {"status": "pending", "reason": "wall_budget"}}, {})
    assert result["pending_cases"] == 1
    assert result["failed_cases"] == 0
    assert result["category_macro_coverage"]["task_success"] == {"evaluated_categories": 1, "total_categories": 2}
    assert result["conservative_category_task_success"] == .5


def test_dag_violations_and_rounds_are_not_retrieval_calls():
    trace = {"research_rounds": 5, "retrieval_calls": 2, "todos": [], "actions": [], "outcome": "completed",
             "todo_history": [{"todos": [{"id": "a", "status": "pending", "blocked_by": []},
                                           {"id": "b", "status": "in_progress", "blocked_by": ["a"]}]}],
             "retrieval_rounds": [{"query": "q", "target_ids": ["a"], "stages": {"rerank": ["x"]}},
                                  {"query": "q", "target_ids": ["b"], "stages": {"rerank": ["x"]}}]}
    metrics = agent_metrics(trace)
    assert metrics["dependency_violations"] == 1
    assert metrics["research_rounds"] == 5
    assert metrics["retrieval_calls"] == 2
    assert metrics["repeated_retrieval_rate"] == .5
