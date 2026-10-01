import pytest

from evals.tuning import progressive_budgets, pareto_frontier, select_validation


def report(name, quality, latency, cost, split="dev"):
    return {"name": name, "eligible_for_tuning": True, "snapshot_verified": True, "scoring_complete": True,
        "experiment": {"split": split, "gold_sha256": "same", "corpus_snapshot_id": "same", "base_runtime_snapshot_id": "same",
                       "judge_fingerprint": "same", "case_ids": ["a", "b"], "fingerprint": name},
        "conservative_task_success_rate": quality, "latency_seconds": {"p95": latency},
        "metrics": {"answer_correctness": {"value": quality}}, "cost": {"value": cost}}


def test_progressive_not_cartesian_and_all_budgets_effective():
    base = {"dense_k": 40, "bm25_k": 40, "rrf_k": 30, "rerank_k": 10, "parent_k": 6}
    candidates = progressive_budgets("rrf", base)
    assert [b.rrf_k for b in candidates] == [30, 50, 80]
    assert all(b.dense_k == 40 and b.bm25_k == 40 for b in candidates)
    assert len(progressive_budgets("recall", base)) == 9


def test_pareto_rejects_test_leakage_or_incomparable_corpus():
    with pytest.raises(ValueError, match="test"):
        pareto_frontier([report("test", .9, 1, 1, "test")])
    changed = report("other", .9, 1, 1)
    changed["experiment"]["corpus_snapshot_id"] = "changed"
    with pytest.raises(ValueError, match="incomparable"):
        pareto_frontier([report("base", .8, 2, 2), changed])


def test_answer_quality_priority_and_unknown_cost_not_cheapest():
    reports = [report("small", .5, 1, .01), report("good", .9, 3, .2), report("waste", .9, 5, .3),
               report("unknown", .9, 1, None)]
    result = pareto_frontier(reports)
    assert {r["name"] for r in result["frontier"]} == {"small", "good"}
    assert result["ineligible"] == ["unknown"]
    assert select_validation([report("v", .9, 3, .2, "validation")])["name"] == "v"


def test_null_answer_score_is_ineligible_not_a_sort_error():
    incomplete = report("missing", .9, 1, 1)
    incomplete["metrics"]["answer_correctness"]["value"] = None
    assert pareto_frontier([incomplete])["ineligible"] == ["missing"]


def test_unreviewed_scoring_spec_report_cannot_select_parameters():
    draft = report("unreviewed", 1, 1, .1)
    draft["eligible_for_tuning"] = False
    assert pareto_frontier([draft])["ineligible"] == ["unreviewed"]


@pytest.mark.parametrize("approval", [None, "true", 1, "missing"])
def test_unknown_or_missing_tuning_approval_is_ineligible(approval):
    draft = report("unknown", 1, 1, .1)
    if approval == "missing":
        draft.pop("eligible_for_tuning")
    else:
        draft["eligible_for_tuning"] = approval
    assert pareto_frontier([draft])["ineligible"] == ["unknown"]


def test_cli_plan_only_writes_budgets_and_compare_never_publishes(tmp_path):
    import json
    from evals.tuning import main
    base = tmp_path / "base.json"
    base.write_text(json.dumps({"dense_k": 40, "bm25_k": 40, "rrf_k": 30, "rerank_k": 10, "parent_k": 6}))
    output = tmp_path / "plan"
    assert main(["plan", "--stage", "rrf", "--base", str(base), "--output-dir", str(output)]) == 0
    assert len(list(output.glob("budget-*.json"))) == 3
    report_path = tmp_path / "report.json"
    report_path.write_text(json.dumps(report("v", .9, 3, .2, "validation")))
    comparison = tmp_path / "comparison.json"
    assert main(["compare", "--split", "validation", "--reports", str(report_path), "--output", str(comparison)]) == 0
    assert json.loads(comparison.read_text())["production_parameters_changed"] is False
