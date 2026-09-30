"""Contract-only verification checks; never live evaluation receipts."""
from types import SimpleNamespace

import pytest

from evals.verification import verify_projection


def test_external_json_cannot_self_attest_real_quality():
    from scripts.verify_acceptance import verify_acceptance
    forged = dict(user_leak_count=0, citation_coverage=1.0, unaudited_answer_count=0,
                  recovery_drill_passed=True, backup_restore_passed=True,
                  evaluation_mode="api", client_provenance="real_query_api", real_query_count=24)
    assert verify_acceptance(forged) == 1


@pytest.mark.asyncio
async def test_saved_verifier_rejects_changed_frozen_gold_before_network(tmp_path):
    import json
    from evals.corpus import build_corpus
    from evals.verification import verify_saved_evaluation
    build_corpus(tmp_path / "corpus")
    manifest = tmp_path / "corpus/manifest.json"
    value = json.loads(manifest.read_text())
    value["cases"][0]["reference_answer"] = "changed gold"
    manifest.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="frozen"):
        await verify_saved_evaluation(tmp_path)


def test_verification_cli_rejects_empty_report_even_with_valid_run(tmp_path, monkeypatch):
    import evals.verification
    from scripts.verify_acceptance import main
    async def verified(_):
        return {"completion_verified": True, "completed_cases": 24, "ragas": {}, "outcome_counts": {}}
    monkeypatch.setattr(evals.verification, "verify_saved_evaluation", verified)
    path = tmp_path / "empty.json"
    path.write_text("{}")
    assert main(["--report", str(path), "--run-dir", str(tmp_path), "--quality-only"]) == 1


def test_terminal_projection_uses_verified_public_status_when_sql_null():
    from datetime import datetime, timedelta
    from evals.saved_report import project_operational
    started = datetime(2026, 9, 17)
    row = project_operational({"started_at": started, "finished_at": started + timedelta(seconds=12),
                               "termination_reason": None}, {"status": "research_round_limit"})
    assert row["terminal_outcome"] == "research_round_limit"
    assert row["elapsed_seconds"] == 12
    assert row["tokens"] is None


def test_saved_report_aggregate_tampering_is_rejected():
    from evals.saved_report import check_summary
    from evals.report import build_summary
    row = SimpleNamespace(case_id="c1", runtime_config_snapshot_id="s1",
        deterministic_metrics={"leakage": 0, "mrr": 0.5},
        ragas_metrics={"status": "available", "faithfulness": 0.5}, audited=True, citation_coverage=1.0)
    summary = build_summary([row], evaluation_mode="api", client_provenance="real_query_api")
    summary.update(real_query_count=1, requested_cases=1, query_failure_count=0, failures=[])
    check_summary([row], summary)
    summary["metrics"]["mrr"] = 1.0
    with pytest.raises(ValueError, match="aggregate"):
        check_summary([row], summary)


def test_projection_checks_scored_answer_and_context_before_counting():
    observed = {"answer": {"segments": [{"text": "actual answer"}]}, "route": "fast_rag",
                "contexts": ["actual context"], "retrieval_rounds": [], "ranking_scope": "per-retrieval"}
    row = SimpleNamespace(answer="actual answer", route="fast_rag", retrieved_contexts=("actual context",),
                          retrieval_rounds=(), ranking_scope="per-retrieval")
    verify_projection(observed, row)
    row.answer = "standard answer echoed instead"
    with pytest.raises(ValueError, match="answer"):
        verify_projection(observed, row)


def test_projection_rejects_swapped_context_order():
    observed = {"answer": {"segments": [{"text": "answer"}]}, "route": "research",
                "contexts": ["first", "second"], "retrieval_rounds": [], "ranking_scope": "per-retrieval"}
    row = SimpleNamespace(answer="answer", route="research", retrieved_contexts=("second", "first"),
                          retrieval_rounds=(), ranking_scope="per-retrieval")
    with pytest.raises(ValueError, match="context"):
        verify_projection(observed, row)
