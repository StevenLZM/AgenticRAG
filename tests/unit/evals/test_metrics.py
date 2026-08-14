"""Deterministic Phase 5 evaluation contracts.

These tests intentionally use only in-memory rows.  The evaluation package must
not call the live QueryGraph, a model provider, or an external retrieval store.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from evals.metrics import (
    aggregate_loop_metrics,
    aggregate_security_metrics,
    mrr,
    ndcg_at_k,
    recall_at_k,
)
from evals.models import EvaluationCase, load_jsonl_dataset, validate_dataset_rows


def test_retrieval_metrics_have_known_values() -> None:
    ranked = ["p3", "p1", "p2"]
    relevant = {"p1", "p2"}

    assert recall_at_k(ranked, relevant, 3) == 1.0
    assert mrr(ranked, relevant) == pytest.approx(0.5)
    assert ndcg_at_k(ranked, relevant, 3) == pytest.approx(0.6934, rel=1e-3)


@pytest.mark.parametrize(
    ("ranked", "relevant", "k"),
    [([], {"p1"}, 5), (["p1"], set(), 5), (["p1", "p1"], {"p1"}, 2), (["p1"], {"p1"}, 0)],
)
def test_retrieval_metrics_are_safe_for_empty_duplicate_and_zero_k(
    ranked: list[str], relevant: set[str], k: int
) -> None:
    assert 0.0 <= recall_at_k(ranked, relevant, k) <= 1.0
    assert 0.0 <= ndcg_at_k(ranked, relevant, k) <= 1.0
    assert 0.0 <= mrr(ranked, relevant, k=k) <= 1.0


def test_retrieval_metrics_bound_k_instead_of_using_negative_slice_semantics() -> None:
    assert recall_at_k(["p1", "p2"], {"p2"}, -1) == 0.0
    assert mrr(["p1", "p2"], {"p2"}, k=1) == 0.0
    assert ndcg_at_k(["p1", "p2"], {"p2"}, 99) == pytest.approx(1.0 / 1.5849625)


def test_event_metrics_filter_user_and_runtime_snapshot_and_dedupe_replays() -> None:
    events = [
        {
            "event_key": "retrieval-1",
            "run_id": "run-1",
            "user_id": "alice",
            "runtime_config_snapshot_id": "snap-a",
            "event_type": "RETRIEVAL_COMPLETED",
            "attributes": {"retrieval_rounds": 2},
        },
        {
            "event_key": "retrieval-1",
            "run_id": "run-1",
            "user_id": "alice",
            "runtime_config_snapshot_id": "snap-a",
            "event_type": "RETRIEVAL_COMPLETED",
            "attributes": {"retrieval_rounds": 2},
        },
        {
            "event_key": "loop-1",
            "run_id": "run-1",
            "user_id": "alice",
            "runtime_config_snapshot_id": "snap-a",
            "event_type": "RUN_COMPLETED",
            "attributes": {"termination_reason": "research_round_limit"},
        },
        {
            "event_key": "other-user",
            "run_id": "run-2",
            "user_id": "bob",
            "runtime_config_snapshot_id": "snap-a",
            "event_type": "RUN_COMPLETED",
            "attributes": {"termination_reason": "research_round_limit"},
        },
        {
            "event_key": "other-snapshot",
            "run_id": "run-3",
            "user_id": "alice",
            "runtime_config_snapshot_id": "snap-b",
            "event_type": "RUN_COMPLETED",
            "attributes": {"termination_reason": "research_round_limit"},
        },
    ]

    result = aggregate_loop_metrics(events, user_id="alice", runtime_config_snapshot_id="snap-a")

    assert result["runs_observed"] == 1
    assert result["average_retrieval_rounds"] == 2.0
    assert result["loop_limit_hit_rate"] == 1.0


def test_security_metrics_are_deterministic_and_ignore_unscoped_rows() -> None:
    events = [
        {
            "event_key": "safe",
            "run_id": "run-1",
            "user_id": "alice",
            "runtime_config_snapshot_id": "snap-a",
            "event_type": "SECURITY_CHECK",
            "attributes": {"user_leak_count": 0},
        },
        {
            "event_key": "leak",
            "run_id": "run-1",
            "user_id": "alice",
            "runtime_config_snapshot_id": "snap-a",
            "event_type": "SECURITY_VIOLATION",
            "attributes": {"user_leak_count": 1, "cross_user_evidence_count": 1},
        },
        {
            "event_key": "mixed",
            "run_id": "run-2",
            "user_id": "alice",
            "runtime_config_snapshot_id": "snap-b",
            "event_type": "SECURITY_VIOLATION",
            "attributes": {"user_leak_count": 99},
        },
    ]

    result = aggregate_security_metrics(events, user_id="alice", runtime_config_snapshot_id="snap-a")

    assert result["runs_observed"] == 1
    assert result["user_leak_count"] == 1
    assert result["cross_user_evidence_count"] == 1
    assert result["security_violation_count"] == 1


def _case(case_id: str = "case-1") -> dict[str, object]:
    return {
        "case_id": case_id,
        "user_id": "eval_user",
        "question": "Which quarter had the highest revenue?",
        "reference_answer": "The third quarter had the highest revenue.",
        "reference_parent_ids": ["parent-1"],
        "expected_route": "fast_rag",
        "tags": ["single-hop"],
        "runtime_config_snapshot_id": "snapshot-phase5-v1",
    }


def test_evaluation_case_is_strict_and_requires_reference() -> None:
    case = EvaluationCase.model_validate(_case())
    assert case.reference_parent_ids == ("parent-1",)
    with pytest.raises(ValidationError):
        EvaluationCase.model_validate({**_case(), "reference_parent_ids": []})
    with pytest.raises(ValidationError):
        EvaluationCase.model_validate({**_case(), "expected_route": "unknown"})
    with pytest.raises(ValidationError):
        EvaluationCase.model_validate({**_case(), "unknown": "rejected"})


def test_dataset_validator_rejects_duplicate_unknown_and_nonfinite_rows(tmp_path: Path) -> None:
    duplicate = [_case("case-1"), _case("case-1")]
    with pytest.raises(ValueError, match="duplicate case_id"):
        validate_dataset_rows(duplicate, dataset_name="baseline")

    bad_jsonl = tmp_path / "bad.jsonl"
    bad_jsonl.write_text(
        json.dumps({**_case(), "score": float("nan")}, allow_nan=True) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="non-finite"):
        load_jsonl_dataset(bad_jsonl, dataset_name="baseline")


def test_fixed_datasets_meet_minimum_case_counts() -> None:
    root = Path(__file__).parents[3] / "evals" / "datasets"
    assert len(load_jsonl_dataset(root / "baseline.jsonl", dataset_name="baseline")) >= 24
    assert len(load_jsonl_dataset(root / "ingestion_fidelity.jsonl", dataset_name="ingestion_fidelity")) >= 12
    assert len(load_jsonl_dataset(root / "security.jsonl", dataset_name="security")) >= 12
