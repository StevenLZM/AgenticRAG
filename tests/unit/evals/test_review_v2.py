import pytest

from evals.gold_v2_models import GoldCaseV2
from tests.unit.evals.test_gold_v2_validation import gold_payload
from evals.review_v2 import gold_review_queue, calibration_queue, audit_gold, audit_calibration


def case():
    return GoldCaseV2.model_validate(gold_payload())


def test_human_review_is_pending_and_bound_to_gold_content():
    rows = gold_review_queue([case()])
    assert audit_gold([case()], rows)["approved"] == 0
    rows[0].update(reviewer="human-reviewer", reviewed_at="2026-10-01T10:00:00+08:00", decision="approve", reason="checked original facts and sufficient chunk sets")
    assert audit_gold([case()], rows)["sample_review_complete"] is True
    rows[0]["case_sha256"] = "tampered"
    with pytest.raises(ValueError, match="binding"):
        audit_gold([case()], rows)


def test_calibration_export_is_blind_and_pending_never_passes():
    c = case()
    results = [{"case_id": c.case_id, "answer": "answer", "verdict": {"fully_correct": True, "contradictions": [], "missing_facts": []}}]
    queue = calibration_queue([c], results, "experiment")
    assert "verdict" not in queue[0] and "task_success" not in queue[0]
    assert queue[0]["human_fully_correct"] is None
    assert audit_calibration([c], results, queue, "experiment")["calibration_sample_complete"] is False
    queue[0].update(reviewer="person", reviewed_at="2026-10-01T10:00:00+08:00", human_fully_correct=False, reason="wrong fact")
    result = audit_calibration([c], results, queue, "experiment")
    assert result["agreement"] == 0 and result["false_positive"] == 1
    assert result["calibration_sample_complete"] is False  # fewer than 100 samples
    results[0]["answer"] = "changed"
    with pytest.raises(ValueError, match="binding"):
        audit_calibration([c], results, queue, "experiment")


def test_calibration_must_not_use_held_out_test_to_tune_judge():
    c = case().model_copy(update={"split": "test"})
    with pytest.raises(ValueError, match="held-out"):
        calibration_queue([c], [{"case_id": c.case_id, "answer": "answer"}], "experiment")
