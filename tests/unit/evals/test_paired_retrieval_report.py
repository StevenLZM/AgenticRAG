from copy import deepcopy

import pytest

from tests.unit.evals.test_paired_retrieval_identity import identity_fixture


def records(identity, *, second=True):
    rows = []
    for arm in (["standard", "ik"] if second else ["standard"]):
        ids = ["c1"] if arm == "standard" else ["c1", "c2"]
        rows.append({"case_id": "case", "arm": arm, "attempt_id": str(len(rows) + 1), "status": "success",
            "interval_verification": "verified",
            "identity_sha256": identity["sha256"], "elapsed_seconds": 0.1,
            "hits": [{"child_id": cid, "rank": n, "score": 1.0} for n, cid in enumerate(ids, 1)]})
    return rows


VERIFIED = {"pre_corpus": True, "post_corpus": True, "pre_lexical": True, "post_lexical": True}


def test_cross_child_and_complete_fact_metrics():
    from evals.paired_retrieval_report import build_paired_report
    case, identity = identity_fixture()
    result = build_paired_report(cases=[case], records=records(identity), identity=identity, verification=VERIFIED)
    assert result["arms"]["standard"]["metrics"]["chunk_recall@40"]["mean"] == 0.5
    assert result["arms"]["ik"]["metrics"]["complete_evidence@40"]["mean"] == 1.0
    assert result["arms"]["ik"]["metrics"]["mrr@40"]["mean"] == 0.5
    assert result["pairs"][0]["changes"]["complete_evidence@40"] == "improve"
    assert result["eligible_for_tuning"] is False
    assert result["stages"]["answer_correctness"] == "not_executed"


def test_partial_pair_has_no_verified_delta_and_keeps_denominator():
    from evals.paired_retrieval_report import build_paired_report
    case, identity = identity_fixture()
    result = build_paired_report(cases=[case], records=records(identity, second=False), identity=identity, verification=VERIFIED)
    assert result["requested_requests"] == 2 and result["valid_pairs"] == 0
    assert result["pairs"][0]["deltas"] is None
    assert result["arms"]["ik"]["metrics"]["fact_recall@40"] == {"mean": None, "n": 0, "denominator": 1,
        "lower_bound": 0.0, "upper_bound": 1.0}


def test_unmapped_is_null_not_zero_and_partial_graded_labels_are_na():
    from evals.paired_retrieval_report import build_paired_report
    from evals.gold_v2_models import GoldCaseV2
    case, identity = identity_fixture()
    raw = case.model_dump(mode="json")
    raw["required_evidence_groups_all_of"][0].update(mapping_status="unmapped", child_ids_any_of=[], child_evidence_sets_any_of=[])
    case = GoldCaseV2.model_validate(raw)
    _, identity = identity_fixture(case)
    result = build_paired_report(cases=[case], records=records(identity), identity=identity, verification=VERIFIED)
    assert result["arms"]["ik"]["metrics"]["fact_recall@40"]["mean"] is None
    assert result["qrel_unmapped_cases"] == 1
    assert result["arms"]["ik"]["graded_ndcg"] == {"status": "not_available", "value": None}


def test_postflight_drift_invalidates_pairs():
    from evals.paired_retrieval_report import build_paired_report
    case, identity = identity_fixture()
    result = build_paired_report(cases=[case], records=records(identity), identity=identity,
                                verification={**VERIFIED, "post_corpus": False})
    assert result["valid_pairs"] == 0 and result["completion_verified"] is False


def test_refusal_evidence_is_not_answer_correctness_and_format_macro():
    from evals.paired_retrieval_report import build_paired_report, render_paired_report
    case, identity = identity_fixture()
    case = case.model_copy(update={"answerable": False})
    _, identity = identity_fixture(case)
    result = build_paired_report(cases=[case], records=records(identity),
                                identity=identity, verification=VERIFIED)
    assert result["pairs"][0]["interpretation"] == "refusal_evidence_recall_only"
    assert result["groups"]["format"].keys() == {"txt"}
    assert "BM25" in render_paired_report(result)
    assert "not_executed" in render_paired_report(result)


def test_short_results_do_not_shrink_qrel_denominator():
    from evals.paired_retrieval_report import build_paired_report
    case, identity = identity_fixture()
    result = build_paired_report(cases=[case], records=records(identity), identity=identity, verification=VERIFIED)
    metrics = result["arms"]["standard"]["metrics"]
    assert metrics["chunk_recall@40"]["mean"] == 0.5
    assert "chunk_recall@50" not in metrics


def test_wrong_identity_or_duplicate_results_are_rejected():
    from evals.paired_retrieval_report import build_paired_report
    case, identity = identity_fixture()
    rows = records(identity)
    with pytest.raises(ValueError):
        build_paired_report(cases=[case], records=[*rows, deepcopy(rows[0])], identity=identity, verification=VERIFIED)


def test_ablation_reports_max_minus_smart_with_unambiguous_labels():
    from evals.paired_retrieval_report import build_paired_report, render_paired_report
    case, identity = identity_fixture(comparison="ik-smart-vs-max")
    rows = records(identity)
    rows[0]["arm"], rows[1]["arm"] = "ik", "ik_max_word"
    report = build_paired_report(cases=[case], records=rows, identity=identity, verification=VERIFIED)
    assert report["arm_order"] == ["ik", "ik_max_word"]
    assert report["pairs"][0]["deltas"]["chunk_recall@40"] == 0.5
    assert report["completion_verified"] is True and report["valid_pairs"] == 1
    markdown = render_paired_report(report)
    assert "ik_smart" in markdown and "ik_max_word" in markdown
    assert "standard 条件均值" not in markdown
    rows[0]["arm"] = "standard"
    with pytest.raises(ValueError, match="foreign"):
        build_paired_report(cases=[case], records=rows, identity=identity, verification=VERIFIED)
