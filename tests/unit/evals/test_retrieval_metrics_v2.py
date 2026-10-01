import math

import pytest

from evals.gold_v2_models import GoldCaseV2
from evals.retrieval_metrics_v2 import score_retrieval_stage, score_context, score_rounds
from tests.unit.evals.test_gold_v2_validation import gold_payload


def case_with_two_facts():
    row = gold_payload()
    group = row["required_evidence_groups_all_of"][0]
    group.update(child_ids_any_of=["old"], child_evidence_sets_any_of=[["old"]], mapping_status="mapped")
    row["required_evidence_groups_all_of"].append({**group, "fact_id": "new", "child_ids_any_of": ["new"],
        "child_evidence_sets_any_of": [["new"]], "parent_ids_any_of": ["new-parent"]})
    return GoldCaseV2.model_validate(row)


def test_one_hop_has_good_mrr_but_incomplete_evidence():
    score = score_retrieval_stage(case_with_two_facts(), ["new"], [1, 10, 50], "child")
    assert score["metrics"]["mrr"] == 1
    assert score["metrics"]["fact_recall@10"] == .5
    assert score["metrics"]["complete_evidence@10"] == 0
    assert score["metrics"]["binary_ndcg@10"] == pytest.approx(1 / (1 + 1 / math.log2(3)))


def test_cross_child_fact_requires_all_pieces_at_actual_rank():
    case = GoldCaseV2.model_validate(gold_payload())
    score = score_retrieval_stage(case, ["c1", "c1", "noise", "c2"], [1, 3, 4], "child")
    assert score["metrics"]["mrr"] == .25
    assert score["metrics"]["fact_recall@3"] == 0
    assert score["metrics"]["fact_recall@4"] == 1


def test_rank31_is_excluded_at30_and_visible_at50():
    case = case_with_two_facts()
    score = score_retrieval_stage(case, [f"n{i}" for i in range(30)] + ["old", "new"], [30, 50], "child")
    assert score["metrics"]["fact_recall@30"] == 0
    assert score["metrics"]["fact_recall@50"] == 1
    assert score["metrics"]["mrr"] == pytest.approx(1 / 31)


def test_unmapped_gold_is_unknown_not_zero_and_graded_is_not_invented():
    row = gold_payload()
    row["required_evidence_groups_all_of"][0].update(mapping_status="unmapped", child_evidence_sets_any_of=[])
    score = score_retrieval_stage(GoldCaseV2.model_validate(row), [], [10], "child")
    assert score["status"] == "qrel_unmapped"
    assert score["metrics"]["fact_recall@10"] is None
    assert score["metrics"]["graded_ndcg@10"] is None


def test_graded_ndcg_ideal_is_frozen_gold_and_unjudged_is_unknown():
    row = gold_payload()
    row["graded_qrels"] = [{"level": "child", "chunk_id": i, "relevance": r, "reviewer": "human", "reason": "checked"}
                           for i, r in [("c1", 1), ("c2", 3), ("n", 0)]]
    case = GoldCaseV2.model_validate(row)
    score = score_retrieval_stage(case, ["c1", "c2"], [2], "child")
    assert score["metrics"]["graded_ndcg@2"] == pytest.approx((1 + 7 / math.log2(3)) / (7 + 1 / math.log2(3)))
    assert score_retrieval_stage(case, ["unknown"], [2], "child")["metrics"]["graded_ndcg@2"] is None


def test_cropped_parent_does_not_claim_full_fact_coverage():
    case = GoldCaseV2.model_validate(gold_payload())
    assert score_context(case, [{"parent_id": "p", "content": "只有旧410", "heading_path": []}])["complete_evidence"] == 0
    assert score_context(case, [{"parent_id": "p", "content": "旧410新470", "heading_path": []}])["complete_evidence"] == 1


def test_multi_round_union_is_coverage_not_fabricated_ranking():
    case = case_with_two_facts()
    result = score_rounds(case, [{"stages": {"dense": ["old"]}}, {"stages": {"dense": ["new"]}}])
    assert result["union"]["dense"]["complete_evidence"] == 1
    assert "mrr" not in result["union"]["dense"]
    assert result["new_fact_gain"]["dense"] == [1, 1]


def test_failed_lane_and_rerank_fallback_have_separate_denominators():
    case = case_with_two_facts()
    result = score_rounds(case, [{"stages": {"dense": [], "bm25": ["old"], "rrf": ["old"], "rerank": ["old"]},
        "stage_status": {"dense": "failed", "bm25": "available", "rrf": "degraded", "rerank": "fallback"}}])
    assert result["rounds"][0]["dense"]["metrics"]["mrr"] is None
    assert result["rounds"][0]["rerank"]["metrics"]["mrr"] is None
    assert result["rounds"][0]["bm25"]["metrics"]["mrr"] == 1
    assert result["stage_counts"]["dense"]["failed"] == 1
    assert result["stage_counts"]["rerank"]["fallback"] == 1


def test_final_context_rank_uses_actual_cropped_fact_text():
    case = GoldCaseV2.model_validate(gold_payload())
    result = score_context(case, [{"parent_id": "p", "content": "无关前言"}, {"parent_id": "p", "content": "旧410新470"}])
    assert result["metrics"]["mrr"] == .5
    assert result["metrics"]["fact_recall@1"] == 0
    assert result["metrics"]["fact_recall@3"] == 1
    assert result["metrics"]["binary_ndcg@3"] == pytest.approx(1 / math.log2(3))
