from copy import deepcopy

import pytest

from evals.gold_v2_models import GoldCaseV2, canonical_hash
from tests.unit.evals.test_gold_v2_validation import gold_payload


def identity_fixture(case_override=None, *, comparison=None, extra_cases=()):
    from evals.paired_retrieval_identity import build_paired_identity
    case = case_override or GoldCaseV2.model_validate(gold_payload())
    content = {"test": "environment"}
    env = {"verified": True, "environment_sha256": canonical_hash(content), "content": content}
    config = {"user_id": "u", "index_generation": "index-v3", "index": "children-v3", "top_k": 40,
              "request_limit": 200, "timeout_s": 30, "wall_s": 900, "settings": {"model": "fixed"}}
    if comparison is not None:
        config["comparison"] = comparison
    identity = build_paired_identity(bundle_audit={"verified": True, "target_snapshot_id": case.snapshot_id,
        "source_files": {"gold": {"sha256": "a" * 64}}}, environment=env, cases=[case, *extra_cases], config=config,
        implementation={"test.py": "fixed"})
    return case, identity


def test_sample_is_fixed_dev_20_per_category():
    from evals.paired_retrieval_identity import select_paired_cases
    rows = []
    for category in ("section_comprehension", "single_hop", "table_calculation", "unanswerable", "multi_hop"):
        for n in range(25):
            raw = gold_payload()
            raw.update(case_id=category + str(n), category=category, split="dev" if n < 24 else "test")
            rows.append(GoldCaseV2.model_validate(raw))
    selected = select_paired_cases(rows)
    assert len(selected) == 100
    assert all(c.split == "dev" for c in selected)
    assert selected == select_paired_cases(list(reversed(rows)))
    with pytest.raises(ValueError):
        select_paired_cases(rows[:10])


@pytest.mark.parametrize("change", ["user_id", "index_generation", "top_k", "request_limit", "model", "query", "code", "gold"])
def test_pair_allows_only_declared_lexical_changes(change):
    from evals.paired_retrieval_identity import assert_paired_comparable
    _, identity = identity_fixture()
    left, right = [deepcopy(identity["arms"][a]) for a in ("standard", "ik")]
    assert_paired_comparable(left, right)
    common = right["common"]
    if change in {"user_id", "index_generation", "top_k", "request_limit"}:
        common["config"][change] = "changed"
    elif change == "model":
        common["config"]["settings"]["model"] = "changed"
    elif change == "query":
        common["cases"][0]["question"] = "changed"
    else:
        common[change] = "changed"
    right["sha256"] = canonical_hash({k: v for k, v in right.items() if k != "sha256"})
    with pytest.raises(ValueError):
        assert_paired_comparable(left, right)


def test_analyzer_ablation_freezes_only_zh_override_and_preserves_legacy_identity():
    import json
    from evals.paired_retrieval_identity import expected_request, validate_identity
    case, legacy = identity_fixture()
    _, ablation = identity_fixture(comparison="ik-smart-vs-max")
    assert set(legacy["arms"]) == {"standard", "ik"}
    assert set(ablation["arms"]) == {"ik", "ik_max_word"}
    validate_identity(json.loads(json.dumps(ablation, sort_keys=True)))
    before = expected_request(ablation, case.case_id, "ik")
    after = expected_request(ablation, case.case_id, "ik_max_word")
    assert before["query"]["bool"]["must"] == {
        "multi_match": {"query": case.question, "fields": ["contextualized_content.zh^2", "contextualized_content"],
                        "type": "best_fields", "tie_breaker": 0.0}}
    assert after["query"]["bool"]["must"] == {
        "dis_max": {"queries": [
            {"match": {"contextualized_content.zh": {"query": case.question, "analyzer": "ik_max_word", "boost": 2}}},
            {"match": {"contextualized_content": case.question}}], "tie_breaker": 0.0}}
    before["query"]["bool"].pop("must")
    after["query"]["bool"].pop("must")
    assert before == after  # scope, candidate window and projection are unchanged
    assert ablation["sha256"] != legacy["sha256"]


@pytest.mark.parametrize("comparison", ["unknown", "ik-vs-dense", ""])
def test_unapproved_comparisons_are_rejected(comparison):
    with pytest.raises(ValueError, match="comparison"):
        identity_fixture(comparison=comparison)


def test_analyzer_change_cannot_be_smuggled_into_declared_profile():
    from evals.paired_retrieval_identity import validate_identity
    _, identity = identity_fixture(comparison="ik-smart-vs-max")
    arm = identity["arms"]["ik_max_word"]
    arm["lexical"]["search_analyzers"]["contextualized_content"] = "ik_max_word"
    arm["sha256"] = canonical_hash({k: v for k, v in arm.items() if k != "sha256"})
    identity["sha256"] = canonical_hash({k: v for k, v in identity.items() if k != "sha256"})
    with pytest.raises(ValueError, match="undeclared lexical"):
        validate_identity(identity)
