import importlib.util
import json

import pytest

from evals.gold_v2_models import CorpusSnapshot, GoldCaseV2
from tests.unit.evals.test_gold_v2_validation import gold_payload, snapshot_payload


def api():
    assert importlib.util.find_spec("evals.gold_answer_spec"), "frozen scoring specification not implemented"
    from evals import gold_answer_spec
    return gold_answer_spec


def test_draft_spec_keeps_reference_conditions_sources_and_never_self_approves():
    mod = api()
    raw = gold_payload()
    raw.update(reference_answer="每人每晚500元人民币，含税。", question="A公司2026年深圳上限？")
    raw["critical_fields"][0]["conditions"] = {"company": "A公司", "year": 2026, "city": "深圳"}
    spec = mod.build_draft_spec(GoldCaseV2.model_validate(raw))
    assert [f.claim for f in spec.required_facts] == ["每人每晚500元人民币", "含税"]
    assert spec.question_conditions == [{"company": "A公司", "year": 2026, "city": "深圳"}]
    assert spec.source_refs[0].sha256 == "a" * 64
    assert spec.review_status == "human_review_pending"


@pytest.mark.parametrize("change", ["case_hash", "claim", "source", "approved"])
def test_tampered_scoring_spec_cannot_import_system_answer_or_fake_approval(tmp_path, change):
    mod = api()
    case = GoldCaseV2.model_validate(gold_payload())
    raw = mod.build_draft_spec(case).model_dump(mode="json")
    if change == "case_hash":
        raw["gold_case_sha256"] = "b" * 64
    elif change == "claim":
        raw["required_facts"][0]["claim"] = "系统答案是600元"
    elif change == "source":
        raw["required_facts"][0]["source_fact_ids"] = []
    else:
        raw["review_status"] = "human_approved"
    path = tmp_path / "spec.jsonl"
    path.write_text(json.dumps(raw, ensure_ascii=False) + "\n")
    with pytest.raises(ValueError):
        mod.load_gold_answer_specs(path, [case], CorpusSnapshot.model_validate(snapshot_payload()))


def test_spec_missing_duplicate_and_foreign_case_rejected(tmp_path):
    mod = api()
    case = GoldCaseV2.model_validate(gold_payload())
    raw = mod.build_draft_spec(case).model_dump(mode="json")
    path = tmp_path / "spec.jsonl"
    path.write_text(json.dumps(raw) + "\n")
    assert set(mod.load_gold_answer_specs(path, [case], CorpusSnapshot.model_validate(snapshot_payload()))) == {"case"}
    path.write_text(json.dumps(raw) + "\n" + json.dumps(raw) + "\n")
    with pytest.raises(ValueError, match="duplicate"):
        mod.load_gold_answer_specs(path, [case], CorpusSnapshot.model_validate(snapshot_payload()))


def test_draft_spec_export_never_overwrites_frozen_file(tmp_path):
    mod = api()
    path = tmp_path / "spec.jsonl"
    path.write_text("frozen")
    with pytest.raises(FileExistsError):
        mod.write_draft_specs(path, [GoldCaseV2.model_validate(gold_payload())])
    assert path.read_text() == "frozen"
