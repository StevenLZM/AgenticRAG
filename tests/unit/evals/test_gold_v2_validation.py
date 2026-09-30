import copy
import hashlib
import json

import pytest

from evals.gold_v2_models import CorpusSnapshot, GoldCaseV2
from evals.gold_v2_validation import load_gold_v2


def snapshot_payload():
    payload = {"schema_version": 2, "captured_at": "2026-09-30T00:00:00Z",
               "scope": {"user_id": "u", "es_url": "http://localhost:9200",
                         "alias": "children", "concrete_indices": ["children-v3"],
                         "cluster_uuid": "cluster", "index_metadata": {}},
               "status_counts": [],
               "documents": [{"document_id": "d", "document_version_id": "v", "content_hash": "a" * 64}],
               "parents": [{"id": "p", "user_id": "u", "document_id": "d", "document_version_id": "v"}],
               "children": [{"child_id": "c1", "parent_id": "p", "user_id": "u", "document_id": "d", "document_version_id": "v"},
                            {"child_id": "c2", "parent_id": "p", "user_id": "u", "document_id": "d", "document_version_id": "v"}]}
    payload["snapshot_id"] = hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True).encode()).hexdigest()
    return payload


def gold_payload():
    return {"schema_version": 2, "case_id": "case", "snapshot_id": snapshot_payload()["snapshot_id"],
            "user_id": "u", "split": "dev", "category": "multi_hop", "question": "增加多少元？",
            "answerable": True, "reference_answer": "增加60元。", "expected_route": "research",
            "source_documents": [{"document_id": "d", "document_version_id": "v", "sha256": "a" * 64,
                                  "filename": "d.txt", "searchable_at_snapshot": True}],
            "required_evidence_groups_all_of": [{"fact_id": "f", "claim": "旧410新470", "document_version_id": "v",
                "source_anchors": ["410", "470"], "parent_ids_any_of": ["p"], "child_ids_any_of": [],
                "child_evidence_sets_any_of": [["c1", "c2"]], "child_mapping_methods": ["range"], "mapping_status": "mapped_multi_child"}],
            "critical_fields": [{"name": "increase", "value": 60, "unit": "CNY/person/night", "tolerance": 0}],
            "derivation": {"operation": "2026_limit-2025_limit", "inputs": {"2025": 410, "2026": 470}},
            "review_status": "original_verified_human_review_pending"}


def load(tmp_path, rows):
    path = tmp_path / "gold.jsonl"
    path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n")
    return load_gold_v2(path, CorpusSnapshot.model_validate(snapshot_payload()))


def test_loads_cross_child_and_keeps_refusal_evidence(tmp_path):
    row = gold_payload()
    case = load(tmp_path, [row])[0]
    assert case.required_evidence_groups_all_of[0].child_evidence_sets_any_of == [["c1", "c2"]]
    refusal = copy.deepcopy(row)
    refusal.update(case_id="refusal", answerable=False, category="unanswerable", derivation=None, critical_fields=[])
    assert load(tmp_path, [refusal])[0].required_evidence_groups_all_of


@pytest.mark.parametrize("mutate", [
    lambda r: r.update(user_id="foreign"),
    lambda r: r["source_documents"][0].update(sha256="b" * 64),
    lambda r: r["required_evidence_groups_all_of"][0].update(document_version_id="foreign"),
    lambda r: r["required_evidence_groups_all_of"][0].update(source_anchors=[]),
    lambda r: r["critical_fields"][0].update(value=61),
    lambda r: r["critical_fields"][0].update(value=float("nan")),
    lambda r: r["critical_fields"][0].update(value=True),
    lambda r: r["critical_fields"][0].update(unit=""),
    lambda r: r["required_evidence_groups_all_of"][0].update(child_ids_any_of=["c1"]),
])
def test_rejects_untrustworthy_gold(tmp_path, mutate):
    row = gold_payload()
    mutate(row)
    with pytest.raises(ValueError):
        load(tmp_path, [row])


def test_duplicate_cases_and_snapshot_tampering_rejected(tmp_path):
    with pytest.raises(ValueError, match="duplicate"):
        load(tmp_path, [gold_payload(), gold_payload()])
    snapshot = snapshot_payload()
    snapshot["children"][0]["parent_id"] = "different"
    with pytest.raises(ValueError, match="hash"):
        CorpusSnapshot.model_validate(snapshot)


def test_original_frozen_corpus_is_supported():
    from pathlib import Path
    root = Path("var/artifacts/evals/formal-corpus-gold-v2-20260930-production")
    if not root.exists():
        pytest.skip("local production artifact unavailable")
    snapshot = CorpusSnapshot.model_validate_json((root / "snapshot.json").read_text())
    cases = load_gold_v2(root / "gold.v2.2.jsonl", snapshot)
    assert len(cases) == 4453
    assert sum(g.mapping_status == "unmapped" for c in cases for g in c.required_evidence_groups_all_of) == 1


def test_numeric_schema_is_finite_and_strict():
    row = gold_payload()
    row["critical_fields"][0]["tolerance"] = float("inf")
    with pytest.raises(ValueError):
        GoldCaseV2.model_validate(row)
