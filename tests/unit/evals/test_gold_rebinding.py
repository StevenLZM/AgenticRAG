from copy import deepcopy
import hashlib
import json

import pytest

from evals.gold_answer_spec import build_draft_spec
from evals.gold_v2_models import GoldCaseV2
from tests.unit.evals.test_gold_v2_validation import gold_payload
from tests.unit.evals.test_migration_evidence import freeze, migration_fixture


def test_only_binding_fields_change():
    from evals.gold_rebinding import rebind_gold_rows, rebind_spec_rows
    rows = [gold_payload()]
    expected = deepcopy(rows)
    changed = rebind_gold_rows(rows, snapshot_id="f" * 64)
    assert changed == [{**expected[0], "snapshot_id": "f" * 64}]
    assert rows == expected
    specs = [build_draft_spec(GoldCaseV2.model_validate(rows[0])).model_dump(mode="json")]
    new = rebind_spec_rows(specs, cases=[GoldCaseV2.model_validate(changed[0])])
    for key in set(specs[0]) - {"snapshot_id", "gold_case_sha256"}:
        assert new[0][key] == specs[0][key]
    assert new[0]["snapshot_id"] == "f" * 64
    assert new[0]["gold_case_sha256"] != specs[0]["gold_case_sha256"]


@pytest.fixture
def inputs(tmp_path):
    old, target, journal, evidence = migration_fixture()
    path = tmp_path / "original.txt"
    path.write_text("事实")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    snapshots = []
    for snapshot in (old, target):
        raw = snapshot.model_dump()
        raw["documents"][0].update(source_path=str(path), content_hash=digest)
        snapshots.append(freeze(raw))
    old, target = snapshots
    evidence["originals"]["v"] = digest
    row = gold_payload()
    row.update(snapshot_id=old.snapshot_id, reference_answer="事实", question="事实是什么？", critical_fields=[], derivation=None)
    row["source_documents"][0].update(sha256=digest)
    row["required_evidence_groups_all_of"] = [{"fact_id": "f", "claim": "事实", "document_version_id": "v",
        "source_anchors": ["事实"], "parent_ids_any_of": ["p"], "child_ids_any_of": ["c"],
        "child_evidence_sets_any_of": [["c"]], "child_mapping_methods": ["rendered_text"], "mapping_status": "mapped"}]
    case = GoldCaseV2.model_validate(row)
    files = {"source_snapshot": old.model_dump(mode="json"), "source_gold": row,
             "source_specs": build_draft_spec(case).model_dump(mode="json"), "journal_path": journal}
    args = {"target": target, "evidence": evidence, "output": tmp_path / "bundle"}
    for name, value in files.items():
        file = tmp_path / (name + ".json")
        file.write_text(json.dumps(value, ensure_ascii=False) + "\n")
        args[name] = file
    return args


def test_publish_load_preserves_sources_and_private_permissions(inputs):
    from evals.gold_rebinding import publish_rebound_bundle, load_rebound_bundle
    before = {k: inputs[k].read_bytes() for k in ("source_snapshot", "source_gold", "source_specs", "journal_path")}
    result = publish_rebound_bundle(**inputs)
    assert result["verified"] is True
    snapshot, cases, specs, audit = load_rebound_bundle(inputs["output"])
    assert snapshot.snapshot_id == inputs["target"].snapshot_id
    assert cases[0].reference_answer == "事实"
    assert specs["case"].review_status == "human_review_pending"
    assert audit["counts"]["cases"] == 1
    assert inputs["output"].stat().st_mode & 0o777 == 0o700
    assert all(p.stat().st_mode & 0o777 == 0o600 for p in inputs["output"].iterdir())
    assert before == {k: inputs[k].read_bytes() for k in before}


@pytest.mark.parametrize("field", ["required_facts", "critical_rules", "review_status"])
def test_sidecar_semantic_drift_is_not_a_rebind(inputs, field):
    from evals.gold_rebinding import publish_rebound_bundle, load_rebound_bundle
    raw = json.loads(inputs["source_specs"].read_text())
    raw[field] = "human_approved" if field == "review_status" else [{}]
    inputs["source_specs"].write_text(json.dumps(raw))
    with pytest.raises(ValueError):
        publish_rebound_bundle(**inputs)
    with pytest.raises(ValueError, match="incomplete"):
        load_rebound_bundle(inputs["output"])


def test_incomplete_or_existing_bundle_refused(inputs):
    from evals.gold_rebinding import publish_rebound_bundle, load_rebound_bundle
    publish_rebound_bundle(**inputs)
    with pytest.raises(FileExistsError):
        publish_rebound_bundle(**inputs)
    (inputs["output"] / "gold.v2.2.jsonl").write_text("tampered")
    with pytest.raises(ValueError, match="hash"):
        load_rebound_bundle(inputs["output"])


def test_unmapped_and_splits_preserved():
    from evals.gold_rebinding import rebind_gold_rows
    rows = [gold_payload()]
    rows[0]["split"] = "test"
    rows[0]["required_evidence_groups_all_of"][0].update(mapping_status="unmapped", child_ids_any_of=[], child_evidence_sets_any_of=[])
    result = rebind_gold_rows(rows, snapshot_id="f" * 64)
    assert result[0]["split"] == "test"
    assert result[0]["required_evidence_groups_all_of"][0]["mapping_status"] == "unmapped"


def test_cross_child_coverage_and_scope_revalidated(inputs):
    from evals.gold_rebinding import publish_rebound_bundle
    raw = json.loads(inputs["source_gold"].read_text())
    raw["required_evidence_groups_all_of"][0].update(source_anchors=["缺失事实"])
    inputs["source_gold"].write_text(json.dumps(raw))
    inputs["source_specs"].write_text(build_draft_spec(GoldCaseV2.model_validate(raw)).model_dump_json())
    with pytest.raises(ValueError, match="anchor|sufficien"):
        publish_rebound_bundle(**inputs)
