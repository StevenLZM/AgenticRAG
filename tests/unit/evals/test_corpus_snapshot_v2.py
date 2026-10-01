from copy import deepcopy

import pytest

from evals.corpus_snapshot import compare_live_snapshot, experiment_fingerprint, verify_alias_definition
from evals.gold_v2_models import CorpusSnapshot
from tests.unit.evals.test_gold_v2_validation import snapshot_payload


@pytest.mark.parametrize("mutation", ["add", "remove", "alias", "uuid", "content", "foreign"])
def test_drift_anywhere_is_detected_even_without_generation_change(mutation):
    frozen = snapshot_payload()
    live = deepcopy(frozen)
    if mutation == "add":
        live["documents"].append({"document_id": "noise", "document_version_id": "noise", "content_hash": "a" * 64})
    if mutation == "remove":
        live["children"].pop()
    if mutation == "alias":
        live["scope"]["concrete_indices"] = ["other"]
    if mutation == "uuid":
        live["scope"]["cluster_uuid"] = "new"
    if mutation == "content":
        live["children"][1]["content"] = "modified"
    if mutation == "foreign":
        live["children"][0]["user_id"] = "other"
    with pytest.raises(ValueError, match="drift"):
        compare_live_snapshot(CorpusSnapshot.model_validate(frozen), live)


def test_snapshot_checks_full_members_and_fingerprint_all_configuration():
    frozen = CorpusSnapshot.model_validate(snapshot_payload())
    assert compare_live_snapshot(frozen, snapshot_payload())["status"] == "matched"
    assert experiment_fingerprint("gold", frozen.snapshot_id, "config1", "judge", ["case"]) != experiment_fingerprint(
        "gold", frozen.snapshot_id, "config2", "judge", ["case"])


def test_robustness_retains_drift_instead_of_relabelling_cases():
    frozen = CorpusSnapshot.model_validate(snapshot_payload())
    live = snapshot_payload()
    live["children"].pop()
    result = compare_live_snapshot(frozen, live, mode="robustness")
    assert result["status"] == "drift"
    assert result["missing_child_ids"] == ["c2"]


@pytest.mark.parametrize("definition", [{"filter": {"term": {"user_id": "foreign"}}}, {"search_routing": "shard2"}])
def test_same_index_uuid_alias_filter_or_routing_cannot_change_visibility(definition):
    snapshot = CorpusSnapshot.model_validate(snapshot_payload())
    with pytest.raises(ValueError, match="alias"):
        verify_alias_definition(snapshot, {"children-v3": {"aliases": {"children": definition}}})


def test_existing_unfiltered_alias_and_explicit_future_definition():
    snapshot = CorpusSnapshot.model_validate(snapshot_payload())
    assert verify_alias_definition(snapshot, {"children-v3": {"aliases": {"children": {}}}}) == {"children-v3": {}}
