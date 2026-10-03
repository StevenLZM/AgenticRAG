from agentic_rag.runtime.models import RuntimeConfigSnapshot
from agentic_rag.runtime.query_composition import build_query_snapshot
from tests.unit.query.test_router_fast_path import SNAPSHOT


def test_legacy_snapshot_hash_stable_and_new_policy_is_explicit():
    old = SNAPSHOT.model_dump(mode="json")
    old.pop("routing_policy_version", None)
    restored = RuntimeConfigSnapshot.model_validate(old)
    assert hasattr(restored, "routing_policy_version")
    assert restored.routing_policy_version is None
    assert restored.snapshot_id == SNAPSHOT.snapshot_id


def test_composed_snapshot_enables_v2():
    from tests.unit.runtime.test_query_composition import _settings
    snapshot = build_query_snapshot(_settings())
    assert snapshot.routing_policy_version == "routing-v2"
    assert "router_v2" in snapshot.prompt_hash_map


def test_lexical_analysis_is_bound_to_runtime_and_provider_identity():
    from tests.unit.runtime.test_query_composition import _settings
    standard = build_query_snapshot(_settings(lexical_analysis="standard"))
    ik = build_query_snapshot(_settings(lexical_analysis="ik"))
    assert standard.retrieval_config_version == "retrieval-v1"
    assert ik.retrieval_config_version == "retrieval-ik-v1"
    assert standard.provider_config_fingerprint != ik.provider_config_fingerprint
    assert standard.snapshot_id != ik.snapshot_id
