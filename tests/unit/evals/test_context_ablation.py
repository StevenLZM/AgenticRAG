import pytest
from evals.context_ablation import (
    SourceSpan,
    contains_gold,
    validate_ablation_configs,
    summarize_recorded,
)


def test_gold_uses_source_coordinates_across_changed_final_child_ids():
    gold = SourceSpan(document_id="doc", version_id="v1", block="p4", start=5, end=15)
    assert contains_gold(
        [dict(child_id="changed-final-hash", spans=[gold.model_dump()])], [gold]
    )
    assert not contains_gold(
        [
            dict(
                child_id="old-gold-id",
                spans=[gold.model_copy(update={"version_id": "v2"}).model_dump()],
            )
        ],
        [gold],
    )


@pytest.fixture
def ablation_configs():
    shared = dict(
        corpus_hash="a",
        acl_hash="acl",
        embedding="e",
        reranker="r",
        recall_k=40,
        rrf_k=30,
        rerank_k=10,
        source_boundaries_hash="320",
        task_budget=24,
        index_manifest_hash="a" * 64,
    )
    configs = {
        letter: {**shared, "base_tokens": 384 if letter == "A" else 320}
        for letter in "ABCDE"
    }
    configs["A"]["source_boundaries_hash"] = "384"
    return configs


@pytest.mark.parametrize("config", ["D", "E", "DE"])
@pytest.mark.parametrize(
    "invalid", [None, "", " ", "a" * 63, "A" * 64, "g" * 64, "a" * 64 + "\n", 123]
)
def test_de_index_identity_must_be_present_and_well_formed(ablation_configs, config, invalid):
    for key in config:
        if invalid is None:
            ablation_configs[key].pop("index_manifest_hash")
        else:
            ablation_configs[key]["index_manifest_hash"] = invalid
    with pytest.raises(ValueError, match="manifest"):
        validate_ablation_configs(ablation_configs)


def test_de_different_valid_index_identities_are_rejected(ablation_configs):
    ablation_configs["E"]["index_manifest_hash"] = "b" * 64
    with pytest.raises(ValueError, match="same index build"):
        validate_ablation_configs(ablation_configs)


def test_same_budget_and_boundaries_required_and_usage_is_recorded(ablation_configs):
    configs = ablation_configs
    validate_ablation_configs(configs)
    configs["D"]["recall_k"] = 80
    with pytest.raises(ValueError, match="budget"):
        validate_ablation_configs(configs)
    result = summarize_recorded(
        [
            dict(
                run_id="run-1",
                config="D",
                build="build-1",
                recall=1.0,
                reciprocal_rank=1.0,
                ndcg=1.0,
                attribution_failures=1,
                source_support_failures=2,
                latency_ms=10.0,
                input_tokens=100,
                output_tokens=20,
                cost=0.01,
            )
        ]
    )
    assert result["D"]["attribution_failures"] == 1
    assert result["D"]["input_tokens"] == 100
    assert result["evidence_kind"] == "recorded_runs_not_fresh_model_quality"
