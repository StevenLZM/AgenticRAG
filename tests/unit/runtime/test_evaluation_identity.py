from types import SimpleNamespace

from agentic_rag.runtime.evaluation_identity import evaluation_config_fingerprint


def test_answer_provider_and_concurrency_change_identity_without_secret_export():
    baseline = SimpleNamespace(deepseek_base_url="https://provider-a", max_concurrent_llm_calls=8, deepseek_api_key="secret")
    identity = evaluation_config_fingerprint(baseline)
    assert evaluation_config_fingerprint(SimpleNamespace(**{**vars(baseline), "deepseek_base_url": "https://provider-b"})) != identity
    assert evaluation_config_fingerprint(SimpleNamespace(**{**vars(baseline), "max_concurrent_llm_calls": 4})) != identity
    assert evaluation_config_fingerprint(SimpleNamespace(**{**vars(baseline), "deepseek_api_key": "rotated-secret"})) == identity
