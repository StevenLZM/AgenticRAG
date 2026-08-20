"""Fail/skip classification for the opt-in real-provider fixture."""

from __future__ import annotations

import pytest

from tests.fixtures import query_services


def test_mixed_missing_and_explicitly_disabled_provider_configuration_fails(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    """A missing base URL may not hide an explicit Mem0 disablement as a skip."""
    monkeypatch.setenv("AGENTIC_RAG_RUN_REAL_QUERY_PROVIDER_E2E", "1")
    monkeypatch.setenv("AGENTIC_RAG_TEST_MYSQL_DSN", "mysql+asyncmy://u:p@127.0.0.1:3306/t")
    monkeypatch.setenv("AGENTIC_RAG_TEST_REDIS_DSN", "redis://127.0.0.1:6379/15")
    monkeypatch.setenv("AGENTIC_RAG_TEST_ELASTICSEARCH_URL", "http://127.0.0.1:9200")
    monkeypatch.setenv("AGENTIC_RAG_MEM0_ENABLED", "false")

    def missing_settings(**kwargs: object) -> object:
        del kwargs
        raise ValueError("deepseek_base_url is missing")

    monkeypatch.setattr(query_services, "Settings", missing_settings)

    with pytest.raises(pytest.fail.Exception, match="invalid DeepSeek/Qwen/Mem0"):
        query_services._require_real_provider_services(tmp_path)


def test_explicit_non_loopback_service_configuration_fails_instead_of_skipping(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    """A supplied unsafe service endpoint is invalid configuration, not an absent opt-in."""
    monkeypatch.setenv("AGENTIC_RAG_RUN_REAL_QUERY_PROVIDER_E2E", "1")
    monkeypatch.setenv("AGENTIC_RAG_TEST_MYSQL_DSN", "mysql+asyncmy://u:p@db.example:3306/t")
    monkeypatch.setenv("AGENTIC_RAG_TEST_REDIS_DSN", "redis://127.0.0.1:6379/15")
    monkeypatch.setenv("AGENTIC_RAG_TEST_ELASTICSEARCH_URL", "http://127.0.0.1:9200")

    with pytest.raises(pytest.fail.Exception, match="local MySQL DSN"):
        query_services._require_real_provider_services(tmp_path)
