import pytest
from pydantic import ValidationError

from agentic_rag.config import Settings


def test_settings_use_local_defaults(monkeypatch):
    monkeypatch.setenv("AGENTIC_RAG_MYSQL_DSN", "mysql+asyncmy://rag:rag@127.0.0.1/rag")
    monkeypatch.setenv("AGENTIC_RAG_REDIS_URL", "redis://127.0.0.1:6379/0")
    monkeypatch.setenv("AGENTIC_RAG_ELASTICSEARCH_URL", "http://localhost:9200")
    monkeypatch.setenv("AGENTIC_RAG_DEEPSEEK_BASE_URL", "https://models.example.invalid/v1")
    monkeypatch.setenv(
        "AGENTIC_RAG_QWEN_EMBEDDING_BASE_URL",
        "https://embeddings.example.invalid/v1",
    )
    settings = Settings()
    assert settings.redis_url == "redis://127.0.0.1:6379/0"
    assert settings.elasticsearch_url == "http://localhost:9200"
    assert settings.embedding_dimensions == 1024
    assert settings.query_worker_count == 1
    assert settings.ingestion_worker_count == 1
    assert settings.mem0_enabled is True
    assert settings.max_research_rounds == 6
    assert settings.max_upload_bytes == 50 * 1024 * 1024
    assert settings.ingestion_pipeline_version == "ingestion-v2"
    assert settings.index_generation == "index-v3"


def test_mem0_can_be_explicitly_disabled_for_local_diagnostics(monkeypatch) -> None:
    monkeypatch.setenv("AGENTIC_RAG_MYSQL_DSN", "mysql+asyncmy://rag:rag@127.0.0.1/rag")
    monkeypatch.setenv("AGENTIC_RAG_DEEPSEEK_BASE_URL", "https://models.example.invalid/v1")
    monkeypatch.setenv(
        "AGENTIC_RAG_QWEN_EMBEDDING_BASE_URL",
        "https://embeddings.example.invalid/v1",
    )
    monkeypatch.setenv("AGENTIC_RAG_MEM0_ENABLED", "0")

    assert Settings().mem0_enabled is False


def test_upload_size_limit_must_be_positive() -> None:
    with pytest.raises(ValidationError):
        Settings(
            mysql_dsn="mysql+asyncmy://rag:rag@127.0.0.1/rag",
            deepseek_base_url="https://models.example.invalid/v1",
            qwen_embedding_base_url="https://embeddings.example.invalid/v1",
            max_upload_bytes=0,
        )


@pytest.mark.parametrize("generation", ["INDEX-V1", " index-v1 ", "index/v1"])
def test_index_generation_rejects_noncanonical_aliases(generation: str) -> None:
    with pytest.raises(ValidationError):
        Settings(
            mysql_dsn="mysql+asyncmy://rag:rag@127.0.0.1/rag",
            deepseek_base_url="https://models.example.invalid/v1",
            qwen_embedding_base_url="https://embeddings.example.invalid/v1",
            index_generation=generation,
        )


def test_new_jobs_cannot_be_configured_for_legacy_index_v1() -> None:
    with pytest.raises(ValidationError, match="legacy"):
        Settings(
            mysql_dsn="mysql+asyncmy://rag:rag@127.0.0.1/rag",
            deepseek_base_url="https://models.example.invalid/v1",
            qwen_embedding_base_url="https://embeddings.example.invalid/v1",
            index_generation="index-v1",
        )
