from agentic_rag.config import Settings


def test_settings_use_local_defaults(monkeypatch):
    monkeypatch.setenv("AGENTIC_RAG_MYSQL_DSN", "mysql+asyncmy://rag:rag@127.0.0.1/rag")
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
