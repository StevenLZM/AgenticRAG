"""Application configuration loaded from environment variables."""

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Validated runtime settings for the application."""

    model_config = SettingsConfigDict(env_prefix="AGENTIC_RAG_", extra="forbid")

    mysql_dsn: str
    redis_url: str = "redis://127.0.0.1:6379/0"
    elasticsearch_url: str = "http://localhost:9200"
    deepseek_base_url: str
    deepseek_api_key: SecretStr | None = None
    qwen_embedding_base_url: str
    qwen_api_key: SecretStr | None = None
    default_user_id: str = "default_user"
    main_model: str = "deepseek-v4-pro"
    light_model: str = "deepseek-v4-flash"
    embedding_model: str = "text-embedding-v3"
    embedding_dimensions: int = 1024
    reranker_model: str = "BAAI/bge-reranker-v2-m3"
    max_concurrent_query_runs: int = 4
    max_concurrent_llm_calls: int = 8
    max_concurrent_reranks: int = 1
    max_parallel_subagents_per_run: int = 3
    max_research_rounds: int = 4
    max_answer_revisions: int = 1
    query_run_timeout_seconds: int = 300
    max_evidence_tokens: int = 12_000
    research_context_soft_limit_tokens: int = 16_000
    query_worker_count: Literal[1] = 1
    ingestion_worker_count: Literal[1] = 1
    max_upload_bytes: int = Field(
        default=50 * 1024 * 1024,
        gt=0,
        le=1024 * 1024 * 1024,
    )
    parser_version: str = "docling-v1"
    ingestion_pipeline_version: str = "ingestion-v1"
    index_generation: str = "index-v1"
    query_checkpoint_path: Path = Path("var/query_checkpoints.sqlite")
    ingestion_checkpoint_path: Path = Path("var/ingestion_checkpoints.sqlite")
    artifact_root: Path = Path("var/artifacts")


@lru_cache
def get_settings() -> Settings:
    """Return the process-wide validated settings instance."""
    return Settings()  # type: ignore[call-arg]
