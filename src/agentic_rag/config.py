"""Application configuration loaded from environment variables."""

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import AliasChoices, Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from agentic_rag.models.indexing import (
    DEFAULT_INDEX_GENERATION,
    LexicalAnalysis,
    validate_writable_index_generation,
)
from agentic_rag.tool_runtime.mcp import McpServerConfig


class Settings(BaseSettings):
    """Validated runtime settings for the application."""

    model_config = SettingsConfigDict(
        env_prefix="AGENTIC_RAG_",
        env_file=".env.local",
        env_file_encoding="utf-8",
        extra="forbid",
    )

    mysql_dsn: str
    redis_url: str = "redis://127.0.0.1:6379/0"
    elasticsearch_url: str = "http://localhost:9200"
    deepseek_base_url: str
    deepseek_api_key: SecretStr | None = None
    deepseek_protocol: Literal["auto", "chat", "responses"] = "auto"
    qwen_embedding_base_url: str
    qwen_api_key: SecretStr | None = None
    mem0_enabled: bool = True
    mem0_collection: str = "agent_memories_v1"
    mem0_embedding_base_url: str | None = None
    mem0_embedding_api_key: SecretStr | None = None
    mem0_embedding_model: str = "text-embedding-v3"
    mem0_llm_enabled: bool = False
    mem0_llm_model: str = "deepseek-v4-flash"
    mem0_llm_base_url: str | None = None
    mem0_llm_api_key: SecretStr | None = None
    mem0_elasticsearch_api_key: SecretStr | None = None
    mem0_elasticsearch_user: str | None = None
    mem0_elasticsearch_password: SecretStr | None = None
    mem0_elasticsearch_verify_certs: bool = True
    mem0_history_db_path: Path = Path("var/mem0/history.db")
    default_user_id: str = "default_user"
    allow_evaluation_requests: bool = False
    main_model: str = "deepseek-v4-pro"
    light_model: str = "deepseek-v4-flash"
    embedding_model: str = "text-embedding-v3"
    embedding_dimensions: int = 1024
    embedding_tokenizer_model: str = "Qwen/Qwen3-Embedding-0.6B"
    reranker_model: str = "BAAI/bge-reranker-v2-m3"
    max_concurrent_query_runs: int = 4
    max_concurrent_llm_calls: int = 8
    max_concurrent_reranks: int = 1
    max_parallel_subagents_per_run: int = 3
    max_research_rounds: int = Field(default=6, ge=1, le=8)
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
    ingestion_pipeline_version: str = "ingestion-v2"
    index_generation: str = DEFAULT_INDEX_GENERATION
    lexical_analysis: LexicalAnalysis = "standard"
    query_checkpoint_path: Path = Path("var/query_checkpoints.sqlite")
    ingestion_checkpoint_path: Path = Path("var/ingestion_checkpoints.sqlite")
    artifact_root: Path = Path("var/artifacts")
    tool_runtime_enabled: bool = True
    tool_invocation_path: Path = Path("var/tool_invocations.sqlite")
    tool_max_calls_per_run: int = Field(default=12, ge=1, le=48)
    tool_max_discoveries_per_run: int = Field(default=4, ge=1, le=12)
    tool_call_timeout_seconds: float = Field(default=20, gt=0, le=60)
    mcp_servers: tuple[McpServerConfig, ...] = ()
    amap_mcp_enabled: bool = False
    amap_mcp_url: str = "https://dashscope.aliyuncs.com/api/v1/mcps/amap-maps/sse"
    amap_mcp_transport: Literal["sse", "streamable_http"] = "sse"
    dashscope_api_key: SecretStr | None = Field(default=None, validation_alias=AliasChoices(
        "DASHSCOPE_API_KEY", "AGENTIC_RAG_DASHSCOPE_API_KEY", "dashscope_api_key"))

    @field_validator("mcp_servers")
    @classmethod
    def _unique_mcp_ids(cls, value: tuple[McpServerConfig, ...]) -> tuple[McpServerConfig, ...]:
        if len({server.id for server in value}) != len(value) or len(value) > 16:
            raise ValueError("MCP service IDs must be unique; maximum 16 services")
        return value

    @field_validator("index_generation")
    @classmethod
    def _canonical_index_generation(cls, value: str) -> str:
        return validate_writable_index_generation(value)

    @field_validator("embedding_tokenizer_model")
    @classmethod
    def _nonblank_embedding_tokenizer(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("embedding_tokenizer_model must not be blank")
        return value

    @field_validator("mem0_collection", "mem0_embedding_model", "mem0_llm_model")
    @classmethod
    def _nonblank_mem0_names(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Mem0 names must not be blank")
        return value.strip()


@lru_cache
def get_settings() -> Settings:
    """Return the process-wide validated settings instance."""
    return Settings()  # type: ignore[call-arg]
