"""Immutable runtime configuration contract."""

import hashlib
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class RuntimeConfigSnapshot(BaseModel):
    """Versioned configuration captured at the start of a runtime operation."""

    model_config = ConfigDict(frozen=True)

    app_version: str
    graph_version: str
    prompt_version: str
    main_model_id: str
    light_model_id: str
    embedding_model: str
    embedding_dimensions: Literal[1024]
    reranker_version: str
    retrieval_config_version: str
    index_generation: str
    memory_config_version: str
    max_research_rounds: int = Field(default=4, ge=1, le=4)
    max_answer_revisions: int = Field(default=1, ge=0, le=1)
    query_run_timeout_seconds: int = Field(default=300, ge=30, le=300)
    max_evidence_tokens: int = Field(default=12_000, ge=1_000, le=12_000)
    research_context_soft_limit_tokens: int = Field(default=16_000, ge=4_000)
    max_parallel_subagents_per_run: int = Field(default=3, ge=1, le=3)

    @property
    def snapshot_id(self) -> str:
        """Return the content-addressed identifier for this configuration."""
        payload = self.model_dump_json(exclude_none=True)
        return hashlib.sha256(payload.encode()).hexdigest()
