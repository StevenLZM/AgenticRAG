"""Immutable runtime configuration contract."""

import hashlib
from collections.abc import Mapping
from types import MappingProxyType
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


PromptHashes = tuple[tuple[str, str], ...]


class RuntimeConfigSnapshot(BaseModel):
    """Versioned configuration captured at the start of a runtime operation."""

    model_config = ConfigDict(frozen=True)

    app_version: str
    graph_version: str
    prompt_version: str
    prompt_hashes: PromptHashes = ()
    main_model_id: str
    light_model_id: str
    deepseek_protocol: Literal["auto", "chat", "responses"] = "auto"
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

    @model_validator(mode="before")
    @classmethod
    def _normalize_prompt_hashes(cls, value: Any) -> Any:
        """Accept a Run-creation mapping but persist an ordered immutable tuple."""
        if not isinstance(value, Mapping):
            return value
        raw_hashes = value.get("prompt_hashes")
        if isinstance(raw_hashes, Mapping):
            normalized = dict(value)
            normalized["prompt_hashes"] = tuple(sorted(raw_hashes.items()))
            return normalized
        return value

    @property
    def prompt_hash_map(self) -> Mapping[str, str]:
        """Return the versioned prompt hashes in a convenient immutable mapping."""
        return MappingProxyType(dict(self.prompt_hashes))

    @property
    def snapshot_id(self) -> str:
        """Return the content-addressed identifier for this configuration."""
        payload = self.model_dump_json(exclude_none=True)
        return hashlib.sha256(payload.encode()).hexdigest()
