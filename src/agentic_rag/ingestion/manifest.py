"""Deterministic manifest for one fully staged document version."""

from __future__ import annotations

import hashlib
import json
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, computed_field, field_validator


class VersionManifest(BaseModel):
    """Counts and immutable versions that gate later publication.

    ``manifest_hash`` is derived from the canonical JSON payload and cannot be
    supplied by callers. The Artifact Store writes exactly ``payload()`` so its
    SHA-256 is the same value.
    """

    model_config = ConfigDict(frozen=True)

    canonical_ast_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    parent_count: int = Field(gt=0)
    child_count: int = Field(gt=0)
    embedding_model: Literal["text-embedding-v3"]
    embedding_dimensions: Literal[1024]
    index_generation: str

    @field_validator("index_generation")
    @classmethod
    def _index_generation_not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("index_generation must not be blank")
        return value

    def payload(self) -> dict[str, Any]:
        """Return the stable payload persisted as the Manifest Artifact."""
        return {
            "canonical_ast_sha256": self.canonical_ast_sha256,
            "parent_count": self.parent_count,
            "child_count": self.child_count,
            "embedding_model": self.embedding_model,
            "embedding_dimensions": self.embedding_dimensions,
            "index_generation": self.index_generation,
        }

    @computed_field  # type: ignore[prop-decorator]
    @property
    def manifest_hash(self) -> str:
        encoded = json.dumps(
            self.payload(),
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()
