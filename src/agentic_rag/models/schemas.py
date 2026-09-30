"""Strict structured-output schemas shared by the query runtime."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints


class RouteDecision(BaseModel):
    """Chat, single-retrieval, or research routing after memory loading."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    route: Literal["chat", "fast_rag", "research"]
    normalized_query: str = Field(min_length=1, max_length=8_000)
    reason_code: str = Field(min_length=1, max_length=256)


class EvidenceGrade(BaseModel):
    """A strict, bounded decision about whether evidence can support an answer."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    decision: Literal["sufficient", "insufficient", "clarify", "refuse"]
    gaps: tuple[
        Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=1_000)], ...
    ] = Field(default=(), max_length=12)
