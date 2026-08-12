"""Strict structured-output schemas shared by the query runtime."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class RouteDecision(BaseModel):
    """The only two routes that a query may take after memory loading."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    route: Literal["fast_rag", "research"]
    normalized_query: str = Field(min_length=1, max_length=8_000)
    reason_code: str = Field(min_length=1, max_length=256)
