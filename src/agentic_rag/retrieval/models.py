"""Pydantic contracts for tenant-scoped retrieval."""

from datetime import date
from typing import Annotated, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    model_validator,
)


class DateRange(BaseModel):
    """An optional inclusive date range supplied as a retrieval selector."""

    start: date | None = None
    end: date | None = None

    @model_validator(mode="after")
    def ordered(self) -> "DateRange":
        """Reject ranges whose end precedes their start."""
        if self.start and self.end and self.start > self.end:
            raise ValueError("date range start must not exceed end")
        return self


class RetrievalRequest(BaseModel):
    """Agent-provided selectors that exclude all server-owned scope fields."""

    model_config = ConfigDict(extra="forbid")

    query: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
    search_type: str | None = None
    document_ids: tuple[str, ...] = ()
    content_types: tuple[str, ...] = ()
    date_range: DateRange | None = None
    top_k_override: int | None = Field(default=None, ge=1, le=80)


class SearchFilter(BaseModel):
    """Fully scoped filter that retrieval adapters must apply server-side."""

    user_id: str
    is_active: Literal[True] = True
    index_generation: str
    search_type: str | None = None
    document_ids: tuple[str, ...] = ()
    content_types: tuple[str, ...] = ()
    date_range: DateRange | None = None


class ChildHit(BaseModel):
    """A scored child chunk returned by one retrieval lane."""

    child_id: str
    parent_id: str
    user_id: str
    document_id: str
    document_version_id: str
    content: str
    ast_locator: str
    lane: Literal["dense", "bm25"]
    lane_rank: int
    score: float


class ParentEvidence(BaseModel):
    """A parent document fragment with the child hits that selected it."""

    parent_id: str
    document_id: str
    document_version_id: str
    content: str
    child_hits: tuple[ChildHit, ...]
    rerank_score: float


class EvidenceBatch(BaseModel):
    """The evidence supplied by one retrieval execution."""

    query: str
    parents: tuple[ParentEvidence, ...]
    degraded_components: tuple[str, ...] = ()
