"""Pydantic contracts for tenant-scoped retrieval."""

from datetime import date
from typing import Annotated, Literal

from pydantic import (
    AliasChoices,
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
    """A child chunk with scores from each completed ranking stage."""

    child_id: str
    parent_id: str
    user_id: str
    document_id: str
    document_version_id: str
    content: str
    ast_locator: str
    lane: Literal["dense", "bm25"]
    lane_rank: int
    retrieval_score: float = Field(
        validation_alias=AliasChoices("retrieval_score", "score")
    )
    rrf_score: float | None = None
    rerank_score: float | None = None

    @property
    def ranking_score(self) -> float:
        """Return the latest available score without mixing ranking stages."""
        if self.rerank_score is not None:
            return self.rerank_score
        if self.rrf_score is not None:
            return self.rrf_score
        return self.retrieval_score


class ParentEvidence(BaseModel):
    """A parent document fragment with the child hits that selected it."""

    parent_id: str
    document_id: str
    document_version_id: str
    content: str
    child_hits: tuple[ChildHit, ...]
    retrieval_score: float = 0.0
    rrf_score: float | None = None
    rerank_score: float | None = None
    heading_path: tuple[str, ...] = ()

    @property
    def ranking_score(self) -> float:
        """Return the score from the latest ranking stage that completed."""
        if self.rerank_score is not None:
            return self.rerank_score
        if self.rrf_score is not None:
            return self.rrf_score
        return self.retrieval_score


class RankedHitRef(BaseModel):
    """Content-free identity at one observed retrieval stage, in list order."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    child_id: str
    parent_id: str
    user_id: str
    document_id: str
    document_version_id: str
    retrieval_score: float | None = None
    rrf_score: float | None = None
    rerank_score: float | None = None


class RetrievalObservation(BaseModel):
    """Checkpoint-only stage rankings; not model prompts or public event payloads."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    schema_version: Literal[1] = 1
    user_id: str
    snapshot_id: str
    index_generation: str
    stages: dict[Literal["dense", "bm25", "rrf", "rerank"], tuple[RankedHitRef, ...]]
    selected_parent_ids: tuple[str, ...]
    hydrated_parent_ids: tuple[str, ...]
    candidate_budget: dict[str, int] = Field(default_factory=dict)
    timings_ms: dict[str, float] = Field(default_factory=dict)
    candidate_counts: dict[str, int] = Field(default_factory=dict)


class EvidenceBatch(BaseModel):
    """The evidence supplied by one retrieval execution.

    ``document_ids`` is propagated from the request by the retrieval graph;
    it is selector metadata, never a tenant, active-version, or index scope.
    """

    query: str
    parents: tuple[ParentEvidence, ...]
    degraded_components: tuple[str, ...] = ()
    document_ids: tuple[str, ...] = ()
    target_ids: tuple[str, ...] = ()
    observation: RetrievalObservation | None = None
