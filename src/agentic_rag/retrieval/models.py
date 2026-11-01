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

from agentic_rag.models.contextual import (
    ContextArtifactRef,
    SHA256_PATTERN,
    text_sha256,
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
    contextualized_content: str | None = None
    representation_version: Literal["context-v1"] | None = None
    content_hash: str | None = Field(default=None, pattern=SHA256_PATTERN)
    source_child_id: str | None = None
    source_content_hash: str | None = Field(default=None, pattern=SHA256_PATTERN)
    context_ref: ContextArtifactRef | None = None
    context_claim_ids: tuple[str, ...] = ()
    build_fingerprint: str | None = None
    context_prefix: str | None = None
    enrichment_status: Literal["enriched", "structural_only", "partial"] | None = None

    @model_validator(mode="after")
    def _valid_representation(self) -> "ChildHit":
        if self.representation_version is None and (
            self.source_child_id is not None
            or self.source_content_hash is not None
            or self.context_ref is not None
            or self.context_claim_ids
            or self.build_fingerprint is not None
            or self.context_prefix is not None
            or self.enrichment_status is not None
        ):
            raise ValueError("context metadata requires representation_version")
        if self.representation_version is not None:
            if (
                "context_claim_ids" not in self.model_fields_set
                or self.context_prefix is None
                or self.enrichment_status is None
            ):
                raise ValueError("context-v1 requires explicit context metadata")
            if not all(
                (
                    self.contextualized_content and self.contextualized_content.strip(),
                    self.content_hash,
                    self.source_child_id and self.source_child_id.strip(),
                    self.source_content_hash,
                    self.context_ref,
                    self.build_fingerprint and self.build_fingerprint.strip(),
                )
            ):
                raise ValueError("context-v1 requires complete representation fields")
            if text_sha256(self.content) != self.source_content_hash:
                raise ValueError("original content hash mismatch")
            if text_sha256(self.contextualized_content) != self.content_hash:
                raise ValueError("search content hash mismatch")
            if bool(self.context_prefix) != bool(self.context_claim_ids) or (
                self.enrichment_status == "structural_only"
            ) != (not self.context_claim_ids):
                raise ValueError(
                    "context prefix, claims and enrichment status disagree"
                )
            prefix = self.context_prefix + "\n\n" if self.context_prefix else ""
            if not self.contextualized_content.startswith(prefix):
                raise ValueError("rendered context prefix mismatch")
            if self.content not in self.contextualized_content[len(prefix) :]:
                raise ValueError("search representation must retain original content")
            if len(self.context_claim_ids) != len(set(self.context_claim_ids)):
                raise ValueError("context claim IDs must be unique")
        return self

    @property
    def search_content(self) -> str:
        """Select validated context-v1 text or explicit legacy original content."""
        self._valid_representation()
        return (
            self.contextualized_content if self.representation_version else self.content
        )

    @property
    def search_identity(self) -> tuple[object, ...]:
        """Immutable source/search identity, excluding lane-specific scores."""
        return (
            self.parent_id,
            self.user_id,
            self.document_id,
            self.document_version_id,
            self.content,
            self.ast_locator,
            self.representation_version,
            self.content_hash,
            self.source_child_id,
            self.source_content_hash,
            self.context_ref,
            self.context_claim_ids,
            self.build_fingerprint,
            self.search_content,
            self.context_prefix,
            self.enrichment_status,
        )

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


class RankedCandidates(BaseModel):
    """Retrieval output at rerank; contains no hydrated parents or grade."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    query: str
    hits: tuple[ChildHit, ...]
    user_id: str
    snapshot_id: str
    index_generation: str
    document_ids: tuple[str, ...] = ()
    degraded_components: tuple[str, ...] = ()
    observation: RetrievalObservation | None = None
