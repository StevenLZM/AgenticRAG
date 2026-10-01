"""Strict structured-output schemas shared by the query runtime."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator


InformationSource = Literal[
    "general", "conversation", "knowledge_base", "external_realtime", "external_lookup", "unknown"
]
GapType = Literal[
    "none", "missing_facts", "multi_step_required", "query_ambiguous",
    "external_realtime_required", "external_lookup_required", "irrelevant_results", "unknown",
]


class RouteAssessment(BaseModel):
    """Model interpretation of a request, never an execution permission."""

    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)
    required_sources: tuple[InformationSource, ...] = Field(min_length=1, max_length=5)
    retrieval_complexity: Literal["none", "single", "multi"]
    needs_clarification: bool
    normalized_query: str = Field(min_length=1, max_length=8_000)
    reason_code: Literal[
        "general_conversation", "conversation_reference", "knowledge_base_lookup",
        "knowledge_base_research", "realtime_information_required", "external_lookup_required",
        "mixed_sources", "clarification_required",
    ]

    @model_validator(mode="after")
    def consistent_sources(self) -> RouteAssessment:
        sources = set(self.required_sources)
        if len(sources) != len(self.required_sources):
            raise ValueError("sources must be unique")
        if "unknown" in sources and len(sources) != 1:
            raise ValueError("unknown must stand alone")
        if ("knowledge_base" in sources) != (self.retrieval_complexity != "none"):
            raise ValueError("retrieval complexity must match knowledge base need")
        return self


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
    gap_type: GapType | None = None
    gaps: tuple[
        Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=1_000)], ...
    ] = Field(default=(), max_length=12)


class EvidenceGradeV2(EvidenceGrade):
    """Strict new output; legacy grades are read through EvidenceGrade only."""

    gap_type: GapType

    @model_validator(mode="after")
    def consistent_gap(self) -> EvidenceGradeV2:
        if self.decision == "sufficient" and self.gap_type != "none":
            raise ValueError("sufficient evidence has no gap")
        if self.decision == "insufficient" and self.gap_type == "none":
            raise ValueError("insufficient evidence needs a gap")
        return self
