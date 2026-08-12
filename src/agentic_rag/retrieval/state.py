"""Safe state values that flow through the retrieval LangGraph subgraph."""

from __future__ import annotations

from typing import TypedDict

from pydantic import BaseModel, ConfigDict

from agentic_rag.domain.models import UserScope
from agentic_rag.retrieval.models import (
    ChildHit,
    EvidenceBatch,
    ParentEvidence,
    RetrievalRequest,
    SearchFilter,
)
from agentic_rag.runtime.models import RuntimeConfigSnapshot


class LaneFailure(BaseModel):
    """Sanitized operational failure recorded for a degraded recall lane."""

    model_config = ConfigDict(frozen=True)

    component: str
    error_type: str
    message: str


class RetrievalState(TypedDict, total=False):
    """Serialized request data and intermediate retrieval results only.

    Process-owned dependencies deliberately live in the compiled graph closure,
    rather than in this state, so an invocation cannot smuggle alternate
    indexes, tenant filters, or model clients into the pipeline.
    """

    request: RetrievalRequest
    scope: UserScope
    snapshot: RuntimeConfigSnapshot
    search_filter: SearchFilter
    dense_hits: list[ChildHit]
    bm25_hits: list[ChildHit]
    fused_hits: list[ChildHit]
    reranked_hits: list[ChildHit]
    selected_parents: list[ParentEvidence]
    hydrated_parents: list[ParentEvidence]
    degraded_components: tuple[str, ...]
    lane_failures: dict[str, LaneFailure]
    candidate_counts: dict[str, int]
    timings_ms: dict[str, float]
    evidence_batch: EvidenceBatch
