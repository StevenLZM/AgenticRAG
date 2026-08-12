"""Single-retrieval Fast RAG path and bounded evidence sufficiency gate."""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Protocol

from pydantic import ValidationError

from agentic_rag.models.schemas import EvidenceGrade, RouteDecision
from agentic_rag.query.evidence_builder import EvidenceBuilder, EvidenceCoverageTarget, PackedEvidence
from agentic_rag.query.state import QueryState, question_from_state, scope_from_state, snapshot_from_state
from agentic_rag.retrieval.graph import RetrievalService, RetrievalUnavailable
from agentic_rag.retrieval.models import RetrievalRequest


class EvidenceGrader(Protocol):
    """Narrow light-model/heuristic boundary for evidence sufficiency."""

    async def grade(
        self,
        question: str,
        packed_evidence: PackedEvidence,
        *,
        scope: object,
        snapshot: object,
    ) -> EvidenceGrade | object: ...


@dataclass(frozen=True, slots=True)
class FastRagDependencies:
    """Process-owned collaborators for the fast path, never graph state."""

    retrieval: RetrievalService
    evidence_builder: EvidenceBuilder
    evidence_grader: EvidenceGrader


async def run_fast_rag(
    state: QueryState, dependencies: FastRagDependencies
) -> dict[str, object]:
    """Retrieve once, pack once, then either generate, research, or terminate."""
    route = _route_from_state(state)
    if route.route != "fast_rag":
        return {"next_node": "research_agent"}
    scope = scope_from_state(state)
    snapshot = snapshot_from_state(state)
    question = question_from_state(state)
    request = RetrievalRequest(query=route.normalized_query)
    try:
        batch = await dependencies.retrieval.retrieve(request, scope, snapshot)
    except asyncio.CancelledError:
        raise
    except (OSError, TimeoutError, ConnectionError, RetrievalUnavailable) as error:
        return {
            "research": {"gaps": ["fast retrieval unavailable"]},
            "errors": [
                *state.get("errors", []),
                {"code": "fast_retrieval_unavailable", "detail": str(error)},
            ],
            "next_node": "research_agent",
        }
    target = EvidenceCoverageTarget(target_id=f"query:{state['run_id']}", description=question)
    packed = dependencies.evidence_builder.build([batch], [target], scope, snapshot)
    base = {
        "evidence": [item.model_dump(mode="json") for item in packed.items],
        "packed_context": packed.model_dump(mode="json"),
    }
    try:
        grade = _grade(
            await dependencies.evidence_grader.grade(
                question, packed, scope=scope, snapshot=snapshot
            )
        )
    except asyncio.CancelledError:
        raise
    except (OSError, TimeoutError, ConnectionError, ValidationError, TypeError, ValueError) as error:
        return {
            **base,
            "research": {"gaps": ["evidence grading unavailable"]},
            "errors": [*state.get("errors", []), {"code": "evidence_grader_unavailable", "detail": str(error)}],
            "next_node": "research_agent",
        }
    if grade.decision == "sufficient":
        return {**base, "next_node": "generate"}
    if grade.decision == "insufficient":
        return {
            **base,
            "research": {"gaps": list(grade.gaps)},
            "next_node": "research_agent",
        }
    return {
        **base,
        "answer": {"status": grade.decision},
        "termination_reason": grade.decision,
        "next_node": "end",
    }


def _route_from_state(state: QueryState) -> RouteDecision:
    route = state.get("route")
    if not isinstance(route, Mapping):
        raise ValueError("fast path requires a route decision")
    return RouteDecision.model_validate(route)


def _grade(value: object) -> EvidenceGrade:
    """Support a strict grade and a ModelResponse-like wrapper from a gateway."""
    if isinstance(value, EvidenceGrade):
        return value
    nested = getattr(value, "value", value)
    if isinstance(nested, Mapping):
        return EvidenceGrade.model_validate(nested)
    raise TypeError("evidence grader must return an EvidenceGrade")
