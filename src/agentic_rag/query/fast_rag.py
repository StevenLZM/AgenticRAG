"""Single-retrieval Fast RAG path and bounded evidence sufficiency gate."""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Protocol

from pydantic import ValidationError

from agentic_rag.models.schemas import EvidenceGrade, RouteDecision
from agentic_rag.observability.logging import emit_degradation
from agentic_rag.query.evidence_builder import EvidenceBuilder, EvidenceCoverageTarget, PackedEvidence
from agentic_rag.query.state import QueryState, question_from_state, scope_from_state, snapshot_from_state
from agentic_rag.retrieval.graph import RetrievalService, RetrievalUnavailable
from agentic_rag.retrieval.models import RetrievalRequest
from agentic_rag.models.schemas import RouteAssessment
from agentic_rag.query.audit import EvidenceGradingUnavailable
from agentic_rag.query.phases import PhaseReporter, report_safely
from agentic_rag.query.routing_context import RoutingContext, reasoning_question
from agentic_rag.query.routing_policy import RuntimeCapabilities, decide_grade, policy_update, technical_failure


class EvidenceGrader(Protocol):
    """Narrow light-model/heuristic boundary for evidence sufficiency."""

    async def grade(
        self,
        question: str,
        packed_evidence: PackedEvidence,
        *,
        scope: object,
        snapshot: object,
        routing_context: RoutingContext | None = None,
        capabilities: RuntimeCapabilities | None = None,
    ) -> EvidenceGrade | object: ...


@dataclass(frozen=True, slots=True)
class FastRagDependencies:
    """Process-owned collaborators for the fast path, never graph state."""

    retrieval: RetrievalService
    evidence_builder: EvidenceBuilder
    evidence_grader: EvidenceGrader
    capabilities: RuntimeCapabilities | None = None


async def run_fast_rag(
    state: QueryState, dependencies: FastRagDependencies, *, report_phase: PhaseReporter | None = None
) -> dict[str, object]:
    """Retrieve once, pack once, then either generate, research, or terminate."""
    route = _route_from_state(state)
    if route.route != "fast_rag":
        return {"next_node": "research_agent"}
    scope = scope_from_state(state)
    snapshot = snapshot_from_state(state)
    question = reasoning_question(state) if dependencies.capabilities is not None else question_from_state(state)
    request = RetrievalRequest(query=route.normalized_query)
    await report_safely(report_phase, "retrieving")
    try:
        batch = await dependencies.retrieval.retrieve(request, scope, snapshot)
    except asyncio.CancelledError:
        raise
    except (OSError, TimeoutError, ConnectionError, RetrievalUnavailable) as error:
        await emit_degradation(
            component="retrieval",
            reason="retrieval_unavailable",
            run_id=state["run_id"],
            snapshot_id=snapshot.snapshot_id,
            attempt=1,
            retryable=True,
            outcome="degraded",
            event_type="RETRIEVAL_DEGRADED",
        )
        if dependencies.capabilities is not None:
            return {**policy_update(state, technical_failure("retrieval_unavailable")),
                    "errors": [*state.get("errors", []), {"code": "retrieval_unavailable"}]}
        return {
            "research": {"gaps": ["fast retrieval unavailable"]},
            "errors": [
                *state.get("errors", []),
                {"code": "fast_retrieval_unavailable", "detail": type(error).__name__},
            ],
            "next_node": "research_agent",
        }
    target = EvidenceCoverageTarget(target_id=f"query:{state['run_id']}", description=question)
    if target.target_id not in batch.target_ids:
        batch = batch.model_copy(update={"target_ids": (*batch.target_ids, target.target_id)})
    packed = dependencies.evidence_builder.build([batch], [target], scope, snapshot)
    base = {
        "evidence": [item.model_dump(mode="json") for item in packed.items],
        "packed_context": packed.model_dump(mode="json"),
        "retrieval_batches": [batch.model_dump(mode="json")],
    }
    await report_safely(report_phase, "auditing")
    try:
        if dependencies.capabilities is not None:
            result = await dependencies.evidence_grader.grade(question, packed, scope=scope, snapshot=snapshot,
                capabilities=dependencies.capabilities, routing_context=(
                    RoutingContext.model_validate(state["routing_context"]) if state.get("routing_context") else None))
        else:
            result = await dependencies.evidence_grader.grade(question, packed, scope=scope, snapshot=snapshot)
        grade = _grade(result)
    except asyncio.CancelledError:
        raise
    except (OSError, TimeoutError, ConnectionError, ValidationError, TypeError, ValueError, EvidenceGradingUnavailable) as error:
        await emit_degradation(
            component="llm",
            reason="model_unavailable",
            run_id=state["run_id"],
            snapshot_id=snapshot.snapshot_id,
            attempt=1,
            retryable=True,
            outcome="degraded",
        )
        if dependencies.capabilities is not None:
            return {**base, **policy_update(state, technical_failure("evidence_grader_unavailable")),
                    "errors": [*state.get("errors", []), {"code": "evidence_grader_unavailable"}]}
        return {
            **base,
            "research": {"gaps": ["evidence grading unavailable"]},
            "errors": [
                *state.get("errors", []),
                {"code": "evidence_grader_unavailable", "detail": type(error).__name__},
            ],
            "next_node": "research_agent",
        }
    if dependencies.capabilities is not None:
        assessment = RouteAssessment.model_validate(state["route_assessment"]) if state.get("route_assessment") else None
        decision = decide_grade(grade, assessment, dependencies.capabilities,
                                research_attempts=state.get("research_attempt_count", 0),
                                max_research_rounds=snapshot.max_research_rounds)
        return {**base, **policy_update(state, decision), "last_evidence_grade": grade.model_dump(mode="json"),
                "research": {**state.get("research", {}), "gaps": list(grade.gaps)}}
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
