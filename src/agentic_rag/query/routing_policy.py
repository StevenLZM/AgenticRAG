"""Deterministic source/capability decisions. This module grants no permissions."""

from typing import Literal

from pydantic import BaseModel, ConfigDict

from agentic_rag.models.schemas import EvidenceGrade, InformationSource, RouteAssessment
from agentic_rag.query.state import QueryState, question_from_state


class RuntimeCapabilities(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    version: Literal["capabilities-v1"] = "capabilities-v1"
    knowledge_base: bool
    external_realtime: Literal[False] = False
    external_lookup: Literal[False] = False


class PolicyDecision(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    next_node: Literal["chat", "fast_rag", "research_agent", "generate", "end"]
    route: Literal["chat", "fast_rag", "research"] | None = None
    response_mode: Literal["conversation", "capability_unavailable", "clarify", "technical_error"] | None = None
    termination_reason: str | None = None
    reason_code: str
    missing_sources: tuple[InformationSource, ...] = ()


def technical_failure(code: str) -> PolicyDecision:
    if code not in {
        "router_unavailable", "router_schema_invalid", "routing_context_unavailable",
        "retrieval_unavailable", "evidence_grader_unavailable",
    }:
        raise ValueError("unknown technical error")
    return PolicyDecision(next_node="chat", route="chat", response_mode="technical_error",
                          termination_reason="cannot_answer", reason_code=code)


def _clarify(missing: tuple[InformationSource, ...] = ()) -> PolicyDecision:
    return PolicyDecision(next_node="chat", route="chat", response_mode="clarify",
                          termination_reason="clarify", reason_code="clarification_required",
                          missing_sources=missing)


def _unavailable(missing: tuple[InformationSource, ...], *, mixed: bool) -> PolicyDecision:
    if mixed:
        return _clarify(missing)
    return PolicyDecision(next_node="chat", route="chat", response_mode="capability_unavailable",
                          termination_reason="cannot_answer", reason_code="capability_unavailable",
                          missing_sources=missing)


def decide_route(assessment: RouteAssessment, capabilities: RuntimeCapabilities) -> PolicyDecision:
    sources = assessment.required_sources
    if assessment.needs_clarification or "unknown" in sources:
        return _clarify()
    missing = tuple(s for s in sources if s in {"external_realtime", "external_lookup"})
    if missing:
        return _unavailable(missing, mixed="knowledge_base" in sources)
    if "knowledge_base" in sources:
        if not capabilities.knowledge_base:
            return technical_failure("retrieval_unavailable")
        if assessment.retrieval_complexity == "multi":
            return PolicyDecision(next_node="research_agent", route="research", reason_code="knowledge_base_research")
        return PolicyDecision(next_node="fast_rag", route="fast_rag", reason_code="knowledge_base_lookup")
    return PolicyDecision(next_node="chat", route="chat", response_mode="conversation", reason_code="general_conversation")


def decide_grade(
    grade: EvidenceGrade, assessment: RouteAssessment | None, capabilities: RuntimeCapabilities,
    *, research_attempts: int, max_research_rounds: int,
) -> PolicyDecision:
    if grade.decision == "refuse":
        return PolicyDecision(next_node="end", termination_reason="refuse", reason_code="refuse")
    if grade.decision == "sufficient":
        return PolicyDecision(next_node="generate", reason_code="sufficient")
    if grade.decision == "clarify" or grade.gap_type == "query_ambiguous":
        return _clarify()
    if grade.gap_type in {"external_realtime_required", "external_lookup_required"}:
        missing: tuple[InformationSource, ...] = (
            "external_realtime" if grade.gap_type == "external_realtime_required" else "external_lookup",
        )
        # An initially mistaken KB classification does not establish a mixed
        # task: the assessment must explicitly require both sources.
        mixed = bool(assessment and "knowledge_base" in assessment.required_sources
                     and any(s in assessment.required_sources for s in missing))
        return _unavailable(missing, mixed=mixed)
    if not capabilities.knowledge_base:
        return technical_failure("retrieval_unavailable")
    if research_attempts >= max_research_rounds:
        return PolicyDecision(next_node="end", termination_reason="research_round_limit", reason_code="research_round_limit")
    return PolicyDecision(next_node="research_agent", route="research", reason_code=grade.gap_type or "unknown")


def policy_update(state: QueryState, decision: PolicyDecision, *, normalized_query: str | None = None) -> dict[str, object]:
    update: dict[str, object] = {
        "next_node": decision.next_node, "response_mode": decision.response_mode,
        "policy_decision": decision.model_dump(mode="json"),
        "termination_reason": decision.termination_reason,
    }
    if decision.route:
        previous = state.get("route") or {}
        update["route"] = {"route": decision.route, "reason_code": decision.reason_code,
                           "normalized_query": normalized_query or previous.get("normalized_query") or question_from_state(state)}
    if decision.next_node == "end" and decision.termination_reason:
        update["answer"] = {"status": decision.termination_reason}
    return update
