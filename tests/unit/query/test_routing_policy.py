import importlib.util

import pytest
from pydantic import ValidationError

from agentic_rag.models.schemas import EvidenceGrade, RouteAssessment


def policy():
    assert importlib.util.find_spec("agentic_rag.query.routing_policy") is not None
    from agentic_rag.query import routing_policy
    return routing_policy


def assessment(sources, complexity="none", clarify=False):
    return RouteAssessment(required_sources=sources, retrieval_complexity=complexity,
                           needs_clarification=clarify, normalized_query="问题",
                           reason_code="general_conversation")


@pytest.mark.parametrize("sources,complexity,clarify,node,mode", [
    (["general"], "none", False, "chat", "conversation"),
    (["conversation"], "none", False, "chat", "conversation"),
    (["external_realtime"], "none", False, "chat", "capability_unavailable"),
    (["external_lookup"], "none", False, "chat", "capability_unavailable"),
    (["knowledge_base"], "single", False, "fast_rag", None),
    (["knowledge_base"], "multi", False, "research_agent", None),
    (["knowledge_base", "external_realtime"], "single", False, "chat", "clarify"),
    (["unknown"], "none", False, "chat", "clarify"),
    (["knowledge_base"], "single", True, "chat", "clarify"),
])
def test_source_to_execution(sources, complexity, clarify, node, mode):
    p = policy()
    result = p.decide_route(assessment(sources, complexity, clarify), p.RuntimeCapabilities(knowledge_base=True))
    assert (result.next_node, result.response_mode) == (node, mode)


@pytest.mark.parametrize("decision,gap,expected", [
    ("sufficient", "none", "generate"), ("refuse", "external_realtime_required", "end"),
    ("clarify", "query_ambiguous", "chat"), ("insufficient", "query_ambiguous", "chat"),
    ("insufficient", "external_realtime_required", "chat"),
    ("insufficient", "external_lookup_required", "chat"),
    ("insufficient", "missing_facts", "research_agent"),
    ("insufficient", "multi_step_required", "research_agent"),
    ("insufficient", "irrelevant_results", "research_agent"),
    ("insufficient", None, "research_agent"),
])
def test_grade_policy(decision, gap, expected):
    p = policy()
    result = p.decide_grade(EvidenceGrade(decision=decision, gap_type=gap), None,
                           p.RuntimeCapabilities(knowledge_base=True), research_attempts=0, max_research_rounds=6)
    assert result.next_node == expected


def test_budget_and_runtime_capability_constraints():
    p = policy()
    caps = p.RuntimeCapabilities(knowledge_base=True)
    result = p.decide_grade(EvidenceGrade(decision="insufficient"), None, caps,
                           research_attempts=6, max_research_rounds=6)
    assert result.termination_reason == "research_round_limit"
    assert p.technical_failure("router_unavailable").response_mode == "technical_error"
    with pytest.raises(ValidationError):
        p.RuntimeCapabilities(knowledge_base=True, external_realtime=True)
    with pytest.raises(ValidationError):
        p.RuntimeCapabilities(knowledge_base=True, mcp_enabled=True)
