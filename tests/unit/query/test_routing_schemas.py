import pytest
from pydantic import ValidationError

from agentic_rag.models import schemas


def test_v2_contract_exists_and_preserves_legacy():
    assert hasattr(schemas, "RouteAssessment")
    assert schemas.EvidenceGrade(decision="insufficient").gap_type is None
    assert schemas.RouteDecision(route="fast_rag", normalized_query="q", reason_code="old").route == "fast_rag"


@pytest.mark.parametrize("sources,complexity", [
    ([], "none"), (["unknown", "general"], "none"),
    (["general", "general"], "none"), (["general"], "single"),
    (["knowledge_base"], "none"),
])
def test_route_assessment_combinations(sources, complexity):
    assert hasattr(schemas, "RouteAssessment")
    with pytest.raises(ValidationError):
        schemas.RouteAssessment(required_sources=sources, retrieval_complexity=complexity,
                                needs_clarification=False, normalized_query="q",
                                reason_code="general_conversation")


def test_v2_grade_rejects_contradictions_and_missing_gap():
    assert hasattr(schemas, "EvidenceGradeV2")
    for payload in [dict(decision="sufficient"),
                    dict(decision="sufficient", gap_type="external_realtime_required"),
                    dict(decision="insufficient", gap_type="none")]:
        with pytest.raises(ValidationError):
            schemas.EvidenceGradeV2(**payload)
