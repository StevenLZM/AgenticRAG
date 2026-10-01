import pytest

from agentic_rag.models.schemas import EvidenceGradeV2, RouteDecision
from agentic_rag.query import audit
from agentic_rag.query.evidence_builder import EvidenceBuilder, EvidenceCoverageTarget
from agentic_rag.query.fast_rag import FastRagDependencies, run_fast_rag
from agentic_rag.query.routing_policy import RuntimeCapabilities
from tests.unit.query.test_chat import Gateway
from tests.unit.query.test_router_fast_path import _batch, _initial_state, FakeGrader, FakeRetrieval, SCOPE, SNAPSHOT


async def test_grader_outage_is_not_semantic_gap():
    packed = EvidenceBuilder().build([_batch()], [EvidenceCoverageTarget(target_id="q", description="q")], SCOPE, SNAPSHOT)
    assert hasattr(audit, "EvidenceGradingUnavailable")
    with pytest.raises(audit.EvidenceGradingUnavailable):
        await audit.EvidenceGrader(Gateway(TimeoutError())).grade("q", packed, scope=SCOPE,
            snapshot=SNAPSHOT, capabilities=RuntimeCapabilities(knowledge_base=True))


@pytest.mark.parametrize("gap,node", [("external_realtime_required", "chat"), ("missing_facts", "research_agent")])
async def test_fast_rag_gap_policy(gap, node):
    state = _initial_state()
    state["route"] = RouteDecision(route="fast_rag", normalized_query="天气", reason_code="simple").model_dump()
    retrieval = FakeRetrieval()
    deps = FastRagDependencies(retrieval, EvidenceBuilder(),
        FakeGrader(EvidenceGradeV2(decision="insufficient", gap_type=gap)),
        capabilities=RuntimeCapabilities(knowledge_base=True))
    update = await run_fast_rag(state, deps)
    assert update["next_node"] == node
    assert len(retrieval.calls) == 1
    if node == "chat":
        assert update["response_mode"] == "capability_unavailable"
        assert update["evidence"]  # retained internally, not a public citation
