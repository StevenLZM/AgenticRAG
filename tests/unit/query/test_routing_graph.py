from dataclasses import replace
import httpx
from openai import APIConnectionError

import pytest
from langgraph.checkpoint.memory import InMemorySaver

from agentic_rag.models.schemas import EvidenceGradeV2, RouteAssessment
from agentic_rag.query.graph import build_query_graph
from agentic_rag.query.routing_context import RoutingContext
from agentic_rag.query.routing_policy import RuntimeCapabilities
from tests.unit.query.test_chat import Gateway
from tests.unit.query.test_graph import _deps, _state
from tests.unit.query.test_router_fast_path import FakeGrader


def classify(source, complexity="none"):
    return RouteAssessment(required_sources=[source], retrieval_complexity=complexity,
        needs_clarification=False, normalized_query="今天北京天气如何",
        reason_code="realtime_information_required")


class Reader:
    def __init__(self):
        self.calls = 0

    async def load(self, scope, *, run_id, thread_id):
        self.calls += 1
        return RoutingContext(run_id=run_id, user_id=scope.user_id, thread_id=thread_id,
                              requested_at="2026-10-01T00:01:00+08:00")


@pytest.mark.parametrize("misroute", [False, True])
async def test_v2_graph_weather_never_researches(misroute):
    deps, memory, retrieval, _ = _deps()
    reader = Reader()
    gateway = Gateway(classify("knowledge_base", "single") if misroute else classify("external_realtime"))
    deps = replace(deps, gateway=gateway, conversations=reader,
                   capabilities=RuntimeCapabilities(knowledge_base=True),
                   evidence_grader=FakeGrader(EvidenceGradeV2(decision="insufficient", gap_type="external_realtime_required")))
    state = _state()
    state["request"].update(question="今天北京天气如何", thread_id="t")
    result = await build_query_graph(deps).ainvoke(state)
    assert result["answer"]["route"] == "chat"
    assert result["answer"]["status"] == "cannot_answer"
    assert "research_agent_loop" not in result["executed_path"]
    assert retrieval.calls == int(misroute)
    assert len(gateway.calls) == 1
    assert len(memory.stored) == 1


async def test_second_run_does_not_reuse_routing_or_memory_context():
    deps, memory, retrieval, _ = _deps()
    reader = Reader()
    deps = replace(deps, gateway=Gateway(classify("external_realtime")),
                   capabilities=RuntimeCapabilities(knowledge_base=True), conversations=reader)
    graph = build_query_graph(deps, InMemorySaver())
    state = _state()
    state["request"]["thread_id"] = "t"
    config = {"configurable": {"thread_id": f"query:{state['scope']['user_id']}:t"}}
    first = await graph.ainvoke(state, config)
    second = dict(state, run_id="run-next")
    result = await graph.ainvoke(second, config)
    assert reader.calls == 2
    assert result["routing_context"]["run_id"] == "run-next"
    assert result["executed_path"] == first["executed_path"]
    assert memory.loads == 2


async def test_v2_document_route_keeps_audits_and_citations():
    deps, _, retrieval, _ = _deps()
    deps = replace(deps, gateway=Gateway(classify("knowledge_base", "single")),
                   capabilities=RuntimeCapabilities(knowledge_base=True),
                   evidence_grader=FakeGrader(EvidenceGradeV2(decision="sufficient", gap_type="none")))
    result = await build_query_graph(deps).ainvoke(_state())
    assert retrieval.calls == 1
    assert result["route"]["route"] == "fast_rag"
    assert "faithfulness" in result["executed_path"] and "citation" in result["executed_path"]
    assert result["answer"]["segments"][0]["evidence_ids"]


async def test_sdk_failure_reaches_finalize_without_retrying_graph():
    deps, memory, retrieval, _ = _deps()
    gateway = Gateway(APIConnectionError(request=httpx.Request("POST", "https://model.invalid")))
    deps = replace(deps, gateway=gateway, capabilities=RuntimeCapabilities(knowledge_base=True))
    result = await build_query_graph(deps).ainvoke(_state())
    assert result["response_mode"] == "technical_error"
    assert result["executed_path"] == ["memory_loader", "route", "chat", "finalize"]
    assert retrieval.calls == 0 and len(gateway.calls) == 1
    assert len(memory.stored) == 1
