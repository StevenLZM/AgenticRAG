"""All strategies use scoped tools without granting new routing permissions."""
import importlib.util
from types import SimpleNamespace

import pytest

from agentic_rag.models.schemas import RouteAssessment
from agentic_rag.query.routing_policy import RuntimeCapabilities, decide_route
from tests.unit.query.test_router_fast_path import _initial_state


def tool_loop():
    assert importlib.util.find_spec("agentic_rag.query.tool_loop") is not None
    from agentic_rag.query import tool_loop
    return tool_loop


def test_maps_capability_routes_to_existing_fast_strategy():
    caps = RuntimeCapabilities(knowledge_base=True, tool_capabilities=("maps.search", "maps.route"))
    request = RouteAssessment(required_sources=("external_lookup",), retrieval_complexity="none",
                              required_capabilities=("maps.route",), execution_complexity="single",
                              needs_clarification=False, normalized_query="从杭州东站到西湖驾车路线",
                              reason_code="external_lookup_required")
    decision = decide_route(request, caps)
    assert decision.route == "fast_rag"
    assert decision.response_mode is None


def test_maps_registration_does_not_enable_unrelated_external_services():
    caps = RuntimeCapabilities(knowledge_base=True, tool_capabilities=("maps.search",))
    request = RouteAssessment(required_sources=("external_realtime",), retrieval_complexity="none",
                              required_capabilities=("finance.quote",), needs_clarification=False,
                              normalized_query="现在股价多少", reason_code="realtime_information_required")
    assert decide_route(request, caps).response_mode == "capability_unavailable"


def test_mixed_document_and_map_routes_research_without_new_route():
    caps = RuntimeCapabilities(knowledge_base=True, tool_capabilities=("maps.route",))
    request = RouteAssessment(required_sources=("knowledge_base", "external_lookup"),
                              retrieval_complexity="single", required_capabilities=("maps.route",),
                              needs_clarification=False, normalized_query="按出差安排规划路线",
                              reason_code="mixed_sources")
    assert decide_route(request, caps).route == "research"


class Gateway:
    def __init__(self, *actions):
        self.actions = iter(actions)
        self.calls = []

    async def complete_structured(self, call, schema):
        self.calls.append(call)
        return SimpleNamespace(value=next(self.actions))


class Runtime:
    def __init__(self):
        self.discovered = []
        self.called = []

    async def discover(self, query, context, limit=5):
        from agentic_rag.tool_runtime.models import ToolDefinition
        self.discovered.append((query, context))
        return (ToolDefinition(tool_id="local.calculator", adapter_id="local", name="calculator",
                               description="计算", input_schema={"type": "object", "properties": {
                                   "expression": {"type": "string"}}, "required": ["expression"]},
                               version="v1", source_kind="calculation", capabilities=("calculate",)),)

    async def call(self, tool_id, arguments, context, call_id):
        from agentic_rag.tool_runtime.models import ToolResult
        self.called.append((tool_id, arguments, context, call_id))
        return ToolResult(call_id=call_id, tool_id=tool_id, source_kind="calculation",
                          status="success", data={"value": 42}, observed_at="2026-10-02T00:00:00+00:00")


@pytest.mark.parametrize("strategy", ["chat", "fast_rag", "research"])
async def test_each_strategy_progressively_discovers_and_calls_same_runtime(strategy):
    loop = tool_loop()
    state = _initial_state()
    state["route"] = {"route": strategy, "normalized_query": "6*7", "reason_code": "test"}
    state["request"]["thread_id"] = "session-1"
    runtime = Runtime()
    gateway = Gateway({"action": "discover_tools", "query": "计算"},
                      {"action": "call_tool", "tool_id": "local.calculator", "arguments": {"expression": "6*7"}})
    state.update(await loop.run_tool_step(state, gateway, runtime, strategy=strategy))
    assert runtime.discovered[0][1].scope.user_id == state["scope"]["user_id"]
    assert "local.calculator" not in gateway.calls[0].messages[-1]["content"]
    state.update(await loop.run_tool_step(state, gateway, runtime, strategy=strategy))
    assert "local.calculator" in gateway.calls[1].messages[-1]["content"]
    assert runtime.called[0][2].session_id == "session-1"
    assert state["tool_state"]["results"][0]["data"]["value"] == 42


async def test_model_cannot_call_tool_that_was_not_loaded():
    loop = tool_loop()
    runtime = Runtime()
    state = _initial_state()
    gateway = Gateway({"action": "call_tool", "tool_id": "mcp.arbitrary.delete", "arguments": {}})
    update = await loop.run_tool_step(state, gateway, runtime, strategy="chat")
    assert not runtime.called
    assert update["tool_state"]["observations"][-1]["error_code"] == "tool_not_loaded"


async def test_fast_escalation_preserves_results_and_total_action_count():
    loop = tool_loop()
    state = _initial_state()
    state["tool_state"] = {"steps": 6, "strategy_steps": {"fast_rag": 6}, "loaded": [],
                           "results": [], "observations": [{"kind": "retained"}]}
    update = await loop.run_tool_step(state, Gateway(), Runtime(), strategy="fast_rag")
    assert update["next_node"] == "research_agent"
    assert update["tool_state"]["steps"] == 6
    assert update["tool_state"]["observations"] == [{"kind": "retained"}]


async def test_tool_result_cannot_be_replaced_with_unverified_model_text():
    loop = tool_loop()
    state = _initial_state()
    state["tool_state"] = {"steps": 1, "loaded": [], "results": [{
        "call_id": "call-1", "tool_id": "local.calculator", "source_kind": "calculation",
        "status": "success", "data": {"value": 42}, "observed_at": "2026-10-02T00:00:00+00:00",
    }], "observations": []}
    update = await loop.run_tool_step(state, Gateway({"action": "answer", "text": "结果是999"}),
                                      Runtime(), strategy="chat")
    assert "999" not in str(update.get("answer", {}))


async def test_finish_selects_final_call_and_limits_cards_together_with_prose():
    loop = tool_loop()
    state = _initial_state()
    results = [{"call_id": call_id, "tool_id": "mcp.amap_maps.maps_text_search",
                "source_kind": "external", "status": "success", "observed_at": "2026-10-02T00:00:00Z",
                "data": {"structured": {"pois": [{"name": f"{name}{i}", "id": f"B0{i}"} for i in range(3)]}}}
               for call_id, name in (("intermediate", "中间结果"), ("final", "最终地点"))]
    state["tool_state"] = {"steps": 3, "results": results}
    result = await loop.run_tool_step(state, Gateway({"action": "finish", "result_ids": ["final"], "max_items": 2}),
                                      Runtime(), strategy="fast_rag")
    assert len(result["answer"]["cards"]) == 2
    assert "中间结果" not in str(result["answer"])
    assert "最终地点2" not in str(result["answer"])
    assert result["tool_state"]["selected_result_ids"] == ["final"]


async def test_finish_rejects_unknown_selected_result():
    loop = tool_loop()
    result = await loop.run_tool_step(_initial_state(), Gateway({"action": "finish", "result_ids": ["made-up"]}),
                                      Runtime(), strategy="chat")
    assert result["answer"] == {"status": "cannot_answer"}
    assert result["next_node"] == "end"


async def test_chat_entry_exposes_progressive_tools():
    from agentic_rag.query.chat import run_chat
    runtime = Runtime()
    result = await run_chat(_initial_state(), Gateway({"action": "discover_tools", "query": "计算"}),
                            tool_runtime=runtime)
    assert result["next_node"] == "chat"
    assert len(runtime.discovered) == 1


async def test_fast_entry_uses_shared_discovery_for_external_queries():
    from agentic_rag.query.fast_rag import FastRagDependencies, run_fast_rag
    from agentic_rag.query.evidence_builder import EvidenceBuilder
    state = _initial_state()
    state["route"] = {"route": "fast_rag", "normalized_query": "路线", "reason_code": "test"}
    state["route_assessment"] = {"required_sources": ["external_lookup"]}
    runtime = Runtime()
    deps = FastRagDependencies(retrieval=None, evidence_builder=EvidenceBuilder(), evidence_grader=None,
                               tool_runtime=runtime, gateway=Gateway({"action": "discover_tools", "query": "路线"}))
    result = await run_fast_rag(state, deps)
    assert result["next_node"] == "fast_rag"
    assert len(runtime.discovered) == 1


async def test_research_entry_uses_shared_discovery_for_mixed_queries():
    from agentic_rag.query.research_loop import ResearchAgentLoop, ResearchLoopDependencies
    from agentic_rag.query.evidence_builder import EvidenceBuilder
    state = _initial_state()
    state["route_assessment"] = {"required_sources": ["knowledge_base", "external_lookup"]}
    runtime = Runtime()
    loop = ResearchAgentLoop(ResearchLoopDependencies(retrieval=None, evidence_builder=EvidenceBuilder(),
        tool_runtime=runtime, gateway=Gateway({"action": "discover_tools", "query": "文档地址"})))
    result = await loop.ainvoke(state)
    assert result["next_node"] == "research_agent"
    assert len(runtime.discovered) == 1


@pytest.mark.parametrize("strategy", ["chat", "fast_rag", "research"])
async def test_graph_checkpoints_tool_steps_and_publishes_checked_result(strategy):
    from dataclasses import replace
    from agentic_rag.models.schemas import RouteDecision
    from agentic_rag.query.graph import build_query_graph, query_checkpoint_config
    from agentic_rag.query.research_loop import ResearchAgentLoop, ResearchLoopDependencies
    from tests.unit.query.test_graph import _deps, _state

    class FlowGateway(Gateway):
        async def complete_structured(self, call, schema):
            if schema is RouteDecision:
                return SimpleNamespace(value=RouteDecision(route=strategy, normalized_query="6*7", reason_code="test"))
            return await super().complete_structured(call, schema)

    gateway = FlowGateway({"action": "discover_tools", "query": "计算"},
                          {"action": "call_tool", "tool_id": "local.calculator", "arguments": {"expression": "6*7"}},
                          {"action": "finish"})
    runtime = Runtime()
    deps, _, _, _ = _deps(route=strategy)
    research = ResearchAgentLoop(ResearchLoopDependencies(gateway=gateway, retrieval=deps.retrieval,
                                  evidence_builder=deps.evidence_builder, tool_runtime=runtime))
    deps = replace(deps, gateway=gateway, tool_runtime=runtime, research_loop=research)
    state = _state()
    state["tool_state"] = {"active": True}
    result = await build_query_graph(deps).ainvoke(state, query_checkpoint_config(state))
    assert len(runtime.called) == 1
    assert result["answer"]["tool_audited"] is True
    assert "42" in str(result["answer"]["segments"])
    assert result["termination_reason"] == "completed"


@pytest.mark.parametrize("valid_external", [False, True])
async def test_mixed_graph_requires_projectable_external_facts_and_document_audit(valid_external):
    from dataclasses import replace
    from agentic_rag.models.schemas import EvidenceGrade
    from agentic_rag.query.audit import FaithfulnessAuditor
    from agentic_rag.query.graph import build_query_graph, query_checkpoint_config
    from agentic_rag.tool_runtime.models import ToolResult
    from tests.unit.query.test_graph import _deps, _state, _batch, FakeGrader
    loop = tool_loop()

    class MixedGateway:
        async def complete_structured(self, call, schema):
            if schema is RouteAssessment:
                return SimpleNamespace(value=RouteAssessment(required_sources=("knowledge_base", "external_lookup"),
                    required_capabilities=("maps.search",), retrieval_complexity="single",
                    normalized_query="总结合同并查询杭州东站", needs_clarification=False, reason_code="mixed_sources"))
            return SimpleNamespace(value={"passed": True, "unsupported_claim_ids": [], "reasons": []})

    class Research:
        async def ainvoke(self, state):
            doc = ToolResult(call_id="doc", tool_id="local.knowledge_search", source_kind="document", status="success",
                             data={"batch": _batch().model_dump(mode="json")}, observed_at="2026-10-02T00:00:00Z")
            body = {"pois": [{"id": "B023B08WDR", "name": "杭州东站"}]} if valid_external else {"unexpected": "unparseable"}
            external = ToolResult(call_id="map", tool_id="mcp.amap_maps.maps_text_search", source_kind="external",
                                  status="success", data={"structured": body}, observed_at="2026-10-02T00:00:00Z")
            staged = {**state, **loop._document_update(state, doc, {"query": "合同"}), "tool_state": {
                "results": [doc.model_dump(mode="json"), external.model_dump(mode="json")],
                "document_queries": ["合同"], "steps": 4}}
            return {**staged, **loop.finish_tools(staged, strategy="research")}

    deps, _, _, _ = _deps()
    gateway = MixedGateway()
    deps = replace(deps, gateway=gateway, faithfulness_auditor=FaithfulnessAuditor(gateway),
                   capabilities=RuntimeCapabilities(knowledge_base=True, tool_capabilities=("maps.search",)),
                   evidence_grader=FakeGrader([EvidenceGrade(decision="sufficient")]), research_loop=Research())
    state = _state()
    result = await build_query_graph(deps).ainvoke(state, query_checkpoint_config(state))
    if valid_external:
        assert result["termination_reason"] == "completed"
        assert result["answer"]["audited"] is True
        assert result["answer"]["tool_audited"] is True
        from agentic_rag.runtime.query_worker import _public_answer_projection
        from agentic_rag.query.state import snapshot_from_state
        public = _public_answer_projection(result["answer"], result,
            runtime_config_snapshot_id=snapshot_from_state(state).snapshot_id, require_audited=True)
        assert public["evidence_parent_ids"]
        assert result["answer"]["cards"][0]["title"] == "杭州东站"
    else:
        assert result["termination_reason"] == "cannot_answer"
        assert not result["answer"].get("segments")
