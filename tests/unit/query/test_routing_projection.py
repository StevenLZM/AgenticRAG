from agentic_rag.runtime.query_worker import _public_answer_projection
from agentic_rag.observability.logging import sanitize_attributes
from agentic_rag.query.routing_policy import routing_summary
from agentic_rag.query.graph import _event_attributes
from agentic_rag.api.query_runs import _sse_event
from types import SimpleNamespace
from datetime import UTC, datetime


def test_chat_terminal_does_not_inherit_retrieval_citations():
    answer = {"route": "chat", "status": "cannot_answer",
              "segments": [{"kind": "content", "text": "尚未接入实时查询。", "evidence_ids": []}]}
    result = {"route": {"route": "chat"}, "evidence": [{"parent_id": "p"}]}
    public = _public_answer_projection(answer, result, runtime_config_snapshot_id="s", require_audited=True)
    assert public is not None
    assert not public.get("evidence_parent_ids")
    assert not public.get("audited")
    assert result["evidence"] == [{"parent_id": "p"}]


def test_routing_telemetry_is_enum_only():
    attrs = sanitize_attributes({"initial_route": "fast_rag", "route": "chat",
        "executed_path": ["route", "fast_rag", "chat"], "gap_type": "external_realtime_required",
        "response_mode": "capability_unavailable", "prompt": "secret"})
    assert attrs["initial_route"] == "fast_rag"
    assert attrs["executed_path"] == ["route", "fast_rag", "chat"]
    assert "prompt" not in attrs
    assert not sanitize_attributes({"executed_path": ["user secret"], "initial_route": "secret"})


def test_initial_route_differs_from_executed_path():
    state = {"initial_route": "fast_rag", "route": {"route": "research"},
             "executed_path": ["route", "fast_rag", "research_agent_loop"],
             "last_evidence_grade": {"gap_type": "missing_facts"}, "prompt": "secret"}
    summary = routing_summary(state)
    assert summary["route"] == "research"
    assert summary["initial_route"] == "fast_rag"
    assert summary["gap_type"] == "missing_facts"
    assert "prompt" not in summary
    assert _event_attributes(state, "ANSWER_FINALIZED")["route"] == "research"


def test_sse_terminal_route_is_safe_and_overrides_initial_route():
    artifacts = SimpleNamespace(describe=lambda uri: uri, read_json=lambda ref: {"attributes": {
        "initial_route": "fast_rag", "route": "chat", "executed_path": ["route", "fast_rag", "chat"],
        "prompt": "private prompt", "memory": "private memory", "tools": ["web_search"],
    }})
    event = SimpleNamespace(event_type="ANSWER_FINALIZED", run_id="r", summary="completed",
                            created_at=datetime.now(UTC), payload_ref="event.json", id=1)
    payload = _sse_event(event, artifacts=artifacts)
    assert '"route":"chat"' in payload
    assert '"initial_route":"fast_rag"' in payload
    assert "private" not in payload and "web_search" not in payload
