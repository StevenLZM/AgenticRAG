"""Opt-in real-provider console and Query API protocol-smoke contract."""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import TYPE_CHECKING
from uuid import uuid4

import pytest

if TYPE_CHECKING:
    from tests.fixtures.query_services import RealQueryRuntime


pytest_plugins = ("tests.fixtures.query_services",)
pytestmark = [pytest.mark.e2e, pytest.mark.integration]


async def test_console_real_graph_api_query_protocol_smoke_has_snapshot_provenance(
    real_query_runtime: "RealQueryRuntime",
) -> None:
    """A live provider run must cross the public console/API boundary safely."""
    page = await real_query_runtime.client.get("/")
    assert page.status_code == 200
    assert 'id="chat-messages"' in page.text
    assert 'id="query-input"' in page.text
    assert 'id="session-list"' in page.text
    memories = await real_query_runtime.client.get("/v1/memories")
    assert memories.status_code == 200
    stored_marker = real_query_runtime.memory_marker
    assert isinstance(stored_marker, str) and stored_marker
    assert any(
        stored_marker in str(memory.get("text", ""))
        for memory in memories.json().get("memories", [])
        if isinstance(memory, Mapping)
    )

    session = await real_query_runtime.client.post(
        "/v1/chat-sessions", json={"creation_request_id": str(uuid4())}
    )
    assert session.status_code == 201
    session_id = session.json()["session_id"]
    created = await real_query_runtime.client.post(
        f"/v1/chat-sessions/{session_id}/turns",
        json={"query": real_query_runtime.seeded_question, "client_request_id": str(uuid4())},
    )
    assert created.status_code == 202
    run_id = created.json()["run_id"]
    assert isinstance(run_id, str) and run_id

    events = await real_query_runtime.read_sse(run_id)
    run = await real_query_runtime.wait_for_terminal(run_id)
    assert run["status"] == "completed"
    assert run["runtime_config_snapshot_id"] == real_query_runtime.snapshot.snapshot_id
    answer = run["answer"]
    assert isinstance(answer, Mapping)
    assert answer["audited"] is True
    assert answer["segments"]
    assert answer["evidence_parent_ids"]
    assert real_query_runtime.seeded_parent_id in answer["evidence_parent_ids"]

    history = await real_query_runtime.client.get(f"/v1/chat-sessions/{session_id}/turns")
    assert history.status_code == 200
    assert [item["run_id"] for item in history.json()["items"]] == [run_id]
    assert history.json()["items"][0]["answer"] == answer
    sources = await real_query_runtime.client.get(
        f"/v1/chat-sessions/{session_id}/turns/{run_id}/sources"
    )
    assert sources.status_code == 200
    assert sources.json()["status"] == "available"
    assert sources.json()["items"]

    degradation = next(
        event for event in events if event["event_type"] == "CIRCUIT_OPEN"
    )
    assert degradation["attributes"] == {
        "attempt": 1,
        "component": "retrieval",
        "outcome": "degraded",
        "reason": "circuit_open",
        "retryable": True,
    }

    summary = real_query_runtime.evaluation_summary
    assert summary["fixture_kind"] == "contract_only"
    assert summary["evaluation_mode"] == "contract"
    assert summary["client_provenance"] == "contract_fixture"
    assert summary["quality_measurement"] is False
    assert summary["runtime_config_snapshot_id"] == real_query_runtime.snapshot.snapshot_id
    assert summary["real_query_count"] == 0
    assert summary["completed_cases"] == 0
    assert summary["requested_cases"] == 0
    assert summary["memory_provider_available"] is True
    assert real_query_runtime.memory_boundary == {"read": True, "write": True}

    encoded = json.dumps({"run": run, "events": events}, ensure_ascii=False)
    assert all(
        marker not in encoded
        for marker in ("Bearer ", "sk-", "chain_of_thought", "prompt", "tool_input")
    )
