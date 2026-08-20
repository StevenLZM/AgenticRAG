"""Opt-in real-provider console and Query API acceptance contract."""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from tests.fixtures.query_services import RealQueryRuntime


pytest_plugins = ("tests.fixtures.query_services",)
pytestmark = [pytest.mark.e2e, pytest.mark.integration]


async def test_console_real_graph_api_query_has_snapshot_provenance_and_gates(
    real_query_runtime: "RealQueryRuntime",
) -> None:
    """A live provider run must cross the public console/API boundary safely."""
    page = await real_query_runtime.client.get("/")
    assert page.status_code == 200
    assert "知识检索控制台" in page.text
    memories = await real_query_runtime.client.get("/v1/memories")
    assert memories.status_code == 200
    stored_marker = real_query_runtime.memory_marker
    assert isinstance(stored_marker, str) and stored_marker
    assert any(
        stored_marker in str(memory.get("text", ""))
        for memory in memories.json().get("memories", [])
        if isinstance(memory, Mapping)
    )

    created = await real_query_runtime.client.post(
        "/v1/query",
        json={"query": real_query_runtime.seeded_question, "wait_seconds": 30},
    )
    assert created.status_code in {200, 202}
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
    assert summary["client_provenance"] == "real_query_api"
    assert summary["runtime_config_snapshot_id"] == real_query_runtime.snapshot.snapshot_id
    assert summary["citation_coverage"] == 1.0
    assert summary["user_leak_count"] == 0
    assert summary["unaudited_answer_count"] == 0
    assert summary["recovery_drill_passed"] is True
    assert summary["backup_restore_passed"] is True
    recovery = summary["recovery_evidence"]
    assert recovery["provider_e2e"] is True
    assert recovery["worker_restarted"] is True
    assert recovery["duplicate_delivery_acked"] is True
    assert recovery["terminal_event_count"] == 1
    backup = summary["backup_restore_evidence"]
    assert backup["provider_e2e"] is True
    assert backup["source_run_id"] == recovery["source_run_id"]
    assert all(backup["services"].values())
    assert backup["source_scope"]["mysql_database"] != backup["restore_scope"]["mysql_database"]
    assert backup["source_scope"]["redis_prefix"] != backup["restore_scope"]["redis_prefix"]
    assert backup["source_scope"]["index_generation"] != backup["restore_scope"]["index_generation"]
    assert summary["memory_provider_available"] is True
    assert real_query_runtime.memory_boundary == {"read": True, "write": True}

    encoded = json.dumps({"run": run, "events": events}, ensure_ascii=False)
    assert all(
        marker not in encoded
        for marker in ("Bearer ", "sk-", "chain_of_thought", "prompt", "tool_input")
    )
