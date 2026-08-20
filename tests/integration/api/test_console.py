"""Same-origin browser console delivery contract."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
from types import SimpleNamespace
from typing import cast

import httpx
import pytest

from agentic_rag.api.app import create_app
from agentic_rag.api.health import ReadinessChecks
from agentic_rag.config import Settings


CONSOLE_PATH = (
    Path(__file__).resolve().parents[3]
    / "src/agentic_rag/api/static/app.js"
)
STABLE_DOM_IDS = {
    "query-form",
    "query-input",
    "run-status",
    "timeline",
    "answer",
    "evidence",
    "audit",
    "provenance",
    "degradation-banner",
    "document-upload",
    "ingestion-status",
    "memory-list",
    "health-grid",
    "snapshot-id",
}


async def _ok() -> None:
    return None


def _app_for_static_test():
    container = SimpleNamespace(
        readiness_checks=ReadinessChecks({"memory": _ok}),
        settings=SimpleNamespace(mem0_enabled=True),
    )

    async def close() -> None:
        return None

    container.close = close
    return create_app(cast(Settings, SimpleNamespace()), container=container)


def _console_contract() -> dict[str, object]:
    script = """
const fs = require("fs");
const vm = require("vm");
const source = fs.readFileSync(process.argv[1], "utf8");
const context = {window: {}, document: {addEventListener() {}}};
vm.runInNewContext(source, context, {filename: process.argv[1]});
const api = (context.window.AgenticRagConsole || {}).contract || {};
const call = (name, ...args) => api[name] ? api[name](...args) : null;
process.stdout.write(JSON.stringify({
  query: call("buildQueryPayload", "  需要检索的问题  "),
  headers: [call("buildSseHeaders", 0), call("buildSseHeaders", 17)],
  unknown: call("eventPresentation", {event_type: "INTERNAL_TOOL_PAYLOAD", summary: "private"}),
  degradation: call("eventPresentation", {
    event_type: "RETRIEVAL_DEGRADED",
    summary: "degraded",
    attributes: {
      component: "dense",
      reason: "lane_timeout",
      outcome: "degraded",
      retryable: true,
      attempt: 1,
      prompt: "never expose this prompt",
      provider_response: "Bearer never expose this provider output"
    }
  }),
  notices: ["RETRIEVAL_DEGRADED", "CIRCUIT_OPEN", "MODEL_REPAIR_EXHAUSTED", "WORKER_DLQ", "AUDIT_REFUSED"].map((type) => call("noticeCodeForEvent", type)),
  terminal: ["research_action_invalid", "research_round_limit", "audit_failed", "cannot_answer", "refuse", "clarify"].map((status) => call("terminalNoticeCode", {answer: {status}})),
  memory: call("memoryErrorPresentation", "provider unavailable"),
  provenance: call("provenanceFor", {evidence_parent_ids: ["parent-1"], route: "research", client_provenance: "api"}, {runtime_config_snapshot_id: "snapshot-1"}),
  answer: call("answerPresentation", {
    runtime_config_snapshot_id: "snapshot-1",
    answer: {
      audited: true,
      segments: [{
        kind: "content",
        text: "safe answer",
        evidence_ids: ["e1"],
        tool_input: "Authorization: Bearer nested-secret"
      }],
      evidence_parent_ids: ["parent-1"],
      route: "research",
      client_provenance: "real_query_api",
      citation_coverage: 1,
      prompt: "Authorization: Bearer prompt-secret",
      provider_response: "provider-secret",
      raw: "raw-secret",
      unknown: "unknown-secret",
      evidence: "evidence-secret",
      audit: "audit-secret"
    }
  })
}));
"""
    completed = subprocess.run(
        ["node", "-e", script, str(CONSOLE_PATH)],
        check=True,
        capture_output=True,
        text=True,
    )
    return cast(dict[str, object], json.loads(completed.stdout))


@pytest.mark.integration
async def test_console_serves_same_origin_html_and_static_assets() -> None:
    app = _app_for_static_test()
    transport = httpx.ASGITransport(app=app, raise_app_exceptions=False)

    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        page = await client.get("/")
        script = await client.get("/static/app.js")
        style = await client.get("/static/app.css")

    assert page.status_code == script.status_code == style.status_code == 200
    assert all(f'id="{element_id}"' in page.text for element_id in STABLE_DOM_IDS)
    assert "localStorage" not in script.text


@pytest.mark.integration
async def test_console_static_mount_rejects_non_asset_files() -> None:
    app = _app_for_static_test()
    transport = httpx.ASGITransport(app=app, raise_app_exceptions=False)

    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        index = await client.get("/static/index.html")
        unknown = await client.get("/static/not-an-asset.txt")

    assert index.status_code == unknown.status_code == 404


def test_console_client_contract_preserves_scope_and_safe_terminal_states() -> None:
    contract = _console_contract()

    assert contract["query"] == {"query": "需要检索的问题", "wait_seconds": 30}
    assert contract["headers"] == [
        {"Accept": "text/event-stream"},
        {"Accept": "text/event-stream", "Last-Event-ID": "17"},
    ]
    assert contract["unknown"] == {
        "label": "进度更新",
        "summary": "进度更新",
        "noticeCode": None,
    }
    assert contract["degradation"] == {
        "label": "RETRIEVAL_DEGRADED",
        "summary": "degraded",
        "noticeCode": "RETRIEVAL_DEGRADED",
        "attributes": {
            "attempt": 1,
            "component": "dense",
            "outcome": "degraded",
            "reason": "lane_timeout",
            "retryable": True,
        },
    }
    assert contract["notices"] == [
        "RETRIEVAL_DEGRADED",
        "CIRCUIT_OPEN",
        "MODEL_REPAIR_EXHAUSTED",
        "WORKER_DLQ",
        "audit_failed",
    ]
    assert contract["terminal"] == [
        "research_action_invalid",
        "research_round_limit",
        "audit_failed",
        "cannot_answer",
        "refuse",
        "clarify",
    ]
    assert contract["memory"] == {
        "className": "error-card",
        "message": "Mem0 不可用：provider unavailable",
    }
    assert contract["provenance"] == {
        "evidence_parent_ids": ["parent-1"],
        "route": "research",
        "runtime_config_snapshot_id": "snapshot-1",
        "client_provenance": "api",
    }
    assert contract["answer"] == {
        "text": "safe answer",
        "evidence": {
            "evidence_ids": ["e1"],
            "evidence_parent_ids": ["parent-1"],
        },
        "audit": {"audited": True, "citation_coverage": 1},
        "provenance": {
            "evidence_parent_ids": ["parent-1"],
            "route": "research",
            "runtime_config_snapshot_id": "snapshot-1",
            "client_provenance": "real_query_api",
        },
    }
    serialized = json.dumps(contract["answer"])
    assert all(secret not in serialized for secret in ("nested-secret", "prompt-secret", "provider-secret", "raw-secret", "unknown-secret", "evidence-secret", "audit-secret"))


@pytest.mark.integration
async def test_console_mem0_provider_failure_is_not_an_empty_list() -> None:
    class UnavailableMemory:
        async def list(self, _scope: object) -> list[object]:
            raise OSError("provider unavailable")

    app = _app_for_static_test()
    app.state.container.settings.default_user_id = "console-user"
    app.state.container.memory_service = UnavailableMemory()
    transport = httpx.ASGITransport(app=app, raise_app_exceptions=False)

    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/v1/memories")

    assert response.status_code == 503
    assert response.json()["error_code"] == "MEMORY_UNAVAILABLE"
