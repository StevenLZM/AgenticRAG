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


CONSOLE_PATH = Path(__file__).resolve().parents[3] / "src/agentic_rag/api/static/app.js"
STABLE_DOM_IDS = {
    "session-list",
    "new-chat",
    "chat-messages",
    "query-form",
    "query-input",
    "send-button",
    "cancel-button",
    "scroll-latest",
    "tools-drawer",
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
const view = require(process.argv[1]);
const run = answer => ({status: 'completed', answer});
const chat = run({route:'chat', segments:[{kind:'content',text:'了解'}]});
const doc = run({route:'research',audited:true,segments:[{kind:'content',text:'safe answer',evidence_ids:['e1'],raw:'nested-secret'}],prompt:'prompt-secret',provider_response:'provider-secret'});
process.stdout.write(JSON.stringify({chat:view.answerPresentation(chat),answer:view.answerPresentation(doc),
 invalidChat:view.answerPresentation(run({...chat.answer,audited:true})),
 pending:view.answerPresentation({...doc,status:'running'}),
 failed:view.answerPresentation({...doc,status:'failed'}),
 unaudited:view.answerPresentation(run({...doc.answer,audited:false}))}));
"""
    result = subprocess.run(
        ["node", "-e", script, str(CONSOLE_PATH.with_name("chat-view.js"))],
        capture_output=True,
        text=True,
        check=True,
    )
    return cast(dict[str, object], json.loads(result.stdout))


@pytest.mark.integration
async def test_console_serves_same_origin_html_and_static_assets() -> None:
    app = _app_for_static_test()
    transport = httpx.ASGITransport(app=app, raise_app_exceptions=False)

    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        page = await client.get("/")
        script = await client.get("/static/app.js")
        style = await client.get("/static/app.css")

    assert page.status_code == script.status_code == style.status_code == 200
    assert script.headers.get("cache-control") == "no-cache"
    assert all(f'id="{element_id}"' in page.text for element_id in STABLE_DOM_IDS)
    assert all(
        f'id="{name}"' not in page.text
        for name in ("timeline", "answer", "evidence", "audit")
    )
    assets = [
        "chat-state.js",
        "chat-api.js",
        "chat-view.js",
        "console-tools.js",
        "app.js",
    ]
    assert [page.text.index(f"/static/{name}") for name in assets] == sorted(
        page.text.index(f"/static/{name}") for name in assets
    )
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        for asset in assets:
            assert (await client.get(f"/static/{asset}")).status_code == 200


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
    assert contract["chat"] == {"text": "了解", "hasSources": False}
    assert contract["answer"] == {"text": "safe answer", "hasSources": True}
    assert all(
        contract[name] is None
        for name in ("invalidChat", "pending", "failed", "unaudited")
    )
    assert "secret" not in json.dumps(contract)


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


def test_console_route_timeline_and_chat_dom_flow() -> None:
    subprocess.run(
        ["node", str(Path(__file__).with_name("console_flow.cjs")), str(CONSOLE_PATH)],
        check=True,
        capture_output=True,
        text=True,
    )


def _terminal_dom(
    answer: dict, mode: str, *, events: list | None = None, **run_fields
) -> dict:
    completed = subprocess.run(
        [
            "node",
            str(Path(__file__).with_name("console_terminal_flow.cjs")),
            str(CONSOLE_PATH),
        ],
        input=json.dumps(
            {
                "run": {
                    "run_id": "run-terminal",
                    "status": "completed",
                    "error_code": None,
                    "answer": answer,
                    **run_fields,
                },
                "mode": mode,
                "events": events or [],
            }
        ),
        check=True,
        capture_output=True,
        text=True,
        timeout=5,
    )
    return json.loads(completed.stdout)


@pytest.mark.parametrize("mode", ["load", "sync", "stream"])
@pytest.mark.parametrize(
    ("status", "text"),
    [
        ("cannot_answer", "目前尚未接入实时数据或外部查询服务，无法核实你请求的信息。"),
        (
            "clarify",
            "我可以分析已上传的资料，但目前无法查询外部实时信息。是否先分析文档部分？",
        ),
        ("cannot_answer", "本次处理所需的服务暂时不可用，未能完成回答，请稍后重试。"),
    ],
)
def test_console_preserves_safe_chat_terminal_explanation(mode, status, text) -> None:
    answer = {
        "route": "chat",
        "status": status,
        "audited": None,
        "segments": [{"kind": "content", "text": text, "evidence_ids": []}],
        "evidence_parent_ids": [],
        "citation_coverage": None,
    }

    for view in _terminal_dom(answer, mode).values():
        assert view["text"] == text
        assert view["notice"] is None


@pytest.mark.parametrize("mode", ["load", "sync", "stream"])
@pytest.mark.parametrize(
    "overrides",
    [
        {"segments": []},
        {"audited": True},
        {"evidence_parent_ids": ["parent-1"]},
        {"citation_coverage": 1},
        {
            "segments": [
                {"kind": "content", "text": "unsafe draft", "evidence_ids": ["e1"]}
            ]
        },
        {"route": "fast_rag"},
        {"route": "research", "audited": True},
    ],
)
def test_console_terminal_fallback_still_blocks_unsafe_or_rag_drafts(
    mode, overrides
) -> None:
    answer = {
        "route": "chat",
        "status": "cannot_answer",
        "segments": [{"kind": "content", "text": "unsafe draft", "evidence_ids": []}],
        **overrides,
    }

    for view in _terminal_dom(answer, mode).values():
        assert view["text"] == "现有证据不足以安全回答，系统未展示草稿。"
        assert view["notice"] == view["text"]


@pytest.mark.parametrize(
    "status",
    ["audit_failed", "refuse", "research_round_limit", "research_action_invalid"],
)
def test_console_chat_route_does_not_bypass_other_terminal_guards(status) -> None:
    answer = {
        "route": "chat",
        "status": status,
        "segments": [{"kind": "content", "text": "unsafe draft", "evidence_ids": []}],
    }

    for view in _terminal_dom(answer, "stream").values():
        assert "unsafe draft" not in view["text"]
        assert view["notice"] == view["text"]


def test_console_chat_explanation_is_not_replaced_by_private_event_logs() -> None:
    answer = {
        "route": "chat",
        "status": "cannot_answer",
        "segments": [
            {"kind": "content", "text": "尚未接入实时服务。", "evidence_ids": []}
        ],
    }
    events = [{"event_type": "COMPONENT_DEGRADED", "summary": "degraded"}]

    for view in _terminal_dom(answer, "stream", events=events).values():
        assert view["text"] == "尚未接入实时服务。"
        assert view["notice"] is None


@pytest.mark.parametrize(
    "run_fields",
    [{"status": "failed"}, {"status": "cancelled"}, {"error_code": "audit_failed"}],
)
def test_console_chat_explanation_does_not_hide_run_failure(run_fields) -> None:
    answer = {
        "route": "chat",
        "status": "cannot_answer",
        "segments": [{"kind": "content", "text": "unsafe draft", "evidence_ids": []}],
    }

    for view in _terminal_dom(answer, "load", **run_fields).values():
        assert "unsafe draft" not in view["text"]
        assert view["notice"] == view["text"]
