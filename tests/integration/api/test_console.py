"""Same-origin browser console delivery contract."""

from __future__ import annotations

from types import SimpleNamespace
from typing import cast

import httpx
import pytest

from agentic_rag.api.app import create_app
from agentic_rag.api.health import ReadinessChecks
from agentic_rag.config import Settings


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


@pytest.mark.integration
async def test_console_serves_same_origin_html_and_static_assets() -> None:
    app = _app_for_static_test()
    transport = httpx.ASGITransport(app=app, raise_app_exceptions=False)

    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        page = await client.get("/")
        script = await client.get("/static/app.js")
        style = await client.get("/static/app.css")

    assert page.status_code == script.status_code == style.status_code == 200
    assert 'id="query-form"' in page.text
    assert 'id="snapshot-id"' in page.text
    assert "/v1/query" in script.text
    assert "Last-Event-ID" in script.text
    assert "localStorage" not in script.text
