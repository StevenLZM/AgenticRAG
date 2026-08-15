"""Public runtime-summary API contract."""

from __future__ import annotations

from types import SimpleNamespace
from typing import cast

import httpx

from agentic_rag.api.app import create_app
from agentic_rag.api.health import ReadinessChecks
from agentic_rag.config import Settings
from agentic_rag.runtime.models import RuntimeConfigSnapshot


SNAPSHOT = RuntimeConfigSnapshot(
    app_version="test-version",
    graph_version="query-v1",
    prompt_version="prompt-v1",
    main_model_id="main-model",
    light_model_id="light-model",
    deepseek_protocol="responses",
    embedding_model="text-embedding-v3",
    embedding_dimensions=1024,
    reranker_version="reranker-v1",
    retrieval_config_version="retrieval-v1",
    index_generation="index-v1",
    memory_config_version="memory-v1",
)


async def _ok() -> None:
    return None


def _container(*, snapshot: object = SNAPSHOT) -> SimpleNamespace:
    container = SimpleNamespace(
        runtime_snapshot=snapshot,
        readiness_checks=ReadinessChecks({"mysql": _ok, "memory": _ok}),
        settings=SimpleNamespace(mem0_enabled=True),
    )

    async def close() -> None:
        return None

    container.close = close
    return container


async def test_runtime_summary_exposes_snapshot_without_secrets() -> None:
    app = create_app(cast(Settings, SimpleNamespace()), container=_container())
    transport = httpx.ASGITransport(app=app, raise_app_exceptions=False)

    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/v1/runtime/summary")

    assert response.status_code == 200
    body = response.json()
    assert body == {
        "runtime_config_snapshot_id": SNAPSHOT.snapshot_id,
        "app_version": "test-version",
        "graph_version": "query-v1",
        "main_model_id": "main-model",
        "light_model_id": "light-model",
        "deepseek_protocol": "responses",
        "embedding_model": "text-embedding-v3",
        "index_generation": "index-v1",
        "memory_enabled": True,
        "memory_available": True,
        "dependencies": {"mysql": "available", "memory": "available"},
    }
    assert "mysql_dsn" not in response.text
    assert "api_key" not in response.text
    assert "prompt_hash" not in response.text


async def test_runtime_summary_fails_closed_when_snapshot_is_invalid() -> None:
    app = create_app(
        cast(Settings, SimpleNamespace()), container=_container(snapshot="not-a-snapshot")
    )
    transport = httpx.ASGITransport(app=app, raise_app_exceptions=False)

    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/v1/runtime/summary")

    assert response.status_code == 503
    assert response.json()["error_code"] == "RUNTIME_SNAPSHOT_UNAVAILABLE"
