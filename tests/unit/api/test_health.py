"""Behavioral tests for application health and error contracts."""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Any, AsyncIterator

import httpx
import pytest
from fastapi import FastAPI

import agentic_rag.api.app as app_module
from agentic_rag.api.app import create_app
from agentic_rag.api.errors import TemporaryDependencyError
from agentic_rag.api.health import (
    ReadinessChecks,
    check_artifacts,
    check_checkpoints,
    check_configuration,
)
from agentic_rag.config import Settings
from agentic_rag.persistence.artifacts import ArtifactRef
from agentic_rag.persistence.repositories import ActiveRunConflict


DEPENDENCY_NAMES = {
    "configuration",
    "mysql",
    "redis",
    "elasticsearch",
    "artifacts",
    "checkpoints",
    "reranker",
}


def _settings(tmp_path: Path, *, with_credentials: bool = True) -> Settings:
    return Settings(
        mysql_dsn="mysql+asyncmy://rag:rag@127.0.0.1:3306/rag",
        redis_url="redis://127.0.0.1:6379/0",
        elasticsearch_url="http://127.0.0.1:9200",
        deepseek_base_url="https://models.example.invalid/v1",
        deepseek_api_key="deepseek-test-key" if with_credentials else None,
        qwen_embedding_base_url="https://embeddings.example.invalid/v1",
        qwen_api_key="qwen-test-key" if with_credentials else None,
        query_checkpoint_path=tmp_path / "query.sqlite",
        ingestion_checkpoint_path=tmp_path / "ingestion.sqlite",
        artifact_root=tmp_path / "artifacts",
    )


async def _ok() -> None:
    return None


def _container(checks: ReadinessChecks) -> SimpleNamespace:
    container = SimpleNamespace(readiness_checks=checks, closed=False)

    async def close() -> None:
        container.closed = True

    container.close = close
    return container


def _app_with_checks(
    monkeypatch: Any, tmp_path: Path, checks: ReadinessChecks
) -> FastAPI:
    monkeypatch.setattr(app_module, "build_container", lambda _settings: _container(checks))
    return create_app(_settings(tmp_path))


async def _get(app: FastAPI, path: str) -> httpx.Response:
    transport = httpx.ASGITransport(app=app, raise_app_exceptions=False)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        return await client.get(path)


async def test_liveness_only_requires_the_application_event_loop(
    monkeypatch: Any, tmp_path: Path
) -> None:
    async def must_not_run() -> None:
        raise AssertionError("liveness must not probe dependencies")

    checks = ReadinessChecks({"mysql": must_not_run})
    app = _app_with_checks(monkeypatch, tmp_path, checks)

    response = await _get(app, "/health/live")

    assert response.status_code == 200
    assert response.json() == {"status": "live"}


async def test_application_shutdown_closes_the_container(
    monkeypatch: Any, tmp_path: Path
) -> None:
    container = _container(ReadinessChecks({"configuration": _ok}))
    monkeypatch.setattr(app_module, "build_container", lambda _settings: container)
    app = create_app(_settings(tmp_path))

    async with app.router.lifespan_context(app):
        assert container.closed is False

    assert container.closed is True


async def test_application_can_reuse_deployment_owned_container(
    tmp_path: Path,
) -> None:
    container = _container(ReadinessChecks({"configuration": _ok}))
    app = create_app(_settings(tmp_path), container=container)

    async with app.router.lifespan_context(app):
        assert container.closed is False

    assert container.closed is False


async def test_readiness_reports_each_failed_dependency_without_provider_details(
    monkeypatch: Any, tmp_path: Path
) -> None:
    async def unavailable_elasticsearch() -> None:
        raise RuntimeError("provider host and secret must never reach the response")

    checks = ReadinessChecks(
        {
            "configuration": _ok,
            "mysql": _ok,
            "redis": _ok,
            "elasticsearch": unavailable_elasticsearch,
            "artifacts": _ok,
            "checkpoints": _ok,
            "reranker": _ok,
        }
    )
    app = _app_with_checks(monkeypatch, tmp_path, checks)

    response = await _get(app, "/health/ready")

    assert response.status_code == 503
    assert response.json() == {
        "status": "unready",
        "dependencies": {
            "configuration": "available",
            "mysql": "available",
            "redis": "available",
            "elasticsearch": "unavailable",
            "artifacts": "available",
            "checkpoints": "available",
            "reranker": "available",
        },
    }
    assert "provider" not in response.text
    assert "secret" not in response.text


async def test_readiness_is_ready_only_when_every_dependency_is_available(
    monkeypatch: Any, tmp_path: Path
) -> None:
    checks = ReadinessChecks({name: _ok for name in sorted(DEPENDENCY_NAMES)})
    app = _app_with_checks(monkeypatch, tmp_path, checks)

    response = await _get(app, "/health/ready")

    assert response.status_code == 200
    assert response.json()["status"] == "ready"
    assert response.json()["dependencies"] == {
        name: "available" for name in sorted(DEPENDENCY_NAMES)
    }


async def test_readiness_rejects_empty_dependency_wiring(
    monkeypatch: Any, tmp_path: Path
) -> None:
    app = _app_with_checks(monkeypatch, tmp_path, ReadinessChecks({}))

    response = await _get(app, "/health/ready")

    assert response.status_code == 503
    assert response.json() == {"status": "unready", "dependencies": {}}


async def test_readiness_require_ready_fails_closed_with_dependency_names() -> None:
    checks = ReadinessChecks({"mysql": _ok, "elasticsearch": _fails})

    with pytest.raises(RuntimeError, match="elasticsearch"):
        await checks.require_ready()


async def test_readiness_require_ready_rejects_empty_checks() -> None:
    with pytest.raises(RuntimeError, match="empty"):
        await ReadinessChecks({}).require_ready()


async def _fails() -> None:
    raise OSError("unavailable")


async def test_configuration_check_requires_credentials_without_calling_models(
    tmp_path: Path,
) -> None:
    missing_credentials = _settings(tmp_path, with_credentials=False)

    error: RuntimeError | None = None
    try:
        await check_configuration(missing_credentials)
    except RuntimeError as caught:
        error = caught

    assert error is not None
    assert str(error) == "required model credentials are not configured"


class _FakeArtifactStore:
    def __init__(self) -> None:
        self.objects: dict[str, dict[str, Any]] = {}

    def put_json(self, relative_path: str, value: dict[str, Any]) -> ArtifactRef:
        self.objects[relative_path] = value
        return ArtifactRef(uri=f"artifact://{relative_path}", sha256="0" * 64, size_bytes=1)

    def read_json(self, ref: ArtifactRef) -> Any:
        return self.objects[ref.uri.removeprefix("artifact://")]

    def delete(self, ref: ArtifactRef) -> None:
        del self.objects[ref.uri.removeprefix("artifact://")]


async def test_artifact_check_round_trips_and_removes_health_probe() -> None:
    store = _FakeArtifactStore()

    await check_artifacts(store)

    assert store.objects == {}


class _FakeCheckpointBackend:
    def __init__(self) -> None:
        self.opened: list[str] = []

    @asynccontextmanager
    async def open_query(self) -> AsyncIterator[object]:
        self.opened.append("query")
        yield object()

    @asynccontextmanager
    async def open_ingestion(self) -> AsyncIterator[object]:
        self.opened.append("ingestion")
        yield object()


async def test_checkpoint_check_initializes_both_async_checkpoint_contexts() -> None:
    checkpoints = _FakeCheckpointBackend()

    await check_checkpoints(checkpoints)

    assert checkpoints.opened == ["query", "ingestion"]


async def test_temporary_dependency_errors_use_sanitized_retryable_503(
    monkeypatch: Any, tmp_path: Path
) -> None:
    app = _app_with_checks(
        monkeypatch,
        tmp_path,
        ReadinessChecks({name: _ok for name in DEPENDENCY_NAMES}),
    )

    async def fail() -> None:
        raise TemporaryDependencyError(
            "model_provider", provider_detail="upstream token=do-not-return"
        )

    app.add_api_route("/temporary-error", fail)

    response = await _get(app, "/temporary-error")

    assert response.status_code == 503
    payload = response.json()
    assert payload["error_code"] == "DEPENDENCY_UNAVAILABLE"
    assert payload["message"] == "A required dependency is temporarily unavailable."
    assert payload["retryable"] is True
    assert payload["degraded_components"] == ["model_provider"]
    assert payload["trace_id"]
    assert "do-not-return" not in response.text


async def test_active_run_conflict_maps_to_non_retryable_409(
    monkeypatch: Any, tmp_path: Path
) -> None:
    app = _app_with_checks(
        monkeypatch,
        tmp_path,
        ReadinessChecks({name: _ok for name in DEPENDENCY_NAMES}),
    )

    async def conflict() -> None:
        raise ActiveRunConflict("mysql constraint details")

    app.add_api_route("/conflict", conflict)

    response = await _get(app, "/conflict")

    assert response.status_code == 409
    assert response.json()["error_code"] == "ACTIVE_RUN_EXISTS"
    assert response.json()["retryable"] is False
    assert "mysql constraint" not in response.text


async def test_request_validation_uses_unified_non_retryable_error(
    monkeypatch: Any, tmp_path: Path
) -> None:
    app = _app_with_checks(
        monkeypatch,
        tmp_path,
        ReadinessChecks({name: _ok for name in DEPENDENCY_NAMES}),
    )

    async def validated_route(count: int) -> dict[str, int]:
        return {"count": count}

    app.add_api_route("/validated", validated_route)

    response = await _get(app, "/validated?count=not-an-integer")

    assert response.status_code == 422
    payload = response.json()
    assert payload["error_code"] == "VALIDATION_ERROR"
    assert payload["message"] == "Request validation failed."
    assert payload["retryable"] is False
    assert "not-an-integer" not in response.text
