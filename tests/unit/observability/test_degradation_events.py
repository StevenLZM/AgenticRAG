"""Explicit, bounded telemetry for operational fallback paths."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from agentic_rag.domain.models import UserScope
from agentic_rag.observability.logging import (
    AgentEventEmitter,
    emit_degradation,
    event_emission_scope,
)
from agentic_rag.observability.metrics import MetricsProjector
from agentic_rag.persistence.artifacts import LocalArtifactStore
from agentic_rag.persistence.repositories import AgentEvent


class RecordingEventRepository:
    def __init__(self) -> None:
        self.events: list[AgentEvent] = []

    async def append(self, event: AgentEvent) -> int:
        persisted = replace(event, id=len(self.events) + 1)
        self.events.append(persisted)
        return persisted.id or 0

    async def list_after(
        self, run_id: str, scope: UserScope, after_id: int, limit: int
    ) -> list[AgentEvent]:
        return [
            event
            for event in self.events
            if event.run_id == run_id
            and event.user_id == scope.user_id
            and (event.id or 0) > after_id
        ][:limit]


@pytest.mark.asyncio
async def test_emit_degradation_persists_safe_user_visible_signal(tmp_path: Path) -> None:
    repository = RecordingEventRepository()
    artifacts = LocalArtifactStore(tmp_path / "artifacts")
    emitter = AgentEventEmitter(repository, artifacts, runtime_config_snapshot_id="snapshot-1")

    async with event_emission_scope(emitter, "run-1", "retrieval", user_id="user-1"):
        await emit_degradation(
            component="dense",
            reason="lane_timeout",
            run_id="run-1",
            snapshot_id="snapshot-1",
            attempt=1,
            retryable=True,
            outcome="degraded",
        )

    event = repository.events[-1]
    assert event.event_type == "RETRIEVAL_DEGRADED"
    assert event.summary == "degraded"
    assert event.payload_ref is not None
    payload = artifacts.read_json(artifacts.describe(event.payload_ref))
    encoded = json.dumps(payload, sort_keys=True)
    assert payload["attributes"] == {
        "attempt": 1,
        "component": "dense",
        "outcome": "degraded",
        "reason": "lane_timeout",
        "retryable": True,
    }
    assert "prompt" not in encoded.lower()
    assert "secret" not in encoded.lower()


@pytest.mark.asyncio
async def test_emit_degradation_falls_back_to_safe_unknown_reason(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level("WARNING")

    await emit_degradation(
        component="dense",
        reason="raw private prompt and provider response",
        run_id=None,
        snapshot_id="snapshot-1",
        attempt=1,
        retryable=False,
        outcome="refused",
    )

    assert "raw private prompt" not in caplog.text
    assert "reason=unknown" in caplog.text


@pytest.mark.asyncio
async def test_metrics_deduplicate_replayed_degradation_event(tmp_path: Path) -> None:
    repository = RecordingEventRepository()
    artifacts = LocalArtifactStore(tmp_path / "artifacts")
    emitter = AgentEventEmitter(repository, artifacts, runtime_config_snapshot_id="snapshot-1")
    for _ in range(2):
        await emitter.emit(
            run_id="run-1",
            user_id="user-1",
            event_type="RETRIEVAL_DEGRADED",
            event_key="degradation-replay",
            summary="degraded",
            attributes={
                "attempt": 1,
                "component": "dense",
                "reason": "lane_timeout",
                "retryable": True,
                "outcome": "degraded",
            },
        )

    projection = await MetricsProjector(
        repository, artifacts, runtime_config_snapshot_id="snapshot-1"
    ).project_window(run_id="run-1", scope=UserScope(user_id="user-1"))

    assert projection.metrics["degraded_component_count"] == 1

