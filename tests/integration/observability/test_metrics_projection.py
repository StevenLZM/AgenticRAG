"""Integration contracts for event-log based online metrics projections."""

from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime, timedelta

import pytest

from agentic_rag.domain.models import UserScope
from agentic_rag.observability.logging import AgentEventEmitter
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


@pytest.mark.integration
async def test_metrics_projection_totals_fixture_events_from_event_log(tmp_path) -> None:
    """Dropping a safe artifact field must change the projection totals."""
    repository = RecordingEventRepository()
    artifacts = LocalArtifactStore(tmp_path / "artifacts")
    emitter = AgentEventEmitter(
        repository,
        artifacts,
        runtime_config_snapshot_id="snapshot-1",
    )
    created_at = datetime(2026, 8, 4, tzinfo=UTC)

    await emitter.emit(
        run_id="run-1",
        user_id="user-1",
        event_type="QUEUE_WAITED",
        attributes={"queue_wait_seconds": 1.5},
        created_at=created_at,
    )
    await emitter.emit(
        run_id="run-1",
        user_id="user-1",
        event_type="RETRIEVAL_COMPLETED",
        attributes={"retrieval_rounds": 2, "candidate_count": 6},
        created_at=created_at + timedelta(seconds=1),
    )
    await emitter.emit(
        run_id="run-1",
        user_id="user-1",
        event_type="LLM_COMPLETED",
        attributes={"input_tokens": 10, "output_tokens": 4, "estimated_cost": 0.01},
        created_at=created_at + timedelta(seconds=2),
    )
    await emitter.emit(
        run_id="run-1",
        user_id="user-1",
        event_type="CITATION_VALIDATED",
        attributes={"cited_claim_count": 3, "claim_count": 3},
        created_at=created_at + timedelta(seconds=3),
    )
    await emitter.emit(
        run_id="run-1",
        user_id="user-1",
        event_type="USER_FEEDBACK",
        attributes={"rating": "up"},
        created_at=created_at + timedelta(seconds=4),
    )

    projection = await MetricsProjector(
        repository,
        artifacts,
        runtime_config_snapshot_id="snapshot-1",
    ).project_window(run_id="run-1", scope=UserScope(user_id="user-1"))

    assert projection.events_processed == 5
    assert projection.metrics["queue_wait_seconds"] == 1.5
    assert projection.metrics["retrieval_rounds"] == 2
    assert projection.metrics["retrieval_candidates"] == 6
    assert projection.metrics["input_tokens"] == 10
    assert projection.metrics["output_tokens"] == 4
    assert projection.metrics["estimated_cost"] == pytest.approx(0.01)
    assert projection.metrics["citation_coverage"] == 1.0
    assert projection.metrics["feedback_count"] == 1


@pytest.mark.integration
async def test_metrics_projection_ignores_other_snapshot_events(tmp_path) -> None:
    """Mixed snapshots cannot silently contaminate a baseline's online metrics."""
    repository = RecordingEventRepository()
    artifacts = LocalArtifactStore(tmp_path / "artifacts")
    emitter = AgentEventEmitter(repository, artifacts, runtime_config_snapshot_id="other")
    await emitter.emit(
        run_id="run-1",
        user_id="user-1",
        event_type="LLM_COMPLETED",
        attributes={"input_tokens": 99},
    )

    projection = await MetricsProjector(
        repository, artifacts, runtime_config_snapshot_id="snapshot-1"
    ).project_window(run_id="run-1", scope=UserScope(user_id="user-1"))

    assert projection.events_processed == 0
    assert projection.metrics["input_tokens"] == 0
