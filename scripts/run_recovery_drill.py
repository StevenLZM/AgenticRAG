"""Run deterministic, isolated API/worker/index recovery drills.

The default command never reads application settings and never opens a network
connection. Every scenario uses the same durable identity shapes as production
(``AgentEvent``, scoped ``UserScope`` and content-addressed Artifacts), backed
by small in-memory fakes that deliberately replay writes.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import tempfile
from collections import Counter
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

# Running this module directly or as ``scripts.run_recovery_drill`` does not
# automatically add the source layout to ``sys.path``. Keep it self-contained
# without requiring an editable install or a caller-owned PYTHONPATH.
source_root = Path(__file__).resolve().parents[1] / "src"
if str(source_root) not in sys.path:
    sys.path.insert(0, str(source_root))

from agentic_rag.domain.models import UserScope  # noqa: E402
from agentic_rag.memory.models import (  # noqa: E402
    MemoryClient,
    MemoryTombstoneStore,
    Tombstone,
)
from agentic_rag.memory.service import MemoryServiceImpl  # noqa: E402
from agentic_rag.persistence.artifacts import ArtifactRef, LocalArtifactStore  # noqa: E402
from agentic_rag.persistence.repositories import AgentEvent  # noqa: E402


DRILL_SCENARIOS: tuple[str, ...] = (
    "api_restart_during_sse",
    "query_worker_termination_after_retrieval",
    "ingestion_worker_termination_after_staging",
    "outbox_mysql_success_redis_failure",
    "es_activation_interruption",
    "missing_artifact_quarantine",
    "mem0_unavailable",
)
ScenarioStatus = Literal["passed", "failed"]


class RecoveryDrillReport(BaseModel):
    """Stable JSON report and final acceptance gate for local recovery drills."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    scenarios: dict[str, ScenarioStatus]
    scenario_invariants: dict[str, bool] = Field(
        default_factory=lambda: {scenario: True for scenario in DRILL_SCENARIOS}
    )
    duplicate_parent_ids: int = Field(
        ge=0, description="duplicate IDs present in the durable Parent projection"
    )
    duplicate_child_ids: int = Field(
        ge=0, description="duplicate IDs present in the durable Child projection"
    )
    duplicate_event_keys: int = Field(
        ge=0, description="duplicate event keys present in durable Event rows"
    )
    replayed_parent_count: int = Field(
        default=0,
        ge=0, description="replayed Parent upsert attempts deduplicated by the fake"
    )
    replayed_child_count: int = Field(
        default=0,
        ge=0, description="replayed Child upsert attempts deduplicated by the fake"
    )
    replayed_event_count: int = Field(
        default=0,
        ge=0, description="replayed Event append attempts deduplicated by event_key"
    )
    artifact_quarantine_count: int = Field(
        default=0,
        ge=0, description="missing or invalid Artifacts quarantined"
    )
    memory_scope_user_ids: tuple[str, ...] = ()
    user_leak_count: int = Field(ge=0)

    @model_validator(mode="before")
    @classmethod
    def _normalize_maps(cls, value: object) -> object:
        if not isinstance(value, dict):
            return value
        normalized = dict(value)
        for field_name in ("scenarios", "scenario_invariants"):
            raw = value.get(field_name)
            if field_name == "scenario_invariants" and raw is None:
                normalized[field_name] = {
                    scenario: True for scenario in DRILL_SCENARIOS
                }
                continue
            if not isinstance(raw, dict) or set(raw) != set(DRILL_SCENARIOS):
                raise ValueError(f"{field_name} must contain exactly the fixed drill set")
            normalized[field_name] = {
                scenario: raw[scenario] for scenario in DRILL_SCENARIOS
            }
        return normalized

    @property
    def gate_passed(self) -> bool:
        return (
            all(status == "passed" for status in self.scenarios.values())
            and all(self.scenario_invariants.values())
            and self.duplicate_parent_ids == 0
            and self.duplicate_child_ids == 0
            and self.duplicate_event_keys == 0
            and self.user_leak_count == 0
        )

    def deterministic_json(self) -> str:
        """Return canonical JSON without timestamps, random IDs or whitespace."""
        return json.dumps(
            self.model_dump(mode="json"),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ) + "\n"


@dataclass(slots=True)
class _IdempotentIds:
    rows: dict[str, str] = field(default_factory=dict)
    attempts: list[str] = field(default_factory=list)

    def upsert(self, identifier: str, user_id: str) -> None:
        self.attempts.append(identifier)
        self.rows.setdefault(identifier, user_id)


@dataclass(slots=True)
class _DurableEventRepository:
    """Faithful EventRepository fake: duplicate event keys return the first row."""

    rows: dict[str, AgentEvent] = field(default_factory=dict)
    attempts: list[str] = field(default_factory=list)

    def append(self, event: AgentEvent) -> int:
        self.attempts.append(event.event_key)
        existing = self.rows.get(event.event_key)
        if existing is not None:
            return existing.id or 0
        assigned = replace(event, id=len(self.rows) + 1)
        self.rows[event.event_key] = assigned
        return assigned.id or 0

    def list_after(
        self, run_id: str, scope: UserScope, after_id: int, limit: int
    ) -> list[AgentEvent]:
        return [
            event
            for event in sorted(self.rows.values(), key=lambda item: item.id or 0)
            if event.run_id == run_id
            and event.user_id == scope.user_id
            and (event.id or 0) > after_id
        ][:limit]


@dataclass(slots=True)
class _ActivationIndex:
    active: dict[str, str] = field(default_factory=dict)
    interrupted: bool = False

    def activate(self, user_id: str, version_id: str) -> None:
        self.active[user_id] = version_id
        if not self.interrupted:
            self.interrupted = True
            raise RuntimeError("activation interrupted after deterministic write")


@dataclass(slots=True)
class _DrillState:
    artifacts: LocalArtifactStore
    expected_user_id: str = "attacker"
    parents: _IdempotentIds = field(default_factory=_IdempotentIds)
    children: _IdempotentIds = field(default_factory=_IdempotentIds)
    events: _DurableEventRepository = field(default_factory=_DurableEventRepository)
    scenario_invariants: dict[str, bool] = field(default_factory=dict)
    scenario_statuses: dict[str, ScenarioStatus] = field(default_factory=dict)
    artifact_quarantine_count: int = 0
    memory_scope_user_ids: tuple[str, ...] = ()
    leaked_user_ids: set[str] = field(default_factory=set)


class _RedisFailure(RuntimeError):
    pass


def run_recovery_drill() -> RecoveryDrillReport:
    """Execute all drills against fresh isolated state and return their report."""
    with tempfile.TemporaryDirectory(prefix="agentic-rag-drill-") as root:
        state = _DrillState(artifacts=LocalArtifactStore(Path(root)))
        runners = {
            "api_restart_during_sse": _api_restart_during_sse,
            "query_worker_termination_after_retrieval": _query_worker_termination_after_retrieval,
            "ingestion_worker_termination_after_staging": _ingestion_worker_termination_after_staging,
            "outbox_mysql_success_redis_failure": _outbox_mysql_success_redis_failure,
            "es_activation_interruption": _es_activation_interruption,
            "missing_artifact_quarantine": _missing_artifact_quarantine,
            "mem0_unavailable": _mem0_unavailable,
        }
        for scenario in DRILL_SCENARIOS:
            try:
                invariant = runners[scenario](state)
            except Exception:
                invariant = False
            state.scenario_invariants[scenario] = invariant is True
            state.scenario_statuses[scenario] = "passed" if invariant is True else "failed"

        attacker = UserScope(user_id=state.expected_user_id)
        visible_events = state.events.list_after("run-1", attacker, after_id=0, limit=100)
        state.leaked_user_ids.update(
            event.user_id
            for event in visible_events
            if event.user_id != state.expected_user_id
        )
        return RecoveryDrillReport(
            scenarios={
                scenario: state.scenario_statuses.get(scenario, "failed")
                for scenario in DRILL_SCENARIOS
            },
            scenario_invariants={
                scenario: state.scenario_invariants.get(scenario, False)
                for scenario in DRILL_SCENARIOS
            },
            duplicate_parent_ids=_duplicates(list(state.parents.rows)),
            duplicate_child_ids=_duplicates(list(state.children.rows)),
            duplicate_event_keys=_duplicates(list(state.events.rows)),
            replayed_parent_count=_replays(state.parents.attempts),
            replayed_child_count=_replays(state.children.attempts),
            replayed_event_count=_replays(state.events.attempts),
            artifact_quarantine_count=state.artifact_quarantine_count,
            memory_scope_user_ids=state.memory_scope_user_ids,
            user_leak_count=len(state.leaked_user_ids),
        )


def _event(key: str, user_id: str, *, run_id: str = "run-1") -> AgentEvent:
    return AgentEvent(
        event_key=key,
        trace_id=run_id,
        run_id=run_id,
        user_id=user_id,
        event_type="PROGRESS",
        summary="completed",
        runtime_config_snapshot_id="snapshot-v1",
        node_name="drill",
    )


def _api_restart_during_sse(state: _DrillState) -> bool:
    attacker = state.expected_user_id
    state.events.append(_event("run-1:retrieval", attacker))
    state.events.append(_event("run-1:answer", attacker))
    # A victim event shares the run id but must remain outside the attacker's
    # reconnect stream because the EventRepository applies UserScope.
    state.events.append(_event("run-1:victim-private", "victim"))
    # Reconnect/replay intentionally appends the same key. The repository
    # returns the original row instead of creating a duplicate Event.
    state.events.append(_event("run-1:answer", attacker))
    visible = state.events.list_after("run-1", UserScope(user_id=attacker), 0, 100)
    return len(visible) == 2 and all(row.user_id == attacker for row in visible)


def _query_worker_termination_after_retrieval(state: _DrillState) -> bool:
    parent_id = "parent:attacker:run-1"
    child_id = "child:attacker:run-1"
    parent_rows_before = set(state.parents.rows)
    child_rows_before = set(state.children.rows)
    parent_attempts_before = len(state.parents.attempts)
    child_attempts_before = len(state.children.attempts)
    state.parents.upsert(parent_id, state.expected_user_id)
    state.children.upsert(child_id, state.expected_user_id)
    # Worker termination after Retrieval causes a deterministic replay.
    state.parents.upsert(parent_id, state.expected_user_id)
    state.children.upsert(child_id, state.expected_user_id)
    state.events.append(_event("run-1:retrieval-completed", state.expected_user_id))
    return (
        parent_id in state.parents.rows
        and child_id in state.children.rows
        and len(set(state.parents.rows) - parent_rows_before) == 1
        and len(set(state.children.rows) - child_rows_before) == 1
        and _replays(state.parents.attempts[parent_attempts_before:]) >= 1
        and _replays(state.children.attempts[child_attempts_before:]) >= 1
    )


def _ingestion_worker_termination_after_staging(state: _DrillState) -> bool:
    parent_id = "parent:attacker:document-1:version-1:0"
    child_id = "child:attacker:document-1:version-1:0"
    parent_rows_before = set(state.parents.rows)
    child_rows_before = set(state.children.rows)
    parent_attempts_before = len(state.parents.attempts)
    child_attempts_before = len(state.children.attempts)
    state.parents.upsert(parent_id, state.expected_user_id)
    state.children.upsert(child_id, state.expected_user_id)
    # Publication restart replays staging writes through the same idempotent
    # deterministic IDs, not a pre-deduplicated input list.
    state.parents.upsert(parent_id, state.expected_user_id)
    state.children.upsert(child_id, state.expected_user_id)
    state.events.append(_event("ingestion:job-1:staged", state.expected_user_id))
    return (
        parent_id in state.parents.rows
        and child_id in state.children.rows
        and len(set(state.parents.rows) - parent_rows_before) == 1
        and len(set(state.children.rows) - child_rows_before) == 1
        and _replays(state.parents.attempts[parent_attempts_before:]) >= 1
        and _replays(state.children.attempts[child_attempts_before:]) >= 1
    )


def _outbox_mysql_success_redis_failure(state: _DrillState) -> bool:
    pending = {"outbox-1": "pending"}
    try:
        _publish_to_redis(pending["outbox-1"], fail=True)
    except _RedisFailure:
        # MySQL remains the durable pending authority after Redis fails.
        assert pending["outbox-1"] == "pending"
    _publish_to_redis(pending["outbox-1"], fail=False)
    pending["outbox-1"] = "dispatched"
    state.events.append(_event("outbox:outbox-1:dispatched", state.expected_user_id))
    return pending["outbox-1"] == "dispatched"


def _es_activation_interruption(state: _DrillState) -> bool:
    index = _ActivationIndex()
    try:
        index.activate(state.expected_user_id, "version-2")
    except RuntimeError:
        pass
    index.activate(state.expected_user_id, "version-2")
    state.events.append(_event("index:attacker:version-2:active", state.expected_user_id))
    return index.active.get(state.expected_user_id) == "version-2"


def _missing_artifact_quarantine(state: _DrillState) -> bool:
    missing = ArtifactRef(
        uri="artifact://documents/attacker/document-1/version-1/missing.json",
        sha256="0" * 64,
        size_bytes=0,
    )
    if state.artifacts.verify(missing):
        return False
    state.artifact_quarantine_count += 1
    state.events.append(_event("ingestion:version-1:quarantined", state.expected_user_id))
    return state.artifact_quarantine_count == 1


class _FixtureMemoryClient(MemoryClient):
    async def add(
        self,
        messages: list[dict[str, str]],
        *,
        user_id: str,
        metadata: dict[str, object],
    ) -> object:
        del messages, user_id, metadata
        return None

    async def search(self, query: str, *, user_id: str, limit: int) -> object:
        del query, user_id, limit
        # Return both namespaces deliberately; MemoryServiceImpl must apply
        # the server-owned scope rather than trusting provider user_id input.
        return {
            "results": [
                {
                    "id": "memory-attacker",
                    "memory": "attacker preference",
                    "user_id": "attacker",
                },
                {
                    "id": "memory-victim",
                    "memory": "victim secret",
                    "user_id": "victim",
                },
            ]
        }

    async def get_all(self, *, user_id: str) -> object:
        del user_id
        return {"results": []}

    async def delete(self, memory_id: str) -> object:
        del memory_id
        return None


class _UnavailableMemoryClient(_FixtureMemoryClient):
    async def search(self, query: str, *, user_id: str, limit: int) -> object:
        del query, user_id, limit
        raise ConnectionError("mem0 unavailable")


class _NoopMemoryTombstones(MemoryTombstoneStore):
    async def request(self, scope: UserScope, memory_id: str) -> Tombstone:
        return Tombstone(user_id=scope.user_id, memory_id=memory_id)

    async def list_pending(self, limit: int = 100) -> list[Tombstone]:
        del limit
        return []

    async def mark_completed(self, scope: UserScope, memory_id: str) -> None:
        del scope, memory_id

    async def mark_retry(self, scope: UserScope, memory_id: str, error: str) -> None:
        del scope, memory_id, error


def _mem0_unavailable(state: _DrillState) -> bool:
    async def read_contexts() -> tuple[object, object]:
        scope = UserScope(user_id=state.expected_user_id)
        tombstones = _NoopMemoryTombstones()
        scoped_service = MemoryServiceImpl(
            _FixtureMemoryClient(), tombstones=tombstones, policy_version="drill-v1"
        )
        unavailable_service = MemoryServiceImpl(
            _UnavailableMemoryClient(), tombstones=tombstones, policy_version="drill-v1"
        )
        return (
            await scoped_service.load_context(scope, "summarize"),
            await unavailable_service.load_context(scope, "summarize"),
        )

    scoped_context, unavailable_context = asyncio.run(read_contexts())
    records = getattr(scoped_context, "records", ())
    state.memory_scope_user_ids = tuple(record.user_id for record in records)
    state.leaked_user_ids.update(
        user_id
        for user_id in state.memory_scope_user_ids
        if user_id != state.expected_user_id
    )
    state.events.append(_event("memory:attacker:degraded", state.expected_user_id))
    return state.memory_scope_user_ids == (state.expected_user_id,) and bool(
        getattr(unavailable_context, "degraded", False)
    ) and not getattr(unavailable_context, "records", ())


def _publish_to_redis(value: str, *, fail: bool) -> None:
    del value
    if fail:
        raise _RedisFailure("isolated redis fault injection")


def _duplicates(values: list[str]) -> int:
    counts = Counter(values)
    return sum(count - 1 for count in counts.values() if count > 1)


def _replays(values: list[str]) -> int:
    return _duplicates(values)


def _atomic_write(path: Path, payload: str) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    temporary_path = Path(temporary)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, destination)
        directory_fd = os.open(destination.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary_path.unlink(missing_ok=True)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="atomic JSON report path")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    report = run_recovery_drill()
    _atomic_write(args.output, report.deterministic_json())
    return 0 if report.gate_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
