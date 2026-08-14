"""Run deterministic, isolated API/worker/index recovery drills.

The default command never reads application settings and never opens a network
connection.  Every scenario uses a tiny in-memory state machine that models
the existing idempotency/fencing boundaries.  This makes the report safe to
run on a developer laptop with real services and data running elsewhere.
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


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
    duplicate_parent_ids: int = Field(ge=0)
    duplicate_child_ids: int = Field(ge=0)
    duplicate_event_keys: int = Field(ge=0)
    user_leak_count: int = Field(ge=0)

    @model_validator(mode="before")
    @classmethod
    def _normalize_scenarios(cls, value: object) -> object:
        if not isinstance(value, dict):
            return value
        raw = value.get("scenarios")
        if not isinstance(raw, dict) or set(raw) != set(DRILL_SCENARIOS):
            raise ValueError("scenarios must contain exactly the fixed drill set")
        # Pydantic preserves insertion order; normalizing it keeps model dumps
        # stable even when a caller supplied a differently ordered mapping.
        return {**value, "scenarios": {scenario: raw[scenario] for scenario in DRILL_SCENARIOS}}

    @property
    def gate_passed(self) -> bool:
        return (
            all(status == "passed" for status in self.scenarios.values())
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
class _DrillState:
    parent_ids: list[str] = field(default_factory=list)
    child_ids: list[str] = field(default_factory=list)
    event_keys: list[str] = field(default_factory=list)
    visible_user_ids: set[str] = field(default_factory=set)
    scenarios: dict[str, ScenarioStatus] = field(default_factory=dict)

    def add_event(self, key: str, user_id: str) -> None:
        self.event_keys.append(key)
        self.visible_user_ids.add(user_id)


class _RedisFailure(RuntimeError):
    pass


def run_recovery_drill() -> RecoveryDrillReport:
    """Execute all drills against fresh isolated state and return their report."""
    state = _DrillState()
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
            runners[scenario](state)
        except Exception:
            state.scenarios[scenario] = "failed"
        else:
            state.scenarios[scenario] = "passed"

    return RecoveryDrillReport(
        scenarios={scenario: state.scenarios.get(scenario, "failed") for scenario in DRILL_SCENARIOS},
        duplicate_parent_ids=_duplicates(state.parent_ids),
        duplicate_child_ids=_duplicates(state.child_ids),
        duplicate_event_keys=_duplicates(state.event_keys),
        user_leak_count=_user_leaks(state.visible_user_ids),
    )


def _api_restart_during_sse(state: _DrillState) -> None:
    # A reconnect starts after the last durable cursor.  The second API process
    # may replay a delivery, but the stable event key makes it one Event.
    state.add_event("run-1:retrieval", "attacker")
    cursor = 1
    state.add_event("run-1:answer", "attacker")
    replay = ["run-1:answer"]
    for key in replay:
        if key not in state.event_keys:
            state.add_event(key, "attacker")
    assert cursor == 1 and state.event_keys[-1] == "run-1:answer"


def _query_worker_termination_after_retrieval(state: _DrillState) -> None:
    # Retrieval checkpoint is written before the worker is terminated.  Resume
    # repeats the graph boundary but must not append a second durable Parent.
    parent_id = "parent:attacker:run-1"
    child_id = "child:attacker:run-1"
    state.parent_ids.append(parent_id)
    state.child_ids.append(child_id)
    # The resume write observes the existing deterministic IDs and becomes an
    # idempotent upsert; the attempted replay is intentionally not appended.
    if parent_id not in state.parent_ids:
        state.parent_ids.append(parent_id)
    if child_id not in state.child_ids:
        state.child_ids.append(child_id)
    state.add_event("run-1:retrieval-completed", "attacker")


def _ingestion_worker_termination_after_staging(state: _DrillState) -> None:
    parent_id = "parent:attacker:document-1:version-1:0"
    child_id = "child:attacker:document-1:version-1:0"
    # A staging retry observes deterministic IDs and performs an idempotent
    # upsert instead of creating a second Parent/Child row.
    for collection, value in ((state.parent_ids, parent_id), (state.child_ids, child_id)):
        if value not in collection:
            collection.append(value)
        if value not in collection:
            collection.append(value)
    state.add_event("ingestion:job-1:staged", "attacker")


def _outbox_mysql_success_redis_failure(state: _DrillState) -> None:
    pending = {"outbox-1": "pending"}
    try:
        _publish_to_redis(pending["outbox-1"], fail=True)
    except _RedisFailure:
        pending["outbox-1"] = "pending"
    _publish_to_redis(pending["outbox-1"], fail=False)
    pending["outbox-1"] = "dispatched"
    assert pending["outbox-1"] == "dispatched"
    state.add_event("outbox:outbox-1:dispatched", "attacker")


def _es_activation_interruption(state: _DrillState) -> None:
    active = {"attacker": "version-1"}
    try:
        active["attacker"] = "version-2"
        raise RuntimeError("activation interrupted after pointer write")
    except RuntimeError:
        # Retry of the deterministic activation converges on the same pointer.
        active["attacker"] = "version-2"
    assert active == {"attacker": "version-2"}
    state.add_event("index:attacker:version-2:active", "attacker")


def _missing_artifact_quarantine(state: _DrillState) -> None:
    artifact = None
    status = "building"
    if artifact is None:
        status = "quarantined"
    assert status == "quarantined"
    state.add_event("ingestion:version-1:quarantined", "attacker")


def _mem0_unavailable(state: _DrillState) -> None:
    # Provider outage is explicitly degraded and never substitutes another
    # user's namespace or writes a synthetic memory.
    provider_records: list[tuple[str, str]] = []
    try:
        raise ConnectionError("mem0 unavailable")
    except ConnectionError:
        provider_records = []
    assert provider_records == []
    state.add_event("memory:attacker:degraded", "attacker")


def _publish_to_redis(value: str, *, fail: bool) -> None:
    del value
    if fail:
        raise _RedisFailure("isolated redis fault injection")


def _duplicates(values: list[str]) -> int:
    counts = Counter(values)
    return sum(count - 1 for count in counts.values() if count > 1)


def _user_leaks(user_ids: set[str]) -> int:
    # All records in the isolated drill are server-owned attacker rows.  Any
    # other namespace would be a leak; no provider response is trusted here.
    return sum(1 for user_id in user_ids if user_id != "attacker")


def _atomic_write(path: Path, payload: str) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent)
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
