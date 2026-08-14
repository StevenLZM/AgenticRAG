"""Fault-injection recovery drills over isolated in-memory state."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from scripts.run_recovery_drill import (
    DRILL_SCENARIOS,
    RecoveryDrillReport,
    run_recovery_drill,
)


pytestmark = pytest.mark.e2e


def test_recovery_drill_report_is_deterministic_and_has_all_scenarios() -> None:
    first = run_recovery_drill()
    second = run_recovery_drill()

    assert isinstance(first, RecoveryDrillReport)
    assert first == second
    assert tuple(first.scenarios) == DRILL_SCENARIOS
    assert all(status == "passed" for status in first.scenarios.values())
    assert first.duplicate_parent_ids == 0
    assert first.duplicate_child_ids == 0
    assert first.duplicate_event_keys == 0
    assert first.user_leak_count == 0
    assert first.replayed_event_count > 0
    assert first.replayed_parent_count > 0
    assert first.replayed_child_count > 0
    assert first.artifact_quarantine_count == 1
    assert all(first.scenario_invariants.values())


def test_recovery_drill_cli_writes_atomic_json_and_returns_success(tmp_path: Path) -> None:
    output = tmp_path / "drills" / "latest.json"
    completed = subprocess.run(
        [sys.executable, "scripts/run_recovery_drill.py", "--output", str(output)],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["scenarios"]["api_restart_during_sse"] == "passed"
    assert payload["duplicate_event_keys"] == 0


def test_recovery_drill_report_rejects_nonzero_gate_counts() -> None:
    report = RecoveryDrillReport(
        scenarios={scenario: "passed" for scenario in DRILL_SCENARIOS},
        scenario_invariants={scenario: True for scenario in DRILL_SCENARIOS},
        duplicate_parent_ids=0,
        duplicate_child_ids=0,
        duplicate_event_keys=0,
        replayed_parent_count=0,
        replayed_child_count=0,
        replayed_event_count=0,
        artifact_quarantine_count=0,
        user_leak_count=0,
    )
    assert report.gate_passed is True

    failed = report.model_copy(update={"duplicate_parent_ids": 1})
    assert failed.gate_passed is False

    failed_scenario = report.model_copy(
        update={
            "scenarios": {
                **report.scenarios,
                DRILL_SCENARIOS[0]: "failed",
            }
        }
    )
    assert failed_scenario.gate_passed is False
