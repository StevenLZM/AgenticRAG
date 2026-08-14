"""Deterministic JSON-only summaries for offline evaluation results."""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from evals.run import EvalCaseResult


class MixedSnapshotError(ValueError):
    """Raised when a report silently combines different runtime snapshots."""


def build_summary(
    results: Iterable[EvalCaseResult],
    *,
    baseline_ids: Mapping[str, str] | None = None,
) -> dict[str, object]:
    """Aggregate immutable case rows without exposing prompts or raw payloads.

    A mixed snapshot report is only valid when the caller supplies explicit
    named baseline IDs.  This makes an A/B comparison intentional and keeps the
    default report comparable across reruns.
    """

    rows = sorted(list(results), key=lambda item: item.case_id)
    snapshots = sorted({item.runtime_config_snapshot_id for item in rows})
    allowed = _normalize_baselines(baseline_ids)
    if len(snapshots) > 1:
        if not allowed or not set(snapshots).issubset(set(allowed.values())):
            raise MixedSnapshotError(
                "results contain mixed runtime_config_snapshot_id values; "
                "supply named baseline_ids to compare them"
            )
    metrics = _average_metrics(rows, "deterministic_metrics")
    ragas = _ragas_summary(rows)
    summary: dict[str, object] = {
        "schema_version": 1,
        "completed_cases": len(rows),
        "case_ids": [row.case_id for row in rows],
        "runtime_config_snapshot_id": snapshots[0] if len(snapshots) == 1 else None,
        "runtime_config_snapshot_ids": snapshots,
        "metrics": metrics,
        "ragas": ragas,
    }
    if allowed:
        summary["baseline_ids"] = dict(sorted(allowed.items()))
    if len(snapshots) > 1:
        summary["comparisons"] = _comparison_summaries(rows, allowed)
    return summary


def write_summary(path: str | Path, summary: Mapping[str, object]) -> None:
    """Write a canonical summary using temp-file + ``os.replace``."""

    encoded = json.dumps(summary, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
    atomic_write_text(Path(path), encoded)


def atomic_write_text(path: Path, content: str) -> None:
    """Atomically replace a UTF-8 text file and remove failed temp files."""

    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temp_path = Path(temporary)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, path)
        _fsync_directory(path.parent)
    finally:
        if temp_path.exists():
            temp_path.unlink()


def _normalize_baselines(
    baseline_ids: Mapping[str, str] | None,
) -> dict[str, str]:
    if baseline_ids is None:
        return {}
    if not isinstance(baseline_ids, Mapping):
        raise ValueError("baseline_ids must be a named baseline mapping of baseline to snapshot")
    normalized: dict[str, str] = {}
    for name, snapshot in baseline_ids.items():
        if not isinstance(name, str) or not name.strip() or not isinstance(snapshot, str) or not snapshot.strip():
            raise ValueError("baseline IDs must map non-empty names to non-empty snapshots")
        normalized[name] = snapshot
    return normalized


def _average_metrics(rows: Sequence[EvalCaseResult], field: str) -> dict[str, float | int]:
    totals: dict[str, float] = {}
    counts: dict[str, int] = {}
    for row in rows:
        values = getattr(row, field)
        for name, value in values.items():
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                continue
            totals[name] = totals.get(name, 0.0) + float(value)
            counts[name] = counts.get(name, 0) + 1
    return {
        name: _integer_if_whole(totals[name] / counts[name])
        for name in sorted(totals)
        if counts[name]
    }


def _ragas_summary(rows: Sequence[EvalCaseResult]) -> dict[str, object]:
    statuses: list[str] = []
    totals: dict[str, float] = {}
    counts: dict[str, int] = {}
    for row in rows:
        values = row.ragas_metrics
        status = values.get("status")
        if isinstance(status, str):
            statuses.append(status)
        for name, value in values.items():
            if name in {"status", "reason"} or isinstance(value, str) or value is None:
                continue
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                continue
            totals[name] = totals.get(name, 0.0) + float(value)
            counts[name] = counts.get(name, 0) + 1
    status = "unavailable" if not statuses or all(value == "unavailable" for value in statuses) else "available"
    result: dict[str, object] = {
        "status": status,
        "metrics": {
            name: _integer_if_whole(totals[name] / counts[name])
            for name in sorted(totals)
            if counts[name]
        },
    }
    if status == "unavailable":
        result["reason"] = "ragas backend is not installed or configured"
    return result


def _comparison_summaries(
    rows: Sequence[EvalCaseResult], baseline_ids: Mapping[str, str]
) -> dict[str, object]:
    result: dict[str, object] = {}
    by_snapshot: dict[str, list[EvalCaseResult]] = {}
    for row in rows:
        by_snapshot.setdefault(row.runtime_config_snapshot_id, []).append(row)
    for name, snapshot in sorted(baseline_ids.items()):
        selected = by_snapshot.get(snapshot, [])
        result[name] = {
            "runtime_config_snapshot_id": snapshot,
            "completed_cases": len(selected),
            "metrics": _average_metrics(selected, "deterministic_metrics"),
            "ragas": _ragas_summary(selected),
        }
    return result


def _integer_if_whole(value: float) -> int | float:
    return int(value) if value.is_integer() else value


def _fsync_directory(path: Path) -> None:
    """Best-effort directory durability; unsupported filesystems are harmless."""

    try:
        descriptor = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    except OSError:
        pass
    finally:
        os.close(descriptor)


__all__ = ["MixedSnapshotError", "atomic_write_text", "build_summary", "write_summary"]
