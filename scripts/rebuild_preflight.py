#!/usr/bin/env python3
"""Non-destructive preflight/report gates for the ingestion-v2/index-v3 rebuild.

This module deliberately does not connect to MySQL, Elasticsearch, Redis, or
the file system.  It validates an operator-supplied dry-run inventory and
emits a report; the destructive backup/delete/re-ingest runbook remains a
separate, explicitly approved operation.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from agentic_rag.models.indexing import validate_index_generation  # noqa: E402


TARGET_PIPELINE_VERSION = "ingestion-v2"
TARGET_INDEX_GENERATION = "index-v3"
TARGET_ALIAS = "agenticrag-children-active"
TARGET_PHYSICAL_INDEX = "agenticrag-children-index-v3"


class RebuildPreflightError(ValueError):
    """Raised when a rebuild dry-run is not safe to report as ready."""


def resolve_physical_index(index_generation: str) -> str:
    """Resolve one exact physical index name and reject wildcard targets."""
    if not isinstance(index_generation, str) or not index_generation.strip():
        raise RebuildPreflightError("index generation target is unresolved")
    if any(marker in index_generation for marker in ("*", "?", "[", "]")):
        raise RebuildPreflightError("wildcard physical index targets are forbidden")
    try:
        validate_index_generation(index_generation)
    except (TypeError, ValueError) as error:
        raise RebuildPreflightError("index generation target is unresolved") from error
    if index_generation != TARGET_INDEX_GENERATION:
        raise RebuildPreflightError(
            f"preflight requires {TARGET_INDEX_GENERATION}, got {index_generation}"
        )
    return f"agenticrag-children-{index_generation}"


def run_preflight(
    *,
    index_generation: str,
    physical_index: str | None = None,
    backup_verified: bool,
    workers_quiescent: bool,
    sources_staged: bool,
    reorder_events: Sequence[Mapping[str, object]] = (),
    parent_count: int = 0,
    parent_token_distribution: Sequence[int] = (),
    source_span_coverage: Mapping[str, object] | None = None,
    structural_boundary_changes: Sequence[Mapping[str, object]] = (),
) -> dict[str, object]:
    """Validate a dry-run inventory and return a JSON-safe readiness report."""
    resolved = resolve_physical_index(index_generation)
    if physical_index is not None and physical_index != resolved:
        raise RebuildPreflightError(
            f"physical index must resolve exactly to {resolved}, got {physical_index}"
        )
    if backup_verified is not True:
        raise RebuildPreflightError("verified backup is required before rebuild")
    if workers_quiescent is not True:
        raise RebuildPreflightError("workers must be quiescent before rebuild")
    if sources_staged is not True:
        raise RebuildPreflightError("source staging inventory is required before rebuild")
    if isinstance(parent_count, bool) or not isinstance(parent_count, int) or parent_count < 0:
        raise RebuildPreflightError("parent_count must be a non-negative integer")

    safe_reorders: list[dict[str, object]] = []
    for event in reorder_events:
        if not isinstance(event, Mapping):
            raise RebuildPreflightError("reorder event is malformed")
        reordered = event.get("reordered", event.get("changed", False)) is True
        triggered = event.get("triggered") is True
        reason = event.get("reason")
        if reordered and not triggered:
            raise RebuildPreflightError(
                "non-triggering source reorder would invalidate the preflight"
            )
        if reordered and (not isinstance(reason, str) or not reason.strip()):
            raise RebuildPreflightError("every triggered reorder requires a reason")
        safe_reorders.append(
            {
                "source_id": _safe_identifier(event.get("source_id")),
                "reordered": reordered,
                "triggered": triggered,
                "reason": reason if isinstance(reason, str) else None,
            }
        )

    token_counts: list[int] = []
    for value in parent_token_distribution:
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise RebuildPreflightError(
                "parent token distribution must contain non-negative integers"
            )
        token_counts.append(value)
    if (parent_count or token_counts) and parent_count != len(token_counts):
        raise RebuildPreflightError(
            "parent token distribution must contain one entry per Parent"
        )
    raw_coverage = dict(source_span_coverage or {})
    coverage = (
        {"before": raw_coverage.get("before"), "after": raw_coverage.get("after")}
        if raw_coverage
        else {}
    )
    before = coverage.get("before")
    after = coverage.get("after")
    if coverage and (
        isinstance(before, bool)
        or isinstance(after, bool)
        or not isinstance(before, int)
        or not isinstance(after, int)
        or before < 0
        or after < 0
    ):
        raise RebuildPreflightError(
            "source-span coverage requires non-negative before and after counts"
        )
    if coverage and before != after:
        raise RebuildPreflightError(
            "source-span coverage changed during the dry-run"
        )
    safe_boundaries: list[dict[str, object]] = []
    for change in structural_boundary_changes:
        if not isinstance(change, Mapping):
            raise RebuildPreflightError("structural-boundary change is malformed")
        raw_reason = change.get("reason")
        reason = raw_reason.strip() if isinstance(raw_reason, str) else None
        safe_boundaries.append(
            {
                "source_id": _safe_identifier(change.get("source_id")),
                "reason": reason or None,
            }
        )
    return {
        "ready": True,
        "mode": "dry-run",
        "pipeline_version": TARGET_PIPELINE_VERSION,
        "index_generation": TARGET_INDEX_GENERATION,
        "alias": TARGET_ALIAS,
        "physical_index": resolved,
        "checks": {
            "backup_verified": True,
            "workers_quiescent": True,
            "sources_staged": True,
            "reorder_reasons_valid": True,
            "exact_index_target": True,
        },
        "reordered_blocks": safe_reorders,
        "structural_boundary_changes": safe_boundaries,
        "parent_count": parent_count,
        "parent_token_distribution": token_counts,
        "source_span_coverage": coverage,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index-generation", required=True)
    parser.add_argument("--physical-index")
    parser.add_argument("--backup-verified", action="store_true")
    parser.add_argument("--workers-quiescent", action="store_true")
    parser.add_argument("--sources-staged", action="store_true")
    parser.add_argument("--reorder-report", type=Path)
    parser.add_argument("--parent-count", type=int, default=0)
    parser.add_argument("--parent-token-distribution", nargs="*", type=int, default=[])
    parser.add_argument("--source-span-before", type=int)
    parser.add_argument("--source-span-after", type=int)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        reorder_events: Sequence[Mapping[str, object]] = ()
        if args.reorder_report is not None:
            value = json.loads(args.reorder_report.read_text(encoding="utf-8"))
            if not isinstance(value, list):
                raise RebuildPreflightError("reorder report must be a JSON array")
            reorder_events = [
                item for item in value if isinstance(item, Mapping)
            ]
            if len(reorder_events) != len(value):
                raise RebuildPreflightError("reorder report contains malformed entries")
        coverage: Mapping[str, object] | None = None
        if args.source_span_before is not None or args.source_span_after is not None:
            coverage = {
                "before": args.source_span_before,
                "after": args.source_span_after,
            }
        report = run_preflight(
            index_generation=args.index_generation,
            physical_index=args.physical_index,
            backup_verified=args.backup_verified,
            workers_quiescent=args.workers_quiescent,
            sources_staged=args.sources_staged,
            reorder_events=reorder_events,
            parent_count=args.parent_count,
            parent_token_distribution=tuple(args.parent_token_distribution),
            source_span_coverage=coverage,
        )
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(
            json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    except (OSError, RebuildPreflightError, json.JSONDecodeError) as error:
        print(f"rebuild preflight refused: {type(error).__name__}: {error}", file=sys.stderr)
        return 2
    return 0


def _safe_identifier(value: object) -> str:
    return value if isinstance(value, str) and value else "unknown"


if __name__ == "__main__":
    raise SystemExit(main())
