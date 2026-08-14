"""Pure deterministic metrics for retrieval and event-derived evaluation."""

from __future__ import annotations

import math
import re
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence, Set
from typing import Any


_EVENT_TYPE = re.compile(r"^[A-Z][A-Z0-9_]{0,63}$")


def recall_at_k(ranked_ids: Sequence[str], relevant_ids: Set[str], k: int) -> float:
    """Return binary relevance recall over the first bounded ``k`` results."""

    cutoff = _bounded_k(k, len(ranked_ids))
    relevant = {value for value in relevant_ids if isinstance(value, str) and value}
    if cutoff == 0 or not relevant:
        return 0.0
    retrieved = {value for value, _rank in _unique_prefix(ranked_ids, cutoff)}
    return len(retrieved.intersection(relevant)) / len(relevant)


def mrr(ranked_ids: Sequence[str], relevant_ids: Set[str], k: int | None = None) -> float:
    """Return reciprocal rank of the first relevant result, or zero."""

    cutoff = _bounded_k(len(ranked_ids) if k is None else k, len(ranked_ids))
    relevant = {value for value in relevant_ids if isinstance(value, str) and value}
    for value, rank in _unique_prefix(ranked_ids, cutoff):
        if value in relevant:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(ranked_ids: Sequence[str], relevant_ids: Set[str], k: int) -> float:
    """Return binary-relevance NDCG with a deterministic ideal ranking."""

    cutoff = _bounded_k(k, len(ranked_ids))
    relevant = {value for value in relevant_ids if isinstance(value, str) and value}
    if cutoff == 0 or not relevant:
        return 0.0
    retrieved = _unique_prefix(ranked_ids, cutoff)
    dcg = sum(
        1.0 / math.log2(rank + 1)
        for value, rank in retrieved
        if value in relevant
    )
    ideal_count = min(cutoff, len(relevant))
    ideal = sum(1.0 / math.log2(index + 2) for index in range(ideal_count))
    return dcg / ideal if ideal else 0.0


def aggregate_loop_metrics(
    events: Iterable[Mapping[str, Any]],
    user_id: str,
    runtime_config_snapshot_id: str,
) -> dict[str, float | int]:
    """Aggregate loop counters from already-recorded, scoped Agent Events.

    The function intentionally consumes event metadata only.  It neither calls
    the graph nor reconstructs retrieval/audit decisions.  Rows from another
    user or runtime snapshot are discarded before any counters are updated.
    """

    scoped = list(_scoped_events(events, user_id=user_id, snapshot_id=runtime_config_snapshot_id))
    by_run: dict[str, dict[str, Any]] = defaultdict(
        lambda: {"rounds": 0.0, "repair_count": 0.0, "loop_limit": False, "terminal": None}
    )
    for event in scoped:
        state = by_run[event["run_id"]]
        event_type = event["event_type"]
        attrs = event["attributes"]
        rounds = _finite_nonnegative(attrs.get("retrieval_rounds"))
        if rounds is not None:
            state["rounds"] = max(float(state["rounds"]), rounds)
        elif event_type == "RETRIEVAL_COMPLETED" and "retrieval_rounds" not in attrs:
            state["rounds"] = float(state["rounds"]) + 1.0
        repairs = _finite_nonnegative(attrs.get("repair_count"))
        if repairs is not None:
            state["repair_count"] = max(float(state["repair_count"]), repairs)
        if "REPAIR" in event_type:
            state["repair_count"] = max(float(state["repair_count"]), 1.0)
        reason = attrs.get("termination_reason")
        if event_type in {"RUN_COMPLETED", "RUN_FAILED", "RUN_CANCELLED", "ANSWER_FINALIZED"}:
            state["terminal"] = reason if isinstance(reason, str) else ""

    runs = len(by_run)
    rounds_total = sum(float(item["rounds"]) for item in by_run.values())
    repair_count = sum(float(item["repair_count"]) for item in by_run.values())
    loop_limit_count = sum(
        1 for item in by_run.values() if item["terminal"] == "research_round_limit"
    )
    repaired_runs = sum(1 for item in by_run.values() if item["repair_count"] > 0)
    return {
        "runs_observed": runs,
        "retrieval_rounds_total": _integer_if_whole(rounds_total),
        "average_retrieval_rounds": rounds_total / runs if runs else 0.0,
        "repair_count": _integer_if_whole(repair_count),
        "repair_rate": repaired_runs / runs if runs else 0.0,
        "loop_limit_count": loop_limit_count,
        "loop_limit_hit_rate": loop_limit_count / runs if runs else 0.0,
    }


def aggregate_security_metrics(
    events: Iterable[Mapping[str, Any]],
    user_id: str,
    runtime_config_snapshot_id: str,
) -> dict[str, float | int]:
    """Aggregate deterministic safety counters from scoped event metadata."""

    counters: dict[str, float | int] = {
        "runs_observed": 0,
        "security_event_count": 0,
        "security_violation_count": 0,
        "user_leak_count": 0,
        "cross_user_evidence_count": 0,
        "forged_evidence_count": 0,
        "prompt_injection_count": 0,
        "hidden_unicode_count": 0,
        "memory_instruction_count": 0,
        "filter_override_count": 0,
    }
    runs: set[str] = set()
    for event in _scoped_events(events, user_id=user_id, snapshot_id=runtime_config_snapshot_id):
        runs.add(event["run_id"])
        event_type = event["event_type"]
        attrs = event["attributes"]
        if event_type.startswith("SECURITY_") or any(
            marker in event_type
            for marker in ("USER_LEAK", "CROSS_USER", "FORGED_EVIDENCE", "PROMPT_INJECTION", "FILTER_OVERRIDE")
        ):
            counters["security_event_count"] += 1
        if "SECURITY_VIOLATION" in event_type:
            counters["security_violation_count"] += 1
        _add_counter(counters, "user_leak_count", attrs.get("user_leak_count"))
        _add_counter(counters, "cross_user_evidence_count", attrs.get("cross_user_evidence_count"))
        _add_counter(counters, "forged_evidence_count", attrs.get("forged_evidence_count"))
        _add_counter(counters, "prompt_injection_count", attrs.get("prompt_injection_count"))
        _add_counter(counters, "hidden_unicode_count", attrs.get("hidden_unicode_count"))
        _add_counter(counters, "memory_instruction_count", attrs.get("memory_instruction_count"))
        _add_counter(counters, "filter_override_count", attrs.get("filter_override_count"))
        marker_map = {
            "USER_LEAK": "user_leak_count",
            "CROSS_USER": "cross_user_evidence_count",
            "FORGED_EVIDENCE": "forged_evidence_count",
            "PROMPT_INJECTION": "prompt_injection_count",
            "HIDDEN_UNICODE": "hidden_unicode_count",
            "MEMORY_INSTRUCTION": "memory_instruction_count",
            "FILTER_OVERRIDE": "filter_override_count",
        }
        for marker, key in marker_map.items():
            if marker in event_type and not _finite_nonnegative(attrs.get(key)):
                counters[key] += 1
    counters["runs_observed"] = len(runs)
    return counters


# Friendly aliases for callers that prefer verb-oriented names.
compute_loop_metrics = aggregate_loop_metrics
compute_security_metrics = aggregate_security_metrics
loop_metrics = aggregate_loop_metrics
security_metrics = aggregate_security_metrics


def _bounded_k(k: int, length: int) -> int:
    if isinstance(k, bool) or not isinstance(k, int) or k <= 0:
        return 0
    return min(k, length)


def _unique_prefix(values: Sequence[str], cutoff: int) -> list[tuple[str, int]]:
    seen: set[str] = set()
    result: list[tuple[str, int]] = []
    for position, value in enumerate(values, start=1):
        if position > cutoff:
            break
        if not isinstance(value, str) or value in seen:
            continue
        seen.add(value)
        result.append((value, position))
    return result


def _scoped_events(
    events: Iterable[Mapping[str, Any]], *, user_id: str, snapshot_id: str
) -> Iterable[dict[str, Any]]:
    seen_keys: set[str] = set()
    for row in events:
        if not isinstance(row, Mapping):
            continue
        if row.get("user_id") != user_id or row.get("runtime_config_snapshot_id") != snapshot_id:
            continue
        run_id = row.get("run_id")
        event_type = row.get("event_type")
        if not isinstance(run_id, str) or not run_id.strip():
            continue
        if not isinstance(event_type, str):
            continue
        event_type = event_type.upper()
        if _EVENT_TYPE.fullmatch(event_type) is None:
            continue
        event_key = row.get("event_key")
        if not isinstance(event_key, str) or not event_key.strip():
            continue
        if event_key in seen_keys:
            continue
        seen_keys.add(event_key)
        attrs = row.get("attributes")
        yield {
            "run_id": run_id,
            "event_type": event_type,
            "attributes": dict(attrs) if isinstance(attrs, Mapping) else {},
        }


def _finite_nonnegative(value: object) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value if value >= 0 else None


def _add_counter(counters: dict[str, float | int], key: str, value: object) -> None:
    number = _finite_nonnegative(value)
    if number is not None:
        counters[key] += number


def _integer_if_whole(number: float) -> int | float:
    return int(number) if number.is_integer() else number
