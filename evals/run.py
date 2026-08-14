"""Offline evaluation runner with resumable, atomic JSONL persistence."""

from __future__ import annotations

import argparse
import asyncio
import inspect
import json
import math
import os
import re
from collections.abc import Awaitable, Mapping, Sequence
from pathlib import Path
from typing import Any, Protocol, cast

from pydantic import BaseModel, ConfigDict, Field, field_validator

from evals.metrics import (
    aggregate_loop_metrics,
    aggregate_security_metrics,
    mrr,
    ndcg_at_k,
    recall_at_k,
)
from evals.models import EvaluationCase, load_jsonl_dataset
from evals.ragas_adapter import RagasAdapter, normalize_ragas_result
from evals.report import MixedSnapshotError, atomic_write_text, build_summary, write_summary


_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$")
_URI = re.compile(r"^[^\x00-\x1f\x7f]{1,1024}$")
_EVENT_REF_SCHEME = re.compile(r"^(?:artifact|fixture|inline)://")
_EVENT_REF_SEGMENT = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$")
_EVENT_REF_UNSAFE = re.compile(
    r"(?:prompt|tool|authorization|secret|password|api[_-]?key|chain[-_ ]of[- ]thought|"
    r"hidden[-_ ]reasoning)",
    re.IGNORECASE,
)
_SECRET_TEXT = re.compile(
    r"(?:api[_ -]?key|authorization:\s*bearer|password\s*=|sk-[A-Za-z0-9_-]{8,}|"
    r"chain[- ]of[- ]thought|hidden reasoning|provider[_ -]?secret)",
    re.IGNORECASE,
)


class SnapshotMismatchError(ValueError):
    """Raised when a query response is not from the case's fixed snapshot."""


class QueryClient(Protocol):
    async def query(self, case: EvaluationCase) -> Mapping[str, object]: ...


class EvalCaseResult(BaseModel):
    """Strict public result row; no prompt, reasoning, or raw tool payloads."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    case_id: str
    runtime_config_snapshot_id: str
    answer: str
    evidence_parent_ids: tuple[str, ...]
    route: str
    events_ref: str
    deterministic_metrics: dict[str, float | int]
    # ``status=unavailable`` is the only non-numeric value permitted.  It keeps
    # missing Ragas explicit without inventing a score.
    ragas_metrics: dict[str, float | str]
    # These are required durable fields.  A pre-Task5/partial row must not be
    # treated as a resumable completed answer and must be recomputed.
    citation_coverage: float = Field(..., ge=0.0, le=1.0)
    audited: bool

    @field_validator("case_id", "runtime_config_snapshot_id", "route", mode="before")
    @classmethod
    def _strict_identifier(cls, value: object) -> object:
        if type(value) is not str or _IDENTIFIER.fullmatch(value.strip()) is None:
            raise ValueError("identifier must be a non-empty safe string")
        return value.strip()

    @field_validator("citation_coverage", mode="before")
    @classmethod
    def _strict_citation_coverage(cls, value: object) -> object:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError("citation_coverage must be numeric")
        return value

    @field_validator("audited", mode="before")
    @classmethod
    def _strict_audited(cls, value: object) -> object:
        if type(value) is not bool:
            raise ValueError("audited must be boolean")
        return value

    @field_validator("answer", mode="before")
    @classmethod
    def _strict_answer(cls, value: object) -> object:
        if type(value) is not str or not value.strip() or len(value) > 20_000:
            raise ValueError("answer must be a bounded non-empty string")
        if _SECRET_TEXT.search(value):
            raise ValueError("answer contains a provider secret or hidden reasoning")
        return value.strip()

    @field_validator("events_ref", mode="before")
    @classmethod
    def _strict_events_ref(cls, value: object) -> object:
        return _safe_events_ref(value)

    @field_validator("evidence_parent_ids", mode="before")
    @classmethod
    def _strict_parent_ids(cls, value: object) -> object:
        if not isinstance(value, (list, tuple)):
            raise ValueError("evidence_parent_ids must be a sequence")
        values = list(value)
        if any(type(item) is not str or _IDENTIFIER.fullmatch(item.strip()) is None for item in values):
            raise ValueError("evidence_parent_ids must contain safe strings")
        if len(set(values)) != len(values):
            raise ValueError("evidence_parent_ids must be unique")
        return tuple(item.strip() for item in values)

    @field_validator("deterministic_metrics", mode="before")
    @classmethod
    def _strict_deterministic_metrics(cls, value: object) -> object:
        return _validate_metric_map(value, allow_status=False)

    @field_validator("ragas_metrics", mode="before")
    @classmethod
    def _strict_ragas_metrics(cls, value: object) -> object:
        return _validate_metric_map(value, allow_status=True)


class EvalRunner:
    """Run fixed evaluation cases through an injected query client.

    The runner never constructs a graph or reaches a provider itself.  This
    keeps tests offline while allowing deployment code to inject the real API or
    QueryGraph adapter.
    """

    def __init__(
        self,
        client: QueryClient | Any,
        *,
        output_dir: str | Path,
        ragas_adapter: RagasAdapter | Any | None = None,
    ) -> None:
        self.client = client
        self.output_dir = Path(output_dir)
        self.ragas_adapter = ragas_adapter if ragas_adapter is not None else RagasAdapter()
        self.results_path = self.output_dir / "results.jsonl"
        self.summary_path = self.output_dir / "summary.json"
        self.quarantined_rows = 0

    async def run(
        self,
        dataset: Sequence[EvaluationCase | Mapping[str, object]],
        *,
        limit: int | None = None,
        resume: bool = True,
        baseline_ids: Mapping[str, str] | None = None,
        recovery_drill_passed: bool = False,
        backup_restore_passed: bool = False,
    ) -> dict[str, object]:
        cases = _normalize_cases(dataset)
        if limit is not None:
            if isinstance(limit, bool) or limit < 0:
                raise ValueError("limit must be a non-negative integer")
            cases = cases[:limit]
        if not cases:
            summary = build_summary(
                [],
                baseline_ids=baseline_ids,
                recovery_drill_passed=recovery_drill_passed,
                backup_restore_passed=backup_restore_passed,
            )
            write_summary(self.summary_path, summary)
            return summary
        snapshots = {case.runtime_config_snapshot_id for case in cases}
        if len(snapshots) > 1 and baseline_ids is None:
            raise MixedSnapshotError("dataset contains mixed runtime snapshots")

        existing = self._load_completed_rows() if resume else {}
        rows: dict[str, EvalCaseResult] = {}
        resumed_case_ids: set[str] = set()
        for case in cases:
            cached = existing.get(case.case_id)
            if cached is not None and cached.runtime_config_snapshot_id == case.runtime_config_snapshot_id:
                rows[case.case_id] = cached
                resumed_case_ids.add(case.case_id)
                continue
            response = await self._query(case)
            result = await self._build_result(case, response)
            rows[case.case_id] = result
            self._write_rows(_ordered_rows(cases, rows))

        # Persist the complete selected dataset even when all rows were resumed;
        # this removes stale/corrupt/duplicate rows from prior interrupted runs.
        ordered = _ordered_rows(cases, rows)
        self._write_rows(ordered)
        summary = build_summary(
            ordered,
            baseline_ids=baseline_ids,
            recovery_drill_passed=recovery_drill_passed,
            backup_restore_passed=backup_restore_passed,
        )
        write_summary(self.summary_path, summary)
        return {
            **summary,
            "completed_cases": len(ordered),
            "resumed_cases": len(resumed_case_ids),
            "quarantined_rows": self.quarantined_rows,
        }

    def _load_completed_rows(self) -> dict[str, EvalCaseResult]:
        if not self.results_path.exists():
            return {}
        result: dict[str, EvalCaseResult] = {}
        self.quarantined_rows = 0
        try:
            lines = self.results_path.read_text(encoding="utf-8").splitlines()
        except OSError:
            return {}
        invalid_case_ids: set[str] = set()
        for line in lines:
            if not line.strip():
                self.quarantined_rows += 1
                continue
            try:
                value = json.loads(line, parse_constant=_reject_nonfinite)
                if not isinstance(value, Mapping):
                    raise ValueError("row must be an object")
                row = EvalCaseResult.model_validate(dict(value))
            except Exception:
                self.quarantined_rows += 1
                continue
            if row.case_id in invalid_case_ids:
                self.quarantined_rows += 1
                continue
            if row.case_id in result:
                # A valid duplicate is not a harmless replay: choosing one row
                # would make resume depend on file order.  Invalidate the case
                # and force one fresh Query invocation.
                result.pop(row.case_id, None)
                invalid_case_ids.add(row.case_id)
                self.quarantined_rows += 2
                continue
            result[row.case_id] = row
        return result

    async def _query(self, case: EvaluationCase) -> Mapping[str, object]:
        method = getattr(self.client, "query", None)
        if method is None:
            method = getattr(self.client, "ainvoke", None)
        if method is None:
            method = getattr(self.client, "run", None)
        if method is None and callable(self.client):
            method = self.client
        if method is None:
            raise TypeError("query client must expose query, ainvoke, run, or __call__")
        value = method(case)
        if inspect.isawaitable(value):
            value = await cast(Awaitable[object], value)
        normalized = _as_mapping(value)
        return normalized

    async def _build_result(
        self, case: EvaluationCase, response: Mapping[str, object]
    ) -> EvalCaseResult:
        snapshot_id = _response_snapshot(response)
        if snapshot_id != case.runtime_config_snapshot_id:
            raise SnapshotMismatchError(
                f"case {case.case_id} expected snapshot {case.runtime_config_snapshot_id!r}, "
                f"got {snapshot_id!r}"
            )
        answer = _response_answer(response)
        evidence_parent_ids = _response_parent_ids(response, "evidence_parent_ids")
        if not evidence_parent_ids:
            evidence_parent_ids = _response_parent_ids(response, "evidence")
        route = _response_route(response, case.expected_route)
        events_ref = _response_events_ref(response, case.case_id)
        ranked_parent_ids = _response_ranked_parent_ids(response, evidence_parent_ids)
        events = _response_events(response)
        deterministic = _deterministic_metrics(
            case,
            ranked_parent_ids=ranked_parent_ids,
            response=response,
            events=events,
        )
        contexts = _response_contexts(response)
        ragas = await self._ragas_metrics(
            answer=answer,
            contexts=contexts,
            reference_answer=case.reference_answer,
        )
        return EvalCaseResult(
            case_id=case.case_id,
            runtime_config_snapshot_id=case.runtime_config_snapshot_id,
            answer=answer,
            evidence_parent_ids=tuple(evidence_parent_ids),
            route=route,
            events_ref=events_ref,
            deterministic_metrics=deterministic,
            ragas_metrics=ragas,
            citation_coverage=_response_citation_coverage(response),
            audited=_response_audited(response),
        )

    async def _ragas_metrics(
        self, *, answer: str, contexts: Sequence[str], reference_answer: str
    ) -> dict[str, float | str]:
        adapter = self.ragas_adapter
        value = adapter.evaluate(
            answer=answer,
            contexts=tuple(contexts),
            reference_answer=reference_answer,
        )
        if inspect.isawaitable(value):
            value = await value
        outcome = normalize_ragas_result(value)
        if outcome.status == "unavailable":
            result: dict[str, float | str] = {"status": "unavailable"}
            if outcome.reason:
                result["reason"] = outcome.reason
            return result
        if outcome.status != "available":
            raise ValueError("Ragas adapter returned an unknown status")
        result = {"status": "available"}
        result.update(outcome.metrics)
        return result

    def _write_rows(self, rows: Sequence[EvalCaseResult]) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        content = "".join(
            json.dumps(row.model_dump(mode="json"), ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
            for row in rows
        )
        atomic_write_text(Path(os.fspath(self.results_path)), content)


class FixtureQueryClient:
    """Deterministic one-process client used by the offline CLI smoke path."""

    async def query(self, case: EvaluationCase) -> Mapping[str, object]:
        return {
            "runtime_config_snapshot_id": case.runtime_config_snapshot_id,
            "answer": case.reference_answer,
            "evidence_parent_ids": list(case.reference_parent_ids),
            "ranked_parent_ids": list(case.reference_parent_ids),
            "route": case.expected_route,
            "events_ref": f"fixture://events/{case.case_id}",
            "events": [],
            "contexts": [case.reference_answer],
            "citation_coverage": 1.0,
            "audited": True,
        }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run an offline Agentic RAG evaluation dataset")
    parser.add_argument("--dataset", required=True, help="path to a baseline/ingestion/security JSONL dataset")
    parser.add_argument("--output", required=True, help="directory for results.jsonl and summary.json")
    parser.add_argument("--limit", type=int, default=None, help="evaluate only the first N cases")
    args = parser.parse_args(argv)
    cases = load_jsonl_dataset(args.dataset)
    from scripts.backup_local import run_backup_restore_drill
    from scripts.run_recovery_drill import run_recovery_drill

    recovery_drill_passed = run_recovery_drill().gate_passed
    backup_restore_passed = run_backup_restore_drill()
    summary = asyncio.run(
        EvalRunner(FixtureQueryClient(), output_dir=args.output).run(
            cases,
            limit=args.limit,
            recovery_drill_passed=recovery_drill_passed,
            backup_restore_passed=backup_restore_passed,
        )
    )
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True, separators=(",", ":")))
    return 0


def _normalize_cases(
    dataset: Sequence[EvaluationCase | Mapping[str, object]],
) -> list[EvaluationCase]:
    result: list[EvaluationCase] = []
    seen: set[str] = set()
    for value in dataset:
        case = value if isinstance(value, EvaluationCase) else EvaluationCase.model_validate(dict(value))
        if case.case_id in seen:
            raise ValueError(f"duplicate case_id {case.case_id!r}")
        seen.add(case.case_id)
        result.append(case)
    return result


def _ordered_rows(cases: Sequence[EvaluationCase], rows: Mapping[str, EvalCaseResult]) -> list[EvalCaseResult]:
    return [rows[case.case_id] for case in cases if case.case_id in rows]


def _as_mapping(value: object) -> Mapping[str, object]:
    if isinstance(value, Mapping):
        return value
    dump = getattr(value, "model_dump", None)
    if callable(dump):
        dumped = dump(mode="json")
        if isinstance(dumped, Mapping):
            return cast(Mapping[str, object], dumped)
    raise TypeError("query client response must be a JSON object")


def _response_snapshot(response: Mapping[str, object]) -> str:
    value = response.get("runtime_config_snapshot_id", response.get("snapshot_id"))
    snapshot = response.get("runtime_config_snapshot")
    if value is None and isinstance(snapshot, Mapping):
        value = snapshot.get("snapshot_id")
    if type(value) is not str or _IDENTIFIER.fullmatch(value.strip()) is None:
        raise SnapshotMismatchError("query response omitted a valid runtime snapshot ID")
    return value.strip()


def _response_answer(response: Mapping[str, object]) -> str:
    value: object = response.get("answer", response.get("final_answer"))
    if isinstance(value, Mapping):
        answer_mapping = value
        value = answer_mapping.get("text", answer_mapping.get("content"))
        if value is None:
            segments = answer_mapping.get("segments")
            if isinstance(segments, Sequence) and not isinstance(segments, (str, bytes)):
                texts = [segment.get("text") for segment in segments if isinstance(segment, Mapping)]
                value = "\n".join(text for text in texts if isinstance(text, str))
    if type(value) is not str or not value.strip():
        raise ValueError("query response omitted a non-empty answer")
    return value.strip()


def _response_parent_ids(response: Mapping[str, object], field: str) -> list[str]:
    value = response.get(field)
    if value is None:
        return []
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise ValueError(f"query response field {field!r} must be a sequence")
    result: list[str] = []
    for item in value:
        parent = item.get("parent_id") if isinstance(item, Mapping) else item
        if type(parent) is not str or _IDENTIFIER.fullmatch(parent.strip()) is None:
            raise ValueError(f"query response field {field!r} contains an invalid parent ID")
        if parent.strip() not in result:
            result.append(parent.strip())
    return result


def _response_ranked_parent_ids(response: Mapping[str, object], fallback: Sequence[str]) -> list[str]:
    for field in ("ranked_parent_ids", "retrieved_parent_ids", "parent_ids"):
        values = _response_parent_ids(response, field)
        if values:
            return values
    return list(fallback)


def _response_route(response: Mapping[str, object], fallback: str) -> str:
    value = response.get("route", fallback)
    if isinstance(value, Mapping):
        value = value.get("route", value.get("name"))
    if type(value) is not str or _IDENTIFIER.fullmatch(value.strip()) is None:
        raise ValueError("query response omitted a valid route")
    return value.strip()


def _response_events_ref(response: Mapping[str, object], case_id: str) -> str:
    value = response.get("events_ref", response.get("events_reference"))
    if value is None:
        value = f"inline://events/{case_id}"
    return _safe_events_ref(value)


def _response_events(response: Mapping[str, object]) -> list[Mapping[str, object]]:
    value = response.get("events", [])
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        return []
    return [cast(Mapping[str, object], item) for item in value if isinstance(item, Mapping)]


def _response_citation_coverage(response: Mapping[str, object]) -> float:
    value = response.get("citation_coverage")
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("query response omitted numeric citation_coverage")
    coverage = float(value)
    if not math.isfinite(coverage) or coverage < 0.0 or coverage > 1.0:
        raise ValueError("query response citation_coverage must be within [0, 1]")
    return coverage


def _response_audited(response: Mapping[str, object]) -> bool:
    value = response.get("audited")
    if type(value) is not bool:
        raise ValueError("query response omitted boolean audited")
    return value


def _response_contexts(response: Mapping[str, object]) -> list[str]:
    value = response.get("contexts", response.get("evidence", []))
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        return []
    contexts: list[str] = []
    for item in value:
        text = item.get("content") if isinstance(item, Mapping) else item
        if type(text) is str and text.strip():
            contexts.append(text.strip())
    return contexts


def _deterministic_metrics(
    case: EvaluationCase,
    *,
    ranked_parent_ids: Sequence[str],
    response: Mapping[str, object],
    events: Sequence[Mapping[str, object]],
) -> dict[str, float | int]:
    relevant = set(case.reference_parent_ids)
    metrics: dict[str, float | int] = {
        "parent_recall_at_6": recall_at_k(ranked_parent_ids, relevant, 6),
        "mrr": mrr(ranked_parent_ids, relevant, 6),
        "ndcg_at_10": ndcg_at_k(ranked_parent_ids, relevant, 10),
        "leakage": _response_leakage(events, case),
    }
    # Loop and security helpers are the sole source for event-derived metrics;
    # the runner does not reconstruct graph or retrieval decisions.
    if events:
        loop = aggregate_loop_metrics(events, case.user_id, case.runtime_config_snapshot_id)
        metrics.update({
            key: value
            for key, value in loop.items()
            if key not in {"runs_observed"}
        })
    return metrics


def _response_leakage(events: Sequence[Mapping[str, object]], case: EvaluationCase) -> int:
    """Read leakage only from Task 2's scoped, deduplicated event reducer."""

    security = aggregate_security_metrics(events, case.user_id, case.runtime_config_snapshot_id)
    leak = security.get("user_leak_count", 0)
    return int(leak) if isinstance(leak, int) else 0


def _safe_events_ref(value: object) -> str:
    if type(value) is not str or _URI.fullmatch(value.strip()) is None:
        raise ValueError("events_ref must be a bounded URI")
    reference = value.strip()
    if _SECRET_TEXT.search(reference) or _EVENT_REF_UNSAFE.search(reference) or not _EVENT_REF_SCHEME.match(reference):
        raise ValueError("events_ref must use an approved artifact, fixture or inline URI")
    remainder = reference.split("://", 1)[1]
    segments = remainder.split("/")
    if not segments or any(
        segment in {"", ".", ".."} or _EVENT_REF_SEGMENT.fullmatch(segment) is None
        for segment in segments
    ):
        raise ValueError("events_ref contains an unsafe URI segment")
    return reference


def _validate_metric_map(value: object, *, allow_status: bool) -> dict[str, float | int | str]:
    if not isinstance(value, Mapping):
        raise ValueError("metrics must be an object")
    result: dict[str, float | int | str] = {}
    for name, metric in value.items():
        if type(name) is not str or _IDENTIFIER.fullmatch(name.strip()) is None:
            raise ValueError("metric names must be safe identifiers")
        normalized_name = name.strip()
        if allow_status and normalized_name == "status":
            if metric not in {"available", "unavailable"}:
                raise ValueError("Ragas status must be available or unavailable")
            result[normalized_name] = cast(str, metric)
            continue
        if allow_status and normalized_name == "reason":
            if type(metric) is not str or not metric.strip() or len(metric) > 1_000:
                raise ValueError("Ragas reason must be a bounded string")
            result[normalized_name] = metric.strip()
            continue
        if isinstance(metric, bool) or not isinstance(metric, (int, float)):
            raise ValueError(f"metric {normalized_name!r} must be numeric")
        numeric = float(metric)
        if not math.isfinite(numeric):
            raise ValueError(f"metric {normalized_name!r} must be finite")
        result[normalized_name] = metric
    if not allow_status and "status" in result:
        raise ValueError("deterministic metrics may not contain status")
    return result


def _reject_nonfinite(value: str) -> Any:
    raise ValueError(f"non-finite JSON constant {value}")


__all__ = [
    "EvalCaseResult",
    "EvalRunner",
    "FixtureQueryClient",
    "MixedSnapshotError",
    "QueryClient",
    "SnapshotMismatchError",
    "main",
]


if __name__ == "__main__":  # pragma: no cover - exercised by CLI smoke tests
    raise SystemExit(main())
