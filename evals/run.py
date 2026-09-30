"""Offline evaluation runner with resumable, atomic JSONL persistence."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import inspect
import json
import math
import os
import re
from collections.abc import Awaitable, Mapping, Sequence
from pathlib import Path
from typing import Any, Literal, Protocol, cast

from pydantic import BaseModel, ConfigDict, Field, field_validator

from evals.metrics import (
    aggregate_loop_metrics,
    aggregate_security_metrics,
    mrr,
    ndcg_at_k,
    recall_at_k,
)
from evals.models import EvaluationCase, SafeIdentifier, load_jsonl_dataset
from evals.ragas_adapter import RagasAdapter, RagasEvaluation, normalize_ragas_result
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


EvaluationMode = Literal["contract", "api"]


class QueryClient(Protocol):
    async def query(self, case: EvaluationCase) -> Mapping[str, object]: ...


class RetrievalRoundResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    ordinal: int = Field(ge=0)
    target_ids: tuple[SafeIdentifier, ...] = Field(default=(), max_length=128)
    stages: dict[Literal["dense", "bm25", "rrf", "rerank"], tuple[SafeIdentifier, ...]]
    parent_ids: tuple[SafeIdentifier, ...] = Field(max_length=128)


class EvalCaseResult(BaseModel):
    """Strict public result row; no prompt, reasoning, or raw tool payloads."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    case_id: str
    case_fingerprint: str | None = Field(default=None, pattern=r"^[a-f0-9]{64}$")
    judge_fingerprint: str | None = Field(default=None, max_length=128)
    judge_metadata: dict[str, str | int] = Field(default_factory=dict)
    retrieved_contexts: tuple[str, ...] = Field(default=(), max_length=100)
    ranking_scope: Literal["contract", "single-retrieval", "per-retrieval"] = "contract"
    retrieval_rounds: tuple[RetrievalRoundResult, ...] = Field(default=(), max_length=128)
    runtime_config_snapshot_id: str
    answer: str
    answer_origin: Literal["model", "terminal_status"] = "model"
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
    evaluation_mode: EvaluationMode = "contract"
    client_provenance: str = "contract_test"

    @field_validator("judge_metadata", mode="before")
    @classmethod
    def _strict_judge_metadata(cls, value: object) -> object:
        if not isinstance(value, Mapping):
            raise ValueError("judge_metadata must be an object")
        return normalize_ragas_result(RagasEvaluation("unavailable", {}, metadata=dict(value))).metadata

    @field_validator("retrieved_contexts", mode="before")
    @classmethod
    def _strict_contexts(cls, value: object) -> object:
        if not isinstance(value, (list, tuple)) or any(
            not isinstance(text, str) or not text.strip() or len(text) > 100_000 for text in value
        ):
            raise ValueError("retrieved_contexts must contain bounded source strings")
        return tuple(value)

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

    @field_validator("evaluation_mode", mode="before")
    @classmethod
    def _strict_evaluation_mode(cls, value: object) -> object:
        if value not in {"contract", "api"}:
            raise ValueError("evaluation_mode must be contract or api")
        return value

    @field_validator("client_provenance", mode="before")
    @classmethod
    def _strict_client_provenance(cls, value: object) -> object:
        if type(value) is not str or _IDENTIFIER.fullmatch(value.strip()) is None:
            raise ValueError("client_provenance must be a safe identifier")
        return value.strip()

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
        evaluation_mode: EvaluationMode = "contract",
        client_provenance: str = "contract_test",
    ) -> None:
        if evaluation_mode not in {"contract", "api"}:
            raise ValueError("evaluation_mode must be contract or api")
        if _IDENTIFIER.fullmatch(client_provenance.strip()) is None:
            raise ValueError("client_provenance must be a safe identifier")
        if client_provenance == "fixture":
            raise ValueError("fixture provenance is not supported")
        self.client = client
        self.output_dir = Path(output_dir)
        self.ragas_adapter = ragas_adapter if ragas_adapter is not None else RagasAdapter()
        self.evaluation_mode = evaluation_mode
        self.client_provenance = client_provenance.strip()
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
            if self._has_persisted_evaluation_state():
                raise ValueError(
                    "empty dataset would overwrite existing evaluation state"
                )
            summary = build_summary(
                [],
                baseline_ids=baseline_ids,
                recovery_drill_passed=recovery_drill_passed,
                backup_restore_passed=backup_restore_passed,
                evaluation_mode=self.evaluation_mode,
                client_provenance=self.client_provenance,
            )
            write_summary(self.summary_path, summary)
            return summary
        snapshots = {case.runtime_config_snapshot_id for case in cases}
        if len(snapshots) > 1 and baseline_ids is None:
            raise MixedSnapshotError("dataset contains mixed runtime snapshots")

        persisted_rows = self._load_completed_rows()
        persisted_failures = self._load_persisted_failures()
        selected_case_ids = {case.case_id for case in cases}
        if (set(persisted_rows) | set(persisted_failures)) - selected_case_ids:
            raise ValueError(
                "subset would discard scored or failed cases; use a separate output directory"
            )
        existing = persisted_rows if resume else {}
        rows: dict[str, EvalCaseResult] = {}
        failures = dict(persisted_failures)
        resumed_case_ids: set[str] = set()
        for case in cases:
            cached = existing.get(case.case_id)
            if (
                cached is not None
                and cached.case_fingerprint == _case_fingerprint(case)
                and cached.runtime_config_snapshot_id == case.runtime_config_snapshot_id
                and cached.evaluation_mode == self.evaluation_mode
                and cached.client_provenance == self.client_provenance
            ):
                if (cached.judge_fingerprint == getattr(self.ragas_adapter, "fingerprint", None)
                        and cached.ragas_metrics.get("status") != "failed"):
                    rows[case.case_id] = cached
                    failures.pop(case.case_id, None)
                    resumed_case_ids.add(case.case_id)
                else:
                    # A judge retry/config change must not repeat a paid RAG query.
                    metrics, metadata = await self._ragas_metrics(
                        case=case, answer=cached.answer, contexts=cached.retrieved_contexts,
                    )
                    rows[case.case_id] = cached.model_copy(update={
                        "ragas_metrics": metrics, "judge_metadata": metadata,
                        "judge_fingerprint": getattr(self.ragas_adapter, "fingerprint", None),
                    })
                    self._write_rows(_ordered_rows(cases, {**existing, **rows}))
                continue
            try:
                response = await self._query(case)
                result = await self._build_result(case, response)
            except Exception as error:
                if self.evaluation_mode != "api":
                    raise
                failures[case.case_id] = {
                    "case_id": case.case_id,
                    "case_fingerprint": _case_fingerprint(case),
                    "status": "timeout" if isinstance(error, TimeoutError) else "evaluation_failed",
                    "error_type": type(error).__name__,
                }
                self._write_failures(cases, failures)
                continue
            failures.pop(case.case_id, None)
            rows[case.case_id] = result
            # Keep later cached rows durable while walking an expanded dataset.
            # A process interruption must not lose their already-paid judge scores.
            self._write_rows(_ordered_rows(cases, {**existing, **rows}))

        # Persist the complete selected dataset even when all rows were resumed;
        # this removes stale/corrupt/duplicate rows from prior interrupted runs.
        ordered = _ordered_rows(cases, rows)
        receipts = []
        if self.evaluation_mode == "api":
            from evals.verification import verify_runtime_rows
            completed = {row.case_id for row in ordered}
            receipts = await verify_runtime_rows([case for case in cases if case.case_id in completed],
                                                 ordered, self.client, self.output_dir)
        self._write_rows(ordered)
        summary = build_summary(
            ordered,
            baseline_ids=baseline_ids,
            recovery_drill_passed=recovery_drill_passed,
            backup_restore_passed=backup_restore_passed,
            evaluation_mode=self.evaluation_mode,
            client_provenance=self.client_provenance,
        )
        if self.evaluation_mode == "api":
            summary["real_query_count"] = len(receipts)
            summary["runtime_verification"] = {"method": "scoped-mysql-checkpoint-v1", "receipts": receipts}
        summary["requested_cases"] = len(cases)
        ordered_failures = [
            failures[case.case_id]
            for case in cases
            if case.case_id in failures
        ]
        summary["query_failure_count"] = len(ordered_failures)
        summary["failures"] = ordered_failures
        if ordered_failures:
            summary["ragas_status"] = "partial" if ordered else "unavailable"
            summary["ragas"]["status"] = summary["ragas_status"]
            summary["ragas"]["query_unavailable_cases"] = len(ordered_failures)
        self._write_failures(cases, failures)
        write_summary(self.summary_path, summary)
        return {
            **summary,
            "completed_cases": len(ordered),
            "resumed_cases": len(resumed_case_ids),
            "quarantined_rows": self.quarantined_rows,
        }

    def _has_persisted_evaluation_state(self) -> bool:
        return any(
            path.exists()
            for path in (
                self.results_path,
                self.summary_path,
                self.output_dir / "failures.jsonl",
            )
        )

    def _load_persisted_failures(self) -> dict[str, dict[str, str]]:
        path = self.output_dir / "failures.jsonl"
        if not path.exists():
            return {}
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except OSError as error:
            raise ValueError("cannot read persisted evaluation failures") from error
        result: dict[str, dict[str, str]] = {}
        for line in lines:
            if not line.strip():
                continue
            try:
                value = json.loads(line, parse_constant=_reject_nonfinite)
                if not isinstance(value, Mapping):
                    raise ValueError("failure row must be an object")
                case_id = value.get("case_id")
                fingerprint = value.get("case_fingerprint")
                status = value.get("status")
                error_type = value.get("error_type")
                if (
                    type(case_id) is not str
                    or _IDENTIFIER.fullmatch(case_id) is None
                    or type(fingerprint) is not str
                    or re.fullmatch(r"[a-f0-9]{64}", fingerprint) is None
                    or status not in {"timeout", "evaluation_failed"}
                    or type(error_type) is not str
                    or _IDENTIFIER.fullmatch(error_type) is None
                    or case_id in result
                ):
                    raise ValueError("invalid persisted failure row")
            except (TypeError, ValueError, json.JSONDecodeError) as error:
                raise ValueError("invalid persisted evaluation failures") from error
            result[case_id] = {
                "case_id": case_id,
                "case_fingerprint": fingerprint,
                "status": status,
                "error_type": error_type,
            }
        return result

    def _write_failures(
        self,
        cases: Sequence[EvaluationCase],
        failures: Mapping[str, Mapping[str, str]],
    ) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        content = "".join(
            json.dumps(failures[case.case_id], sort_keys=True) + "\n"
            for case in cases
            if case.case_id in failures
        )
        atomic_write_text(self.output_dir / "failures.jsonl", content)

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
                raw_row = dict(value)
                if "evaluation_mode" not in raw_row or "client_provenance" not in raw_row:
                    raise ValueError("result row is missing evaluation provenance")
                row = EvalCaseResult.model_validate(raw_row)
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
        ragas, judge_metadata = await self._ragas_metrics(
            case=case, answer=answer, contexts=contexts,
        )
        return EvalCaseResult(
            case_id=case.case_id,
            case_fingerprint=_case_fingerprint(case),
            judge_fingerprint=getattr(self.ragas_adapter, "fingerprint", None),
            judge_metadata=judge_metadata,
            retrieved_contexts=tuple(contexts),
            ranking_scope=response.get("ranking_scope", "contract"),
            retrieval_rounds=tuple({key: row[key] for key in ("ordinal", "target_ids", "stages", "parent_ids")}
                                   for row in response.get("retrieval_rounds", [])),
            runtime_config_snapshot_id=case.runtime_config_snapshot_id,
            answer=answer,
            answer_origin=("terminal_status" if isinstance(response.get("answer"), Mapping)
                           and response["answer"].get("status") and not response["answer"].get("segments") else "model"),
            evidence_parent_ids=tuple(evidence_parent_ids),
            route=route,
            events_ref=events_ref,
            deterministic_metrics=deterministic,
            ragas_metrics=ragas,
            citation_coverage=_response_citation_coverage(response),
            audited=_response_audited(response),
            evaluation_mode=self.evaluation_mode,
            client_provenance=self.client_provenance,
        )

    async def _ragas_metrics(
        self, *, case: EvaluationCase, answer: str, contexts: Sequence[str]
    ) -> tuple[dict[str, float | str], dict[str, str | int]]:
        adapter = self.ragas_adapter
        if not case.answerable and getattr(adapter, "supports_refusal", False) is not True:
            return {"status": "unavailable", "reason": "unanswerable case requires separate refusal evaluation"}, {}
        value = adapter.evaluate(
            question=case.question,
            answer=answer,
            contexts=tuple(contexts),
            reference_answer=case.reference_answer,
            answerable=case.answerable,
        )
        if inspect.isawaitable(value):
            value = await value
        outcome = normalize_ragas_result(value)
        if outcome.status in {"unavailable", "failed"}:
            result: dict[str, float | str] = {"status": outcome.status}
            if outcome.reason:
                result["reason"] = outcome.reason
            return result, outcome.metadata
        if outcome.status != "available":
            raise ValueError("Ragas adapter returned an unknown status")
        result = {"status": "available"}
        result.update(outcome.metrics)
        return result, outcome.metadata

    def _write_rows(self, rows: Sequence[EvalCaseResult]) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        content = "".join(
            json.dumps(row.model_dump(mode="json"), ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
            for row in rows
        )
        atomic_write_text(Path(os.fspath(self.results_path)), content)


def _case_fingerprint(case: EvaluationCase) -> str:
    """Bind resume to all validated case inputs, including gold and user scope."""
    encoded = json.dumps(
        case.model_dump(mode="json"), ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run an Agentic RAG evaluation dataset")
    parser.add_argument("--dataset", required=True, help="path to a baseline/ingestion/security JSONL dataset")
    parser.add_argument("--output", required=True, help="directory for results.jsonl and summary.json")
    parser.add_argument("--limit", type=int, default=None, help="evaluate only the first N cases")
    parser.add_argument(
        "--mode",
        choices=("api",),
        default="api",
        help="business evaluation requires the deployed HTTP Query API",
    )
    parser.add_argument("--base-url", default=None, help="public API base URL for --mode api")
    parser.add_argument("--stack-allocation", type=Path, help="isolated runtime allocation ledger for trusted collection")
    args = parser.parse_args(argv)
    if not args.base_url:
        parser.error("--base-url is required for API evaluation")
    if args.stack_allocation is None:
        parser.error("--stack-allocation is required for scoped runtime collection")
    cases = load_jsonl_dataset(args.dataset)
    async def evaluate() -> dict[str, object]:
        from agentic_rag.config import Settings
        from evals.clients import HttpQueryClient
        from evals.judge import JudgeConfig, RagasJudge
        from evals.collector import RuntimeCollector
        from evals.real_stack import isolated_environment
        from sqlalchemy.ext.asyncio import create_async_engine

        # Fail missing dependencies/configuration before submitting any query.
        settings = Settings()
        allocation = json.loads(args.stack_allocation.read_text())
        environment = isolated_environment(settings, allocation, root=Path.cwd())
        if args.base_url.rstrip("/") != allocation["api_url"].rstrip("/"):
            raise ValueError("API endpoint differs from isolated allocation")
        if any(case.user_id != allocation["user_id"] for case in cases):
            raise ValueError("dataset user differs from isolated allocation")
        judge = RagasJudge(JudgeConfig.from_settings(settings))
        engine = create_async_engine(environment["AGENTIC_RAG_MYSQL_DSN"])
        collector = RuntimeCollector(engine, Path(environment["AGENTIC_RAG_QUERY_CHECKPOINT_PATH"]))
        client = HttpQueryClient(args.base_url, collector=collector, timeout_seconds=360,
                                 ledger_dir=Path(args.output) / "queries")
        try:
            return await EvalRunner(
                client,
                output_dir=args.output,
                ragas_adapter=RagasAdapter(judge),
                evaluation_mode="api",
                client_provenance=HttpQueryClient.provenance,
            ).run(
                cases,
                limit=args.limit,
            )
        finally:
            await client.aclose()
            await judge.aclose()
            await engine.dispose()

    summary = asyncio.run(evaluate())
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
        if not value and answer_mapping.get("status"):
            # The existing console renders these fixed notices for safe public
            # terminal states. This is observed UI output, never reference text.
            value = {
                "cannot_answer": "现有证据不足以安全回答，系统未展示草稿。",
                "refuse": "该请求已被安全拒绝，系统未继续生成回答。",
                "clarify": "需要补充问题或上下文后才能继续检索。",
                "research_action_invalid": "研究动作不符合安全约束，系统未继续执行。",
                "audit_failed": "回答未通过审计，系统不会展示未审计结果。",
                "research_round_limit": "已达到全局研究轮次上限，系统停止继续研究。",
            }.get(answer_mapping["status"])
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
    if response.get("ranking_scope") == "per-retrieval":
        return []  # Deliberately inapplicable, never substitute final citations.
    if "ranked_parent_ids" in response:
        return _response_parent_ids(response, "ranked_parent_ids")
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
    metrics: dict[str, float | int] = {"leakage": _response_leakage(events, case)}
    if case.answerable and response.get("ranking_scope") != "per-retrieval":
        metrics.update({
            "parent_recall_at_6": recall_at_k(ranked_parent_ids, relevant, 6),
            "mrr": mrr(ranked_parent_ids, relevant, 6),
            "ndcg_at_10": ndcg_at_k(ranked_parent_ids, relevant, 10),
        })
    if case.answerable and "final_context_parent_ids" in response:
        final_ids = set(_response_parent_ids(response, "final_context_parent_ids"))
        metrics["final_evidence_coverage"] = len(final_ids & relevant) / len(relevant)
    # Loop and security helpers are the sole source for event-derived metrics;
    # the runner does not reconstruct graph or retrieval decisions.
    if "retrieval_rounds" in response:
        metrics["retrieval_rounds_total"] = len(response["retrieval_rounds"])
    elif events:
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
            if metric not in {"available", "unavailable", "failed"}:
                raise ValueError("Ragas status must be available, unavailable, or failed")
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
    "EvaluationMode",
    "EvalRunner",
    "MixedSnapshotError",
    "QueryClient",
    "SnapshotMismatchError",
    "main",
]


if __name__ == "__main__":  # pragma: no cover - exercised by CLI smoke tests
    raise SystemExit(main())
