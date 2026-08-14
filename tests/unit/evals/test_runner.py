"""Offline runner contracts.

The runner tests deliberately inject an in-memory query client.  A real model,
database, retrieval store, or network must never be needed to exercise resume
or report persistence.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from evals.models import EvaluationCase
from evals.ragas_adapter import RagasUnavailable
from evals.report import MixedSnapshotError, build_summary
from evals.run import EvalCaseResult, EvalRunner, SnapshotMismatchError


def _case(case_id: str = "case-1", snapshot: str = "snapshot-v1") -> EvaluationCase:
    return EvaluationCase.model_validate(
        {
            "case_id": case_id,
            "user_id": "eval_user",
            "question": "Which parent is relevant?",
            "reference_answer": "Parent one is relevant.",
            "reference_parent_ids": ["parent-1"],
            "expected_route": "fast_rag",
            "tags": ["single-hop"],
            "runtime_config_snapshot_id": snapshot,
        }
    )


class FakeQueryClient:
    def __init__(self, *, snapshot: str | None = None) -> None:
        self.call_count = 0
        self.snapshot = snapshot

    async def query(self, case: EvaluationCase) -> dict[str, object]:
        self.call_count += 1
        return {
            "answer": case.reference_answer,
            "evidence_parent_ids": ["parent-1"],
            "ranked_parent_ids": ["parent-1", "parent-2"],
            "route": case.expected_route,
            "events_ref": f"artifact://events/{case.case_id}",
            "runtime_config_snapshot_id": self.snapshot or case.runtime_config_snapshot_id,
            "events": [],
            "citation_coverage": 1.0,
            "audited": True,
        }


@pytest.mark.asyncio
async def test_eval_runner_resumes_completed_cases_without_reexecution(tmp_path: Path) -> None:
    client = FakeQueryClient()
    runner = EvalRunner(client, output_dir=tmp_path)

    first = await runner.run([_case()])
    second = await runner.run([_case()])

    assert client.call_count == 1
    assert first["completed_cases"] == second["completed_cases"] == 1
    assert len((tmp_path / "results.jsonl").read_text(encoding="utf-8").splitlines()) == 1


@pytest.mark.asyncio
async def test_corrupt_result_row_is_ignored_and_recomputed(tmp_path: Path) -> None:
    results = tmp_path / "results.jsonl"
    results.write_text(
        '{"case_id":"case-1","runtime_config_snapshot_id":"snapshot-v1",\n',
        encoding="utf-8",
    )
    client = FakeQueryClient()

    await EvalRunner(client, output_dir=tmp_path).run([_case()])

    assert client.call_count == 1
    rows = [json.loads(line) for line in results.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 1
    assert rows[0]["case_id"] == "case-1"


@pytest.mark.asyncio
async def test_duplicate_valid_result_rows_are_quarantined_and_recomputed(tmp_path: Path) -> None:
    first_client = FakeQueryClient()
    await EvalRunner(first_client, output_dir=tmp_path).run([_case()])
    row = (tmp_path / "results.jsonl").read_text(encoding="utf-8")
    (tmp_path / "results.jsonl").write_text(row + row, encoding="utf-8")

    second_client = FakeQueryClient()
    summary = await EvalRunner(second_client, output_dir=tmp_path).run([_case()])

    assert second_client.call_count == 1
    assert summary["quarantined_rows"] == 2
    assert len((tmp_path / "results.jsonl").read_text(encoding="utf-8").splitlines()) == 1


@pytest.mark.asyncio
async def test_runner_rejects_client_response_from_different_snapshot(tmp_path: Path) -> None:
    with pytest.raises(SnapshotMismatchError):
        await EvalRunner(
            FakeQueryClient(snapshot="snapshot-other"), output_dir=tmp_path
        ).run([_case()])


@pytest.mark.asyncio
async def test_result_and_summary_are_replaced_atomically(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    replacements: list[tuple[str, str]] = []
    original_replace = __import__("os").replace

    def record_replace(source: str, target: str) -> None:
        replacements.append((source, target))
        original_replace(source, target)

    monkeypatch.setattr("evals.run.os.replace", record_replace)
    await EvalRunner(FakeQueryClient(), output_dir=tmp_path).run([_case()])

    assert {Path(target).name for _source, target in replacements} >= {"results.jsonl", "summary.json"}
    assert not list(tmp_path.glob("*.tmp"))


def test_report_rejects_mixed_runtime_snapshots() -> None:
    one = EvalCaseResult(
        case_id="case-1",
        runtime_config_snapshot_id="snapshot-a",
        answer="a",
        evidence_parent_ids=("parent-1",),
        route="fast_rag",
        events_ref="artifact://one",
        deterministic_metrics={"parent_recall_at_6": 1.0},
        ragas_metrics={"status": "unavailable"},
    )
    two = one.model_copy(update={"case_id": "case-2", "runtime_config_snapshot_id": "snapshot-b"})

    with pytest.raises(MixedSnapshotError):
        build_summary([one, two])

    with pytest.raises(ValueError, match="named baseline"):
        build_summary([one, two], baseline_ids=["snapshot-a", "snapshot-b"])


def test_summary_emits_only_verified_acceptance_gate_values() -> None:
    row = EvalCaseResult(
        case_id="case-acceptance",
        runtime_config_snapshot_id="snapshot-v1",
        answer="audited answer",
        evidence_parent_ids=("parent-1",),
        route="fast_rag",
        events_ref="fixture://events/case-acceptance",
        deterministic_metrics={"leakage": 0},
        ragas_metrics={"status": "unavailable"},
        citation_coverage=1.0,
        audited=True,
    )

    summary = build_summary(
        [row], recovery_drill_passed=True, backup_restore_passed=True
    )

    assert summary["user_leak_count"] == 0
    assert summary["citation_coverage"] == 1.0
    assert summary["unaudited_answer_count"] == 0
    assert summary["recovery_drill_passed"] is True
    assert summary["backup_restore_passed"] is True


def test_summary_citation_gate_uses_worst_verified_case() -> None:
    common = {
        "runtime_config_snapshot_id": "snapshot-v1",
        "answer": "audited answer",
        "evidence_parent_ids": ("parent-1",),
        "route": "fast_rag",
        "events_ref": "fixture://events/coverage",
        "deterministic_metrics": {"leakage": 0},
        "ragas_metrics": {"status": "unavailable"},
        "audited": True,
    }
    strong = EvalCaseResult(case_id="coverage-strong", citation_coverage=1.0, **common)
    weak = EvalCaseResult(case_id="coverage-weak", citation_coverage=0.5, **common)

    assert build_summary([strong, weak])["citation_coverage"] == 0.5


@pytest.mark.parametrize(
    "events_ref",
    [
        "https://example.test/events",
        "artifact://../secret",
        "artifact://events with spaces",
        "artifact://authorization: bearer secret",
        "artifact://tool-output/1",
        "artifact://events/secret",
        "prompt text without a URI",
        "artifact://events\nnext",
    ],
)
def test_eval_case_result_events_ref_accepts_only_safe_uri_references(events_ref: str) -> None:
    with pytest.raises(ValueError):
        EvalCaseResult(
            case_id="case-1",
            runtime_config_snapshot_id="snapshot-v1",
            answer="answer",
            evidence_parent_ids=("parent-1",),
            route="fast_rag",
            events_ref=events_ref,
            deterministic_metrics={"parent_recall_at_6": 0.0},
            ragas_metrics={"status": "unavailable"},
        )


@pytest.mark.asyncio
async def test_runner_extracts_task2_deterministic_metrics(tmp_path: Path) -> None:
    class MetricClient(FakeQueryClient):
        async def query(self, case: EvaluationCase) -> dict[str, object]:
            result = await super().query(case)
            result.update(
                {
                    "ranked_parent_ids": ["wrong", "parent-1", "parent-1"],
                    "leaked_parent_ids": ["other-user-parent"],
                }
            )
            return result

    summary = await EvalRunner(MetricClient(), output_dir=tmp_path).run([_case()])
    row = json.loads((tmp_path / "results.jsonl").read_text(encoding="utf-8"))

    assert row["deterministic_metrics"]["parent_recall_at_6"] == 1.0
    assert row["deterministic_metrics"]["mrr"] == pytest.approx(0.5)
    assert row["deterministic_metrics"]["ndcg_at_10"] == pytest.approx(1.0 / 1.5849625)
    assert row["deterministic_metrics"]["leakage"] == 0
    assert summary["metrics"]["parent_recall_at_6"] == 1.0


@pytest.mark.asyncio
async def test_leakage_comes_only_from_scoped_security_events(tmp_path: Path) -> None:
    class EventClient(FakeQueryClient):
        async def query(self, case: EvaluationCase) -> dict[str, object]:
            result = await super().query(case)
            result.update(
                {
                    "leaked_parent_ids": ["untrusted-claim"],
                    "user_leak_count": 99,
                    "events": [
                        {
                            "event_key": "security-1",
                            "run_id": "run-1",
                            "user_id": case.user_id,
                            "runtime_config_snapshot_id": case.runtime_config_snapshot_id,
                            "event_type": "SECURITY_VIOLATION",
                            "attributes": {"user_leak_count": 1},
                        }
                    ],
                }
            )
            return result

    await EvalRunner(EventClient(), output_dir=tmp_path).run([_case()])
    row = json.loads((tmp_path / "results.jsonl").read_text(encoding="utf-8"))
    assert row["deterministic_metrics"]["leakage"] == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "ragas_result",
    [
        {"status": "available", "metrics": {}},
        {"status": "available", "metrics": {"faithfulness": "not-a-score"}},
        {"status": "available", "metrics": {"faithfulness": float("nan")}},
        {"status": "unavailable", "metrics": {"faithfulness": 0.2}},
        {"status": "available", "metrics": {"faithfulness": 0.2}, "raw": "payload"},
    ],
)
async def test_runner_rejects_malformed_injected_ragas_results(
    tmp_path: Path, ragas_result: dict[str, object]
) -> None:
    class InvalidRagas:
        async def evaluate(self, **_: object) -> dict[str, object]:
            return ragas_result

    with pytest.raises((TypeError, ValueError)):
        await EvalRunner(
            FakeQueryClient(), output_dir=tmp_path, ragas_adapter=InvalidRagas()
        ).run([_case()])


@pytest.mark.asyncio
async def test_ragas_unavailable_is_explicit_and_never_a_fake_score(tmp_path: Path) -> None:
    result = await EvalRunner(FakeQueryClient(), output_dir=tmp_path).run([_case()])
    assert result["ragas"]["status"] == "unavailable"
    assert result["ragas"]["metrics"] == {}
    with pytest.raises(RagasUnavailable):
        # The exception remains available to callers that require strict Ragas.
        raise RagasUnavailable("ragas is not installed")


@pytest.mark.asyncio
async def test_injected_ragas_adapter_receives_only_public_offline_fields(tmp_path: Path) -> None:
    calls: list[dict[str, object]] = []

    class FakeRagas:
        async def evaluate(self, **kwargs: object) -> dict[str, object]:
            calls.append(dict(kwargs))
            return {"faithfulness": 0.75, "answer_relevancy": 0.5, "context_precision": 1.0}

    await EvalRunner(
        FakeQueryClient(), output_dir=tmp_path, ragas_adapter=FakeRagas()
    ).run([_case()])

    assert calls == [
        {
            "answer": "Parent one is relevant.",
            "contexts": (),
            "reference_answer": "Parent one is relevant.",
        }
    ]
    row = json.loads((tmp_path / "results.jsonl").read_text(encoding="utf-8"))
    assert row["ragas_metrics"]["faithfulness"] == 0.75


def test_eval_case_result_rejects_unknown_or_malformed_fields() -> None:
    with pytest.raises(ValueError):
        EvalCaseResult.model_validate(
            {
                "case_id": "case-1",
                "runtime_config_snapshot_id": "snapshot-v1",
                "answer": "answer",
                "evidence_parent_ids": ["parent-1"],
                "route": "fast_rag",
                "events_ref": "artifact://events",
                "deterministic_metrics": {"parent_recall_at_6": float("nan")},
                "ragas_metrics": {"status": "unavailable"},
                "unexpected": True,
            }
        )


@pytest.mark.parametrize(
    "answer",
    ["chain-of-thought: hidden reasoning", "authorization: Bearer secret-token"],
)
def test_eval_case_result_rejects_secrets_and_hidden_reasoning(answer: str) -> None:
    with pytest.raises(ValueError, match="secret|reasoning"):
        EvalCaseResult(
            case_id="case-1",
            runtime_config_snapshot_id="snapshot-v1",
            answer=answer,
            evidence_parent_ids=("parent-1",),
            route="fast_rag",
            events_ref="artifact://events",
            deterministic_metrics={"parent_recall_at_6": 0.0},
            ragas_metrics={"status": "unavailable"},
        )


def test_cli_smoke_uses_deterministic_fixture(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    from evals.run import main

    dataset = tmp_path / "baseline.jsonl"
    dataset.write_text(_case().model_dump_json() + "\n", encoding="utf-8")
    output = tmp_path / "cli-out"

    assert main(["--dataset", str(dataset), "--output", str(output), "--limit", "1"]) == 0
    assert (output / "results.jsonl").exists()
    assert (output / "summary.json").exists()
    assert "completed_cases" in capsys.readouterr().out
