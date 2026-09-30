"""Entry-point contracts, not scored business evaluations."""
import importlib.util
import json
from pathlib import Path

import pytest


def entry():
    assert importlib.util.find_spec("evals.command") is not None, "one-command entry is missing"
    from evals import command
    return command


def test_document_inventory_rejects_extra_unindexed_documents(tmp_path):
    command = entry()
    (tmp_path / "a.txt").write_text("a")
    manifest = {"documents": [{"filename": "a.txt"}]}
    command.check_inventory(tmp_path, manifest)
    (tmp_path / "nested").mkdir()
    (tmp_path / "nested/b.pdf").write_bytes(b"extra")
    with pytest.raises(ValueError, match="inventory"):
        command.check_inventory(tmp_path, manifest)


def test_output_requires_explicit_resume_and_binding(tmp_path):
    command = entry()
    output = tmp_path / "experiment"
    binding = {"snapshot": "s1", "gold": "sha1"}
    with command.experiment(output, binding, resume=False):
        assert json.loads((output / "experiment.json").read_text()) == binding
        with pytest.raises(ValueError, match="running"):
            with command.experiment(output, binding, resume=True):
                pass
    with pytest.raises(ValueError, match="exists"):
        with command.experiment(output, binding, resume=False):
            pass
    with pytest.raises(ValueError, match="binding"):
        with command.experiment(output, {**binding, "snapshot": "s2"}, resume=True):
            pass
    with command.experiment(output, binding, resume=True):
        assert output.exists()


def test_report_keeps_partial_failures_and_metric_denominators():
    command = entry()
    report = command.render_report({"requested_cases": 3, "completed_cases": 2,
        "real_query_count": 2, "metrics": {"parent_recall@6": 1},
        "metric_sample_counts": {"parent_recall@6": 1},
        "ragas": {"status": "failed"}, "failures": [{"case_id": "q3", "reason": "timeout"}]},
        verified=False)
    assert "未完成" in report and "q3" in report and "timeout" in report
    assert "parent_recall@6 | 1 | 1" in report
    assert "2 / 3" in report


def test_report_displays_ragas_denominators_and_terminal_outcomes():
    command = entry()
    report = command.render_report({"requested_cases": 24, "completed_cases": 24,
        "ragas": {"status": "available", "metrics": {"faithfulness": 0.98, "correct_refusal": 0.25},
                  "metric_sample_counts": {"faithfulness": 20, "correct_refusal": 4}},
        "outcome_counts": {"completed": 20, "research_round_limit": 3, "cannot_answer": 1}}, verified=True)
    assert "faithfulness | 0.98 | 20" in report
    assert "correct_refusal | 0.25 | 4" in report
    assert "research_round_limit" in report


@pytest.mark.asyncio
async def test_verifier_reads_separate_output_not_original_results(tmp_path, monkeypatch):
    from evals import saved_report
    monkeypatch.setattr(saved_report, "load_frozen_inputs", lambda *a, **kw: ({}, {}, {"cases": []}))
    source = tmp_path / "source"
    output = tmp_path / "new"
    output.mkdir()
    (output / "summary.json").write_text("{}")
    (output / "results.jsonl").write_text("")
    with pytest.raises(ValueError, match="aggregate"):
        await saved_report.verify_saved_evaluation(source, output_dir=output)


@pytest.mark.asyncio
async def test_failed_preflight_has_no_output(tmp_path, monkeypatch):
    command = entry()
    async def unavailable(*args):
        raise ValueError("missing index")
    monkeypatch.setattr(command, "preflight", unavailable)
    from argparse import Namespace
    output = tmp_path / "new"
    with pytest.raises(ValueError, match="missing index"):
        await command.run(Namespace(resume=None, source_run=tmp_path, docs_dir=None,
                                    gold=None, check=False, output_dir=output))
    assert not output.exists()


@pytest.mark.asyncio
async def test_output_cannot_write_inside_source_runtime(tmp_path, monkeypatch):
    command = entry()
    from argparse import Namespace
    root = tmp_path / "source"
    root.mkdir()
    (root / "gold-mapping.json").write_text("{}")
    async def ready(*args):
        return ({"manifest_sha256": "m", "allocation_id": "a", "index_generation": "g"}, {},
                {"runtime_config_snapshot_id": "s"}, 8, 24)
    monkeypatch.setattr(command, "preflight", ready)
    output = root / "runtime" / "nested"
    with pytest.raises(ValueError, match="overlaps"):
        await command.run(Namespace(resume=None, source_run=root, docs_dir=None,
                                    gold=None, check=False, output_dir=output))
    assert not output.exists()


@pytest.mark.asyncio
async def test_partial_evaluation_writes_report_and_nonzero_exit(tmp_path, monkeypatch):
    command = entry()
    from argparse import Namespace
    root = tmp_path / "source"
    root.mkdir()
    (root / "gold-mapping.json").write_text("{}")
    async def ready(*args):
        return ({"manifest_sha256": "m", "allocation_id": "a", "index_generation": "g"}, {},
                {"runtime_config_snapshot_id": "s"}, 8, 24)
    async def evaluate(*args, output_dir, **kwargs):
        (output_dir / "failures.jsonl").write_text(json.dumps({"case_id": "q2", "reason": "evaluation_failed"}) + "\n")
        (output_dir / "summary.json").write_text(json.dumps({"requested_cases": 24, "completed_cases": 1,
            "failures": [{"case_id": "q2", "reason": "evaluation_failed"}]}))
        return 1
    monkeypatch.setattr(command, "preflight", ready)
    monkeypatch.setattr(command, "evaluate_cases", evaluate)
    output = tmp_path / "experiment"
    code = await command.run(Namespace(resume=None, source_run=root, docs_dir=None,
                                      gold=None, check=False, output_dir=output))
    assert code == 1
    assert json.loads((output / "report.json").read_text())["completion_verified"] is False
    assert "q2" in (output / "report.md").read_text()


@pytest.mark.asyncio
async def test_verification_error_reports_saved_rows_without_summary(tmp_path, monkeypatch):
    command = entry()
    from argparse import Namespace
    from evals.run import EvalCaseResult
    root = tmp_path / "source"
    root.mkdir()
    (root / "gold-mapping.json").write_text("{}")
    async def ready(*args):
        return ({"manifest_sha256": "m", "allocation_id": "a", "index_generation": "g"}, {},
                {"runtime_config_snapshot_id": "s"}, 8, 24)
    async def interrupted(*args, output_dir, **kwargs):
        row = EvalCaseResult(case_id="q1", runtime_config_snapshot_id="s", answer="answer",
            evidence_parent_ids=[], route="fast_rag", events_ref="inline://events/q1",
            deterministic_metrics={}, ragas_metrics={"status": "available", "faithfulness": 1},
            citation_coverage=1, audited=True)
        (output_dir / "results.jsonl").write_text(row.model_dump_json() + "\n")
        (output_dir / "failures.jsonl").write_text(json.dumps({"case_id": "q2", "status": "evaluation_failed"}) + "\n")
        raise ValueError("checkpoint no longer matches")
    monkeypatch.setattr(command, "preflight", ready)
    monkeypatch.setattr(command, "evaluate_cases", interrupted)
    output = tmp_path / "experiment"
    assert await command.run(Namespace(resume=None, source_run=root, docs_dir=None,
        gold=None, check=False, output_dir=output)) == 1
    report = json.loads((output / "report.json").read_text())
    assert report["completed_cases"] == 1
    assert report["real_query_count"] == 0
    assert report["failures"][0]["case_id"] == "q2"
    assert report["completion_verified"] is False


def test_partial_report_ignores_stale_summary_and_handles_corrupt_row(tmp_path):
    command = entry()
    (tmp_path / "summary.json").write_text('{"real_query_count": 24, "completed_cases": 24}')
    (tmp_path / "results.jsonl").write_text("broken json\n")
    result = command.partial_summary(tmp_path, 24)
    assert result["real_query_count"] == 0
    assert result["completed_cases"] == 0
    assert result["artifact_errors"]


@pytest.mark.asyncio
async def test_resume_check_rejects_changed_binding_without_writes(tmp_path, monkeypatch):
    command = entry()
    from argparse import Namespace
    root = tmp_path / "source"
    root.mkdir()
    (root / "gold-mapping.json").write_text("{}")
    output = tmp_path / "experiment"
    output.mkdir()
    record = json.dumps({"source_run": str(root), "snapshot": "old"})
    (output / "experiment.json").write_text(record)
    async def ready(*args):
        return ({"manifest_sha256": "m", "allocation_id": "a", "index_generation": "g"}, {},
                {"runtime_config_snapshot_id": "new"}, 8, 24)
    monkeypatch.setattr(command, "preflight", ready)
    with pytest.raises(ValueError, match="binding"):
        await command.run(Namespace(resume=output, source_run=None, docs_dir=None,
                                    gold=None, check=True, output_dir=None))
    assert list(output.iterdir()) == [output / "experiment.json"]
    assert (output / "experiment.json").read_text() == record


def test_shell_help_works_without_starting_services():
    import subprocess
    path = Path("scripts/eval_rag.sh")
    assert path.exists(), "one-command shell entry is missing"
    result = subprocess.run(["bash", str(path), "--help"], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert "--resume" in result.stdout and "--docs-dir" in result.stdout
