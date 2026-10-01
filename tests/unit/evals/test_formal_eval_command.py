import json

import pytest

from evals.formal_command import select_cases, durable_operation, FormalHttpClient, score_case, import_query_ledger
from evals.gold_v2_models import GoldCaseV2
from tests.unit.evals.test_gold_v2_validation import gold_payload


def cases():
    rows = []
    for category in ("single_hop", "multi_hop", "table_calculation", "unanswerable", "section_comprehension"):
        for i in range(6):
            row = gold_payload()
            row.update(case_id=f"{category}:{i}", category=category)
            rows.append(GoldCaseV2.model_validate(row))
    return rows


def test_smoke_is_stratified_not_first20_templates():
    selected = select_cases(cases(), split="dev", smoke=20)
    assert len(selected) == 20
    assert {c.category for c in selected} == {c.category for c in cases()}
    assert select_cases(cases(), split="dev", smoke=20) == selected


@pytest.mark.asyncio
async def test_completed_judge_operation_resumes_without_call(tmp_path):
    count = 0
    async def score():
        nonlocal count
        count += 1
        return {"score": .75}
    path = tmp_path / "metric.json"
    first = await durable_operation(path, "binding", score)
    assert await durable_operation(path, "binding", score) == first
    assert count == 1
    with pytest.raises(ValueError, match="binding"):
        await durable_operation(path, "changed", score)


@pytest.mark.asyncio
async def test_uncertain_or_failed_judge_is_not_blindly_recharged(tmp_path):
    path = tmp_path / "metric.json"
    path.write_text(json.dumps({"binding": "b", "status": "running"}))
    async def forbidden():
        raise AssertionError("must not run")
    with pytest.raises(ValueError, match="reconcil"):
        await durable_operation(path, "b", forbidden)


@pytest.mark.asyncio
async def test_unknown_post_refuses_second_submission(tmp_path):
    class Transport:
        calls = 0
        async def post(self, *args, **kwargs):
            self.calls += 1
            raise TimeoutError()
    transport = Transport()
    client = FormalHttpClient(transport, None, "runtime", {"session_id": "e"}, tmp_path)
    case = cases()[0]
    with pytest.raises(TimeoutError):
        await client.query(case)
    with pytest.raises(ValueError, match="reconcil"):
        await client.query(case)
    assert transport.calls == 1


@pytest.mark.asyncio
async def test_existing_result_without_query_ledger_cannot_submit_again(tmp_path):
    class Transport:
        async def post(self, *args, **kwargs):
            raise AssertionError("missing ledger must not resubmit a paid query")
    client = FormalHttpClient(Transport(), None, "runtime", {}, tmp_path)
    with pytest.raises(ValueError, match="missing query ledger"):
        await client.query(cases()[0], require_existing=True)


@pytest.mark.asyncio
async def test_empty_terminal_answer_is_never_invented_or_sent_to_judge(tmp_path):
    case = cases()[0].model_copy(update={"answerable": False, "critical_fields": []})
    class Judge:
        metadata_v2 = {}
        async def verdict(self, *args):
            raise AssertionError("an empty reply must not be invented")
    trace = {"context_items": [], "retrieval_rounds": [], "answer": "", "outcome": "cannot_answer",
             "run_id": "r", "route": "research", "answer_origin": "terminal_status", "research_rounds": 1,
             "retrieval_calls": 0}
    result = await score_case(case, trace, Judge(), tmp_path, "binding")
    assert result["answer"] == ""
    assert result["task_success"]["value"] == 0


@pytest.mark.asyncio
async def test_ragas_uses_exact_rendered_context_with_headings(tmp_path):
    from agentic_rag.safety.context import DataEnvelope
    envelopes = [DataEnvelope(source_label="document:d", evidence_id="e", content="470元", heading_path=("2026年虚构公司政策",)).render(),
                 DataEnvelope(source_label="document:n", evidence_id="n", content="不相关片段").render()]
    case = cases()[0].model_copy(update={"critical_fields": []})
    class Judge:
        metadata_v2 = {}
        async def verdict(self, *args):
            return {"fully_correct": True, "contradictions": [], "missing_facts": []}
        async def evaluate_v2(self, **kwargs):
            assert kwargs["contexts"] == envelopes
            return {"metrics": {}, "metadata": {}}
    trace = {"context_items": [{"parent_id": "p", "document_id": "d", "evidence_id": "e", "content": "470元", "heading_path": ["2026年虚构公司政策"]},
                               {"parent_id": "n", "document_id": "n", "evidence_id": "n", "content": "不相关片段"}],
             "rendered_context": "\n".join(envelopes), "retrieval_rounds": [],
             "answer": "470元", "outcome": "completed", "run_id": "r", "route": "research",
             "answer_origin": "model", "research_rounds": 1, "retrieval_calls": 0}
    await score_case(case, trace, Judge(), tmp_path, "binding")


def test_reuse_query_keeps_original_run_and_refuses_unknown_post(tmp_path):
    from evals.gold_v2_models import canonical_hash
    case = cases()[0]
    source, target = tmp_path / "old", tmp_path / "new"
    source.mkdir()
    binding = canonical_hash({"case": case.model_dump(mode="json"), "snapshot": "runtime"})
    path = source / (case.case_id + ".json")
    path.write_text(json.dumps({"binding": binding, "status": "completed", "run_id": "paid-existing-run"}))
    import_query_ledger(source, target, case, "runtime")
    assert json.loads((target / path.name).read_text())["run_id"] == "paid-existing-run"
    path.write_text(json.dumps({"binding": binding, "status": "submitting"}))
    with pytest.raises(ValueError, match="reconcil"):
        import_query_ledger(source, tmp_path / "other", case, "runtime")


def test_continue_can_fill_pending_only_with_identical_query_deployment():
    from evals.formal_command import validate_continuation
    source = {"runtime_config_snapshot_id": "same-runtime-including-evaluation"}
    validate_continuation(source, "same-runtime-including-evaluation")
    with pytest.raises(ValueError, match="deployment"):
        validate_continuation(source, "different-code-or-provider")


@pytest.mark.asyncio
async def test_cached_case_requires_judge_artifacts_without_paid_recovery(tmp_path):
    case = cases()[0].model_copy(update={"critical_fields": []})
    class Judge:
        metadata_v2 = {}
        async def verdict(self, *args):
            raise AssertionError("must not automatically charge for missing cache evidence")
    trace = {"context_items": [], "rendered_context": "", "retrieval_rounds": [], "answer": "a", "outcome": "completed",
             "run_id": "r", "route": "research", "answer_origin": "model", "research_rounds": 1, "retrieval_calls": 0}
    with pytest.raises(ValueError, match="cached"):
        await score_case(case, trace, Judge(), tmp_path, "b", require_cached=True)


@pytest.mark.asyncio
@pytest.mark.parametrize("with_fields", [False, True])
async def test_grounded_operation_error_stays_unknown_and_resumes_without_new_calls(tmp_path, with_fields):
    import inspect
    assert "answer_spec" in inspect.signature(score_case).parameters
    from evals.gold_answer_spec import build_draft_spec
    from evals.grounded_judge import GroundedJudgeError
    from evals.answer_judge import answer_requests
    case = cases()[0] if with_fields else cases()[0].model_copy(update={"critical_fields": []})
    class Judge:
        metadata_v2 = {}
        calls = 0
        async def extract_grounded_facts(self, *args):
            self.calls += 1
            raise GroundedJudgeError("source_quote_missing_or_ambiguous", {"bad_quote": "虚构星尘"})
        async def extract_fields(self, *args):
            self.calls += 1
            raise GroundedJudgeError("unit_not_supported_by_quote", {"unit_evidence": []})
        async def verdict(self, *args):
            self.calls += 1
            return {"fully_correct": True, "missing_facts": [], "contradictions": []}
        async def evaluate_v2(self, **kwargs):
            # Real durable coordinator tested; only paid model boundary replaced.
            metrics = dict(kwargs["completed"])
            for name in answer_requests(case.question, "470元", [], case.reference_answer, True):
                if name not in metrics:
                    self.calls += 1
                    metrics[name] = {"status": "available", "value": 1.0}
                    kwargs["on_metric"](name, metrics[name])
            return {"metrics": metrics, "metadata": {}}
    trace = {"context_items": [], "rendered_context": "", "retrieval_rounds": [], "answer": "470元", "outcome": "completed",
             "run_id": "r", "route": "research", "answer_origin": "model", "research_rounds": 1, "retrieval_calls": 0}
    judge = Judge()
    first = await score_case(case, trace, judge, tmp_path, "b", answer_spec=build_draft_spec(case))
    assert first["task_success"]["value"] is None
    assert first["grounded"]["status"] == "failed"
    assert first["failure_class"] == "judge_error"
    if with_fields:
        assert first["fields"]["reason"] == "unit_not_supported_by_quote"
    assert json.loads((tmp_path / "judges" / case.case_id / "grounded_extract.json").read_text())["grading_evidence"] == {"bad_quote": "虚构星尘"}
    count = judge.calls
    again = await score_case(case, trace, judge, tmp_path, "b", answer_spec=build_draft_spec(case), require_cached=True)
    assert again == first and judge.calls == count


def test_cli_exposes_frozen_answer_spec_input(monkeypatch):
    from evals import command
    async def run(args):
        assert str(args.answer_spec) == "spec.jsonl"
        return 0
    monkeypatch.setattr(command, "run", run)
    assert command.main(["--answer-spec", "spec.jsonl"]) == 0
