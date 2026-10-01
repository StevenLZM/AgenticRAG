import pytest

from evals.answer_judge import FormalRagasJudge, answer_requests, evaluate_task_success, formal_judge_config
from evals.gold_v2_models import GoldCaseV2
from tests.unit.evals.test_gold_v2_validation import gold_payload
from tests.unit.evals.test_ragas import config, Metric


@pytest.mark.asyncio
async def test_factual_precision_recall_f1_share_one_verified_claim_sample():
    from types import SimpleNamespace
    from evals.answer_judge import shared_factual_scores
    class Factual:
        calls = 0
        async def _decompose_claims(self, text):
            self.calls += 1
            return ["a", "b", "c"] if text == "answer" else ["a", "b", "d"]
        async def _verify_claims(self, claims, reference):
            self.calls += 1
            return SimpleNamespace(statements=[SimpleNamespace(statement=c, reason="checked", verdict=int(c != "c" and c != "d")) for c in claims])
    metric = Factual()
    scores = await shared_factual_scores(metric, "answer", "gold")
    assert metric.calls == 4
    assert scores["counts"] == {"tp": 2, "fp": 1, "fn": 1}
    assert scores["precision"] == pytest.approx(2 / 3)
    assert scores["recall"] == pytest.approx(2 / 3)
    assert scores["f1"] == pytest.approx(2 / 3)
    assert len(scores["response_claims"]) == 3


@pytest.mark.asyncio
async def test_factual_incomplete_nli_is_unknown_not_false_perfect_score():
    from types import SimpleNamespace
    from evals.answer_judge import shared_factual_scores
    class Factual:
        async def _decompose_claims(self, text):
            return ["a", "b"]
        async def _verify_claims(self, claims, reference):
            return SimpleNamespace(statements=[SimpleNamespace(statement="a", reason="checked", verdict=1)])
    with pytest.raises(ValueError, match="claim"):
        await shared_factual_scores(Factual(), "answer", "gold")


@pytest.mark.asyncio
async def test_indexed_nli_binds_each_decision_to_original_claim_without_text_copy():
    from evals.answer_judge import indexed_nli
    class LLM:
        async def agenerate(self, prompt, schema):
            assert '"claim_id": 0' in prompt and "原始公司名称" in prompt
            return schema.model_validate({"checks": [
                {"claim_id": 1, "supported": False, "reason": "not supported"},
                {"claim_id": 0, "supported": True, "reason": "supported"}]})
    result = await indexed_nli(LLM(), ["原始公司名称", "另一声明"], "reference")
    assert [s.statement for s in result.statements] == ["原始公司名称", "另一声明"]
    assert [s.verdict for s in result.statements] == [1, 0]


@pytest.mark.asyncio
@pytest.mark.parametrize("identities", [[0], [0, 0], [0, 2]])
async def test_indexed_nli_rejects_missing_duplicate_and_foreign_claim_ids(identities):
    from evals.answer_judge import indexed_nli
    class LLM:
        async def agenerate(self, prompt, schema):
            return schema.model_validate({"checks": [{"claim_id": identity, "supported": True, "reason": "checked"} for identity in identities]})
    with pytest.raises(ValueError, match="claim"):
        await indexed_nli(LLM(), ["first", "second"], "reference")


@pytest.mark.asyncio
async def test_shared_factual_metrics_persist_one_sample_and_resume_without_call(monkeypatch):
    from types import SimpleNamespace
    class Factual:
        calls = 0
        async def _decompose_claims(self, text):
            self.calls += 1
            return ["a", "b"]
        async def _verify_claims(self, claims, reference):
            self.calls += 1
            return SimpleNamespace(statements=[SimpleNamespace(statement=c, reason="checked", verdict=int(c == "a")) for c in claims])
    metric = Factual()
    monkeypatch.setattr(FormalRagasJudge, "_load_metrics", lambda self: {
        name: metric if name == "factual_correctness_f1" else Metric(.75)
        for name in answer_requests("q", "a", ["c"], "r", True)})
    backend = FormalRagasJudge(config(max_attempts=1))
    first = await backend.evaluate_v2(question="q", answer="a", contexts=["c"], reference_answer="r")
    facts = [v for k, v in first["metrics"].items() if k.startswith("factual_correctness_")]
    assert metric.calls == 4 and {f["value"] for f in facts} == {.5}
    assert len({f["sample_id"] for f in facts}) == 1
    again = await backend.evaluate_v2(question="q", answer="a", contexts=["c"], reference_answer="r", completed=first["metrics"])
    assert again == first and metric.calls == 4
    partial = {"factual_correctness_precision": {"status": "running", "value": None}}
    incomplete = await backend.evaluate_v2(question="q", answer="a", contexts=["c"], reference_answer="r", completed=partial)
    assert incomplete["metrics"]["factual_correctness_recall"]["status"] == "failed"
    assert metric.calls == 4


def test_correctness_compares_final_answer_to_gold_not_context():
    requests = answer_requests("question", "wrong but fluent", ["wrong context"], "gold", True)
    assert requests["answer_correctness"] == {"user_input": "question", "response": "wrong but fluent", "reference": "gold"}
    assert requests["factual_correctness_f1"] == {"response": "wrong but fluent", "reference": "gold"}
    assert "context_recall" in requests


@pytest.mark.asyncio
async def test_partial_judge_failure_preserves_successful_metrics(monkeypatch):
    monkeypatch.setattr(FormalRagasJudge, "_load_metrics", lambda self: {
        name: Metric(error=RuntimeError("sk-do-not-log")) if name == "context_recall" else Metric(.75)
        for name in answer_requests("q", "a", ["c"], "r", True)})
    backend = FormalRagasJudge(config(max_attempts=1))
    result = await backend.evaluate_v2(question="q", answer="a", contexts=["c"], reference_answer="r", answerable=True)
    assert result["metrics"]["answer_correctness"]["value"] == .75
    assert result["metrics"]["context_recall"]["value"] is None
    assert result["metrics"]["context_recall"]["status"] == "failed"
    assert "sk-" not in str(result)


def test_hard_wrong_field_and_budget_limit_never_pass():
    case = GoldCaseV2.model_validate(gold_payload())
    verdict = {"fully_correct": True, "contradictions": []}
    assert evaluate_task_success(case, "completed", verdict, {"all_critical_fields_pass": 0})["value"] == 0
    assert evaluate_task_success(case, "research_round_limit", verdict, {"all_critical_fields_pass": 1})["value"] == 0
    assert evaluate_task_success(case, "completed", None, {"all_critical_fields_pass": 1})["value"] is None


def test_formal_deepseek_judge_explicitly_disables_default_thinking(monkeypatch):
    from types import SimpleNamespace
    from evals.judge import JudgeConfig
    monkeypatch.setattr(JudgeConfig, "from_settings", lambda settings: config())
    monkeypatch.setenv("RAG_EVAL_JUDGE_BASE_URL", "https://api.deepseek.com")
    monkeypatch.delenv("RAG_EVAL_JUDGE_THINKING", raising=False)
    settings = SimpleNamespace(main_model="answer-model", light_model="deepseek-v4-flash", deepseek_base_url="https://api.deepseek.com")
    configured = formal_judge_config(settings)
    assert configured.thinking_mode == "disabled"
    assert configured.max_output_tokens == 8192


@pytest.mark.asyncio
async def test_grounded_extractor_cannot_rewrite_company_and_receives_no_gold(monkeypatch):
    assert hasattr(FormalRagasJudge, "extract_grounded_facts"), "grounded extraction missing"
    from tests.unit.evals.test_source_selection import selected_fact, ref
    answer = "虚构星河04公司上限500元。"
    class LLM:
        async def agenerate(self, prompt, schema):
            assert '"reference":' not in prompt and '"critical_fields":' not in prompt
            return schema.model_validate({"facts": [selected_fact([0], subject_spans=[
                ref("r0", "虚构星尘04公司")])],
                "segments": [{"segment_id": 0, "factual": True}]})
    monkeypatch.setattr(FormalRagasJudge, "_load_metrics", lambda self: {})
    judge = FormalRagasJudge(config(max_attempts=1))
    judge._llm = LLM()
    with pytest.raises(ValueError, match="selection_quote"):
        await judge.extract_grounded_facts("上限是多少？", answer)


@pytest.mark.asyncio
async def test_grounded_provider_contract_selects_clauses_instead_of_counting_characters(monkeypatch):
    import json
    from tests.unit.evals.test_source_selection import selected_fact, ref
    answer = "😀旧470元，新530元，增加60元。"
    raw = {"facts": [selected_fact([0], value_spans=[ref("r2", "60")],
        unit_spans=[ref("r2", "元")])], "segments": [{"segment_id": 0, "factual": True}]}
    class LLM:
        async def agenerate(self, prompt, schema):
            payload = json.loads(prompt.split("\n", 1)[1])
            assert {"span_id": "r2", "source": "response", "text": "增加60元。"} in payload["source_catalog"]
            assert "reference_answer" not in payload and "gold" not in payload
            return schema.model_validate(raw)
    monkeypatch.setattr(FormalRagasJudge, "_load_metrics", lambda self: {})
    judge = FormalRagasJudge(config(max_attempts=1))
    judge._llm = LLM()
    facts = await judge.extract_grounded_facts("增加多少？", answer)
    assert facts[0].unit_spans[0].start == 17
    assert facts[0].value_spans[0].quote == "60"


@pytest.mark.asyncio
async def test_fields_use_numbered_provenance_and_preserve_missing_evidence_as_error(monkeypatch):
    from tests.unit.evals.test_source_selection import ref
    from evals.grounded_judge import GroundedJudgeError
    case = GoldCaseV2.model_validate(gold_payload())
    answer = "每人每晚旧470元，新530元，增加60元。"
    raw = {"fields": [{"name": "increase", "values": [{"value": 60, "unit": "CNY/person/night",
        "conditions": {}, "evidence": ref("r2", "增加60元。"),
        "unit_evidence": [ref("r2", "元"), ref("r0", "每人每晚")], "condition_evidence": {}}]}]}
    class LLM:
        async def agenerate(self, prompt, schema):
            return schema.model_validate(raw)
    monkeypatch.setattr(FormalRagasJudge, "_load_metrics", lambda self: {})
    judge = FormalRagasJudge(config(max_attempts=1))
    judge._llm = LLM()
    result = await judge.extract_fields(case, answer)
    assert result["all_critical_fields_pass"] == 1
    raw["fields"][0]["values"][0]["unit_evidence"] = []
    with pytest.raises(GroundedJudgeError, match="unit") as error:
        await judge.extract_fields(case, answer)
    assert error.value.grading_evidence["selection"] == raw


@pytest.mark.asyncio
async def test_grounded_scores_use_separate_gold_denominator_and_original_text(monkeypatch):
    assert hasattr(FormalRagasJudge, "evaluate_grounded"), "grounded evaluator missing"
    from evals.gold_answer_spec import build_draft_spec
    from evals.grounded_facts import GroundedFact
    from tests.unit.evals.test_grounded_facts import fact, span
    answer = "A公司500元。B公司600元。"
    case = GoldCaseV2.model_validate(gold_payload()).model_copy(update={"reference_answer": "A公司500元。B公司600元。含税。"})
    spec = build_draft_spec(case)
    facts = [GroundedFact.model_validate(fact(answer, fact_id=i, evidence_spans=[span(answer, part)]))
             for i, part in enumerate(["A公司500元。", "B公司600元。"])]
    class LLM:
        async def agenerate(self, prompt, schema):
            assert answer in prompt and "增加多少元" in prompt
            return schema.model_validate({"extraction_faithful": True, "extraction_complete": True,
                "extraction_reason": "原文完整", "response_checks": [
                    {"fact_id": 0, "verdict": "supported", "duplicate_of": None, "reason": "支持"},
                    {"fact_id": 1, "verdict": "supported", "duplicate_of": None, "reason": "支持"}],
                "reference_checks": [{"fact_id": f.fact_id, "covered": i < 2, "reason": "核验"}
                    for i, f in enumerate(spec.required_facts)]})
    monkeypatch.setattr(FormalRagasJudge, "_load_metrics", lambda self: {})
    judge = FormalRagasJudge(config(max_attempts=1))
    judge._llm = LLM()
    result = await judge.evaluate_grounded(case.question, answer, facts, spec)
    assert result["precision"] == 1
    assert result["recall"] == pytest.approx(2 / 3)
    assert result["f1"] == pytest.approx(.8)
    assert result["all_required_facts_covered"] is False
    assert result["eligible_for_tuning"] is False


def test_grounded_judge_failure_cannot_be_hidden_by_positive_verdict():
    import inspect
    assert "grounded" in inspect.signature(evaluate_task_success).parameters
    case = GoldCaseV2.model_validate(gold_payload())
    verdict = {"fully_correct": True, "contradictions": [], "missing_facts": []}
    result = evaluate_task_success(case, "completed", verdict, {"all_critical_fields_pass": 1},
        grounded={"status": "failed", "value": None})
    assert result["value"] is None and result["reason"] == "judge_error"
    result = evaluate_task_success(case, "completed", verdict, {"all_critical_fields_pass": 1},
        grounded={"status": "available", "all_required_facts_covered": False, "contradicted": 0})
    assert result["value"] == 0 and result["reason"] == "missing_required_facts"


@pytest.mark.parametrize("kind", ["missing", "duplicate", "foreign", "unfaithful", "incomplete", "bad_duplicate"])
def test_grounded_checks_fail_closed_on_misalignment_or_lost_meaning(kind):
    from evals.gold_answer_spec import build_draft_spec
    from evals.grounded_facts import GroundedFact
    from evals.grounded_judge import GroundedChecks, grounded_scores
    from tests.unit.evals.test_grounded_facts import fact
    spec = build_draft_spec(GoldCaseV2.model_validate(gold_payload()))
    facts = [GroundedFact.model_validate(fact("增加60元。"))]
    response = {"fact_id": 0, "verdict": "supported", "duplicate_of": None, "reason": "核验"}
    raw = {"extraction_faithful": True, "extraction_complete": True, "extraction_reason": "已检查",
           "response_checks": [response], "reference_checks": [{"fact_id": spec.required_facts[0].fact_id, "covered": True, "reason": "核验"}]}
    if kind == "missing":
        raw["response_checks"] = []
    elif kind == "duplicate":
        raw["response_checks"] = [response, response]
    elif kind == "foreign":
        response["fact_id"] = 3
    elif kind == "unfaithful":
        raw["extraction_faithful"] = False
    elif kind == "incomplete":
        raw["extraction_complete"] = False
    else:
        response["duplicate_of"] = 0
    with pytest.raises(ValueError):
        grounded_scores(facts, spec, GroundedChecks.model_validate(raw))


def test_grounded_duplicate_not_double_counted_and_optional_not_in_recall_denominator():
    from evals.gold_answer_spec import build_draft_spec, ReferenceFact
    from evals.grounded_facts import GroundedFact
    from evals.grounded_judge import GroundedChecks, grounded_scores
    from tests.unit.evals.test_grounded_facts import fact
    spec = build_draft_spec(GoldCaseV2.model_validate(gold_payload()))
    spec = spec.model_copy(update={"optional_facts": [ReferenceFact(fact_id="optional", claim="补充说明",
        source_fact_ids=spec.required_facts[0].source_fact_ids)]})
    facts = [GroundedFact.model_validate(fact("增加60元。", fact_id=i)) for i in range(2)]
    checks = GroundedChecks.model_validate({"extraction_faithful": True, "extraction_complete": True,
        "extraction_reason": "已检查", "response_checks": [
            {"fact_id": 0, "verdict": "supported", "duplicate_of": None, "reason": "支持"},
            {"fact_id": 1, "verdict": "supported", "duplicate_of": 0, "reason": "相同事实"}],
        "reference_checks": [{"fact_id": f.fact_id, "covered": True, "reason": "已覆盖"} for f in spec.required_facts]})
    result = grounded_scores(facts, spec, checks)
    assert result["response_total"] == result["supported"] == 1
    assert result["required_total"] == len(spec.required_facts)
    assert result["recall"] == result["precision"] == 1
