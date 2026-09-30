"""Judge contracts; doubles never count as live Ragas evidence."""

from types import SimpleNamespace

import pytest

from evals.judge import JudgeConfig, JudgePreflightError, RagasJudge
from evals.ragas_adapter import RagasAdapter


def config(**changes):
    return JudgeConfig.model_validate({
        "judge_model": "judge-test", "judge_base_url": "https://judge.example/v1",
        "judge_api_key": "contract-only-key", "embedding_model": "embedding-test",
        "embedding_base_url": "https://embedding.example/v1",
        "embedding_api_key": "contract-embedding-key", "retry_delay_seconds": 0,
        **changes,
    })


class Metric:
    def __init__(self, value=0.75, error=None):
        self.value, self.error, self.calls = value, error, []

    async def ascore(self, **kwargs):
        self.calls.append(kwargs)
        if self.error:
            raise self.error
        return SimpleNamespace(value=self.value)


def judge(monkeypatch, **metrics):
    scorers = {name: Metric() for name in ("faithfulness", "answer_relevancy", "context_precision")}
    scorers["correct_refusal"] = Metric("pass")
    scorers.update(metrics)
    monkeypatch.setattr(RagasJudge, "_load_metrics", lambda self: scorers)
    return RagasJudge(config()), scorers


@pytest.mark.asyncio
async def test_judge_receives_question_ordered_contexts_and_reference(monkeypatch):
    backend, scorers = judge(monkeypatch)
    outcome = await RagasAdapter(backend).evaluate(
        question="问题", answer="回答", contexts=["证据一", "证据二"], reference_answer="金标"
    )
    assert outcome.status == "available"
    assert len(outcome.metrics) == 3
    assert scorers["faithfulness"].calls[0]["user_input"] == "问题"
    assert scorers["context_precision"].calls[0]["retrieved_contexts"] == ["证据一", "证据二"]
    assert scorers["context_precision"].calls[0]["reference"] == "金标"
    assert outcome.metadata["judge_model"] == "judge-test"


@pytest.mark.asyncio
@pytest.mark.parametrize("changes", [{"question": ""}, {"contexts": []}, {"reference_answer": ""}])
async def test_judge_rejects_incomplete_answerable_inputs(monkeypatch, changes):
    backend, scorers = judge(monkeypatch)
    with pytest.raises(ValueError):
        await backend.evaluate(**{
            "question": "问题", "answer": "回答", "contexts": ["证据"],
            "reference_answer": "金标", **changes,
        })
    assert not scorers["faithfulness"].calls


@pytest.mark.asyncio
async def test_provider_failure_has_bounded_attempts_no_scores_or_secrets(monkeypatch):
    metric = Metric(error=RuntimeError("sk-sensitive-must-not-be-logged"))
    backend, _ = judge(monkeypatch, faithfulness=metric)
    outcome = await backend.evaluate(question="问题", answer="回答", contexts=["证据"], reference_answer="金标")
    assert outcome.status == "failed"
    assert outcome.metrics == {}
    assert len(metric.calls) == 2
    assert "sensitive" not in str(outcome.as_dict())


@pytest.mark.asyncio
async def test_nonfinite_score_is_failed_not_fabricated(monkeypatch):
    backend, _ = judge(monkeypatch, faithfulness=Metric(float("nan")))
    outcome = await backend.evaluate(question="问题", answer="回答", contexts=["证据"], reference_answer="金标")
    assert outcome.status == "failed" and not outcome.metrics


@pytest.mark.asyncio
async def test_unanswerable_uses_refusal_judge_only(monkeypatch):
    backend, scorers = judge(monkeypatch)
    outcome = await backend.evaluate(question="未知问题", answer="资料没有说明", contexts=[],
                                     reference_answer="无法确定", answerable=False)
    assert outcome.metrics == {"correct_refusal": 1.0}
    assert not scorers["faithfulness"].calls
    assert len(scorers["correct_refusal"].calls) == 1


def test_missing_credentials_fail_preflight():
    with pytest.raises(JudgePreflightError):
        RagasJudge(config(judge_api_key=""))
    assert "contract-only-key" not in repr(config())
