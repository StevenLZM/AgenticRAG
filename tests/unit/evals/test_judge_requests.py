import json

import pytest


def setup_artifact(tmp_path):
    directory = tmp_path / "judges/case"
    directory.mkdir(parents=True)
    (directory / "fields.json").write_text(json.dumps({"status": "available", "result": {}}))


def record(**updates):
    return {"kind": "llm", "status": "completed", "input_tokens": 10, "output_tokens": 5,
            "cached_input_tokens": 0, "operation": "case:fields", "model": "model", **updates}


@pytest.mark.parametrize("saved", [None, {"binding": "foreign", "requests": [record()]},
    {"binding": "bound", "requests": []}, {"binding": "bound", "requests": [record(input_tokens=-1)]},
    {"binding": "bound", "requests": [record(operation="other:fields")]},
    {"binding": "bound", "requests": [record(cached_input_tokens=11)]}])
def test_resume_rejects_missing_foreign_malformed_or_truncated_paid_ledger(tmp_path, saved):
    from evals.judge_requests import reconcile_judge_requests
    setup_artifact(tmp_path)
    if saved is not None:
        (tmp_path / "judge-requests.json").write_text(json.dumps(saved))
    with pytest.raises(ValueError, match="judge requests"):
        reconcile_judge_requests(tmp_path, "bound")


def test_zero_call_resume_loads_all_attempts_without_changing_ledger(tmp_path):
    from evals.judge_requests import reconcile_judge_requests
    setup_artifact(tmp_path)
    requests = [record(status="failed", input_tokens=None, output_tokens=None), record()]
    path = tmp_path / "judge-requests.json"
    path.write_text(json.dumps({"binding": "bound", "requests": requests}))
    before = path.read_bytes()
    assert reconcile_judge_requests(tmp_path, "bound") == requests
    assert path.read_bytes() == before


def test_new_experiment_without_paid_artifacts_can_start_empty(tmp_path):
    from evals.judge_requests import reconcile_judge_requests
    assert reconcile_judge_requests(tmp_path, "bound") == []


@pytest.mark.asyncio
async def test_resume_restores_cumulative_budget_before_any_provider_call(tmp_path):
    from types import SimpleNamespace
    from evals.answer_judge import FormalRagasJudge
    setup_artifact(tmp_path)
    path = tmp_path / "judge-requests.json"
    path.write_text(json.dumps({"binding": "bound", "requests": [record()]}))
    before = path.read_bytes()
    judge = object.__new__(FormalRagasJudge)
    judge.max_requests = 1
    judge.configure_requests(tmp_path, "bound")
    calls = []
    async def create(*args, **kwargs):
        calls.append(kwargs)
        raise AssertionError("must not call provider")
    resource = SimpleNamespace(create=create)
    judge._instrument_client(SimpleNamespace(chat=SimpleNamespace(completions=resource)), "llm")
    with pytest.raises(RuntimeError, match="budget_exhausted"):
        await resource.create()
    assert not calls and judge.request_count == 1
    assert path.read_bytes() == before


def test_report_cannot_trust_foreign_request_ledger_without_new_calls(tmp_path):
    from evals.report_v2 import write_formal_report
    from tests.unit.evals.test_formal_eval_command import cases
    (tmp_path / "judge-requests.json").write_text(json.dumps({"binding": "foreign", "requests": [record()]}))
    with pytest.raises(ValueError, match="binding"):
        write_formal_report(tmp_path, cases()[:1], [], {}, {"fingerprint": "bound"}, {})
    assert not (tmp_path / "report.json").exists()
