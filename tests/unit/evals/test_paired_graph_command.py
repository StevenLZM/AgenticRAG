from types import SimpleNamespace

import pytest


async def test_paired_runner_alternates_and_completed_resume_makes_no_calls(tmp_path):
    from evals.paired_graph_command import run_pairs
    calls = []
    cases = [SimpleNamespace(case_id="first"), SimpleNamespace(case_id="second")]
    async def query(case, arm, directory):
        calls.append((case.case_id, arm, "query"))
        return {"actual": "query trace"}
    async def score(case, arm, trace, directory):
        calls.append((case.case_id, arm, "score"))
        return {"case_id": case.case_id, "actual": trace["actual"]}
    kwargs = dict(cases=cases, root=tmp_path, binding="binding", query=query, score=score)
    first = await run_pairs(**kwargs)
    assert [(c, a) for c, a, operation in calls if operation == "query"] == [
        ("first", "ik"), ("first", "ik_max_word"), ("second", "ik_max_word"), ("second", "ik")]
    assert len(calls) == 8
    before = {str(p): p.read_bytes() for p in tmp_path.rglob("*.json")}
    assert await run_pairs(**kwargs) == first
    assert len(calls) == 8
    assert {str(p): p.read_bytes() for p in tmp_path.rglob("*.json")} == before


async def test_unknown_query_is_not_resubmitted_or_dropped_from_denominator(tmp_path):
    from evals.paired_graph_command import operation_binding, run_pairs
    from evals.gold_rebinding import _write
    cases = [SimpleNamespace(case_id="case")]
    _write(tmp_path / "ik" / "cases" / "case" / "query.json", {
        "binding": operation_binding("binding", "case", "ik", "query"), "status": "running"})
    async def query(case, arm, directory):
        assert arm != "ik"
        raise TimeoutError()
    async def score(*args):
        pytest.fail("missing query result must not be scored")
    records = await run_pairs(cases=cases, root=tmp_path, binding="binding", query=query, score=score)
    assert len(records) == 2
    assert records[0]["query"]["status"] == "running"
    assert records[1]["query"]["status"] == "failed"


async def test_copied_query_from_other_arm_is_rejected_before_paid_call(tmp_path):
    from evals.paired_graph_command import run_pairs
    cases = [SimpleNamespace(case_id="case")]
    async def query(case, arm, directory):
        return {"actual": arm}
    async def score(case, arm, trace, directory):
        return {"answer": trace["actual"]}
    kwargs = dict(cases=cases, root=tmp_path, binding="binding", query=query, score=score)
    await run_pairs(**kwargs)
    from evals.gold_rebinding import _json, _write
    _write(tmp_path / "ik_max_word" / "cases" / "case" / "query.json",
           _json(tmp_path / "ik" / "cases" / "case" / "query.json"))
    with pytest.raises(ValueError, match="binding"):
        await run_pairs(**kwargs)
