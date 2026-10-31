import json

import pytest

from agentic_rag.orchestration.finalization import bound_inputs


def test_oversized_full_answer_requires_explicit_reduction():
    evidence = [{"evidence_id": "e1", "content": "x" * 2000}]
    with pytest.raises(ValueError, match="context budget"):
        bound_inputs(evidence, [], [], {"g1": ["e1"]}, budget=1000, mode="full")


def test_partial_selection_keeps_complete_calculation_support_and_marks_omitted_coverage():
    evidence = [{"evidence_id": "e1", "content": "value is 2"}, {"evidence_id": "e2", "content": "x" * 2000}]
    analysis = {"task_key": "a1", "result": {"findings": [], "calculations": [{"operands": [{"evidence_ref": "e1", "source_kind": "evidence"}]}]}}
    chosen, analyses, assessments, gaps = bound_inputs(evidence, [analysis], [{"group_key": "g1", "acceptance": "accepted"}], {"g1": ["e1", "e2"]}, budget=1000, mode="partial")
    assert [e["evidence_id"] for e in chosen] == ["e1"]
    assert analyses == [analysis]
    assert assessments[0]["acceptance"] == "partial"
    assert gaps[0]["reason"] == "answer_context_budget"
    assert len(json.dumps({"evidence": chosen, "analyses": analyses}, ensure_ascii=False)) <= 1000


@pytest.mark.parametrize("stored_limit", ["default", None, 1000])
async def test_report_freeze_uses_persisted_input_limit_and_preserves_legacy_runs(store, stored_limit):
    from agentic_rag.orchestration.agents import AnalysisDraft
    from agentic_rag.orchestration.coordinator import Coordinator
    from agentic_rag.orchestration.execution import StageExecutor
    from agentic_rag.orchestration.evidence import EvidencePreparer
    from agentic_rag.orchestration.finalization import freeze_answer
    from agentic_rag.orchestration.store_sql import lock_run, save_run
    from tests.unit.orchestration.test_evidence import Hydrator, SNAPSHOT
    from tests.unit.orchestration.test_full_path import Gateway, Retrieval

    statement = ("The cited source provides the consolidated annual revenue. " * 65).strip()

    class LongReport(Gateway):
        async def complete_structured(self, call, schema):
            response = await super().complete_structured(call, schema)
            if issubclass(schema, AnalysisDraft):
                raw = response.value.model_dump(mode="json")
                ref = raw["calculations"][0]["operands"][0]["evidence_ref"]
                raw["findings"] = [{"statement": statement, "evidence_refs": [ref], "criterion_ids": ["revenue"]} for _ in range(4)]
                return response.model_copy(update={"value": schema.model_validate(raw)})
            return response

    async with store.session_factory.begin() as db:
        run = await lock_run(db, "run-1")
        if stored_limit is None:
            run["data"].pop("answer_input_char_limit", None)
        elif stored_limit != "default":
            run["data"]["answer_input_char_limit"] = stored_limit
        await save_run(db, run)
    executor = StageExecutor(store, LongReport(), retrieval=Retrieval(), preparer=EvidencePreparer(Hydrator()))
    coordinator = Coordinator(store)
    for stage in ("route", "supervisor", "retrieval", "retrieval", "prepare", "grade", "analysis", "analysis_grade"):
        claim = await store.claim((await store.list_ready(stage))[0]["id"], stage)
        result = await store.submit_result(claim, await executor(claim))
        await store.receive_result(result)
        await coordinator.advance("run-1")
    before = (await store.read_run("run-1"))["data"]
    assert before["tasks"]["a1"]["acceptance"] == "accepted"
    restored = type(store)(store.session_factory)
    await restored.initialize(run_id="run-1", user_id="user-1", question="Compare revenues", snapshot=SNAPSHOT)

    async def freeze(mode):
        async with restored.session_factory.begin() as db:
            run = await lock_run(db, "run-1")
            await freeze_answer(db, run, mode=mode)
            await save_run(db, run)

    if stored_limit == "default":
        await freeze("full")
    else:
        with pytest.raises(ValueError, match="context budget"):
            await freeze("full")
        await freeze("partial")
    frozen = (await restored.read_run("run-1"))["data"]
    assert frozen.get("answer_input_char_limit") == before.get("answer_input_char_limit")
    inputs = frozen["answer"]["input"]
    if stored_limit == "default":
        report = inputs["analyses"][0]["result"]
        assert [f["statement"] for f in report["findings"][:4]] == [statement] * 4
        assert {r for f in report["findings"] for r in f["evidence_refs"]} <= {e["evidence_id"] for e in inputs["evidence"]}
        assert not inputs["limitations"]
    else:
        assert not inputs["analyses"]
        assert any(g.get("missing_analysis") == "a1" for g in inputs["limitations"])
    await restored.initialize(run_id="run-1", user_id="user-1", question="Compare revenues", snapshot=SNAPSHOT)
    assert (await restored.read_run("run-1"))["data"]["answer"] == frozen["answer"]
