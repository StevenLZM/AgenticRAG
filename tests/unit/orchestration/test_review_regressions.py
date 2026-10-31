from importlib import import_module

import pytest
from sqlalchemy import update

from agentic_rag.orchestration.contracts import CandidateResult, DispatchPlan
from tests.unit.orchestration.test_runtime import complete, prepare_and_grade_fast
from tests.unit.orchestration.test_evidence import ranked, candidate
from tests.unit.orchestration.test_contracts import task_payload, group_payload
from tests.unit.orchestration.test_agents import analysis_task, Execution


async def fast_candidate(store):
    await complete(store, "route", {"kind": "fact", "reason": "lookup", "requires_calculation": False, "multiple_targets": False})
    await complete(store, "fast_retrieval", CandidateResult(origin_key="fast", origin="fast_pipeline", criterion_ids=("answer",), candidates=(ranked(),)).model_dump(mode="json"))


async def test_fast_preparation_failure_schedules_a_bounded_recovery_decision(store):
    await fast_candidate(store)
    work = (await store.list_ready("prepare"))[0]
    claim = await store.claim(work["id"], "prepare")
    await store.fail(claim, "HydrationFailure")
    from tests.unit.orchestration.test_full_path import Broker
    from agentic_rag.orchestration.service import CoordinationService
    await CoordinationService(store, Broker(), worker_id="recovery").tick(public_runs=False)
    assert len(await store.list_ready("supervisor")) == 1
    assert (await store.read_run("run-1"))["data"]["route"] == "research"


async def test_invalid_full_answer_decision_is_consumed_and_replanned(store):
    await complete(store, "route", {"kind": "research", "reason": "many sources", "requires_calculation": False, "multiple_targets": True})
    work = (await store.list_ready("supervisor"))[0]
    result_id = await complete(store, "supervisor", {"input_hash": work["payload"]["input_hash"], "decision": {"action": "answer", "reason": "unsafe premature full answer"}})
    state = await store.read_run("run-1")
    assert result_id not in state["data"]["pending_results"]
    assert state["data"]["feedback"][-1]["decision_rejected"] == "ValueError"
    assert len(await store.list_ready("supervisor")) == 1


async def test_invented_continuation_group_is_rejected_without_poisoning_reception(store):
    await complete(store, "route", {"kind": "research", "reason": "research", "requires_calculation": False, "multiple_targets": True})
    work = (await store.list_ready("supervisor"))[0]
    result_id = await complete(store, "supervisor", {"input_hash": work["payload"]["input_hash"], "decision": {
        "action": "continue", "reason": "invalid group", "group_key": "invented", "task_keys": ["missing"],
        "gaps": [{"criterion_id": "missing", "description": "Missing fact"}],
    }})
    state = (await store.read_run("run-1"))["data"]
    assert result_id not in state["pending_results"]
    assert "current retrieval group" in state["feedback"][-1]["reason"]
    assert len(await store.list_ready("supervisor")) == 1


async def test_user_supplied_numbers_are_computed_inside_analysis_without_retrieval():
    module = import_module("agentic_rag.orchestration.agents")
    from agentic_rag.orchestration.contracts import content_hash
    question = "Calculate 100 + 120"
    operands = [{"value": n, "source_kind": "user_input", "evidence_ref": "query", "bundle_hash": content_hash({"question": question}), "quote": question,
                 "unit": "number", "entity": "user_input", "period": "unspecified"} for n in (100, 120)]
    execution = Execution({"calculations": [{"id": "total", "operation": "sum", "operands": operands, "criterion_ids": ["revenue"]}]})
    execution.question = question
    result = await module.AnalysisAgent().execute(analysis_task(), [], {}, execution, dependency_results={})
    assert result.calculations[0].value == 220
    assert result.calculations[0].operands[0].source_kind == "user_input"
    assert result.status == "completed"
    operands[0]["value"] = 999
    execution.responses = [{"calculations": [{"id": "total", "operation": "sum", "operands": operands, "criterion_ids": ["revenue"]}]}]
    bad = await module.AnalysisAgent().execute(analysis_task(), [], {}, execution, dependency_results={})
    assert bad.status == "needs_input"


async def test_fast_candidates_are_reused_by_gap_only_group_preparation(store):
    await fast_candidate(store)
    await prepare_and_grade_fast(store, sufficient=False)
    task = task_payload("r2")
    task["criteria"] = [{"id": "missing", "description": "Missing metric"}]
    group = group_payload("r2")
    group["criteria"] = [{"id": "answer", "description": "Original lookup"}, {"id": "missing", "description": "Missing metric"}]
    group["retained_fast_criteria"] = ["answer"]
    raw = DispatchPlan.model_validate({"tasks": [task], "groups": [group]})
    state = await store.read_run("run-1")
    work = await store.submit_plan("run-1", "reuse-fast", state["version"], raw)
    claim = await store.claim(work[0], "gap")
    await store.submit_result(claim, candidate("r2", "missing", ranked("p2")).model_dump(mode="json"))
    frozen = await store.collect_group("run-1", "g1")
    assert {m["origin"] for m in frozen["members"] if "origin" in m} == {"fast_pipeline"}
    assert len(frozen["members"]) == 2
    from agentic_rag.orchestration.execution import StageExecutor
    from agentic_rag.orchestration.evidence import EvidencePreparer
    from tests.unit.orchestration.test_evidence import Hydrator
    prepared_work = next(w for w in await store.list_ready("prepare") if w["payload"]["group_key"] == "g1")
    claim = await store.claim(prepared_work["id"], "prepare-new")
    bundle = await StageExecutor(store, None, preparer=EvidencePreparer(Hydrator()))(claim)
    assert {i["parent_id"] for i in bundle["packed"]["items"]} == {"p1", "p2"}
    assert {t for i in bundle["packed"]["items"] for t in i["covered_target_ids"]} >= {"answer", "missing"}
    await store.submit_result(claim, bundle)
    assert not await store.list_ready("fast_retrieval")


async def test_source_revocation_blocks_new_deltas_and_historical_replay(store):
    from datetime import UTC, datetime
    from sqlalchemy import insert
    from agentic_rag.persistence.repositories import documents, document_versions
    from agentic_rag.orchestration import schema
    from agentic_rag.orchestration.streaming import append_delta, read_chunks
    from agentic_rag.orchestration.authorization import SourceAccessRevoked
    from tests.unit.orchestration.test_streaming import frozen
    claim = await frozen(store)
    async with store.session_factory.begin() as db:
        connection = await db.connection()
        await connection.run_sync(lambda conn: documents.metadata.create_all(conn, tables=[documents, document_versions]))
        current = datetime.now(UTC)
        await db.execute(insert(documents).values(id="d-p1", user_id="user-1", source_type="upload", filename="test", mime_type="text/plain", content_hash="hash", status="active", active_version_id=None, source_trust="untrusted", created_at=current, updated_at=current))
        await db.execute(insert(document_versions).values(id="v-p1", document_id="d-p1", version_no=1, parser_version="test", pipeline_version="test", embedding_version="test", index_generation="index-test", status="active", created_at=current))
        run = await store.read_run("run-1")
        run["data"]["enforce_source_access"] = True
        await db.execute(update(schema.runs).where(schema.runs.c.id == "run-1").values(data=run["data"]))
    await append_delta(store, claim, "authorized prefix")
    async with store.session_factory.begin() as db:
        await db.execute(update(documents).where(documents.c.id == "d-p1").values(status="deleted"))
    with pytest.raises(SourceAccessRevoked):
        await append_delta(store, claim, "must not publish")
    with pytest.raises(SourceAccessRevoked):
        await read_chunks(store, "run-1", "user-1", 1, 0)
