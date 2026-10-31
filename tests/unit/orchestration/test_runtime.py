from importlib import import_module

from agentic_rag.orchestration.contracts import CandidateResult
from tests.unit.orchestration.test_store import dispatch, finish_candidate, prepare_and_grade  # noqa: F401
from tests.unit.orchestration.test_evidence import ranked


async def complete(store, stage, payload):
    work = (await store.list_ready(stage))[0]
    claim = await store.claim(work["id"], f"worker-{stage}")
    result = await store.submit_result(claim, payload)
    await store.receive_result(result)
    await import_module("agentic_rag.orchestration.coordinator").Coordinator(store).advance("run-1")
    return result


async def test_fast_lookup_has_no_child_task_or_supervisor_and_freezes_after_grade(store):
    await complete(store, "route", {"kind": "fact", "reason": "one lookup", "requires_calculation": False, "multiple_targets": False})
    state = await store.read_run("run-1")
    assert state["data"]["tasks"] == {}
    assert not await store.list_ready("supervisor")
    result = CandidateResult(origin_key="fast", origin="fast_pipeline", criterion_ids=("answer",), candidates=(ranked(),))
    await complete(store, "fast_retrieval", result.model_dump(mode="json"))
    await prepare_and_grade_fast(store)
    state = await store.read_run("run-1")
    assert state["data"]["answer"]["input_hash"]
    assert state["data"]["tasks"] == {}
    assert len(await store.list_ready("answer")) == 1
    assert not await store.list_ready("supervisor")


async def prepare_and_grade_fast(store, *, sufficient=True):
    from agentic_rag.orchestration.evidence import EvidencePreparer
    from agentic_rag.orchestration.contracts import Criterion
    from tests.unit.orchestration.test_evidence import SCOPE, SNAPSHOT, Hydrator
    work = (await store.list_ready("prepare"))[0]
    frozen = await store.read_snapshot(work["payload"]["snapshot_id"])
    raw = (await store.read_result(frozen["members"][0]["results"][0]["result_id"]))["payload"]
    prepared = await EvidencePreparer(Hydrator()).prepare([CandidateResult.model_validate(raw)], [Criterion(id="answer", description="Compare revenues")], SCOPE, SNAPSHOT)
    await complete(store, "prepare", prepared.model_dump(mode="json"))
    ref = next(iter(prepared.packed.manifest))
    await complete(store, "grade", {"bundle_hash": prepared.bundle_hash, "decision": "sufficient" if sufficient else "insufficient",
        "coverage": [{"criterion_id": "answer", "status": "supported" if sufficient else "unsupported", "evidence_refs": [ref]}]})


async def test_fast_insufficiency_promotes_same_run_with_evidence_and_spent_budget(store):
    await complete(store, "route", {"kind": "fact", "reason": "lookup", "requires_calculation": False, "multiple_targets": False})
    item = (await store.list_ready("fast_retrieval"))[0]
    claim = await store.claim(item["id"], "fast")
    await store.consume_budget(claim, "queries")
    result_id = await store.submit_result(claim, CandidateResult(origin_key="fast", origin="fast_pipeline", criterion_ids=("answer",), candidates=(ranked(),)).model_dump(mode="json"))
    await store.receive_result(result_id)
    await import_module("agentic_rag.orchestration.coordinator").Coordinator(store).advance("run-1")
    await prepare_and_grade_fast(store, sufficient=False)
    state = await store.read_run("run-1")
    assert state["data"]["route"] == "research"
    assert state["data"]["budget"]["queries"]["used"] == 1
    assert state["data"]["groups"]["fast"]["prepared_ref"]
    assert len(await store.list_ready("supervisor")) == 1
    assert not await store.list_ready("fast_retrieval")
