"""Contract enforcement depends on evidence changes, not new IDs or model confidence."""
import pytest

from agentic_rag.orchestration.contracts import DispatchPlan
from agentic_rag.orchestration.store import BudgetExceeded
from tests.unit.orchestration.test_contracts import task_payload, group_payload
from tests.unit.orchestration.test_store import dispatch, finish_candidate, prepare_and_grade
from tests.unit.orchestration.test_evidence import candidate, ranked


async def test_required_source_cannot_be_waived_by_a_sufficient_model_grade(store):
    task = task_payload()
    task["sources"]["document_ids"] = []
    task["sources"]["required_document_ids"] = ["missing-report"]
    work = await store.submit_plan("run-1", "must-use", 0, DispatchPlan.model_validate({"tasks": [task], "groups": [group_payload("r1")]}))
    await finish_candidate(store, work[0], "r1")
    grade = await prepare_and_grade(store)
    await store.accept_grade("run-1", "g1", grade)
    state = (await store.read_run("run-1"))["data"]
    assert state["tasks"]["r1"]["acceptance"] != "accepted"
    assert state["groups"]["g1"]["acceptance"] != "accepted"
    assert state["groups"]["g1"]["contract_gaps"][0]["criterion_id"] == "revenue"
    assert "missing-report" in state["groups"]["g1"]["contract_gaps"][0]["description"]


async def test_new_round_with_identical_evidence_and_coverage_stops_at_no_progress_limit(store):
    task = task_payload()
    task["budget"] = {"max_no_progress_rounds": 1}
    work = await store.submit_plan("run-1", "initial", 0, DispatchPlan.model_validate({"tasks": [task], "groups": [group_payload("r1")]}))
    await finish_candidate(store, work[0], "r1")
    grade = await prepare_and_grade(store, decision="insufficient")
    await store.accept_grade("run-1", "g1", grade)
    state = await store.read_run("run-1")
    gaps = [{"criterion_id": "revenue", "description": "Need comparable revenue"}]
    work = await store.continue_tasks("run-1", "new-decision", state["version"], "g1", ["r1"], gaps)
    claim = await store.claim(work[0], "different-worker")
    repeated = candidate("r1", "revenue", ranked()).model_copy(update={"round_id": 2, "queries": ("a paraphrased query",)})
    await store.submit_result(claim, repeated.model_dump(mode="json"))
    grade = await prepare_and_grade(store, decision="insufficient")
    await store.accept_grade("run-1", "g1", grade)
    state = await store.read_run("run-1")
    assert state["data"]["tasks"]["r1"]["state"] == "blocked"
    with pytest.raises(BudgetExceeded, match="progress"):
        await store.continue_tasks("run-1", "another-id", state["version"], "g1", ["r1"], gaps)


async def test_new_task_id_does_not_reset_the_same_goal_contract(store):
    await dispatch(store)
    state = await store.read_run("run-1")
    replacement = task_payload("r3")
    replacement["objective"] = "  FIND consolidated 2024 revenue with its source.  "
    group = {**group_payload("r3"), "key": "g2"}
    with pytest.raises(ValueError, match="existing.*task|same.*contract"):
        await store.submit_plan("run-1", "fresh-call-id", state["version"], DispatchPlan.model_validate({"tasks": [replacement], "groups": [group]}))


async def ready_distinct_criteria(store):
    from agentic_rag.orchestration.store_sql import lock_run, save_run
    first, second = task_payload(), task_payload("r2")
    second["criteria"] = [{"id": "profit", "description": "Profit"}]
    group = group_payload("r1", "r2")
    group["criteria"].append({"id": "profit", "description": "Profit"})
    await store.submit_plan("run-1", "initial", 0, DispatchPlan.model_validate({"tasks": [first, second], "groups": [group]}))
    async with store.session_factory.begin() as db:
        run = await lock_run(db, "run-1")
        run["data"]["groups"]["g1"]["status"] = "feedback_ready"
        await save_run(db, run)
    return await store.read_run("run-1")


@pytest.mark.parametrize("selected,related,criterion", [
    (["r1"], ["r2"], "revenue"),
    (["r1"], ["foreign"], "revenue"),
    (["r1", "r2"], ["r1"], "revenue"),
    (["r1", "r2"], [], "revenue"),
    (["r2"], ["r2"], "revenue"),
])
async def test_continuation_requires_a_relevant_gap_for_every_selected_task(store, selected, related, criterion):
    before = await ready_distinct_criteria(store)
    with pytest.raises(ValueError, match="gap|criterion"):
        await store.continue_tasks("run-1", "invalid-gap-target", before["version"], "g1", selected,
                                   [{"criterion_id": criterion, "description": "Find missing metric", "related_task_keys": related}])
    assert await store.read_run("run-1") == before


async def test_unassociated_gaps_are_assigned_only_to_matching_task_criteria(store):
    before = await ready_distinct_criteria(store)
    await store.continue_tasks("run-1", "valid-gap-targets", before["version"], "g1", ["r1", "r2"], [
        {"criterion_id": "revenue", "description": "Find missing revenue"},
        {"criterion_id": "profit", "description": "Find missing profit"},
    ])
    tasks = (await store.read_run("run-1"))["data"]["tasks"]
    assert [g["criterion_id"] for g in tasks["r1"]["gaps"]] == ["revenue"]
    assert [g["criterion_id"] for g in tasks["r2"]["gaps"]] == ["profit"]
