"""Durable state, fencing, idempotency and fan-in exercised through SQL."""

from datetime import UTC, datetime, timedelta
from importlib import import_module

import pytest
from sqlalchemy import select, update

from agentic_rag.orchestration.contracts import DispatchPlan
from tests.unit.orchestration.test_contracts import group_payload, task_payload
from tests.unit.orchestration.test_evidence import SNAPSHOT, candidate, ranked


def plan():
    return DispatchPlan.model_validate({
        "tasks": [task_payload(), task_payload("r2")], "groups": [group_payload("r1", "r2")],
    })


async def dispatch(store):
    state = await store.read_run("run-1")
    return await store.submit_plan("run-1", "decision-1", state["version"], plan())


async def finish_candidate(store, work_id, key):
    claim = await store.claim(work_id, f"worker-{key}", 30)
    assert claim is not None
    return await store.submit_result(claim, candidate(key, "revenue", ranked()).model_dump(mode="json"))


async def test_plan_replay_is_idempotent_and_conflicting_replay_fails(store):
    first = await dispatch(store)
    second = await dispatch(store)
    assert first == second
    assert len(await store.list_ready("retrieval")) == 2
    with pytest.raises(ValueError, match="decision|replay"):
        changed = plan().model_copy(update={"tasks": (plan().tasks[0].model_copy(update={"objective": "different"}), plan().tasks[1])})
        await store.submit_plan("run-1", "decision-1", 0, changed)


async def test_restart_keeps_candidate_receipts_but_never_unlocks_raw_evidence(store):
    work = await dispatch(store)
    result_id = await finish_candidate(store, work[0], "r1")
    restored = type(store)(store.session_factory)
    assert await restored.collect_group("run-1", "g1") is None
    state = await restored.read_run("run-1")
    assert state["data"]["tasks"]["r1"]["result_refs"] == [result_id]
    assert state["data"]["tasks"]["r1"]["acceptance"] == "pending"
    assert not await restored.list_ready("analysis")
    await finish_candidate(restored, work[1], "r2")
    snapshot = await restored.collect_group("run-1", "g1")
    assert snapshot["collection_completeness"] == "complete"
    assert len(snapshot["members"]) == 2
    assert await restored.collect_group("run-1", "g1") == snapshot
    assert len(await restored.list_ready("prepare")) == 1


async def test_expired_owner_cannot_overwrite_a_new_claim(store):
    work = await dispatch(store)
    old = await store.claim(work[0], "old", 30)
    schema = import_module("agentic_rag.orchestration.schema")
    async with store.session_factory.begin() as db:
        await db.execute(update(schema.work_items).where(schema.work_items.c.id == old.work_id).values(
            lease_until=datetime.now(UTC) - timedelta(minutes=1),
        ))
    await store.recover()
    new = await store.claim(work[0], "new", 30)
    assert new is not None and new.lease_token > old.lease_token
    with pytest.raises(import_module("agentic_rag.orchestration.store").StaleClaim):
        await store.submit_result(old, candidate("r1", "revenue", ranked()).model_dump(mode="json"))
    await store.submit_result(new, candidate("r1", "revenue", ranked()).model_dump(mode="json"))


async def test_timeout_snapshot_exposes_missing_member_and_ignores_late_append(store):
    work = await dispatch(store)
    await finish_candidate(store, work[0], "r1")
    schema = import_module("agentic_rag.orchestration.schema")
    async with store.session_factory.begin() as db:
        row = (await db.execute(select(schema.runs).where(schema.runs.c.id == "run-1"))).mappings().one()
        data = dict(row["data"])
        data["groups"]["g1"]["join_deadline"] = (datetime.now(UTC) - timedelta(seconds=5)).isoformat()
        await db.execute(update(schema.runs).where(schema.runs.c.id == "run-1").values(data=data))
    frozen = await store.collect_group("run-1", "g1")
    assert frozen["collection_completeness"] == "partial"
    assert frozen["missing_members"] == ["r2"]
    await finish_candidate(store, work[1], "r2")
    assert await store.collect_group("run-1", "g1") == frozen


async def test_result_listener_deduplicates_inbox_and_recovers_pending_semantics(store):
    work = await dispatch(store)
    result_id = await finish_candidate(store, work[0], "r1")
    assert await store.receive_result(result_id) == 1
    assert await store.receive_result(result_id) == 1
    state = await store.read_run("run-1")
    assert state["data"]["inbox_next_seq"] == 1
    inbox = await store.read_inbox("run-1")
    assert [event["result_id"] for event in inbox] == [result_id]
    await store.reduce_inbox("run-1")
    restored = type(store)(store.session_factory)
    state = await restored.read_run("run-1")
    assert state["data"]["applied_inbox_seq"] == 1
    assert await restored.read_inbox("run-1") == []


async def test_budget_debit_survives_restart_and_cannot_be_exceeded(store):
    work = await dispatch(store)
    claim = await store.claim(work[0], "worker", 30)
    for _ in range(3):
        await store.consume_budget(claim, "queries", 1)
    restored = type(store)(store.session_factory)
    with pytest.raises(import_module("agentic_rag.orchestration.store").BudgetExceeded):
        await restored.consume_budget(claim, "queries", 1)
    state = await restored.read_run("run-1")
    assert state["data"]["tasks"]["r1"]["usage"]["queries"] == 3


async def test_cancel_fences_result_writes_and_new_claims(store):
    work = await dispatch(store)
    claim = await store.claim(work[0], "worker", 30)
    await store.cancel("run-1", "user-1")
    assert await store.claim(work[1], "another", 30) is None
    with pytest.raises(import_module("agentic_rag.orchestration.store").StaleClaim):
        await store.submit_result(claim, candidate("r1", "revenue", ranked()).model_dump(mode="json"))


async def test_two_independent_stores_cannot_claim_the_same_work(store):
    import asyncio
    work = await dispatch(store)
    other = type(store)(store.session_factory)
    claims = await asyncio.gather(store.claim(work[0], "a", 30), other.claim(work[0], "b", 30))
    assert sum(claim is not None for claim in claims) == 1


async def prepare_and_grade(store, *, decision="sufficient"):
    from agentic_rag.orchestration.contracts import CandidateResult, Criterion, CriterionCoverage, GradeResult
    from agentic_rag.orchestration.evidence import EvidencePreparer
    from tests.unit.orchestration.test_evidence import Hydrator, SCOPE
    frozen = await store.collect_group("run-1", "g1")
    candidates = []
    for member in frozen["members"]:
        for ref in member["results"]:
            row = await store.read_result(ref["result_id"])
            candidates.append(CandidateResult.model_validate(row["payload"]))
    prepared = await EvidencePreparer(Hydrator()).prepare(
        candidates, [Criterion.model_validate(c) for c in frozen["criteria"]], SCOPE, SNAPSHOT,
    )
    work = (await store.list_ready("prepare"))[0]
    claim = await store.claim(work["id"], "prepare", 30)
    await store.submit_result(claim, prepared.model_dump(mode="json"))
    work = (await store.list_ready("grade"))[0]
    claim = await store.claim(work["id"], "grade", 30)
    refs = tuple(prepared.packed.manifest)
    grade = GradeResult(
        bundle_hash=prepared.bundle_hash, decision=decision,
        coverage=(CriterionCoverage(criterion_id="revenue",
                                    status="supported" if decision == "sufficient" else "unsupported",
                                    evidence_refs=refs if decision == "sufficient" else ()),),
    )
    return await store.submit_result(claim, grade.model_dump(mode="json"))


async def test_grade_acceptance_pins_real_bundle_and_only_then_completes_tasks(store):
    work = await dispatch(store)
    await finish_candidate(store, work[0], "r1")
    await finish_candidate(store, work[1], "r2")
    grade_ref = await prepare_and_grade(store)
    state = await store.read_run("run-1")
    assert state["data"]["tasks"]["r1"]["acceptance"] == "pending"
    await store.accept_grade("run-1", "g1", grade_ref)
    state = await store.read_run("run-1")
    assert state["data"]["groups"]["g1"]["acceptance"] == "accepted"
    assert state["data"]["tasks"]["r1"]["acceptance"] == "accepted"
    assert state["data"]["tasks"]["r1"]["accepted_ref"] == grade_ref


async def test_supplement_reuses_other_member_and_does_not_reset_budget_or_context(store):
    work = await dispatch(store)
    first = await finish_candidate(store, work[0], "r1")
    claim = await store.claim(work[1], "r2", 30)
    await store.consume_budget(claim, "queries", 1)
    second = await store.submit_result(claim, candidate("r2", "revenue", ranked()).model_dump(mode="json"))
    old_grade = await prepare_and_grade(store, decision="insufficient")
    state = await store.read_run("run-1")
    new_work = await store.continue_tasks(
        "run-1", "supplement-1", state["version"], "g1", ["r2"],
        [{"criterion_id": "revenue", "description": "Need consolidated revenue", "related_task_keys": ["r2"]}],
    )
    state = await store.read_run("run-1")
    assert state["data"]["tasks"]["r1"]["result_refs"] == [first]
    assert state["data"]["tasks"]["r2"]["context_ref"] == second
    assert state["data"]["tasks"]["r2"]["usage"]["queries"] == 1
    assert len(new_work) == 1
    assert await store.collect_group("run-1", "g1") is None
    new_claim = await store.claim(new_work[0], "r2-new", 30)
    await store.submit_result(new_claim, candidate("r2", "revenue", ranked()).model_copy(
        update={"round_id": 2},
    ).model_dump(mode="json"))
    frozen = await store.collect_group("run-1", "g1")
    assert frozen["aggregation_version"] == 2
    assert frozen["members"][0]["results"][0]["result_id"] == first
    assert len(frozen["members"][1]["results"]) == 2
    with pytest.raises(import_module("agentic_rag.orchestration.store").StaleClaim):
        await store.accept_grade("run-1", "g1", old_grade)


async def test_recovery_republishes_result_after_transport_loses_dispatched_notice(store):
    work = await dispatch(store)
    result_id = await finish_candidate(store, work[0], "r1")
    for notice in await store.pending_outbox():
        await store.dispatched(notice["id"])
    await store.recover()
    pending = await store.pending_outbox()
    assert any(n["aggregate_id"] == result_id for n in pending)
