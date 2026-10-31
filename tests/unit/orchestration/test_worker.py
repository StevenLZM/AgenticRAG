import asyncio
from importlib import import_module
import pytest
from sqlalchemy import update

from tests.unit.orchestration.test_store import dispatch  # noqa: F401
from tests.unit.orchestration.test_evidence import candidate, ranked


async def test_two_independent_workers_execute_once_and_join_after_restart(store):
    module = import_module("agentic_rag.orchestration.worker")
    work = await dispatch(store)
    calls = []
    async def execute(claim):
        calls.append(claim.payload["task_key"])
        await asyncio.sleep(0.01)
        return candidate(claim.payload["task_key"], "revenue", ranked()).model_dump(mode="json")
    one = module.AgentWorker(store, execute, worker_id="one", role="retrieval")
    two = module.AgentWorker(type(store)(store.session_factory), execute, worker_id="two", role="retrieval")
    await asyncio.gather(one.run_one(work[0]), two.run_one(work[0]), two.run_one(work[1]))
    assert sorted(calls) == ["r1", "r2"]
    restored = type(store)(store.session_factory)
    await restored.recover()
    assert len(await restored.list_ready("prepare")) == 1


async def test_provider_crash_keeps_reserved_budget_across_new_store(store):
    from agentic_rag.orchestration.budgeting import reserve, settle
    claim = await store.claim((await dispatch(store))[0], "one")
    reservation, output, _ = await reserve(store, claim, 512)
    restored = type(store)(store.session_factory)
    state = await restored.read_run("run-1")
    assert state["data"]["tasks"]["r1"]["usage"]["model_calls"] == 2
    assert state["data"]["budget"]["output_tokens"]["used"] == 1024
    await settle(restored, claim, reservation, 1, 40)
    state = await restored.read_run("run-1")
    assert state["data"]["budget"]["output_tokens"]["used"] == 40
    await settle(restored, claim, reservation, 1, 40)
    assert (await restored.read_run("run-1"))["data"]["budget"]["output_tokens"]["used"] == 40


async def test_supervisor_contract_has_room_for_parallel_plan_and_bounded_deepseek_mode(store):
    from agentic_rag.orchestration.execution import ModelExecution
    from agentic_rag.orchestration.supervisor import RouteDecision
    from agentic_rag.runtime.model_gateway import ModelResponse
    from dataclasses import replace
    claim = await store.claim((await store.list_ready("route"))[0]["id"], "planner")
    claim = replace(claim, state={**claim.state, "snapshot": {**claim.state["snapshot"], "main_model_id": "deepseek-v4-pro"}})
    class Gateway:
        async def complete_structured(self, call, schema):
            assert call.max_output_tokens == 8192
            assert call.thinking == "disabled"
            assert "server-owned rule" in call.messages[0]["content"]
            assert "user text" in call.messages[1]["content"]
            return ModelResponse(value=schema(kind="research", reason="parallel", requires_calculation=False, multiple_targets=True),
                                 requested_model=call.requested_model, actual_model=call.requested_model,
                                 input_tokens=100, output_tokens=50, attempts=1, latency_ms=1)
    await ModelExecution(store, claim, Gateway()).structured("supervisor", {"instructions": "server-owned rule", "question": "user text"}, RouteDecision)
    state = (await store.read_run("run-1"))["data"]
    assert state["budget"]["output_tokens"]["used"] == 50


async def test_complete_repair_usage_preserves_task_budget_for_independent_assessment(store):
    from agentic_rag.orchestration.execution import ModelExecution
    from agentic_rag.orchestration.supervisor import RouteDecision
    from agentic_rag.runtime.model_gateway import ModelResponse
    from agentic_rag.orchestration.budgeting import reserve
    claim = await store.claim((await dispatch(store))[0], "one")
    class Gateway:
        async def complete_structured(self, call, schema):
            assert call.max_output_tokens == 100000
            assert call.max_provider_attempts == 2
            pending = (await store.read_run("run-1"))["data"]
            assert pending["tasks"]["r1"]["usage"]["output_tokens"] == 200000
            return ModelResponse(value=schema(kind="fact", reason="test", requires_calculation=False, multiple_targets=False),
                                 requested_model=call.requested_model, actual_model=call.requested_model,
                                 input_tokens=100, output_tokens=584, attempts=2, latency_ms=1, usage_complete=True)
    await ModelExecution(store, claim, Gateway()).structured("analysis", {}, RouteDecision)
    assert (await store.read_run("run-1"))["data"]["tasks"]["r1"]["usage"]["output_tokens"] == 584
    _, output, _ = await reserve(store, claim)
    assert output >= 128


async def test_default_task_budget_covers_three_rounds_of_planning_and_summary_repairs(store):
    from agentic_rag.orchestration.execution import ModelExecution
    from agentic_rag.orchestration.supervisor import RouteDecision
    from agentic_rag.runtime.model_gateway import ModelResponse
    from tests.unit.orchestration.test_evidence import SNAPSHOT
    claim = await store.claim((await dispatch(store))[0], "research")

    class Gateway:
        async def complete_structured(self, call, schema):
            assert call.max_output_tokens >= 3072
            assert call.max_provider_attempts == 2
            return ModelResponse(value=schema(kind="fact", reason="test", requires_calculation=False, multiple_targets=False),
                                 requested_model="test", actual_model="test", input_tokens=100, output_tokens=6144,
                                 attempts=2, latency_ms=1, usage_complete=True)

    # Each retrieval round needs a query plan and a summary. Allow room for
    # one structured-output repair per stage and retain charges after restart.
    for round_index in range(3):
        execution = ModelExecution(store, claim, Gateway())
        for role in ("retrieval_plan", "retrieval_summary"):
            await execution.structured(role, {}, RouteDecision)
        store = type(store)(store.session_factory)
        await store.initialize(run_id="run-1", user_id="user-1", question="Compare revenues", snapshot=SNAPSHOT)
        state = (await store.read_run("run-1"))["data"]
        assert state["tasks"]["r1"]["usage"]["model_calls"] == (round_index + 1) * 4
        assert state["tasks"]["r1"]["usage"]["output_tokens"] == (round_index + 1) * 12288
        assert state["budget"]["output_tokens"]["used"] == (round_index + 1) * 12288
        assert not state["reservations"]


async def test_global_capacity_is_shared_between_store_instances(store):
    work = await dispatch(store)
    limits = {"model": 1, "tenant_model": 1, "rerank": 1}
    one, two = type(store)(store.session_factory, capacity_limits=limits), type(store)(store.session_factory, capacity_limits=limits)
    claim = await one.claim(work[0], "one")
    assert claim is not None
    assert await two.claim(work[1], "two") is None
    await one.submit_result(claim, candidate("r1", "revenue", ranked()).model_dump(mode="json"))
    assert await two.claim(work[1], "two") is not None


async def test_stolen_capacity_lease_cannot_be_renewed_as_owned_work(store):
    from agentic_rag.orchestration import schema
    from agentic_rag.orchestration.store import StaleClaim
    work = await dispatch(store)
    limited = type(store)(store.session_factory, capacity_limits={"model": 1, "tenant_model": 1, "rerank": 1})
    claim = await limited.claim(work[0], "one")
    async with store.session_factory.begin() as db:
        await db.execute(update(schema.resources).where(schema.resources.c.resource == "global:model").values(owner="another-work", token=99))
    with pytest.raises(StaleClaim, match="capacity"):
        await limited.heartbeat(claim)


async def test_child_concurrency_cap_includes_analysis_alongside_retrieval(store):
    from agentic_rag.orchestration.contracts import DispatchPlan
    from agentic_rag.orchestration.store_sql import lock_run, save_run
    from tests.unit.orchestration.test_contracts import task_payload, group_payload
    async with store.session_factory.begin() as db:
        run = await lock_run(db, "run-1")
        run["data"]["snapshot"]["max_parallel_subagents_per_run"] = 1
        await save_run(db, run)
    state = await store.read_run("run-1")
    plan = DispatchPlan.model_validate({"tasks": [task_payload(), task_payload("a1", role="analysis")], "groups": [group_payload("r1")]})
    work = await store.submit_plan("run-1", "mixed-roles", state["version"], plan)
    assert await store.claim(work[0], "retrieval") is not None
    assert await store.claim(work[1], "analysis") is None


async def test_run_admission_is_shared_and_waiting_does_not_refresh_deadline(store):
    from datetime import UTC, datetime, timedelta
    from agentic_rag.orchestration.store_sql import utc
    from tests.unit.orchestration.test_evidence import SNAPSHOT
    one = type(store)(store.session_factory, max_active_runs=1)
    two = type(store)(store.session_factory, max_active_runs=1)
    deadline = datetime.now(UTC) + timedelta(seconds=100)
    assert not await one.initialize(run_id="run-2", user_id="user-2", question="queued", snapshot=SNAPSHOT, absolute_deadline=deadline)
    await store.cancel("run-1", "user-1")
    results = await asyncio.gather(
        one.initialize(run_id="run-2", user_id="user-2", question="queued", snapshot=SNAPSHOT, absolute_deadline=deadline),
        two.initialize(run_id="run-3", user_id="user-3", question="queued", snapshot=SNAPSHOT, absolute_deadline=deadline),
    )
    assert sum(results) == 1
    winner = "run-2" if results[0] else "run-3"
    assert utc((await store.read_run(winner))["deadline"]) == deadline
    past = datetime.now(UTC) - timedelta(seconds=10)
    assert await two.initialize(run_id="expired", user_id="user-4", question="too late", snapshot=SNAPSHOT, absolute_deadline=past)
    assert (await store.read_run("expired"))["status"] == "failed"


async def test_initial_broker_outage_does_not_prevent_sql_receipt_and_join(store, monkeypatch):
    from redis.exceptions import ConnectionError as RedisConnectionError
    from agentic_rag.orchestration.service import CoordinationService
    from tests.unit.orchestration.test_full_path import Broker
    from tests.unit.orchestration.test_store import finish_candidate
    work = await dispatch(store)
    await finish_candidate(store, work[0], "r1")
    await finish_candidate(store, work[1], "r2")
    service = CoordinationService(store, Broker(), worker_id="recovery")
    async def unavailable():
        raise RedisConnectionError("initial notification transport unavailable")
    monkeypatch.setattr(service, "reconcile_public_runs", unavailable)
    await service.tick()
    state = (await store.read_run("run-1"))["data"]
    assert state["inbox_next_seq"] == 2
    assert state["groups"]["g1"]["snapshot_id"]


async def test_missing_provider_usage_keeps_a_conservative_token_charge(store):
    from agentic_rag.orchestration.budgeting import reserve, settle
    claim = await store.claim((await dispatch(store))[0], "provider")
    reservation, output, _ = await reserve(store, claim, 512)
    await settle(store, claim, reservation, 1, None)
    budget = (await store.read_run("run-1"))["data"]["budget"]
    assert budget["output_tokens"]["used"] == output
    assert budget["model_calls"]["used"] == 1


async def test_last_model_call_can_finish_without_reserving_an_unavailable_repair(store):
    from agentic_rag.orchestration.budgeting import reserve
    from agentic_rag.orchestration.contracts import DispatchPlan
    from agentic_rag.orchestration.execution import ModelExecution
    from agentic_rag.orchestration.store import BudgetExceeded
    from agentic_rag.orchestration.supervisor import RouteDecision
    from agentic_rag.runtime.model_gateway import ModelResponse
    from tests.unit.orchestration.test_contracts import group_payload, task_payload
    from tests.unit.orchestration.test_evidence import SNAPSHOT
    task = {**task_payload(), "budget": {"max_model_calls": 6, "max_output_tokens": 4096}}
    plan = DispatchPlan.model_validate({"tasks": [task], "groups": [group_payload("r1")]})
    state = await store.read_run("run-1")
    work = await store.submit_plan("run-1", "small-budget", state["version"], plan)
    claim = await store.claim(work[0], "last-call")
    await store.consume_budget(claim, "model_calls", 5)
    class Gateway:
        async def complete_structured(self, call, schema):
            assert call.max_provider_attempts == 1
            return ModelResponse(value=schema(kind="fact", reason="test", requires_calculation=False, multiple_targets=False),
                                 requested_model="test", actual_model="test", input_tokens=10, output_tokens=20, attempts=1, latency_ms=1)
    await ModelExecution(store, claim, Gateway()).structured("summary", {}, RouteDecision)
    restored = type(store)(store.session_factory)
    await restored.initialize(run_id="run-1", user_id="user-1", question="Compare revenues", snapshot=SNAPSHOT)
    task = (await restored.read_run("run-1"))["data"]["tasks"]["r1"]
    assert task["spec"]["budget"]["max_model_calls"] == 6
    assert task["spec"]["budget"]["max_output_tokens"] == 4096
    assert task["usage"]["model_calls"] == 6
    with pytest.raises(BudgetExceeded):
        await reserve(restored, claim)
