"""Bounded, restartable supervisor transitions; SQL owns every causal handoff."""
from typing import Any
from copy import deepcopy

from sqlalchemy import insert, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from agentic_rag.orchestration import schema
from agentic_rag.orchestration.assessment import accept_locked
from agentic_rag.orchestration.contracts import CandidateResult, content_hash
from agentic_rag.orchestration.continuation import continue_locked
from agentic_rag.orchestration.dispatch import collect_locked, enqueue_task, submit_plan_locked
from agentic_rag.orchestration.finalization import freeze_answer
from agentic_rag.orchestration.store import CoordinationStore, BudgetExceeded
from agentic_rag.orchestration.store_sql import enqueue, lock_run, now, save_run, utc
from agentic_rag.orchestration.supervisor import RouteDecision, SupervisorDecision


def decision_hash(data: dict[str, Any]) -> str:
    return content_hash({
        "route": data["route"], "tasks": {k: {f: t.get(f) for f in (
            "spec", "revision", "input_version", "round_id", "state", "acceptance", "accepted_ref",
        )} for k, t in data["tasks"].items()},
        "groups": data["groups"], "accepted": data["accepted"], "feedback": data.get("feedback", []),
    })


async def schedule_supervisor(db: AsyncSession, run: dict[str, Any]) -> None:
    data = run["data"]
    if data.get("answer") or data["phase"] == "routing" or data["route"] == "fast":
        return
    active = (await db.execute(select(schema.work_items.c.id).where(
        schema.work_items.c.run_id == run["id"], schema.work_items.c.status.in_(("queued", "running", "retry_wait")),
        schema.work_items.c.stage.in_(("supervisor", "retrieval", "prepare", "grade", "analysis", "analysis_grade")),
    ))).first()
    if active:
        return
    if data["decision_count"] >= 8:
        await freeze_answer(db, run, mode="partial")
        return
    data["decision_count"] += 1
    await enqueue(db, run["id"], "supervisor", f"supervisor:{data['decision_count']}", {"input_hash": decision_hash(data)})


async def fast_group(db: AsyncSession, run: dict[str, Any], result_id: str, payload: dict[str, Any]) -> None:
    candidate = CandidateResult.model_validate(payload)
    if candidate.origin != "fast_pipeline" or candidate.origin_key != "fast" or candidate.criterion_ids != ("answer",):
        raise ValueError("fast pipeline identity mismatch")
    criteria = [{"id": "answer", "description": run["data"]["question"], "required": True}]
    frozen = {"run_id": run["id"], "group_key": "fast", "group_revision": 1, "aggregation_version": 1,
              "members": [{"task_key": "fast", "origin": "fast_pipeline", "required": True, "status": "succeeded",
                           "revision": 1, "input_version": 1, "round_id": 1,
                           "results": [{"result_id": result_id, "hash": content_hash(payload)}]}],
              "criteria": criteria, "objective": run["data"]["question"], "missing_members": [], "collection_completeness": "complete"}
    snapshot_id = content_hash(frozen)
    await db.execute(insert(schema.snapshots).values(id=snapshot_id, run_id=run["id"], group_key="fast", group_revision=1, aggregation_version=1, data=frozen))
    run["data"]["groups"]["fast"] = {"origin": "fast_pipeline", "group_revision": 1, "aggregation_version": 1,
        "status": "preparing", "snapshot_id": snapshot_id, "spec": {"criteria": criteria, "members": []}}
    await enqueue(db, run["id"], "prepare", "prepare:fast:1:1", {"group_key": "fast", "group_revision": 1, "aggregation_version": 1, "snapshot_id": snapshot_id})


class Coordinator:
    def __init__(self, store: CoordinationStore) -> None:
        self.store = store

    async def advance(self, run_id: str) -> None:
        from agentic_rag.orchestration.authorization import SourceAccessRevoked
        try:
            await self._advance(run_id)
        except SourceAccessRevoked:
            await self.store.abort(run_id, "SOURCE_ACCESS_REVOKED")

    async def _advance(self, run_id: str) -> None:
        await self.store.reduce_inbox(run_id)
        async with self.store.session_factory.begin() as db:
            run = await lock_run(db, run_id)
            if run["status"] != "active" or utc(run["deadline"]) <= await now(db):
                return
            data = run["data"]
            if data.get("answer"):
                return
            pending = list(data["pending_results"])
            for result_id in pending:
                row = (await db.execute(select(schema.results).where(schema.results.c.id == result_id))).mappings().one()
                payload, stage = row["payload"], row["stage"]
                if "error" in payload:
                    data.setdefault("feedback", []).append({"stage": stage, "error": payload["error"]})
                    if stage in ("route", "fast_retrieval"):
                        data["route"], data["phase"] = "research", "research"
                    if stage in ("prepare", "grade"):
                        group = data["groups"][payload["work_payload"]["group_key"]]
                        group["status"] = "failed"
                        if data["route"] == "fast":
                            data["route"], data["phase"] = "research", "research"
                elif stage == "route":
                    route = RouteDecision.model_validate(payload)
                    data["requires_analysis"] = route.kind == "analysis" or route.requires_calculation
                    if route.kind == "clarify":
                        await freeze_answer(db, run, mode="partial", clarification=route.clarification or "请明确查询范围。")
                    elif route.fast:
                        data["route"], data["phase"] = "fast", "fast_retrieval"
                        data["budget"]["assessments"]["used"] += 1
                        await enqueue(db, run_id, "fast_retrieval", "fast:1", {})
                    else:
                        data["route"], data["phase"] = "research", "research"
                elif stage == "fast_retrieval":
                    await fast_group(db, run, result_id, payload)
                elif stage == "grade":
                    group_key = next((k for k, g in data["groups"].items() if g.get("grade_ref") == result_id), None)
                    if group_key is not None:
                        await accept_locked(db, run, group_key, result_id)
                        group = data["groups"][group_key]
                        data.setdefault("feedback", []).append({"group_key": group_key, "grade": payload, "acceptance": group["acceptance"]})
                        if group_key == "fast" and data["route"] == "fast":
                            if group["acceptance"] == "accepted":
                                await freeze_answer(db, run)
                            else:
                                data["route"], data["phase"] = "research", "research"
                elif stage == "supervisor":
                    if payload["input_hash"] == decision_hash(data):
                        before = deepcopy(run)
                        try:
                            async with db.begin_nested():
                                decision = SupervisorDecision.model_validate(payload["decision"])
                                if decision.action == "dispatch":
                                    assert decision.plan is not None
                                    await submit_plan_locked(db, run, result_id, run["version"], decision.plan)
                                elif decision.action == "continue":
                                    assert decision.group_key is not None
                                    await continue_locked(db, run, result_id, run["version"], decision.group_key, list(decision.task_keys), [g.model_dump(mode="json") for g in decision.gaps])
                                else:
                                    await freeze_answer(db, run, mode=decision.answer_mode if decision.action == "answer" else "partial", clarification=decision.question)
                        except (ValueError, BudgetExceeded) as error:
                            run.clear()
                            run.update(before)
                            data = run["data"]
                            data.setdefault("feedback", []).append({"decision_rejected": type(error).__name__, "reason": str(error)[:500]})
                elif stage == "analysis_grade":
                    from agentic_rag.orchestration.analysis_assessment import accept_analysis
                    await accept_analysis(db, run, result_id, payload)
                data["pending_results"].remove(result_id)
                if data.get("answer"):
                    break
            if not data.get("answer"):
                for key in data["groups"]:
                    await collect_locked(db, run, key)
                # Re-version downstream analyses if a supplement replaced their
                # accepted inputs; old results cannot satisfy the new dependency.
                for key, task in data["tasks"].items():
                    if task["spec"]["agent_type"] != "analysis":
                        continue
                    refs = {d["task_key"]: data["tasks"][d["task_key"]].get("accepted_ref") for d in task["spec"]["dependencies"]}
                    if task.get("dependency_results") is not None and refs != task["dependency_results"]:
                        task["input_version"] += 1
                        task["dependency_results"] = refs
                        task["acceptance"], task["accepted_ref"], task["state"] = "pending", None, "waiting_dependencies"
                        task["assessment_ref"], task["analysis_result_ref"] = None, None
                        items = (await db.execute(select(schema.work_items).where(schema.work_items.c.run_id == run_id, schema.work_items.c.status.in_(("queued", "running", "retry_wait"))))).mappings()
                        for item in items:
                            if item["payload"].get("task_key") == key:
                                await db.execute(update(schema.work_items).where(schema.work_items.c.id == item["id"]).values(status="cancelled", lease_until=None))
                    if task["state"] == "waiting_dependencies":
                        if await enqueue_task(db, run, key):
                            task["dependency_results"] = refs
                await schedule_supervisor(db, run)
            await save_run(db, run)
