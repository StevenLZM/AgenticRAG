"""Atomic plan dispatch and immutable multi-member evidence joins."""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Any

from sqlalchemy import insert, select
from sqlalchemy.ext.asyncio import AsyncSession

from agentic_rag.orchestration import schema
from agentic_rag.orchestration.contracts import DispatchPlan, TaskSpec, content_hash
from agentic_rag.orchestration.store_sql import enqueue, lock_run, now, save_run, utc
from agentic_rag.orchestration.validation import validate_dispatch

if TYPE_CHECKING:
    from agentic_rag.orchestration.store import CoordinationStore


async def enqueue_task(db: AsyncSession, run: dict[str, Any], key: str) -> str | None:
    task = run["data"]["tasks"][key]
    for dependency in task["spec"]["dependencies"]:
        upstream = run["data"]["tasks"][dependency["task_key"]]
        if upstream["acceptance"] != "accepted":
            if not (dependency["allow_partial"] and upstream["acceptance"] == "partial"):
                return None
        required = set(dependency["criterion_ids"])
        if required and not required <= set(upstream.get("supported_criteria", ())):
            return None
    payload = {
        "task_key": key, "revision": task["revision"],
        "input_version": task["input_version"], "round_id": task["round_id"],
        "dependency_results": {d["task_key"]: run["data"]["tasks"][d["task_key"]].get("accepted_ref")
                               for d in task["spec"]["dependencies"]},
    }
    work_id = await enqueue(db, run["id"], task["spec"]["agent_type"],
                            f"task:{key}:{task['revision']}:{task['input_version']}:{task['round_id']}", payload)
    task["state"] = "ready"
    return work_id


async def submit_plan(
    store: CoordinationStore, run_id: str, decision_id: str, expected_version: int, plan: DispatchPlan,
) -> list[str]:
    async with store.session_factory.begin() as db:
        run = await lock_run(db, run_id)
        return await submit_plan_locked(db, run, decision_id, expected_version, plan)


async def submit_plan_locked(db: AsyncSession, run: dict[str, Any], decision_id: str, expected_version: int, plan: DispatchPlan) -> list[str]:
    from agentic_rag.orchestration.store import BudgetExceeded, StaleClaim
    run_id = run["id"]
    old = (await db.execute(select(schema.decisions).where(
        schema.decisions.c.run_id == run_id, schema.decisions.c.decision_id == decision_id,
    ))).mappings().one_or_none()
    digest = content_hash(plan)
    if old:
        if old["content_hash"] != digest:
            raise ValueError("decision replay changed its commands")
        return list(old["work_ids"])
    current = await now(db)
    if run["status"] != "active" or utc(run["deadline"]) <= current or run["data"].get("answer") is not None:
        raise StaleClaim("run no longer accepts dispatch")
    if run["version"] != expected_version:
        raise StaleClaim("supervisor input version changed")
    data = run["data"]
    existing = {key: TaskSpec.model_validate(task["spec"]) for key, task in data["tasks"].items()}
    validate_dispatch(plan, existing)
    if len(existing) + len(plan.tasks) > 24:
        raise BudgetExceeded("run task limit exhausted")
    prior_members = {m["task_key"] for g in data["groups"].values() for m in g["spec"]["members"]}
    if any(m.task_key in prior_members for g in plan.groups for m in g.members):
        raise ValueError("existing group members require continuation, not reassignment")
    if any(g.retained_fast_criteria for g in plan.groups) and not data["groups"].get("fast", {}).get("snapshot_id"):
        raise ValueError("no fast evidence snapshot is available for reuse")
    if any(group.key in data["groups"] for group in plan.groups):
        raise ValueError("existing group requires an explicit new version")
    account = data["budget"]["assessments"]
    if account["used"] + len(plan.groups) > account["limit"]:
        raise BudgetExceeded("group assessment reservation exhausted")
    account["used"] += len(plan.groups)
    for spec in plan.tasks:
        data["tasks"][spec.key] = {
            "spec": spec.model_dump(mode="json"), "revision": 1, "input_version": 1, "round_id": 1,
            "state": "waiting_dependencies", "acceptance": "pending", "result_refs": [],
            "context_ref": None, "usage": {}, "supported_criteria": [], "accepted_ref": None,
        }
    for group_spec in plan.groups:
        data["groups"][group_spec.key] = {
            "spec": group_spec.model_dump(mode="json"), "group_revision": 1, "aggregation_version": 1,
            "status": "collecting", "snapshot_id": None,
            "join_deadline": min(current + timedelta(seconds=group_spec.join_timeout_seconds),
                                 utc(run["deadline"]) - timedelta(seconds=5)).isoformat(),
            "expected": {member.task_key: {
                field: data["tasks"][member.task_key][field] for field in ("revision", "input_version", "round_id")
            } for member in group_spec.members},
            "retained_refs": [],
        }
    work = []
    for spec in plan.tasks:
        work_id = await enqueue_task(db, run, spec.key)
        if work_id:
            work.append(work_id)
    data["phase"] = "research"
    await db.execute(insert(schema.decisions).values(
        run_id=run_id, decision_id=decision_id, input_version=expected_version,
        content_hash=digest, work_ids=work,
    ))
    await save_run(db, run)
    return work

async def collect_group(store: CoordinationStore, run_id: str, group_key: str) -> dict[str, Any] | None:
    async with store.session_factory.begin() as db:
        run = await lock_run(db, run_id)
        return await collect_locked(db, run, group_key)


async def collect_locked(db: AsyncSession, run: dict[str, Any], group_key: str) -> dict[str, Any] | None:
    group = run["data"]["groups"][group_key]
    if group["snapshot_id"]:
        return (await db.execute(select(schema.snapshots.c.data).where(
            schema.snapshots.c.id == group["snapshot_id"],
        ))).scalar_one()
    current = await now(db)
    if run["status"] != "active" or utc(run["deadline"]) <= current:
        return None
    members, pending, missing = [], [], []
    for member in group["spec"]["members"]:
        key = member["task_key"]
        task = run["data"]["tasks"][key]
        expected = group["expected"][key]
        references = []
        current_done = False
        for result_id in task["result_refs"]:
            row = (await db.execute(select(schema.results).where(schema.results.c.id == result_id))).mappings().one()
            payload = row["payload"]
            if (payload.get("task_revision") == expected["revision"]
                    and payload.get("input_version") == expected["input_version"]
                    and (payload.get("round_id") == expected["round_id"] or result_id in group["retained_refs"])):
                references.append({"result_id": result_id, "hash": row["content_hash"]})
                current_done = current_done or payload.get("round_id") == expected["round_id"]
        # Retained earlier rounds enrich a supplement but cannot masquerade as
        # completion of the newly requested round.
        status = "succeeded" if current_done else task["state"] if task["state"] in ("failed", "cancelled", "blocked") else "pending"
        if member["required"] and status == "pending":
            pending.append(key)
        if member["required"] and status != "succeeded":
            missing.append(key)
        members.append({"task_key": key, "required": member["required"], **expected,
                        "status": status, "results": references})
    if pending and current < utc(datetime.fromisoformat(group["join_deadline"])):
        return None
    retained = group["spec"].get("retained_fast_criteria", [])
    if retained:
        baseline = (await db.execute(select(schema.snapshots.c.data).where(
            schema.snapshots.c.id == run["data"]["groups"]["fast"]["snapshot_id"],
        ))).scalar_one()
        members.extend({**member, "criterion_ids": retained} for member in baseline["members"])
    frozen = {
        "run_id": run["id"], "group_key": group_key, "group_revision": group["group_revision"],
        "aggregation_version": group["aggregation_version"], "members": members,
        "criteria": group["spec"]["criteria"], "objective": group["spec"]["objective"],
        "missing_members": missing, "collection_completeness": "partial" if missing else "complete",
    }
    snapshot_id = content_hash(frozen)
    await db.execute(insert(schema.snapshots).values(
        id=snapshot_id, run_id=run["id"], group_key=group_key,
        group_revision=group["group_revision"], aggregation_version=group["aggregation_version"], data=frozen,
    ))
    group["snapshot_id"], group["status"] = snapshot_id, "preparing"
    for member in members:
        if member.get("origin") == "fast_pipeline":
            continue
        task = run["data"]["tasks"][member["task_key"]]
        if task["state"] == "waiting_group":
            task["state"] = "waiting_grade"
    await enqueue(db, run["id"], "prepare",
                  f"prepare:{group_key}:{group['group_revision']}:{group['aggregation_version']}",
                  {"group_key": group_key, "group_revision": group["group_revision"],
                   "aggregation_version": group["aggregation_version"], "snapshot_id": snapshot_id})
    await save_run(db, run)
    return frozen
