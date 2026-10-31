"""Budgeted same-task supplementation keeps private context and successful peers."""

from __future__ import annotations

from datetime import timedelta
from typing import TYPE_CHECKING, Any

from sqlalchemy import insert, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from agentic_rag.orchestration import schema
from agentic_rag.orchestration.contracts import Gap, content_hash
from agentic_rag.orchestration.dispatch import enqueue_task
from agentic_rag.orchestration.store_sql import lock_run, now, save_run, utc

if TYPE_CHECKING:
    from agentic_rag.orchestration.store import CoordinationStore


async def continue_tasks(
    store: CoordinationStore, run_id: str, decision_id: str, expected_version: int,
    group_key: str, task_keys: list[str], gaps: list[dict[str, Any]],
) -> list[str]:
    async with store.session_factory.begin() as db:
        run = await lock_run(db, run_id)
        return await continue_locked(db, run, decision_id, expected_version, group_key, task_keys, gaps)


async def continue_locked(db: AsyncSession, run: dict[str, Any], decision_id: str, expected_version: int,
                          group_key: str, task_keys: list[str], gaps: list[dict[str, Any]]) -> list[str]:
    from agentic_rag.orchestration.store import BudgetExceeded, StaleClaim
    run_id = run["id"]
    parsed = [Gap.model_validate(gap) for gap in gaps]
    digest = content_hash({"group_key": group_key, "tasks": task_keys, "gaps": gaps})
    old = (await db.execute(select(schema.decisions).where(
        schema.decisions.c.run_id == run_id, schema.decisions.c.decision_id == decision_id,
    ))).mappings().one_or_none()
    if old:
        if old["content_hash"] != digest:
            raise ValueError("decision replay changed its continuation")
        return list(old["work_ids"])
    current = await now(db)
    if (run["status"] != "active" or utc(run["deadline"]) <= current
            or run["data"]["answer"] is not None or run["version"] != expected_version):
        raise StaleClaim("continuation input is stale")
    data = run["data"]
    group = data["groups"].get(group_key)
    if group is None or group.get("origin") == "fast_pipeline":
        raise ValueError("continuation must name a current retrieval group")
    members = {m["task_key"] for m in group["spec"]["members"]}
    if not task_keys or len(set(task_keys)) != len(task_keys) or not set(task_keys) <= members:
        raise ValueError("continuation must name current group members")
    if not parsed or any(gap.criterion_id not in {c["id"] for c in group["spec"]["criteria"]} for gap in parsed):
        raise ValueError("continuation needs concrete current-criterion gaps")
    if group["status"] != "feedback_ready":
        raise ValueError("continuation requires independent grading feedback")
    task_criteria = {key: {c["id"] for c in data["tasks"][key]["spec"]["criteria"]} for key in task_keys}
    assigned: dict[str, list[Gap]] = {key: [] for key in task_keys}
    for gap in parsed:
        targets = set(gap.related_task_keys) if gap.related_task_keys else {
            key for key, criteria in task_criteria.items() if gap.criterion_id in criteria
        }
        if not targets or not targets <= set(task_keys) or any(gap.criterion_id not in task_criteria[key] for key in targets):
            raise ValueError("gap must target selected tasks with the matching criterion")
        for key in targets:
            assigned[key].append(gap)
    if any(not gaps_for_task for gaps_for_task in assigned.values()):
        raise ValueError("every selected task needs an applicable gap")
    account = data["budget"]["assessments"]
    if group["aggregation_version"] >= group["spec"]["max_aggregations"] or account["used"] >= account["limit"]:
        raise BudgetExceeded("group supplementation budget exhausted")
    for key in task_keys:
        task = data["tasks"][key]
        if task.get("no_progress_rounds", 0) >= task["spec"]["budget"]["max_no_progress_rounds"]:
            raise BudgetExceeded("task made no evidence or coverage progress")
        if task["round_id"] >= task["spec"]["budget"]["max_rounds"]:
            raise BudgetExceeded("task round budget exhausted")
        if task["usage"].get("queries", 0) >= task["spec"]["budget"]["max_queries"]:
            raise BudgetExceeded("task query budget exhausted")
    account["used"] += 1
    group["retained_refs"] = list(dict.fromkeys(
        ref for member in group["spec"]["members"] for ref in data["tasks"][member["task_key"]]["result_refs"]
    ))
    group["aggregation_version"] += 1
    group["snapshot_id"], group["status"], group["grade_ref"] = None, "collecting", None
    group["prepared_ref"], group["accepted_grade_ref"], group["acceptance"] = None, None, "pending"
    data["accepted"].pop(group_key, None)
    group["join_deadline"] = min(current + timedelta(seconds=group["spec"]["join_timeout_seconds"]),
                                 utc(run["deadline"]) - timedelta(seconds=5)).isoformat()
    work = []
    for key in task_keys:
        task = data["tasks"][key]
        task["round_id"] += 1
        task["gaps"] = [gap.model_dump(mode="json") for gap in assigned[key]]
        task["acceptance"], task["accepted_ref"] = "pending", None
        group["expected"][key] = {field: task[field] for field in ("revision", "input_version", "round_id")}
        # Fence older attempts before the new round becomes claimable.
        old_items = (await db.execute(select(schema.work_items).where(
            schema.work_items.c.run_id == run_id, schema.work_items.c.status.in_(("queued", "running", "retry_wait")),
        ))).mappings().all()
        for item in old_items:
            if item["payload"].get("task_key") == key:
                await db.execute(update(schema.work_items).where(schema.work_items.c.id == item["id"]).values(status="cancelled", lease_until=None))
        work_id = await enqueue_task(db, run, key)
        if work_id:
            work.append(work_id)
    await db.execute(insert(schema.decisions).values(
        run_id=run_id, decision_id=decision_id, input_version=expected_version,
        content_hash=digest, work_ids=work,
    ))
    await save_run(db, run)
    return work
