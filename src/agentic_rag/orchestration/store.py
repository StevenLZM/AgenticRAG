"""Short SQL transactions own work leases, results and supervisor reception."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any

from sqlalchemy import insert, select, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from agentic_rag.orchestration import schema
from agentic_rag.orchestration.contracts import CandidateResult, DispatchPlan, content_hash
from agentic_rag.orchestration.store_sql import (
    RESULT_STREAM, WAKE_STREAM, enqueue, lock_run, notify, now, save_run, utc,
)
from agentic_rag.runtime.ids import new_id
from agentic_rag.runtime.models import RuntimeConfigSnapshot


class StaleClaim(RuntimeError):
    """A cancelled, expired, replaced or completed execution cannot write."""


class BudgetExceeded(RuntimeError):
    """A durable task or Run budget is exhausted."""


@dataclass(frozen=True)
class WorkClaim:
    work_id: str
    run_id: str
    stage: str
    owner: str
    lease_token: int
    payload: dict[str, Any]
    state: dict[str, Any]
    deadline: datetime
    user_id: str


class CoordinationStore:
    def __init__(self, session_factory: async_sessionmaker[AsyncSession], *, capacity_limits: dict[str, int] | None = None, enforce_source_access: bool = True, max_active_runs: int | None = None) -> None:
        self.session_factory = session_factory
        self.capacity_limits = capacity_limits or {}
        self.enforce_source_access = enforce_source_access
        if max_active_runs is not None and max_active_runs < 1:
            raise ValueError("max_active_runs must be positive")
        self.max_active_runs = max_active_runs

    async def initialize(
        self, *, run_id: str, user_id: str, question: str,
        snapshot: RuntimeConfigSnapshot, timeout_seconds: int = 300, absolute_deadline: datetime | None = None,
    ) -> bool:
        if not run_id or not user_id or not question.strip() or not 1 <= timeout_seconds <= 300:
            raise ValueError("invalid server run context")
        try:
            async with self.session_factory.begin() as db:
                if self.max_active_runs is not None:
                    from agentic_rag.orchestration.capacity import lock_admission
                    await lock_admission(db)
                old = (await db.execute(select(schema.runs).where(schema.runs.c.id == run_id))).mappings().one_or_none()
                if old:
                    if old["user_id"] != user_id or old["data"]["question"] != question:
                        raise ValueError("run identity conflict")
                    return True
                current = await now(db)
                deadline = current + timedelta(seconds=min(timeout_seconds, snapshot.query_run_timeout_seconds))
                if absolute_deadline is not None:
                    deadline = min(deadline, utc(absolute_deadline))
                expired = deadline <= current
                if self.max_active_runs is not None and not expired:
                    active = (await db.execute(select(schema.runs.c.id).where(schema.runs.c.status.in_(("active", "finalizing"))))).all()
                    if len(active) >= self.max_active_runs:
                        return False
                data: dict[str, Any] = {
                    "enforce_source_access": self.enforce_source_access,
                    "question": question, "snapshot": snapshot.model_dump(mode="json"),
                    "phase": "routing", "route": None, "tasks": {}, "groups": {},
                    "accepted": {}, "pending_results": [], "pending_semantic_results": [],
                    "inbox_next_seq": 0, "applied_inbox_seq": 0, "decision_count": 0,
                    # Aggregate accepted evidence and reports have a separate
                    # envelope from the per-bundle retrieval packing budget.
                    "answer_input_char_limit": 1_000_000,
                    "budget": {kind: {"used": 0, "limit": limit} for kind, limit in (
                        ("queries", 24), ("model_calls", 256), ("output_tokens", 1048576), ("assessments", 16),
                    )},
                    "answer": None,
                }
                if expired:
                    data["termination_reason"] = "deadline"
                await db.execute(insert(schema.runs).values(
                    id=run_id, user_id=user_id, version=0, status="failed" if expired else "active", deadline=deadline,
                    data=data, created_at=current,
                ))
                if not expired:
                    await enqueue(db, run_id, "route", "route:1", {})
                return True
        except IntegrityError:
            existing = await self.read_run(run_id)
            if existing["user_id"] != user_id or existing["data"]["question"] != question:
                raise ValueError("run identity conflict")
            return True

    async def read_run(self, run_id: str) -> dict[str, Any]:
        async with self.session_factory() as db:
            row = (await db.execute(select(schema.runs).where(schema.runs.c.id == run_id))).mappings().one()
            return dict(row)

    async def get_work(self, work_id: str) -> dict[str, Any] | None:
        async with self.session_factory() as db:
            row = (await db.execute(select(schema.work_items).where(schema.work_items.c.id == work_id))).mappings().one_or_none()
            return dict(row) if row else None

    async def read_result(self, result_id: str) -> dict[str, Any]:
        async with self.session_factory() as db:
            return dict((await db.execute(select(schema.results).where(schema.results.c.id == result_id))).mappings().one())

    async def read_snapshot(self, snapshot_id: str) -> dict[str, Any]:
        async with self.session_factory() as db:
            return (await db.execute(select(schema.snapshots.c.data).where(schema.snapshots.c.id == snapshot_id))).scalar_one()

    async def list_ready(self, stage: str, limit: int = 100) -> list[dict[str, Any]]:
        async with self.session_factory() as db:
            current = await now(db)
            rows = await db.execute(select(schema.work_items).where(
                schema.work_items.c.stage == stage, schema.work_items.c.status.in_(("queued", "retry_wait")),
                schema.work_items.c.not_before <= current,
            ).order_by(schema.work_items.c.created_at, schema.work_items.c.id).limit(limit))
            return [dict(row) for row in rows.mappings()]

    async def claim(self, work_id: str, owner: str, lease_seconds: int = 30) -> WorkClaim | None:
        if not owner.strip() or not 1 <= lease_seconds <= 300:
            raise ValueError("invalid lease")
        hint = await self.get_work(work_id)
        if hint is None:
            return None
        async with self.session_factory.begin() as db:
            run = await lock_run(db, hint["run_id"])
            item = (await db.execute(select(schema.work_items).where(schema.work_items.c.id == work_id).with_for_update())).mappings().one_or_none()
            if item is None:
                return None
            current = await now(db)
            if run["status"] not in ("active", "finalizing") or utc(run["deadline"]) <= current:
                return None
            if item["status"] not in ("queued", "retry_wait") or utc(item["not_before"]) > current:
                return None
            if item["stage"] in ("retrieval", "fast_retrieval", "analysis"):
                active = (await db.execute(select(schema.work_items.c.id).where(
                    schema.work_items.c.run_id == run["id"],
                    schema.work_items.c.stage.in_(("retrieval", "fast_retrieval", "analysis")),
                    schema.work_items.c.status == "running", schema.work_items.c.lease_until > current,
                ))).all()
                if len(active) >= run["data"]["snapshot"]["max_parallel_subagents_per_run"]:
                    return None
            from agentic_rag.orchestration.capacity import acquire
            if not await acquire(db, dict(item), run["user_id"], self.capacity_limits, lease_seconds):
                return None
            token = item["lease_token"] + 1
            await db.execute(update(schema.work_items).where(schema.work_items.c.id == work_id).values(
                status="running", owner=owner, lease_token=token,
                lease_until=min(current + timedelta(seconds=lease_seconds), utc(run["deadline"])),
                attempt_count=item["attempt_count"] + 1,
            ))
            return WorkClaim(work_id, run["id"], item["stage"], owner, token,
                             dict(item["payload"]), run["data"], utc(run["deadline"]), run["user_id"])

    async def _verified(self, db: AsyncSession, claim: WorkClaim) -> tuple[dict[str, Any], dict[str, Any]]:
        run = await lock_run(db, claim.run_id)
        item = (await db.execute(select(schema.work_items).where(schema.work_items.c.id == claim.work_id))).mappings().one()
        current = await now(db)
        if (run["status"] not in ("active", "finalizing") or utc(run["deadline"]) <= current
                or item["run_id"] != claim.run_id or item["status"] != "running"
                or item["owner"] != claim.owner or item["lease_token"] != claim.lease_token
                or item["lease_until"] is None or utc(item["lease_until"]) <= current):
            raise StaleClaim("execution authority lost")
        key = item["payload"].get("task_key")
        if key is not None:
            task = run["data"]["tasks"].get(key)
            if task is None or any(task[name] != item["payload"][name] for name in ("revision", "input_version", "round_id")):
                raise StaleClaim("task input was superseded")
        for upstream, result_ref in item["payload"].get("dependency_results", {}).items():
            if run["data"]["tasks"][upstream].get("accepted_ref") != result_ref:
                raise StaleClaim("dependency input was superseded")
        group_key = item["payload"].get("group_key")
        if group_key is not None:
            group = run["data"]["groups"].get(group_key)
            if group is None or any(group[name] != item["payload"][name] for name in ("group_revision", "aggregation_version")):
                raise StaleClaim("group version was superseded")
        return run, dict(item)

    async def heartbeat(self, claim: WorkClaim, lease_seconds: int = 30) -> None:
        async with self.session_factory.begin() as db:
            run, _ = await self._verified(db, claim)
            current = await now(db)
            await db.execute(update(schema.work_items).where(schema.work_items.c.id == claim.work_id).values(
                lease_until=min(current + timedelta(seconds=lease_seconds), utc(run["deadline"])),
            ))
            from agentic_rag.orchestration.capacity import renew, requirements
            await renew(db, claim.work_id, claim.lease_token, lease_seconds,
                        expected_resources=set(requirements(claim.stage, claim.user_id, self.capacity_limits)))

    async def consume_budget(self, claim: WorkClaim, kind: str, amount: int = 1) -> int:
        if amount < 1 or kind not in ("queries", "model_calls", "output_tokens"):
            raise ValueError("invalid budget debit")
        async with self.session_factory.begin() as db:
            run, item = await self._verified(db, claim)
            account = run["data"]["budget"][kind]
            if account["used"] + amount > account["limit"]:
                raise BudgetExceeded(f"run {kind} exhausted")
            task_key = item["payload"].get("task_key")
            if task_key:
                task = run["data"]["tasks"][task_key]
                used = task["usage"].get(kind, 0)
                limit_key = {"queries": "max_queries", "model_calls": "max_model_calls"}.get(kind)
                if limit_key and used + amount > task["spec"]["budget"][limit_key]:
                    raise BudgetExceeded(f"task {kind} exhausted")
                task["usage"][kind] = used + amount
            account["used"] += amount
            await save_run(db, run, changed=False)
            return account["limit"] - account["used"]

    async def submit_result(self, claim: WorkClaim, payload: dict[str, Any]) -> str:
        if len(json.dumps(payload, ensure_ascii=False).encode()) > 2_000_000:
            raise ValueError("stage result exceeds the durable artifact bound")
        async with self.session_factory.begin() as db:
            run, item = await self._verified(db, claim)
            result_id = new_id()
            if claim.stage == "grade":
                from agentic_rag.orchestration.assessment import constrain_sources
                payload = await constrain_sources(db, run, item, payload)
            if claim.stage == "retrieval":
                result = CandidateResult.model_validate(payload)
                key = item["payload"]["task_key"]
                task = run["data"]["tasks"][key]
                if (result.origin_key != key or result.origin != "agent_task"
                        or result.task_revision != task["revision"] or result.input_version != task["input_version"]
                        or result.round_id != task["round_id"]):
                    raise ValueError("candidate result identity mismatch")
                task["result_refs"].append(result_id)
                task["context_ref"] = result_id
                task["state"] = "waiting_group"
            await db.execute(insert(schema.results).values(
                id=result_id, run_id=claim.run_id, work_id=claim.work_id, stage=claim.stage,
                payload=payload, content_hash=content_hash(payload), created_at=await now(db),
            ))
            if claim.stage in ("prepare", "grade"):
                from agentic_rag.orchestration.assessment import complete_evidence_stage
                await complete_evidence_stage(db, run, item, result_id, payload)
            if claim.stage in ("analysis", "analysis_grade"):
                from agentic_rag.orchestration.analysis_assessment import complete_analysis
                await complete_analysis(db, run, item, result_id, payload)
            await db.execute(update(schema.work_items).where(schema.work_items.c.id == claim.work_id).values(
                status="succeeded", lease_until=None,
            ))
            from agentic_rag.orchestration.capacity import release
            await release(db, claim.work_id, claim.lease_token)
            await notify(db, claim.run_id, RESULT_STREAM, result_id, f"result:{result_id}")
            await notify(db, claim.run_id, WAKE_STREAM, claim.run_id, f"completed:{result_id}")
            await save_run(db, run)
            return result_id

    async def receive_result(self, result_id: str) -> int:
        async with self.session_factory.begin() as db:
            result = (await db.execute(select(schema.results).where(schema.results.c.id == result_id))).mappings().one()
            run = await lock_run(db, result["run_id"])
            old = (await db.execute(select(schema.inbox.c.seq).where(schema.inbox.c.result_id == result_id))).scalar_one_or_none()
            if old is not None:
                return old
            seq = run["data"]["inbox_next_seq"] + 1
            await db.execute(insert(schema.inbox).values(
                run_id=run["id"], seq=seq, result_id=result_id, received_at=await now(db),
            ))
            run["data"]["inbox_next_seq"] = seq
            await notify(db, run["id"], WAKE_STREAM, run["id"], f"inbox:{run['id']}:{seq}")
            await save_run(db, run)
            return seq

    async def read_inbox(self, run_id: str) -> list[dict[str, Any]]:
        async with self.session_factory() as db:
            run = (await db.execute(select(schema.runs).where(schema.runs.c.id == run_id))).mappings().one()
            rows = await db.execute(select(schema.inbox).where(
                schema.inbox.c.run_id == run_id, schema.inbox.c.seq > run["data"]["applied_inbox_seq"],
            ).order_by(schema.inbox.c.seq))
            return [dict(row) for row in rows.mappings()]

    async def reduce_inbox(self, run_id: str) -> None:
        async with self.session_factory.begin() as db:
            run = await lock_run(db, run_id)
            rows = (await db.execute(select(schema.inbox).where(
                schema.inbox.c.run_id == run_id, schema.inbox.c.seq > run["data"]["applied_inbox_seq"],
            ).order_by(schema.inbox.c.seq))).mappings().all()
            if not rows:
                return
            # Cursor progress means the immutable result remains in durable
            # pending state, not that a model has already accepted it.
            for row in rows:
                if row["result_id"] not in run["data"]["pending_results"]:
                    run["data"]["pending_results"].append(row["result_id"])
                run["data"]["applied_inbox_seq"] = row["seq"]
            await save_run(db, run)

    async def submit_plan(self, run_id: str, decision_id: str, expected_version: int, plan: DispatchPlan) -> list[str]:
        from agentic_rag.orchestration.dispatch import submit_plan
        return await submit_plan(self, run_id, decision_id, expected_version, plan)

    async def collect_group(self, run_id: str, group_key: str) -> dict[str, Any] | None:
        from agentic_rag.orchestration.dispatch import collect_group
        return await collect_group(self, run_id, group_key)

    async def accept_grade(self, run_id: str, group_key: str, result_id: str) -> None:
        from agentic_rag.orchestration.assessment import accept_grade
        await accept_grade(self, run_id, group_key, result_id)

    async def continue_tasks(
        self, run_id: str, decision_id: str, expected_version: int, group_key: str,
        task_keys: list[str], gaps: list[dict[str, Any]],
    ) -> list[str]:
        from agentic_rag.orchestration.continuation import continue_tasks
        return await continue_tasks(self, run_id, decision_id, expected_version, group_key, task_keys, gaps)

    async def cancel(self, run_id: str, user_id: str) -> None:
        async with self.session_factory.begin() as db:
            run = await lock_run(db, run_id)
            if run["user_id"] != user_id:
                raise KeyError(run_id)
            if run["status"] in ("completed", "failed", "cancelled"):
                return
            run["status"] = "cancelled"
            from agentic_rag.orchestration.streaming import terminate_output
            await terminate_output(db, run, "cancelled", cancelled=True)
            await db.execute(update(schema.work_items).where(
                schema.work_items.c.run_id == run_id, schema.work_items.c.status.in_(("queued", "running", "retry_wait")),
            ).values(status="cancelled", lease_until=None))
            await save_run(db, run)

    async def recover(self) -> int:
        from agentic_rag.orchestration.recovery import recover
        return await recover(self)

    async def pending_outbox(self, limit: int = 100) -> list[dict[str, Any]]:
        async with self.session_factory() as db:
            rows = await db.execute(select(schema.outbox).where(
                schema.outbox.c.status == "pending", schema.outbox.c.next_attempt_at <= await now(db),
            ).order_by(schema.outbox.c.created_at, schema.outbox.c.id).limit(limit))
            return [dict(row) for row in rows.mappings()]

    async def dispatched(self, event_id: str) -> None:
        async with self.session_factory.begin() as db:
            await db.execute(update(schema.outbox).where(schema.outbox.c.id == event_id).values(status="dispatched"))

    async def fail(self, claim: WorkClaim, code: str, *, retryable: bool = False) -> None:
        from agentic_rag.orchestration.capacity import release
        from agentic_rag.orchestration.recovery import terminal_failure
        from agentic_rag.orchestration.store_sql import WORK_STREAMS
        async with self.session_factory.begin() as db:
            run, item = await self._verified(db, claim)
            await release(db, claim.work_id, claim.lease_token)
            if retryable and item["attempt_count"] < 3 and claim.stage not in ("answer", "quality"):
                generation = item["dispatch_generation"] + 1
                await db.execute(update(schema.work_items).where(schema.work_items.c.id == claim.work_id).values(
                    status="retry_wait", owner=None, lease_until=None, error_code=code,
                    dispatch_generation=generation, not_before=await now(db) + timedelta(seconds=2 ** item["attempt_count"]),
                ))
                await notify(db, claim.run_id, WORK_STREAMS[claim.stage], claim.work_id, f"work:{claim.work_id}:{generation}")
            else:
                await terminal_failure(db, run, item, code)
            await save_run(db, run)

    async def abort(self, run_id: str, reason: str) -> None:
        async with self.session_factory.begin() as db:
            run = await lock_run(db, run_id)
            if run["status"] not in ("active", "finalizing"):
                return
            from agentic_rag.orchestration.streaming import terminate_output
            await terminate_output(db, run, reason)
            run["status"], run["data"]["termination_reason"] = "failed", reason
            await db.execute(update(schema.work_items).where(schema.work_items.c.run_id == run_id,
                schema.work_items.c.status.in_(("queued", "running", "retry_wait"))).values(status="cancelled", lease_until=None))
            await save_run(db, run)
