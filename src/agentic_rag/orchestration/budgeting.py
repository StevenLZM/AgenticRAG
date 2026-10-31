"""Durable conservative reservations cover provider calls, repairs and crashes."""
from typing import Any

from agentic_rag.orchestration.store_sql import save_run
from agentic_rag.runtime.ids import new_id


async def reserve(store: Any, claim: Any, max_output: int = 2048, *, max_attempts: int = 2) -> tuple[str, int, int]:
    from agentic_rag.orchestration.store import BudgetExceeded
    async with store.session_factory.begin() as db:
        run, item = await store._verified(db, claim)
        data = run["data"]
        accounts = [data["budget"]]
        task = data["tasks"].get(item["payload"].get("task_key"))
        remaining_output = data["budget"]["output_tokens"]["limit"] - data["budget"]["output_tokens"]["used"]
        remaining_calls = data["budget"]["model_calls"]["limit"] - data["budget"]["model_calls"]["used"]
        if task:
            remaining_output = min(remaining_output, task["spec"]["budget"]["max_output_tokens"] - task["usage"].get("output_tokens", 0))
            remaining_calls = min(remaining_calls, task["spec"]["budget"]["max_model_calls"] - task["usage"].get("model_calls", 0))
        calls = min(max_attempts, remaining_calls)
        output = min(max_output, remaining_output // max(1, calls))
        if calls < 1 or output < 128:
            raise BudgetExceeded("provider reservation exhausted")
        amounts = {"model_calls": calls, "output_tokens": output * calls}
        for account in accounts:
            for kind, amount in amounts.items():
                account[kind]["used"] += amount
        if task:
            for kind, amount in amounts.items():
                task["usage"][kind] = task["usage"].get(kind, 0) + amount
        reservation_id = new_id()
        data.setdefault("reservations", {})[reservation_id] = {"work_id": claim.work_id, "task_key": item["payload"].get("task_key"), **amounts}
        await save_run(db, run, changed=False)
        return reservation_id, output, calls


async def settle(store: Any, claim: Any, reservation_id: str, calls: int, output_tokens: int | None) -> None:
    async with store.session_factory.begin() as db:
        run, _ = await store._verified(db, claim)
        reservation = run["data"].get("reservations", {}).get(reservation_id)
        if reservation is None:
            return
        if reservation["work_id"] != claim.work_id or not 1 <= calls <= reservation["model_calls"]:
            raise ValueError("provider usage exceeded reservation")
        if output_tokens is None:
            # A completed response with unreported usage is not free. Retain
            # the per-call upper bound while refunding known unused attempts.
            output_tokens = reservation["output_tokens"] * calls // reservation["model_calls"]
            run["data"].setdefault("estimated_usage", []).append({"work_id": claim.work_id, "output_tokens": output_tokens})
        if not 0 <= output_tokens <= reservation["output_tokens"]:
            raise ValueError("provider usage exceeded reservation")
        task = run["data"]["tasks"].get(reservation["task_key"])
        for kind, used in (("model_calls", calls), ("output_tokens", output_tokens)):
            refund = reservation[kind] - used
            run["data"]["budget"][kind]["used"] -= refund
            if task:
                task["usage"][kind] -= refund
        del run["data"]["reservations"][reservation_id]
        await save_run(db, run, changed=False)
