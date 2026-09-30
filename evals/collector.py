"""Private, scoped runtime observation projection. Missing evidence is an error."""
import sqlite3
import asyncio
from pathlib import Path

from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from sqlalchemy import select
from agentic_rag.persistence.repositories import agent_runs

from agentic_rag.retrieval.models import RetrievalObservation
from agentic_rag.runtime.models import RuntimeConfigSnapshot
from agentic_rag.query.public_answer import project_public_answer


class RuntimeCollector:
    """Read only explicitly scoped Run fields and its local checkpoint."""

    def __init__(self, engine, checkpoint_path: Path):
        self.engine = engine
        self.checkpoint_path = checkpoint_path

    async def collect(self, *, run_id, user_id, snapshot_id, question):
        fields = ("id", "user_id", "runtime_config_snapshot_id", "question", "status", "answer", "route", "checkpoint_thread_id")
        async with self.engine.connect() as connection:
            run = (await connection.execute(select(*(agent_runs.c[name] for name in fields)).where(
                agent_runs.c.id == run_id, agent_runs.c.user_id == user_id,
                agent_runs.c.runtime_config_snapshot_id == snapshot_id, agent_runs.c.question == question,
            ))).mappings().one_or_none()
        if run is None:
            raise ValueError("scoped run unavailable")
        if not run["checkpoint_thread_id"].startswith(f"query:{user_id}:"):
            raise ValueError("checkpoint thread scope mismatch")
        state = await asyncio.to_thread(read_checkpoint, self.checkpoint_path, run["checkpoint_thread_id"])
        return project_verified_state(run, state, run_id=run_id, user_id=user_id,
                                       snapshot_id=snapshot_id, question=question)

    async def collect_v2(self, *, run_id, user_id, snapshot_id, question):
        from evals.trace_v2 import project_case_trace
        fields = ("id", "user_id", "runtime_config_snapshot_id", "question", "status", "answer", "route",
                  "checkpoint_thread_id", "error_code", "created_at", "started_at", "finished_at")
        async with self.engine.connect() as connection:
            run = (await connection.execute(select(*(agent_runs.c[name] for name in fields)).where(
                agent_runs.c.id == run_id, agent_runs.c.user_id == user_id,
                agent_runs.c.runtime_config_snapshot_id == snapshot_id, agent_runs.c.question == question,
            ))).mappings().one_or_none()
        if run is None or not run["checkpoint_thread_id"].startswith(f"query:{user_id}:eval-v2-"):
            raise ValueError("scoped isolated evaluation run unavailable")
        state = await asyncio.to_thread(read_checkpoint, self.checkpoint_path, run["checkpoint_thread_id"])
        trace = project_case_trace(dict(run), state, snapshot_id=snapshot_id)
        if trace["memory_policy"] != "disabled":
            raise ValueError("evaluation was not memory isolated")
        if run["created_at"] and run["finished_at"]:
            trace["total_seconds"] = (run["finished_at"] - run["created_at"]).total_seconds()
        return trace


def read_checkpoint(path: Path, thread_id: str):
    """Read one local runtime checkpoint without creating/migrating its database."""
    connection = sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)
    try:
        row = connection.execute(
            "SELECT type, checkpoint FROM checkpoints WHERE thread_id=? AND checkpoint_ns='' "
            "ORDER BY checkpoint_id DESC LIMIT 1", (thread_id,),
        ).fetchone()
        if row is None or len(row[1]) > 20_000_000:
            raise ValueError("checkpoint unavailable or oversized")
        checkpoint = JsonPlusSerializer(pickle_fallback=False).loads_typed((row[0], row[1]))
        state = checkpoint.get("channel_values")
        if not isinstance(state, dict):
            raise ValueError("checkpoint has no state")
        return state
    finally:
        connection.close()


def project_verified_state(run, state, *, run_id, user_id, snapshot_id, question):
    if (run["id"] != run_id or run["user_id"] != user_id
            or run["runtime_config_snapshot_id"] != snapshot_id or run["question"] != question
            or state["run_id"] != run_id or state["scope"]["user_id"] != user_id
            or state["request"]["question"] != question):
        raise ValueError("runtime scope, run, question or snapshot mismatch")
    snapshot = RuntimeConfigSnapshot.model_validate(state["runtime_config_snapshot"])
    if snapshot.snapshot_id != snapshot_id:
        raise ValueError("checkpoint snapshot mismatch")
    route = state.get("route", {}).get("route")
    if route not in {"fast_rag", "research", "chat"} or run.get("route") not in {None, route}:
        raise ValueError("actual route is unavailable or inconsistent")
    termination = state.get("termination_reason")
    answer = dict(state.get("answer") or {"status": termination})
    if termination == "completed":
        answer.pop("status", None)
    else:
        answer["status"] = termination
    projected = project_public_answer(answer,
        evidence_parent_ids=list(dict.fromkeys(item["parent_id"] for item in state.get("evidence", []))),
        route=route, runtime_config_snapshot_id=snapshot_id if termination == "completed" else None,
        require_audited=termination == "completed" and route != "chat")
    persisted = project_public_answer(run.get("answer"))
    if run["status"] != "completed" or projected is None or persisted != projected:
        raise ValueError("terminal public answer differs from checkpoint projection")
    pack = state.get("packed_context", {})
    if route != "chat" and pack.get("index_generation") != snapshot.index_generation:
        raise ValueError("packed context belongs to another generation")
    contexts = [item["content"] for item in pack.get("items", [])]
    if any(not isinstance(content, str) or not content.strip() for content in contexts):
        raise ValueError("invalid observed context")
    rounds = []
    for ordinal, batch in enumerate(state.get("retrieval_batches", [])):
        trace = RetrievalObservation.model_validate(batch.get("observation"))
        if (trace.user_id != user_id or trace.snapshot_id != snapshot_id
                or trace.index_generation != snapshot.index_generation):
            raise ValueError("retrieval observation belongs to another scope")
        if set(trace.stages) != {"dense", "bm25", "rrf", "rerank"}:
            raise ValueError("retrieval stages missing")
        stages = {}
        for name, hits in trace.stages.items():
            if any(hit.user_id != user_id for hit in hits):
                raise ValueError("foreign retrieval hit")
            stages[name] = list(dict.fromkeys(hit.parent_id for hit in hits))
        hydrated = [parent["parent_id"] for parent in batch["parents"]]
        if hydrated != list(trace.hydrated_parent_ids):
            raise ValueError("hydrated ranking differs from batch")
        rounds.append({"ordinal": ordinal, "target_ids": batch.get("target_ids", []),
                       "stages": stages, "parent_ids": hydrated,
                       "observation": trace.model_dump(mode="json")})
    return {"run_id": run_id, "user_id": user_id, "runtime_config_snapshot_id": snapshot_id,
            "answer": dict(run["answer"]),
            "route": route, "contexts": contexts, "rendered_context": pack.get("rendered_context", ""),
            "retrieval_rounds": rounds, "ranking_scope": "single-retrieval" if len(rounds) == 1 else "per-retrieval",
            "ranked_parent_ids": rounds[0]["parent_ids"] if len(rounds) == 1 else None,
            "final_context_parent_ids": [item["parent_id"] for item in pack.get("items", [])]}
