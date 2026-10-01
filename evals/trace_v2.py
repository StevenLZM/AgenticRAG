"""Scoped projections of actual checkpoints, not expected routes or TopK guesses."""
from __future__ import annotations

from agentic_rag.query.public_answer import project_public_answer
from agentic_rag.retrieval.models import RetrievalObservation
from agentic_rag.runtime.models import RuntimeConfigSnapshot


def project_case_trace(run: dict, state: dict, *, snapshot_id: str) -> dict:
    snapshot = RuntimeConfigSnapshot.model_validate(state["runtime_config_snapshot"])
    if (run["id"] != state["run_id"] or run["user_id"] != state["scope"]["user_id"]
            or run["question"] != state["request"]["question"] or snapshot.snapshot_id != snapshot_id
            or run["runtime_config_snapshot_id"] != snapshot_id):
        raise ValueError("trace scope/run/question/snapshot mismatch")
    route = state.get("route", {}).get("route")
    answer = project_public_answer(run.get("answer"), runtime_config_snapshot_id=snapshot_id)
    outcome = state.get("termination_reason") or run["status"]
    if run["status"] == "completed":
        from evals.collector import project_verified_state
        project_verified_state(run, state, run_id=run["id"], user_id=run["user_id"],
                               snapshot_id=snapshot_id, question=run["question"])
        if route != "chat" and "packed_context" not in state:
            raise ValueError("actual context missing")
    else:
        outcome = run.get("error_code") or run["status"]
    pack = state.get("packed_context")
    if pack is not None and pack.get("index_generation") != snapshot.index_generation:
        raise ValueError("context generation mismatch")
    items = pack.get("items", []) if isinstance(pack, dict) else None
    rounds = []
    for ordinal, batch in enumerate(state.get("retrieval_batches", [])):
        observation = RetrievalObservation.model_validate(batch.get("observation"))
        if (observation.user_id != run["user_id"] or observation.snapshot_id != snapshot_id
                or observation.index_generation != snapshot.index_generation
                or any(h.user_id != run["user_id"] for hits in observation.stages.values() for h in hits)):
            raise ValueError("foreign retrieval observation")
        parent_ids = [p["parent_id"] for p in batch.get("parents", [])]
        if parent_ids != list(observation.hydrated_parent_ids):
            raise ValueError("actual Parent order mismatch")
        degraded = set(batch.get("degraded_components", []))
        stage_status = {name: "failed" if name in degraded else "available" for name in ("dense", "bm25")}
        stage_status.update(rrf="degraded" if degraded & {"dense", "bm25"} else "available",
                            rerank="fallback" if "reranker" in degraded else "available", parent="available")
        rounds.append({"ordinal": ordinal, "query": batch.get("query"), "target_ids": batch.get("target_ids", []),
                       "stages": {k: [h.child_id for h in v] for k, v in observation.stages.items()},
                       "stage_status": stage_status, "degraded_components": sorted(degraded),
                       "parent_ids": parent_ids, "observation": observation.model_dump(mode="json")})
    research = state.get("research", {})
    # Deliberately omit model thinking/planning prompts. Only public actions,
    # immutable evidence refs, and constrained Todo state belong in this trace.
    observations = [{k: o[k] for k in ("kind", "ok", "type", "tool", "todo_id", "todo_ids", "completed_todo_ids", "blocked_todo_ids", "status", "action", "reason", "error_code") if k in o}
                    for o in research.get("observations", []) if isinstance(o, dict)]
    terminal_status = outcome
    # The public terminal status intentionally hides operational errors. The
    # evaluator must retain that distinction, rather than credit an outage as
    # an evidence-grounded refusal.
    if outcome == "cannot_answer" and observations:
        final = observations[-1]
        cause = final.get("error_code") or final.get("reason")
        if cause in {"model_unavailable", "research_evidence_invalid", "research_action_invalid",
                     "subagent_failed", "tool_unavailable"}:
            outcome = cause
    return {"schema_version": 2, "run_id": run["id"], "user_id": run["user_id"],
            "runtime_config_snapshot_id": snapshot_id, "route": route, "outcome": outcome,
            "run_status": run["status"], "question": run["question"], "terminal_status": terminal_status,
            "answer": "\n".join(s.text for s in answer.segments) if answer else "",
            "answer_origin": "model" if answer and answer.segments else "terminal_status",
            "public_answer": answer.model_dump(mode="json") if answer else None,
            "context_items": items, "rendered_context": pack.get("rendered_context") if pack else None,
            "retrieval_rounds": rounds, "ranking_scope": "per-retrieval", "retrieval_calls": len(rounds),
            "research_rounds": state.get("research_attempt_count", 0), "todos": research.get("todos", []),
            "actions": observations, "audit_results": state.get("audit_results", []),
            "errors": [{"code": e.get("code")} for e in state.get("errors", [])],
            "memory_policy": snapshot.evaluation.memory_policy if snapshot.evaluation else "standard",
            "provider_usage": {"status": "unknown", "input_tokens": None, "output_tokens": None, "cost": None},
            "ttft_seconds": None, "total_seconds": None}
