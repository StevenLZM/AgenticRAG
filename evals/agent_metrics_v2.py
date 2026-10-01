"""Evidence-grounded Agent diagnostics, independent from answer similarity."""
from __future__ import annotations

from itertools import combinations


def agent_metrics(trace):
    violations = set()
    for state in trace.get("todo_history", []):
        todos = {t["id"]: t for t in state.get("todos", [])}
        for key, todo in todos.items():
            if todo.get("status") in {"in_progress", "completed"}:
                for dep in todo.get("blocked_by", []):
                    if dep not in todos or todos[dep].get("status") != "completed":
                        violations.add((key, dep))
    queries = [r.get("query") for r in trace.get("retrieval_rounds", []) if r.get("query")]
    groups = {}
    for batch in trace.get("retrieval_rounds", []):
        for target in batch.get("target_ids", []):
            groups.setdefault(target, set()).update(batch.get("stages", {}).get("rerank", []))
    overlaps = [len(a & b) / len(a | b) for a, b in combinations(groups.values(), 2) if a or b]
    return {"research_rounds": trace.get("research_rounds"), "retrieval_calls": trace.get("retrieval_calls"),
            "action_count": len(trace.get("actions", [])),
            "dependency_violations": len(violations) if trace.get("todo_history") is not None else None,
            "repeated_retrieval_rate": (len(queries) - len(set(queries))) / len(queries) if queries else None,
            "subagent_chunk_overlap": sum(overlaps) / len(overlaps) if overlaps else None,
            "budget_exhausted": float(trace.get("outcome") == "research_round_limit"),
            "todo_history_status": "observed" if trace.get("todo_history") is not None else "unknown"}
