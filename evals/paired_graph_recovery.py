"""Explicit, offline continuation of the single verified terminal projection failure.

This does not resume a model call or invent an answer. Source artifacts remain
immutable, and only the completed research-limit state can be imported.
"""
from __future__ import annotations

from agentic_rag.runtime.models import RuntimeConfigSnapshot
from evals.costs_v2 import provider_requests
from evals.gold_rebinding import _json, file_sha256
from evals.gold_v2_models import canonical_hash
from evals.paired_graph_runtime import arm_snapshot, project_graph_result

REPAIR_FILES = {"evals/collector.py", "evals/paired_graph_runtime.py", "evals/paired_graph_command.py",
                "evals/paired_graph_recovery.py"}


def recover_terminal_trace(directory, *, case, snapshot):
    journal, state = _json(directory / "query.json"), _json(directory / "state.json")
    history = _json(directory / "history.json")
    if (journal.get("status") != "failed" or journal.get("error_type") != "ValueError"
            or state.get("termination_reason") != "research_round_limit" or state.get("next_node") != "end"
            or state["request"]["question"] != case.question or state["scope"]["user_id"] != case.user_id
            or RuntimeConfigSnapshot.model_validate(state["runtime_config_snapshot"]).snapshot_id != snapshot.snapshot_id
            or not history or history[-1].get("termination_reason") != "research_round_limit"
            or history[-1].get("next_node") != "end"):
        raise ValueError("source is not the scoped completed projection-failure case")
    trace = project_graph_result(state, error_code=None)
    budget = _json(directory / "provider-budget.json")
    requests = provider_requests(_json(directory / "events.json"))
    if len(requests["requests"]) != len(budget):
        requests["status"] = "incomplete"
    trace.update(total_seconds=history[-1]["elapsed_seconds"], grade_history=_json(directory / "grades.json"),
        history=history, provider_requests=requests, provider_budget_attempts=len(budget),
        recovery={"kind": "offline_terminal_projection_v1", "new_provider_calls": 0,
                  "timing_source": "last_completed_graph_transition", "source_directory": str(directory.resolve())})
    trace["provider_usage"] = {"status": requests["status"], "cost": None,
        **{key: sum(r[key] for r in requests["requests"]) if requests["status"] == "available" else None
           for key in ("input_tokens", "output_tokens")}}
    return trace


def prepare_continuation(source, current_identity, cases):
    """Prove the approved import changes evaluator bookkeeping only, not the RAG system."""
    from evals.paired_graph_command import operation_binding
    original = _json(source / "experiment.json")
    if "continuation" in original or set(original) != set(current_identity):
        raise ValueError("unsupported continuation source")
    if any(original[k] != current_identity[k] for k in original if k != "implementation"):
        raise ValueError("continuation changes production configuration, cases, gold or budgets")
    old_code, new_code = original["implementation"], current_identity["implementation"]
    if any(old_code[k] != new_code[k] for k in old_code if k != "files"):
        raise ValueError("continuation changes runtime or dependency identity")
    changed = {p for p in old_code["files"].keys() | new_code["files"].keys()
               if old_code["files"].get(p) != new_code["files"].get(p)}
    if not changed or not changed <= REPAIR_FILES:
        raise ValueError("continuation exceeds approved projection repair")
    case, arm = cases[0], "ik"
    directory = source / arm / "cases" / case.case_id
    journals = list(source.glob("*/cases/*/query.json"))
    phases = list((source / "phases").glob("*.json"))
    if (journals != [directory / "query.json"] or not phases
            or any(_json(p)["status"] != "verified" for p in phases)
            or list(source.glob("*/judge-requests.json")) or list(source.glob("*/cases/*/score.json"))):
        raise ValueError("continuation requires exactly one verified unscored source query")
    original_binding = canonical_hash(original)
    if _json(directory / "query.json")["binding"] != operation_binding(original_binding, case.case_id, arm, "query"):
        raise ValueError("source query binding mismatch")
    original_snapshot = arm_snapshot(RuntimeConfigSnapshot.model_validate(original["base_snapshot"]), arm,
                                    original_binding, case.snapshot_id)
    trace = recover_terminal_trace(directory, case=case, snapshot=original_snapshot)
    paths = sorted(source.rglob("*.json"))
    if any(p.is_symlink() or source.resolve() not in p.resolve().parents for p in paths):
        raise ValueError("unsafe continuation source")
    metadata = {"source_directory": str(source.resolve()), "source_experiment_sha256": file_sha256(source / "experiment.json"),
        "source_artifacts": {str(p.relative_to(source)): file_sha256(p) for p in paths},
        "case_id": case.case_id, "arm": arm, "changed_evaluator_files": sorted(changed),
        "runtime_dataset_sha256": original_binding, "recovered_trace_sha256": canonical_hash(trace),
        "new_query_allowance": 39, "replayed_queries": 0}
    return metadata, trace


def assert_source_environment(source, current):
    before = _json(source / "lexical-pre.json")["content"]
    if {k: v for k, v in before.items() if k != "implementation"} != {
            k: v for k, v in current["content"].items() if k != "implementation"}:
        raise ValueError("continuation ES/plugin/environment changed")
