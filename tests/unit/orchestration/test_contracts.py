"""Dispatch contracts must reject ambiguous authority and join deadlocks."""

from importlib import import_module

import pytest
from pydantic import ValidationError


def task_payload(key="r1", *, role="retrieval", dependencies=()):
    return {
        "key": key,
        "agent_type": role,
        "objective": "Find consolidated 2024 revenue with its source.",
        "criteria": [{"id": "revenue", "description": "2024 consolidated revenue"}],
        "output": {"schema_name": f"{role}.v1", "max_items": 16},
        "tools": {"allowed": ["retrieve"] if role == "retrieval" else ["read_evidence", "calculate"],
                  "instructions": "Use the allowed tools and preserve source references."},
        "sources": {"document_ids": ["doc-1"], "instructions": "Use the authorized annual report."},
        "boundary": {"in_scope": "2024 consolidated revenue", "excluded_scope": ["business calculations"],
                     "stop_conditions": ["budget exhausted", "no new evidence"],
                     "escalate_when": ["new source scope is required"]},
        "dependencies": list(dependencies),
    }


def group_payload(*members):
    return {
        "key": "g1", "objective": "Prepare comparable revenue evidence",
        "criteria": [{"id": "revenue", "description": "2024 consolidated revenue"}],
        "members": [{"task_key": key, "required": True} for key in members],
    }


@pytest.mark.parametrize("missing", ["objective", "criteria", "output", "tools", "sources", "boundary"])
def test_dispatch_requires_all_delegation_contract_parts(missing):
    module = import_module("agentic_rag.orchestration.contracts")
    raw = task_payload()
    raw.pop(missing)
    with pytest.raises(ValidationError):
        module.TaskSpec.model_validate(raw)


@pytest.mark.parametrize("tools", [["calculate"], ["retrieve", "call_tool"], []])
def test_retrieval_cannot_receive_math_or_generic_tool_authority(tools):
    module = import_module("agentic_rag.orchestration.contracts")
    raw = task_payload()
    raw["tools"]["allowed"] = tools
    with pytest.raises(ValidationError):
        module.TaskSpec.model_validate(raw)


def test_dispatch_rejects_same_group_dependency_barrier():
    module = import_module("agentic_rag.orchestration.contracts")
    validate = import_module("agentic_rag.orchestration.validation").validate_dispatch
    plan = module.DispatchPlan.model_validate({
        "tasks": [task_payload(), task_payload("r2", dependencies=[{"task_key": "r1"}])],
        "groups": [group_payload("r1", "r2")],
    })
    with pytest.raises(ValueError, match="group|barrier"):
        validate(plan)


def test_optional_only_group_cannot_freeze_before_any_result():
    module = import_module("agentic_rag.orchestration.contracts")
    raw = group_payload("r1")
    raw["members"][0]["required"] = False
    with pytest.raises(ValidationError, match="required"):
        module.RetrievalGroupSpec.model_validate(raw)


def test_independent_retrieval_members_are_valid_and_missing_dependency_is_not():
    module = import_module("agentic_rag.orchestration.contracts")
    validate = import_module("agentic_rag.orchestration.validation").validate_dispatch
    plan = module.DispatchPlan.model_validate({
        "tasks": [task_payload(), task_payload("r2")], "groups": [group_payload("r1", "r2")],
    })
    validate(plan)
    bad = plan.model_copy(update={"tasks": (
        module.TaskSpec.model_validate(task_payload(dependencies=[{"task_key": "unknown"}])),
        plan.tasks[1],
    )})
    with pytest.raises(ValueError, match="dependency"):
        validate(bad)


def test_source_priority_cannot_expand_hard_document_scope():
    module = import_module("agentic_rag.orchestration.contracts")
    raw = task_payload()
    raw["sources"]["preferred_document_ids"] = ["outside"]
    with pytest.raises(ValidationError, match="source|document"):
        module.TaskSpec.model_validate(raw)


@pytest.mark.parametrize("prior_analysis", [False, True])
def test_analysis_dependency_must_be_a_retrieval_task(prior_analysis):
    module = import_module("agentic_rag.orchestration.contracts")
    validate = import_module("agentic_rag.orchestration.validation").validate_dispatch
    first = module.TaskSpec.model_validate(task_payload("a1", role="analysis"))
    next_task = task_payload("a2", role="analysis", dependencies=[{"task_key": "a1"}])
    next_task["objective"] = "Compare the earlier analysis with another result"
    plan = module.DispatchPlan.model_validate({"tasks": [next_task] if prior_analysis else [first.model_dump(), next_task]})
    with pytest.raises(ValueError, match="analysis dependencies must name retrieval tasks"):
        validate(plan, {"a1": first} if prior_analysis else None)


@pytest.mark.parametrize("refs", [None, []])
def test_findings_without_sources_are_schema_errors_eligible_for_bounded_model_repair(refs):
    from agentic_rag.orchestration.contracts import Finding
    raw = {"statement": "A claimed inference", "kind": "inference"}
    if refs is not None:
        raw["evidence_refs"] = refs
    with pytest.raises(ValidationError, match="evidence_refs"):
        Finding.model_validate(raw)
