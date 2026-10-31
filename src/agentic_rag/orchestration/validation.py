"""Reject dependency/join deadlocks before durable dispatch."""

from collections.abc import Mapping

from agentic_rag.orchestration.contracts import DispatchPlan, TaskSpec, content_hash


def task_fingerprint(task: TaskSpec) -> str:
    """Stable contract identity excludes fresh task/criterion IDs and budgets.

    This catches equivalent normalized contracts, not arbitrary paraphrases;
    evidence/coverage deltas independently stop unproductive continuations.
    """
    def normalized(value):
        if isinstance(value, str):
            return " ".join(value.casefold().split())
        if isinstance(value, dict):
            return {k: normalized(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [normalized(v) for v in value]
        return value
    raw = task.model_dump(mode="json", exclude={"key", "budget"})
    raw["criteria"] = [{k: v for k, v in criterion.items() if k != "id"} for criterion in raw["criteria"]]
    return content_hash(normalized(raw))


def validate_dispatch(plan: DispatchPlan, existing_tasks: Mapping[str, TaskSpec] | None = None) -> None:
    tasks = dict(existing_tasks or {})
    previous = {task_fingerprint(task) for task in tasks.values()}
    for task in plan.tasks:
        if task.key in tasks:
            raise ValueError(f"duplicate task key: {task.key}")
        if task_fingerprint(task) in previous:
            raise ValueError("same goal contract already has an existing task; continue it without resetting its budget")
        tasks[task.key] = task
    for task in plan.tasks:
        if task.agent_type == "analysis" and any(
            dependency.task_key in tasks and tasks[dependency.task_key].agent_type != "retrieval"
            for dependency in task.dependencies
        ):
            raise ValueError("analysis dependencies must name retrieval tasks; use one analysis task with all required retrieval inputs, not a chain of analysis tasks")
    group_keys: set[str] = set()
    membership: dict[str, str] = {}
    for group in plan.groups:
        if group.key in group_keys:
            raise ValueError("duplicate group key")
        group_keys.add(group.key)
        for member in group.members:
            if member.task_key not in tasks or tasks[member.task_key].agent_type != "retrieval":
                raise ValueError("group member must name a retrieval task")
            if member.task_key in membership:
                raise ValueError("a task may belong to only one active retrieval group")
            membership[member.task_key] = group.key
        criterion_ids = {c.id for c in group.criteria}
        supplied = {c.id for m in group.members for c in tasks[m.task_key].criteria}
        supplied.update(group.retained_fast_criteria)
        if not criterion_ids <= supplied:
            raise ValueError("group criteria must be assigned to its member tasks")
    for task in plan.tasks:
        if task.agent_type == "retrieval" and task.key not in membership:
            raise ValueError("retrieval task needs an explicit group")
    visiting: set[str] = set()
    ancestors: dict[str, set[str]] = {}

    def visit(key: str) -> set[str]:
        if key in visiting:
            raise ValueError("dependency cycle")
        if key in ancestors:
            return ancestors[key]
        visiting.add(key)
        found: set[str] = set()
        for dependency in tasks[key].dependencies:
            if dependency.task_key not in tasks:
                raise ValueError(f"unknown dependency: {dependency.task_key}")
            found.add(dependency.task_key)
            found.update(visit(dependency.task_key))
        visiting.remove(key)
        ancestors[key] = found
        return found

    for key in tasks:
        visit(key)
    for group in plan.groups:
        members = {m.task_key for m in group.members}
        if any(ancestors[key] & members for key in members):
            raise ValueError("same-group dependency creates a join barrier deadlock")
