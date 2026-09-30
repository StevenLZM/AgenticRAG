"""Dependency scheduling invariants, independent of model choices."""
import pytest

from agentic_rag.query.todos import InvalidTodoTransition, TodoItem, TodoReducer


def item(todo_id, **values):
    return TodoItem(id=todo_id, title=todo_id, owner="supervisor", **values)


def test_checkpoint_dependency_name_is_canonical():
    todo = item("b", dependencies=("a",))
    assert todo.model_dump(mode="json")["blocked_by"] == ["a"]
    assert "dependencies" not in todo.model_dump()
    with pytest.raises(ValueError):
        item("b", dependencies=("a",), blocked_by=("a",))


def test_draft_chain_unblocks_only_after_completion():
    from agentic_rag.query.todos import TodoDraft
    todos = TodoReducer.append_drafts((), (
        TodoDraft(key="a", title="locate"),
        TodoDraft(key="b", title="extract", blocked_by=("a",)),
    ), owner="supervisor")
    assert todos[1].blocked_by == ("todo-1",)
    assert TodoReducer.view(todos).ready_ids == ("todo-1",)
    with pytest.raises(InvalidTodoTransition):
        TodoReducer.claim_many(todos, ("todo-1", "todo-2"))
    claimed = TodoReducer.claim_many(todos, ("todo-1",))
    assert claimed[0].attempts == 1
    done = TodoReducer.complete(claimed, "todo-1", evidence_ids=("e1", "e2"))
    assert TodoReducer.view(done).ready_ids == ("todo-2",)
    assert done[0].evidence_ids == ("e1", "e2")


def test_failed_prerequisite_blocks_transitive_descendants_without_mutating_status():
    todos = (item("a", status="blocked"), item("b", blocked_by=("a",)),
             item("c", blocked_by=("b",)))
    assert TodoReducer.view(todos).upstream_blocked_ids == ("b", "c")
    assert todos[1].status == "pending"


def test_execution_retry_is_bounded_and_completed_work_is_immutable():
    todos = (item("a"),)
    for _ in range(2):
        todos = TodoReducer.claim_many(todos, ("a",))
        todos = TodoReducer.block_many(todos, ("a",))
        if todos[0].attempts == 1:
            todos = TodoReducer.apply_agent_updates(todos, (("a", "pending"),))
    with pytest.raises(InvalidTodoTransition):
        TodoReducer.apply_agent_updates(todos, (("a", "pending"),))
    with pytest.raises(InvalidTodoTransition):
        TodoReducer.complete(todos, "a", evidence_ids=("forged",))


@pytest.mark.parametrize("edges", [
    {"a": ("a",)}, {"a": ("b",), "b": ("a",)},
    {"a": ("unknown",)}, {"a": (), "b": ("a", "a")},
    {"a": (), "b": ("a",), "c": ("b",), "d": ("c",), "e": ("d",)},
])
def test_invalid_dag_rejected(edges):
    with pytest.raises(ValueError):
        TodoReducer.validate(tuple(item(key, blocked_by=value) for key, value in edges.items()))


def test_duplicate_draft_keys_are_atomic():
    from agentic_rag.query.todos import TodoDraft
    existing = (item("todo-1"),)
    with pytest.raises(InvalidTodoTransition):
        TodoReducer.append_drafts(existing, (
            TodoDraft(key="a", title="one"), TodoDraft(key="a", title="two"),
        ), owner="supervisor")
    assert len(existing) == 1
