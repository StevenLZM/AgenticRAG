"""Invariant tests for the supervisor-owned research Todo reducer."""

from __future__ import annotations

import pytest

from agentic_rag.query.todos import (
    InvalidTodoTransition,
    TodoItem,
    TodoReducer,
    TodoUpdate,
)


def _todo(**changes: object) -> TodoItem:
    values: dict[str, object] = {
        "id": "todo-1",
        "title": "Find the contract notice period",
        "owner": "researcher-1",
        "dependencies": (),
    }
    values.update(changes)
    return TodoItem(**values)


def test_completed_todo_requires_evidence_or_result_reference() -> None:
    item = _todo(status="in_progress")

    with pytest.raises(InvalidTodoTransition, match="evidence"):
        TodoReducer.apply(item, TodoUpdate(status="completed"), actor="researcher-1")


def test_only_owner_or_supervisor_can_change_a_todo() -> None:
    item = _todo(status="pending")

    with pytest.raises(InvalidTodoTransition, match="owner"):
        TodoReducer.apply(item, TodoUpdate(status="blocked"), actor="researcher-2")

    changed = TodoReducer.apply(item, TodoUpdate(status="blocked"), actor="supervisor")

    assert changed.status == "blocked"


def test_dependencies_must_complete_before_work_starts() -> None:
    dependent = _todo(id="todo-2", dependencies=("todo-1",))
    prerequisite = _todo(id="todo-1", status="pending")

    with pytest.raises(InvalidTodoTransition, match="dependencies"):
        TodoReducer.apply(
            dependent,
            TodoUpdate(status="in_progress"),
            actor="researcher-1",
            todos=(prerequisite, dependent),
        )


def test_rejects_dependency_cycles_and_duplicate_ids() -> None:
    first = _todo(id="todo-1", dependencies=("todo-2",))
    second = _todo(id="todo-2", dependencies=("todo-1",))

    with pytest.raises(InvalidTodoTransition, match="cycle"):
        TodoReducer.validate((first, second))

    with pytest.raises(InvalidTodoTransition, match="duplicate"):
        TodoReducer.validate((_todo(), _todo()))


def test_todo_order_and_ids_remain_deterministic() -> None:
    items = TodoReducer.create(
        ["Second research question", "First research question"], owner="researcher-1"
    )

    assert [item.id for item in items] == ["todo-1", "todo-2"]
    assert [item.title for item in items] == [
        "Second research question",
        "First research question",
    ]


def test_append_todos_uses_stable_non_colliding_ids() -> None:
    """Appending model-proposed work must not replace an existing Todo."""
    existing = TodoReducer.create(["root"], owner="supervisor")

    result = TodoReducer.append(
        existing,
        ["retrieve policy", "check exception"],
        owner="supervisor",
    )

    assert [item.id for item in result] == ["todo-1", "todo-2", "todo-3"]


def test_append_todos_fills_sparse_ids_without_collision() -> None:
    """Checkpoint recovery may retain a sparse but otherwise valid Todo collection."""
    existing = (
        _todo(id="todo-1"),
        _todo(id="todo-3"),
    )

    result = TodoReducer.append(existing, ["fill gap", "continue"], owner="supervisor")

    assert [item.id for item in result] == ["todo-1", "todo-3", "todo-2", "todo-4"]
