"""Deterministic, supervisor-governed Todo contracts for research runs."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


TodoStatus = Literal["pending", "in_progress", "completed", "blocked", "skipped"]
SUPERVISOR_OWNER = "supervisor"


class InvalidTodoTransition(ValueError):
    """Raised when an untrusted model request violates Todo invariants."""


class TodoItem(BaseModel):
    """JSON-serializable unit of work owned by one agent or the supervisor."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str = Field(min_length=1)
    title: str = Field(min_length=1, max_length=2_000)
    owner: str = Field(min_length=1)
    status: TodoStatus = "pending"
    dependencies: tuple[str, ...] = ()
    evidence_ids: tuple[str, ...] = ()
    result_ref: str | None = None


class TodoUpdate(BaseModel):
    """A partial, constrained update requested by the research model."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    status: TodoStatus | None = None
    evidence_ids: tuple[str, ...] | None = None
    result_ref: str | None = Field(default=None, min_length=1)


class TodoReducer:
    """Apply model-proposed Todo changes without granting arbitrary mutation."""

    @staticmethod
    def create(titles: list[str], *, owner: str) -> tuple[TodoItem, ...]:
        if not owner.strip():
            raise ValueError("owner must not be blank")
        return tuple(
            TodoItem(id=f"todo-{position}", title=title, owner=owner)
            for position, title in enumerate(titles, start=1)
        )

    @staticmethod
    def validate(todos: tuple[TodoItem, ...] | list[TodoItem]) -> None:
        seen: set[str] = set()
        ids = {todo.id for todo in todos}
        for todo in todos:
            if todo.id in seen:
                raise InvalidTodoTransition("duplicate todo id")
            seen.add(todo.id)
            unknown = set(todo.dependencies) - ids
            if unknown:
                raise InvalidTodoTransition("dependency does not exist")
            if todo.id in todo.dependencies:
                raise InvalidTodoTransition("dependency cycle")
        adjacency = {todo.id: todo.dependencies for todo in todos}
        visiting: set[str] = set()
        visited: set[str] = set()

        def visit(todo_id: str) -> None:
            if todo_id in visiting:
                raise InvalidTodoTransition("dependency cycle")
            if todo_id in visited:
                return
            visiting.add(todo_id)
            for dependency in adjacency[todo_id]:
                visit(dependency)
            visiting.remove(todo_id)
            visited.add(todo_id)

        for todo_id in sorted(adjacency):
            visit(todo_id)

    @classmethod
    def apply(
        cls,
        item: TodoItem,
        update: TodoUpdate,
        *,
        actor: str,
        todos: tuple[TodoItem, ...] | list[TodoItem] = (),
    ) -> TodoItem:
        if actor != item.owner and actor != SUPERVISOR_OWNER:
            raise InvalidTodoTransition("only the owner or supervisor may mutate a todo")
        cls.validate(tuple(todos) if todos else (item,))
        status = update.status or item.status
        allowed = {
            "pending": {"pending", "in_progress", "blocked", "skipped"},
            "in_progress": {"in_progress", "completed", "blocked", "skipped"},
            "completed": {"completed"},
            "blocked": {"blocked", "pending", "skipped"},
            "skipped": {"skipped"},
        }
        if status not in allowed[item.status]:
            raise InvalidTodoTransition("invalid todo status transition")
        if status == "in_progress":
            by_id = {todo.id: todo for todo in todos}
            if any(by_id[dependency].status != "completed" for dependency in item.dependencies):
                raise InvalidTodoTransition("dependencies must be completed before work starts")
        evidence_ids = update.evidence_ids if update.evidence_ids is not None else item.evidence_ids
        result_ref = update.result_ref if update.result_ref is not None else item.result_ref
        if status == "completed" and not (evidence_ids or result_ref):
            raise InvalidTodoTransition("completed todo requires evidence or a result reference")
        return item.model_copy(
            update={"status": status, "evidence_ids": evidence_ids, "result_ref": result_ref}
        )

    @classmethod
    def apply_many(
        cls,
        todos: tuple[TodoItem, ...],
        updates: tuple[tuple[str, TodoUpdate], ...],
        *,
        actor: str,
    ) -> tuple[TodoItem, ...]:
        cls.validate(todos)
        by_id = {todo.id: todo for todo in todos}
        if len(by_id) != len(todos):
            raise InvalidTodoTransition("duplicate todo id")
        for todo_id, update in updates:
            if todo_id not in by_id:
                raise InvalidTodoTransition("todo does not exist")
            by_id[todo_id] = cls.apply(by_id[todo_id], update, actor=actor, todos=tuple(by_id.values()))
        result = tuple(by_id[todo.id] for todo in todos)
        cls.validate(result)
        return result
