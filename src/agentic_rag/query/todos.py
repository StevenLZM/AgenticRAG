"""Deterministic, supervisor-governed Todo contracts for research runs."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


TodoStatus = Literal["pending", "in_progress", "completed", "blocked", "skipped"]
SUPERVISOR_OWNER = "supervisor"
MAX_TODOS = 12
MAX_BLOCKERS_PER_TODO = 8
MAX_DAG_PATH_NODES = 4
MAX_TODO_ATTEMPTS = 2


class InvalidTodoTransition(ValueError):
    """Raised when an untrusted model request violates Todo invariants."""


class TodoItem(BaseModel):
    """JSON-serializable unit of work owned by one agent or the supervisor."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str = Field(min_length=1)
    title: str = Field(min_length=1, max_length=2_000)
    owner: str = Field(min_length=1)
    status: TodoStatus = "pending"
    blocked_by: tuple[str, ...] = ()
    evidence_ids: tuple[str, ...] = ()
    result_ref: str | None = None
    attempts: int = Field(default=0, ge=0, le=MAX_TODO_ATTEMPTS)

    @model_validator(mode="before")
    @classmethod
    def read_legacy_dependencies(cls, value: object) -> object:
        if not isinstance(value, Mapping) or "dependencies" not in value:
            return value
        if "blocked_by" in value:
            raise ValueError("use blocked_by only")
        normalized = dict(value)
        normalized["blocked_by"] = normalized.pop("dependencies")
        return normalized

    @field_validator("id", "title", "owner")
    @classmethod
    def nonblank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("todo fields must not be blank")
        return value.strip()

    @property
    def dependencies(self) -> tuple[str, ...]:
        """Read compatibility for existing in-process callers; never serialized."""
        return self.blocked_by


class TodoDraft(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    key: str = Field(pattern=r"^[a-z][a-z0-9_-]{0,63}$")
    title: str = Field(min_length=1, max_length=2_000)
    blocked_by: tuple[str, ...] = Field(default=(), max_length=MAX_BLOCKERS_PER_TODO)


class TodoDependencyInput(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    todo_id: str
    evidence_ids: tuple[str, ...] = ()
    result_ref: str | None = None


@dataclass(frozen=True, slots=True)
class TodoDagView:
    ready_ids: tuple[str, ...]
    waiting_ids: tuple[str, ...]
    upstream_blocked_ids: tuple[str, ...]
    in_progress_ids: tuple[str, ...]
    completed_ids: tuple[str, ...]
    blocked_ids: tuple[str, ...]
    skipped_ids: tuple[str, ...]


class TodoUpdate(BaseModel):
    """A partial, constrained update requested by the research model."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    status: TodoStatus | None = None
    evidence_ids: tuple[str, ...] | None = None
    result_ref: str | None = Field(default=None, min_length=1)


class TodoReducer:
    """Apply model-proposed Todo changes without granting arbitrary mutation."""

    @classmethod
    def append_drafts(
        cls, todos: tuple[TodoItem, ...], drafts: Sequence[TodoDraft], *, owner: str
    ) -> tuple[TodoItem, ...]:
        cls.validate(todos)
        keys = [draft.key for draft in drafts]
        existing = {todo.id for todo in todos}
        if not drafts or len(keys) != len(set(keys)) or existing.intersection(keys):
            raise InvalidTodoTransition("invalid or ambiguous draft keys")
        additions = cls.append(todos, [draft.title for draft in drafts], owner=owner)[len(todos):]
        key_ids = dict(zip(keys, (todo.id for todo in additions), strict=True))
        resolved = []
        for draft, item in zip(drafts, additions, strict=True):
            if any(ref not in key_ids and ref not in existing for ref in draft.blocked_by):
                raise InvalidTodoTransition("dependency does not exist")
            resolved.append(item.model_copy(update={
                "blocked_by": tuple(key_ids.get(ref, ref) for ref in draft.blocked_by)
            }))
        result = (*todos, *resolved)
        cls.validate(result)
        return result

    @classmethod
    def view(cls, todos: tuple[TodoItem, ...]) -> TodoDagView:
        cls.validate(todos)
        by_id = {todo.id: todo for todo in todos}

        def upstream_failed(todo: TodoItem) -> bool:
            return any(
                by_id[dep].status in {"blocked", "skipped"}
                or upstream_failed(by_id[dep]) for dep in todo.blocked_by
            )

        pending = [todo for todo in todos if todo.status == "pending"]
        failed = tuple(todo.id for todo in pending if upstream_failed(todo))
        ready = tuple(todo.id for todo in pending if todo.attempts < MAX_TODO_ATTEMPTS
                      and all(by_id[dep].status == "completed" for dep in todo.blocked_by))
        return TodoDagView(
            ready_ids=ready,
            waiting_ids=tuple(todo.id for todo in pending if todo.id not in (*failed, *ready)),
            upstream_blocked_ids=failed,
            in_progress_ids=tuple(todo.id for todo in todos if todo.status == "in_progress"),
            completed_ids=tuple(todo.id for todo in todos if todo.status == "completed"),
            blocked_ids=tuple(todo.id for todo in todos if todo.status == "blocked"),
            skipped_ids=tuple(todo.id for todo in todos if todo.status == "skipped"),
        )

    @classmethod
    def recover_interrupted(cls, todos: tuple[TodoItem, ...]) -> tuple[TodoItem, ...]:
        cls.validate(todos)
        return tuple(todo.model_copy(update={"status": "blocked"})
                     if todo.status == "in_progress" or (
                         todo.status == "pending" and todo.attempts >= MAX_TODO_ATTEMPTS
                     ) else todo for todo in todos)

    @classmethod
    def claim_many(cls, todos: tuple[TodoItem, ...], todo_ids: tuple[str, ...]) -> tuple[TodoItem, ...]:
        if not todo_ids or len(set(todo_ids)) != len(todo_ids):
            raise InvalidTodoTransition("todo selection must be unique")
        if not set(todo_ids).issubset(cls.view(todos).ready_ids):
            raise InvalidTodoTransition("todo is not ready")
        return tuple(todo.model_copy(update={"status": "in_progress", "attempts": todo.attempts + 1})
                     if todo.id in todo_ids else todo for todo in todos)

    @classmethod
    def complete(cls, todos: tuple[TodoItem, ...], todo_id: str, *,
                 evidence_ids: tuple[str, ...] = (), result_ref: str | None = None) -> tuple[TodoItem, ...]:
        if not evidence_ids and not result_ref:
            raise InvalidTodoTransition("completed todo requires evidence or result")
        item = next((todo for todo in todos if todo.id == todo_id), None)
        if item is None or item.status != "in_progress":
            raise InvalidTodoTransition("only in-progress todo can complete")
        return cls.apply_many(todos, ((todo_id, TodoUpdate(
            status="completed", evidence_ids=evidence_ids, result_ref=result_ref,
        )),), actor=SUPERVISOR_OWNER)

    @classmethod
    def block_many(cls, todos: tuple[TodoItem, ...], todo_ids: tuple[str, ...]) -> tuple[TodoItem, ...]:
        cls.validate(todos)
        active = {todo.id for todo in todos if todo.status in {"pending", "in_progress"}}
        if len(set(todo_ids)) != len(todo_ids) or not set(todo_ids).issubset(active):
            raise InvalidTodoTransition("invalid active todo selection")
        return tuple(todo.model_copy(update={"status": "blocked"})
                     if todo.id in todo_ids else todo for todo in todos)

    @classmethod
    def apply_agent_updates(
        cls, todos: tuple[TodoItem, ...], updates: tuple[tuple[str, Literal["pending", "skipped"]], ...]
    ) -> tuple[TodoItem, ...]:
        cls.validate(todos)
        if not updates or len({key for key, _ in updates}) != len(updates):
            raise InvalidTodoTransition("invalid todo updates")
        by_id = {todo.id: todo for todo in todos}
        for key, status in updates:
            item = by_id.get(key)
            if item is None:
                raise InvalidTodoTransition("todo does not exist")
            if status == "pending":
                if item.status != "blocked" or item.attempts >= MAX_TODO_ATTEMPTS:
                    raise InvalidTodoTransition("todo cannot be retried")
            elif status != "skipped" or item.status not in {"pending", "blocked"}:
                raise InvalidTodoTransition("todo cannot be skipped")
        result = tuple(todo.model_copy(update={"status": dict(updates)[todo.id]})
                       if todo.id in dict(updates) else todo for todo in todos)
        new_ids = {key for key, status in updates if status == "pending"}
        if new_ids.intersection(cls.view(result).upstream_blocked_ids):
            raise InvalidTodoTransition("todo has terminal dependency")
        return result

    @staticmethod
    def create(titles: list[str], *, owner: str) -> tuple[TodoItem, ...]:
        if not owner.strip():
            raise ValueError("owner must not be blank")
        return tuple(
            TodoItem(id=f"todo-{position}", title=title, owner=owner)
            for position, title in enumerate(titles, start=1)
        )

    @classmethod
    def append(
        cls,
        todos: tuple[TodoItem, ...],
        titles: list[str],
        *,
        owner: str,
    ) -> tuple[TodoItem, ...]:
        """Append supervisor-owned work with deterministic IDs."""
        if not owner.strip():
            raise ValueError("owner must not be blank")
        cls.validate(todos)
        used_ids = {todo.id for todo in todos}
        next_position = 1
        additions: list[TodoItem] = []
        for title in titles:
            while f"todo-{next_position}" in used_ids:
                next_position += 1
            todo_id = f"todo-{next_position}"
            additions.append(TodoItem(id=todo_id, title=title, owner=owner))
            used_ids.add(todo_id)
            next_position += 1
        result = (*todos, *additions)
        cls.validate(result)
        return result

    @staticmethod
    def validate(todos: tuple[TodoItem, ...] | list[TodoItem]) -> None:
        if len(todos) > MAX_TODOS:
            raise InvalidTodoTransition("todo limit exceeded")
        seen: set[str] = set()
        ids = {todo.id for todo in todos}
        for todo in todos:
            if len(todo.blocked_by) > MAX_BLOCKERS_PER_TODO:
                raise InvalidTodoTransition("dependency limit exceeded")
            if len(set(todo.blocked_by)) != len(todo.blocked_by):
                raise InvalidTodoTransition("duplicate dependency")
            if not 0 <= todo.attempts <= MAX_TODO_ATTEMPTS:
                raise InvalidTodoTransition("todo attempt limit reached")
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
        depths: dict[str, int] = {}

        def visit(todo_id: str) -> int:
            if todo_id in visiting:
                raise InvalidTodoTransition("dependency cycle")
            if todo_id in depths:
                return depths[todo_id]
            visiting.add(todo_id)
            depth = 1 + max((visit(dependency) for dependency in adjacency[todo_id]), default=0)
            visiting.remove(todo_id)
            if depth > MAX_DAG_PATH_NODES:
                raise InvalidTodoTransition("dependency path limit exceeded")
            depths[todo_id] = depth
            return depth

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
