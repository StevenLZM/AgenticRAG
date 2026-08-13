"""Bounded one-action-at-a-time research loop for the complex query route."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Annotated, Any, Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field, RootModel, TypeAdapter, ValidationError

from agentic_rag.query.context import ContextBuilder
from agentic_rag.query.evidence_builder import EvidenceBuilder
from agentic_rag.query.state import QueryState, scope_from_state, snapshot_from_state
from agentic_rag.query.subagents import EvidenceReducer, SubagentDispatcher
from agentic_rag.query.todos import InvalidTodoTransition, TodoItem, TodoReducer, TodoUpdate
from agentic_rag.query.tools import ResearchContext, ResearchToolset, RetrievalPort
from agentic_rag.runtime.model_gateway import ModelCall, ModelGateway, StructuredOutputValidationError, load_prompt


class UpdateTodos(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    action: Literal["update_todos"]
    updates: tuple["TodoActionUpdate", ...] = ()


class TodoActionUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    todo_id: str = Field(min_length=1)
    status: Literal["pending", "in_progress", "completed", "blocked", "skipped"] | None = None
    evidence_ids: tuple[str, ...] | None = None
    result_ref: str | None = None


class RetrieveEvidence(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    action: Literal["retrieve_evidence"]
    query: str = Field(min_length=1, max_length=8_000)
    todo_id: str | None = None


class DelegateResearch(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    action: Literal["delegate_research"]
    todo_ids: tuple[str, ...] = Field(min_length=1)


class CalculatorCall(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    action: Literal["calculator"]
    expression: str = Field(min_length=1, max_length=1_000)


class SubmitEvidence(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    action: Literal["submit_evidence"]
    evidence_ids: tuple[str, ...] = ()


class CannotAnswer(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    action: Literal["cannot_answer"]
    reason: str = Field(default="insufficient verified evidence", min_length=1, max_length=1_000)


ResearchAction = Annotated[
    UpdateTodos | RetrieveEvidence | DelegateResearch | CalculatorCall | SubmitEvidence | CannotAnswer,
    Field(discriminator="action"),
]
_RESEARCH_ACTION: TypeAdapter[Any] = TypeAdapter(ResearchAction)


class StructuredGateway(Protocol):
    async def complete_structured(self, call: ModelCall, schema: type[object]) -> object: ...


@dataclass(frozen=True, slots=True)
class ResearchLoopDependencies:
    """Process-owned dependencies; no clients or compiled graphs enter QueryState."""

    gateway: StructuredGateway | ModelGateway
    retrieval: RetrievalPort
    evidence_builder: EvidenceBuilder
    context_builder: ContextBuilder | None = None
    subagents: SubagentDispatcher | None = None


class ResearchAgentLoop:
    """Ask the model for one strict action, observe it, then re-enter the agent."""

    def __init__(self, dependencies: ResearchLoopDependencies) -> None:
        self._gateway = dependencies.gateway
        self._context = dependencies.context_builder or ContextBuilder()
        self._tools = ResearchToolset(dependencies.retrieval, dependencies.evidence_builder)
        self._subagents = dependencies.subagents

    async def ainvoke(self, state: QueryState) -> dict[str, object]:
        snapshot = snapshot_from_state(state)
        context = ResearchContext(scope=scope_from_state(state), snapshot=snapshot)
        research = _research_state(state)
        todos = _todos(research.get("todos"))
        observations = _observations(research.get("observations"))
        evidence = _evidence(state)
        retrieval_batches = _retrieval_batches(state)
        for round_number in range(snapshot.max_research_rounds):
            action = await self._next_action(state, research, todos, observations)
            result = await self._execute(action, context, todos, observations, evidence, state, round_number)
            todos, observations, evidence = result.todos, result.observations, result.evidence
            retrieval_batches.extend(result.retrieval_batches)
            research = {**research, "todos": _dump_todos(todos), "observations": observations}
            if result.submitted:
                return _result(research, evidence, retrieval_batches=retrieval_batches, submitted=True)
            if result.cannot_answer:
                return _result(
                    {**research, "cannot_answer": True},
                    evidence,
                    retrieval_batches=retrieval_batches,
                    cannot_answer=True,
                    termination_reason=result.termination_reason or "cannot_answer",
                )
        blocked = tuple(
            todo.model_copy(update={"status": "blocked"})
            if todo.status in {"pending", "in_progress"}
            else todo
            for todo in todos
        )
        return _result(
            {**research, "todos": _dump_todos(blocked), "observations": observations},
            evidence,
            retrieval_batches=retrieval_batches,
            termination_reason="research_round_limit",
        )

    async def _next_action(
        self,
        state: QueryState,
        research: dict[str, object],
        todos: tuple[TodoItem, ...],
        observations: list[dict[str, object]],
    ) -> ResearchAction:
        staged = dict(state)
        staged["research"] = {
            **research,
            "todos": _dump_todos(todos),
            "observations": observations,
        }
        prompt_context = await self._context.build(staged)
        call = ModelCall(
            model_role="main",
            snapshot=snapshot_from_state(state),
            messages=(
                {"role": "system", "content": load_prompt("research_agent_v1").content},
                {"role": "user", "content": json.dumps(prompt_context, ensure_ascii=False, separators=(",", ":"))},
            ),
        )
        try:
            response = await self._gateway.complete_structured(call, _ResearchActionSchema)
            raw = getattr(response, "value", response)
            return _parse_action(raw)
        except asyncio.CancelledError:
            raise
        except (StructuredOutputValidationError, ValidationError, TypeError, ValueError) as error:
            return CannotAnswer(action="cannot_answer", reason=f"research_action_invalid: {error}")
        except (OSError, TimeoutError, ConnectionError) as error:
            return CannotAnswer(action="cannot_answer", reason=f"research action unavailable: {error}")

    async def _execute(
        self,
        action: ResearchAction,
        context: ResearchContext,
        todos: tuple[TodoItem, ...],
        observations: list[dict[str, object]],
        evidence: list[dict[str, object]],
        state: QueryState,
        round_number: int,
    ) -> "_Step":
        if isinstance(action, UpdateTodos):
            try:
                updated = TodoReducer.apply_many(
                    todos,
                    tuple(
                        (change.todo_id, TodoUpdate(status=change.status, evidence_ids=change.evidence_ids, result_ref=change.result_ref))
                        for change in action.updates
                    ),
                    actor="supervisor",
                )
                observation = {"kind": "todo_update", "ok": True, "round": round_number + 1}
                return _Step(updated, [*observations, observation], evidence)
            except InvalidTodoTransition as error:
                return _Step(todos, [*observations, {"kind": "todo_update", "ok": False, "error": str(error)}], evidence)
        if isinstance(action, RetrieveEvidence):
            try:
                target_id = action.todo_id or f"query:{state['run_id']}"
                _batch, packed = await self._tools.retrieve_evidence(query=action.query, ctx=context, target_id=target_id)
            except asyncio.CancelledError:
                raise
            except (OSError, TimeoutError, ConnectionError, ValueError) as error:
                return _Step(
                    _block_active(todos),
                    [*observations, {"kind": "retrieval", "ok": False, "error": str(error)}],
                    evidence,
                )
            additions = [item.model_dump(mode="json") for item in packed.items]
            merged = _merge_evidence(evidence, additions)
            return _Step(
                todos,
                [*observations, {"kind": "retrieval", "ok": True, "evidence_ids": [item["evidence_id"] for item in additions]}],
                merged,
                retrieval_batches=[_batch.model_dump(mode="json")],
            )
        if isinstance(action, CalculatorCall):
            observation = await self._tools.calculator(action.expression)
            return _Step(todos, [*observations, {"kind": "calculator", **observation}], evidence)
        if isinstance(action, SubmitEvidence):
            known = {item.get("evidence_id") for item in evidence}
            # Evidence identifiers are server-derived; a model cannot submit an invented one.
            if action.evidence_ids and not set(action.evidence_ids).issubset(known):
                return _Step(todos, [*observations, {"kind": "submit", "ok": False, "error": "unknown evidence id"}], evidence)
            return _Step(todos, [*observations, {"kind": "submit", "ok": True}], evidence, submitted=True)
        if isinstance(action, DelegateResearch):
            if self._subagents is None:
                return _Step(todos, [*observations, {"kind": "delegate", "ok": False, "error": "subagents unavailable"}], evidence)
            by_id = {todo.id: todo for todo in todos}
            if not set(action.todo_ids).issubset(by_id):
                return _Step(todos, [*observations, {"kind": "delegate", "ok": False, "error": "todo does not exist"}], evidence)
            selected = tuple(by_id[todo_id] for todo_id in action.todo_ids)
            completed = frozenset(todo.id for todo in todos if todo.status == "completed")
            try:
                delegated = await self._subagents.delegate(
                    selected,
                    context,
                    max_parallel=snapshot_from_state(state).max_parallel_subagents_per_run,
                    resolved_todo_ids=completed,
                )
            except asyncio.CancelledError:
                raise
            except (OSError, TimeoutError, ConnectionError, ValueError) as error:
                return _Step(todos, [*observations, {"kind": "delegate", "ok": False, "error": str(error)}], evidence)
            if delegated.results:
                packed = EvidenceReducer.merge(
                    delegated.results,
                    expected_index_generation=context.snapshot.index_generation,
                )
                merged = _merge_evidence(evidence, [item.model_dump(mode="json") for item in packed.items])
            else:
                # A full timeout has no evidence to reduce; keep parent evidence
                # and let the dispatcher-provided blocked IDs drive Todo state.
                merged = evidence
            completed_ids = {
                result.todo_id: result
                for result in delegated.results
                if result.evidence.items
            }
            blocked_ids = set(delegated.blocked_todo_ids)
            blocked_ids.update(
                result.todo_id
                for result in delegated.results
                if not result.evidence.items
            )
            updated = tuple(
                todo.model_copy(update={
                    "status": "completed",
                    "evidence_ids": tuple(item.evidence_id for item in completed_ids[todo.id].evidence.items),
                }) if todo.id in completed_ids else (
                    todo.model_copy(update={"status": "blocked"}) if todo.id in blocked_ids else todo
                )
                for todo in todos
            )
            return _Step(
                updated,
                [*observations, {
                    "kind": "delegate", "ok": True,
                    "completed_todo_ids": sorted(completed_ids), "blocked_todo_ids": sorted(blocked_ids),
                }],
                merged,
            )
        assert isinstance(action, CannotAnswer)
        reason = "research_action_invalid" if action.reason.startswith("research_action_invalid:") else "cannot_answer"
        return _Step(
            todos,
            [*observations, {"kind": "cannot_answer", "reason": action.reason}],
            evidence,
            cannot_answer=True,
            termination_reason=reason,
        )


class _ResearchActionSchema(RootModel[ResearchAction]):
    """Strict discriminated union passed to ModelGateway for one repair owner."""


@dataclass(frozen=True, slots=True)
class _Step:
    todos: tuple[TodoItem, ...]
    observations: list[dict[str, object]]
    evidence: list[dict[str, object]]
    submitted: bool = False
    cannot_answer: bool = False
    termination_reason: str | None = None
    retrieval_batches: list[dict[str, object]] = field(default_factory=list)


def _parse_action(value: object) -> ResearchAction:
    if isinstance(value, BaseModel):
        value = value.model_dump(mode="json", exclude_none=True)
    if not isinstance(value, Mapping):
        raise ValueError("research action must be an object")
    return _RESEARCH_ACTION.validate_python(dict(value))


def _research_state(state: QueryState) -> dict[str, object]:
    value = state.get("research")
    return dict(value) if isinstance(value, Mapping) else {}


def _todos(value: object) -> tuple[TodoItem, ...]:
    if not isinstance(value, list):
        return ()
    try:
        parsed = tuple(TodoItem.model_validate(item) for item in value if isinstance(item, Mapping))
        TodoReducer.validate(parsed)
        return parsed
    except (ValidationError, InvalidTodoTransition) as error:
        raise ValueError("invalid serialized research todos") from error


def _observations(value: object) -> list[dict[str, object]]:
    return [dict(item) for item in value if isinstance(item, Mapping)] if isinstance(value, list) else []


def _evidence(state: QueryState) -> list[dict[str, object]]:
    value = state.get("evidence")
    return [dict(item) for item in value if isinstance(item, Mapping)] if isinstance(value, list) else []


def _retrieval_batches(state: QueryState) -> list[dict[str, object]]:
    value = state.get("retrieval_batches")
    return [dict(item) for item in value if isinstance(item, Mapping)] if isinstance(value, list) else []


def _dump_todos(todos: tuple[TodoItem, ...]) -> list[dict[str, object]]:
    return [todo.model_dump(mode="json") for todo in todos]


def _merge_evidence(existing: list[dict[str, object]], additions: list[dict[str, object]]) -> list[dict[str, object]]:
    merged = {str(item.get("evidence_id")): item for item in existing if item.get("evidence_id")}
    for item in additions:
        evidence_id = item.get("evidence_id")
        if evidence_id:
            merged[str(evidence_id)] = item
    return [merged[key] for key in sorted(merged)]


def _block_active(todos: tuple[TodoItem, ...]) -> tuple[TodoItem, ...]:
    return tuple(todo.model_copy(update={"status": "blocked"}) if todo.status in {"pending", "in_progress"} else todo for todo in todos)


def _result(
    research: dict[str, object],
    evidence: list[dict[str, object]],
    *,
    retrieval_batches: list[dict[str, object]] | None = None,
    submitted: bool = False,
    cannot_answer: bool = False,
    termination_reason: str | None = None,
) -> dict[str, object]:
    return {
        "research": {**research, "submitted": submitted, "cannot_answer": cannot_answer},
        "evidence": evidence,
        "retrieval_batches": retrieval_batches or [],
        "next_node": "generate" if submitted else "end",
        "termination_reason": termination_reason,
    }
