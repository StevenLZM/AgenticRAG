# Research AgentLoop Todo DAG Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement a checkpointed, dependency-aware Todo DAG inside the existing Research Agent loop so only ready tasks execute and completed dependency results flow to downstream work.

**Architecture:** Keep scheduling policy as pure deterministic code in `query/todos.py`, execute one model-selected action per invocation of the existing `research_agent_loop` node, and self-loop that node between checkpointed actions. The Agent creates and revises the plan, while server code owns IDs, DAG validation, readiness, execution claims, evidence-bound completion, and submission gates.

**Tech Stack:** Python 3.11, Pydantic v2, asyncio, LangGraph, pytest/pytest-asyncio, Ruff, mypy.

**Spec:** `docs/superpowers/specs/2026-09-08-research-agentloop-todo-dag-design.md`

## Global Constraints

## Execution record (2026-09-08)

Tasks 1–6 have been implemented in the existing working tree on `codex/research-todo-dag`, preserving pre-existing repairs. The detailed checkboxes below remain the original procedure, not a claim that its per-task commits were performed; commits are intentionally deferred because files include earlier uncommitted work.

Verified: unit suite plus SQLite checkpoint, runtime and API integration tests: **675 passed, 6 skipped**. Ruff passes. Query-package mypy passes; full-source mypy has an existing unrelated `testing/e2e_harness.py:547` `ChildHit(score=...)` error. Live-provider end-to-end acceptance and application restart were not performed.

Implementation adjustments: dependent delegate tasks require Supervisor-resolved `queries`; upstream failure classification is transitive; post-grader replanning uses a persistent latch. See the design implementation notes.

### Original constraints

- Do not add a Planner node, Scheduler node, Redis Todo stream, SQL table, or SQL migration.
- Persist `blocked_by`; accept legacy `dependencies` only while reading old checkpoint JSON, and reject payloads containing both names.
- Limit one Run to 12 Todos, 8 blockers per Todo, a longest path of 4 Todo nodes, and 2 execution attempts per Todo.
- The Agent may request only retry and skip transitions; only server code may claim or complete a Todo and attach Evidence IDs/result references.
- `available_todo_ids` remains for one compatibility release but must equal `ready_todo_ids` exactly.
- Execute at most one structured model action per `ResearchAgentLoop.ainvoke()` call.
- Default `max_research_rounds` is 6, its maximum is 8, the Run timeout remains 300 seconds, and graph recursion limit remains 50.
- Preserve the current immutable scope, runtime snapshot, evidence-manifest checks, evidence-token limit, cancellation propagation, and maximum of three parallel Subagents.
- A successful retrieval Todo means one validated retrieval result set was produced; it does not make one Parent globally sufficient or bypass the Evidence Grader.
- The final answer remains outside the Todo DAG and is generated only by the existing audited generation path.
- The current checkout contains pre-existing uncommitted FastRAG repair changes. Before executing this plan, preserve those changes and use an execution workspace that includes them; never overwrite, reset, or stage unrelated edits.

---

### Task 1: Canonical Todo model and deterministic DAG policy

**Files:**

- Modify: `src/agentic_rag/query/todos.py`
- Modify: `tests/unit/query/test_todos.py`

**Interfaces:**

- Consumes: checkpoint JSON containing canonical `blocked_by` or legacy `dependencies` and server-owned Todo outcomes.
- Produces: `TodoDraft`, `TodoDependencyInput`, `TodoDagView`, `TodoReducer.append_drafts()`, `TodoReducer.view()`, `TodoReducer.recover_interrupted()`, `TodoReducer.claim_many()`, `TodoReducer.complete()`, `TodoReducer.block_many()`, and `TodoReducer.apply_agent_updates()`.

- [ ] **Step 1: Replace test fixtures with canonical `blocked_by` and add checkpoint migration tests.**

Add tests proving legacy input is accepted and immediately dumps canonically, while ambiguous input fails:

```python
def test_legacy_dependencies_load_but_dump_as_blocked_by() -> None:
    item = TodoItem.model_validate({
        "id": "todo-2",
        "title": "extract education",
        "owner": "supervisor",
        "dependencies": ["todo-1"],
    })

    assert item.blocked_by == ("todo-1",)
    assert item.model_dump(mode="json")["blocked_by"] == ["todo-1"]
    assert "dependencies" not in item.model_dump(mode="json")


def test_checkpoint_rejects_both_dependency_field_names() -> None:
    with pytest.raises(ValueError, match="blocked_by"):
        TodoItem.model_validate({
            "id": "todo-2",
            "title": "extract education",
            "owner": "supervisor",
            "blocked_by": ["todo-1"],
            "dependencies": ["todo-1"],
        })
```

- [ ] **Step 2: Add DAG creation, limit, cycle, and readiness tests.**

Cover same-action key resolution, existing-ID references, atomic rejection, duplicate references, all four limit types, and stable ready/waiting views:

```python
def test_append_drafts_resolves_keys_and_computes_frontier() -> None:
    todos = TodoReducer.append_drafts((), (
        TodoDraft(key="locate", title="locate resume"),
        TodoDraft(key="extract", title="extract education", blocked_by=("locate",)),
        TodoDraft(key="check", title="cross check", blocked_by=("extract",)),
    ), owner="supervisor")

    assert [item.id for item in todos] == ["todo-1", "todo-2", "todo-3"]
    assert todos[1].blocked_by == ("todo-1",)
    assert todos[2].blocked_by == ("todo-2",)
    assert TodoReducer.view(todos).ready_ids == ("todo-1",)
    assert TodoReducer.view(todos).waiting_ids == ("todo-2", "todo-3")


def test_pending_descendant_of_failed_dependency_is_not_ready() -> None:
    parent = _todo(id="todo-1", status="blocked")
    child = _todo(id="todo-2", blocked_by=("todo-1",))

    view = TodoReducer.view((parent, child))

    assert view.upstream_blocked_ids == ("todo-2",)
    assert view.ready_ids == ()
    assert child.status == "pending"
```

- [ ] **Step 3: Run the Todo tests and confirm the new contract is red.**

Run: `conda run -n agentic-rag pytest --import-mode=importlib tests/unit/query/test_todos.py -q`

Expected: FAIL because `TodoItem.blocked_by`, draft creation, DAG views, and constrained transition methods do not exist.

- [ ] **Step 4: Implement canonical models and legacy-input normalization.**

Use a before-validator that removes only the legacy key and rejects ambiguity; do not configure a serialization alias that can write the old name:

```python
MAX_TODOS = 12
MAX_BLOCKERS_PER_TODO = 8
MAX_DAG_PATH_NODES = 4
MAX_TODO_ATTEMPTS = 2


class TodoItem(BaseModel):
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
    def _read_legacy_dependencies(cls, value: object) -> object:
        if not isinstance(value, Mapping):
            return value
        if "blocked_by" in value and "dependencies" in value:
            raise ValueError("use blocked_by only")
        if "dependencies" not in value:
            return value
        normalized = dict(value)
        normalized["blocked_by"] = normalized.pop("dependencies")
        return normalized


class TodoDraft(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    key: str = Field(pattern=r"^[a-z][a-z0-9_-]{0,63}$")
    title: str = Field(min_length=1, max_length=2_000)
    blocked_by: tuple[str, ...] = ()


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
```

- [ ] **Step 5: Implement atomic draft resolution and bounded DAG validation.**

`append_drafts()` must assign IDs before resolving references, reject a reference that matches both a local key and an existing ID, validate the combined graph, and return no partial additions on failure:

```python
class TodoReducer:
    @classmethod
    def append_drafts(
        cls,
        todos: tuple[TodoItem, ...],
        drafts: Sequence[TodoDraft],
        *,
        owner: str,
    ) -> tuple[TodoItem, ...]:
        cls.validate(todos)
        if not owner.strip() or not drafts:
            raise InvalidTodoTransition("owner and drafts are required")
        if len(todos) + len(drafts) > MAX_TODOS:
            raise InvalidTodoTransition("todo limit exceeded")

        keys = [draft.key for draft in drafts]
        if len(keys) != len(set(keys)):
            raise InvalidTodoTransition("duplicate todo draft key")
        existing_ids = {todo.id for todo in todos}
        if existing_ids.intersection(keys):
            raise InvalidTodoTransition("ambiguous todo reference")

        used_ids = set(existing_ids)
        key_to_id: dict[str, str] = {}
        next_position = 1
        for draft in drafts:
            while f"todo-{next_position}" in used_ids:
                next_position += 1
            assigned = f"todo-{next_position}"
            key_to_id[draft.key] = assigned
            used_ids.add(assigned)
            next_position += 1

        additions: list[TodoItem] = []
        for draft in drafts:
            if not draft.title.strip():
                raise InvalidTodoTransition("todo title must not be blank")
            resolved: list[str] = []
            for reference in draft.blocked_by:
                if reference in key_to_id:
                    resolved.append(key_to_id[reference])
                elif reference in existing_ids:
                    resolved.append(reference)
                else:
                    raise InvalidTodoTransition("dependency does not exist")
            additions.append(TodoItem(
                id=key_to_id[draft.key],
                title=draft.title.strip(),
                owner=owner,
                blocked_by=tuple(resolved),
            ))

        result = (*todos, *additions)
        cls.validate(result)
        return result
```

Extend `validate()` with the bounded depth-first walk below. Keep iteration in
persisted Todo order so ready IDs and dumps remain deterministic:

```python
if len(todos) > MAX_TODOS:
    raise InvalidTodoTransition("todo limit exceeded")
by_id = {todo.id: todo for todo in todos}
if len(by_id) != len(todos):
    raise InvalidTodoTransition("duplicate todo id")
for todo in todos:
    if len(todo.blocked_by) > MAX_BLOCKERS_PER_TODO:
        raise InvalidTodoTransition("dependency limit exceeded")
    if len(set(todo.blocked_by)) != len(todo.blocked_by):
        raise InvalidTodoTransition("duplicate dependency")
    if todo.id in todo.blocked_by:
        raise InvalidTodoTransition("dependency cycle")
    if not set(todo.blocked_by).issubset(by_id):
        raise InvalidTodoTransition("dependency does not exist")

visiting: set[str] = set()
depth_cache: dict[str, int] = {}

def path_nodes(todo_id: str) -> int:
    if todo_id in visiting:
        raise InvalidTodoTransition("dependency cycle")
    if todo_id in depth_cache:
        return depth_cache[todo_id]
    visiting.add(todo_id)
    blockers = by_id[todo_id].blocked_by
    depth = 1 + max((path_nodes(item) for item in blockers), default=0)
    visiting.remove(todo_id)
    if depth > MAX_DAG_PATH_NODES:
        raise InvalidTodoTransition("dependency path limit exceeded")
    depth_cache[todo_id] = depth
    return depth

for todo in todos:
    path_nodes(todo.id)
```

- [ ] **Step 6: Implement readiness and constrained transitions.**

Add these exact reducer boundaries:

```python
class TodoReducer:
    @classmethod
    def view(cls, todos: tuple[TodoItem, ...]) -> TodoDagView:
        cls.validate(todos)
        by_id = {todo.id: todo for todo in todos}
        buckets: dict[str, list[str]] = {
            "ready": [], "waiting": [], "upstream_blocked": [],
            "in_progress": [], "completed": [], "blocked": [], "skipped": [],
        }
        for todo in todos:
            if todo.status != "pending":
                buckets[todo.status].append(todo.id)
                continue
            blockers = tuple(by_id[item].status for item in todo.blocked_by)
            if any(status in {"blocked", "skipped"} for status in blockers):
                buckets["upstream_blocked"].append(todo.id)
            elif all(status == "completed" for status in blockers):
                buckets["ready"].append(todo.id)
            else:
                buckets["waiting"].append(todo.id)
        return TodoDagView(
            ready_ids=tuple(buckets["ready"]),
            waiting_ids=tuple(buckets["waiting"]),
            upstream_blocked_ids=tuple(buckets["upstream_blocked"]),
            in_progress_ids=tuple(buckets["in_progress"]),
            completed_ids=tuple(buckets["completed"]),
            blocked_ids=tuple(buckets["blocked"]),
            skipped_ids=tuple(buckets["skipped"]),
        )

    @classmethod
    def recover_interrupted(
        cls, todos: tuple[TodoItem, ...]
    ) -> tuple[TodoItem, ...]:
        recovered = tuple(
            todo.model_copy(update={"status": "blocked"})
            if todo.status == "in_progress" else todo
            for todo in todos
        )
        cls.validate(recovered)
        return recovered

    @classmethod
    def claim_many(
        cls, todos: tuple[TodoItem, ...], todo_ids: tuple[str, ...]
    ) -> tuple[TodoItem, ...]:
        if not todo_ids or len(todo_ids) != len(set(todo_ids)):
            raise InvalidTodoTransition("todo selection must be unique")
        by_id = {todo.id: todo for todo in todos}
        if not set(todo_ids).issubset(cls.view(todos).ready_ids):
            raise InvalidTodoTransition("todo is not ready")
        if any(by_id[todo_id].attempts >= MAX_TODO_ATTEMPTS for todo_id in todo_ids):
            raise InvalidTodoTransition("todo attempt limit reached")
        selected = set(todo_ids)
        claimed = tuple(
            todo.model_copy(update={
                "status": "in_progress",
                "attempts": todo.attempts + 1,
            }) if todo.id in selected else todo
            for todo in todos
        )
        cls.validate(claimed)
        return claimed

    @classmethod
    def complete(
        cls,
        todos: tuple[TodoItem, ...],
        todo_id: str,
        *,
        evidence_ids: tuple[str, ...] = (),
        result_ref: str | None = None,
    ) -> tuple[TodoItem, ...]:
        if not evidence_ids and not result_ref:
            raise InvalidTodoTransition("completed todo requires evidence or result")
        by_id = {todo.id: todo for todo in todos}
        item = by_id.get(todo_id)
        if item is None or item.status != "in_progress":
            raise InvalidTodoTransition("only in-progress todo can complete")
        by_id[todo_id] = item.model_copy(update={
            "status": "completed",
            "evidence_ids": tuple(dict.fromkeys(evidence_ids)),
            "result_ref": result_ref,
        })
        result = tuple(by_id[todo.id] for todo in todos)
        cls.validate(result)
        return result

    @classmethod
    def block_many(
        cls, todos: tuple[TodoItem, ...], todo_ids: tuple[str, ...]
    ) -> tuple[TodoItem, ...]:
        selected = set(todo_ids)
        by_id = {todo.id: todo for todo in todos}
        if len(selected) != len(todo_ids) or not selected.issubset(by_id):
            raise InvalidTodoTransition("invalid todo selection")
        if any(by_id[item].status not in {"pending", "in_progress"} for item in selected):
            raise InvalidTodoTransition("only active todos can be blocked")
        result = tuple(
            todo.model_copy(update={"status": "blocked"})
            if todo.id in selected else todo
            for todo in todos
        )
        cls.validate(result)
        return result

    @classmethod
    def apply_agent_updates(
        cls,
        todos: tuple[TodoItem, ...],
        updates: tuple[tuple[str, Literal["pending", "skipped"]], ...],
    ) -> tuple[TodoItem, ...]:
        if len({todo_id for todo_id, _status in updates}) != len(updates):
            raise InvalidTodoTransition("duplicate todo update")
        by_id = {todo.id: todo for todo in todos}
        for todo_id, status in updates:
            item = by_id.get(todo_id)
            if item is None:
                raise InvalidTodoTransition("todo does not exist")
            if status == "pending":
                if item.status != "blocked" or item.attempts >= MAX_TODO_ATTEMPTS:
                    raise InvalidTodoTransition("todo cannot be retried")
                if any(by_id[dep].status in {"blocked", "skipped"} for dep in item.blocked_by):
                    raise InvalidTodoTransition("todo has terminal dependency")
            elif item.status not in {"pending", "blocked"}:
                raise InvalidTodoTransition("todo cannot be skipped")
            by_id[todo_id] = item.model_copy(update={"status": status})
        result = tuple(by_id[todo.id] for todo in todos)
        cls.validate(result)
        return result
```

`claim_many()` must require unique IDs from `view().ready_ids`, change them to
`in_progress`, and increment attempts. `complete()` must accept only an
in-progress Todo and require evidence or `result_ref`. `apply_agent_updates()`
must allow only `blocked -> pending` below the attempt limit and
`pending|blocked -> skipped`. `recover_interrupted()` changes checkpointed
in-progress items to blocked without incrementing attempts.

- [ ] **Step 7: Run focused tests, lint, and type checking.**

Run: `conda run -n agentic-rag pytest --import-mode=importlib tests/unit/query/test_todos.py -q && conda run -n agentic-rag ruff check src/agentic_rag/query/todos.py tests/unit/query/test_todos.py && MYPYPATH=src conda run -n agentic-rag mypy --explicit-package-bases src/agentic_rag/query/todos.py`

Expected: all Todo tests pass; old checkpoint input serializes canonically; invalid graphs and illegal transitions fail before execution.

- [ ] **Step 8: Commit the pure Todo DAG layer.**

```bash
git add src/agentic_rag/query/todos.py tests/unit/query/test_todos.py
git commit -m "feat: add research todo dag policy"
```

---

### Task 2: Dependency-aware action and prompt context contracts

**Files:**

- Modify: `src/agentic_rag/query/research_loop.py`
- Modify: `src/agentic_rag/query/context.py`
- Modify: `src/agentic_rag/prompts/research_agent_v1.md`
- Modify: `tests/unit/query/test_research_loop.py`

**Interfaces:**

- Consumes: `TodoDraft`, canonical serialized Todos, `TodoReducer.view()`, current Evidence Manifest, and grader gaps.
- Produces: strict `CreateTodos.items`, constrained `TodoActionUpdate`, Todo-targeted retrieval/calculator actions, and a transition contract whose executable IDs equal the ready DAG frontier.

- [ ] **Step 1: Add strict action-schema tests.**

Test the discriminated union through `_ResearchActionSchema` so old title-only creation, a completion update, and a tool action without a Todo ID are rejected:

```python
@pytest.mark.parametrize("action", [
    {"action": "create_todos", "titles": ["old shape"]},
    {"action": "update_todos", "updates": [{"todo_id": "todo-1", "status": "completed"}]},
    {"action": "retrieve_evidence", "query": "education"},
    {"action": "calculator", "expression": "1+1"},
])
def test_research_action_schema_rejects_dag_bypasses(action: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        _ResearchActionSchema.model_validate(action)
```

Also test a three-draft plan using local keys is accepted.

- [ ] **Step 2: Add ContextBuilder frontier tests.**

Build a state containing one completed prerequisite, one ready child, one
waiting child, one upstream-blocked child, and one directly blocked Todo. Assert:

```python
assert built["transition_contract"] == {
    "plan_required": False,
    "ready_todo_ids": ["ready"],
    "available_todo_ids": ["ready"],
    "waiting_todo_ids": ["waiting"],
    "upstream_blocked_todo_ids": ["downstream-of-failure"],
    "blocked_todo_ids": ["failed"],
    "retryable_todo_ids": ["failed"],
    "known_evidence_ids": ["e1"],
    "unresolved_todos_remain": True,
    "submit_evidence_valid": False,
}
```

Add an empty-plan assertion for `plan_required=true` and
`submit_evidence_valid=false`, and a terminal-plan assertion showing submission
becomes valid only with at least one Todo and verified evidence. Add a Grader
re-entry case where prior `submitted=true` plus non-empty gaps requires a new
plan fragment before resubmission.

- [ ] **Step 3: Run the schema/context tests and confirm they fail.**

Run: `conda run -n agentic-rag pytest --import-mode=importlib tests/unit/query/test_research_loop.py -q`

Expected: FAIL because actions still use title-only creation, calculator/retrieval can bypass Todo selection, and context exposes all pending/in-progress IDs.

- [ ] **Step 4: Replace the action payloads with the approved contracts.**

Use the domain `TodoDraft` directly so the Pydantic JSON Schema and reducer
cannot drift:

```python
class CreateTodos(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    action: Literal["create_todos"]
    items: tuple[TodoDraft, ...] = Field(min_length=1)


class TodoActionUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    todo_id: str = Field(min_length=1)
    status: Literal["pending", "skipped"]


class RetrieveEvidence(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    action: Literal["retrieve_evidence"]
    todo_id: str = Field(min_length=1)
    query: str = Field(min_length=1, max_length=8_000)


class CalculatorCall(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    action: Literal["calculator"]
    todo_id: str = Field(min_length=1)
    expression: str = Field(min_length=1, max_length=1_000)
```

- [ ] **Step 5: Build the transition contract from the deterministic DAG view.**

Parse canonical/legacy Todo mappings with `TodoItem.model_validate()`, call
`TodoReducer.view()`, and derive retryable IDs from blocked Todos with
`attempts < MAX_TODO_ATTEMPTS`. Preserve unresolved Todo details for model
planning, but make both `ready_todo_ids` and compatibility
`available_todo_ids` equal only `view.ready_ids`.

Compute planning and submission exactly as:

```python
active = bool(
    view.ready_ids
    or view.waiting_ids
    or view.upstream_blocked_ids
    or view.in_progress_ids
)
plan_required = not todos or (
    research.get("submitted") is True and bool(research.get("gaps"))
)
submit_evidence_valid = (
    bool(todos) and bool(known_evidence_ids) and not active and not plan_required
)
```

- [ ] **Step 6: Update the Research prompt.**

State that the first action normally creates a plan, `blocked_by` points to
prerequisites, execution actions may use only `ready_todo_ids`, independent
ready IDs may be delegated together, waiting/upstream-blocked IDs are invalid,
and only the server completes tasks. Explicitly say `submit_evidence` is valid
only when the transition contract says so.

- [ ] **Step 7: Run focused tests, lint, and prompt-contract tests.**

Run: `conda run -n agentic-rag pytest --import-mode=importlib tests/unit/query/test_research_loop.py tests/unit/test_documentation_contracts.py -q && conda run -n agentic-rag ruff check src/agentic_rag/query/context.py src/agentic_rag/query/research_loop.py tests/unit/query/test_research_loop.py`

Expected: strict action JSON Schema and context views agree; no waiting Todo is advertised as executable; prompt contract tests pass.

- [ ] **Step 8: Commit the model-facing DAG contract.**

```bash
git add src/agentic_rag/query/research_loop.py src/agentic_rag/query/context.py src/agentic_rag/prompts/research_agent_v1.md tests/unit/query/test_research_loop.py tests/unit/test_documentation_contracts.py
git commit -m "feat: expose ready research todo frontier"
```

---

### Task 3: Dynamic dependency inputs and guarded parallel Subagents

**Files:**

- Modify: `src/agentic_rag/query/subagents.py`
- Modify: `src/agentic_rag/runtime/query_composition.py`
- Modify: `tests/unit/query/test_subagents.py`
- Modify: `tests/unit/runtime/test_query_composition.py`

**Interfaces:**

- Consumes: selected ready `TodoItem` objects, current `PackedEvidence`, current Memory summary, direct `TodoDependencyInput` values, immutable scope, and completed Todo IDs.
- Produces: per-call `ChildResearchState` containing current dependency outputs and current manifest; preserves existing `DelegationResult` and deterministic `EvidenceReducer` behavior.

- [ ] **Step 1: Add dynamic-child-context tests.**

Create a completed parent with Evidence ID `e-parent`, a ready child blocked by
that parent, and a current pack containing `e-parent`. Assert the worker sees:

```python
assert child.dependency_inputs == (
    TodoDependencyInput(
        todo_id="parent",
        evidence_ids=("e-parent",),
        result_ref=None,
    ),
)
assert "parent evidence" in child.dependency_context
assert child.evidence_manifest == current_pack.manifest
```

Call the same dispatcher a second time with a different pack and prove the
second child sees the second manifest, not constructor-time state. Keep the
existing concurrency, partial timeout, cancellation, and unresolved-dependency
tests.

- [ ] **Step 2: Add defense-in-depth frontier tests.**

Assert delegation rejects a waiting Todo, an upstream-blocked Todo, duplicate
Todo IDs, and a batch containing a parent and its child. Assert independent
ready siblings run concurrently and preserve stable result order.

- [ ] **Step 3: Run Subagent/composition tests and confirm the dynamic-input assertions fail.**

Run: `conda run -n agentic-rag pytest --import-mode=importlib tests/unit/query/test_subagents.py tests/unit/runtime/test_query_composition.py -q`

Expected: FAIL because child manifests are constructor snapshots and no dependency inputs/context are present.

- [ ] **Step 4: Move mutable research inputs to the `delegate()` call.**

Use this interface:

```python
@dataclass(frozen=True, slots=True)
class ChildResearchState:
    todo_id: str
    question: str
    scope: Mapping[str, object]
    retrieval_filter: Mapping[str, object]
    memory_summary: str
    evidence_manifest: Mapping[str, Mapping[str, object]]
    dependency_inputs: tuple[TodoDependencyInput, ...]
    dependency_context: str
```

Change `delegate()` to keep positional parameters
`items: Sequence[TodoItem]` and `context: ResearchContext`, require keyword-only
`packed_evidence: PackedEvidence`,
`dependency_inputs: Mapping[str, tuple[TodoDependencyInput, ...]]`, and
`memory_summary: str`, retain keyword defaults `max_parallel=3`,
`timeout_seconds=20`, and `resolved_todo_ids=frozenset()`, and return
`DelegationResult`.

Remove `memory_summary` and `evidence_manifest` from persistent dispatcher
constructor state. `build_subagent_dispatcher()` continues to compose only
process-owned tools, concurrency, and worker dependencies.

- [ ] **Step 5: Render bounded dependency evidence.**

For each child, union only the Evidence IDs referenced by its direct dependency
inputs, select matching items from `packed_evidence` in pack order, verify each
manifest entry with the existing `_matches_manifest()`, and render with
`DataEnvelope`. Reject a referenced Evidence ID missing from the current pack.
Because this is a subset of the already bounded pack, do not introduce a second
token budget or a second cropper.

- [ ] **Step 6: Keep the dispatcher guard independent of the prompt.**

Before creating asyncio tasks, require every requested Todo to be unfinished,
require every `blocked_by` ID to be in `resolved_todo_ids`, and reject duplicate
requested IDs. Do not infer readiness from child output or from Evidence
Manifest coverage.

- [ ] **Step 7: Run focused tests, lint, and type checking.**

Run: `conda run -n agentic-rag pytest --import-mode=importlib tests/unit/query/test_subagents.py tests/unit/runtime/test_query_composition.py -q && conda run -n agentic-rag ruff check src/agentic_rag/query/subagents.py src/agentic_rag/runtime/query_composition.py tests/unit/query/test_subagents.py && MYPYPATH=src conda run -n agentic-rag mypy --explicit-package-bases src/agentic_rag/query/subagents.py src/agentic_rag/runtime/query_composition.py`

Expected: downstream children see current dependency evidence, independent
siblings remain bounded and parallel, and unresolved work starts no child task.

- [ ] **Step 8: Commit dependency-aware Subagent dispatch.**

```bash
git add src/agentic_rag/query/subagents.py src/agentic_rag/runtime/query_composition.py tests/unit/query/test_subagents.py tests/unit/runtime/test_query_composition.py
git commit -m "feat: pass todo dependency results to subagents"
```

---

### Task 4: One-action Research loop with evidence-bound Todo lifecycle

**Files:**

- Modify: `src/agentic_rag/query/research_loop.py`
- Modify: `tests/unit/query/test_research_loop.py`

**Interfaces:**

- Consumes: strict `ResearchAction`, `TodoReducer` DAG methods, current `PackedEvidence`, raw retrieval batches, and dependency-aware `SubagentDispatcher.delegate()`.
- Produces: one checkpoint-safe action result with `next_node` equal to `research_agent`, `generate`, or `end`; automatic server-owned claim/complete/block transitions around tools.

- [ ] **Step 1: Add one-action and empty-plan tests.**

Assert an empty Research state no longer creates a synthetic root. One
`create_todos` response must call the gateway once, execute no retrieval, and
return:

```python
assert result["next_node"] == "research_agent"
assert result["research_attempt_count"] == 1
assert [item["id"] for item in result["research"]["todos"]] == ["todo-1", "todo-2"]
assert all("dependencies" not in item for item in result["research"]["todos"])
```

Add a pre-exhausted state assertion proving zero gateway/tool calls and
`research_round_limit` termination.

- [ ] **Step 2: Add action-guard and direct-tool lifecycle tests.**

Cover all of these cases:

- retrieval for a waiting/missing/blocked/completed Todo returns
  `todo_not_ready` without calling retrieval;
- successful targeted retrieval performs
  `pending -> in_progress -> completed`, increments attempts once, and attaches
  every validated Evidence ID from the returned set;
- an empty or failed retrieval blocks only its selected Todo;
- calculator completes its selected Todo with a server-created result reference;
- a model completion update cannot enter the action schema;
- retry changes a directly blocked Todo to pending but cannot exceed two
  attempts;
- skip leaves descendants pending and classified as upstream-blocked.

- [ ] **Step 3: Add delegation lifecycle and completion-boundary tests.**

Script two independent ready siblings and one child blocked by both. Assert the
siblings can be delegated in one action, successful children complete with all
their evidence, timeout children block, and the dependent child is not accepted
in the same batch. Assert one Parent carrying a Todo target does not discard
other complementary Parents and does not bypass the later Evidence Grader.

- [ ] **Step 4: Add submit-gate and grader-reentry tests.**

Assert submission is rejected while any Todo is ready, waiting, upstream
blocked, or in progress; it succeeds only with terminal Todos and current
manifest IDs. After an insufficient Evidence Grader result re-enters Research,
assert the next action may append a new DAG fragment while preserving the
Run-global action count and prior evidence.

- [ ] **Step 5: Run Research loop tests and confirm the old inner-loop behavior fails.**

Run: `conda run -n agentic-rag pytest --import-mode=importlib tests/unit/query/test_research_loop.py tests/unit/query/test_subagents.py -q`

Expected: FAIL because one invocation currently consumes multiple actions,
creates a root Todo, does not transition direct-retrieval Todos, and permits
submission with active work.

- [ ] **Step 6: Refactor `ainvoke()` to exactly one action.**

Retain the existing scope, checkpoint, packed-evidence, observation, and raw
batch loading before this control block. Replace the inner `while` with this
single-action control shape, and add an explicit `next_node` parameter to
`_result()`. Before recovery, collect checkpointed in-progress IDs; after
`recover_interrupted()` append one safe `todo_interrupted` observation listing
only those IDs:

```python
attempt = int(state.get("research_attempt_count", 0))
if attempt >= snapshot.max_research_rounds:
    active_ids = tuple(
        todo.id for todo in todos if todo.status in {"pending", "in_progress"}
    )
    limited = TodoReducer.block_many(todos, active_ids) if active_ids else todos
    return _result(
        {**research, "todos": _dump_todos(limited), "observations": observations},
        packed,
        retrieval_batches=retrieval_batches,
        research_attempt_count=attempt,
        termination_reason="research_round_limit",
        next_node="end",
    )

action = await self._next_action(state, research, todos, observations, packed)
step = await self._execute(
    action, context, todos, observations, packed, state, attempt
)
todos = step.todos
observations = step.observations
retrieval_batches.extend(step.retrieval_batches)
if step.retrieval_batches:
    packed = self._rebuild_working_pack(state, todos, retrieval_batches, context)

next_node = (
    "generate" if step.submitted
    else "end" if step.cannot_answer
    else "research_agent"
)
return _result(
    {
        **research,
        "todos": _dump_todos(todos),
        "observations": observations,
        "submitted": step.submitted,
        "cannot_answer": step.cannot_answer,
    },
    packed,
    retrieval_batches=retrieval_batches,
    submitted=step.submitted,
    cannot_answer=step.cannot_answer,
    research_attempt_count=attempt + 1,
    termination_reason=step.termination_reason,
    next_node=next_node,
)
```

Do not retain a `while` loop inside `ainvoke()`. A nonterminal model/action
validation observation returns `next_node="research_agent"` while budget
remains. `submit_evidence` returns `generate`; `cannot_answer`, evidence
corruption, and exhausted budget return `end`.

- [ ] **Step 7: Guard and reduce each action through TodoReducer.**

For retrieval/calculator, validate readiness, claim before invoking the tool,
then complete or block the selected Todo. For delegation, validate the whole
selection before claiming any item, pass the current pack/dependency inputs to
the dispatcher, and reduce successful/blocked children in stable Todo-ID order.
For `create_todos`, call `append_drafts()` once. For `update_todos`, call only
`apply_agent_updates()`.

Build direct dependency inputs exactly from completed blocker items:

```python
def _dependency_inputs(
    todo: TodoItem, by_id: Mapping[str, TodoItem]
) -> tuple[TodoDependencyInput, ...]:
    return tuple(
        TodoDependencyInput(
            todo_id=dependency.id,
            evidence_ids=dependency.evidence_ids,
            result_ref=dependency.result_ref,
        )
        for dependency_id in todo.blocked_by
        if (dependency := by_id[dependency_id]).status == "completed"
    )
```

- [ ] **Step 8: Enforce evidence and submission boundaries.**

A retrieval/delegation completion must use Evidence IDs that remain present in
the merged validated pack. Preserve every raw batch for the post-Research
Evidence Builder. Check `submit_evidence` against a fresh `TodoReducer.view()`
and the current manifest; never treat target coverage as global evidence
sufficiency.

- [ ] **Step 9: Run loop, Todo, Subagent, lint, and type checks.**

Run: `conda run -n agentic-rag pytest --import-mode=importlib tests/unit/query/test_todos.py tests/unit/query/test_research_loop.py tests/unit/query/test_subagents.py -q && conda run -n agentic-rag ruff check src/agentic_rag/query tests/unit/query && MYPYPATH=src conda run -n agentic-rag mypy --explicit-package-bases src/agentic_rag/query`

Expected: every loop call performs at most one model action; only ready Todos
reach tools; tool outcomes and evidence references drive server-owned status.

- [ ] **Step 10: Commit the one-action lifecycle.**

```bash
git add src/agentic_rag/query/research_loop.py tests/unit/query/test_research_loop.py
git commit -m "feat: execute one guarded research todo action"
```

---

### Task 5: LangGraph self-loop, checkpoint boundary, and action budget

**Files:**

- Modify: `src/agentic_rag/query/graph.py`
- Modify: `src/agentic_rag/runtime/models.py`
- Modify: `src/agentic_rag/config.py`
- Modify: `src/agentic_rag/runtime/query_composition.py`
- Modify: `src/agentic_rag/api/query_runs.py`
- Modify: `.env.example`
- Modify: `docs/local-operations.md`
- Modify: `tests/unit/query/test_graph.py`
- Modify: `tests/unit/runtime/test_models.py`
- Modify: `tests/unit/runtime/test_query_composition.py`
- Modify: `tests/unit/test_config.py`
- Modify: `tests/integration/persistence/test_sqlite_checkpoint.py`

**Interfaces:**

- Consumes: one-action Research updates whose `next_node` is `research_agent`, `generate`, or `end`.
- Produces: same-node Research continuation with a LangGraph checkpoint after each action, Run-global six-action default/eight-action maximum, and one terminal `RESEARCH_LOOP_COMPLETED` event.

- [ ] **Step 1: Add graph self-loop tests.**

Use the real `ResearchAgentLoop` with scripted actions
`create_todos -> delegate_research -> delegate_research -> submit_evidence`.
Assert four gateway calls, four invocations of the existing graph node, no new
node name, dependency order, and successful arrival at the Evidence Builder.

Add a fake one-action loop that returns `research_agent` twice and `generate`
once, then assert the graph calls it three times. This isolates edge routing
from retrieval behavior.

- [ ] **Step 2: Add checkpoint-resume coverage.**

Compile the Query graph with the SQLite checkpointer. Let the first Research
action return normally, make the scripted gateway raise on the second Research
action, then load the latest checkpoint and assert canonical `blocked_by`,
action count, Todo statuses, packed manifest, raw batches, and the pending
`research_agent_loop` next node are present. Replace the failing gateway, resume
with the same server-owned checkpoint namespace, and prove the first action is
not repeated.

- [ ] **Step 3: Add budget and configuration tests.**

Assert:

```python
settings = Settings(
    mysql_dsn="mysql+asyncmy://rag:rag@127.0.0.1/rag",
    deepseek_base_url="https://models.example.invalid/v1",
    qwen_embedding_base_url="https://embeddings.example.invalid/v1",
)
assert settings.max_research_rounds == 6
assert RuntimeConfigSnapshot(**SNAPSHOT_DATA).max_research_rounds == 6
assert RuntimeConfigSnapshot(**SNAPSHOT_DATA, max_research_rounds=8).max_research_rounds == 8
with pytest.raises(ValidationError):
    RuntimeConfigSnapshot(**SNAPSHOT_DATA, max_research_rounds=9)
with pytest.raises(ValidationError):
    Settings(
        mysql_dsn="mysql+asyncmy://rag:rag@127.0.0.1/rag",
        deepseek_base_url="https://models.example.invalid/v1",
        qwen_embedding_base_url="https://embeddings.example.invalid/v1",
        max_research_rounds=9,
    )
```

Update composition and API fallback tests so every new Run snapshots the same
value. Assert a resumed run at its limit issues no model or retrieval call.

- [ ] **Step 4: Run graph/config/checkpoint tests and confirm they fail.**

Run: `conda run -n agentic-rag pytest --import-mode=importlib tests/unit/query/test_graph.py tests/unit/runtime/test_models.py tests/unit/runtime/test_query_composition.py tests/unit/test_config.py tests/integration/persistence/test_sqlite_checkpoint.py -q`

Expected: FAIL because `after_research()` has no self-edge, default/maximum are
2/4, and one action is not checkpointed before the next.

- [ ] **Step 5: Add the same-node conditional edge.**

Route without adding a node:

```python
def after_research(state: QueryState) -> str:
    next_node = state.get("next_node")
    if next_node == "research_agent":
        return "research_agent_loop"
    if next_node == "generate":
        return "evidence_builder"
    return "finalize"


builder.add_conditional_edges(
    "research_agent_loop",
    after_research,
    {
        "research_agent_loop": "research_agent_loop",
        "evidence_builder": "evidence_builder",
        "finalize": "finalize",
    },
)
```

On a continuing step emit a bounded `PROGRESS` event with summary
`research_step_completed`. Emit `RESEARCH_LOOP_COMPLETED` only when the update
routes to `generate` or `end`, preserving its existing finite projection
attributes.

- [ ] **Step 6: Raise the snapshotted action budget consistently.**

Change `Settings.max_research_rounds`, `RuntimeConfigSnapshot.max_research_rounds`,
the API fallback, `.env.example`, and operations documentation from 2 to 6.
Declare the Settings field as `Field(default=6, ge=1, le=8)` and change the
Runtime snapshot upper bound from 4 to 8. Keep
`query_run_timeout_seconds=300` and graph recursion limit 50 unchanged.

- [ ] **Step 7: Run graph, configuration, checkpoint, lint, and type checks.**

Run: `conda run -n agentic-rag pytest --import-mode=importlib tests/unit/query/test_graph.py tests/unit/runtime/test_models.py tests/unit/runtime/test_query_composition.py tests/unit/test_config.py tests/integration/persistence/test_sqlite_checkpoint.py -q && conda run -n agentic-rag ruff check src/agentic_rag/query/graph.py src/agentic_rag/runtime/models.py src/agentic_rag/config.py src/agentic_rag/runtime/query_composition.py src/agentic_rag/api/query_runs.py && MYPYPATH=src conda run -n agentic-rag mypy --explicit-package-bases src/agentic_rag/query/graph.py src/agentic_rag/runtime/models.py src/agentic_rag/config.py src/agentic_rag/runtime/query_composition.py src/agentic_rag/api/query_runs.py`

Expected: the same Research node checkpoints and resumes one action at a time;
the six/eight action contract is consistent at every Run-creation boundary.

- [ ] **Step 8: Commit graph checkpointing and budgets.**

```bash
git add src/agentic_rag/query/graph.py src/agentic_rag/runtime/models.py src/agentic_rag/config.py src/agentic_rag/runtime/query_composition.py src/agentic_rag/api/query_runs.py .env.example docs/local-operations.md tests/unit/query/test_graph.py tests/unit/runtime/test_models.py tests/unit/runtime/test_query_composition.py tests/unit/test_config.py tests/integration/persistence/test_sqlite_checkpoint.py
git commit -m "feat: checkpoint research todo actions"
```

---

### Task 6: End-to-end DAG regression and documentation alignment

**Files:**

- Modify: `docs/superpowers/specs/2026-08-04-production-agentic-rag-design.md`
- Modify: `docs/development-progress.md`
- Modify: `tests/unit/test_documentation_contracts.py`
- Modify: `tests/unit/query/test_research_loop.py`
- Modify: `tests/unit/query/test_graph.py`
- Verify: every source and test file modified in Tasks 1-5

**Interfaces:**

- Consumes: complete embedded Todo DAG, one-action graph loop, current evidence pipeline, and documented production architecture.
- Produces: one end-to-end dependency-chain regression and documentation that points to the approved detailed design without claiming a separate scheduler.

- [ ] **Step 1: Add the full education-query task-chain regression.**

Use deterministic fake retrieval/Subagent outputs for this plan:

```text
todo-1 locate resume
  -> todo-2 extract education
    -> todo-3 cross-check school, degree, major, and dates
```

Assert the model sees only one ready frontier per checkpoint, each downstream
child receives direct predecessor evidence, final packed evidence retains
complementary Parents, submission occurs after all three tasks complete, and
the Evidence Grader—not Todo target coverage—decides sufficiency.

- [ ] **Step 2: Add failure-path regression for a broken prerequisite.**

Timeout `todo-1`; assert `todo-2` and `todo-3` remain pending and appear as
upstream-blocked, neither child starts, submission is invalid, retry is bounded,
and `cannot_answer` terminates without generation when progress is impossible.

- [ ] **Step 3: Run the end-to-end focused tests.**

Run: `conda run -n agentic-rag pytest --import-mode=importlib tests/unit/query/test_todos.py tests/unit/query/test_research_loop.py tests/unit/query/test_subagents.py tests/unit/query/test_graph.py -q`

Expected: the success chain and broken-prerequisite path pass deterministically
without a Scheduler node or unbounded Agent loop.

- [ ] **Step 4: Align architecture and progress documentation.**

In the production design's Dynamic Todo section, state that
`blocked_by` is canonical, link to
`docs/superpowers/specs/2026-09-08-research-agentloop-todo-dag-design.md`, and
clarify that the existing Research node self-loops one action at a time. Record
the implementation and verification commands in `docs/development-progress.md`.
Do not modify rebuild procedures or claim that local document data was changed.

- [ ] **Step 5: Run the full unit suite.**

Run: `conda run -n agentic-rag pytest --import-mode=importlib tests/unit -q`

Expected: PASS with no regression in FastRAG, evidence packing, model-gateway,
query graph, runtime composition, console, or documentation contracts.

- [ ] **Step 6: Run integration tests covering Query runtime and persistence.**

Run: `conda run -n agentic-rag pytest --import-mode=importlib tests/integration/runtime/test_query_worker.py tests/integration/persistence/test_sqlite_checkpoint.py tests/integration/api/test_query_runs.py -q`

Expected: PASS; query cancellation/timeouts, worker lifecycle, checkpoint
namespacing, SSE event safety, and API Run snapshots remain valid.

- [ ] **Step 7: Run repository-wide static checks.**

Run: `conda run -n agentic-rag ruff check src tests && MYPYPATH=src conda run -n agentic-rag mypy --explicit-package-bases src/agentic_rag`

Expected: Ruff and mypy both exit zero.

- [ ] **Step 8: Review the final diff for scope and checkpoint compatibility.**

Run: `git diff --check && git diff --stat && git status --short`

Expected: no whitespace errors; only files listed in this plan plus pre-existing
user changes are present. Verify the DAG implementation did not add a graph
node, Redis stream, SQL migration, or final-answer Subagent action.

- [ ] **Step 9: Commit the end-to-end regression and documentation.**

```bash
git add docs/superpowers/specs/2026-08-04-production-agentic-rag-design.md docs/development-progress.md tests/unit/test_documentation_contracts.py tests/unit/query/test_research_loop.py tests/unit/query/test_graph.py
git commit -m "test: cover research todo dependency chain"
```

## Execution Notes

- Run tasks in order because Tasks 2-5 consume exact types and signatures from
  earlier tasks.
- Do not use multiple implementation agents concurrently on Tasks 1-5; they
  modify shared Todo, Research loop, and test files.
- After each task, inspect `git status --short` before staging so pre-existing
  user edits are not accidentally committed.
- If an existing uncommitted change overlaps a planned hunk, preserve its
  behavior and reconcile the DAG work around it; do not reset the file.
