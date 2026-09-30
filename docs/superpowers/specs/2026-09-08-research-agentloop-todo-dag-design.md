# Research AgentLoop Todo DAG Task System Design

**Date:** 2026-09-08

**Status:** Implemented in working tree; automated regression verified, live-model acceptance pending

## Implementation notes (2026-09-08)

- The production child is a retrieval worker, not another LLM loop. `delegate_research.queries` supplies concrete queries keyed by Todo ID and is required for dependent tasks. The Supervisor resolves upstream facts into these queries; current manifest, dependency evidence and calculator results also flow into child state.
- `upstream_blocked` propagates transitively. A persistent `research.needs_replan` gate prevents repeated submission after the grader reports gaps until new tasks are added.
- One action is checkpointed per graph-node invocation. SQLite close/reopen recovery is covered by integration tests; this is action-boundary recovery, not an exactly-once guarantee for a tool interrupted before its checkpoint commits.
- This change does not restart running services or rewrite existing Run configuration snapshots.

## Context

The Research route already has the beginnings of a Todo subsystem, but it does
not yet implement dependency-aware planning end to end:

- `TodoItem.dependencies` can represent edges and `TodoReducer.validate()` can
  reject missing dependencies, self-dependencies, and cycles.
- `CreateTodos` accepts only titles, so the Research Agent cannot create those
  edges in a real run.
- `ContextBuilder.transition_contract.available_todo_ids` currently includes
  every pending or in-progress Todo, even when a dependency is unresolved.
- Direct retrieval is not tied to a Todo lifecycle. Delegation performs a
  second dependency check, but the Agent can still request a waiting Todo and
  receive only a generic execution failure.
- `ResearchAgentLoop.ainvoke()` executes its whole inner `while` loop inside one
  LangGraph node invocation. Individual Agent actions therefore do not form
  durable checkpoint boundaries.
- The current default of two Research actions cannot execute a plan-then-work-
  then-submit chain.

This design completes that subsystem as a small task system embedded in the
existing Agent loop. It does not introduce a separate Planner graph node,
Scheduler graph node, Redis task stream, or database table.

## Goals

- Let the Research Agent create multiple Todos and express `blocked_by`
  dependencies between them in one structured action.
- Treat the Todo collection as a server-validated DAG and expose only its ready
  frontier for execution.
- Execute independent ready Todos in parallel while preserving dependency order
  across frontiers.
- Carry completed dependency outputs into downstream supervisor and Subagent
  context.
- Keep status transitions, completion evidence, IDs, scope, and limits owned by
  deterministic server code.
- Checkpoint after every Agent-selected action by looping through the existing
  `research_agent_loop` graph node.
- Preserve compatibility with checkpoints that contain the old `dependencies`
  field while writing only the canonical `blocked_by` field.

## Non-goals

- A general workflow engine, distributed task scheduler, or cross-Run task
  queue.
- A second planning LLM call outside `ResearchAgentLoop`.
- Mutable graph edges after a Todo has been created.
- Recursive Subagent planning or Subagent-generated final answers.
- Exactly-once execution for side-effecting tools. Research tools remain
  read-only or idempotent; process death may replay the last uncheckpointed
  tool call.
- Replacing the existing post-Research Evidence Builder, Evidence Grader,
  Generator, faithfulness audit, or citation audit.
- A SQL schema migration. The DAG remains inside checkpoint-safe Query state.

## Selected Architecture

### 1. The task system stays inside the Agent loop

`ResearchAgentLoop` remains the only owner of research planning and action
selection. A pure Todo policy in `query/todos.py` validates the graph, computes
the ready frontier, and applies state transitions before and after a tool call.
It is a helper called by the loop, not an independently scheduled component.

The existing LangGraph node performs one Agent action per invocation and then
routes back to itself when more work remains:

```text
research_agent_loop
    -> load and validate Todo DAG
    -> derive ready/waiting/blocked views
    -> ask Agent for exactly one action
    -> validate and execute that action
    -> reduce tool result into Todo/evidence state
    -> return checkpoint-safe state
    -> research_agent_loop | evidence_builder | finalize
```

This gives each Agent action a LangGraph checkpoint boundary without adding a
Planner or Scheduler node. The deterministic DAG reconciliation itself does not
consume a Research action; the single model-selected action does.

### 2. Canonical Todo contract

The persisted contract is:

```python
TodoStatus = Literal[
    "pending",
    "in_progress",
    "completed",
    "blocked",
    "skipped",
]

class TodoItem(BaseModel):
    id: str
    title: str
    owner: str
    status: TodoStatus = "pending"
    blocked_by: tuple[str, ...] = ()
    evidence_ids: tuple[str, ...] = ()
    result_ref: str | None = None
    attempts: int = 0
```

`B.blocked_by == ("A",)` means the directed edge `A -> B`: A must complete
before B may start.

For backward compatibility, `TodoItem` accepts `dependencies` only while
loading old checkpoint JSON. A payload containing both `blocked_by` and
`dependencies` is invalid. `model_dump(mode="json")` always emits
`blocked_by`, so the next successful checkpoint canonicalizes old state.

The first Research entry starts with an empty Todo collection. The original
question already exists in immutable Query state, so it is not duplicated as a
synthetic root Todo. The transition contract sets `plan_required=true`, and the
first valid progress action is `create_todos` or `cannot_answer`; submission is
not valid without at least one Todo. This avoids an orphaned root Todo when the
Agent creates a real multi-step plan.

### 3. Bounded, immutable DAG creation

`create_todos` accepts drafts instead of title strings:

```json
{
  "action": "create_todos",
  "items": [
    {
      "key": "locate_resume",
      "title": "定位刘泽明的当前简历文档",
      "blocked_by": []
    },
    {
      "key": "extract_education",
      "title": "从已定位文档中提取教育经历",
      "blocked_by": ["locate_resume"]
    },
    {
      "key": "cross_check",
      "title": "核对学校、专业、学历和时间范围",
      "blocked_by": ["extract_education"]
    }
  ]
}
```

`key` is local to one action. A `blocked_by` reference may name a draft key in
the same action or an existing Todo ID. The server rejects an ambiguous
reference that matches both. It assigns stable `todo-N` IDs in input order,
resolves references, validates the combined graph, and appends the entire batch
atomically. Partial creation is forbidden.

The V1 limits are fixed and server-owned:

- at most 12 Todos in one Research Run;
- at most 8 direct blockers on one Todo;
- at most 4 Todo nodes on the longest dependency path;
- at most 2 execution attempts per Todo;
- duplicate draft keys, duplicate blocker references, blank titles, unknown
  references, self-dependencies, and cycles are invalid.

Edges are immutable after creation. Replanning appends replacement Todos and
skips obsolete pending/blocked Todos instead of rewriting history.

### 4. Persisted status versus derived readiness

Unresolved `blocked_by` edges do not set `status="blocked"`. Readiness is a
derived view:

```python
ready(todo) = (
    todo.status == "pending"
    and all(by_id[dep].status == "completed" for dep in todo.blocked_by)
)
```

The complete view contains:

- `ready_ids`: pending Todos whose blockers are all completed;
- `waiting_ids`: pending Todos with a pending or in-progress blocker;
- `upstream_blocked_ids`: pending Todos with a blocked or skipped blocker;
- `in_progress_ids`, `completed_ids`, `blocked_ids`, and `skipped_ids` from
  persisted status.

`blocked` is reserved for a Todo whose own execution failed, timed out, returned
no usable result, or reached its attempt limit. `skipped` means the supervisor
decided the Todo is no longer required. A pending descendant of either remains
pending and appears in `upstream_blocked_ids`; it does not silently become
completed or executable.

An `in_progress` Todo found at loop entry is an interrupted checkpoint. The
policy converts it to `blocked` and records `todo_interrupted`; the Agent may
explicitly retry it if the attempt limit permits. Normal tool execution claims
and finishes a Todo inside one graph-node invocation, so `in_progress` is not a
normal steady checkpoint state.

### 5. Server-owned state transitions

The Agent may request only two lifecycle changes through `update_todos`:

- `blocked -> pending` to retry when `attempts < 2` and its dependencies permit
  future execution;
- `pending | blocked -> skipped` to remove obsolete work from the active plan.

The Agent cannot set `in_progress`, `completed`, evidence IDs, result refs,
owners, IDs, or dependencies.

The server owns execution transitions:

```text
pending --claim--> in_progress --verified result--> completed
                              \--failure/empty----> blocked
blocked --Agent retry request---------------------> pending
pending|blocked --Agent skip request--------------> skipped
```

Claiming increments `attempts` and is allowed only for a member of the current
ready frontier. Completion requires at least one server-validated Evidence ID
or a server-created non-retrieval `result_ref`.

For retrieval work, completion means the requested retrieval operation produced
a validated result set for that Todo. It does not mean one Parent is sufficient
to answer the entire user question. All selected complementary Parents remain
in the evidence set, and the existing Evidence Grader remains the authority on
global answer sufficiency.

### 6. Action guards

The structured Research actions become:

- `create_todos(items=[...])` appends one atomic DAG fragment;
- `update_todos(updates=[...])` requests only retry or skip;
- `retrieve_evidence(todo_id, query)` requires one ready Todo;
- `delegate_research(todo_ids)` requires unique ready Todos;
- `calculator(todo_id, expression)` requires one ready Todo and stores a
  server-created result reference;
- `submit_evidence(evidence_ids)` requires a terminal Todo plan and verified
  evidence;
- `cannot_answer(reason)` terminates safely.

Every execution action is checked against a freshly computed DAG view after the
model response is parsed. Prompt instructions are advisory; server action guards
are authoritative.

`delegate_research` may contain multiple Todos from the same ready frontier.
Because every selected ID must already be ready before the batch begins, a Todo
cannot depend on another Todo in that same batch. The existing per-Run Subagent
semaphore continues to cap physical parallelism at three.

### 7. Dependency outputs flow downstream

Scheduling order alone is insufficient for a useful dependency. Each completed
Todo retains its server-derived `evidence_ids` or `result_ref`. For every ready
Todo, the loop builds dependency input from its direct blockers:

```python
class TodoDependencyInput(BaseModel):
    todo_id: str
    evidence_ids: tuple[str, ...]
    result_ref: str | None
```

The supervisor context receives these inputs with the complete current packed
evidence. A delegated child additionally receives:

- dependency Todo IDs and their evidence/result references;
- a bounded `dependency_context` rendered only from referenced evidence;
- the current server-owned Evidence Manifest at dispatch time;
- immutable scope, retrieval filters, and read-only Memory summary.

The manifest and dependency context are invocation inputs, not constructor-time
snapshots. This prevents a downstream child from seeing the empty/stale manifest
that existed when the Query worker composed the dispatcher.

### 8. Context contract

Before every model action, `ContextBuilder` exposes:

```json
{
  "transition_contract": {
    "plan_required": false,
    "ready_todo_ids": ["todo-2", "todo-3"],
    "waiting_todo_ids": ["todo-4"],
    "upstream_blocked_todo_ids": [],
    "blocked_todo_ids": [],
    "retryable_todo_ids": [],
    "known_evidence_ids": ["evidence-1"],
    "unresolved_todos_remain": true,
    "submit_evidence_valid": false
  }
}
```

For one compatibility release, `available_todo_ids` remains present but is an
exact alias of `ready_todo_ids`; it no longer means all pending/in-progress
Todos.

`plan_required` is true when no Todo exists. It is also true when the Evidence
Grader has returned new gaps after a previous submission and no new active Todo
has been added yet. Creating a new DAG fragment clears the prior submitted flag
and makes the new work active.

`submit_evidence_valid` is true only when:

- at least one Todo exists;
- at least one server-verified Evidence ID exists;
- no Todo is pending or in progress;
- every submitted Evidence ID belongs to the current packed manifest.

Blocked and skipped Todos are terminal for this predicate. The Agent may submit
partial verified evidence, but the post-Research Evidence Grader can still mark
it insufficient and return explicit gaps. On re-entry the Agent may append a
new DAG fragment, subject to the same global action and Todo limits.

### 9. One-action graph self-loop and budgets

`ResearchAgentLoop.ainvoke()` executes at most one model action. Non-terminal
results set `next_node="research_agent"`; `submit_evidence` sets
`next_node="generate"`; terminal failure sets `next_node="end"`.

`QueryGraph.after_research()` routes `research_agent` back to the existing
`research_agent_loop` node. LangGraph writes a checkpoint between node
invocations. `research_attempt_count` remains a Run-global counter and never
resets after Evidence Grader re-entry.

The default `max_research_rounds` becomes 6 and its allowed maximum becomes 8.
A four-node dependency path requires at most six normal Agent actions: create
the plan, execute four successive frontiers, and submit. The existing 300-second
Run timeout remains the outer deadline and can terminate earlier. The graph
recursion limit remains 50.

Reconciliation, readiness calculation, deterministic ID assignment, claim, and
result reduction do not increment `research_attempt_count`; each structured
model decision increments it exactly once.

### 10. Failure handling

- Invalid old checkpoint data, duplicate IDs, unknown dependencies, or a cycle
  fails closed before any model or tool call.
- A model-selected waiting, upstream-blocked, missing, completed, or skipped
  Todo produces `todo_not_ready` and does not invoke a tool.
- Partial Subagent completion preserves validated results. Timed-out/empty
  children become blocked while successful children become completed.
- Mixed index generations, missing raw batches, evidence-manifest mismatch, or
  an over-budget merge remains a terminal evidence failure.
- When the action limit is reached, pending and in-progress Todos become blocked,
  the Run terminates with `research_round_limit`, and no additional model call
  occurs.
- Parent cancellation still cancels every child task and propagates
  `CancelledError`.

## End-to-End Example

For `刘泽明教育经历`, the first Agent action creates:

```text
todo-1 定位当前简历
   |
   v
todo-2 提取教育经历
   |
   v
todo-3 核对学校、专业、学历和日期
```

Execution proceeds as follows:

1. Only `todo-1` is ready. The Agent delegates it; validated evidence completes
   it and stores its Evidence IDs.
2. The next checkpoint makes `todo-2` ready. Its child receives `todo-1`'s
   evidence references and bounded text, then retrieves the education section.
3. The next checkpoint makes `todo-3` ready. It receives the extracted section
   as dependency context and performs a narrower cross-check retrieval.
4. All Todos are terminal and the packed manifest is non-empty, so
   `submit_evidence` is valid.
5. The existing Evidence Builder rebuilds from raw batches, the Evidence Grader
   checks sufficiency, and only then does the Generator create the user answer.

The final answer is not a Todo and no Subagent can generate it.

## File Responsibility Map

- `src/agentic_rag/query/todos.py`: canonical Todo models, checkpoint field
  migration, DAG validation/view, stable creation, and server-owned transitions.
- `src/agentic_rag/query/research_loop.py`: structured task actions, one-action
  loop step, action guards, claims, result reduction, and submission gate.
- `src/agentic_rag/query/context.py`: dependency-aware transition contract and
  dependency output projection.
- `src/agentic_rag/query/subagents.py`: ready-frontier defense, dynamic manifest,
  and bounded dependency inputs for children.
- `src/agentic_rag/query/graph.py`: self-loop routing and per-step/final Research
  events using the existing node.
- `src/agentic_rag/query/state.py`: checkpoint-safe state declaration only; no
  client or scheduler object enters state.
- `src/agentic_rag/runtime/models.py`, `src/agentic_rag/config.py`,
  `src/agentic_rag/runtime/query_composition.py`, and
  `src/agentic_rag/api/query_runs.py`: immutable six-action default/eight-action
  maximum snapshot contract.
- `src/agentic_rag/prompts/research_agent_v1.md`: action semantics and
  ready-frontier instructions.
- Tests under `tests/unit/query`, `tests/unit/runtime`, and `tests/unit`, plus the
  SQLite checkpoint integration test: DAG, transition, prompt, graph, budget,
  compatibility, cancellation, and recovery coverage.

## Acceptance Criteria

- A single `create_todos` action can create the three-node example and persists
  only resolved `blocked_by` Todo IDs.
- Old checkpoint JSON containing only `dependencies` loads and is rewritten
  using only `blocked_by` after one successful step.
- Cycles, missing references, duplicate keys/IDs/edges, excessive size/depth,
  and mixed old/new dependency fields are rejected before execution.
- Context and server guards expose/accept only the ready DAG frontier.
- Completing a prerequisite automatically makes its pending dependent ready at
  the next Agent action without a scheduler node or model status update.
- Two independent ready Todos can run concurrently; their shared dependent
  cannot join the same dispatch batch.
- Downstream supervisor and child calls receive current dependency evidence and
  the current manifest.
- Model actions cannot forge completion evidence or set server-owned statuses.
- Direct retrieval, calculator, successful delegation, timeout, retry, skip,
  submit, action-limit, cancellation, and checkpoint re-entry paths have tests.
- A real graph test completes create -> execute -> execute -> submit by revisiting
  the same `research_agent_loop` node, with checkpoint-safe JSON after every
  action.

## Approval Boundary

This design authorizes an implementation plan and subsequent code changes only
when the user explicitly starts plan execution. It does not authorize deleting
indices, rebuilding documents, changing uploaded data, or performing production
operations.
