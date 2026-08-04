# Agentic RAG Phase 4 Query Runtime Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Deliver durable Fast RAG and true Agentic research Runs with memory, dynamic Todo planning, parallel Subagents, mandatory evidence/audit gates, cancellation and reconnectable APIs.

**Architecture:** API/Run Gateway persists each Query Run and Outbox notification; a single Query Worker Claims it and invokes a checkpointed QueryGraph under bounded concurrency. QueryGraph loads Memory once, routes fast or research, uses the deterministic Retrieval Subgraph, packs evidence, generates structured answer segments and fails closed through mandatory audit nodes.

**Tech Stack:** FastAPI, LangGraph StateGraph and Send, SQLite Checkpointer, DeepSeek through an OpenAI-compatible adapter, mem0ai 2.0.12 AsyncMemory, Redis Streams, MySQL Run/Event repositories, pytest.

## Global Constraints

- Prerequisites: Phases 1–3 gates pass; the Query implementation must consume their exact ports rather than query ES/MySQL directly.
- Query Worker count remains one; default concurrency is Run 4, LLM 8, Reranker 1 and Subagent 3 per Run.
- Main Agent/generation model is `deepseek-v4-pro`; routing, grading, audit, compaction and memory extraction use `deepseek-v4-flash`.
- Memory is loaded once before routing. Do not expose `search_memory` inside ResearchAgentLoop.
- Agent-visible actions are only `update_todos`, `retrieve_evidence`, `delegate_research`, `calculator`, `submit_evidence` and `cannot_answer`.
- Every normal answer passes Evidence Grader, Faithfulness Audit and deterministic Citation Validator; one answer revision is allowed.
- No hidden reasoning or unreviewed draft is persisted or streamed.

---

### Task 1: ModelGateway and Structured Output Validation

**Files:**
- Create: `src/agentic_rag/runtime/model_gateway.py`
- Create: `src/agentic_rag/models/schemas.py`
- Create: `src/agentic_rag/prompts/__init__.py`
- Create: `src/agentic_rag/prompts/router_v1.md`
- Create: `src/agentic_rag/prompts/research_agent_v1.md`
- Create: `src/agentic_rag/prompts/evidence_grader_v1.md`
- Create: `src/agentic_rag/prompts/generator_v1.md`
- Create: `src/agentic_rag/prompts/faithfulness_v1.md`
- Create: `src/agentic_rag/prompts/context_compactor_v1.md`
- Create: `src/agentic_rag/prompts/memory_extractor_v1.md`
- Create: `tests/unit/runtime/test_model_gateway.py`

**Interfaces:**
- Produces: `ModelGateway.complete()`, `ModelGateway.complete_structured()`, `ModelResponse`, prompt Hash/version loader.
- Consumes: model IDs and retry/timeout settings from `RuntimeConfigSnapshot`.

- [ ] **Step 1: Write structured-response and retry-ownership tests**

```python
async def test_structured_call_rejects_invalid_schema_without_returning_partial(fake_client):
    fake_client.responses = ['{"route":"unknown"}', '{"route":"fast_rag","normalized_query":"q","reason_code":"simple"}']
    result = await ModelGateway(fake_client).complete_structured(ROUTE_CALL, RouteDecision)
    assert result.value.route == "fast_rag"
    assert result.attempts == 2
```

- [ ] **Step 2: Verify red state**

Run: `pytest tests/unit/runtime/test_model_gateway.py -q`
Expected: FAIL because ModelGateway is absent.

- [ ] **Step 3: Implement one retry owner and immutable prompt versions**

```python
class ModelResponse(BaseModel, Generic[T]):
    value: T
    requested_model: str
    actual_model: str
    input_tokens: int
    output_tokens: int
    attempts: int
    latency_ms: int

class ModelGateway(Protocol):
    async def complete(self, call: ModelCall) -> ModelResponse[str]: ...
    async def complete_structured(self, call: ModelCall, schema: type[T]) -> ModelResponse[T]: ...
```

Use the provider's OpenAI-compatible endpoint behind an injected client. Retry only timeout, 429, 5xx and temporary connection errors with exponential backoff plus jitter, at most two retries. Schema errors receive one repair attempt within this Gateway; Graph Nodes do not repeat an already retried call. Each prompt file has fixed `ROLE`, `TRUST BOUNDARY`, `INPUT`, `ALLOWED DECISIONS/ACTIONS`, `OUTPUT SCHEMA` and `FAIL-CLOSED RULES` sections. Research prompt lists only the six approved actions; generator forbids Evidence IDs outside the Manifest; grader/auditor state their non-overlapping responsibilities. Load prompts from these versioned files and store their content Hash in the Run snapshot.

- [ ] **Step 4: Run gateway tests**

Run: `pytest tests/unit/runtime/test_model_gateway.py -q && mypy src/agentic_rag/runtime/model_gateway.py`
Expected: PASS for timeout retry, schema repair, actual model ID and usage capture.

- [ ] **Step 5: Commit ModelGateway**

```bash
git add src/agentic_rag/runtime/model_gateway.py src/agentic_rag/models src/agentic_rag/prompts tests/unit/runtime
git commit -m "feat: add structured model gateway"
```

### Task 2: mem0ai MemoryService Boundary

**Files:**
- Create: `src/agentic_rag/memory/__init__.py`
- Create: `src/agentic_rag/memory/models.py`
- Create: `src/agentic_rag/memory/service.py`
- Create: `src/agentic_rag/memory/mem0_adapter.py`
- Create: `tests/unit/memory/test_service.py`
- Create: `tests/integration/memory/test_mem0_adapter.py`

**Interfaces:**
- Produces: `MemoryService.load_context()`, `extract_and_store()`, `list()`, `delete()`, `reconcile_deletions()`.
- Consumes: mem0ai `AsyncMemory`, `MemoryTombstoneRepository`, ModelGateway light model, UserScope.

- [ ] **Step 1: Write user-source-only memory and tombstone tests**

```python
async def test_assistant_claim_is_not_written_without_user_confirmation(memory_service):
    await memory_service.extract_and_store(
        scope=UserScope(user_id="u1"), run_id="r1",
        messages=[user("我喜欢简洁回答"), assistant("你住在上海")],
    )
    stored = await memory_service.list(UserScope(user_id="u1"))
    assert any("简洁" in item.text for item in stored)
    assert all("上海" not in item.text for item in stored)
```

- [ ] **Step 2: Verify red state**

Run: `pytest tests/unit/memory/test_service.py -q`
Expected: FAIL because MemoryService is absent.

- [ ] **Step 3: Implement the thin Mem0 boundary**

```python
class MemoryType(StrEnum):
    SEMANTIC = "semantic"
    EPISODIC = "episodic"
    PROCEDURAL = "procedural"

class MemoryService(Protocol):
    async def load_context(self, scope: UserScope, query: str, limit: int = 10) -> MemoryContext: ...
    async def extract_and_store(self, scope: UserScope, run_id: str, messages: Sequence[PublicMessage]) -> None: ...
    async def list(self, scope: UserScope) -> list[MemoryRecord]: ...
    async def delete(self, scope: UserScope, memory_id: str) -> None: ...
```

Configure mem0ai with `agent_memories_v1`, Qwen 1024-dimensional embeddings and the same `user_id` namespace. Metadata includes Memory type, source Run/message IDs and policy version. Wrap loaded Memory in an untrusted Data Envelope. Deletion writes MySQL Tombstone first, calls Mem0 delete, and marks complete only after a subsequent search no longer returns the Memory.

- [ ] **Step 4: Run unit and local Mem0/ES tests**

Run: `pytest tests/unit/memory/test_service.py -q && pytest -m integration tests/integration/memory/test_mem0_adapter.py -q`
Expected: user isolation, source filtering, graceful read/write degradation and deletion reconciliation pass.

- [ ] **Step 5: Commit memory boundary**

```bash
git add src/agentic_rag/memory tests/unit/memory tests/integration/memory
git commit -m "feat: integrate scoped mem0 memory"
```

### Task 3: Query State, Memory Load, Router and Fast RAG

**Files:**
- Create: `src/agentic_rag/query/state.py`
- Create: `src/agentic_rag/query/router.py`
- Create: `src/agentic_rag/query/fast_rag.py`
- Create: `tests/unit/query/test_router_fast_path.py`

**Interfaces:**
- Produces: `QueryState`, `RouteDecision`, `MemoryContextLoader`, `route_query()`, `run_fast_rag()`.
- Consumes: ModelGateway, MemoryService and RetrievalService.

- [ ] **Step 1: Write route and fast-path escalation tests**

```python
async def test_fast_path_retrieves_once_then_escalates_on_insufficient(router_graph, deps):
    deps.router.return_value = RouteDecision(route="fast_rag", normalized_query="q", reason_code="simple")
    deps.grader.return_value = EvidenceGrade(decision="insufficient", gaps=("缺少第二份合同",))
    state = await router_graph.ainvoke(INITIAL_STATE)
    assert deps.retrieval.calls == 1
    assert state["next_node"] == "research_agent"
```

- [ ] **Step 2: Verify red state**

Run: `pytest tests/unit/query/test_router_fast_path.py -q`
Expected: FAIL because Query state/router are absent.

- [ ] **Step 3: Implement JSON-serializable state and strict routing**

```python
class QueryState(TypedDict):
    request: dict[str, Any]
    run_id: str
    messages: list[dict[str, Any]]
    memory_context: dict[str, Any]
    route: dict[str, Any]
    research: dict[str, Any]
    evidence: list[dict[str, Any]]
    packed_context: dict[str, Any]
    runtime_config_snapshot: dict[str, Any]
    answer: dict[str, Any]
    audit_results: list[dict[str, Any]]
    revision_count: int
    errors: list[dict[str, Any]]
    termination_reason: str | None
```

Memory loader executes exactly once before Router. Router schema is `fast_rag | research`; invalid outputs fail closed after Gateway repair. Fast RAG makes one high-level RetrievalService call and never calls ES/MySQL adapters directly.

- [ ] **Step 4: Run state and fast route tests**

Run: `pytest tests/unit/query/test_router_fast_path.py -q`
Expected: Memory loads once, fast retrieval calls once, and insufficient evidence routes to Research.

- [ ] **Step 5: Commit query entry flow**

```bash
git add src/agentic_rag/query/state.py src/agentic_rag/query/router.py src/agentic_rag/query/fast_rag.py tests/unit/query
git commit -m "feat: add memory routed fast rag"
```

### Task 4: Dynamic Todo ResearchAgentLoop and Tools

**Files:**
- Create: `src/agentic_rag/query/todos.py`
- Create: `src/agentic_rag/query/tools.py`
- Create: `src/agentic_rag/query/calculator.py`
- Create: `src/agentic_rag/query/research_loop.py`
- Create: `src/agentic_rag/query/context.py`
- Create: `tests/unit/query/test_todos.py`
- Create: `tests/unit/query/test_calculator.py`
- Create: `tests/unit/query/test_research_loop.py`

**Interfaces:**
- Produces: `TodoReducer`, `ResearchAction` union, `ResearchAgentLoop`, `ResearchToolset`, `ContextBuilder`.
- Consumes: ModelGateway main/light models, RetrievalService, EvidenceBuilder, calculator and EventRepository.

- [ ] **Step 1: Write Todo invariants and observation-loop tests**

```python
def test_completed_todo_requires_evidence_or_result_ref():
    with pytest.raises(InvalidTodoTransition):
        TodoReducer.apply(ITEM_IN_PROGRESS, TodoUpdate(status="completed"))

async def test_agent_reenters_after_tool_observation(loop, scripted_model):
    scripted_model.actions = [RetrieveEvidence(request=REQ), SubmitEvidence(evidence_ids=("e1",))]
    result = await loop.ainvoke(RESEARCH_STATE)
    assert scripted_model.call_count == 2
    assert result["submitted"] is True

def test_calculator_rejects_function_calls():
    with pytest.raises(UnsafeExpression):
        SafeCalculator().evaluate("__import__('os').system('id')")
```

- [ ] **Step 2: Verify red state**

Run: `pytest tests/unit/query/test_todos.py tests/unit/query/test_research_loop.py -q`
Expected: FAIL because loop contracts are absent.

- [ ] **Step 3: Implement the action union and true loop**

```python
ResearchAction = Annotated[
    UpdateTodos | RetrieveEvidence | DelegateResearch | CalculatorCall | SubmitEvidence | CannotAnswer,
    Field(discriminator="action"),
]

class ResearchToolset:
    async def retrieve_evidence(self, request: RetrievalRequest, ctx: ResearchContext) -> EvidenceBatch: ...
    async def calculator(self, expression: str) -> CalculatorResult: ...
```

The Agent node always returns one structured action. Reducer rejects dependency cycles, illegal ownership changes and completion without Evidence/result reference. Calculator parses Python expression AST and permits only numeric literals, parentheses and `+ - * / // % **`; it rejects names, attributes, calls, containers and results exceeding configured magnitude. Tool Observation returns to the Agent. ContextBuilder retains system constraints, original question, Memory summary, unfinished Todos, latest Observation, Evidence Manifest and grader gaps; when estimated context exceeds 16000 Tokens, the light model compacts only older Observations/completed Todo detail while raw public messages/events remain in MySQL. Enforce four research rounds and Graph recursion limit 50.

- [ ] **Step 4: Run Todo/loop tests**

Run: `pytest tests/unit/query/test_todos.py tests/unit/query/test_calculator.py tests/unit/query/test_research_loop.py -q`
Expected: dynamic creation/revision, multi-hop retrieval, cannot-answer and loop-limit behavior pass.

- [ ] **Step 5: Commit ResearchAgentLoop**

```bash
git add src/agentic_rag/query/todos.py src/agentic_rag/query/tools.py src/agentic_rag/query/calculator.py src/agentic_rag/query/research_loop.py src/agentic_rag/query/context.py tests/unit/query
git commit -m "feat: add dynamic research agent loop"
```

### Task 5: Parallel Research Subagents

**Files:**
- Modify: `src/agentic_rag/query/research_loop.py`
- Create: `src/agentic_rag/query/subagents.py`
- Create: `src/agentic_rag/runtime/concurrency.py`
- Create: `tests/unit/query/test_subagents.py`

**Interfaces:**
- Produces: `ConcurrencyManager`, `SubagentDispatcher.delegate()`, `EvidenceReducer.merge()` and LangGraph `Send` branch integration.
- Consumes: restricted ResearchAgentLoop and Runtime settings.

- [ ] **Step 1: Write fan-out, partial-join and cancellation tests**

```python
async def test_subagents_are_bounded_and_partial_results_survive_timeout(dispatcher):
    result = await dispatcher.delegate(INDEPENDENT_ITEMS, max_parallel=3, timeout_seconds=20)
    assert dispatcher.max_observed_parallelism <= 3
    assert result.completed_evidence
    assert result.blocked_todo_ids == ("slow-todo",)
```

- [ ] **Step 2: Verify red state**

Run: `pytest tests/unit/query/test_subagents.py -q`
Expected: FAIL because dispatcher/reducer are absent.

- [ ] **Step 3: Implement per-invocation Subgraphs and deterministic merge**

Implement `ConcurrencyManager` with injected Run/LLM/Reranker limits and a per-Run Subagent Semaphore. Supervisor may delegate only Todos without unresolved dependencies. Each `Send` payload contains assigned question, server-built Filter, read-only Memory summary and Evidence Manifest. Subagents cannot delegate again or generate final answers. Use per-invocation state, sort reducer input by Todo ID, deduplicate by stable Evidence ID, keep completed results on timeout, mark unfinished Todos blocked, and cancel all children when parent Run ends.

```python
class SubagentDispatcher:
    async def delegate(
        self,
        items: Sequence[Todo],
        context: ResearchContext,
        max_parallel: int = 3,
        timeout_seconds: int = 20,
    ) -> SubagentJoinResult: ...

class EvidenceReducer:
    def merge(self, results: Sequence[SubagentResult]) -> tuple[EvidenceItem, ...]: ...
```

- [ ] **Step 4: Run Subagent tests**

Run: `pytest tests/unit/query/test_subagents.py -q`
Expected: bound, isolation, deterministic reducer, timeout and cancellation tests pass.

- [ ] **Step 5: Commit Multi-Agent execution**

```bash
git add src/agentic_rag/query/research_loop.py src/agentic_rag/query/subagents.py src/agentic_rag/runtime/concurrency.py tests/unit/query/test_subagents.py
git commit -m "feat: add bounded research subagents"
```

### Task 6: Generation and Mandatory Audit Nodes

**Files:**
- Create: `src/agentic_rag/query/generation.py`
- Create: `src/agentic_rag/query/audit.py`
- Create: `tests/unit/query/test_audit.py`

**Interfaces:**
- Produces: `AnswerDraft`, `EvidenceGrader`, `FaithfulnessAuditor`, `CitationValidator`, `render_final_answer()`.
- Consumes: PackedEvidence, ModelGateway and current UserScope/version repository checks.

- [ ] **Step 1: Write claim coverage, invalid citation and revision-limit tests**

```python
def test_citation_validator_requires_evidence_for_every_content_segment():
    draft = AnswerDraft(segments=(AnswerSegment(kind="content", text="期限为三年", evidence_ids=()),))
    result = CitationValidator().validate(draft, PACKED_EVIDENCE, UserScope(user_id="u1"))
    assert result.passed is False

async def test_second_failed_revision_refuses(graph):
    result = await graph.ainvoke(state_with_two_failed_audits())
    assert result["termination_reason"] == "audit_failed"
    assert result["answer"] == {}
```

- [ ] **Step 2: Verify red state**

Run: `pytest tests/unit/query/test_audit.py -q`
Expected: FAIL because audit types and nodes are absent.

- [ ] **Step 3: Implement the three gates and one shared repair loop**

```python
class AnswerSegment(BaseModel):
    kind: Literal["content", "heading", "separator", "references"]
    text: str
    evidence_ids: tuple[str, ...] = ()

class AnswerDraft(BaseModel):
    segments: tuple[AnswerSegment, ...]
```

Evidence Grader returns `sufficient | insufficient | clarify | refuse` plus gaps. Faithfulness checks semantic support only. Citation Validator deterministically requires every content Segment to cite Manifest Evidence owned by the current user and active version; only format kinds may omit Evidence. Faithfulness or citation failure returns to Generate once, then refuses without returning the draft.

- [ ] **Step 4: Run audit tests**

Run: `pytest tests/unit/query/test_audit.py -q`
Expected: sufficient, research-gap, clarify, refuse, repair and fail-closed paths pass.

- [ ] **Step 5: Commit audits**

```bash
git add src/agentic_rag/query/generation.py src/agentic_rag/query/audit.py tests/unit/query/test_audit.py
git commit -m "feat: enforce answer evidence audits"
```

### Task 7: Compile QueryGraph

**Files:**
- Create: `src/agentic_rag/query/graph.py`
- Create: `tests/unit/query/test_graph.py`

**Interfaces:**
- Produces: `build_query_graph(deps, checkpointer) -> CompiledStateGraph`.
- Consumes: Tasks 2–6 and Phase 3 Retrieval/Evidence services.

- [ ] **Step 1: Write fixed-path graph topology tests**

```python
async def test_every_normal_answer_passes_all_three_gates(query_graph, events):
    result = await query_graph.ainvoke(SIMPLE_REQUEST, config=THREAD_CONFIG)
    assert result["termination_reason"] == "completed"
    types = events.types_for(result["run_id"])
    assert types.index("EVIDENCE_GRADED") < types.index("FAITHFULNESS_AUDITED") < types.index("CITATION_VALIDATED")
```

- [ ] **Step 2: Verify red state**

Run: `pytest tests/unit/query/test_graph.py -q`
Expected: FAIL because QueryGraph is absent.

- [ ] **Step 3: Compile the approved macro graph**

```text
START -> memory_loader -> route
route -> fast_rag | research_agent_loop
fast/research -> evidence_grader
grader sufficient -> evidence_builder -> generate -> faithfulness -> citation -> finalize
grader insufficient -> research_agent_loop
grader clarify/refuse -> terminal
audit failure -> generate once -> refuse on second failure
```

Before calling EvidenceBuilder, map the Fast-path question or current Todo list into Phase 3 `EvidenceCoverageTarget` values; do not introduce a reverse dependency from Phase 3 to Todo types. Use the Query SQLite Checkpointer and `checkpoint_thread_id={user_id}:{thread_id}`. Finalize persists the public final answer/events first, then invokes MemoryService extraction as a degradable post-finalize side effect.

- [ ] **Step 4: Run QueryGraph tests**

Run: `pytest tests/unit/query/test_graph.py tests/unit/query/test_router_fast_path.py tests/unit/query/test_research_loop.py tests/unit/query/test_audit.py -q`
Expected: all macro routes and mandatory gates pass.

- [ ] **Step 5: Commit QueryGraph**

```bash
git add src/agentic_rag/query/graph.py tests/unit/query/test_graph.py
git commit -m "feat: compile audited query graph"
```

### Task 8: Run Manager, Concurrency and Query Worker

**Files:**
- Create: `src/agentic_rag/runtime/run_manager.py`
- Modify: `src/agentic_rag/runtime/concurrency.py`
- Create: `src/agentic_rag/runtime/query_worker.py`
- Create: `scripts/run_query_worker.py`
- Create: `tests/integration/runtime/test_query_worker.py`

**Interfaces:**
- Produces: `RunManager.create()`, `request_cancel()`, `QueryWorker.run_one()`/`run_forever()`.
- Consumes: Run/Outbox/Event repositories, Redis broker, QueryGraph, Checkpoint backend and `ConcurrencyManager` from Task 5.

- [ ] **Step 1: Write durable creation, single-thread and crash-resume tests**

```python
@pytest.mark.integration
async def test_create_run_and_outbox_are_atomic(run_manager, db):
    run = await run_manager.create(SCOPE, "thread-1", QUERY, SNAPSHOT)
    assert await db.has_outbox("query_run", run.id)

@pytest.mark.integration
async def test_worker_resumes_same_run_after_crash(worker_fixture):
    await worker_fixture.run_one(fail_after="retrieval")
    result = await worker_fixture.run_one()
    assert result.status == RunStatus.COMPLETED
    assert worker_fixture.duplicate_event_keys == set()
```

- [ ] **Step 2: Verify red state**

Run: `pytest -m integration tests/integration/runtime/test_query_worker.py -q`
Expected: FAIL because runtime classes are absent.

- [ ] **Step 3: Implement durable execution and cooperative cancellation**

```python
class ConcurrencyManager:
    def __init__(self, run_limit: int = 4, llm_limit: int = 8, rerank_limit: int = 1):
        self.run_slots = asyncio.Semaphore(run_limit)
        self.llm_slots = asyncio.Semaphore(llm_limit)
        self.rerank_slots = asyncio.Semaphore(rerank_limit)

class RunManager(Protocol):
    async def create(self, scope: UserScope, thread_id: str, query: str,
                     snapshot: RuntimeConfigSnapshot) -> QueryRun: ...
    async def request_cancel(self, scope: UserScope, run_id: str) -> RunStatus: ...
```

Create Run and Outbox atomically. Worker consumes `agenticrag:jobs:query`, Claims MySQL lease, checks cancel before each Graph node/tool/subagent, heartbeats, invokes Graph with the stable checkpoint thread, and ACKs only after terminal status. Reclaim expired pending entries with `XAUTOCLAIM`; dead-letter after three Claim attempts. Enforce 300-second Run timeout and graceful shutdown that stops new Claims before waiting for active nodes.

- [ ] **Step 4: Run runtime integration tests**

Run: `pytest -m integration tests/integration/runtime/test_query_worker.py -q`
Expected: duplicate notifications, active-thread conflict, cancellation, timeout, lease reclaim and checkpoint resume pass without duplicate events.

- [ ] **Step 5: Commit Query runtime**

```bash
git add src/agentic_rag/runtime scripts/run_query_worker.py tests/integration/runtime
git commit -m "feat: add durable query worker"
```

### Task 9: Query, SSE, Memory and Feedback APIs

**Files:**
- Create: `src/agentic_rag/api/query_runs.py`
- Create: `src/agentic_rag/api/memories.py`
- Create: `src/agentic_rag/api/feedback.py`
- Modify: `src/agentic_rag/api/app.py`
- Create: `tests/integration/api/test_query_runs.py`
- Create: `tests/e2e/test_query_runtime.py`

**Interfaces:**
- Produces: all `/v1/query-runs`, `/v1/query`, `/v1/memories` and `/v1/feedback` endpoints from the design.
- Consumes: RunManager, EventRepository, MemoryService and UserScope dependency.

- [ ] **Step 1: Write SSE reconnection, sync-wrapper and cancellation API tests**

```python
@pytest.mark.integration
async def test_sse_reconnect_uses_mysql_event_cursor(client, completed_run):
    first = await read_sse(client, completed_run.id, stop_after=2)
    resumed = await read_sse(client, completed_run.id, last_event_id=first[-1].id)
    assert not ({e.id for e in first} & {e.id for e in resumed})

async def test_sync_query_returns_202_after_wait_timeout(client, slow_run):
    response = await client.post("/v1/query", json={"query": "复杂研究", "wait_seconds": 0})
    assert response.status_code == 202
    assert response.json()["run_id"]
```

- [ ] **Step 2: Verify red state**

Run: `pytest -m integration tests/integration/api/test_query_runs.py -q`
Expected: FAIL because routes are absent.

- [ ] **Step 3: Implement scoped APIs and MySQL-backed SSE**

`POST /v1/query-runs` returns 202. GET and cancel verify Run ownership. SSE polls `agent_events.id > Last-Event-ID`, emits heartbeat comments and redacted public progress only. `/v1/query` creates the same durable Run, waits at most 30 seconds, returns 200 only for an audited completed result and otherwise returns 202 without starting a second Run. Memory deletion is tombstone-backed; feedback verifies Run ownership before appending `USER_FEEDBACK`.

```python
class QueryRequest(BaseModel):
    query: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
    thread_id: str | None = None
    wait_seconds: int = Field(default=30, ge=0, le=30)
```

If `(user_id, thread_id)` already has an active Run, return unified 409 `ACTIVE_RUN_EXISTS` and set `Location: /v1/query-runs/{existing_run_id}`; do not enqueue a duplicate. All other failures use Phase 1 `ApiError`.

- [ ] **Step 4: Run the Phase 4 gate**

Run: `pytest tests/unit/query tests/unit/memory tests/unit/runtime -q && pytest -m integration tests/integration/runtime tests/integration/api tests/integration/memory -q && pytest -m e2e tests/e2e/test_query_runtime.py -q`
Expected: Fast RAG, multi-hop, Subagent, audit repair/refuse, restart, cancel, SSE reconnect, Memory isolation and feedback all pass.

- [ ] **Step 5: Commit public Query APIs**

```bash
git add src/agentic_rag/api tests/integration/api tests/e2e/test_query_runtime.py
git commit -m "feat: expose durable query apis"
```
