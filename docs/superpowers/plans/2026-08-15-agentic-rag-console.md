# Agentic RAG 控制台与 Query Runtime Gaps Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 修复 Query Runtime 的子 Agent、Todo 与全局研究预算缺口，并交付一个由 FastAPI 同源托管、可通过真实 Graph/API 查询验收的 Agentic RAG 中文控制台。

**Architecture:** QueryState 继续只保存 JSON 值，但增加跨 Graph 重入持久化的 `research_attempt_count` 和服务器生成的 Todo；生产组合根创建真正的 `SubagentDispatcher`，并把同一个 `ConcurrencyManager` 传给 QueryGraph 与 QueryWorker。FastAPI 新增无密钥运行时摘要端点和静态控制台，页面通过现有 Query Run/SSE、文档、Mem0、Health API 获取状态，所有降级、熔断、重试、审计和拒答均由后端白名单字段驱动。最终门禁使用当前 RuntimeConfigSnapshot、真实 client provenance、Graph/API 模式和隔离 MySQL/Redis/Elasticsearch/Mem0 资源，不以 fixture 结果替代真实验收。

**Tech Stack:** Python 3.11、FastAPI、LangGraph、原生 HTML/CSS/JavaScript、Pydantic v2、pytest/pytest-asyncio、MySQL、Redis Streams、Elasticsearch、mem0ai 2.0.12、DeepSeek/Qwen OpenAI-compatible clients。

## Global Constraints

- 所有命令使用 `conda run -n agentic-rag` 前缀，不依赖 `.venv`，不引入 Node/npm 构建链。
- QueryState、SSE payload、运行时摘要只能包含 JSON 基本值和后端白名单字段；不得向浏览器、事件、Artifact 或日志写入 prompt、隐藏推理、原始工具载荷、Memory 原文、Authorization、API key 或 provider 原始响应。
- `user_id` 只由服务端 `Settings.default_user_id` 注入；页面请求体和静态脚本不得携带可切换的用户标识。
- 研究全局上限使用当前 `RuntimeConfigSnapshot.max_research_rounds`，在整个 Query 生命周期累计，不因 Grader 重新进入 ResearchAgentLoop 而归零；超限必须产生安全事件并以 `research_round_limit` 业务终态结束。
- 子 Agent 只能使用服务器创建的 `ResearchContext`、作用域过滤和当前 snapshot；不得递归创建子 Agent、修改父级答案或替换 evidence provenance。
- 所有可恢复任务遵循 red-green-refactor：先写可观察的失败测试，再写最小实现，最后运行聚焦回归、Ruff、mypy 和 `git diff --check` 后提交。
- Mem0 默认启用；初始化或运行时故障只能进入明确的 memory degraded 状态，`GET/DELETE /v1/memories` 不得把不可用伪装为空成功。
- 最终验收必须同时满足 `user_leak_count=0`、`citation_coverage=1.0`、`unaudited_answer_count=0`、恢复门禁和备份门禁为 true，且报告包含真实 Graph/API 来源和当前 snapshot ID。

## 文件与边界地图

| 文件 | 职责 |
|---|---|
| `src/agentic_rag/query/state.py` | QueryState 的 checkpoint-safe 字段、初始化和 snapshot/scope 边界 |
| `src/agentic_rag/query/todos.py` | 服务器生成 Todo、ID、依赖和状态不变量 |
| `src/agentic_rag/query/research_loop.py` | 严格研究动作、Todo 创建/更新、全局研究计数和委派入口 |
| `src/agentic_rag/query/graph.py` | 固定 Graph 拓扑、研究计数合并、有限事件属性和终止路由 |
| `src/agentic_rag/runtime/query_composition.py` | 模型、检索、证据、子 Agent 和共享并发预算的生产组合 |
| `src/agentic_rag/runtime/query_worker.py` | Query Outbox 消费、租约、Graph 恢复、终态和生命周期事件 |
| `scripts/run_query_worker.py` | 将生产组合出的共享并发管理器注入 Worker |
| `src/agentic_rag/api/runtime_summary.py` | 不含密钥的 snapshot、模型、索引、Mem0 和依赖状态摘要 |
| `src/agentic_rag/api/app.py` | 同源静态页面、`/static` 资源和运行时摘要路由装配 |
| `src/agentic_rag/api/static/index.html` | 中文控制台结构和可访问的状态区域 |
| `src/agentic_rag/api/static/app.css` | 三栏响应式布局、状态色和移动端布局 |
| `src/agentic_rag/api/static/app.js` | Query/SSE、文档、Mem0、Health、降级提示和有限 UI 状态 |
| `tests/unit/query/test_todos.py`、`test_research_loop.py` | Todo 创建、委派和全局预算单元契约 |
| `tests/unit/runtime/test_query_composition.py` | 生产 SubagentDispatcher/Concurrency 注入契约 |
| `tests/unit/api/test_runtime_summary.py`、`tests/integration/api/test_console.py` | 摘要、静态页面和 UI 使用的 API 边界 |
| `tests/e2e/test_query_console_real_services.py` | 真实 API/Graph、SSE、snapshot、provenance、Mem0 和审计门禁 |
| `tests/unit/test_documentation_contracts.py` | 中文运行手册和总体路线图的关键契约 |
| `docs/local-operations.md`、`docs/development-progress.md`、`docs/superpowers/plans/2026-08-04-agentic-rag-implementation-roadmap.md` | 中文运行手册、进度和总体路线图 |

---

### Task 1: 创建 Todo 并实施跨 Graph 全局研究预算

**Files:**
- Modify: `src/agentic_rag/query/state.py`
- Modify: `src/agentic_rag/query/todos.py`
- Modify: `src/agentic_rag/query/research_loop.py`
- Modify: `src/agentic_rag/query/graph.py`
- Test: `tests/unit/query/test_todos.py`
- Test: `tests/unit/query/test_research_loop.py`
- Test: `tests/unit/query/test_graph.py`

**Interfaces:**
- `QueryState` 新增必填 checkpoint-safe 字段 `research_attempt_count: int`；`new_query_state(run_id: str, question: str, scope: UserScope, snapshot: RuntimeConfigSnapshot, messages: list[dict[str, Any]] | None = None)` 将其设为 `0`。
- `TodoReducer.append(todos: tuple[TodoItem, ...], titles: list[str], *, owner: str) -> tuple[TodoItem, ...]` 按现有数量分配稳定的 `todo-{n}` ID，合并后调用 `TodoReducer.validate`。
- 新增严格动作模型 `CreateTodos(action: Literal["create_todos"], titles: tuple[str, ...])`，加入 `ResearchAction` discriminator；模型不能指定 `user_id`、任意 Todo ID 或 owner。
- `ResearchAgentLoop.ainvoke(state: QueryState) -> dict[str, object]` 每次动作尝试前读取 `state["research_attempt_count"]`，递增并在返回值中携带新的 count；count 达到 snapshot 上限时返回 `termination_reason="research_round_limit"`、将未完成 Todo 标记为 blocked，且不再调用 Gateway。

- [ ] **Step 1: Write the failing tests**

```python
def test_append_todos_uses_stable_non_colliding_ids() -> None:
    existing = TodoReducer.create(["root"], owner="supervisor")
    result = TodoReducer.append(existing, ["retrieve policy", "check exception"], owner="supervisor")
    assert [item.id for item in result] == ["todo-1", "todo-2", "todo-3"]


async def test_empty_research_state_creates_root_todo_and_can_delegate() -> None:
    state = _state_without_research_todos()
    loop = ResearchAgentLoop(_deps(actions=[
        {"action": "create_todos", "titles": ["Find notice period"]},
        {"action": "delegate_research", "todo_ids": ["todo-2"]},
        {"action": "submit_evidence"},
    ], dispatcher=_recording_dispatcher()))
    result = await loop.ainvoke(state)
    assert [todo["id"] for todo in result["research"]["todos"]] == ["todo-1", "todo-2"]
    assert result["research"]["observations"][0]["kind"] == "todo_created"


async def test_research_attempt_count_survives_reentry_and_stops_at_snapshot_limit() -> None:
    first = _state_with_research_attempt_count(3)
    loop = ResearchAgentLoop(_deps(actions=[{"action": "calculator", "expression": "1+1"}]))
    update = await loop.ainvoke(first)
    assert update["research_attempt_count"] == 4
    resumed = {**first, **update}
    gateway = _deps(actions=[]).gateway
    limited = await ResearchAgentLoop(_deps(gateway=gateway)).ainvoke(resumed)
    assert limited["termination_reason"] == "research_round_limit"
    assert gateway.calls == 0
```

测试模块中的 `_state_without_research_todos()` 使用现有 `_state()` 后写入 `research={"todos": [], "observations": []}`；`_state_with_research_attempt_count(value: int)` 使用同一状态并设置 `research_attempt_count=value`；`_deps(actions: list[dict[str, object]] | None = None, *, gateway: ScriptedGateway | None = None, dispatcher: object | None = None)` 返回现有 `ResearchLoopDependencies(gateway=gateway or ScriptedGateway(actions or []), retrieval=FakeRetrieval(), evidence_builder=EvidenceBuilder(), subagents=dispatcher)`；`_recording_dispatcher()` 的 `delegate` 返回带一个非空 `PackedEvidence` 的 `DelegationResult`。这样测试同时覆盖服务器自动创建根 Todo 和严格 action schema 的追加路径。

- [ ] **Step 2: Run the tests to verify they fail**

Run:

```bash
conda run -n agentic-rag pytest --import-mode=importlib \
  tests/unit/query/test_todos.py tests/unit/query/test_research_loop.py \
  tests/unit/query/test_graph.py -q
```

Expected: FAIL because `QueryState` has no `research_attempt_count`, `TodoReducer.append` and `CreateTodos` do not exist, and the loop still uses a fresh `range(snapshot.max_research_rounds)` on every invocation.

- [ ] **Step 3: Write the minimal implementation**

Implement the following concrete flow:

```python
# research_loop.py
todos = _todos(research.get("todos"))
if not todos:
    todos = TodoReducer.create([question_from_state(state)], owner=SUPERVISOR_OWNER)
    observations.append({"kind": "todo_created", "todo_ids": [todo.id for todo in todos]})

attempt = int(state.get("research_attempt_count", 0))
while attempt < snapshot.max_research_rounds:
    attempt += 1
    action = await self._next_action(state, research, todos, observations)
    step = await self._execute(
        action, context, todos, observations, evidence, state, attempt - 1
    )
    todos, observations, evidence = step.todos, step.observations, step.evidence
    retrieval_batches.extend(step.retrieval_batches)
    research = {**research, "todos": _dump_todos(todos), "observations": observations}
    if step.submitted or step.cannot_answer:
        return _result(
            research,
            evidence,
            retrieval_batches=retrieval_batches,
            submitted=step.submitted,
            cannot_answer=step.cannot_answer,
            research_attempt_count=attempt,
            termination_reason=step.termination_reason,
        )
return _result(
    {**research, "todos": _dump_todos(_block_active(todos)), "observations": observations},
    evidence,
    retrieval_batches=retrieval_batches,
    research_attempt_count=attempt,
    termination_reason="research_round_limit",
)
```

Add `CreateTodos` to the strict union and execute it with `TodoReducer.append`; reject blank/duplicate titles with a safe observation. Update `_result` to include `research_attempt_count`, update `QueryState` initialization, and add finite numeric `research_attempt_count`/`todo_count` attributes to `RESEARCH_LOOP_COMPLETED` for the console and metrics. Keep `CancelledError`, provider errors, invalid action schemas and evidence provenance behavior unchanged.

- [ ] **Step 4: Run the focused tests and static checks**

```bash
conda run -n agentic-rag pytest --import-mode=importlib \
  tests/unit/query/test_todos.py tests/unit/query/test_research_loop.py \
  tests/unit/query/test_graph.py -q
conda run -n agentic-rag ruff check src/agentic_rag/query tests/unit/query
conda run -n agentic-rag mypy src/agentic_rag/query
```

Expected: all focused tests pass; no new Ruff or mypy errors; a second Graph research entry cannot issue a fifth action call.

- [ ] **Step 5: Commit**

```bash
git add src/agentic_rag/query/state.py src/agentic_rag/query/todos.py \
  src/agentic_rag/query/research_loop.py src/agentic_rag/query/graph.py \
  tests/unit/query/test_todos.py tests/unit/query/test_research_loop.py \
  tests/unit/query/test_graph.py
git commit -m "fix: create research todos and enforce global budget"
```

### Task 2: 将 SubagentDispatcher 接入生产组合根

**Files:**
- Modify: `src/agentic_rag/runtime/query_composition.py`
- Modify: `src/agentic_rag/query/graph.py`
- Modify: `src/agentic_rag/runtime/query_worker.py`
- Modify: `scripts/run_query_worker.py`
- Test: `tests/unit/runtime/test_query_composition.py`
- Test: `tests/unit/query/test_subagents.py`
- Test: `tests/integration/runtime/test_query_pipeline.py`

**Interfaces:**
- `build_query_dependencies(container: object, settings: Settings, *, child_index: str | None = None, concurrency: ConcurrencyManager | None = None) -> QueryGraphDependencies` 接受可注入的共享并发预算。
- `build_subagent_dispatcher(*, retrieval: RetrievalPort, evidence_builder: EvidenceBuilder, snapshot: RuntimeConfigSnapshot, concurrency: ConcurrencyManager) -> SubagentDispatcher` 是可单测的组合辅助函数，负责创建闭包 child worker。
- `QueryGraphDependencies.concurrency: ConcurrencyManager | None` 保存与子 Agent/Worker 共用的进程级预算，不进入 QueryState。
- 组合根内部创建 `SubagentDispatcher(tools=ResearchToolset(retrieval, evidence_builder), concurrency=shared_concurrency, worker=child_worker)`，并将其传给 `ResearchLoopDependencies.subagents`。
- `child_worker(child: ChildResearchState, tools: SubagentTools) -> PackedEvidence` 使用闭包捕获的当前 `RuntimeConfigSnapshot`，从 `child.scope` 重建 `UserScope`，调用 `tools.retrieve_evidence(query=child.question, context=ResearchContext(scope=child_scope, snapshot=snapshot), target_id=child.todo_id)`；不创建新的 Graph、Gateway 或客户端。
- `scripts/run_query_worker.py` 将 `dependencies.concurrency or ConcurrencyManager()` 传给 `QueryWorker(concurrency=shared_concurrency)`，确保运行槽位和子 Agent 槽位由同一进程预算管理。

- [ ] **Step 1: Write the failing tests**

```python
recorded_requests: list[tuple[object, object, object]] = []


class RecordingRetrieval:
    async def retrieve(self, request: object, scope: UserScope, snapshot: RuntimeConfigSnapshot) -> EvidenceBatch:
        recorded_requests.append((request, scope, snapshot))
        return _batch_for_scope(scope)  # fixture helper returns a provenance-valid EvidenceBatch


def test_production_composition_injects_real_subagent_dispatcher() -> None:
    dependencies = query_composition.build_subagent_dispatcher(
        retrieval=RecordingRetrieval(),
        evidence_builder=EvidenceBuilder(),
        snapshot=SNAPSHOT,
        concurrency=ConcurrencyManager(),
    )
    assert isinstance(dependencies, SubagentDispatcher)


async def test_composed_child_worker_uses_parent_scope_and_current_snapshot() -> None:
    recording_retrieval = RecordingRetrieval()
    dispatcher = query_composition.build_subagent_dispatcher(
        retrieval=recording_retrieval,
        evidence_builder=EvidenceBuilder(),
        snapshot=CONTEXT.snapshot,
        concurrency=ConcurrencyManager(),
    )
    result = await dispatcher.delegate([_todo("todo-1")], CONTEXT)
    assert result.results[0].evidence.index_generation == CONTEXT.snapshot.index_generation
    assert recorded_requests[0][1].user_id == CONTEXT.scope.user_id
```

在测试模块中定义 `_batch_for_scope(scope)`，返回与 `scope.user_id`、`SNAPSHOT.index_generation` 和 `EvidenceBuilder` manifest 完全匹配的最小 `EvidenceBatch`。再增加一条流水线测试，脚本化研究动作依次为 `create_todos`、`delegate_research`、`submit_evidence`，断言子 Agent 调用了检索 fake，且观察记录中不存在 `subagents unavailable`。

- [ ] **Step 2: Run the tests to verify they fail**

```bash
conda run -n agentic-rag pytest --import-mode=importlib \
  tests/unit/runtime/test_query_composition.py tests/unit/query/test_subagents.py \
  tests/integration/runtime/test_query_pipeline.py -q
```

Expected: FAIL because `build_query_dependencies` currently leaves `ResearchLoopDependencies.subagents` as `None` and the worker has no shared composition concurrency field.

- [ ] **Step 3: Write the minimal implementation**

Add a composition-local child worker and wire it as follows:

```python
shared = concurrency or ConcurrencyManager(
    run_limit=settings.max_concurrent_query_runs,
    llm_limit=settings.max_concurrent_llm_calls,
    reranker_limit=settings.max_concurrent_reranks,
    per_run_subagent_limit=snapshot.max_parallel_subagents_per_run,
)
tools = ResearchToolset(retrieval, evidence_builder)

async def child_worker(child: ChildResearchState, child_tools: SubagentTools) -> PackedEvidence:
    child_scope = UserScope.model_validate(dict(child.scope))
    context = ResearchContext(scope=child_scope, snapshot=snapshot)
    return await child_tools.retrieve_evidence(
        query=child.question, context=context, target_id=child.todo_id
    )

subagents = SubagentDispatcher(
    tools=tools,
    concurrency=shared,
    worker=child_worker,
    memory_summary="",
    evidence_manifest={},
)
```

Store `shared` on `QueryGraphDependencies`, pass `subagents` to `ResearchLoopDependencies`, and pass the same object into `QueryWorker`. Preserve scope/provenance checks in `EvidenceReducer`; a child failure, timeout, generation mismatch or cancellation must remain observable and fail closed.

- [ ] **Step 4: Run the focused tests and static checks**

```bash
conda run -n agentic-rag pytest --import-mode=importlib \
  tests/unit/runtime/test_query_composition.py tests/unit/query/test_subagents.py \
  tests/integration/runtime/test_query_pipeline.py -q
conda run -n agentic-rag ruff check src/agentic_rag/runtime src/agentic_rag/query scripts tests/unit/runtime tests/unit/query
conda run -n agentic-rag mypy src/agentic_rag/runtime src/agentic_rag/query
```

Expected: child retrieval is executed under the server scope, the same snapshot reaches the evidence builder, and the worker and Graph share one concurrency budget.

- [ ] **Step 5: Commit**

```bash
git add src/agentic_rag/runtime/query_composition.py src/agentic_rag/query/graph.py \
  src/agentic_rag/runtime/query_worker.py scripts/run_query_worker.py \
  tests/unit/runtime/test_query_composition.py tests/unit/query/test_subagents.py \
  tests/integration/runtime/test_query_pipeline.py
git commit -m "feat: wire bounded research subagents into production"
```

### Task 3: 增加运行时摘要并交付同源中文控制台

**Files:**
- Create: `src/agentic_rag/api/runtime_summary.py`
- Modify: `src/agentic_rag/api/app.py`
- Modify: `pyproject.toml`
- Create: `src/agentic_rag/api/static/index.html`
- Create: `src/agentic_rag/api/static/app.css`
- Create: `src/agentic_rag/api/static/app.js`
- Test: `tests/unit/api/test_runtime_summary.py`
- Test: `tests/integration/api/test_console.py`

**Interfaces:**
- `RuntimeSummaryResponse` 字段固定为：`runtime_config_snapshot_id: str`、`app_version: str`、`graph_version: str`、`main_model_id: str`、`light_model_id: str`、`deepseek_protocol: Literal["auto", "chat", "responses"]`、`embedding_model: str`、`index_generation: str`、`memory_enabled: bool`、`memory_available: bool`、`dependencies: dict[str, Literal["available", "unavailable"]]`。
- `GET /v1/runtime/summary` 从 `request.app.state.container.runtime_snapshot` 和 `readiness_checks.run()` 读取数据，任何 snapshot 解析失败返回 503；响应不包含 DSN、API key、prompt hash、Memory text 或 provider 原文。
- `GET /` 返回 `index.html`；`/static` 只服务 `app.css` 与 `app.js`，打包配置将 `api/static/*` 纳入 wheel。
- `app.js` 暴露 `loadRuntimeSummary()`、`submitQuery(question)`、`streamRun(runId)`、`loadRun(runId)`、`uploadDocument(file)`、`loadMemories()` 和 `deleteMemory(memoryId)`；SSE 重连用内存中的 `Last-Event-ID`，不把完整查询/答案/Memory 写入 LocalStorage。

- [ ] **Step 1: Write the failing tests**

```python
async def test_runtime_summary_exposes_snapshot_without_secrets() -> None:
    app = _app_with_runtime_snapshot_and_readiness()
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        response = await client.get("/v1/runtime/summary")
    assert response.status_code == 200
    body = response.json()
    assert body["runtime_config_snapshot_id"] == SNAPSHOT.snapshot_id
    assert "mysql_dsn" not in response.text
    assert "api_key" not in response.text


async def test_console_serves_same_origin_html_and_static_assets() -> None:
    app = create_app_for_static_test()
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        page = await client.get("/")
        script = await client.get("/static/app.js")
        style = await client.get("/static/app.css")
    assert page.status_code == script.status_code == style.status_code == 200
    assert 'id="query-form"' in page.text
    assert "/v1/query" in script.text
```

`_app_with_runtime_snapshot_and_readiness()` 使用 `SimpleNamespace(runtime_snapshot=SNAPSHOT, readiness_checks=ReadinessChecks({"mysql": _ok, "memory": _ok}), settings=SimpleNamespace(mem0_enabled=True))`；`create_app_for_static_test()` 使用 `create_app(settings, container=container)`，其中 `container.close`、`readiness_checks` 和页面所需路由依赖均为异步 fake，避免在静态路由测试中连接外部服务。

- [ ] **Step 2: Run the tests to verify they fail**

```bash
conda run -n agentic-rag pytest --import-mode=importlib \
  tests/unit/api/test_runtime_summary.py tests/integration/api/test_console.py -q
```

Expected: FAIL during collection or with 404 because `runtime_summary.py`, the static mount and asset files do not exist.

- [ ] **Step 3: Write the minimal implementation**

Add the summary router and mount static files in `create_app`:

```python
static_dir = Path(__file__).with_name("static")
app.mount("/static", StaticFiles(directory=static_dir), name="static")

@app.get("/", include_in_schema=False)
async def console() -> FileResponse:
    return FileResponse(static_dir / "index.html")
```

Build the page with these stable IDs: `query-form`, `query-input`, `run-status`, `timeline`, `answer`, `evidence`, `audit`, `provenance`, `degradation-banner`, `document-upload`, `ingestion-status`, `memory-list`, `health-grid`, `snapshot-id`. The JavaScript must:

1. call `/v1/runtime/summary` on load and render dependency chips;
2. post `{query, wait_seconds: 30}` to `/v1/query`, keep the returned `run_id`, and never send `user_id`;
3. consume `/v1/query-runs/{run_id}/events` with a fetch-based SSE reader, retaining only the last numeric event ID in memory;
4. render only the allowlisted event types and map unknown events to `进度更新`;
5. show explicit Chinese states for `RETRIEVAL_DEGRADED`、`CIRCUIT_OPEN`、`MODEL_REPAIR_EXHAUSTED`、`WORKER_DLQ`、`LEASE_LOST`、`COMPONENT_DEGRADED`、`research_action_invalid`、`audit_failed`、`research_round_limit`、`subagents unavailable` and empty Todo creation;
6. show answer `audited`、citation coverage、Parent IDs、route、snapshot and `client_provenance` only when the backend response contains them;
7. upload documents and poll ingestion jobs, load/delete Mem0 records, and show provider unavailable as an error card rather than an empty list.

Use semantic HTML and CSS media queries for a 3-column desktop layout that collapses to one column below 900px; do not add a frontend dependency or a client-side user switcher. Add `agentic_rag = ["api/static/*"]` to package data so the installed wheel serves the page.

- [ ] **Step 4: Run the focused tests and static checks**

```bash
conda run -n agentic-rag pytest --import-mode=importlib \
  tests/unit/api/test_runtime_summary.py tests/integration/api/test_console.py \
  tests/integration/api/test_query_runs.py -q
conda run -n agentic-rag ruff check src/agentic_rag/api tests/unit/api tests/integration/api
conda run -n agentic-rag python -m build --wheel --no-isolation
```

Expected: page, CSS and JavaScript are available from the installed package; summary and existing API routes preserve scope and redaction behavior.

- [ ] **Step 5: Commit**

```bash
git add src/agentic_rag/api/runtime_summary.py src/agentic_rag/api/app.py \
  src/agentic_rag/api/static/index.html src/agentic_rag/api/static/app.css \
  src/agentic_rag/api/static/app.js pyproject.toml \
  tests/unit/api/test_runtime_summary.py tests/integration/api/test_console.py
git commit -m "feat: add agentic rag console and runtime summary"
```

### Task 4: 用真实 Graph/API、Mem0 和当前 snapshot 完成全链路 E2E

**Files:**
- Create: `tests/e2e/test_query_console_real_services.py`
- Modify: `tests/fixtures/query_services.py` (only to expose the shared API/worker/container fixture without duplicating provider clients)
- Modify: `scripts/run_real_query_acceptance.py` (add console page/SSE assertions to the existing real API flow)
- Modify: `scripts/verify_acceptance.py` only if the new console result needs a strictly validated, already-existing gate field

**Interfaces:**
- Test setup requires `AGENTIC_RAG_RUN_REAL_QUERY_PROVIDER_E2E=1`, `AGENTIC_RAG_TEST_MYSQL_DSN`, `AGENTIC_RAG_TEST_REDIS_DSN`, `AGENTIC_RAG_TEST_ELASTICSEARCH_URL` and configured DeepSeek/Qwen/Mem0 variables; missing explicit variables produce a clear skip, while configured provider failure fails.
- The acceptance client uses `create_app(settings, container=container)`, the production `QueryWorker`, and the seeded active ES generation; it does not inject the fixture client into the acceptance path.
- Every accepted `QueryRunResponse` must match the container snapshot ID, contain `audited=true`, audited segments and at least one server evidence Parent ID; the resulting `EvalRunner` summary must contain `client_provenance="real_query_api"`, the same snapshot ID, citation coverage `1.0` and no raw secret/prompt/tool fields.
- The `real_query_runtime` fixture exposes `client: httpx.AsyncClient`, `snapshot: RuntimeConfigSnapshot`, `seeded_question: str`, `read_sse(run_id: str) -> list[dict[str, object]]`, `wait_for_terminal(run_id: str) -> dict[str, object]` and `evaluation_summary: Mapping[str, object]`.

- [ ] **Step 1: Write the failing E2E assertions**

```python
@pytest.mark.e2e
@pytest.mark.integration
async def test_console_real_graph_api_query_has_snapshot_provenance_and_gates(real_query_runtime) -> None:
    page = await real_query_runtime.client.get("/")
    assert page.status_code == 200
    created = await real_query_runtime.client.post(
        "/v1/query", json={"query": real_query_runtime.seeded_question, "wait_seconds": 30}
    )
    assert created.status_code in {200, 202}
    run_id = created.json()["run_id"]
    events = await real_query_runtime.read_sse(run_id)
    run = await real_query_runtime.wait_for_terminal(run_id)
    assert run["runtime_config_snapshot_id"] == real_query_runtime.snapshot.snapshot_id
    assert run["answer"]["audited"] is True
    assert run["answer"]["segments"]
    assert real_query_runtime.evaluation_summary["client_provenance"] == "real_query_api"
    assert real_query_runtime.evaluation_summary["runtime_config_snapshot_id"] == real_query_runtime.snapshot.snapshot_id
    assert real_query_runtime.evaluation_summary["citation_coverage"] == 1.0
    assert all(secret not in json.dumps(events, ensure_ascii=False) for secret in ("Bearer ", "sk-", "chain_of_thought"))
```

- [ ] **Step 2: Run the real test before implementation**

```bash
set -a; source .env.local; set +a
AGENTIC_RAG_RUN_REAL_QUERY_PROVIDER_E2E=1 \
conda run -n agentic-rag pytest --import-mode=importlib \
  tests/e2e/test_query_console_real_services.py -q -s
```

Expected: the new test is absent/fails until it is connected to the existing production fixture and page route; if any required service variable is absent, pytest must report the explicit opt-in skip rather than silently pass.

- [ ] **Step 3: Connect the production test path**

Reuse the existing isolated seeding, current snapshot, real Query Worker, Mem0 provider and ASGI API setup. Extend the acceptance script to record `console_page_status`, `sse_last_event_id`, `client_provenance`, `runtime_config_snapshot_id`, `memory_provider_available`, `degradation_events`, `citation_coverage`, `user_leak_count`, `unaudited_answer_count`, `recovery_drill_passed` and `backup_restore_passed`. Keep all temporary DB/index/Redis prefixes isolated and clean them in `finally`; do not weaken `verify_acceptance.py` or convert provider errors into fixture success.

Ensure at least one controlled degradation (for example a retriever lane outage or a circuit-open fake within the isolated run) produces both a structured warning log and an allowlisted durable event; the UI must display the same reason/outcome/retryable fields without exposing provider text. A normal Mem0 provider failure must produce `memory_provider_degraded` and `/v1/memories` 503, while a healthy run must prove a real memory read/write boundary.

- [ ] **Step 4: Run the real acceptance and all gates**

```bash
set -a; source .env.local; set +a
export HF_HOME=/tmp/agentic-rag-hf MEM0_TELEMETRY=0
export AGENTIC_RAG_RUN_REAL_QUERY_PROVIDER_E2E=1
conda run -n agentic-rag python scripts/run_real_query_acceptance.py \
  --output var/artifacts/evals/real-api-current
conda run -n agentic-rag python scripts/verify_acceptance.py \
  --report var/artifacts/evals/real-api-current/summary.json
conda run -n agentic-rag pytest --import-mode=importlib \
  tests/e2e/test_query_console_real_services.py -q -s
```

Expected: the real report has a positive query count, `client_provenance=real_query_api`, current snapshot ID, Mem0 available, citation coverage 1.0, zero leaks, zero unaudited answers, recovery and backup true, and `verify_acceptance.py` exits 0. Missing service variables skip only the opt-in test and never create a false PASS report.

- [ ] **Step 5: Commit**

```bash
git add tests/e2e/test_query_console_real_services.py tests/fixtures/query_services.py \
  scripts/run_real_query_acceptance.py scripts/verify_acceptance.py
git commit -m "test: accept real graph api console query"
```

### Task 5: 中文运行手册、路线图和发布状态收敛

**Files:**
- Modify: `docs/local-operations.md`
- Modify: `docs/development-progress.md`
- Modify: `docs/superpowers/plans/2026-08-04-agentic-rag-implementation-roadmap.md`
- Create: `tests/unit/test_documentation_contracts.py`

**Interfaces:**
- 文档中的命令、环境变量、API 路径、事件名、状态枚举和文件路径必须保持英文原样；说明、标题、故障解释、验收步骤和页面操作全部使用中文。
- 文档必须明确区分 Query Outbox（MySQL 同事务持久化待投递意图）与 Query Worker（Redis 领取、租约、心跳、恢复、重试/DLQ、Graph 执行和终态 ACK），并给出生产级必要性。
- 文档必须记录 `SubagentDispatcher` 已接入、Todo 初始创建/追加、全局 `research_attempt_count`、Mem0 默认启用、页面启动方式和降级/熔断日志字段。

- [ ] **Step 1: Write the documentation acceptance checks**

```python
def test_chinese_operations_docs_cover_new_runtime_contracts() -> None:
    operations = Path("docs/local-operations.md").read_text()
    roadmap = Path("docs/superpowers/plans/2026-08-04-agentic-rag-implementation-roadmap.md").read_text()
    assert "Query Outbox" in operations and "Query Worker" in operations
    assert "SubagentDispatcher" in operations and "research_attempt_count" in operations
    assert "Agentic RAG 控制台" in operations
    assert "真实 Graph/API" in roadmap and "Mem0" in roadmap
```

- [ ] **Step 2: Run the checks before editing**

```bash
conda run -n agentic-rag pytest --import-mode=importlib \
  tests/unit/test_documentation_contracts.py -q
```

Expected: FAIL because控制台启动、全局预算、子 Agent 已接入和真实 API/SSE 验收内容尚未在运行手册与总体路线图中完整出现。

- [ ] **Step 3: Rewrite the Chinese operational sections**

在运行手册中新增“控制台启动与查询”“降级/熔断/重试日志”“子 Agent/Todo/研究预算”“真实验收与门禁”四节，给出 `python scripts/run_api.py`、`python scripts/run_query_worker.py`、`curl http://127.0.0.1:8000/`、SSE 重连和 `verify_acceptance.py` 命令；说明 UI 只展示白名单字段，`memory_provider_degraded`、`MODEL_REPAIR_EXHAUSTED`、`CIRCUIT_OPEN`、`WORKER_DLQ` 等信号如何排查。

在总体路线图和开发进度中将已完成项目、当前 snapshot/provenance 证据、真实 MySQL/Redis/Elasticsearch/Mem0 测试和最终 PASS 条件写成中文；保留代码标识和命令可复制性，删除“实现分支尚未合并”之类与当前 `main` 不符的陈旧描述，并把剩余的生产鉴权/RBAC 和 reranker 分数标定单独列为上线审批事项。

- [ ] **Step 4: Run documentation and full static checks**

```bash
conda run -n agentic-rag pytest --import-mode=importlib \
  tests/unit/test_documentation_contracts.py -q
conda run -n agentic-rag ruff check src tests evals scripts
conda run -n agentic-rag mypy src evals scripts
git diff --check
```

Expected: 文档契约通过，代码和脚本静态检查无新增错误，所有命令仍能在 `agentic-rag` Conda 环境执行。

- [ ] **Step 5: Commit**

```bash
git add docs/local-operations.md docs/development-progress.md \
  docs/superpowers/plans/2026-08-04-agentic-rag-implementation-roadmap.md \
  tests/unit/test_documentation_contracts.py
git commit -m "docs: document console and query runtime contracts in Chinese"
```

## Final Verification Checklist

- [ ] `conda run -n agentic-rag pytest --import-mode=importlib -m "not integration and not e2e and not live_model" -q`
- [ ] `conda run -n agentic-rag pytest --import-mode=importlib -m integration -q`（真实服务变量缺失时只出现明确 skip）
- [ ] `conda run -n agentic-rag pytest --import-mode=importlib -m e2e -q`
- [ ] `conda run -n agentic-rag ruff check src tests evals scripts`
- [ ] `conda run -n agentic-rag mypy src evals scripts`
- [ ] `conda run -n agentic-rag python -m evals.validate_datasets evals/datasets`
- [ ] `scripts/run_recovery_drill.py` 与备份恢复真实隔离测试通过，且没有临时文件残留
- [ ] 真实 Graph/API 查询返回当前 snapshot 和 `client_provenance=real_query_api`
- [ ] 真实 Mem0 读写或明确 provider degraded 证据存在，不能把空列表当作成功
- [ ] `verify_acceptance.py` 以真实 Graph/API summary 返回 `ACCEPTANCE PASSED`
- [ ] 页面能显示 `queued/running/completed/refuse/audit_failed/research_round_limit` 以及所有降级/熔断/重试/DLQ 提示，未知事件统一脱敏为“进度更新”
