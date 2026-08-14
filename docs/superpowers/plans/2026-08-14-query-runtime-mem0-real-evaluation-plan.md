# Query Runtime, Mem0 and Real Evaluation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (\`- [ ]\`) syntax for tracking.

**Goal:** Make the documented Query API/Worker path runnable end-to-end, connect the installed Mem0 provider safely, and make final evaluation PASS require a real Graph/API client while emitting explicit degradation telemetry.

**Architecture:** Keep the three-process local topology. API creates \`agent_runs\` and a \`query_run\` Outbox row atomically; Query Worker owns a filtered Outbox dispatcher plus the Query Stream consumer; Ingestion Worker continues to dispatch only ingestion rows. A production composition root creates \`QueryGraphDependencies\` and \`AsyncMemory\` from validated settings. Evaluation has explicit \`fixture\`, \`graph\`, and \`api\` modes, with fixture mode prohibited from final acceptance.

**Tech Stack:** Python 3.11, FastAPI, LangGraph, SQLAlchemy async MySQL, Redis Streams, Elasticsearch 8, mem0ai 2.0.12, OpenAI-compatible Async clients, pytest, Ruff, mypy, Alembic.

## Global Constraints

- Execute all commands in \`conda run -n agentic-rag\`.
- Preserve server-owned \`UserScope\`, \`RuntimeConfigSnapshot\`, checkpoint namespace, and fail-closed evidence/audit contracts.
- Never write prompts, hidden reasoning, raw tool payloads, memory text, credentials, or provider responses to durable telemetry.
- Every fallback, retry, circuit-open, outage, refusal, and DLQ path emits a bounded structured event/log with component, reason, run ID when available, snapshot ID, attempt, retryable, and degraded/refused outcome.
- Query and ingestion Outbox rows are filtered by \`aggregate_type\`; each worker dispatches only its own rows.
- Redis publication occurs before \`mark_dispatched\`; failed publication remains retryable.
- Real-service tests use explicit disposable local MySQL/Redis/Elasticsearch resources and never infer or mutate the configured production namespace.
- Mem0 calls always use the server-owned user ID and \`infer=False\); durable memory is created only from the existing structured light-model extractor.
- Fixture evaluation is a smoke mode only. Final acceptance requires \`evaluation_mode\` of \`graph\` or \`api\` and real-client provenance.

---

### Task 1: Filtered Query Outbox Dispatcher

**Files:**
- Modify: \`src/agentic_rag/persistence/repositories.py:607-614,1302-1367\`
- Modify: \`src/agentic_rag/persistence/outbox.py:11-37\`
- Modify: \`scripts/run_ingestion_worker.py:89-108,124-133\`
- Modify: \`src/agentic_rag/runtime/query_worker.py:77-121,run_forever\`
- Create: \`tests/unit/persistence/test_query_outbox_dispatch.py\`
- Modify: \`tests/integration/runtime/test_query_worker.py\`

**Interfaces:**
- \`OutboxRepository.list_pending(limit: int, aggregate_type: str | None = None) -> list[OutboxRecord]\`.
- \`OutboxRepository.claim_pending(limit: int, aggregate_type: str | None = None) -> list[OutboxRecord]\`.
- \`OutboxDispatcher(outbox, broker, aggregate_type: Literal["query_run", "ingestion_job"])\`.
- \`QueryWorker(runs, broker, graph_factory, worker_id, outbox_dispatcher: OutboxDispatcher, outbox_interval_seconds: float = 1.0)\`.

- [ ] **Step 1: Write the failing tests.**

~~~python
async def test_query_claim_excludes_ingestion_outbox_rows(fake_outbox):
    rows = await fake_outbox.claim_pending(10, aggregate_type="query_run")
    assert {row.aggregate_type for row in rows} == {"query_run"}

async def test_query_dispatch_marks_only_published_query_row(fake_outbox, broker):
    dispatcher = OutboxDispatcher(fake_outbox, broker, aggregate_type="query_run")
    assert await dispatcher.dispatch_once() == 1
    assert fake_outbox.dispatched == ["query-outbox-1"]
    assert broker.published == [("agenticrag:jobs:query", "run-1")]
~~~

- [ ] **Step 2: Run the focused tests and verify the expected missing-argument failure.**

Run: \`conda run -n agentic-rag pytest --import-mode=importlib tests/unit/persistence/test_query_outbox_dispatch.py tests/integration/runtime/test_query_worker.py -q\`.

Expected: FAIL because the repository and dispatcher do not accept an aggregate type and Query Worker has no dispatcher loop.

- [ ] **Step 3: Add aggregate filtering to the repository protocol and SQL adapter.**

Add the optional predicate \`task_outbox.c.aggregate_type == aggregate_type\` to both list and claim queries. Pass \`aggregate_type="ingestion_job"\` from the existing Ingestion Worker dispatcher and \`aggregate_type="query_run"\` from the Query Worker dispatcher.

- [ ] **Step 4: Add the Query Worker dispatcher loop.**

Start the dispatcher as a sibling task to the consumer in \`run_forever\`; cancel it during shutdown. Redis publish failure calls \`schedule_retry\`, emits \`OUTBOX_RETRY_SCHEDULED\`, and does not stop Query consumption. Successful publication emits \`OUTBOX_DISPATCHED\` only after \`mark_dispatched\`.

- [ ] **Step 5: Run focused tests and the worker regression suite.**

Run the focused files from Step 2. Expected: mixed query/ingestion rows, Redis failure retry, shutdown, duplicate delivery and reclaim tests all pass.

- [ ] **Step 6: Run static checks and commit.**

Run \`conda run -n agentic-rag ruff check src tests\` and \`conda run -n agentic-rag mypy src/agentic_rag/persistence src/agentic_rag/runtime\`.

~~~bash
git add src/agentic_rag/persistence/repositories.py src/agentic_rag/persistence/outbox.py src/agentic_rag/runtime/query_worker.py scripts/run_ingestion_worker.py tests/unit/persistence/test_query_outbox_dispatch.py tests/integration/runtime/test_query_worker.py
git commit -m "feat: dispatch query outbox records from query worker"
~~~

### Task 2: Production Query Dependency Composition and Runnable Worker

**Files:**
- Create: \`src/agentic_rag/runtime/query_composition.py\`
- Modify: \`src/agentic_rag/bootstrap.py:40-163\`
- Modify: \`src/agentic_rag/config.py:26-62\`
- Modify: \`scripts/run_query_worker.py:24-70\`
- Create: \`tests/unit/runtime/test_query_composition.py\`
- Create: \`tests/integration/runtime/test_query_worker_startup.py\`

**Interfaces:**
- \`async def build_query_dependencies(container: AppContainer, settings: Settings) -> QueryGraphDependencies\`.
- \`async def close_query_dependencies(dependencies: QueryGraphDependencies) -> None\`.
- \`async def run(settings: Settings, dependencies_factory: DependenciesFactory = build_query_dependencies) -> None\`.

- [ ] **Step 1: Write RED tests for composition and startup.**

~~~python
async def test_build_query_dependencies_binds_same_snapshot_to_gateway_and_emitter(settings, container):
    dependencies = await build_query_dependencies(container, settings)
    assert dependencies.gateway.snapshot.snapshot_id == dependencies.event_emitter.runtime_config_snapshot_id

def test_query_worker_main_no_longer_exits_with_deployment_owned_error(settings, fake_factory):
    assert query_worker_entrypoint(settings, fake_factory) == 0
~~~

The integration test must assert that a missing required model/retrieval dependency fails with a safe \`QueryCompositionError\`, not an empty graph.

- [ ] **Step 2: Run RED.**

Run: \`conda run -n agentic-rag pytest --import-mode=importlib tests/unit/runtime/test_query_composition.py tests/integration/runtime/test_query_worker_startup.py -q\`.

Expected: FAIL because the composition root does not exist and CLI \`main()\` intentionally raises.

- [ ] **Step 3: Add validated settings for Query composition.**

Add process-owned values needed by construction: \`query_evaluation_mode\`, \`mem0_enabled\`, \`mem0_collection\`, \`mem0_embedding_base_url\`, \`mem0_embedding_api_key\`, \`mem0_embedding_model\`, \`mem0_llm_model\`, and explicit local provider flags. Keep secrets as \`SecretStr\`; reject blank/placeholder credentials in readiness checks.

- [ ] **Step 4: Implement the composition root.**

Construct OpenAI-compatible Async clients, \`ModelGateway\`, Qwen query embedder, ES vector/BM25 adapters, \`Reranker\`, \`RetrievalService\`, ParentFetcher, EvidenceBuilder/grader, ResearchAgentLoop, AnswerGenerator, FaithfulnessAuditor, CitationValidator, resolver, event emitter and trace recorder from one immutable \`RuntimeConfigSnapshot\`. Owned clients are closed by \`close_query_dependencies\`.

- [ ] **Step 5: Wire the entry point and worker lifecycle.**

~~~python
async def run(settings: Settings, dependencies_factory=build_query_dependencies) -> None:
    container = build_container(settings)
    dependencies = await dependencies_factory(container, settings)
    try:
        async with container.checkpoints.open_query() as checkpointer:
            await QueryWorker(runs=runs, broker=container.broker, graph_factory=graph_factory, worker_id="query-worker", outbox_dispatcher=query_dispatcher).run_forever(stop_event=stop_event)
    finally:
        await close_query_dependencies(dependencies)
        await container.close()
~~~

The documented command remains \`conda run -n agentic-rag python scripts/run_query_worker.py\`; missing configuration gives a precise message.

- [ ] **Step 6: Verify startup, static checks and commit.**

Run the two focused files, \`ruff check src scripts tests\`, and \`mypy src\`. Commit \`feat: add production query dependency composition\`.

### Task 3: Real Query End-to-End Harness

**Files:**
- Create: \`tests/e2e/test_query_pipeline_real_services.py\`
- Create: \`tests/integration/runtime/test_query_pipeline.py\`
- Create or modify: \`tests/fixtures/query_services.py\`
- Modify: \`docs/local-operations.md\`

**Interfaces:**
- \`RealQueryFixture\` provisions isolated MySQL, Redis, ES generation, checkpoint and artifact paths.
- \`DeterministicQueryProvider\` implements existing ModelGateway and embedding ports without bypassing QueryGraph.
- \`run_real_query_pipeline(fixture) -> QueryRun\` exercises public API/worker boundaries.

- [ ] **Step 1: Write RED tests for the request path.**

~~~python
@pytest.mark.e2e
@pytest.mark.integration
async def test_post_query_run_reaches_persisted_audited_answer(real_query_fixture):
    response = await real_query_fixture.api.post("/v1/query-runs", json={"query": "what is in the seeded document?", "thread_id": "e2e"})
    assert response.status_code == 202
    run = await real_query_fixture.wait_for_terminal(response.json()["run_id"])
    assert run.status == "completed"
    assert run.answer["audited"] is True
    assert await real_query_fixture.sse_contains("ANSWER_FINALIZED")
~~~

- [ ] **Step 2: Run RED and confirm the missing dispatcher/composition behavior.**

Run with explicit service variables: \`conda run -n agentic-rag pytest --import-mode=importlib tests/integration/runtime/test_query_pipeline.py tests/e2e/test_query_pipeline_real_services.py -q\`.

- [ ] **Step 3: Build isolated fixtures and the deterministic provider.**

Use a generated MySQL database/schema, Redis stream prefix, ES \`agenticrag-children-e2e-*\` generation, temporary checkpoints, and artifacts. Seed one document through the real ingestion pipeline and wait for its active version.

- [ ] **Step 4: Execute the real API → Outbox → Worker → Graph path.**

Start API and Query Worker through their composition roots, poll the persisted Run with a bounded timeout, assert no cross-user events, and test duplicate Redis delivery, cancellation and one worker restart.

- [ ] **Step 5: Run deterministic and provider E2E gates.**

Run the focused integration test, then \`conda run -n agentic-rag pytest -m e2e tests/e2e/test_query_pipeline_real_services.py -q -s\` with explicit disposable services. Missing variables skip; configured but unhealthy services fail.

- [ ] **Step 6: Commit the E2E harness and operations documentation.**

Commit \`test: cover real query worker pipeline\`.

### Task 4: Production Mem0 AsyncMemory Integration

**Files:**
- Modify: \`src/agentic_rag/config.py:26-62\`
- Modify: \`src/agentic_rag/memory/mem0_adapter.py:17-67\`
- Create: \`src/agentic_rag/memory/factory.py\`
- Modify: \`src/agentic_rag/bootstrap.py:106-163\`
- Modify: \`src/agentic_rag/api/health.py\`
- Create: \`tests/unit/memory/test_mem0_factory.py\`
- Modify: \`tests/integration/memory/test_mem0_adapter.py\`
- Create: \`tests/e2e/test_mem0_real_services.py\`

**Interfaces:**
- \`def build_mem0_config(settings: Settings) -> dict[str, object]\`.
- \`async def build_memory_service(container: AppContainer, settings: Settings, snapshot: RuntimeConfigSnapshot) -> MemoryService\`.
- \`Mem0Adapter.add(messages, *, user_id, metadata, infer: bool = False)\` forwards to \`AsyncMemory.add\` with the server-owned user ID.

- [ ] **Step 1: Write RED tests for factory configuration and Bootstrap selection.**

~~~python
def test_mem0_config_contains_elasticsearch_collection_and_1024_dimensions(settings):
    config = build_mem0_config(settings)
    assert config["vector_store"]["config"]["collection_name"] == "agent_memories_v1"
    assert config["vector_store"]["config"]["embedding_model_dims"] == 1024

async def test_enabled_mem0_builds_async_memory_and_disabled_mode_is_degraded(container, settings, snapshot):
    service = await build_memory_service(container, settings, snapshot)
    assert service is not unavailable_service
~~~

- [ ] **Step 2: Run RED.**

Run: \`conda run -n agentic-rag pytest --import-mode=importlib tests/unit/memory/test_mem0_factory.py tests/integration/memory/test_mem0_adapter.py -q\`.

Expected: FAIL because Bootstrap always returns \`_UnavailableMemory\` and the integration test is a placeholder.

- [ ] **Step 3: Implement \`AsyncMemory.from_config\` construction.**

Build a complete Mem0 config with Elasticsearch URL, collection, Qwen endpoint/model/key, and Mem0 LLM only when provider inference is explicitly enabled. The application extractor remains authoritative and calls Mem0 with \`infer=False\`.

- [ ] **Step 4: Preserve tenant and tombstone boundaries.**

Wrap AsyncMemory in \`Mem0Adapter\`; pass \`user_id\` into \`search/get_all/add\`, retain strict response filtering, and keep delete verification/search and MySQL tombstone reconciliation. Provider exceptions emit \`MEMORY_PROVIDER_DEGRADED\`.

- [ ] **Step 5: Replace the placeholder integration test.**

With \`AGENTIC_RAG_TEST_MEM0_ENABLED=1\` and explicit local ES/Mem0 variables, add a namespaced memory, search it, verify another user cannot see it, delete it, verify search absence, and run reconciliation. Missing configuration skips; configured failures fail.

- [ ] **Step 6: Run Mem0 gates and commit.**

Run \`pytest tests/unit/memory tests/integration/memory -q\`, the opt-in real Mem0/ES test, Ruff and mypy. Commit \`feat: connect production mem0 memory service\`.

### Task 5: Real Evaluation Modes and Acceptance Provenance

**Files:**
- Modify: \`evals/run.py:132-379\`
- Modify: \`evals/report.py:20-63\`
- Modify: \`scripts/verify_acceptance.py:12-57\`
- Create: \`evals/clients.py\`
- Modify: \`tests/unit/evals/test_runner.py\`
- Create: \`tests/integration/evals/test_real_query_evaluation.py\`
- Create: \`tests/e2e/test_query_evaluation_api.py\`
- Modify: \`docs/local-operations.md\`

**Interfaces:**
- \`EvaluationMode = Literal["fixture", "graph", "api"]\`.
- \`GraphQueryClient.query(case: EvaluationCase) -> Mapping[str, object]\`.
- \`HttpQueryClient.query(case: EvaluationCase) -> Mapping[str, object]\`.
- \`EvalRunner(client, output_dir, evaluation_mode: EvaluationMode, client_provenance: str)\`.
- \`verify_acceptance\` requires \`evaluation_mode in {"graph", "api"}\` and non-fixture provenance.

- [ ] **Step 1: Write RED tests for provenance and fixture rejection.**

~~~python
def test_fixture_summary_cannot_pass_final_acceptance(valid_fixture_summary):
    assert verify_acceptance(valid_fixture_summary) == 1

async def test_graph_mode_calls_real_query_client_and_persists_provenance(runner, cases):
    summary = await runner.run(cases)
    assert summary["evaluation_mode"] == "graph"
    assert summary["client_provenance"] == "real_query_graph"
~~~

- [ ] **Step 2: Run RED.**

Run: \`conda run -n agentic-rag pytest --import-mode=importlib tests/unit/evals/test_runner.py tests/integration/evals/test_real_query_evaluation.py -q\`.

Expected: fixture summaries lack provenance and \`verify_acceptance\` only checks numeric gates.

- [ ] **Step 3: Add Graph and HTTP clients.**

\`GraphQueryClient\` invokes production \`build_query_graph\` with fixed snapshot and isolated dependencies. \`HttpQueryClient\` submits \`/v1/query-runs\`, polls GET/SSE until terminal, and returns only the strict answer/evidence/events projection accepted by \`EvalCaseResult\`. Neither client synthesizes reference answers.

- [ ] **Step 4: Extend strict result and summary schemas.**

Persist \`evaluation_mode\`, \`client_provenance\`, \`real_query_count\`, and \`ragas_status\`. Reject missing/fixture provenance when final acceptance is requested. Keep deterministic metrics delegated to \`evals.metrics\` and Ragas normalized through \`RagasAdapter\`.

- [ ] **Step 5: Add real evaluation CLI modes.**

~~~text
python -m evals.run --mode fixture --dataset evals/datasets/baseline.jsonl --output var/artifacts/evals/fixture
python -m evals.run --mode graph --dataset evals/datasets/baseline.jsonl --output var/artifacts/evals/graph
python -m evals.run --mode api --base-url http://127.0.0.1:8000 --dataset evals/datasets/baseline.jsonl --output var/artifacts/evals/api
~~~

The default may remain \`fixture\` for an explicit smoke command, but it must print \`SMOKE ONLY\` and cannot satisfy \`verify_acceptance\`.

- [ ] **Step 6: Run evaluation and static gates; commit.**

Run fixture unit tests, isolated Graph evaluation, API evaluation with real local services, Ruff, mypy and \`verify_acceptance\`. Commit \`feat: require real query provenance for acceptance\`.

### Task 6: Explicit Degradation and Circuit Telemetry

**Files:**
- Modify: \`src/agentic_rag/observability/logging.py\`
- Modify: \`src/agentic_rag/observability/metrics.py\`
- Modify: \`src/agentic_rag/retrieval/graph.py\`
- Modify: \`src/agentic_rag/query/fast_rag.py\`
- Modify: \`src/agentic_rag/query/router.py\`
- Modify: \`src/agentic_rag/query/audit.py\`
- Modify: \`src/agentic_rag/runtime/query_worker.py\`
- Modify: \`src/agentic_rag/memory/service.py\`
- Create: \`tests/unit/observability/test_degradation_events.py\`
- Modify: \`tests/integration/observability/test_metrics_projection.py\`

**Interfaces:**
- \`emit_degradation(*, component: str, reason: str, run_id: str | None, snapshot_id: str, attempt: int, retryable: bool, outcome: Literal["degraded", "refused", "dlq"]) -> Awaitable[None]\`.
- \`CircuitState\` remains process-owned and never enters QueryState/checkpoint state.

- [ ] **Step 1: Write RED tests for user-visible and durable signals.**

~~~python
async def test_retrieval_single_lane_degradation_emits_safe_event(retrieval_service, request, emitter):
    result = await retrieval_service.retrieve(request)
    assert result.degraded is True
    assert emitter.events[-1].event_type == "RETRIEVAL_DEGRADED"
    assert "prompt" not in json.dumps(emitter.events[-1].model_dump()).lower()

def test_circuit_open_projects_retryable_degraded_metric(events):
    assert project(events).degraded_count == 1
~~~

- [ ] **Step 2: Run RED and identify silent fallback boundaries.**

Run: \`conda run -n agentic-rag pytest --import-mode=importlib tests/unit/observability/test_degradation_events.py tests/unit/retrieval tests/unit/query -q\`.

- [ ] **Step 3: Add one bounded emission helper.**

Validate component/reason against safe enums, derive a stable event key from run/operation/attempt, attach snapshot and safe counts, and use existing \`AgentEventEmitter\`/logger. Telemetry failure must not change the business result but is counted locally.

- [ ] **Step 4: Instrument fallback boundaries.**

Add events/logs for retrieval lane failure/degradation, reranker unavailable, memory outage, model retry/repair exhaustion, circuit-open, audit refusal, Query Outbox retry, lease loss, cancellation and DLQ. Preserve existing safe API error codes and include the exact \`retryable\` boolean and \`degraded_components\` tuple on each affected response.

- [ ] **Step 5: Verify redaction and metric projection.**

Run focused observability/query/retrieval tests and assert payloads exclude prompts, reasoning, provider responses, memory text and secrets. Verify cursor projection does not double-count replayed degradation events.

- [ ] **Step 6: Commit telemetry hardening.**

Commit \`feat: expose explicit degradation and circuit telemetry\`.

### Task 7: Final Real-Service Gate, Documentation and Release Review

**Files:**
- Modify: \`docs/development-progress.md\`
- Modify: \`docs/local-operations.md\`
- Modify: \`docs/superpowers/plans/2026-08-04-agentic-rag-implementation-roadmap.md\`
- Modify: \`.superpowers/sdd/2026-08-04-agentic-rag-phase-5-evaluation-operations/progress.md\`
- Create: \`tests/e2e/test_release_query_gate.py\`

- [ ] **Step 1: Write the release gate test.**

Require Query Worker startup, Query E2E completion, Mem0 provider status, \`evaluation_mode != fixture\`, hard acceptance gates, and explicit degradation telemetry checks.

- [ ] **Step 2: Run the release gate against isolated local services.**

Run migrations, static checks, unit tests, integration tests, deterministic E2E, real Query E2E, Mem0 E2E, graph/API evaluation, backup/recovery drills and acceptance verification in Conda environment \`agentic-rag\`.

- [ ] **Step 3: Update status documents with evidence.**

Separate module completion from production-readiness gates. Correct the roadmap header and Task 5 ledger SHA to the final commit.

- [ ] **Step 4: Perform final verification and release review.**

Run \`ruff check src tests evals scripts\`, \`mypy src\`, \`pytest --import-mode=importlib -q\`, real-service markers, \`git diff --check\`, wheel build/install smoke and \`alembic upgrade head\` on a disposable database. Request independent review before merging to \`main\`.

- [ ] **Step 5: Commit documentation and release gate.**

Commit \`docs: record real query and mem0 production gates\`.
