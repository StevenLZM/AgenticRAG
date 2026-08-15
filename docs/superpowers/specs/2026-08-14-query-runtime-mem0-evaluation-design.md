# Query Runtime, Mem0 and Real Evaluation Design

**Date:** 2026-08-14  
**Status:** Approved design

## Goal

Close the production Query path, connect the installed Mem0 provider, and make
the acceptance report distinguish deterministic fixture smoke from a real
QueryGraph/API evaluation.

## Scope

This change covers three connected deliverables:

1. Query Outbox dispatch and a runnable Query Worker composition root.
2. A real `mem0.AsyncMemory` adapter with the existing tenant-safe
   `MemoryServiceImpl` boundary.
3. Real Graph/API evaluation modes and explicit degradation/circuit telemetry.

The existing three-process local topology remains:

```text
API ── MySQL Run + Query Outbox ── Query Worker
                                      ├─ Outbox dispatcher → Redis query stream
                                      └─ Query consumer → QueryGraph → Run answer
Ingestion Worker ── ingestion Outbox dispatcher → Redis ingestion stream
```

## Query Outbox and Worker

`RunManager.create()` continues to insert `agent_runs` and a `query_run` row in
`task_outbox` in one MySQL transaction. The Query Worker owns a bounded
dispatcher loop that claims only `query_run` rows, publishes their existing
`stream_name`, then marks the row dispatched. Publication uses the existing
stable outbox/attempt deduplication key. A Redis failure leaves the row pending
and emits a structured retry event.

The worker composition root constructs all `QueryGraphDependencies` from
process-owned settings and clients: model gateway, Qwen query embeddings,
retrieval/rerank, parent/evidence services, research/subagents, generation and
audits, memory, checkpoint, and telemetry. Missing required production
dependencies fail startup with a safe reason and a structured log; they do not
silently produce an empty answer.

## Mem0

The composition root constructs `mem0.AsyncMemory.from_config()` when
`AGENTIC_RAG_MEM0_ENABLED=1`. The fixed application contract is collection
`agent_memories_v1`, Elasticsearch as the vector store, and Qwen-compatible
1024-dimensional embeddings. The adapter passes the server-owned `user_id` to
every provider operation and calls Mem0 with `infer=False`; only the existing
light-model structured extractor may turn finalized user facts into durable
memory.

The provider remains behind the existing narrow `Mem0Adapter` and
`MemoryServiceImpl`. MySQL tombstones remain authoritative for deletion and
reconciliation. Provider outage is an explicit degraded read/write outcome;
cross-user or malformed provider data is a security refusal. Both outcomes
emit structured logs/events without raw memory text.

## Real Query E2E

The deterministic E2E uses isolated MySQL, Redis, Elasticsearch index
generation, SQLite checkpoints and artifact directory. It injects a
deterministic model/embedding client but uses the production RunManager,
OutboxDispatcher, Query Worker, QueryGraph and persistence adapters. It verifies
POST → 202 → Outbox → Redis → Worker → retrieval/evidence/audit → persisted
answer → GET/SSE, plus duplicate delivery, cancellation and worker restart.

An opt-in provider smoke can replace only the model/embedding clients with
DeepSeek/Qwen. Missing provider credentials produce an explicit skip, never a
fabricated acceptance result.

## Evaluation modes

`evals.run` exposes three modes:

- `fixture`: offline framework smoke only; it cannot satisfy final acceptance.
- `graph`: real in-process QueryGraph with isolated local services.
- `api`: real HTTP API plus Query Worker and isolated local services.

Results persist `evaluation_mode` and `client_provenance`. The acceptance
verifier rejects fixture mode, mixed snapshots, missing provenance, leakage,
unaudited answers, or incomplete recovery/backup gates. Deterministic retrieval
metrics continue to use the Task 2 helpers. Ragas remains optional but strict:
unconfigured means `status=unavailable`, while a configured offline backend
must return validated Faithfulness, Answer Relevancy and Context Precision
values.

## Degradation, retry and circuit observability

Every operational fallback, retry exhaustion, circuit-open event, provider
outage, single-lane retrieval degradation, audit refusal, and worker DLQ path
must emit a structured event/log with:

- `event_type` and bounded `component`/`reason` enums;
- `run_id` when available;
- server-owned user-scope hash, never raw user data;
- `runtime_config_snapshot_id`;
- `retryable`, `attempt`, and `degraded`/`refused` outcome;
- safe latency/count attributes only.

Telemetry must never include prompts, hidden reasoning, raw tool payloads,
provider responses, memory text, authorization headers, or secrets. The same
event key must be stable across replay. User-visible API responses retain a
safe error/degraded code and retry guidance; logs/events provide the operator
diagnostic reason.

## Failure boundaries

- MySQL transaction failure: no Redis publish is attempted.
- Redis publish failure: Outbox remains retryable and the Worker continues
  consuming existing deliveries.
- Query lease loss: no terminal event or ACK is emitted.
- Model/Reranker/Mem0 outage: use the documented fail-closed/degraded path and
  emit one bounded event per stable operation key.
- Unknown termination reason, malformed provider data, snapshot mismatch, or
  unauthorized evidence: refuse or DLQ according to the existing contracts.

## Verification gates

The implementation is complete only when:

1. Query Worker starts from the documented command with real dependency
   injection.
2. A real isolated Query E2E reaches a persisted answer and SSE event.
3. Query Outbox retry/replay and worker reclaim are covered.
4. Mem0 add/search/delete/reconcile and outage/tenant isolation are covered.
5. Graph/API evaluation is distinguishable from fixture smoke and final
   acceptance rejects fixture-only reports.
6. Degradation/circuit paths have both safe user-facing signals and structured
   durable telemetry.
7. Ruff, mypy, unit/integration/E2E tests, migrations and wheel/install smoke
   pass in `conda` environment `agentic-rag`.
