# Phase 5 Task 1 — Trace Recorder and Online Metrics Projection

## Scope delivered

- Added local OpenTelemetry-compatible `SpanRecord` and async `TraceRecorder`.
  Each top-level span receives a recorded implicit `run` root, and nested spans
  retain the same trace ID with an in-trace parent span ID.
- Added `AgentEventEmitter`, which adapts safe event metadata to the existing
  `EventRepository` and writes only sanitized optional attribute artifacts via
  the existing `ArtifactStore` contract.
- Added `MetricsProjector.project_window()`, which reads only scoped
  `EventRepository.list_after()` rows and safe artifacts for one immutable
  RuntimeConfigSnapshot ID.  It projects queue/run/node latency, concurrency
  saturation, retrieval rounds/candidates, outcome and degraded rates,
  citation coverage, model token/call/cost totals, outbox/lease/reconciler
  operational counts, and feedback counts.

## Privacy and operational boundaries

- Prompt, credential, hidden-reasoning, token/authorization, and raw Tool
  payload fields are dropped before either a span, structured log record, or
  Artifact is created.
- Observability is append/projection-only: no telemetry value is passed to the
  router, retrieval, audit, loop, or refusal logic.
- Emitter timestamps remain `None` unless explicitly supplied.  The existing
  SQL repository owns first-write timestamp generation, preserving its
  deterministic event-key replay contract.
- No persistence schema or existing repository/Artifact/RuntimeConfigSnapshot
  contract was changed.

## TDD evidence

1. RED: `.venv/bin/pytest tests/unit/observability/test_tracing.py -q` failed
   during collection with `ModuleNotFoundError: agentic_rag.observability`.
2. GREEN: added the local observability package, then fixed two focused test
   failures: legitimate `input_tokens` had been over-redacted and
   `SpanRecord.model_dump()` was not JSON serializable by default.
3. Regression RED/GREEN: added the event-key replay test, observed it fail
   because generated timestamps conflicted with idempotent repository replay,
   and changed the emitter to defer timestamps to the repository.  The same
   cycle made implicit roots recorded spans so parent IDs resolve within every
   trace.

## Verification

- `/Users/steven/miniconda3/envs/agentic-rag/bin/pytest tests/unit/observability/test_tracing.py -q` — 5 passed.
- `/Users/steven/miniconda3/envs/agentic-rag/bin/pytest -m integration tests/integration/observability/test_metrics_projection.py -q` — 2 passed.
- Adjacent EventRepository/ArtifactStore/query graph regression subset — 50 passed.
- Combined focused and adjacent suite — 57 passed.
- `/Users/steven/miniconda3/envs/agentic-rag/bin/ruff check src tests` — clean.
- `/Users/steven/miniconda3/envs/agentic-rag/bin/mypy src/agentic_rag/observability` — clean.
- `/Users/steven/miniconda3/envs/agentic-rag/bin/mypy src` — 3 pre-existing
  errors in `src/agentic_rag/persistence/migrations.py` from constructing
  `Settings()` without required current fields; no Task 1 file is implicated.
- `git diff --check` — clean.

## Commit

`053e8e476bd3e363d2e81242802a4d94e9f48b37` — `feat: add local traces and metrics`.

## Fix round 1

Reviewer reproduction exposed six gaps in the first implementation.  Each was
captured with a focused failing test before its production change:

1. `sanitize_attributes()` now denies raw Tool input/output/response,
   chain-of-thought, hidden reasoning, messages, prompt, and provider/raw
   payload fragments. `sanitize_summary()` is allowlist-only and returns the
   generic `telemetry event` for all unknown summaries.
2. Event artifacts use a canonical SHA-256 address over user, Run, event key,
   and sanitized attributes; raw event keys are restricted and never become a
   path segment. Existing matching content is reused without a write, while a
   conflicting payload receives a new address before the existing repository
   key-conflict contract rejects it. Regression coverage verifies unchanged
   first content, distinct conflicting/cross-user paths, and path-alias reject.
3. `QueryGraphDependencies` and `QueryWorker` accept an optional
   `TraceRecorder`. The actual queue, graph-node, memory, LLM, retrieval,
   rerank, and audit boundaries create spans only when the recorder snapshot
   matches the Run snapshot. No injected recorder preserves the original path;
   no span feeds any decision.
4. `MetricsWindow.reducer_state` serializes active Run/node start timestamps,
   outcomes, and rate flags. A subsequent `project_window()` accepts that state
   so start/end events separated by a cursor page compute the same latency.
5. Outcome, repair, degraded, and loop flags are per Run, so all projected
   rates are bounded by one despite duplicate durable deliveries.
6. Trace ContextVar state now owns recorder identity, Run ID, and snapshot
   context. Cross-recorder and cross-Run nesting is rejected rather than
   inheriting another trace.

Fix-round verification:

- Observability + query graph + query worker: 36 passed.
- Metrics projection integration: 4 passed.
- `ruff check src tests`: clean.
- `mypy src/agentic_rag/observability src/agentic_rag/query/graph.py src/agentic_rag/runtime/query_worker.py`: clean.
- `git diff --check`: clean.

Fix-round-2 commit: `27cfa8c01ef8ba4167389b47336a1d16760d7adf` — `fix: enforce safe observability events`.

Follow-up: the legacy `_event` repository fallback now independently validates
event identifiers and converts arbitrary summaries to the generic safe summary,
so observability remains fail-closed even without an injected emitter. The
dedicated fallback regression and the query/worker/observability suite pass.
Follow-up commit: `9b5ac4f444567e10471c77fc483dc21669b1da1f` — `fix: sanitize fallback graph events`.

Fix commit: `99b9210b326c88c02fdb89e3a5d315b1c8baead1` — `fix: harden local observability telemetry`.

## Fix round 2

- Replaced attribute denylisting with a strict finite telemetry allowlist.
  Only approved numeric counters, terminal outcomes, ratings, and enumerated
  degraded components cross the boundary; unknown keys and all Tool/raw input,
  result, analysis, thinking, and instruction fields are discarded. Numbers
  must also be finite. Emitter event types/node names are safe identifiers and
  raw non-allowlisted summaries are rejected before any write.
- `QueryGraphDependencies.event_emitter` and `QueryWorker.event_emitter` now
  drive the real safe projection source. Graph events emit retrieval,
  citation, repair/outcome attributes; workers emit queue wait. Emitter
  snapshot IDs must equal the Run snapshot or the event is skipped. No decision
  path reads these results. Model usage is not yet instrumented because the
  current ModelGateway has no safely injected TraceRecorder/Emitter boundary;
  this task therefore records no fabricated usage data.
- Research-loop delegation is now additionally wrapped in a real `tool` span.
- Metrics cursor reducer state accumulates citation numerator/denominator, and
  artifact payloads are accepted only when canonical sanitized content hashes
  to the durable Event URI. Tampered content is ignored.
- Outcome rates are set only from one terminal Event's `termination_reason`;
  transient/refusal-start events cannot become a final Run outcome.

Fix-round-2 verification:

- Query/runtime/observability unit suite: 127 passed.
- Metrics projection integration suite: 7 passed.
- `ruff check src tests`: clean.
- Relevant mypy: clean.
- `git diff --check`: clean.

## Fix round 3

- Added a task-local `event_emission_scope()` and `emit_model_usage()` boundary.
  Real `ModelGateway.complete()` and `complete_structured()` now emit only
  provider-derived finite input/output token counts, attempts, and latency when
  entered from a matching safe scope. Event keys are deterministic SHA-256
  values over server-owned Run/operation/sequence semantics. Pricing is not
  emitted: no authoritative pricing source is available in this deployment.
- The real graph span wrapper and worker queue-to-graph invocation enter that
  scope only when the injected emitter snapshot matches the Run snapshot.
  Nested operation names compose deterministically, preventing two distinct
  graph LLM operations in one Run from sharing an event key. A no-scope gateway
  call remains behaviorally unchanged.
- Graph projection events now use stable keys, research completion extracts
  attributes from `{**state, **update}`, and real graph-boundary events record
  node start/completion plus finite measured node latency. `ANSWER_FINALIZED`
  repair counts are added to the online projection.
- The deployment worker composition forwards the exact optional recorder and
  emitter from `QueryGraphDependencies`, preserving one shared snapshot
  baseline across worker and graph. Snapshot mismatch emits no queue or graph
  telemetry and never changes execution.

Fix-round-3 TDD and verification:

1. RED: the initial ModelGateway usage test failed on importing the absent
   `event_emission_scope`; after implementing the scope and gateway boundary,
   a deterministic provider-shaped graph regression verified that a real graph
   invocation emits `LLM_COMPLETED` through the nested graph scope.
2. GREEN focused observability/query/runtime suite: 65 passed.
3. Relevant unit/integration suite (`observability`, `query`, `runtime`): 139
   passed.
4. Full suite with `--import-mode=importlib`: 466 passed, 38 skipped. Plain
   collection is affected by pre-existing duplicate test-module basenames and
   stale `__pycache__`; importlib mode isolates module names without modifying
   workspace artifacts.
5. `ruff check` for changed source/tests — clean; `mypy` for observability,
   graph, model gateway, and query worker — clean; `git diff --check` — clean.

## Fix round 4

- The real Query Worker now emits a stable `RUN_STARTED` event immediately
  after a successful lease claim and one terminal `RUN_COMPLETED`,
  `RUN_FAILED`, or `RUN_CANCELLED` event only after the fenced Run finish
  succeeds. Lifecycle keys are Run-level (`run_id + event_type`) so reclaim
  and claim-generation changes cannot inflate totals; a lost lease leaves the
  delivery pending and publishes no terminal outcome. Terminal events carry a
  finite server-measured `run_latency_seconds` attribute, and the projector
  uses that authoritative value once instead of adding a second timestamp
  delta.
- `MetricsWindow.estimated_cost_status` explicitly reports `unavailable` when
  no priced cost observation exists. The current immutable
  `RuntimeConfigSnapshot` has no provider pricing table, so the real
  `ModelGateway` emits only provider usage tokens/attempts/latency and never a
  fabricated zero or estimated price. A supplied, sanitized cost field is
  retained for externally priced observations and reports `observed`; the
  status is carried through cursor reducer state.

Fix-round-4 RED/GREEN coverage includes stable lifecycle/latency events for
success, timeout, cancellation and lease loss, replay fencing, snapshot
mismatch, explicit cost-unavailable semantics, and non-duplicated run/node
latency projection.
