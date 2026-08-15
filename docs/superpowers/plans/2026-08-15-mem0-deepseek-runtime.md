# Mem0 Default and DeepSeek Structured Runtime Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Enable Mem0 by default with explicit degraded/readiness semantics and make DeepSeek structured Graph calls protocol-aware, diagnosable, and deterministic under provider outages or schema-invalid responses.

**Architecture:** Mem0 remains behind the existing `MemoryService` boundary. Settings provide safe local embedding/auth fallbacks, while provider construction failures return the existing unavailable service, emit a structured degradation signal, and keep query execution available but readiness unhealthy. `ModelGateway` remains the sole retry/repair owner; it gains explicit protocol selection, provider-compatible JSON request hints, safe fenced-JSON normalization, and non-sensitive diagnostics.

**Tech Stack:** Python 3.11, Pydantic v2, pydantic-settings, OpenAI-compatible async clients, Mem0 AsyncMemory, pytest/pytest-asyncio, Ruff, mypy, FastAPI readiness checks.

## Global Constraints

- Mem0 is enabled by default; a provider/configuration failure must not silently make readiness healthy.
- A Mem0 failure degrades memory only; retrieval, citation validation, and answer audit remain fail-closed.
- `mem0_llm_enabled` remains false by default; application `ModelGateway` performs the lightweight memory extraction.
- Model retries remain owned by `ModelGateway` and remain bounded at the existing maximum of two retries plus one schema repair.
- Raw prompts, hidden reasoning, tool payloads, and raw model output must never enter durable logs, events, or repair telemetry.
- Structured output must still pass the existing Pydantic schema; normalization may remove transport formatting only and must not relax the schema.
- Explicit Mem0 settings override fallbacks. Remote Elasticsearch requires explicit authentication; unauthenticated access is allowed only for loopback URLs.
- Real-service tests are opt-in and must use isolated collection/index/database/key namespaces.

---

### Task 1: Default Mem0 composition and readiness semantics

**Files:**
- Modify: `src/agentic_rag/config.py:Settings.mem0_enabled` and Mem0-related validators/defaults.
- Modify: `src/agentic_rag/memory/factory.py:build_mem0_config`, `_build_memory_service_sync`, and degradation logging.
- Test: `tests/unit/memory/test_mem0_factory.py`.
- Test: `tests/unit/test_config.py`.
- Modify: `docs/local-operations.md` and `docs/development-progress.md`.

**Interfaces:**
- `build_mem0_config(settings: object) -> dict[str, object]` keeps its current return shape.
- Mem0 embedding URL/key resolution is `mem0_embedding_*` first, then `qwen_embedding_*`.
- `UnavailableMemoryService(reason: str)` remains the degraded service boundary and exposes `available=False`, `degraded=True`.

- [ ] **Step 1: Write the failing tests**

Add tests asserting the default `Settings` value is `mem0_enabled is True`, `build_mem0_config` falls back to `qwen_embedding_base_url` and `qwen_api_key`, loopback Elasticsearch permits no auth, remote Elasticsearch without auth raises `MemoryCompositionError`, and an enabled provider construction failure returns a degraded service while logging `memory_provider_degraded`.

- [ ] **Step 2: Run the focused tests to verify RED**

Run:

```bash
conda run -n agentic-rag pytest --import-mode=importlib tests/unit/memory/test_mem0_factory.py tests/unit/test_config.py -q
```

Expected: failures for the false default, missing fallback resolution, and current unconditional Elasticsearch authentication requirement.

- [ ] **Step 3: Implement the minimal composition change**

Set `mem0_enabled: bool = True`. Resolve embedding values with explicit Mem0 values first and Qwen values second. Treat `localhost`, `127.0.0.1`, and `::1` as the only unauthenticated Elasticsearch hosts; require API key or username/password for all other hosts. Preserve the existing catch-to-`UnavailableMemoryService` path and emit a log with `component="mem0"`, `reason`, `outcome="degraded"`, and a stable `degraded=true` field. Do not enable Mem0’s own LLM inference.

- [ ] **Step 4: Run focused and readiness regression tests**

Run:

```bash
conda run -n agentic-rag pytest --import-mode=importlib tests/unit/memory/test_mem0_factory.py tests/unit/test_config.py tests/unit/runtime/test_query_composition.py -q
```

Expected: all tests pass, and disabled-mode compatibility tests still pass when a test explicitly supplies `mem0_enabled=False`.

- [ ] **Step 5: Update Chinese operations documentation**

Document the default-enabled behavior, fallback variables, loopback/remote ES auth rule, degraded query behavior, readiness status, and the exact `memory_provider_degraded` log fields in `docs/local-operations.md`; update the progress entry with the test command and result.

- [ ] **Step 6: Commit**

```bash
git add src/agentic_rag/config.py src/agentic_rag/memory/factory.py tests/unit/memory/test_mem0_factory.py tests/unit/test_config.py tests/unit/runtime/test_query_composition.py docs/local-operations.md docs/development-progress.md
git commit -m "feat: enable mem0 memory by default"
```

### Task 2: Explicit DeepSeek protocol and structured-response diagnostics

**Files:**
- Modify: `src/agentic_rag/config.py` to add a validated `deepseek_protocol` setting with `auto`, `chat`, and `responses` values.
- Modify: `src/agentic_rag/runtime/model_gateway.py` (`ModelCall`, `_create`, `complete_structured`, `_validate_schema`, and safe diagnostic helpers).
- Modify: `src/agentic_rag/observability/logging.py` only if an existing structured event helper needs a new non-sensitive field.
- Test: `tests/unit/runtime/test_model_gateway.py`.
- Test: `tests/unit/runtime/test_query_composition.py` if snapshot/config propagation is required.
- Modify: `docs/local-operations.md` with DeepSeek protocol and failure-diagnosis guidance.

**Interfaces:**
- `Settings.deepseek_protocol: Literal["auto", "chat", "responses"] = "auto"`.
- `ModelCall.protocol` is optional and inherits the snapshot/application default when omitted.
- `_create` selects the requested protocol explicitly; `auto` prefers Chat when both APIs are present and falls back to Responses only when Chat is unavailable.
- Structured calls add a provider-compatible JSON request hint, but `_validate_schema` remains authoritative.
- Diagnostic events contain only `schema_name`, `protocol`, `requested_model`, `actual_model`, `attempt`, `error_class`, `output_length`, and `output_sha256`.

- [ ] **Step 1: Write the failing tests**

Add tests that `chat` chooses `chat.completions.create` even when `responses.create` exists, `responses` chooses `responses.create`, `auto` prefers Chat, structured Chat requests contain `response_format={"type":"json_object"}`, fenced JSON validates after safe stripping, and schema exhaustion emits a diagnostic without the raw invalid output.

- [ ] **Step 2: Run the focused tests to verify RED**

Run:

```bash
conda run -n agentic-rag pytest --import-mode=importlib tests/unit/runtime/test_model_gateway.py -q
```

Expected: failures because protocol selection is currently based only on client attributes, structured requests do not send JSON hints, and fenced JSON is passed directly to `json.loads`.

- [ ] **Step 3: Implement minimal protocol-aware requests**

Add the validated protocol setting and make `_create` select the client path deterministically. For Chat structured calls, add the JSON object response hint. For Responses, add the compatible JSON text-format hint only when the selected provider/client accepts it; if a fake/provider rejects the optional hint with a parameter error, surface a categorized `protocol_error` rather than retrying it as an outage. Keep provider retries and the single repair attempt unchanged.

- [ ] **Step 4: Implement safe response normalization and diagnostics**

Strip only a leading/trailing Markdown JSON fence before validation. On validation failure, retain the existing repair message behavior but emit safe metadata (schema class name, protocol, model, attempt, error class, byte length, SHA-256) and never persist raw output. Preserve `MODEL_REPAIR_EXHAUSTED` and classify outage/circuit/protocol/schema errors distinctly.

- [ ] **Step 5: Run focused and Graph regression tests**

Run:

```bash
conda run -n agentic-rag pytest --import-mode=importlib tests/unit/runtime/test_model_gateway.py tests/unit/query tests/unit/observability -q
```

Expected: all existing route/grade/generation/audit tests pass, including cancellation, retry bounds, and no-raw-content assertions.

- [ ] **Step 6: Update Chinese diagnostics documentation and commit**

Document the meaning of `provider_outage`, `protocol_error`, `model_schema_invalid`, `model_unavailable`, and `circuit_open`, including the relevant log fields and the one-repair limit.

```bash
git add src/agentic_rag/config.py src/agentic_rag/runtime/model_gateway.py src/agentic_rag/observability/logging.py tests/unit/runtime/test_model_gateway.py tests/unit/runtime/test_query_composition.py docs/local-operations.md
git commit -m "fix: make deepseek structured calls protocol aware"
```

### Task 3: Real Graph/API acceptance with Mem0 and current snapshot

**Files:**
- Modify: `tests/e2e/test_query_pipeline_real_services.py` to add an opt-in Mem0-backed case using isolated user/collection/index namespaces.
- Modify: `tests/integration/runtime/test_query_pipeline.py` only for deterministic provider/provenance assertions.
- Modify: `scripts/run_api_eval.py` or the existing API acceptance helper only if it cannot pass the current snapshot and client provenance through the report.
- Modify: `docs/local-operations.md` with exact Conda commands and environment gates.
- Modify: `docs/development-progress.md` with acceptance evidence and known opt-in skips.

**Interfaces:**
- Real acceptance must use the compiled Graph/API path, not a direct service call.
- The report must include `evaluation_mode=api`, `client_provenance=real_query_api`, the current `runtime_config_snapshot_id`, `citation_coverage`, `unaudited_answer_count`, `user_leak_count`, `recovery_drill_passed`, `backup_restore_passed`, and explicit `memory_provider` status.

- [ ] **Step 1: Write the failing acceptance assertions**

Require the real API response to carry the current snapshot ID, `real_query_api` provenance, an audited answer, citation coverage of 1.0 for the seeded document, zero user leaks, and a non-degraded Mem0 memory status when the opt-in variables are present. Add an explicit skip reason when Mem0 or external services are not configured.

- [ ] **Step 2: Run the acceptance test to verify the new assertions fail before wiring**

Run:

```bash
conda run -n agentic-rag pytest --import-mode=importlib tests/integration/runtime/test_query_pipeline.py tests/e2e/test_query_pipeline_real_services.py -q
```

Expected: the new Mem0/provenance assertions fail until the composition and report wiring are complete, or skip only when the required opt-in variables are absent.

- [ ] **Step 3: Wire the real Mem0 settings and provenance assertions**

Use isolated MySQL/Redis/Elasticsearch/Mem0 namespaces, initialize the required index/alias and migration state, run one Graph/API query, and verify memory status and provenance in the acceptance report. Do not weaken citation, audit, leakage, recovery, or backup gates when memory is degraded.

- [ ] **Step 4: Run the full verification matrix**

```bash
conda run -n agentic-rag pytest --import-mode=importlib tests -q
conda run -n agentic-rag ruff check src tests evals scripts
conda run -n agentic-rag mypy src evals
conda run -n agentic-rag python -m evals.validate_datasets evals/datasets
```

When local services and Mem0 credentials are available, also run the real-service acceptance command and `scripts/verify_acceptance.py` against the generated summary. Record exact pass/skip counts and any provider outage/schema diagnostic events in the Chinese progress document.

- [ ] **Step 5: Commit**

```bash
git add tests/integration/runtime/test_query_pipeline.py tests/e2e/test_query_pipeline_real_services.py scripts docs/local-operations.md docs/development-progress.md
git commit -m "test: verify real graph api and mem0 provenance"
```

## Self-review checklist

- Mem0 default-on is covered without removing explicit disabled-mode tests.
- Local unauthenticated ES remains supported only for loopback; remote auth remains mandatory.
- Provider outage and schema-invalid outputs have separate categories and safe diagnostics.
- JSON normalization never relaxes Pydantic validation or stores raw output.
- API/Graph acceptance checks the current snapshot, real client provenance, citation, audit, leak, recovery, backup, and memory status gates.
- Documentation remains Chinese and includes exact commands, environment gates, and degradation behavior.
