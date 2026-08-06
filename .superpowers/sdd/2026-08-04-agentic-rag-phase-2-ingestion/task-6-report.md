# Task 6 implementation report

Status: DONE

## Scope delivered

- Added JSON-only `IngestionState`, durable runtime projection, Artifact pointers, and fenced Job Claim identity.
- Added the fixed LangGraph sequence `load_job -> upload_safety_gate -> parse_fragments -> assemble_canonical -> content_safety_gate -> validate_canonical -> chunk -> embed_and_stage -> publish -> finalize` with the mandatory quarantine short-circuit.
- Compiled the graph with the injected Phase 1 Checkpoint port and stable `thread_id=ingestion:{job_id}`. A takeover refreshes only owner/generation with `aupdate_state`, then resumes the pending LangGraph task with `ainvoke(None)`.
- Added `DefaultIngestionPipeline`, which composes the existing Phase 2 scanner, Docling parser, assembler, content scanner, deterministic Parent/Child chunker, IndexWriter, VersionPublisher, Artifact Store, and MySQL lifecycle adapter. Graph state contains references and JSON values, never clients or full parsed documents.
- Added `SqlAlchemyIngestionJobStore` with atomic Claim, heartbeat/assert lease fencing, trusted runtime load, retry/failure transition, completed/quarantined terminal transitions, and explicit trusted local quarantine approve/reject.
- Added Redis Streams worker behavior for new and reclaimed messages, heartbeat cancellation on lease loss, terminal-only ACK, retry preservation, third-attempt dead-letter, stable stream/group/dead-stream names, and an in-process periodic Reconciler loop.
- Added local scripts for worker launch and explicit quarantined-version review. No Docker or external scheduler was introduced.
- Extended the local Artifact Store with integrity-checked byte read and URI description so a resumed graph can reconstruct the original immutable source reference from durable SQL identity.

## TDD evidence

Initial RED:

- `conda run -n agentic-rag python -m pytest tests/unit/ingestion/test_worker.py -q`
  - collection failed with `ModuleNotFoundError: agentic_rag.ingestion.state`.
- `conda run -n agentic-rag python -m pytest -m e2e tests/e2e/test_ingestion_pipeline.py -q`
  - collection failed with `ModuleNotFoundError: agentic_rag.ingestion.graph`.

Final targeted GREEN:

- `conda run -n agentic-rag python -m pytest tests/unit/ingestion/test_worker.py -q`
  - `3 passed`.
- `conda run -n agentic-rag python -m pytest -m e2e tests/e2e/test_ingestion_pipeline.py -q`
  - `3 passed`.
- The E2E test uses a real `AsyncSqliteSaver`, injects a crash after deterministic staging, resumes with a new lease generation, proves parsing is not repeated, proves staging can repeat without duplicate Parent/Child IDs, and verifies the exact checkpoint node sequence and JSON serialization.

## Phase 2 gate

- `conda run -n agentic-rag python -m pytest tests/unit/ingestion tests/unit/safety -q`
  - `121 passed`.
- `conda run -n agentic-rag python -m pytest -m integration tests/integration/ingestion -q`
  - `9 passed, 7 skipped`, with existing Docling warnings. The passed Docling fixture matrix covers text PDF, scanned PDF/OCR, plain text, and Excel. The seven MySQL/Elasticsearch lifecycle cases remain intentionally gated by explicit disposable test DSNs.
- `conda run -n agentic-rag python -m pytest -m e2e tests/e2e/test_ingestion_pipeline.py -q`
  - `3 passed`.
- `conda run -n agentic-rag python -m pytest -q`
  - `241 passed, 28 skipped`, 18 existing third-party/deprecation warnings.

## Static verification

- `conda run -n agentic-rag ruff check src scripts tests` — clean.
- `conda run -n agentic-rag mypy src scripts/run_ingestion_worker.py scripts/review_quarantined_version.py tests/unit/ingestion/test_worker.py tests/e2e/test_ingestion_pipeline.py` — 45 source files clean.
- `conda run -n agentic-rag python -m py_compile scripts/run_ingestion_worker.py scripts/review_quarantined_version.py` — clean.
- `conda run -n agentic-rag python scripts/review_quarantined_version.py --help` — CLI contract rendered successfully.

## Known limitations

- Real MySQL/Elasticsearch/Redis end-to-end execution requires the existing explicit disposable DSN environment variables; the default suite never guesses or mutates developer services.
- The local worker constructs the Docling tokenizer on startup and may require the configured Hugging Face model to already be cached or locally downloadable.
- SQLite checkpointing deliberately retains the Phase 1 single-ingestion-worker constraint; horizontal workers require replacing the Checkpoint port first.

## Official review fix round 1

The review of `ce5edd0` found one Critical delivery-generation defect, five
Important runtime/convergence defects, and two Important test-depth gaps. This
round closes them without adding a second scheduler or an alternate ingestion
architecture:

- Quarantine approval now atomically rotates the outbox attempt generation and
  recreates a missing row. Redis therefore appends a new notification after the
  old quarantined notification was ACKed, while retries inside either generation
  remain Lua-deduplicated. Explicit Redis regressions cover both upload-time and
  post-parse quarantine states through approval, new Claim, and completion.
- Terminal failure now atomically records durable DLQ intent, fails an
  unpublished Version and pointerless Document, and preserves an existing active
  Document. A failure after publication instead converges the Job to COMPLETED;
  reconciler publication also resolves a stranded Job to COMPLETED.
- DLQ publication is idempotent per `{job_id}:{attempt_count}`. A crash or Redis
  failure between the SQL terminal commit and Dead Stream write leaves the
  message pending; terminal fast-path delivery repairs the DLQ before ACK.
  Expired leases use the same three-attempt ceiling and rotate the outbox
  generation for both retries and terminal repair delivery.
- The durable Claim fence is threaded through Parser Fragment writes, Canonical
  and Chunk Artifact writes, embedding batches, Parent/Child staging, Manifest
  persistence/attachment, Elasticsearch bulk batches, and every publication
  mutation. Focused fault-injection tests prove work stops at parser,
  cross-store staging, and publication boundaries after lease loss.
- `run_forever()` isolates broker and per-message failures, applies bounded
  exponential backoff with structured logs, keeps the reconciler resilient, and
  supports stop-claiming/drain-current-work shutdown. Heartbeat lease loss
  cancels and awaits the graph; tests cover broker recovery, graceful draining,
  and cancellation without ACK.
- The tokenizer model is an explicit validated setting aligned to the Qwen
  embedding family (`Qwen/Qwen3-Embedding-0.6B`), instead of a hidden MiniLM
  launcher constant.
- Alembic revision `0005_ingestion_dead_letter_state` adds the paired durable DLQ
  fields and invariant. Real MySQL schema inspection and real Redis DLQ Lua
  behavior are opt-in behind explicit disposable local DSNs.
- A new production-composition E2E covers text, Excel, text PDF, and scanned/OCR
  PDF through real DocumentService, outbox/Redis, Worker, LangGraph with SQLite,
  Docling, Artifact Store, MySQL Parent staging, Elasticsearch Child staging,
  publication, and final lifecycle state. It injects a crash after real
  cross-store staging and resumes the same message, then verifies active rows
  and deterministic IDs. Only embedding is a deterministic test adapter, so no
  external paid model API is required.

Review-driven RED evidence included the durable DLQ crash window, the third
expired-lease lifecycle, missing/unchanged quarantine outbox delivery generation,
and publication-after-final-commit lifecycle mismatch. Focused GREEN evidence:

```text
compound side-effect lease boundaries + worker shutdown/heartbeat: 5 passed
Task 6 focused worker/store/lifecycle/graph set: 93 passed, 3 DSN-gated skipped
adjacent Task 5 lifecycle suites: 46 passed
```

Fresh verification after the fix round:

```text
conda run -n agentic-rag python -m pytest tests/unit/ingestion tests/unit/safety -q
133 passed

conda run -n agentic-rag python -m pytest -m integration tests/integration/ingestion -q
9 passed, 9 skipped, 18 existing Docling warnings

conda run -n agentic-rag python -m pytest -m e2e tests/e2e -q
3 passed, 1 skipped

conda run -n agentic-rag python -m pytest -q
256 passed, 32 skipped, 18 existing Docling warnings

conda run -n agentic-rag ruff check src scripts tests
All checks passed!

conda run -n agentic-rag mypy src scripts/run_ingestion_worker.py \
  scripts/review_quarantined_version.py tests/unit/ingestion/test_worker.py \
  tests/unit/ingestion/test_worker_store.py tests/e2e/test_ingestion_pipeline.py \
  tests/e2e/test_ingestion_pipeline_real_services.py \
  tests/integration/ingestion/test_quarantine_approval_delivery.py
Success: no issues found in 48 source files
```

The three-service E2E and real Redis/MySQL contracts were collected but skipped
because their explicit test DSNs were absent. Once any test DSN is configured,
unavailable infrastructure fails the test rather than silently falling back to
fakes or guessed developer services.
