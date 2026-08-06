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
