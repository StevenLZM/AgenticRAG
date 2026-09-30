# Real RAG Evaluation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace simulated business evaluation with reproducible real uploads, ingestion, HTTP queries and real Ragas scoring for 8 documents and 24 cases.

**Architecture:** Use the existing production API/Worker pipeline in a separately configured stack. Freeze source-level gold before queries, bind it to actual ingestion provenance, collect true ranked retrieval outputs, and persist every case and judge outcome for resumability.

**Tech Stack:** Python 3.11, FastAPI, MySQL, Redis Streams, Elasticsearch, Docling, Qwen embeddings, configured generation/judge providers, Ragas, pytest.

**Spec:** `docs/superpowers/specs/2026-09-16-real-rag-evaluation-design.md`

**Execution checkpoint (2026-09-18):** Tasks 1–7 completed for the approved first real evaluation:8 real ingestions,24 actual HTTP Runs and Ragas results, live readonly report verification, retired seeded-quality entry point, failure-safe resume, report/docs and scoped review. Results and limitations: `docs/rag-evaluation-report.md`; exact resources/recovery: `docs/rag-evaluation-progress.md`. Correct refusal1/4; provider tokens unavailable, release drills not performed. Completion is not a release-quality pass. Checklists below preserve the original plan; the recovery ledger is authoritative for execution status.

## Global Constraints

- Resume from `docs/rag-evaluation-progress.md`; update it after each independently verified deliverable.
- Existing dirty-worktree changes belong to the user. No resets, bulk staging, or automatic commits.
- Business evaluation has no fixture/fixed-embedding/gold-answer shortcut. Unit-test doubles never produce release-quality evidence.
- Only synthetic evaluation data; never print provider secrets. Do not alter global `.env.local` or stop business services.
- Use real HTTP upload/query and real providers; model/network failure is recorded, never replaced by fake success.
- Execution order is Task 1 through Task 7. Each task follows failing test → minimal change → verification → recovery-note update.

### Task 1: Gold/metric contracts and remove fixture entry

**Files:** `evals/metrics.py`, `evals/models.py`, `evals/run.py`, `evals/report.py`, `tests/unit/evals/test_metrics.py`, `tests/unit/evals/test_runner.py`, `tests/integration/evals/test_real_query_evaluation.py`.

**Interfaces:** Keep `recall_at_k`, `mrr`, `ndcg_at_k`; extend `EvaluationCase` with answerability and compatible empty references for explicitly unanswerable cases. CLI rejects fixture. Reports do not infer realness merely from a caller string.

- [ ] Add and run `assert ndcg_at_k(["p1"], {"p1", "p2"}, 10) == pytest.approx(1 / (1 + 1 / math.log2(3)))`; expect failure on current implementation.
- [ ] Use requested K, not returned-result count, for ideal ranking length; verify empty/nonpositive/duplicate cases.
- [ ] Add gold tests: answerable cases require references; unanswerable cases require no gold parents; unsupported/missing fields fail validation.
- [ ] Add CLI test rejecting `--mode fixture`; remove `FixtureQueryClient` and fixture business defaults. Relabel fake Graph boundary tests as contract-only.
- [ ] Test unit evals and API evaluator contracts. Record exact output and remaining migration work in recovery log.

### Task 2: Versioned synthetic corpus and 24 pre-labelled cases

**Files:** new `evals/corpus.py`, `evals/datasets/real_corpus/`, `tests/unit/evals/test_corpus.py`; generated artifacts under run-specific `var/artifacts/evals/`.

**Interfaces:** `build_corpus(output_dir: Path) -> Path` returns a manifest path. Manifest includes filenames, sha256, logical doc IDs, source fact anchors, questions/reference answers/answerability/tags.

- [ ] Test exactly 8 nonempty supported files, exactly 24 unique cases (12 single, 8 multi, 4 unanswerable), references resolve to declared source facts, no gold derived from system retrieval.
- [ ] Generate consistent fictional facts with distractors (distinct department, dates, products); avoid random expected answers. Freeze manifest SHA256.
- [ ] Read PDF and spreadsheet skills before creating those formats; render/inspect PDF pages and verify workbook values/formulas.
- [ ] Run corpus tests and validate generated files through the production upload validator without bypasses. Record manifest and file hashes.

### Task 3: Real Ragas judge and strict error reporting

**Files:** `evals/ragas_adapter.py`, new `evals/judge.py`, `pyproject.toml`, `tests/unit/evals/test_ragas.py`, runner/report consumers.

**Interfaces:** Judge consumes question, answer, ordered contexts, reference and returns finite named metrics plus model/version metadata; unavailable configuration raises a typed preflight error. Per-case provider errors are explicit failures.

- [ ] Tests reject missing question/context/config and fake available scores; verify bounded retries/error records without running real providers in unit tests.
- [ ] Inspect/pin compatible Ragas dependencies; install required evaluation dependencies without silently upgrading the live app's core runtime.
- [ ] Configure real judge/embeddings from existing authorized provider settings, never hard-code secrets. Use official Ragas APIs verified against the installed version.
- [ ] Verify provider preflight with synthetic non-sensitive input after config checks; record response metadata, not credentials.

### Task 4: Isolated stack and actual upload/ingestion

**Files:** new `scripts/run_real_rag_evaluation.py`, new `evals/real_stack.py`, `evals/upload.py`, `tests/unit/evals/test_upload.py`.

**Interfaces:** Upload orchestrator records `(sha256, job_id, document_id, document_version_id, status)` and resumes completed uploads. Stack returns base URL and scoped collector dependencies; exact allocated resources are ledgered before use.

- [ ] Test rejected upload, timeout, failed ingestion and interrupted-resume behavior; each must remain a failed/pending stage, not a completed case.
- [ ] Preflight MySQL/Redis/ES/providers and privileges. Allocate isolated API/Worker configs and namespaces with no shared consumer/dispatcher collisions.
- [ ] Protect the shared `agenticrag-children-active` alias: use an isolated ES instance (preferred) or tested isolated alias configuration before any publication; a unique index generation alone is insufficient.
- [ ] Start real API/ingestion/query processes; POST multipart files over HTTP and poll jobs to completion.
- [ ] Resolve pre-labelled source anchors to actual Parent/AST spans in matching user/document version. Persist mapping; fail ambiguous/unmapped anchors rather than silently replacing gold.

### Task 5: Truthful runtime collection

**Files:** `evals/clients.py`, new `evals/collector.py`, scoped query/retrieval observability or artifacts, `tests/unit/evals/test_clients.py` and collector tests.

**Interfaces:** Collector takes run_id, user_id, snapshot_id and returns actual route, final answer, ranked per-stage/per-round parents, ordered contexts, terminal outcome and usage. It never substitutes gold, expected_route or citations as ranks.

- [ ] Test foreign-user/stale-snapshot rejection, actual-route projection, nonempty contexts on answerable success, per-round rank preservation, unavailable observation reported explicitly.
- [ ] Persist/retrieve bounded trusted runtime outputs without exposing raw prompts or chain-of-thought through the public API.
- [ ] Use scoped read-only collector with the real HTTP client's run_id. Distinguish single-round retrieval metrics from multihop final evidence coverage.

### Task 6: Real preflight cases and full run

**Files:** real orchestrator, runner result schema, report, run-specific manifest/results/summary.

- [ ] Run 3 preflight cases (single, multi, no-answer), verify actual upload IDs, source mapping, HTTP Run IDs and real judge outputs.
- [ ] Resolve implementation defects with targeted regression tests; do not tune gold to outputs or silently discard failing cases.
- [ ] Run all 24 cases with durable per-stage resume hashes. Record success/failure/timeout/refusal and missing-metric reasons for every case.
- [ ] Confirm Ragas coverage, retrieval-stage completeness and no fake provenance; publish scores even when low, label incomplete runs honestly.

### Task 7: Review, report and handoff

**Files:** `docs/local-operations.md`, `docs/development-progress.md`, `docs/rag-evaluation-progress.md`, `scripts/verify_acceptance.py` and related tests.

- [ ] Run `conda run -n agentic-rag pytest --import-mode=importlib tests/unit/evals tests/integration/evals tests/integration/api/test_query_runs.py -q` plus affected runtime/ingestion tests.
- [ ] Run Ruff on touched Python and `git diff --check`; request focused read-only review.
- [ ] Document run commands, restart/resume behavior, real metrics with valid sample counts, remaining gaps and preserved resource names.
- [ ] Mark complete only when upload→ingestion→query→judge→report really ran; otherwise record the exact next command/blocker. No cleanup without exact validated evaluation targets.
