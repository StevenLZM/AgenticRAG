# Task 5 Fix Round 2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close the three scoped Task 5 convergence findings without adding Phase 2 Task 6 or Phase 3 behavior.

**Architecture:** Keep completed deletion tombstones eligible for idempotent physical sweeps while reporting completion only on the first fenced-to-completed transition. Isolate every outbox redispatch exception at the item boundary. Fail closed for pointerless active Documents by physically deactivating their active Versions and atomically moving the invalid Document state to failed.

**Tech Stack:** Python 3.11, SQLAlchemy asyncio, MySQL/SQLite, Redis Streams, Elasticsearch 8, pytest, Ruff, mypy.

## Global Constraints

- Work only in Phase 2 Task 5 files and adjacent Task 5 regressions.
- Preserve Parent then Child activation, old Child then old Parent deactivation, and SQL finalization order.
- Keep real-service tests behind explicit disposable DSNs.
- Use `conda run -n agentic-rag` for every Python verification command.

---

### Task 1: Repeatable completed-deletion sweep

**Files:**
- Modify: `src/agentic_rag/ingestion/reconciler.py`
- Modify: `src/agentic_rag/persistence/lifecycle.py`
- Test: `tests/unit/ingestion/test_publisher_reconciler_unit.py`
- Test: `tests/unit/persistence/test_lifecycle.py`

**Interfaces:**
- Produces: `DeletedDocument.first_completion: bool` and `mark_deletion_reconciled(document_id) -> bool`.
- Preserves: completed tombstones remain scan candidates; only a first successful completion appears in `ReconcileReport.reconciled_deletions`.

- [x] Write a failing interleaving test that completes cleanup, recreates Parent, Child, and Artifact data afterward, runs reconciliation again, and observes all late data removed without reporting the Document twice.
- [x] Run the focused test and confirm completed tombstones are currently omitted.
- [x] Select both eligible `fenced` and `completed` deleted Documents, return whether the row is a first completion, and make the completion update return a boolean transition result.
- [x] Run deletion unit and SQL lifecycle tests to green.

### Task 2: Outbox SQL-mark failure isolation

**Files:**
- Modify: `src/agentic_rag/ingestion/reconciler.py`
- Test: `tests/unit/ingestion/test_publisher_reconciler_unit.py`
- Test: `tests/integration/persistence/test_redis_streams.py`

**Interfaces:**
- Preserves: `OutboxRedispatcher.redispatch(row)` publishes then marks SQL dispatched.
- Produces: per-item `Exception` isolation while `BaseException` still propagates.

- [x] Write a failing reconciler test where Redis publication succeeds and SQL marking raises `RuntimeError`, then assert a later deletion or pointer repair completes in the same pass.
- [x] Run it and confirm `RuntimeError` currently escapes.
- [x] Catch `Exception` around each complete redispatch operation and continue without adding the Job to `redispatched_jobs`.
- [x] Retain the real Redis Lua same-generation retry assertion and run focused tests to green.

### Task 3: Pointerless active Document repair

**Files:**
- Modify: `src/agentic_rag/ingestion/reconciler.py`
- Modify: `src/agentic_rag/persistence/lifecycle.py`
- Test: `tests/unit/persistence/test_lifecycle.py`
- Test: `tests/unit/ingestion/test_publisher_reconciler_unit.py`

**Interfaces:**
- Preserves: existing mismatch action `deactivate` and `resolve_deactivated_version(version_id)`.
- Invariant: resolving any active Version for an `ACTIVE`/null-pointer Document also moves the Document to `FAILED`; subsequent mismatch candidates deactivate every remaining physical and SQL-active Version.

- [x] Write a failing SQLite lifecycle test with an active/null Document and active Versions; run all classified deactivations and expect a failed/null Document with no active Versions.
- [x] Run it and confirm the old deactivation path leaves active/null drift.
- [x] Extend the locked deactivation resolution so an active/null Document is atomically moved to failed when its physical Version is resolved; keep all remaining active Versions eligible for later physical deactivation.
- [x] Run pointer classification and reconciler tests to green.

### Task 4: Report, review, verification, and commit

**Files:**
- Modify: `.superpowers/sdd/2026-08-04-agentic-rag-phase-2-ingestion/task-5-report.md`

- [x] Run focused tests for all three RED-to-GREEN cases and adjacent lifecycle suites.
- [x] Run `conda run -n agentic-rag python -m pytest -q`.
- [x] Run `conda run -n agentic-rag ruff check .`, `conda run -n agentic-rag mypy src`, targeted test mypy, and `git diff --check`.
- [x] Leave the next official scoped review to the controller after commit.
- [x] Update the implementation report with exact evidence, then commit one reviewable fix-round change.
