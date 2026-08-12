# Phase 4 Task 5 report — bounded research subagents

## Delivered

- Added `ConcurrencyManager` with injectable global Run/LLM/Reranker budgets and a per-Run subagent semaphore capped by server configuration.
- Added isolated `ChildResearchState`, narrow retrieval/calculator-only `SubagentTools`, bounded timeout/cancellation cleanup, and deterministic `EvidenceReducer`.
- Wired `DelegateResearch` into `ResearchAgentLoop`: it accepts only server-known Todo IDs, passes completed dependency IDs, preserves completed results, blocks timed-out Todos, and never grants final-answer or recursive-delegation capability.
- Evidence reduction sorts source results and evidence identifiers deterministically, keeps the first stable provenance record, and rejects manifest/item provenance mismatches.

## TDD evidence

Initial RED run, before the new modules existed:

```text
6 failed — ModuleNotFoundError: agentic_rag.query.subagents
```

After implementation:

```text
tests/unit/query/test_subagents.py: 7 passed
tests/unit/query + runtime + memory + retrieval + persistence: 206 passed
ruff check src tests: passed
mypy src: Success: no issues found in 72 source files
git diff --check: passed
```

## Scope

Task 6 audits, Task 7 QueryGraph, and Task 8/9 persistence/API work were not implemented.
