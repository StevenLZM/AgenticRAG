# Phase 4 Task 3 report — query entry flow

## Scope delivered

- JSON-only `QueryState` creation and reconstruction helpers; process-owned
  services are captured in `QueryRuntimeDependencies` / `FastRagDependencies`.
- A `MemoryContextLoader` that loads once before routing and stores only the
  JSON-safe `MemoryContext` representation in state.
- A light-model router using the versioned `router_v1` prompt. Structured
  validation/operational failures fail closed to the research route.
- A Fast RAG path that constructs a server-owned `RetrievalRequest`, invokes
  `RetrievalService.retrieve()` once, packs a query coverage target and uses a
  strict evidence-grade decision to generate, research, clarify, or refuse.
- `EvidenceGrade`, with bounded, nonblank gaps and forbidden extra fields.

## TDD evidence

Initial RED command:

```text
conda run -n agentic-rag python -m pytest --import-mode=importlib tests/unit/query/test_router_fast_path.py -q
6 failed: missing EvidenceGrade, query.router and query.fast_rag (expected)
```

Final checks:

```text
conda run -n agentic-rag python -m pytest --import-mode=importlib tests/unit/query/test_router_fast_path.py -q
7 passed

conda run -n agentic-rag ruff check src/agentic_rag/models src/agentic_rag/query tests/unit/query/test_router_fast_path.py
All checks passed

conda run -n agentic-rag mypy src/agentic_rag/models src/agentic_rag/query
Success: no issues found in 9 source files

git diff --check
clean
```

An earlier focused regression run also completed with `178 passed` across
query/runtime/memory/retrieval/persistence before the final added retrieval
degradation test.
