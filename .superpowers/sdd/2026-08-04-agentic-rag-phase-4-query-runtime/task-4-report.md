# Phase 4 Task 4 report — Dynamic Todo ResearchAgentLoop and Tools

## Scope completed

- Added deterministic, ownership-checked Todo contracts and reducer.
- Added AST-only bounded arithmetic calculator; it never calls `eval`.
- Added strict six-action research union, constrained tools, contextual prompt builder, and a four-round observation loop.
- Retrieval remains through the injected high-level `RetrievalService` boundary; all loop output remains JSON-compatible.
- Invalid model actions fail closed. Tool failures retain prior evidence and block active work; cancellation propagates.

## TDD evidence

Initial RED run (before production modules existed):

```text
ModuleNotFoundError: No module named 'agentic_rag.query.todos'
ModuleNotFoundError: No module named 'agentic_rag.query.calculator'
```

The research-loop/context RED run then failed because their modules were absent. The GREEN suite passed after implementation.

## Verification

```text
conda run -n agentic-rag python -m pytest --import-mode=importlib \
  tests/unit/query/test_todos.py tests/unit/query/test_calculator.py \
  tests/unit/query/test_research_loop.py -q
16 passed in 0.17s

conda run -n agentic-rag python -m pytest --import-mode=importlib \
  tests/unit/query tests/unit/runtime tests/unit/memory tests/unit/retrieval \
  tests/unit/persistence -q
198 passed in 2.63s

ruff check ...
All checks passed!

mypy ...
Success: no issues found in 5 source files

git diff --check
(clean)
```
