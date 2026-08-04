# Production Agentic RAG Implementation Roadmap

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Deliver the approved single-host production Agentic RAG system as five ordered, independently reviewable implementation phases.

**Architecture:** Build the durable storage and runtime contracts first, then ingestion, deterministic retrieval, QueryGraph/Agent runtime, and finally evaluation and production acceptance. Each phase has its own detailed plan and must pass its verification gate before the next phase starts.

**Tech Stack:** Python 3.11, FastAPI, LangGraph, DeepSeek, Qwen `text-embedding-v3`, Docling, Elasticsearch 8, MySQL, Redis Streams, SQLite Checkpointer, mem0ai 2.0.12, BGE Cross-Encoder, Ragas, pytest.

## Global Constraints

- Run locally without Docker, Kubernetes, microservices, Celery, Dramatiq, RQ or ARQ.
- Use LangGraph as the only Graph and Agent orchestration framework.
- Main model: DeepSeek `deepseek-v4-pro`; light model: `deepseek-v4-flash`.
- Embedding: Qwen `text-embedding-v3`, exactly 1024 dimensions.
- Reranker: `BAAI/bge-reranker-v2-m3`, loaded once in the Query Worker.
- Elasticsearch endpoint defaults to `http://localhost:9200`; MySQL and Redis endpoints come from environment settings; Redis defaults to `redis://127.0.0.1:6379/0`.
- MySQL stores Parent, Run, Job, Outbox, audit and deletion state; ES stores active Child search records; local files store versioned Artifacts.
- SQLite Checkpoint uses separate Query and Ingestion files in WAL mode and exactly one writer process per file.
- `user_id` defaults to `default_user`, is injected by the API, and cannot be supplied or changed by an Agent or retrieval request.
- V1 assumes trusted callers; `user_id` is a data scope, not authentication.
- Retrieved documents and Memory are untrusted data and can never override system instructions or Tool schemas.
- No Token or cost budget routing; Token and estimated cost are observability fields only.
- Unit tests never require live model APIs. Integration tests requiring local ES, MySQL or Redis use explicit pytest markers and fail with a clear dependency message when selected without the service.
- Development and evaluation environments install with `conda run -n agentic-rag python -m pip install -e '.[dev,eval]'`.
- Run all commands inside the Conda environment `agentic-rag`; use `conda run -n agentic-rag ...` in non-interactive scripts.
- Every task follows red-green-refactor, ends with focused tests, and creates one reviewable commit.

---

## Ordered Phase Plans

1. [Phase 1 — Foundation and Persistence](./2026-08-04-agentic-rag-phase-1-foundation.md)
   Creates the Python package, typed contracts, settings, MySQL schema, repositories, Outbox/Redis ports, SQLite Checkpoint and Artifact Store.

2. [Phase 2 — Document Ingestion](./2026-08-04-agentic-rag-phase-2-ingestion.md)
   Delivers upload safety, Docling Fragment/Canonical AST assembly, Parent-Child Chunking, embedding/index staging, Publisher, Reconciler and Ingestion Worker recovery.

3. [Phase 3 — Retrieval and Evidence](./2026-08-04-agentic-rag-phase-3-retrieval.md)
   Delivers server-scoped Dense/BM25 retrieval, RRF, Cross-Encoder reranking, Parent fetch, RetrievalPipelineGraph and deterministic EvidenceBuilder.

4. [Phase 4 — Query and Agent Runtime](./2026-08-04-agentic-rag-phase-4-query-runtime.md)
   Delivers ModelGateway, mem0ai MemoryService, QueryGraph, true ResearchAgentLoop, Subagents, mandatory audits, durable Query Runs, cancellation, SSE and query APIs.

5. [Phase 5 — Evaluation and Production Acceptance](./2026-08-04-agentic-rag-phase-5-evaluation-operations.md)
   Delivers tracing and metrics, offline evaluation, adversarial/load/recovery suites, migrations, backup/restore, process scripts and the final acceptance gate.

## Phase Gates

| Phase | Required evidence before continuing |
|---|---|
| 1 | Unit suite passes; migrations upgrade/downgrade on a clean local database; Redis duplicate notification and SQLite replay tests pass. |
| 2 | Text PDF, scanned PDF, text and Excel fixtures produce valid Canonical AST, Parent/Child records and a recoverable active version. |
| 3 | Hybrid retrieval integration tests pass with zero user leakage; deterministic Recall@6, NDCG@10 and MRR fixtures produce expected values. |
| 4 | Fast RAG, multi-hop loop, Subagent, audit, cancellation, checkpoint recovery and SSE reconnection tests pass. |
| 5 | Full unit/integration/e2e/eval suite passes; backup restoration and worker interruption drills pass; hard quality gates remain zero violations. |

## Design Coverage Map

| Approved design area | Owning implementation tasks |
|---|---|
| Local three-process runtime, Run lifecycle, Outbox, Lease, cancellation and Checkpoint | Phase 1 Tasks 3–6; Phase 4 Tasks 8–9 |
| QueryGraph routing, Fast RAG, ResearchAgentLoop, Todo, Subagent and compaction | Phase 4 Tasks 3–7 |
| Dense/BM25 filters, RRF, Cross-Encoder, Parent fetch and EvidenceBuilder | Phase 3 Tasks 1–6 |
| Docling AST, cross-page assembly, Parent-Child Chunking, indexing and reconciliation | Phase 2 Tasks 1–6 |
| Working/long-term memory, Mem0 lifecycle and deletion | Phase 1 Task 5; Phase 4 Task 2 |
| Evidence, Faithfulness and citation gates | Phase 4 Tasks 6–7 |
| Safety boundaries for upload, retrieved context, Memory and Tool input | Phase 2 Tasks 1–2; Phase 3 Tasks 1 and 6; Phase 4 Tasks 2, 4 and 6 |
| Runtime versioning, tracing, online metrics and three-layer evaluation | Phase 1 Task 2; Phase 5 Tasks 1–3 |
| Retry/degradation, load/backpressure, recovery and local operations | Phase 3 Tasks 3 and 5; Phase 4 Task 8; Phase 5 Tasks 4–5 |

## Final Verification Commands

```bash
ruff check src tests evals
mypy src
pytest -m "not integration and not e2e and not live_model" -q
pytest -m integration -q
pytest -m e2e -q
pytest -m live_model tests/smoke -q
python -m evals.run --dataset evals/datasets/baseline.jsonl --output artifacts/evals/final
python scripts/verify_acceptance.py --report artifacts/evals/final/summary.json
```

The final verifier must exit non-zero if `user_leak_count != 0`, citation coverage is below 100%, an unaudited answer was returned, a required service is unhealthy, or the recovery drill did not complete.
