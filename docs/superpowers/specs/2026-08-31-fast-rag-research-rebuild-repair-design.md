# FastRAG Evidence and Research Escalation Repair Design

**Date:** 2026-08-31

## Context

Run `01a056d8-2f55-73ab-9e46-9cdf40545116` started on the FastRAG route for the question `刘泽明工作经历`, packed three parent fragments, failed the evidence-sufficiency gate, and escalated into the Research Agent loop. The Research loop completed one delegated retrieval, then exhausted model retries and returned `cannot_answer` after 221.906 seconds.

The incident has four independent but compounding causes:

1. Parent headings exist in MySQL and Elasticsearch but are dropped when a stored parent is hydrated into `ParentEvidence` and packed into prompt evidence. Employer and job-title facts therefore help retrieval but are invisible to the evidence grader and answer generator.
2. The canonical PDF reading order places right-column date ranges after unrelated left-column content. Parent chunking then flushes on every paragraph/list transition, producing tiny fragments such as `业绩：获` and separating dates, headings, duties, and continuations.
3. The Evidence Builder uses literal full-question substring matching for target coverage and caps generic searches at three parents per document. A name-heavy profile block can displace work-history evidence even when the reranker found the relevant section.
4. Delegated evidence is added to a local Research list but not to the next supervisor prompt. Delegated results do not carry raw retrieval batches to the graph's post-Research Evidence Builder. Research structured calls also receive only generic JSON mode without a complete action schema, causing schema-repair calls that multiply the 30-second provider timeout.

## Goals

- A work-history question must expose employer, job title, date range, duties, and continuation text to the evidence grader when those facts exist in the source document.
- FastRAG must answer the reproduced work-history query without entering Research when the rebuilt source contains sufficient evidence.
- If Research is required, every successful delegated retrieval must be visible to the next supervisor action and must remain rebuildable from server-derived retrieval batches.
- Structured model calls must receive an exact schema contract and remain within a bounded total retry-and-repair deadline.
- The local dataset must be rebuilt from preserved source artifacts after deleting the old physical Elasticsearch index.
- No SQL schema migration is required.

## Non-goals

- General-purpose resume understanding or an LLM-based document-layout repair system.
- A production zero-downtime, dual-index rollout.
- Hard deletion of source artifacts, query history, or database audit records.
- A general PII-redaction subsystem. The rebuild must not log or expose personal contact fields, but broader document-level PII policy is separate work.

## Selected Rollout Strategy

The repair uses a destructive local rebuild with an explicit backup boundary:

1. Stop Query and Ingestion workers and verify that no query run or ingestion job is active.
2. Create and verify a local backup using the existing backup tooling.
3. Copy each active document source artifact into a rebuild staging directory outside its document artifact scope. Deduplicate the staging inventory by `(user_id, content_hash)` so repeated uploads of the same resume are restored once.
4. Soft-delete the existing active documents and run deletion reconciliation to completion. This deactivates old MySQL parents and removes their document-scoped artifacts, so the staging copy is mandatory.
5. Delete only the resolved physical index `agenticrag-children-index-v2` and remove or verify the active alias. Wildcards and unresolved environment-variable targets are forbidden.
6. Deploy the repair with `ingestion_pipeline_version=ingestion-v2` and `index_generation=index-v3`.
7. Re-upload the staged source files, wait for every ingestion job to publish, then start the Query worker.
8. Run structural, retrieval, query, and latency acceptance checks. If any hard gate fails, stop queries and restore the verified backup.

Although the old index is deleted, the new index uses `index-v3` rather than reusing `index-v2`. Chunk identity, contextualized text, and embeddings have changed, so a new generation prevents stale checkpoints and evidence from appearing compatible with rebuilt data.

## Architecture

### 1. Layout-aware canonical ordering

Add a deterministic layout-association pass to `GlobalAssembler` before heading-hierarchy repair. The pass handles a narrowly defined PDF pattern: a short date-range paragraph in a right column aligned with a heading in the left column.

A date block is relocated immediately after a heading only when all of the following are true:

- both blocks have a single-page bounding box on the same page;
- the candidate text matches the repository-owned date-range grammar;
- the date block is geometrically to the right of the heading;
- their vertical overlap exceeds the configured deterministic threshold;
- there is exactly one best heading candidate within the tolerance.

Ambiguous or missing geometry leaves the source order unchanged. The pass never fabricates text, removes provenance, or crosses pages. The canonical source references and bounding boxes remain attached to their original blocks.

This associates the observed dates with the headings on the same visual row: education dates with their institutions, the 2024-present range with the first employer, the 2019-2024 range with the Jingdong heading, and project dates with their project headings.

### 2. Section-coherent Parent chunks

Change `ParentBuilder` so a transition among regular prose types (`paragraph`, `list_item`, `formula`, and `other`) does not automatically flush an undersized Parent. Adjacent regular pieces under the same heading are coalesced until the normal target limits require a split. A coalesced Parent has `content_type="mixed"` and retains every source span in order.

Tables, code, and logs remain structural atoms and preserve the existing fail-closed row fallback. Headings still form hard section boundaries. Parent and Child token limits, source-resolution invariants, deterministic IDs, and page provenance remain unchanged.

The result must keep `业绩：获` with its continuation and keep a job's aligned date, duties, and accomplishments in the same reasoning unit whenever the token budget allows.

### 3. Preserve heading provenance through evidence packing

Carry `heading_path` through the complete query boundary:

- persisted `ParentChunk` hydration;
- retrieval `ParentEvidence`;
- prompt-safe `EvidenceItem`;
- `EvidenceManifestEntry`;
- authorization and manifest consistency checks.

`DataEnvelope` renders `heading_path` as a separate untrusted-data field next to raw content. Heading text is not prepended to source content and is never treated as an instruction. Citation authorization checks the server-hydrated heading against the manifest so a model cannot invent or replace it.

No Alembic migration is needed because `parent_chunks.heading_path` already exists and Elasticsearch Child documents already contain the field.

### 4. Deterministic evidence coverage and document limits

Replace literal full-question substring coverage with server-owned retrieval provenance. A retrieval batch carries the target IDs for which it was executed; parents selected from that batch cover those targets unless later cropping removes all usable content.

Retain the three-parent-per-document diversity cap when multiple documents are represented. If all valid candidates belong to one document, allow up to the retrieval parent limit of six, still bounded by `max_evidence_tokens`. This prevents a profile header from consuming one third of the only document's evidence budget while preserving cross-document diversity.

The change does not add query-specific company names, resume heuristics, or model-based selection rules.

### 5. One canonical Research evidence working set

The Research loop owns a `PackedEvidence` working set, not an independent evidence list plus a stale `packed_context`.

- At loop entry, validate the checkpointed `packed_context` against the current runtime snapshot.
- Direct retrieval returns both its raw `EvidenceBatch` and its packed evidence.
- Subagent retrieval returns both values as well; `SubagentResult` no longer discards the batch.
- Merge current and new packs deterministically by evidence ID, verify manifest equality and index generation, and enforce the snapshot evidence-token limit.
- Before every supervisor action, stage the merged `packed_context`, evidence list, Todos, and observations into `ContextBuilder`.
- On submission, return the merged packed context and every raw retrieval batch to the graph.

The post-Research graph remains the final trust boundary: it rebuilds evidence from the server-derived batches, grades it, and only then generates an answer. A delegate-only successful path therefore reaches the Evidence Builder instead of failing with `research_batches_missing`.

### 6. Structured-action contract and bounded model budget

`complete_structured` must expose the exact Pydantic JSON Schema to the model while retaining generic JSON-object provider mode for compatibility. The schema is server-owned and added as a system-level contract; manual prompt examples may clarify semantics but are not the source of truth.

The Research action prompt must also state the allowed transition for the current state: available Todo IDs, known evidence IDs, whether unresolved Todos remain, and when `submit_evidence` is valid.

Apply one total deadline to the complete structured operation, including transient retries and one schema repair. The repair request receives at most one attempt and uses only the remaining deadline. A repair must not start when insufficient budget remains. The underlying HTTP client timeout must be greater than the per-operation deadline so the gateway, not the SDK client, owns timeout classification.

Telemetry distinguishes:

- initial request versus schema repair;
- provider timeout versus gateway total-deadline exhaustion;
- supervisor LLM, delegated retrieval, and tool execution spans.

The graph must not wrap the entire Research loop in misleading `retrieval.tool` spans.

## Failure Handling

- Layout association fails open to original ordering only for ambiguous geometry; all provenance validation continues to fail closed.
- Evidence with a heading or manifest mismatch is rejected before prompt construction.
- Mixed index generations, missing delegated batches, or an over-budget evidence merge terminate Research with a safe typed reason and no answer.
- Model timeout or invalid schema returns `model_unavailable` or `research_action_invalid` without exposing provider text.
- Rebuild preflight aborts before deletion if source staging, backup verification, service quiescence, or exact index resolution fails.
- Rebuild validation failure keeps Query traffic stopped until the backup is restored or the defect is corrected and ingestion is rerun.

## Testing Strategy

### Unit tests

- Canonical assembler associates right-column date ranges with the uniquely aligned heading and leaves ambiguous layouts unchanged.
- ParentBuilder coalesces paragraph/list transitions into one mixed Parent without crossing headings or structural atoms; every AST locator still resolves exactly.
- Parent hydration and Evidence Builder preserve heading paths, render them as untrusted data, and reject heading-manifest mismatches.
- Evidence coverage follows retrieval target provenance; a single-document result may pack six parents while multi-document diversity remains capped.
- Research action two receives delegated evidence text and manifest IDs from action one.
- Delegate results propagate raw batches, and a delegate-submit sequence reaches post-Research evidence grading.
- Structured-call retry tests prove that initial retries plus repair cannot exceed the total deadline and that repair attempt counters are labeled separately.

### Integration tests

- MySQL Parent hydration round-trips `heading_path` from the existing column.
- The ingestion pipeline fixture produces coherent job-history Parents and index-v3 Child documents with matching headings and provenance.
- The real graph completes a delegate-only Research path without `research_batches_missing`.
- Elasticsearch alias and physical-index checks reject an unexpected generation.

### Rebuild acceptance

- The active alias points only to `agenticrag-children-index-v3`.
- Every active document version has `pipeline_version=ingestion-v2` and `index_generation=index-v3`.
- Manifest parent/child counts match MySQL and Elasticsearch counts.
- Only one active document exists per staged `(user_id, content_hash)`.
- The rebuilt resume places each visually aligned date range in the corresponding section and contains no split `业绩：获` continuation.
- `刘泽明工作经历` completes on FastRAG with an audited answer citing employer/title/date evidence and emits no Research-loop event.
- `刘泽明教育经历` remains audited and correct.
- A scripted Research regression demonstrates that delegated evidence is visible on the next action and can be submitted successfully.
- No raw personal fields appear in logs or degradation-event artifacts.

## Implementation Order

1. Add failing layout and mixed-Parent regression fixtures.
2. Implement canonical layout association and regular-piece coalescing.
3. Add failing heading-provenance and evidence-selection tests, then implement the query-model changes.
4. Add failing Research state-flow tests, then implement batch and packed-context propagation.
5. Add failing schema-budget and observability tests, then implement the model-gateway and span changes.
6. Add the guarded local rebuild command and operational documentation.
7. Run unit, integration, security, and real-service acceptance suites.
8. Execute the approved backup-delete-reingest runbook against the local environment.

## File Responsibility Map

- `src/agentic_rag/ingestion/assembler.py`: deterministic layout association.
- `src/agentic_rag/ingestion/chunker.py`: section-coherent mixed Parent construction.
- `src/agentic_rag/persistence/repositories.py`: hydrate existing heading metadata.
- `src/agentic_rag/retrieval/models.py` and `src/agentic_rag/retrieval/parents.py`: preserve heading and target provenance.
- `src/agentic_rag/query/evidence_builder.py` and `src/agentic_rag/safety/context.py`: pack, render, and authorize heading-aware evidence.
- `src/agentic_rag/query/subagents.py`, `tools.py`, `research_loop.py`, `context.py`, and `graph.py`: propagate one canonical Research evidence working set and raw batches.
- `src/agentic_rag/runtime/model_gateway.py` and `query_composition.py`: schema contract, retry budget, and timeout ownership.
- `src/agentic_rag/models/indexing.py` and configuration documentation: index-v3 and ingestion-v2 rollout contract.
- `scripts/`: guarded local rebuild orchestration using existing backup/restore boundaries.
- `tests/`: unit, integration, graph, and real-service regressions described above.

## Approval Boundary

This design authorizes creation of an implementation plan only. Deleting Elasticsearch data, deactivating documents, re-uploading source files, or changing production code requires a separately approved execution step after the implementation has passed its tests and the rebuild preflight identifies the exact targets.
