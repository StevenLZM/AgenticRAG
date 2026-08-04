# Agentic RAG Phase 2 Document Ingestion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ingest PDF, scanned PDF, text and Excel into a validated Canonical AST, deterministic Parent/Child chunks and an atomically published searchable document version.

**Architecture:** The document API persists file, version, job and Outbox records first. A single Ingestion Worker executes a checkpointed LangGraph pipeline through upload safety, Docling parsing, global assembly, chunking, embedding, staging and publication; an in-process Reconciler repairs notification and cross-store inconsistencies.

**Tech Stack:** FastAPI multipart upload, Docling, LangGraph, Qwen embeddings through an adapter, MySQL repositories, Redis Streams, Elasticsearch 8, local Artifact Store, pytest.

## Global Constraints

- Prerequisite: Phase 1 gate passes and its repository/broker/artifact interfaces remain unchanged.
- All source-derived content is untrusted data; document text cannot select tools, filters, users or pipeline code paths.
- The API transaction creates `documents`, `document_versions`, `ingestion_jobs` and `task_outbox`; Redis is never the Job state authority.
- Fragment and Canonical AST use `DoclingDocument` plus the approved thin Envelope.
- Parent uses structure-semantic chunking; Child uses Docling HybridChunker with 384 Token maximum and no default overlap.
- New versions remain invisible until Parent, Child, Artifact and Manifest validation passes.

---

### Task 1: Document Upload, Safety Gate and Durable Job Creation

**Files:**
- Create: `src/agentic_rag/ingestion/__init__.py`
- Create: `src/agentic_rag/safety/__init__.py`
- Create: `src/agentic_rag/api/documents.py`
- Create: `src/agentic_rag/ingestion/models.py`
- Create: `src/agentic_rag/safety/uploads.py`
- Create: `tests/unit/safety/test_uploads.py`
- Create: `tests/integration/api/test_documents.py`

**Interfaces:**
- Produces: `UploadSafetyScanner.scan() -> UploadDecision`, `DocumentService.create_upload() -> IngestionJob`, `POST /v1/documents`, `GET /v1/ingestion-jobs/{job_id}`, `DELETE /v1/documents/{document_id}`.
- Consumes: `DocumentRepository`, `IngestionJobRepository`, `OutboxRepository`, `ArtifactStore`, `UserScope`.

- [ ] **Step 1: Write failing upload and atomic-job tests**

```python
def test_mime_mismatch_is_rejected(scanner):
    decision = scanner.scan(filename="report.pdf", declared_mime="application/pdf", content=b"PK\x03\x04")
    assert decision.status == UploadSafetyStatus.REJECTED

@pytest.mark.integration
async def test_upload_creates_job_and_outbox_in_one_transaction(client, mysql):
    response = await client.post("/v1/documents", files={"file": ("notes.txt", b"hello", "text/plain")})
    assert response.status_code == 202
    assert await mysql.has_matching_job_and_outbox(response.json()["job_id"])
```

- [ ] **Step 2: Verify red state**

Run: `pytest tests/unit/safety/test_uploads.py tests/integration/api/test_documents.py -q`
Expected: FAIL because upload contracts and routes are absent.

- [ ] **Step 3: Implement the scanner and upload transaction**

```python
class UploadSafetyStatus(StrEnum):
    ACCEPTED = "accepted"
    QUARANTINED = "quarantined"
    REJECTED = "rejected"

class UploadDecision(BaseModel):
    status: UploadSafetyStatus
    detected_mime: str
    content_hash: str
    reasons: tuple[str, ...] = ()

class UploadSafetyScanner(Protocol):
    def scan(self, filename: str, declared_mime: str, content: bytes) -> UploadDecision: ...
```

Allow only PDF, plain text and Excel MIME/signature pairs. Normalize the basename, reject traversal and unsupported containers, and scan parsed text for invisible Unicode and instruction-like content; heuristics quarantine rather than silently remove text. Store the original Artifact before creating the database transaction, and remove the unreferenced Artifact if the transaction fails.

- [ ] **Step 4: Run safety and API tests**

Run: `pytest tests/unit/safety/test_uploads.py -q && pytest -m integration tests/integration/api/test_documents.py -q`
Expected: accepted upload returns 202; mismatch rejects; suspicious content quarantines and creates no active index version.

- [ ] **Step 5: Commit upload flow**

```bash
git add src/agentic_rag/api/documents.py src/agentic_rag/ingestion/models.py src/agentic_rag/safety/uploads.py tests
git commit -m "feat: add durable document uploads"
```

### Task 2: Docling Fragment Parser and Canonical AST Assembler

**Files:**
- Create: `src/agentic_rag/ingestion/parser.py`
- Create: `src/agentic_rag/ingestion/assembler.py`
- Create: `src/agentic_rag/safety/content.py`
- Create: `tests/fixtures/documents/`
- Create: `tests/unit/ingestion/test_assembler.py`
- Create: `tests/integration/ingestion/test_docling_parser.py`

**Interfaces:**
- Produces: `DocumentParser.parse_batches() -> AsyncIterator[FragmentAst]`, `GlobalAssembler.assemble() -> CanonicalAst`, `ContentSafetyScanner.scan(CanonicalAst) -> UploadDecision`.
- Consumes: original `ArtifactRef`, `DocumentEnvelope`, Docling converters.

- [ ] **Step 1: Write fixture-based cross-page merge and dedup tests**

```python
def test_assembler_merges_cross_page_paragraph_and_removes_repeated_footer():
    canonical = GlobalAssembler().assemble(fragment_fixture("cross_page_paragraph.json"))
    assert canonical.text_blocks[0].text == "完整的跨页段落"
    assert all("CONFIDENTIAL" not in block.text for block in canonical.text_blocks)
    assert canonical.text_blocks[0].provenance.page_from == 1
    assert canonical.text_blocks[0].provenance.page_to == 2
```

- [ ] **Step 2: Verify red state**

Run: `pytest tests/unit/ingestion/test_assembler.py -q`
Expected: FAIL because Fragment/Canonical AST types and assembler do not exist.

- [ ] **Step 3: Implement thin Docling envelopes and deterministic assembly passes**

```python
class DocumentEnvelope(BaseModel):
    document_id: str
    document_version_id: str
    user_id: str
    source_type: Literal["pdf", "scanned_pdf", "excel", "text"]
    source_uri: str
    content_hash: str
    parser_version: str
    pipeline_version: str
    created_at: datetime

class FragmentAst(BaseModel):
    envelope: DocumentEnvelope
    batch_no: int
    page_from: int
    page_to: int
    docling_document: dict[str, Any]
```

Assembly order is fixed: normalize reading order → join cross-page paragraphs/lists/tables → remove repeated headers/footers and OCR duplicates by normalized Hash plus location rule → repair heading hierarchy → validate provenance and references. After assembly, scan extracted/OCR text for hidden Unicode and instruction-like content; quarantine the version before Chunking when the policy rejects it. Save every Fragment and the final Canonical AST as versioned JSON Artifacts.

- [ ] **Step 4: Run unit and Docling integration tests**

Run: `pytest tests/unit/ingestion/test_assembler.py -q && pytest -m integration tests/integration/ingestion/test_docling_parser.py -q`
Expected: all four document types produce valid envelopes and traceable page/region provenance.

- [ ] **Step 5: Commit parser and assembler**

```bash
git add src/agentic_rag/ingestion/parser.py src/agentic_rag/ingestion/assembler.py src/agentic_rag/safety/content.py tests
git commit -m "feat: assemble canonical document ast"
```

### Task 3: Parent-Child Chunking

**Files:**
- Create: `src/agentic_rag/ingestion/chunker.py`
- Create: `tests/unit/ingestion/test_chunker.py`

**Interfaces:**
- Produces: `ParentBuilder.build(CanonicalAst) -> list[ParentChunk]`, `ChildBuilder.build(ParentChunk) -> list[ChildChunk]`.
- Consumes: Canonical AST from Task 2 and tokenizer resolved from embedding settings.

- [ ] **Step 1: Write boundary, determinism and provenance tests**

```python
def test_parent_child_ids_are_deterministic(canonical_ast):
    first = ChunkingPipeline(TOKENIZER).build(canonical_ast)
    second = ChunkingPipeline(TOKENIZER).build(canonical_ast)
    assert [(p.id, [c.id for c in p.children]) for p in first] == [(p.id, [c.id for c in p.children]) for p in second]
    assert max(c.token_count for p in first for c in p.children) <= 384
```

- [ ] **Step 2: Verify red state**

Run: `pytest tests/unit/ingestion/test_chunker.py -q`
Expected: FAIL because chunking classes are absent.

- [ ] **Step 3: Implement approved hybrid chunking**

```python
class ParentChunk(BaseModel):
    id: str
    document_version_id: str
    ordinal: int
    heading_path: tuple[str, ...]
    content_type: str
    content: str
    page_from: int
    page_to: int
    ast_locator: str
    content_hash: str

class ChildChunk(BaseModel):
    id: str
    parent_id: str
    ordinal: int
    contextualized_content: str
    token_count: int
    ast_locator: str
```

Parent targets 1200–1800 Tokens and never exceeds 2400 unless an indivisible table/code/log atom requires the documented row fallback. Child uses Docling HybridChunker at 384 Tokens, includes heading context, has no default overlap, and inherits Parent/user/version metadata server-side.

- [ ] **Step 4: Run chunking tests**

Run: `pytest tests/unit/ingestion/test_chunker.py -q`
Expected: deterministic IDs, complete provenance, structure boundaries and Token limits pass.

- [ ] **Step 5: Commit chunking**

```bash
git add src/agentic_rag/ingestion/chunker.py tests/unit/ingestion/test_chunker.py
git commit -m "feat: add parent child chunking"
```

### Task 4: Embedding, Staging Index and Manifest

**Files:**
- Create: `src/agentic_rag/models/__init__.py`
- Create: `src/agentic_rag/models/embeddings.py`
- Create: `src/agentic_rag/ingestion/indexer.py`
- Create: `src/agentic_rag/ingestion/manifest.py`
- Create: `tests/unit/ingestion/test_manifest.py`
- Create: `tests/integration/ingestion/test_staging_index.py`

**Interfaces:**
- Produces: `EmbeddingPort.embed_documents()`, `IndexWriter.stage() -> VersionManifest`.
- Consumes: Parent/Child chunks and MySQL/ES adapters.

- [ ] **Step 1: Write dimension and manifest-count tests**

```python
async def test_indexer_rejects_wrong_embedding_dimension(fake_embedding, indexer, chunks):
    fake_embedding.dimensions = 1536
    with pytest.raises(EmbeddingDimensionError):
        await indexer.stage(chunks)

def test_manifest_hash_is_deterministic(manifest_factory):
    assert manifest_factory().manifest_hash == manifest_factory().manifest_hash
```

- [ ] **Step 2: Verify red state**

Run: `pytest tests/unit/ingestion/test_manifest.py -q`
Expected: FAIL because embedding and manifest contracts are absent.

- [ ] **Step 3: Implement batch embedding and invisible staging writes**

```python
class EmbeddingPort(Protocol):
    async def embed_documents(self, texts: Sequence[str]) -> list[list[float]]: ...
    async def embed_query(self, text: str) -> list[float]: ...

class VersionManifest(BaseModel):
    canonical_ast_sha256: str
    parent_count: int
    child_count: int
    embedding_model: Literal["text-embedding-v3"]
    embedding_dimensions: Literal[1024]
    index_generation: str
    manifest_hash: str
```

Stage Parent rows as inactive in MySQL and Child records as `is_active=false` in the versioned ES Index. Validate every vector length before bulk indexing. Persist the Manifest Artifact and its Hash on `document_versions` only after both stores report the expected counts.

- [ ] **Step 4: Run manifest and local ES/MySQL tests**

Run: `pytest tests/unit/ingestion/test_manifest.py -q && pytest -m integration tests/integration/ingestion/test_staging_index.py -q`
Expected: incorrect dimensions fail closed; valid staging remains invisible to active filters.

- [ ] **Step 5: Commit staging index**

```bash
git add src/agentic_rag/models src/agentic_rag/ingestion/indexer.py src/agentic_rag/ingestion/manifest.py tests
git commit -m "feat: stage embedded document versions"
```

### Task 5: Publisher, Reconciler and Delete Reconciliation

**Files:**
- Create: `src/agentic_rag/ingestion/publisher.py`
- Create: `src/agentic_rag/ingestion/reconciler.py`
- Create: `tests/integration/ingestion/test_publisher_reconciler.py`

**Interfaces:**
- Produces: `VersionPublisher.publish(version_id)`, `IngestionReconciler.run_once() -> ReconcileReport`.
- Consumes: Manifest, document/version repositories, ES adapter, Artifact Store, Outbox repository.

- [ ] **Step 1: Write interrupted-publication recovery tests**

```python
@pytest.mark.integration
async def test_reconciler_finishes_interrupted_activation(system):
    await system.publisher.publish(system.version_id, fail_after="new_children_active")
    report = await system.reconciler.run_once()
    assert report.repaired_versions == [system.version_id]
    assert await system.only_highest_version_is_searchable()
```

- [ ] **Step 2: Verify red state**

Run: `pytest -m integration tests/integration/ingestion/test_publisher_reconciler.py -q`
Expected: FAIL because publication and reconciliation are absent.

- [ ] **Step 3: Implement manifest-gated publication and periodic repair**

Publication sequence is fixed: verify Artifact Hash/counts → activate new Parent → activate new Child → deactivate old Child → deactivate old Parent → update MySQL `active_version_id` and document/version status. Each write is idempotent by deterministic ID. Reconciler scans pending Outbox, expired Jobs, stale building versions, active-pointer mismatches and deleted documents, and returns explicit repaired/quarantined IDs.

```python
class ReconcileReport(BaseModel):
    redispatched_jobs: tuple[str, ...] = ()
    reclaimed_jobs: tuple[str, ...] = ()
    repaired_versions: tuple[str, ...] = ()
    quarantined_versions: tuple[str, ...] = ()
    reconciled_deletions: tuple[str, ...] = ()
```

- [ ] **Step 4: Run publication, rollback and deletion tests**

Run: `pytest -m integration tests/integration/ingestion/test_publisher_reconciler.py -q`
Expected: every injected failure converges to exactly one active version or keeps the old version active; no partial new version is returned by active retrieval filters.

- [ ] **Step 5: Commit lifecycle management**

```bash
git add src/agentic_rag/ingestion/publisher.py src/agentic_rag/ingestion/reconciler.py tests/integration/ingestion
git commit -m "feat: publish and reconcile document versions"
```

### Task 6: Checkpointed IngestionGraph and Worker Recovery

**Files:**
- Create: `src/agentic_rag/ingestion/state.py`
- Create: `src/agentic_rag/ingestion/graph.py`
- Create: `src/agentic_rag/ingestion/worker.py`
- Create: `scripts/run_ingestion_worker.py`
- Create: `scripts/review_quarantined_version.py`
- Create: `tests/e2e/test_ingestion_pipeline.py`

**Interfaces:**
- Produces: compiled `IngestionGraph`, `IngestionWorker.run_forever()`, trusted quarantine review CLI.
- Consumes: all Phase 2 services and Phase 1 Stream/Checkpoint ports.

- [ ] **Step 1: Write crash-resume and duplicate-delivery E2E tests**

```python
@pytest.mark.e2e
async def test_worker_resumes_after_index_stage_without_duplicate_chunks(local_system):
    job = await local_system.upload("tests/fixtures/documents/sample.pdf")
    await local_system.worker.run_one(job.id, fail_after="stage_index")
    await local_system.worker.run_one(job.id)
    assert await local_system.job_status(job.id) == "completed"
    assert await local_system.count_duplicate_parent_or_child_ids(job.document_id) == 0
```

- [ ] **Step 2: Verify red state**

Run: `pytest -m e2e tests/e2e/test_ingestion_pipeline.py -q`
Expected: FAIL because Graph and Worker are absent.

- [ ] **Step 3: Implement the fixed ingestion state machine**

```text
load_job -> upload_safety_gate -> parse_fragments -> assemble_canonical
         -> content_safety_gate -> validate_canonical -> chunk -> embed_and_stage
         -> publish -> finalize
```

Every node reads/writes JSON-serializable references, uses deterministic idempotency keys and checks Job cancellation/lease ownership before side effects. Worker consumes `agenticrag:jobs:ingestion`, Claims MySQL Job, heartbeats, invokes Graph with `thread_id=ingestion:{job_id}`, ACKs only after terminal status, reclaims expired pending messages, and dead-letters after three attempts. Run Reconciler on a fixed periodic loop inside the same process.

- [ ] **Step 4: Run the Phase 2 gate**

Run: `pytest tests/unit/ingestion tests/unit/safety -q && pytest -m integration tests/integration/ingestion -q && pytest -m e2e tests/e2e/test_ingestion_pipeline.py -q`
Expected: all four formats ingest; restart and duplicate delivery produce one active version without duplicate IDs.

- [ ] **Step 5: Commit the ingestion runtime**

```bash
git add src/agentic_rag/ingestion scripts tests/e2e/test_ingestion_pipeline.py
git commit -m "feat: add recoverable ingestion graph"
```
