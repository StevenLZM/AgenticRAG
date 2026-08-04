# Agentic RAG Phase 3 Retrieval Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Deliver a deterministic, tenant-scoped hybrid Parent-Child retrieval pipeline and capacity-safe Evidence context for both Fast RAG and Agentic research.

**Architecture:** A service-only Filter Builder injects user and active-version scope, then ES Dense and BM25 lanes run concurrently. RRF, local Cross-Encoder, Parent aggregation and MySQL fetch form a fixed LangGraph Subgraph; EvidenceBuilder separately packs only validated Parent evidence for generation.

**Tech Stack:** LangGraph, Elasticsearch async client, Qwen embeddings, `BAAI/bge-reranker-v2-m3`, MySQL Parent repository, Pydantic, pytest.

## Global Constraints

- Prerequisites: Phase 1 and Phase 2 gates pass with at least one active version fixture.
- Metadata is a mandatory filter on Dense and BM25, never an independent recall lane or Agent-visible Tool.
- Agent input cannot supply or override `user_id`, `is_active`, `document_version_id` validity or ES Index Generation.
- Default pipeline: Dense 40 + BM25 40 → RRF 30 → Cross-Encoder 30 → Child 10 → Parent 6, at most two selected Children per Parent.
- Dense-only, BM25-only and RRF-only degraded paths are explicit results; both recall lanes failing is a closed failure.

---

### Task 1: Retrieval Contracts and Server-Side Filter Builder

**Files:**
- Create: `src/agentic_rag/retrieval/__init__.py`
- Create: `src/agentic_rag/retrieval/adapters/__init__.py`
- Create: `src/agentic_rag/retrieval/models.py`
- Create: `src/agentic_rag/retrieval/ports.py`
- Create: `src/agentic_rag/retrieval/filters.py`
- Create: `tests/unit/retrieval/test_filters.py`

**Interfaces:**
- Produces: `RetrievalRequest`, `SearchFilter`, `ChildHit`, `ParentEvidence`, `EvidenceBatch`, `FilterBuilder.build(request, scope, snapshot)`.
- Consumes: `UserScope`, `RuntimeConfigSnapshot`.

- [ ] **Step 1: Write tests proving Agent filters cannot override tenant scope**

```python
def test_filter_builder_injects_user_active_and_generation():
    request = RetrievalRequest(query="合同期限", search_type="policy")
    result = FilterBuilder().build(request, UserScope(user_id="u1"), SNAPSHOT)
    assert result.user_id == "u1"
    assert result.is_active is True
    assert result.index_generation == SNAPSHOT.index_generation

def test_request_schema_rejects_user_id():
    with pytest.raises(ValidationError):
        RetrievalRequest.model_validate({"query": "x", "user_id": "u2"})
```

- [ ] **Step 2: Verify red state**

Run: `pytest tests/unit/retrieval/test_filters.py -q`
Expected: FAIL because retrieval models are absent.

- [ ] **Step 3: Implement strict request and filter models**

```python
class RetrievalRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    query: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
    search_type: str | None = None
    document_ids: tuple[str, ...] = ()
    content_types: tuple[str, ...] = ()
    date_range: DateRange | None = None
    top_k_override: int | None = Field(default=None, ge=1, le=80)

class DateRange(BaseModel):
    start: date | None = None
    end: date | None = None

    @model_validator(mode="after")
    def ordered(self) -> "DateRange":
        if self.start and self.end and self.start > self.end:
            raise ValueError("date range start must not exceed end")
        return self

class SearchFilter(BaseModel):
    user_id: str
    is_active: Literal[True] = True
    index_generation: str
    search_type: str | None = None
    document_ids: tuple[str, ...] = ()
    content_types: tuple[str, ...] = ()
    date_range: DateRange | None = None

class ChildHit(BaseModel):
    child_id: str
    parent_id: str
    user_id: str
    document_id: str
    document_version_id: str
    content: str
    ast_locator: str
    lane: Literal["dense", "bm25"]
    lane_rank: int
    score: float

class ParentEvidence(BaseModel):
    parent_id: str
    document_id: str
    document_version_id: str
    content: str
    child_hits: tuple[ChildHit, ...]
    rerank_score: float

class EvidenceBatch(BaseModel):
    query: str
    parents: tuple[ParentEvidence, ...]
    degraded_components: tuple[str, ...] = ()
```

- [ ] **Step 4: Run contract tests**

Run: `pytest tests/unit/retrieval/test_filters.py -q && mypy src/agentic_rag/retrieval`
Expected: PASS.

- [ ] **Step 5: Commit retrieval contracts**

```bash
git add src/agentic_rag/retrieval tests/unit/retrieval
git commit -m "feat: define scoped retrieval contracts"
```

### Task 2: ES Dense and BM25 Adapters

**Files:**
- Create: `src/agentic_rag/retrieval/adapters/elasticsearch.py`
- Create: `tests/integration/retrieval/test_elasticsearch_search.py`

**Interfaces:**
- Produces: `VectorIndex.search(query_vector, filter, top_k)`, `LexicalIndex.search(query_text, filter, top_k)`.
- Consumes: active Child ES Alias and `SearchFilter` from Task 1.

- [ ] **Step 1: Write integration tests for both lanes and zero leakage**

```python
@pytest.mark.integration
async def test_dense_and_bm25_apply_identical_user_filter(index_fixture):
    dense = await index_fixture.vector.search(index_fixture.query_vector, index_fixture.filter("u1"), 40)
    lexical = await index_fixture.lexical.search("termination clause", index_fixture.filter("u1"), 40)
    assert dense and lexical
    assert {hit.user_id for hit in dense + lexical} == {"u1"}
```

- [ ] **Step 2: Verify red state**

Run: `pytest -m integration tests/integration/retrieval/test_elasticsearch_search.py -q`
Expected: FAIL because ES adapters are absent.

- [ ] **Step 3: Implement independent adapters with one shared filter serializer**

```python
class VectorIndex(Protocol):
    async def search(self, query_vector: Sequence[float], filter: SearchFilter, top_k: int) -> list[ChildHit]: ...

class LexicalIndex(Protocol):
    async def search(self, query_text: str, filter: SearchFilter, top_k: int) -> list[ChildHit]: ...
```

Dense uses the ES vector field and BM25 uses the Child contextualized content. Both call the same `serialize_filter()` and request only fields needed for fusion. Validate query vectors have exactly 1024 dimensions and reject Index Generation mismatch before sending the request.

- [ ] **Step 4: Run local ES tests**

Run: `pytest -m integration tests/integration/retrieval/test_elasticsearch_search.py -q`
Expected: both lanes return active records only and never return the other user fixture.

- [ ] **Step 5: Commit ES retrieval**

```bash
git add src/agentic_rag/retrieval/adapters tests/integration/retrieval
git commit -m "feat: add elastic hybrid recall adapters"
```

### Task 3: RRF Fusion and Cross-Encoder Reranking

**Files:**
- Create: `src/agentic_rag/retrieval/fusion.py`
- Create: `src/agentic_rag/retrieval/reranker.py`
- Create: `tests/unit/retrieval/test_fusion.py`
- Create: `tests/unit/retrieval/test_reranker.py`

**Interfaces:**
- Produces: `rrf_fuse(lanes, k=60, limit=30)`, `Reranker.rerank(query, candidates, limit=10)`.
- Consumes: `ChildHit` lists; local model instance supplied at bootstrap.

- [ ] **Step 1: Write exact-ranking and degradation tests**

```python
def test_rrf_deduplicates_and_uses_stable_tie_break():
    fused = rrf_fuse([[hit("a"), hit("b")], [hit("b"), hit("c")]], k=60, limit=30)
    assert [item.child_id for item in fused] == ["b", "a", "c"]

async def test_reranker_timeout_returns_rrf_order(service, candidates):
    service.model.raise_timeout = True
    result = await service.rerank("q", candidates, limit=10)
    assert result.degraded is True
    assert result.hits == candidates[:10]
```

- [ ] **Step 2: Verify red state**

Run: `pytest tests/unit/retrieval/test_fusion.py tests/unit/retrieval/test_reranker.py -q`
Expected: FAIL because fusion and reranker are absent.

- [ ] **Step 3: Implement pure RRF and a single-loaded reranker service**

```python
def reciprocal_rank_score(rank: int, k: int = 60) -> float:
    return 1.0 / (k + rank)
```

Sum scores for duplicate Child IDs and break ties by best lane rank, then `child_id`. Reranker accepts no more than 30 candidates, runs model inference through a bounded executor/Semaphore, enforces retrieval timeout, and returns a structured `RerankResult(hits, degraded, model_version)`.

- [ ] **Step 4: Run deterministic ranking tests**

Run: `pytest tests/unit/retrieval/test_fusion.py tests/unit/retrieval/test_reranker.py -q`
Expected: exact order and timeout degradation pass.

- [ ] **Step 5: Commit fusion and reranking**

```bash
git add src/agentic_rag/retrieval/fusion.py src/agentic_rag/retrieval/reranker.py tests/unit/retrieval
git commit -m "feat: add rrf and cross encoder reranking"
```

### Task 4: Parent Aggregation and Fetch

**Files:**
- Create: `src/agentic_rag/retrieval/parents.py`
- Create: `tests/unit/retrieval/test_parent_aggregation.py`
- Create: `tests/integration/retrieval/test_parent_fetch.py`

**Interfaces:**
- Produces: `aggregate_parents(hits, max_children_per_parent=2, limit=6)`, `ParentFetcher.fetch(ids, scope) -> list[ParentEvidence]`.
- Consumes: reranked Child hits and `ParentRepository`.

- [ ] **Step 1: Write aggregation and ownership tests**

```python
def test_parent_aggregation_caps_children_and_keeps_best_parent_order():
    parents = aggregate_parents(RERANKED_HITS, max_children_per_parent=2, limit=6)
    assert all(len(parent.child_hits) <= 2 for parent in parents)
    assert len(parents) <= 6

@pytest.mark.integration
async def test_parent_fetch_fails_closed_on_scope_mismatch(parent_fetcher):
    with pytest.raises(ParentScopeViolation):
        await parent_fetcher.fetch(["other-user-parent"], UserScope(user_id="u1"))
```

- [ ] **Step 2: Verify red state**

Run: `pytest tests/unit/retrieval/test_parent_aggregation.py -q`
Expected: FAIL because aggregation is absent.

- [ ] **Step 3: Implement stable aggregation and batched scoped fetch**

Aggregate by Parent ID using the highest rerank score and best Child ordinal; fetch all Parent rows in one scoped MySQL query requiring `user_id`, active Parent status and active document version. Preserve requested order and fail the whole EvidenceBatch if any returned Parent violates scope/version.

```python
class ParentFetcher:
    async def fetch(self, parent_ids: Sequence[str], scope: UserScope) -> list[ParentEvidence]:
        rows = await self.parents.fetch_active(parent_ids=parent_ids, user_id=scope.user_id)
        if {row.id for row in rows} != set(parent_ids):
            raise ParentScopeViolation("one or more parents are missing, inactive, or outside scope")
        by_id = {row.id: row for row in rows}
        return [to_parent_evidence(by_id[parent_id]) for parent_id in parent_ids]
```

- [ ] **Step 4: Run unit and MySQL tests**

Run: `pytest tests/unit/retrieval/test_parent_aggregation.py -q && pytest -m integration tests/integration/retrieval/test_parent_fetch.py -q`
Expected: PASS with stable ordering and fail-closed ownership.

- [ ] **Step 5: Commit Parent retrieval**

```bash
git add src/agentic_rag/retrieval/parents.py tests
git commit -m "feat: aggregate and fetch parent evidence"
```

### Task 5: RetrievalPipelineGraph and Explicit Degradation

**Files:**
- Create: `src/agentic_rag/retrieval/state.py`
- Create: `src/agentic_rag/retrieval/graph.py`
- Create: `tests/unit/retrieval/test_graph.py`
- Create: `tests/integration/retrieval/test_pipeline.py`

**Interfaces:**
- Produces: `build_retrieval_graph(deps) -> CompiledStateGraph`, `RetrievalService.retrieve(request, scope, snapshot) -> EvidenceBatch`.
- Consumes: Tasks 1–4 adapters and embedding query port.

- [ ] **Step 1: Write graph lane-failure tests**

```python
async def test_graph_degrades_to_bm25_when_dense_fails(graph, deps):
    deps.vector.fail = True
    result = await graph.ainvoke(REQUEST_STATE)
    assert result["evidence_batch"].degraded_components == ("dense",)

async def test_graph_fails_closed_when_both_lanes_fail(graph, deps):
    deps.vector.fail = deps.lexical.fail = True
    with pytest.raises(RetrievalUnavailable):
        await graph.ainvoke(REQUEST_STATE)
```

- [ ] **Step 2: Verify red state**

Run: `pytest tests/unit/retrieval/test_graph.py -q`
Expected: FAIL because the graph is absent.

- [ ] **Step 3: Implement the fixed subgraph**

```text
validate_and_filter
  -> [embed_and_dense, bm25]
  -> rrf_fusion
  -> cross_encoder
  -> parent_aggregation
  -> parent_fetch
  -> evidence_batch
```

Dense and BM25 use `asyncio.gather(return_exceptions=True)` with lane-specific timeouts. The graph emits structured timings, candidate counts and degradation metadata, but exposes only the high-level `retrieve_evidence` service to Agents.

- [ ] **Step 4: Run graph and integration tests**

Run: `pytest tests/unit/retrieval/test_graph.py -q && pytest -m integration tests/integration/retrieval/test_pipeline.py -q`
Expected: normal and single-lane paths return Parent evidence; dual failure raises `RetrievalUnavailable`.

- [ ] **Step 5: Commit retrieval graph**

```bash
git add src/agentic_rag/retrieval/state.py src/agentic_rag/retrieval/graph.py tests
git commit -m "feat: add deterministic retrieval graph"
```

### Task 6: EvidenceBuilder and Packed Context

**Files:**
- Create: `src/agentic_rag/query/__init__.py`
- Create: `src/agentic_rag/query/evidence_builder.py`
- Create: `src/agentic_rag/safety/context.py`
- Create: `tests/unit/query/test_evidence_builder.py`

**Interfaces:**
- Produces: `EvidenceBuilder.build(batches, coverage_targets, scope, snapshot) -> PackedEvidence`, `DataEnvelope.render()`.
- Consumes: one or more `EvidenceBatch` values from Task 5.

- [ ] **Step 1: Write capacity, diversity and citation-manifest tests**

```python
def test_builder_never_exceeds_capacity_and_only_manifests_included_evidence(builder):
    packed = builder.build(BATCHES, COVERAGE_TARGETS, UserScope(user_id="u1"), SNAPSHOT)
    assert packed.token_count <= 12_000
    assert set(packed.manifest) == {item.evidence_id for item in packed.items}
    assert max(Counter(item.document_id for item in packed.items).values()) <= 3
```

- [ ] **Step 2: Verify red state**

Run: `pytest tests/unit/query/test_evidence_builder.py -q`
Expected: FAIL because EvidenceBuilder is absent.

- [ ] **Step 3: Implement deterministic packing and untrusted-data envelopes**

```python
class PackedEvidence(BaseModel):
    items: tuple[EvidenceItem, ...]
    manifest: Mapping[str, EvidenceManifestEntry]
    rendered_context: str
    token_count: int

class EvidenceCoverageTarget(BaseModel):
    target_id: str
    description: str

class EvidenceItem(BaseModel):
    evidence_id: str
    parent_id: str
    document_id: str
    document_version_id: str
    content: str
    ast_locator: str
    covered_target_ids: tuple[str, ...]

class EvidenceManifestEntry(BaseModel):
    evidence_id: str
    parent_id: str
    document_id: str
    document_version_id: str
    ast_locator: str

class DataEnvelope(BaseModel):
    source_label: str
    evidence_id: str
    content: str

    def render(self) -> str:
        return json.dumps({
            "trust": "untrusted_data",
            "source": self.source_label,
            "evidence_id": self.evidence_id,
            "content": self.content,
        }, ensure_ascii=False)

class EvidenceBuilder:
    def build(self, batches: Sequence[EvidenceBatch], coverage_targets: Sequence[EvidenceCoverageTarget], scope: UserScope,
              snapshot: RuntimeConfigSnapshot, max_tokens: int = 12_000) -> PackedEvidence: ...
```

Deduplicate by Parent/version, cover distinct Todos before adding repeated evidence, cap general searches at three Parents per document, preserve direct single-document queries, and structurally crop long Parents around matched Child locators. Render each item inside a source-labeled Data Envelope that explicitly states content is untrusted data.

- [ ] **Step 4: Run Phase 3 gate**

Run: `pytest tests/unit/retrieval tests/unit/query/test_evidence_builder.py -q && pytest -m integration tests/integration/retrieval -q`
Expected: all deterministic, degradation, capacity and zero-leakage tests pass.

- [ ] **Step 5: Commit EvidenceBuilder**

```bash
git add src/agentic_rag/query src/agentic_rag/safety/context.py tests/unit/query
git commit -m "feat: build bounded evidence context"
```
