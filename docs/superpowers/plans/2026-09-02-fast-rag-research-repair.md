# FastRAG Evidence and Research Escalation Repair Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Repair canonical document ordering, semantic Parent packing, heading-aware evidence provenance, Research evidence propagation, and structured-call deadline handling so the rebuilt FastRAG path answers work-history questions safely.

**Architecture:** Keep one Docling-baseline canonical ordering pipeline, add one deterministic local heading/date relationship pass, and feed its output to the existing unified section-aware ParentBuilder and HybridChunker ChildBuilder. Make `EvidenceBatch` the server-owned retrieval provenance carrier and make `PackedEvidence` the single Research working set; the graph still rebuilds evidence from raw batches before grading or generation. Keep provider JSON-object mode for compatibility while attaching the exact Pydantic schema as a server-owned system contract and enforcing one deadline over initial retries plus one repair.

**Tech Stack:** Python 3.11, Pydantic v2, Docling, HybridChunker, LangGraph, SQLAlchemy asyncio, Elasticsearch 8, OpenAI-compatible async clients, pytest/pytest-asyncio, Ruff, mypy.

**Spec:** `docs/superpowers/specs/2026-08-31-fast-rag-research-rebuild-repair-design.md`

## Global Constraints

- Every source type and page layout uses one canonical-ordering pipeline and one Parent/Child Chunker; no resume-specific or two-column Chunker.
- Docling order remains the baseline; never replace it with a global `y`/`x` bounding-box sort.
- Ambiguous or missing geometry preserves the exact original candidate IDs, text, provenance, and sequence; the layout pass is deterministic and idempotent.
- Headings are hard Parent boundaries; tables, code, and logs remain structural atoms with the existing fail-closed row fallback.
- Parent and Child token limits, source-resolution invariants, deterministic IDs, and page provenance remain unchanged.
- `heading_path` is rendered as a separate untrusted-data field and is checked against the server-owned manifest; it is never prepended as instructions.
- Retrieval target coverage is server-owned provenance, not literal full-question substring matching.
- Mixed index generations, missing delegated batches, malformed manifests, and over-budget merges fail closed with no answer.
- Structured model calls receive the exact Pydantic JSON Schema; one total deadline covers transient retries and at most one schema repair.
- The underlying provider client timeout is greater than the per-operation gateway deadline; gateway code owns timeout classification.
- No SQL schema migration is required.
- This implementation changes code, tests, and operational documentation only; it does not delete indices, deactivate documents, re-upload sources, or run the destructive rebuild.
- The working tree already contains unrelated uncommitted changes; never reset, checkout, or commit a file whose pre-existing changes have not been explicitly isolated.

---

### Task 1: Add conservative canonical heading/date association

**Files:**
- Modify: `src/agentic_rag/ingestion/assembler.py` in `GlobalAssembler.assemble` and the candidate-order helpers near `_normalize_reading_order`.
- Test: `tests/unit/ingestion/test_assembler.py`.

**Interfaces:**
- Consumes: the `_Candidate` list returned by `_normalize_reading_order`, including `Provenance.regions[*].bbox`, `kind`, `role`, `source_refs`, and `source_order`.
- Produces: `_associate_aligned_date_ranges(candidates: Sequence[_Candidate]) -> list[_Candidate]`, called after Docling graph order normalization and before cross-page joining.

- [ ] **Step 1: Write failing ordering and safety tests.** Add fixtures that construct a Docling body graph whose depth-first traversal visits a heading, nested duty/list groups, then a same-row right-hand date. Assert the current candidate order puts the date after the duty and the new expected order puts it immediately after the heading. Add cases for a single aligned pair, mismatched vertical overlap, two date candidates for one heading, table/form/picture-like blocks, cross-page blocks, ordinary right-aligned report dates, and a true multi-column article; each must retain the exact pre-pass candidate sequence. Add an idempotence assertion that applying the pass twice returns equal candidates.

```python
def test_aligned_date_is_relocated_only_for_a_consistent_metadata_band() -> None:
    candidates = _reproduced_body_tree_candidates()
    repaired = _associate_aligned_date_ranges(candidates)
    assert [item.text for item in repaired] == [
        "甲公司 / 高级工程师", "2022.01-2024.03", "负责平台建设",
        "乙公司 / 工程师", "2020.01-2021.12", "负责服务开发",
    ]
    assert _associate_aligned_date_ranges(repaired) == repaired
```

- [ ] **Step 2: Run the focused tests to verify the new assertions fail.**

Run: `conda run -n agentic-rag pytest --import-mode=importlib tests/unit/ingestion/test_assembler.py -q`

Expected: the new expected adjacency test fails because no relationship pass exists; all unrelated baseline tests remain the comparison signal.

- [ ] **Step 3: Implement the minimal deterministic relationship pass.** Add repository-owned constants for the date-range grammar, vertical-overlap threshold, and right-band tolerance. Extract a candidate's only same-page bounding box only when it has one region and one page. Match standalone `paragraph`/`other` blocks with `role is None` against the date grammar, require the date box to be to the right of a heading box, require normalized vertical overlap above the threshold, choose exactly one best heading by overlap/distance with ties rejected, and require at least two compatible pairs on the page whose date x positions form one band. Move each accepted date candidate immediately after its heading while preserving every other candidate's relative order. Do not mutate candidate IDs, text, provenance, source references, or geometry. Call the pass before `_join_cross_page_content`.

- [ ] **Step 4: Run the assembler unit tests and lint the file.**

Run: `conda run -n agentic-rag pytest --import-mode=importlib tests/unit/ingestion/test_assembler.py -q && conda run -n agentic-rag ruff check src/agentic_rag/ingestion/assembler.py tests/unit/ingestion/test_assembler.py`

Expected: all assembler tests pass, including no-op and idempotence cases; Ruff reports no errors.

---

### Task 2: Make regular Parent chunks section-coherent

**Files:**
- Modify: `src/agentic_rag/ingestion/chunker.py` in `ParentBuilder._append_regular_piece`.
- Test: `tests/unit/ingestion/test_chunker.py`.

**Interfaces:**
- Consumes: `_ParentPiece` values with `content_type` in `paragraph`, `list_item`, `formula`, or `other`.
- Produces: the existing `ParentChunk` contract, with `content_type="mixed"` when a coalesced Parent contains multiple regular types.

- [ ] **Step 1: Write failing mixed-boundary tests.** Build a canonical AST containing one heading, a paragraph, a list item beginning with `业绩：获`, and its continuation. Configure a small tokenizer window so the combined content fits one Parent and assert one Parent retains all source spans, has `content_type == "mixed"`, and keeps the heading path. Add paragraph/list/formula/other combinations across PDF-like and text-like envelopes, plus tests proving headings and structural atoms still flush.

```python
def test_parent_builder_coalesces_regular_type_transitions() -> None:
    parents = ParentBuilder(_Tokenizer(), target_min_tokens=1, target_max_tokens=200).build(
        _ast("工作经历", ("负责平台建设", "业绩：获", "得优秀交付奖"))
    )
    assert len(parents) == 1
    assert parents[0].content_type == "mixed"
    assert "业绩：获得优秀交付奖" in parents[0].content.replace("\n", "")
    assert parents[0].heading_path == ("工作经历",)
```

- [ ] **Step 2: Run the focused chunker tests to confirm the old type-transition flush fails the new assertion.**

Run: `conda run -n agentic-rag pytest --import-mode=importlib tests/unit/ingestion/test_chunker.py -q`

Expected: the new mixed Parent test fails with more than one Parent or a non-mixed type.

- [ ] **Step 3: Remove only the regular-type flush branch.** In `_append_regular_piece`, calculate the joined candidate regardless of the previous regular piece's type. Keep the existing target-max and hard-max decisions, flush only when the normal token limits require it, and retain the current structural-atom and heading flushes. Let `_materialize` derive `mixed` from the set of piece content types.

- [ ] **Step 4: Run chunker regressions and static checks.**

Run: `conda run -n agentic-rag pytest --import-mode=importlib tests/unit/ingestion/test_chunker.py tests/unit/ingestion/test_assembler.py -q && conda run -n agentic-rag ruff check src/agentic_rag/ingestion/chunker.py tests/unit/ingestion/test_chunker.py`

Expected: all existing table/code/list locator and token-limit tests pass; only intended Parent boundaries change.

---

### Task 3: Preserve heading and retrieval-target provenance

**Files:**
- Modify: `src/agentic_rag/persistence/repositories.py` (`ParentChunk` and `SqlAlchemyParentRepository.get_many`).
- Modify: `src/agentic_rag/retrieval/models.py`, `src/agentic_rag/retrieval/parents.py`.
- Modify: `src/agentic_rag/query/evidence_builder.py`, `src/agentic_rag/safety/context.py`, `src/agentic_rag/query/audit.py`.
- Modify: `src/agentic_rag/query/fast_rag.py`, `src/agentic_rag/query/tools.py`, and `src/agentic_rag/retrieval/graph.py` only where the server-owned batch target metadata is attached.
- Test: `tests/unit/persistence/test_mysql.py`, `tests/unit/retrieval/test_parent_aggregation.py`, `tests/unit/query/test_evidence_builder.py`, `tests/unit/query/test_audit.py`, `tests/unit/query/test_router_fast_path.py`.

**Interfaces:**
- Consumes: existing `parent_chunks.heading_path` JSON and retrieval target IDs assigned by the server at the retrieval boundary.
- Produces: `ParentEvidence.heading_path`, `EvidenceItem.heading_path`, `EvidenceManifestEntry.heading_path`, `EvidenceBatch.target_ids`, and `DataEnvelope.render()` output containing a separate `heading_path` untrusted-data field.

- [ ] **Step 1: Add failing round-trip, manifest, and target-provenance tests.** Assert MySQL hydration copies a JSON heading path into `ParentChunk`, `ParentFetcher.fetch/hydrate` copies it into `ParentEvidence`, and `EvidenceBuilder` copies it into the item and manifest. Assert `DataEnvelope.render()` exposes `heading_path` as data without prepending it to content. Add a manifest mismatch test that rejects a different heading path. Add a batch with `target_ids=("todo-work",)` whose content does not contain the target description; assert the target is still covered, while a batch with no matching target ID is not covered.

```python
def test_target_coverage_comes_from_server_batch_provenance() -> None:
    packed = EvidenceBuilder().build(
        [batch(parent("p1", content="Employer: Acme", heading_path=("Work",)), target_ids=("todo-work",))],
        [EvidenceCoverageTarget(target_id="todo-work", description="a phrase absent from content")],
        SCOPE,
        SNAPSHOT,
    )
    assert packed.items[0].covered_target_ids == ("todo-work",)
    assert packed.items[0].heading_path == ("Work",)
```

- [ ] **Step 2: Run the focused provenance tests and record every constructor failure.**

Run: `conda run -n agentic-rag pytest --import-mode=importlib tests/unit/persistence/test_mysql.py tests/unit/retrieval/test_parent_aggregation.py tests/unit/query/test_evidence_builder.py tests/unit/query/test_audit.py -q`

Expected: new provenance assertions fail before the fields and server-owned metadata are wired; existing fixtures identify constructors that need explicit empty defaults or target metadata.

- [ ] **Step 3: Add backward-compatible immutable fields and hydrate them.** Add `heading_path: tuple[str, ...] = ()` to persistence/retrieval/evidence models so old test fixtures remain valid. Read and validate the existing SQL JSON value as a tuple in `SqlAlchemyParentRepository`, preserve it through both ParentFetcher paths, and include it in evidence IDs' manifest consistency comparisons. Add `heading_path` to `DataEnvelope` with a default empty tuple and render JSON keys `trust`, `source`, `evidence_id`, `heading_path`, and `content`.

- [ ] **Step 4: Add server-owned target IDs and replace substring coverage.** Add `target_ids: tuple[str, ...] = ()` to `EvidenceBatch`. `ResearchToolset.retrieve_evidence` must attach the requested `target_id` to the returned batch; FastRag must attach `query:<run_id>` and retain the batch in `retrieval_batches`. `EvidenceBuilder._candidates` must derive target IDs only from `batch.target_ids`, and `_pack_candidate` must retain those IDs whenever the fitted content remains non-empty. Determine the single-document six-parent allowance from the valid candidate set, not from an empty selector list; keep the three-parent-per-document cap when multiple documents are present and cap a direct single-document result at six items.

- [ ] **Step 5: Make authorization compare headings.** Include `heading_path` in `_entry_matches_item`, `_parent_record`, and `_authorization_allows`; a heading mismatch must fail before prompt construction/publication. Do not add a migration because the SQL column and Elasticsearch Child field already exist.

- [ ] **Step 6: Run focused query, retrieval, persistence, and lint checks.**

Run: `conda run -n agentic-rag pytest --import-mode=importlib tests/unit/persistence tests/unit/retrieval tests/unit/query/test_evidence_builder.py tests/unit/query/test_audit.py tests/unit/query/test_router_fast_path.py -q && conda run -n agentic-rag ruff check src/agentic_rag/persistence/repositories.py src/agentic_rag/retrieval src/agentic_rag/query src/agentic_rag/safety/context.py tests/unit/persistence tests/unit/retrieval tests/unit/query`

Expected: heading paths round-trip, target coverage no longer depends on literal question text, a one-document result may include six Parents, and scope/manifest tests still fail closed.

---

### Task 4: Make Research own one packed evidence working set

**Files:**
- Modify: `src/agentic_rag/query/subagents.py`, `src/agentic_rag/query/tools.py`.
- Modify: `src/agentic_rag/query/research_loop.py`, `src/agentic_rag/query/context.py`, `src/agentic_rag/query/graph.py`.
- Test: `tests/unit/query/test_subagents.py`, `tests/unit/query/test_research_loop.py`, `tests/unit/query/test_graph.py`.

**Interfaces:**
- Consumes: `EvidenceBatch` plus `PackedEvidence` from direct and delegated retrieval; checkpointed `packed_context` and `retrieval_batches`.
- Produces: `SubagentResult(todo_id, evidence, batch)`, one validated merged `PackedEvidence`, and graph updates containing both `packed_context` and every raw retrieval batch.

- [ ] **Step 1: Write failing propagation tests.** Add a dispatcher test whose worker returns `(EvidenceBatch, PackedEvidence)` and assert the `SubagentResult` retains the raw batch. Add a Research loop test where action one delegates a Todo and action two's captured prompt contains the delegated evidence text and manifest ID. Add a delegate-then-submit graph test asserting post-Research `evidence_builder` receives the raw batch and does not return `research_batches_missing`. Add a checkpoint test with a wrong index generation or over-budget packed context and assert safe termination without an answer.

```python
async def test_second_supervisor_action_sees_delegated_evidence() -> None:
    prompts: list[str] = []
    # The scripted gateway records the user message for each action.
    result = await loop.ainvoke(state)
    assert "delegated evidence" in prompts[1]
    assert "delegated-evidence-id" in prompts[1]
    assert result["retrieval_batches"]
```

- [ ] **Step 2: Run the focused Research tests to demonstrate stale-context behavior.**

Run: `conda run -n agentic-rag pytest --import-mode=importlib tests/unit/query/test_subagents.py tests/unit/query/test_research_loop.py tests/unit/query/test_graph.py -q`

Expected: the new action-two and raw-batch assertions fail because the loop currently updates only a local evidence list and `SubagentResult` has no batch.

- [ ] **Step 3: Propagate raw batches from child tools and dispatcher.** Change production child workers to return `(EvidenceBatch, PackedEvidence)`. Keep a temporary `batch: EvidenceBatch | None = None` default on `SubagentResult` so existing isolated fixtures remain valid, but production delegate results must carry the batch. `SubagentTools.retrieve_evidence` returns both values; dispatcher normalizes the worker result and never discards the batch.

- [ ] **Step 4: Add deterministic packed merge and validation helpers.** Parse an existing checkpointed `PackedEvidence`, verify its index generation, item/manifest equality, non-negative bounded token count, and snapshot limit. Merge packs by evidence ID in deterministic order, reject conflicting manifests/generations, render each item through `DataEnvelope`, and enforce `snapshot.max_evidence_tokens`. Preserve each item's heading path and manifest exactly.

- [ ] **Step 5: Thread the merged pack through every loop action.** Change `_next_action` to receive the current pack and stage `packed_context`, `evidence`, Todo state, observations, known evidence IDs, and unresolved Todo IDs before calling `ContextBuilder`. Direct retrieval merges its pack and appends its batch. Delegation merges all child packs and appends every non-null child batch. `_result` returns the merged `packed_context`, evidence list, and all raw batches; `submit_evidence` validates IDs against the merged pack.

- [ ] **Step 6: Keep the post-Research graph as the trust boundary and fix spans.** The graph's evidence-builder node must rebuild from `EvidenceBatch` values, grade only the rebuilt pack, and refuse malformed/missing batches. Remove the outer `retrieval.tool` span around the whole Research loop; retain a graph-node span and add distinct supervisor-LLM, delegated-retrieval, and tool-execution scopes where the existing tracing/event ports support them.

- [ ] **Step 7: Run Research/graph regressions and static checks.**

Run: `conda run -n agentic-rag pytest --import-mode=importlib tests/unit/query/test_subagents.py tests/unit/query/test_research_loop.py tests/unit/query/test_graph.py tests/unit/query/test_router_fast_path.py -q && conda run -n agentic-rag ruff check src/agentic_rag/query`

Expected: action two sees delegated evidence, delegate-submit reaches post-Research grading, mixed generations and missing batches refuse safely, and no graph span labels the full Research loop as one retrieval tool.

---

### Task 5: Enforce exact structured schemas and one total model deadline

**Files:**
- Modify: `src/agentic_rag/runtime/model_gateway.py`.
- Modify: `src/agentic_rag/runtime/query_composition.py`.
- Modify: `src/agentic_rag/query/context.py`, `src/agentic_rag/query/research_loop.py`, and `src/agentic_rag/prompts/research_agent_v1.md`.
- Test: `tests/unit/runtime/test_model_gateway.py`, `tests/unit/runtime/test_query_composition.py`, `tests/unit/query/test_research_loop.py`.

**Interfaces:**
- Consumes: a Pydantic model class passed to `ModelGateway.complete_structured` and the current Research state.
- Produces: provider requests containing a server-owned exact JSON Schema plus generic JSON-object mode, bounded `ModelResponse.attempts`, safe timeout/degradation classifications, and an action prompt that states valid Todo/evidence transitions.

- [ ] **Step 1: Write failing schema/deadline tests.** Assert the structured provider messages contain the full `schema.model_json_schema()` under a system-level contract while `response_format`/`text.format` remains generic JSON-object mode. Use an injected clock and fake client that times out/retries to assert initial retries plus repair never exceed the operation deadline, repair runs at most once, and repair is skipped when no budget remains. Assert telemetry labels initial versus schema repair and provider timeout versus gateway deadline exhaustion. Assert the composed DeepSeek client timeout exceeds the snapshot operation timeout.

```python
async def test_structured_retry_and_repair_share_one_deadline() -> None:
    gateway = ModelGateway(client, max_retries=2, clock=clock, sleep=advance_clock)
    with pytest.raises(StructuredOutputValidationError):
        await gateway.complete_structured(call.model_copy(update={"timeout_seconds": 1.0}), RouteDecision)
    assert clock.value <= 1.0
    assert len(client.responses.calls) <= 4  # initial attempts plus one repair attempt
```

- [ ] **Step 2: Run the focused model and composition tests to confirm the old behavior fails.**

Run: `conda run -n agentic-rag pytest --import-mode=importlib tests/unit/runtime/test_model_gateway.py tests/unit/runtime/test_query_composition.py tests/unit/query/test_research_loop.py -q`

Expected: schema-contract, repair-at-most-once, or deadline assertions fail before the gateway is changed; existing protocol tests provide compatibility constraints.

- [ ] **Step 3: Add schema contract messages without changing provider compatibility.** Pass the schema through `_request_with_retries` to `_create`; append a server-owned system message containing a canonical JSON rendering of `model_json_schema()` and retain the existing generic JSON-object response hints. Include the schema name and phase in safe diagnostics, never raw provider output.

- [ ] **Step 4: Bound the complete structured operation.** Inject a monotonic clock, calculate one deadline from `call.effective_timeout_seconds`, pass remaining time to each initial retry, cap schema repair to one provider attempt, and bound backoff by remaining time. Add typed timeout classification for provider timeout and gateway total-deadline exhaustion; emit a repair-skipped event when the remaining budget cannot safely start repair. Preserve cancellation and interpreter control-flow propagation.

- [ ] **Step 5: State Research transitions explicitly.** Extend `ContextBuilder` output with available Todo IDs, known evidence IDs, `unresolved_todos_remain`, and `submit_evidence_valid`; update `research_agent_v1.md` so the model is told that this state contract is server-owned. `_next_action` supplies the current pack and Todo snapshot, so the schema and prompt agree on valid actions.

- [ ] **Step 6: Give the gateway timeout ownership at composition.** Configure the OpenAI-compatible DeepSeek client timeout to `snapshot.query_run_timeout_seconds + 5` (or an equivalent strictly greater value) while leaving embedding timeout behavior unchanged. Do not enforce a provider timeout smaller than the gateway deadline inside the fake-client-compatible gateway constructor.

- [ ] **Step 7: Run model, Research, lint, and type checks.**

Run: `conda run -n agentic-rag pytest --import-mode=importlib tests/unit/runtime/test_model_gateway.py tests/unit/runtime/test_query_composition.py tests/unit/query/test_research_loop.py -q && conda run -n agentic-rag ruff check src/agentic_rag/runtime src/agentic_rag/query && MYPYPATH=src conda run -n agentic-rag mypy --explicit-package-bases src/agentic_rag/runtime src/agentic_rag/query`

Expected: provider request compatibility tests pass, total deadline is bounded, telemetry contains no raw output, and mypy/Ruff are clean for changed runtime/query modules.

---

### Task 6: Publish the v2/v3 rollout contract without running the rebuild

**Files:**
- Modify: `src/agentic_rag/models/indexing.py`, `src/agentic_rag/config.py`, and any configuration documentation that states the active pipeline/index defaults.
- Modify: `tests/unit/test_config.py`, `tests/unit/test_documentation_contracts.py`, and relevant ingestion/retrieval fixtures that assert the active generation.
- Create or modify: `scripts/` only for a guarded, dry-run rebuild preflight/report command if the existing scripts do not expose the required checks; no command may delete data or resolve wildcard index targets.

**Interfaces:**
- Consumes: existing backup/restore tooling, artifact inventory, settings validation, and exact physical index naming.
- Produces: explicit `ingestion_pipeline_version="ingestion-v2"`, `index_generation="index-v3"` contract and a dry-run report containing reordered blocks/reasons, Parent counts, token distributions, and source-span coverage.

- [ ] **Step 1: Write failing default/version and preflight tests.** Assert the repaired settings contract uses ingestion-v2/index-v3, the active alias is `agenticrag-children-active`, the physical target resolves exactly to `agenticrag-children-index-v3`, and a preflight rejects unresolved or wildcard index targets and any non-triggering source reorder.

- [ ] **Step 2: Run the focused settings/documentation tests to verify current v2/v1 defaults fail the new contract.**

Run: `conda run -n agentic-rag pytest --import-mode=importlib tests/unit/test_config.py tests/unit/test_documentation_contracts.py -q`

Expected: the new contract assertions fail against the current defaults.

- [ ] **Step 3: Update only the explicit rollout defaults and guards.** Change the writable default generation and ingestion pipeline default to v3/v2, keep legacy-generation rejection intact, and make any preflight command require an explicit resolved physical index plus a verified backup and quiescent workers before it can report ready. Do not invoke deletion, soft-delete, upload, or restore from tests or this implementation turn.

- [ ] **Step 4: Run settings, ingestion, staging, and documentation tests.**

Run: `conda run -n agentic-rag pytest --import-mode=importlib tests/unit/test_config.py tests/unit/test_documentation_contracts.py tests/unit/ingestion tests/unit/persistence/test_staging.py -q && conda run -n agentic-rag ruff check src/agentic_rag/config.py src/agentic_rag/models/indexing.py scripts`

Expected: the code advertises v2/v3 consistently while legacy and ambiguous targets remain rejected.

---

### Task 7: Full verification and handoff

**Files:**
- Verify all modified files from Tasks 1–6; do not include unrelated pre-existing changes in the handoff.
- Test: unit, integration, and opt-in real-service suites named below.

- [ ] **Step 1: Run the complete non-live unit suite.**

Run: `conda run -n agentic-rag pytest --import-mode=importlib -m "not integration and not e2e and not live_model" -q`

Expected: all non-live tests pass; any failure is fixed with a focused regression test before continuing.

- [ ] **Step 2: Run integration tests that do not mutate durable local production data.**

Run: `conda run -n agentic-rag pytest --import-mode=importlib -m integration tests/integration/ingestion tests/integration/retrieval tests/integration/runtime -q`

Expected: configured local services pass; missing service variables produce the repository's explicit skip behavior.

- [ ] **Step 3: Run static and diff checks.**

Run: `conda run -n agentic-rag ruff check src tests evals scripts && MYPYPATH=src conda run -n agentic-rag mypy --explicit-package-bases src evals scripts && git diff --check`

Expected: Ruff, mypy, and whitespace checks pass for our changes; pre-existing unrelated diagnostics are recorded separately.

- [ ] **Step 4: Review acceptance evidence without executing destructive operations.** Confirm code/tests cover aligned date placement, no-op/idempotence, mixed Parent continuation, heading manifest authorization, six-parent single-document selection, delegated evidence visibility, raw-batch graph rebuild, exact schema contract, bounded repair deadline, v2/v3 naming, and PII-safe diagnostics. Leave the backup-delete-reingest runbook for a separately approved execution turn.
