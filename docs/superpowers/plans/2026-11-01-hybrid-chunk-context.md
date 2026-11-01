# Hybrid Chunk Context Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the approved contextual retrieval design, including grounded ingestion, retrieval/rerank propagation, query business scope, supporting evidence, durable builds, and offline acceptance tooling.

**Architecture:** Keep original Canonical AST/Parent/Child evidence and derive an immutable searchable representation. Server-owned authorization remains separate from business scope. Versioned context artifacts connect ingestion, ES candidates, reranking, and citation-validated source evidence; opt-in composition preserves legacy operation.

**Tech Stack:** Python 3.11, Pydantic 2, LangGraph, SQLAlchemy/Alembic, Elasticsearch, existing ModelGateway and ArtifactStore, pytest.

**Spec:** [approved design](/Users/steven/.codex/worktrees/36dd/AgenticRAG/docs/superpowers/specs/2026-10-31-hybrid-chunk-context-design.md).

## Global Constraints

- Work only in `/Users/steven/.codex/worktrees/36dd/AgenticRAG`; preserve unrelated changes. Baseline `f875fb0`, existing untracked design belongs to this task.
- Use `/Users/steven/miniconda3/envs/agentic-rag/bin/python` for checks. No live services, downloaded models, external model calls, active index changes, migrations against user databases, restarts, pushes, or publication in this development run.
- Original `content`, Parent text and source locators remain original. `contextualized_content` is the single search representation. New representation version is `context-v1`; absence is explicit legacy behavior.
- `320` base tokens, `64` prefix tokens, `384` final tokens, `6000` context input tokens, `16` source blocks, `2` reference hops; actual tokenizer measurements include separators. Never truncate original child content or qualifying exceptions.
- All LLM semantic claims, including same-paragraph claims, require a distinct verifier call. Unknown, ambiguous, contradicted and insufficient claims are omitted; quote/identity/hash integrity failures fail the build. Model output cannot set trusted use permissions or source coordinates.
- Search-only LLM claims never authorize ACL or exclusionary business filters. Coverage certificates must bind the current corpus/version universe; unresolved or empty scope must not become an unfiltered search.
- Immutable content-addressed artifacts, durable budget reservation, lease fencing and first accepted result reuse are required. Retry cannot reset spend or overwrite selected results.
- Context-dependent answers require original supporting sources, including accepted claims omitted from the rendered prefix. Maximum 4 required source excerpts per selected child and 25% of evidence budget for supporting context; missing dependencies remain explicit.
- New behavior is disabled by default; no full enterprise IAM is introduced. Trusted profile and entity/coverage sources are injectable and default to absent.
- Full feature scope is code and offline verification, not production rollout. Implement migration/release guards and commands without executing a real release.

## Review Focus

- Same company name can denote customer/subject/example: a real quote alone must not establish applicability (Tasks 1, 3, 4).
- New text can silently disappear during RRF/worker serialization or survive mismatched IDs: round-trip and conflict tests bind representation hashes (Tasks 2, 5).
- Prefix budgeting can drop the qualifier required by the answer: evidence dependencies are computed from all accepted claims (Task 5).
- Expired leases, crashed model calls and profile changes must not reset cost or mutate a running query's meaning (Tasks 3, 4).
- New source versions created after cutover must remain available under rollback, without resurrecting deleted versions (Task 6).

## Execution and verification

Each task uses a failing behavioral test first, then implementation, focused regression, self-review and a separate task review. Shared-interface changes must be recorded in the report for the next task. The controller runs the complete offline suite after integration; expensive/infrastructure/live suites remain separately reported. Agent implementers do not spawn agents. Commits use managed-worktree-compatible tooling if permitted; otherwise keep reviewable worktree patches and record this in the ledger.

### Task 1: Grounded context contracts and deterministic source operations

**Files:** Create `src/agentic_rag/ingestion/contextualization/{__init__,models,sources,rendering}.py`, shared lightweight `src/agentic_rag/models/contextual.py`, and `tests/unit/ingestion/test_context_sources.py`, `test_context_rendering.py`.

**Interfaces:**
- `models/contextual.py` exports frozen Pydantic `ContextArtifactRef(uri, sha256, size_bytes)`, `CanonicalSourceRef`, `ContextClaim`, `ContextArtifact`, `SearchRepresentation`; it must not import ingestion/chunker or Docling.
- `CanonicalSourceRef` binds owner/document/version/AST hash/block/path/half-open char span/quote hash and derived page range. `ContextClaim` binds field/value/scope/source refs/status/verification/allowed_uses/exclusions; artifacts retain all accepted and rejected claims plus rendered IDs.
- `SourceCatalog.from_ast(ast: CanonicalAst, ast_sha256: str) -> SourceCatalog`; `resolve_quote(source_id: str, quote: str) -> CanonicalSourceRef`; `verify_source(ref: CanonicalSourceRef) -> str`; `covers_child(refs, child) -> bool` require all body spans.
- `render_context(child: ChildChunk, claims: Sequence[ContextClaim], *, count_tokens: Callable[[str],int], build_fingerprint: str, context_ref: ContextArtifactRef | None) -> SearchRepresentation` retains original text, filters usable claims, budgets whole fields, hashes final bytes, and derives final ID. A context reference may be attached after content-addressed artifact persistence without changing text identity.
- Keep `SearchRepresentation` explicit: source_child_id, id, content, source_content_hash, base_contextualized_content, context_prefix, contextualized_content, content_hash, context_claim_ids, representation_version, enrichment_status, token_count, build_fingerprint, optional context_ref. Pure model validators verify text/hash coherence and required context for semantic claims.

- [ ] Write `test_real_quote_wrong_owner_or_version_is_rejected`, `test_ambiguous_quote_is_not_silently_first_match`, `test_partial_scope_cannot_label_whole_child`, and `test_overflow_drops_whole_claim_preserving_exclusion_and_body`. Assert exact source text, 384 limit, unchanged content, rejected forged IDs and changed final hash on changed context.
- [ ] Run these tests and record the expected missing-behavior failures.
- [ ] Implement contracts, catalog and renderer; derive source coordinates server-side and distinguish artifact integrity errors from insufficient semantic claims.
- [ ] Run the new files plus `tests/unit/ingestion/test_chunker.py`; check legacy input remains accepted. Report exact interfaces and any deviations before review.
- [ ] Produce task diff/report and commit if permitted.

### Task 2: Versioned retrieval representation through ES and rerank

**Files:** Modify `ingestion/chunker.py`, `persistence/elasticsearch.py`, `retrieval/{models,fusion,reranker,graph,state}.py`, `retrieval/adapters/elasticsearch.py`; extend tests in `tests/unit/retrieval/` and add `test_context_representation.py`.

**Interfaces:**
- Extend ChildChunk with optional `search_representation: SearchRepresentation | None`, while preserving legacy fields/validators/serialization; provide one deterministic adapter producing a final indexed ChildChunk from the raw child and representation, without losing original source_child_id.
- Extend ChildHit with optional contextualized_content, representation_version, content_hash, source_child_id, source_content_hash, context_ref, context_claim_ids and build_fingerprint; a `search_content` property selects validated new representation or legacy original content. New `context-v1` hits require required hashes and references; structural_only still has valid enhanced text.
- ES uses the final contextualized_content for embedding and BM25, returns the same projection in both lanes, and new strict mapping admits the added metadata. Legacy documents lacking fields keep the old behavior.
- RRF verifies duplicate identities and search hashes before merging; reranker passes `(query, hit.search_content)`; enforce input lengths through an optional process-owned pair tokenizer/counting adapter. Preserve full document, bounded query windows (at most 2), fixed max-score aggregation; explicit RRF fallback when impossible.

- [ ] Write failing tests: ES dense/BM25 round trip, rerank sees the context rather than only the original sentence, legacy hit remains original, new incomplete hit fails, same ID/conflicting hash fails, JSON/RankedCandidates round trip preserves metadata, and long pair never silently clips document.
- [ ] Implement the projection, models and inference adapter while keeping generation policy explicit. Do not duplicate search_text as a second mutable field.
- [ ] Run `tests/unit/retrieval`, relevant staging/manifest unit tests, and focused orchestration candidate serialization tests. Report production composition hooks needed by Task 6.
- [ ] Produce task diff/report and commit if permitted.

### Task 3: Context extraction and durable ingestion integration

**Files:** Create `ingestion/contextualization/{map,assembly,service,gateway,store}.py`, `persistence/context_builds.py`, prompts `context_extract_v1.md` and `context_verify_v1.md`; modify `ingestion/{graph,state,indexer,manifest,publisher,chunker}.py`, `config.py`, `runtime/models.py`, Alembic migration after current head. Add ingestion source/service/recovery tests and SQLite-backed persistence tests where feasible.

**Interfaces:**
- `ContextualizationService.enrich(canonical: CanonicalAst, parents: Sequence[ParentChunk], *, canonical_ref: ContextArtifactRef, build: ContextBuildConfig, claim: IngestionJobClaim, before_side_effect: Callable[[], Awaitable[None]]) -> ContextBuildResult` returns original parents with derived children and immutable artifact refs; constructor owns source/catalog/model/tokenizer/store dependencies.
- `ContextBuildConfig` freezes model IDs/provider fingerprint/prompt hashes/token budgets/model usage prices and finite request/token/cost/concurrency caps. Disabled/unconfigured extraction yields structural_only; enabled provider without finite budget fails configuration.
- `ContextModelPort.extract(input: ContextInput) -> ExtractionResponse` and `.verify(input: VerificationInput) -> VerificationResponse`; production gateway adapter uses existing ModelGateway structured calls, source IDs/quotes only, independent verifier call for every semantic candidate and no hidden unconstrained retry. Implement document/section map, bounded source selection and explicit references with 2-hop/16-block/6000-token limits.
- `ContextBuildStore` reserves durable usage before model dispatch, settles usage, and accepts a ready artifact under owner/document/version/fingerprint/source-child key and valid lease generation. SQLAlchemy production store must check durable Job lease in the same transaction as accepted-pointer update. Source artifact writes can be orphaned but cannot supersede accepted output. Test double store may be supplied for unit tests, never silently selected in production.
- Add opt-in graph `contextualize` node between chunk and embed; state carries refs only. Default legacy topology/recovery remains supported. New chunking budget is configurable (320 for enabled context pipeline, legacy 384 otherwise).
- Manifest version 2 binds context artifact root/enriched-chunks hashes, fingerprint/representation and child ID/hash digest; version 1 payload/hash remains byte-compatible. Publisher validates new manifests against stored artifacts/child identities, not counts alone.

- [ ] Add failing tests for independent verification of same-paragraph and cross-paragraph claims, unsupported scope and invented units, provider timeout partial fallback, no overwritten raw artifact, retry reuse, pre-call reservation across crash, stale lease rejection, changed model/prompt producing distinct build, and old manifest hash preservation.
- [ ] Implement bounded generation, persistent store/migration, graph and indexing integration. Reuse existing safety gates; refuse cross-owner/version sources.
- [ ] Run context tests, `tests/unit/ingestion`, focused staging/publisher tests using offline doubles, and migration schema tests. Exercise an enabled ingestion graph end-to-end with recorded model responses, not a network call.
- [ ] Produce task diff/report and commit if permitted.

### Task 4: Business scope query compilation without changing authorization

**Files:** Create `retrieval/query_context.py`, `retrieval/query_compiler.py`, and tests `test_query_compiler.py`; modify `retrieval/{models,state,graph,filters,ports}.py`, ES lexical adapter, query `fast_rag.py`, `tools.py`, ToolContext/native adapter, orchestration context/execution/agents, config/snapshot and composition call sites as needed.

**Interfaces:**
- Frozen `QueryBusinessContext` captures source question, task intent, trusted profile revision and permitted task target selectors. `TrustedBusinessProfileProvider` defaults to absent. `BusinessScopeCoverage` binds owner/corpus snapshot/generation/version-set hash/mapping revision and completeness; exact validation against a trusted provider is required before business exclusions.
- `QueryCompiler.compile(request: RetrievalRequest, scope: UserScope, snapshot: RuntimeConfigSnapshot, business: QueryBusinessContext | None) -> QueryPlan` (async if provider lookups require it) returns original_query, scope resolutions/provenance, dense_query, lexical_core_query, lexical_scope_terms, rerank_query and a server-validated effective filter. Persist immutable plan through injected ArtifactStore when enabled, carrying query_plan_ref/hash through EvidenceBatch/RankedCandidates. Define the immutable QueryPlan model in shared retrieval contract module so Task 5 can use it.
- Retrieval service's `retrieve` and `retrieve_candidates` accept optional keyword-only business_context and compile at the common boundary; process-owned dependencies cannot be supplied by the model. Default off behavior and existing test doubles remain supported through composition, not blanket exception swallowing.
- Precedence: explicit user/current task target, confirmed conversation reference, then trusted profile for explicitly self-directed policy requests. Never append own company to an explicit other-company/general query. No profile means no inferred default. Ambiguity is explicit; empty intersection never means no filter.
- Dense/rerank consume full semantic query; BM25 core remains required with supplementary scope should terms in one lexical lane. Keep exact user/task document_ids bounds even without coverage certificates.

- [ ] Write failing tests for self-directed profile enrichment, explicit other company override, general question unchanged, missing/ambiguous profile, conflicting task target, incomplete/stale coverage, empty document intersection and unavailable target without ACL widening.
- [ ] Implement frozen compilation and propagation across both direct and tool/runtime/worker paths. Production resolution may use injectable trusted entity directory; implement explicit-input behavior with source spans and reject uncertain aliases instead of inventing values. Do not require enterprise IAM or infer company from user_id.
- [ ] Run retrieval, query fast path/tool, orchestration agent and ToolRuntime focused tests; serialize/resume a QueryPlan while profile provider revision changes and assert original meaning remains fixed.
- [ ] Produce task diff/report and commit if permitted.

### Task 5: Context source hydration and complete citation evidence

**Files:** Create `retrieval/context_evidence.py`; modify `query/{evidence_builder,audit,answer_sources,research_loop,subagents}.py`, `retrieval/{models,graph,parents}.py`, `orchestration/{evidence,contracts,assessment}.py` and source models/API readers as required; add `tests/unit/query/test_context_evidence.py` and extend audit/answer-source/orchestration tests.

**Interfaces:**
- `EvidenceRequirementResolver.resolve(query_plan: QueryPlan | None, hit: ChildHit, artifact: ContextArtifact) -> EvidenceRequirements` binds original source, task scope and all accepted claims (not only rendered IDs); resolves missing/conflicting states explicitly.
- `ContextSourceHydrator.hydrate(hits, scope, snapshot, query_plan) -> SupportingContext` loads context artifacts and authorized canonical excerpts; uses durable active version, generation/build membership and permissions, not only cached ES metadata. Same-Parent support deduplicates, without inventing hits/scores.
- Versioned evidence union supports parent_chunk and canonical_excerpt, with source-kind-specific provenance, content hashes and nullable parent_id for excerpts. Trusted metadata evidence is explicitly disabled initially. EvidenceManifestEntry and item identity projections remain exact.
- EvidenceBuilder packs body plus required scope sources as an indivisible group within total/25%/4-excerpt caps; missing support cannot be marked covered. Pass requirements and status to grader; arbitrary extra generated scope assertion must still be rejected by faithfulness/citation checks.
- Both full retrieval and Supervisor EvidencePreparer use the shared source service. Implement authoritative canonical source authorization branch in CitationValidator and public AnswerSource page extraction without weakening old parent checks.

- [ ] Failing tests cover off-Parent company/audience paragraphs, accepted-but-not-rendered exception, missing/forged/revoked/stale source, insufficient supporting budget, correct excerpt citation/public source pages, duplicate candidate hashes and JSON/checkpoint round trips.
- [ ] Implement source hydration, typed evidence union, grouped packing and all legacy consumers; existing parent-only evidence contracts continue to work.
- [ ] Run evidence/audit/answer-sources/research/subagent/orchestration suites and a fake-provider ingestion → ES projection → rerank → source hydration → citation validation end-to-end test.
- [ ] Produce task diff/report and commit if permitted.

### Task 6: Build registry release controls production wiring and acceptance

**Files:** Create/extend `persistence/context_builds.py`, `ingestion/contextualization/builds.py`, `ingestion/contextualization/release.py`, Alembic schema, `scripts/rebuild_context_index.py`, offline evaluation helper under `evals/`, and `docs/contextual-retrieval.md`; modify bootstrap, query/ingestion/orchestration composition, settings/example config, publisher/reconciler and deletion lifecycle as necessary; add focused integration/release/evaluation/composition tests.

**Interfaces:**
- Durable retrieval_builds registry uniquely binds owner/document_version/index_generation, immutable manifest ref/parent hash/fingerprint/state; existing document_versions generation fields are compatibility projections for selected build, not multiple-build history.
- `stage_search_representation` rebuilds child representations for an active source version without setting Parent/version building or overwriting source. Publisher verifies ID/hash collections and current source validity.
- Release journal records before/after mapping and stage, requires explicit drained/frozen readiness, checks all active source versions/deletions, and supports recoverable switch/rollback across separate DB/ES operations. Rollback readiness requires a compatible old-generation build for every currently effective source version; mutations during rollback window enqueue/record dual-build requirements. Missing build => not_ready; deletion is never undone.
- Rebuild/release CLI defaults to plan/dry-run, uses explicit source/target/manifest/config, never selects active index for in-place mutations, and refuses execute without validated journal/readiness inputs. No CLI is executed against real infrastructure in this task.
- Production composition wires context service, structured model adapters, SQL store, token counters, artifact resolvers, query compiler, scope source hydration and compatible reranker adapters when flags enabled. All flags false preserve current startup. Missing enabled dependencies fail configuration rather than silently using test doubles or ungrounded output.
- Offline ablation helper matches gold by document/version/source spans, reports A/B/C/D/E configurations and equal recall budgets, attribution/source-support failures plus token/cost/latency usage. Do not claim LLM quality from mocks.

- [ ] Add failing tests for opt-in runtime wiring, disabled startup compatibility, active source rebuild preserving Parent/version, resume at each release boundary, new version preventing incomplete rollback, deleted source never reactivated, durable usage/schema round trip and source-span gold matching across changed child IDs.
- [ ] Implement build/release and composition, add finite budget validation and documented activation/rebuild/rollback commands. Integration paths must invoke the new services, not leave standalone unused classes.
- [ ] Run relevant offline integration tests and the complete offline pytest suite, then Ruff on changed Python files and `git diff --check`. Capture any pre-existing/environment failures precisely; do not expand to paid/live suites.
- [ ] Update spec status/operations guide with actual code and validation boundaries, create final task report and request whole-branch review.

## Completion evidence

- [ ] All six task reviews accepted or explicitly recorded with concrete unresolved limitations.
- [ ] Whole-branch spec and production-path review complete, material findings fixed.
- [ ] Focused and broad offline results recorded with exact commands.
- [ ] No deployment/index/paid-evaluation side effects occurred.
- [ ] Final delivery distinguishes implemented behavior, offline verification, and future live acceptance.
