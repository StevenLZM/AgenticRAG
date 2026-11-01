# Hybrid Chunk Context Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the approved contextual retrieval design, including grounded ingestion, retrieval/rerank propagation, query business scope, supporting evidence, durable builds, and offline acceptance tooling.

**Architecture:** Keep original Canonical AST/Parent/Child evidence and derive an immutable searchable representation. Server-owned authorization remains separate from business scope. Versioned context artifacts connect ingestion, ES candidates, reranking, and citation-validated source evidence; opt-in composition preserves legacy operation.

**Tech Stack:** Python 3.11, Pydantic 2, LangGraph, SQLAlchemy/Alembic, Elasticsearch, existing ModelGateway and ArtifactStore, pytest.

**Spec:** [approved design](/Users/steven/.codex/worktrees/36dd/AgenticRAG/docs/superpowers/specs/2026-10-31-hybrid-chunk-context-design.md).

## Global Constraints

- Work only in `/Users/steven/.codex/worktrees/36dd/AgenticRAG`; preserve unrelated changes. Baseline `f875fb0`, existing untracked design belongs to this task.
- Use `/Users/steven/.codex/worktrees/36dd/AgenticRAG/.venv/bin/python` for checks and set `RAG_EVAL_PYTHON` to that same absolute path. The local venv inherits the existing conda environment and adds the declared pypdf evaluation dependency. No live services, downloaded models, external model calls, active index changes, migrations against user databases, restarts, pushes, or publication in this development run.
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

- [x] Write `test_real_quote_wrong_owner_or_version_is_rejected`, `test_ambiguous_quote_is_not_silently_first_match`, `test_partial_scope_cannot_label_whole_child`, and `test_overflow_drops_whole_claim_preserving_exclusion_and_body`. Assert exact source text, 384 limit, unchanged content, rejected forged IDs and changed final hash on changed context.
- [x] Run these tests and record the expected missing-behavior failures.
- [x] Implement contracts, catalog and renderer; derive source coordinates server-side and distinguish artifact integrity errors from insufficient semantic claims.
- [x] Run the new files plus `tests/unit/ingestion/test_chunker.py`; check legacy input remains accepted. Report exact interfaces and any deviations before review.
- [x] Produce task diff/report and commit if permitted.

### Task 2: Versioned retrieval representation through ES and rerank

**Files:** Modify `ingestion/chunker.py`, `persistence/elasticsearch.py`, `retrieval/{models,fusion,reranker,graph,state}.py`, `retrieval/adapters/elasticsearch.py`; extend tests in `tests/unit/retrieval/` and add `test_context_representation.py`.

**Interfaces:**
- Extend ChildChunk with optional `search_representation: SearchRepresentation | None`, while preserving legacy fields/validators/serialization; provide one deterministic adapter producing a final indexed ChildChunk from the raw child and representation, without losing original source_child_id.
- Extend ChildHit with optional contextualized_content, representation_version, content_hash, source_child_id, source_content_hash, context_ref, context_claim_ids and build_fingerprint; a `search_content` property selects validated new representation or legacy original content. New `context-v1` hits require required hashes and references; structural_only still has valid enhanced text.
- ES uses the final contextualized_content for embedding and BM25, returns the same projection in both lanes, and new strict mapping admits the added metadata. Legacy documents lacking fields keep the old behavior. Adapter/store constructors accept an explicit expected representation policy for context-v1 generations; that policy rejects missing version/fields instead of silently downgrading new-generation records to legacy. Legacy mapping metadata/startup validation remains unchanged by default; new schema is opt-in.
- RRF verifies duplicate identities and search hashes before merging; reranker passes `(query, hit.search_content)`; enforce input lengths through an optional process-owned pair tokenizer/counting adapter. Preserve full document, bounded query windows (at most 2), fixed max-score aggregation; explicit RRF fallback when impossible.

- [x] Write failing tests: ES dense/BM25 round trip, rerank sees the context rather than only the original sentence, legacy hit remains original, new incomplete hit fails, same ID/conflicting hash fails, JSON/RankedCandidates round trip preserves metadata, and long pair never silently clips document.
- [x] Implement the projection, models and inference adapter while keeping generation policy explicit. Do not duplicate search_text as a second mutable field.
- [x] Run `tests/unit/retrieval`, relevant staging/manifest unit tests, and focused orchestration candidate serialization tests. Report production composition hooks needed by Task 6.
- [x] Produce task diff/report and commit if permitted.

### Task 3: Context extraction and durable ingestion integration

**Files:** Create `ingestion/contextualization/{map,assembly,service,gateway,store}.py`, `persistence/context_builds.py`, prompts `context_extract_v1.md` and `context_verify_v1.md`; modify `ingestion/{graph,state,indexer,manifest,publisher,chunker}.py`, `config.py`, `runtime/models.py`, Alembic migration after current head. Add ingestion source/service/recovery tests and SQLite-backed persistence tests where feasible.

**Interfaces:**
- `ContextualizationService.enrich(canonical: CanonicalAst, parents: Sequence[ParentChunk], *, canonical_ref: ContextArtifactRef, build: ContextBuildConfig, claim: IngestionJobClaim, before_side_effect: Callable[[], Awaitable[None]]) -> ContextBuildResult` returns original parents with derived children and immutable artifact refs; constructor owns source/catalog/model/tokenizer/store dependencies.
- `ContextBuildConfig` freezes model IDs/provider fingerprint/prompt hashes/token budgets/model usage prices and finite request/token/cost/concurrency caps. Disabled/unconfigured extraction yields structural_only; enabled provider without finite budget fails configuration.
- `ContextModelPort.extract(input: ContextInput) -> ExtractionResponse` and `.verify(input: VerificationInput) -> VerificationResponse`; production gateway adapter uses existing ModelGateway structured calls, source IDs/quotes only, independent verifier call for every semantic candidate and no hidden unconstrained retry. Implement document/section map, bounded source selection and explicit references with 2-hop/16-block/6000-token limits.
- `ContextBuildStore` reserves durable usage before model dispatch, settles usage, and accepts a ready artifact under owner/document/version/fingerprint/source-child key and valid lease generation. SQLAlchemy production store must check durable Job lease in the same transaction as accepted-pointer update. Source artifact writes can be orphaned but cannot supersede accepted output. Test double store may be supplied for unit tests, never silently selected in production.
- Add opt-in graph `contextualize` node between chunk and embed; state carries refs only. Default legacy topology/recovery remains supported. New chunking budget is configurable (320 for enabled context pipeline, legacy 384 otherwise).
- Manifest version 2 binds context artifact root/enriched-chunks hashes, fingerprint/representation and child ID/hash digest; version 1 payload/hash remains byte-compatible. Publisher validates new manifests against stored artifacts/child identities, not counts alone.

- [x] Add failing tests for independent verification of same-paragraph and cross-paragraph claims, unsupported scope and invented units, provider timeout partial fallback, no overwritten raw artifact, retry reuse, pre-call reservation across crash, stale lease rejection, changed model/prompt producing distinct build, and old manifest hash preservation.
- [x] Implement bounded generation, persistent store/migration, graph and indexing integration. Reuse existing safety gates; refuse cross-owner/version sources.
- [x] Run context tests, `tests/unit/ingestion`, focused staging/publisher tests using offline doubles, and migration schema tests. Exercise an enabled ingestion graph end-to-end with recorded model responses, not a network call.
- [x] Produce task diff/report and commit if permitted.

### Task 4: Business scope query compilation without changing authorization

**Files:** Create `retrieval/query_context.py`, `retrieval/query_compiler.py`, and tests `test_query_compiler.py`; modify `retrieval/{models,state,graph,filters,ports}.py`, ES lexical adapter, query `fast_rag.py`, `tools.py`, ToolContext/native adapter, orchestration context/execution/agents, config/snapshot and composition call sites as needed.

**Interfaces:**
- Frozen `QueryBusinessContext` captures source question, task intent, trusted profile revision and permitted task target selectors. `TrustedBusinessProfileProvider` defaults to absent. `BusinessScopeCoverage` binds owner/corpus snapshot/generation/version-set hash/mapping revision and completeness; exact validation against a trusted provider is required before business exclusions.
- `QueryCompiler.compile(request: RetrievalRequest, scope: UserScope, snapshot: RuntimeConfigSnapshot, business: QueryBusinessContext | None) -> QueryPlan` (async if provider lookups require it) returns original_query, scope resolutions/provenance, dense_query, lexical_core_query, lexical_scope_terms, rerank_query, protected_query_context and a server-validated effective filter. When protected_query_context is nonempty, rerank_query is exactly protected_query_context + "\n\n" + original semantic question body; pass that prefix to the reranker keyword so every bounded query window retains all business qualifiers. Persist immutable plan through injected ArtifactStore when enabled, carrying query_plan_ref/hash through EvidenceBatch/RankedCandidates. Define the immutable QueryPlan model in shared retrieval contract module so Task 5 can use it.
- Retrieval service's `retrieve` and `retrieve_candidates` accept optional keyword-only business_context and compile at the common boundary; process-owned dependencies cannot be supplied by the model. Default off behavior and existing test doubles remain supported through composition, not blanket exception swallowing.
- Precedence: explicit user/current task target, confirmed conversation reference, then trusted profile for explicitly self-directed policy requests. Never append own company to an explicit other-company/general query. No profile means no inferred default. Ambiguity is explicit; empty intersection never means no filter.
- Dense/rerank consume full semantic query; BM25 core remains required with supplementary scope should terms in one lexical lane. Keep exact user/task document_ids bounds even without coverage certificates.

- [x] Write failing tests for self-directed profile enrichment, explicit other company override, general question unchanged, missing/ambiguous profile, conflicting task target, incomplete/stale coverage, empty document intersection and unavailable target without ACL widening.
- [x] Implement frozen compilation and propagation across both direct and tool/runtime/worker paths. Production resolution may use injectable trusted entity directory; implement explicit-input behavior with source spans and reject uncertain aliases instead of inventing values. Do not require enterprise IAM or infer company from user_id.
- [x] Run retrieval, query fast path/tool, orchestration agent and ToolRuntime focused tests; serialize/resume a QueryPlan while profile provider revision changes and assert original meaning remains fixed.
- [x] Produce task diff/report and commit if permitted.

### Task 5: Context source hydration and complete citation evidence

**Files:** Create `retrieval/context_evidence.py`; modify `query/{evidence_builder,audit,answer_sources,research_loop,subagents}.py`, `retrieval/{models,graph,parents}.py`, `orchestration/{evidence,contracts,assessment}.py` and source models/API readers as required; add `tests/unit/query/test_context_evidence.py` and extend audit/answer-source/orchestration tests.

**Interfaces:**
- `EvidenceRequirementResolver.resolve(query_plan: QueryPlan | None, hit: ChildHit, artifact: ContextArtifact) -> EvidenceRequirements` binds original source, task scope and all accepted claims (not only rendered IDs); resolves missing/conflicting states explicitly.
- `ContextSourceHydrator.hydrate(hits, scope, snapshot, query_plan) -> SupportingContext` loads context artifacts and authorized canonical excerpts; uses durable active version, generation/build membership and permissions, not only cached ES metadata. Same-Parent support deduplicates, without inventing hits/scores.
- Versioned evidence union supports parent_chunk and canonical_excerpt, with source-kind-specific provenance, content hashes and nullable parent_id for excerpts. Trusted metadata evidence is explicitly disabled initially. EvidenceManifestEntry and item identity projections remain exact.
- EvidenceBuilder packs body plus required scope sources as an indivisible group within total/25%/4-excerpt caps; missing support cannot be marked covered. Pass requirements and status to grader; arbitrary extra generated scope assertion must still be rejected by faithfulness/citation checks.
- Both full retrieval and Supervisor EvidencePreparer use the shared source service. Implement authoritative canonical source authorization branch in CitationValidator and public AnswerSource page extraction without weakening old parent checks.

- [x] Failing tests cover off-Parent company/audience paragraphs, accepted-but-not-rendered exception, missing/forged/revoked/stale source, insufficient supporting budget, correct excerpt citation/public source pages, duplicate candidate hashes and JSON/checkpoint round trips.
- [x] Implement source hydration, typed evidence union, grouped packing and all legacy consumers; existing parent-only evidence contracts continue to work.
- [x] Run evidence/audit/answer-sources/research/subagent/orchestration suites and a fake-provider ingestion → ES projection → rerank → source hydration → citation validation end-to-end test.
- [x] Produce task diff/report and commit if permitted.

### Task 6: Build registry release controls production wiring and acceptance

**Files:** Create/extend `persistence/context_builds.py`, `ingestion/contextualization/builds.py`, `ingestion/contextualization/release.py`, Alembic schema, `scripts/rebuild_context_index.py`, offline evaluation helper under `evals/`, and `docs/contextual-retrieval.md`; modify bootstrap, query/ingestion/orchestration composition, settings/example config, publisher/reconciler and deletion lifecycle as necessary; add focused integration/release/evaluation/composition tests.

**Interfaces:**
- Durable retrieval_builds registry uniquely binds owner/document_version/index_generation, immutable manifest ref/parent hash/fingerprint/state; existing document_versions generation fields are compatibility projections for selected build, not multiple-build history.
- `stage_search_representation` rebuilds child representations for an active source version without setting Parent/version building or overwriting source. Publisher verifies ID/hash collections and current source validity.
- Release journal records before/after mapping and stage, requires explicit drained/frozen readiness, checks all active source versions/deletions, and supports recoverable switch/rollback across separate DB/ES operations. Rollback readiness requires a compatible old-generation build for every currently effective source version; mutations during rollback window enqueue/record dual-build requirements. Missing build => not_ready; deletion is never undone.
- Rebuild/release CLI defaults to plan/dry-run, uses explicit source/target/manifest/config, never selects active index for in-place mutations, and refuses execute without validated journal/readiness inputs. No CLI is executed against real infrastructure in this task.
- Production composition wires context service, structured model adapters, SQL store, token counters, artifact resolvers, query compiler, scope source hydration and compatible reranker adapters when flags enabled. All flags false preserve current startup. Missing enabled dependencies fail configuration rather than silently using test doubles or ungrounded output.
- Offline ablation helper matches gold by document/version/source spans, reports A/B/C/D/E configurations and equal recall budgets, attribution/source-support failures plus token/cost/latency usage. Do not claim LLM quality from mocks.

- [x] Add failing tests for opt-in runtime wiring, disabled startup compatibility, active source rebuild preserving Parent/version, resume at each release boundary, new version preventing incomplete rollback, deleted source never reactivated, durable usage/schema round trip and source-span gold matching across changed child IDs.
- [x] Implement build/release and composition, add finite budget validation and documented activation/rebuild/rollback commands. Integration paths must invoke the new services, not leave standalone unused classes.
- [x] Run relevant offline integration tests and the complete offline pytest suite, then Ruff on changed Python files and `git diff --check`. Capture any pre-existing/environment failures precisely; do not expand to paid/live suites.
- [x] Update spec status/operations guide with actual code and validation boundaries, create final task report and request whole-branch review.

## Completion evidence

- [x] All six task reviews accepted or explicitly recorded with concrete unresolved limitations.
- [x] Whole-branch spec and production-path review complete, material findings fixed.
- [x] Focused and broad offline results recorded with exact commands.
- [x] No deployment/index/paid-evaluation side effects occurred.
- [x] Final delivery distinguishes implemented behavior, offline verification, and future live acceptance.

## Final acceptance record

Date: 2026-11-01. Source baseline: `f875fb0e1673bdbe4c85272fbf88d9ca54e045b8`. Final functional code: `7ca5388b3d8934727a24090b61206f238ffa32d3`. The subsequent completion commit changes documentation only.

The final controller-run command, from this worktree, exited 0:

```sh
RAG_EVAL_PYTHON=/Users/steven/.codex/worktrees/36dd/AgenticRAG/.venv/bin/python .venv/bin/python -m pytest -q -m 'not integration and not e2e and not live_model'
```

Result: **1917 passed, 35 skipped, 147 deselected in 34.78s**, no warnings. This is the final-tree offline gate. Skipped and excluded tests remain unverified; no live model or service acceptance is implied. The earlier Task 6 full run exposed one disabled-hook fixture incompatibility; it was fixed, covered by focused tests, and included in this final passing run.

Final Ruff command used all changed Python paths relative to the source baseline and exited 0 (107 files, `All checks passed!`):

```sh
python3 - <<'CHECK_PY'
import subprocess
paths = subprocess.check_output(
    ['git', 'diff', '--name-only', '--diff-filter=ACMR', 'f875fb0', 'HEAD', '--', '*.py'],
    text=True,
).splitlines()
subprocess.run(
    ['/Users/steven/miniconda3/envs/agentic-rag/bin/python', '-m', 'ruff', 'check', *paths],
    check=True,
)
CHECK_PY

git diff --check f875fb0 HEAD
git diff --check
```

The final fix wave also ran this focused command (68 passed, no warnings) before its independent scoped review:

```sh
RAG_EVAL_PYTHON=/Users/steven/.codex/worktrees/36dd/AgenticRAG/.venv/bin/python .venv/bin/python -m pytest -q tests/unit/runtime/test_query_worker_entrypoint.py tests/unit/runtime/test_query_composition.py tests/unit/runtime/test_context_composition.py tests/unit/persistence/test_elasticsearch_staging.py tests/unit/evals/test_context_ablation.py
```

Review closure:

| Scope | Commit range | Final disposition |
|---|---|---|
| Task 1 contracts/rendering | `6d8bb83..737b488` | Accepted after one fix round |
| Task 2 retrieval/rerank | `737b488..2d5d16f` | Accepted |
| Task 3 ingestion/budgets | `2d5d16f..a974396` | Accepted after one fix round |
| Task 4 query scope | `a974396..3f4bc17` | Accepted after two fix rounds |
| Task 5 evidence/disclosure | `3f4bc17..4af9dc8` | Accepted after three fix rounds |
| Task 6 composition/rebuild/release | `4af9dc8..95c4352` | Accepted after one fix round |
| Whole branch | `f875fb0..95c4352` | One Important and two Minor findings assigned to final fix wave |
| Final fix wave | `95c4352..7ca5388` | All three addressed; no new findings |

The final Important finding was controlled Query Worker startup repairing an ES alias before readiness. Controlled startup now uses read-only target validation; the actual run tests assert no alias/index writes even when migration 0012 or the alias is missing. The two Minor findings were missing D/E index identity passing equality and undocumented permanent concurrency pinning. D/E now requires valid equal SHA-256 identities; the guide explicitly records that no limiter resize exists.

No live migration, index rebuild/switch, model call/download, service restart, deployment, push or merge was performed. The existing externally managed detached worktree is preserved. Profile/directory/coverage providers and the signed maintenance authority remain explicit deployment integrations. Real MySQL/ES behavior, deployed tokenizers, provider costs, semantic accuracy and retrieval improvements require separately authorized real-service evaluation.

## Rulings I made

Chronological decisions made during implementation, including superseded decisions. The costs below are retained for review; the exact original ledger wording follows the table. Decisions 17 and 18 record a rejected intermediate implementation and its replacement, not two alternative active policies.

| # | Decision | Cost or consequence |
|---|---|---|
| 1 | 按已确认设计直接开发，不重复申请实施许可。 | 若范围理解有偏差，需要返工。 |
| 2 | 沿用隔离的托管worktree，保留本地提交。 | 集成和部署仍需后续操作。 |
| 3 | 拆成六项顺序实现，每项独立审查。 | 串行审查增加交付耗时。 |
| 4 | 搜索表示草稿可暂缺context_ref，发布前必须补齐。 | 调用方若绕过发布校验会破坏来源合同。 |
| 5 | rerank分窗始终保留完整范围前缀。 | 超长且无法完整覆盖的查询会降级。 |
| 6 | 模型只尝试一次，SDK不重试，未知消费保留预留。 | 少一些自动修复成功率，费用统计更保守。 |
| 7 | 发布按Child ID与文本哈希回读验证。 | 增加ES读回成本。 |
| 8 | 生产模型并发使用SQL跨Worker限制。 | 需要新增持久化容量与故障恢复依赖。 |
| 9 | 来源闭包策略升级为whole-block-v2。 | 旧策略产物需要重新构建。 |
| 10 | 显式范围不重复拼接；范围无法安全分窗时保留RRF结果。 | 部分长查询无法使用rerank。 |
| 11 | 完整保存任务目标继承链，统一经可信目录编译。 | 增加冻结状态与范围冲突处理。 |
| 12 | 来源可用与语义支持分开，原文候选仍须语义审核。 | 增加审核成本，并可能产生明确证据缺口。 |
| 13 | 来源校验覆盖Supervisor冻结、流式、重放与公开展示。 | 增加这些路径的回源开销。 |
| 14 | 从权威Parent与真实Child坐标重建裁剪文本。 | 新增裁剪合同；不一致内容会被拒绝。 |
| 15 | 跨Parent支持保留独立原文摘录。 | 摘录会占用四来源及25%证据预算。 |
| 16 | 新格式无context_ref的Parent也验证原文；历史schema1保留兼容。 | 历史记录不获得追溯性认证保证。 |
| 17 | 曾采用逐项版本标记区分混合来源，随后由下一项替代。 | 该尝试引入降级绕过，已修复并增加回归。 |
| 18 | 新schema2快照每条来源都必须携带可校验的原文证明。 | 早期未证明的混合快照需要重新准备。 |
| 19 | 旧格式有真实定位时也按坐标裁剪；不可证明的历史裁剪拒绝提升。 | 部分历史来源需要重新准备。 |
| 20 | 独立重建任务表与真实租约，保留旧上传表兼容。 | 维护第二套有限任务生命周期。 |
| 21 | 维护门禁按open、draining、frozen推进。 | 维护期间新请求和删除需重试。 |
| 22 | 首次启用与切换依赖外部可信部署证明和逐实例就绪。 | 上线需要接入可信签发方；签名不等于实际观测进程。 |
| 23 | 发布执行不按超时抢占；未知ES写入须签名静止证明后恢复。 | 异常窗口可能需要人工协同恢复。 |
| 24 | 0012在首次启用前添加执行保护；已armed的0011拒绝自动升级。 | 该未发布中间状态需要单独恢复方案。 |

<details>
<summary>Original chronological ledger rulings</summary>

1. Ruling: proceed directly from explicit implementation authorization using reviewed design; no additional plan permission question — user already authorized work, developer instructions prioritize completing it — implementation choices remain reviewable.

2. Ruling: native managed detached worktree is already isolated; local commits succeed; no branch creation/push/deploy needed.

3. Ruling: six sequential implementation tasks with independent task reviews; controller owns integration planning and verification, no parallel implementers.

4. Ruling: SearchRepresentation can be a pre-persistence draft with context_ref=None; an explicit validate_for_indexing/require_persisted gate must require the reference before context-v1 publication. with_context_ref preserves text/ID. This avoids circular artifact identity without weakening published-record validation.

5. Ruling: Reranker gains protected_query_context keyword. Full query must begin with exact protected context plus two newlines; only remaining body may be windowed, and each window retains the complete prefix. Task 4 QueryPlan compiler owns this deterministic composition. Missing/full query coverage cannot silently truncate to two windows.

6. Ruling: Task3 uses max_provider_attempts=1 and SDK retries=0; zero schema repair/network retry is within spec's at-most-one allowance. Test no hidden extra dispatch. Conservative request reservations retain failed/unknown spend. Cost: fewer repaired successes, explicit safe structural fallback.

7. Ruling: Task3 adds ES child ID/hash readback needed to enforce manifest-v2 publication; old v1 invocation path remains compatible. Task6 still owns multi-build registry and release orchestration. Counts alone cannot meet accepted spec.

8. Ruling: Task3 process-local semaphore is only local safety. Task6 must wire a durable cross-Worker model resource limiter at production ContextModelPort boundary (reuse existing resource leases where compatible); enabled production without global limiter fails configuration. This preserves clear Task3 core/Task6 composition ownership while satisfying spec global concurrency.

9. Ruling: bump default ContextBuildConfig.selection_version whole-block-v1→whole-block-v2 in the fix because accepted dependency-closure semantics changed. New default fingerprint prevents ready-result reuse from pre-fix policy. Existing explicit old configuration/produced artifacts require a new build; no live artifact migration is being performed.

10. Ruling: preserve no-duplicate explicit query qualifiers. Task4 adds Reranker allow_query_windows=True default and frozen QueryPlan allow_query_windows=False when explicit qualifiers remain outside protected prefix. Fit full pairs still score; overlong pairs explicitly degrade to existing RRF reason rather than dropping qualifiers. Conditional keyword preserves legacy fakes/calls. Cost: fewer long explicit queries can rerank; avoids repetitive normal inputs and aligns approved no-duplicate semantics.

11. Ruling: round2 stores immutable TaskBusinessObjective(text,source_ref) lineage on QueryBusinessContext; pure for_task appends lineage instead of committing substring-only projection. Common async compile resolves all objectives with its configured directory and validates/narrows sequentially from frozen assignment before recall. Preserve source/profile, legitimate broad-parent→narrow-child paths and reject narrow-parent→new-target child. This centralizes authority without injecting process objects into checkpoints.

12. Ruling: Task5 EvidenceRequirement status resolved/missing/conflicting denotes source dependency availability; independent semantic_status=pending prevents treating available text/literal matches as factual support. Self-contained original body may supply a pending source candidate if scope literal exists, then existing grader must assess applicability. No literal-based semantic acceptance; conflicts/counterevidence/missing cross-Parent dependencies remain gaps. Task5 will test actual applicability vs customer-mention same literal. SQL authority uses active version/generation/current durable manifest membership; Task6 extends membership port for multi-build registry.

13. Ruling: Task5 also updates actual Supervisor evidence disclosure consumers orchestration/finalization.py, authorization.py, streaming.py and api/supervisor_stream.py. Current individual-item freeze cropping and parent-only source_snapshot would drop closure/new source metadata. Reuse centralized grouping/identity projection; new contextual Parent and excerpt disclosure use same authority recheck, not mere doc/version existence. Task6 injects context_sources into CoordinationStore and API-side stores. Existing parent checks cannot weaken.

14. Ruling: Task5 I1 adds bounded parent_excerpt_length and reconstructs cropped text from authoritative Parent + validated member Child anchor through shared deterministic crop helper, then binds hash/locator/heading/retained support. Authorizers return verified_content and real heading/pages; public stored full/truncated prefix and flags must match verified projection. Frozen locator is not its own authority.

15. Ruling: Task5 I2 keeps a separately authorized canonical excerpt for cross-Parent support; local Parent dedup only carries its own requirement relations, no global source-item suppression. Independent support still counts under4-source/25% caps. This avoids ambiguous body/support membership fields.

16. Ruling: Task5 I1 also binds new schema2/private-provenance Parents without context_refs to scoped ParentRepository originals; historical schema1 without private provenance preserves compatibility. Crop anchors must use verified source coordinates rather than first equal child text; include repeated-text and legitimate end-crop coverage. This closes the new-type bypass while preserving existing source contracts.

17. Ruling: Task5 N1 adds private per-item AnswerSource.evidence_schema_version (1/2/None historical) emitted from the EvidenceItem. Mixed schema2 snapshots may carry explicitly legacy items; genuine schema2 items require matching private provenance, missing both marker/provenance in schema2 fails, and old schema2 provenance without marker is revalidated. Marker/provenance mismatch fails; public DTO omits marker. This distinguishes supported formats without granting authority; historical schema1 compatibility remains the declared limit.

18. Ruling: replace marker-selected weaker treatment inside new schema2 snapshots by authenticated disclosure proof for every item; normalize legacy Parent items in mixed snapshots to original-body hash/crop/locator proof, preserve citation IDs. All schema2 items require proof; historical schema1 remains compatible; genuine markerless older schema2 can revalidate. A mutable marker/allowlist cannot be authority. This supersedes the unsafe part of round2 ruling; cost is stricter validation of newly emitted legacy/mixed source views and possible crop-contract work, not a general historical migration.

19. Ruling: Task5 round3 normalization uses actual legacy item content hash/length plus immutable identity/locator/heading and preserves citation ID. For new legacy packing with valid AstLocator, reuse coordinate crop helper so repeated text is correctly located and normal mixed sources remain provable. Historical opaque crops without sufficient coordinates fail closed on promotion and require reprepare; no first-text-match heuristic may replace original coordinate proof.

20. Ruling: Task6 permits focused modules persistence/context_release.py, persistence/context_registry.py, ingestion/contextualization/builds.py and release.py, runtime/context_composition.py, and limiter extension. Use separate retrieval_rebuild_jobs rather than altering ingestion_jobs because existing select(table) would break flags-off unmigrated startup. Explicit server-owned rebuild claim extends ContextBuildStore transactional reservation/accept fence against real rebuild table lease; no fabricated upload claim/external-only lease assertion. Dedicated worker/CLI queued/expired recovery executes dual-build requirements; completion is build state, not active source. Release drain counts both job families; deletion revokes rebuilds. Cost: second focused job lifecycle, justified by isolation and backward schema compatibility.

21. Ruling: Task6 gate uses open→draining→frozen. Draining blocks new admissions but existing queued work may claim/heartbeat/finalize under old configuration until SQL queued/running/retry_wait work drains; frozen then blocks claims/publication. This prevents drain deadlock. Rejected deletes must remain explicit maintenance rejection; committed revocation/deletion invalidates all builds. Global armed release cannot be bypassed merely by process-local context flags=false; implementer to specify independent global guard and validated all-role deployment/readiness boundary.

22. Ruling: first install/arm uses externally trusted maintenance deployment attestation, not a new single-host PID-only restriction. Add minimal MaintenanceReceiptVerifier + concrete Ed25519 verifier (existing cryptography dependency); production trusts configured public key/issuer, CLI never self-signs stop proofs. Signed payload binds full role/instance inventory, old deployment stop evidence references, new deployment/config hash, from/to generation, expiry and one-use nonce; durable journal consumes nonce/hash. Actual post-validation instance receipts must exactly cover expected inventory before reopen. Absent trusted authority blocks execute/arm safely; dry-run reports dependency. Physical stopping and inventory completeness are trust promises of the configured deployment manager, not facts observable from DB/signature. Independent context_release_enabled governs legacy/context deployments. Cost: external deployment-authority integration is required for live activation, documented rather than faked.

23. Ruling: Task6 I1 uses durable exclusive execution token plus effect-uncertain latch, with no timed takeover. SQL expiry cannot fence an already dispatched ES request. If crash/cancel/unknown failure leaves external writes uncertain, block resume/rollback/open until a trusted Ed25519 recovery receipt proves old executor stopped and related ES writes settled/cancelled. Bind receipt to exact journal, execution token/fence, current maintenance receipt, target/config, expiry/nonce; consume once only against current owner. CLI cannot self-certify. Normal known-complete boundaries release safely; every continuation/DB stage/ES dispatch/reopen still checks active journal/token/receipt and monotonic state. Cost: external quiescence proof required only for ambiguous-write recovery; avoids unsafe automatic lease expiry and a larger ES-wide fencing architecture. Implementer must document/test busy/uncertain/recovery/replay/late-continuation paths.

24. Ruling: add0012 after static0011 for release execution fence. Require it before initial arm. Already-armed pre-fence0011 cannot safely reconstruct unknown old execution ownership; migration refuses before anyDDL and leaves data unchanged. No automatic migration of that unapproved/unreleased live state is in this task. Test fresh sequential migrations and pre-DDL refusal; enabled startup readiness/guide require0012. Document actual recovery CLI and execution-token lookup. No userDB migration.

</details>
