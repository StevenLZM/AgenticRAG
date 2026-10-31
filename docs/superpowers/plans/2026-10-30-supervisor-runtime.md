# Supervisor Retrieval Analysis Runtime Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [x]`) syntax for tracking.

**Goal:** Implement the approved supervisor design with isolated agents, durable dispatch/reception, grouped evidence preparation, a direct fast path, analysis-owned calculations and persisted answer streaming.

**Architecture:** Add a versioned SQL-backed coordination runtime alongside the legacy QueryGraph. Independent workers claim bounded stages; every state transition, result reference, budget update and notification intent commits atomically. Reuse the existing retrieval, evidence, authorization, model and Redis adapters through explicit ports.

**Tech Stack:** Python 3.11, Pydantic 2, SQLAlchemy async/MySQL, Redis Streams, LangGraph, FastAPI, existing browser client.

**Spec:** ../specs/2026-10-30-supervisor-retrieval-analysis-design.md

## Global Constraints

- Retrieval Agent pipeline ends at rerank. Evidence preparation and grading are separate stages.
- Task dispatch requires objective/criteria, output contract, tool/source guidance and task boundary.
- A group has at least one required member; no member may require this same group's grade before it executes.
- Fast fact lookup does not create Retrieval Agent tasks or call the supervisor planner.
- Business mathematical calculations are accessible only to Analysis Report Agent.
- Task contexts, identities, source permissions and budgets are server owned and versioned.
- SQL is the recovery source of truth; Redis is a replayable notification transport.
- Use absolute UTC deadlines, leases and fencing. Restart, retries and route promotion do not reset budgets.
- A frozen member snapshot and evidence bundle are immutable; old results never unlock newer dependencies.
- Streaming begins only after accepted inputs are frozen. Published output is not labelled pre-audited.
- Preserve legacy Runs and output contracts. Select the new version at Run creation; do not reinterpret old checkpoints.
- No changes to live databases, active indexes, running services or paid model evaluation in this development task.
- Work in the existing isolated worktree. Keep all changes reviewable in the working tree; do not push or merge.

## Review Focus

- Acyclic task dependencies can still deadlock against a group join: reject that combination before dispatch (Task 1).
- Result arrival racing join timeout or a stale worker return must not create two snapshots or overwrite accepted evidence (Task 2).
- Role/tool restrictions and source selectors must survive provider-generated invalid actions (Tasks 1 and 3).
- A fast-path promotion or supervisor crash must not double-debit or restart successful retrieval work (Tasks 2 and 4).
- A provider stream that yields a prefix then fails must not be retried into that prefix; cancellation/reconnect must preserve the epoch (Task 5).

## Verification Environment

Python: `/Users/steven/LzmWorkSpace/AgenticRAG/var/eval-venv-ragas042/bin/python` (existing environment including optional evaluation dependencies).
Set `RAG_EVAL_PYTHON` to that same path for subprocess evaluation CLI tests.
Run tests from the repository root with `python -m pytest`; live-model tests stay opt-in.
Baseline command: `python -m pytest -m 'not integration and not e2e and not live_model' -q`.
Each task records failing-then-passing targeted tests in the execution ledger. Full offline suite and static checks run after integration; real MySQL/Redis tests use disposable resources only.

## File Map

- `retrieval/models.py`, `retrieval/graph.py`: expose ranked candidates without hydration; retain the legacy retrieval adapter.
- `query/evidence_builder.py`: preserve target associations across deduplication and target-aware packing.
- `orchestration/contracts.py`, `validation.py`: task, group, result, budget, decision and stream contracts.
- `orchestration/schema.py`, `store.py`: SQL coordination state, immutable results, leased work, Inbox/Outbox and chunks.
- `orchestration/evidence.py`, `agents.py`, `supervisor.py`: independently callable evidence/agent/model stages.
- `orchestration/coordinator.py`, `worker.py`, `composition.py`: atomic transitions, queue transport and deployment-owned wiring.
- `orchestration/streaming.py`: input freeze, generation/quality lifecycle and persisted publication.
- `alembic/versions/0009_supervisor_runtime.py`: additive tables and version-safe migration.
- Runtime settings/snapshot, Run repository/API and worker entrypoints: dispatch new Runs to their selected executor.
- API/static chat modules: versioned incremental output and quality status.

## Task 1: Retrieval boundary and validated task/evidence contracts

**Files:** retrieval models/graph; query evidence builder; new orchestration contracts/validation/evidence; tests/unit/orchestration/test_contracts.py, test_evidence.py; tests/unit/retrieval/test_graph.py.

**Interfaces:**
- Produces `RetrievalService.retrieve_candidates(request, scope, snapshot) -> RankedCandidates`.
- Produces `TaskSpec`, `RetrievalGroupSpec`, `DispatchPlan`, `CandidateResult`, `PreparedBundle`, `GradeResult`, `AnalysisResult` and `validate_dispatch(plan)`.
- Produces `EvidencePreparer.prepare(candidates, criteria, scope, snapshot) -> PreparedBundle`; hydration occurs here.

- [x] Write behavioral tests: missing mandatory dispatch parts/role tool mismatch are rejected; optional-only and join dependency cycles are rejected; task/criterion keys are unique and dependencies resolve.
- [x] Write retrieval test asserting the candidate method returns ranked hits and never hydrates parents; retain the legacy method's output.
- [x] Write preparation tests for cross-user input rejection, dedup preserving both task/criterion links, independent query score ordering, and bounded target coverage.
- [x] Run these tests and observe missing behavior failures.
- [x] Implement the typed contracts, validators and retrieval/preparation boundaries using existing filter, rerank, ParentFetcher and EvidenceBuilder code.
- [x] Run the new tests plus existing retrieval/evidence-builder tests. Expected: all pass.

## Task 2: Durable coordination store and recovery invariants

**Files:** orchestration schema/store; migration 0009; tests/unit/orchestration/test_store.py; tests/integration/orchestration/test_store_mysql.py.

**Interfaces:**
- `CoordinationStore(session_factory)` opens short transactions; `initialize(run)`, `claim(work_id, owner, lease_seconds)`, `heartbeat(claim)`, `submit_result(claim, result)`.
- `submit_plan(run_id, decision_id, expected_version, plan)` persists validated tasks/groups/notifications once.
- `collect_group(group_id)` freezes exact members at all-required settlement or join deadline; `apply_feedback(run_id)` reduces Inbox without losing semantic feedback.
- `recover()` reestablishes dispatch, join, lease and Inbox wake responsibilities.
- Worker-facing `WorkClaim` contains exact input/version/lease and server scope, never broker-provided authority.

- [x] Write tests using a file SQL database and independent sessions for idempotent decisions, lease theft/stale results, atomic result/outbox, deadline preservation and budget exhaustion.
- [x] Write tests for one successful member plus timeout, duplicate/out-of-order notification, same-version single snapshot, no early analysis, and retained results on continuation.
- [x] Observe failures, then implement additive tables and transactions. Run row serialization owns coordination; large immutable stage results are separate records with hashes.
- [x] Add MySQL integration tests for concurrent claims/commit ordering and migration upgrade/downgrade on disposable schema.
- [x] Run targeted store tests. Expected: persistence survives constructing a new store and old leases cannot write.

## Task 3: Private-context agents, independent grading and supervisor decisions

**Files:** orchestration agents/evidence/supervisor; role prompts; tests/unit/orchestration/test_agents.py, test_supervisor.py.

**Interfaces:**
- `RetrievalAgent.execute(task, context, execution) -> RetrievalResult` plans bounded queries and returns candidates/findings with context.
- `AnalysisAgent.execute(task, bundles, context, execution) -> AnalysisResult` performs validated calculation actions and records operand provenance.
- `GroupGrader.grade(bundle, criteria, execution) -> GradeResult` validates supported evidence IDs and criterion coverage.
- `Supervisor.decide(state, execution) -> SupervisorDecision` produces typed submit/continue/analyze/answer/clarify actions without raw child conversations.

- [x] Write tests for context isolation/resume, source restrictions, planner attempts to call calculator from retrieval, missing mathematical operands, unsupported citations and bounded no-progress.
- [x] Write tests proving evidence is delivered to the grader after preparation and analysis only accepts the contracted evidence versions.
- [x] Observe failures, implement structured model requests and deterministic action/tool enforcement.
- [x] Run all agent and contract tests. Expected: malicious/out-of-role actions fail closed and a missing-source calculation yields needs_input.

## Task 4: Worker transport, fast route and application integration

**Files:** orchestration coordinator/worker/composition; config/runtime snapshot; run repository/bootstrap; scripts/run_agent_worker.py; worker compatibility; tests/unit/orchestration/test_worker.py, test_runtime.py.

**Interfaces:**
- `AgentWorker.run_one()` / `run_forever(stop_event)` claim independent work with heartbeats and explicit ACK responsibility.
- `Coordinator.advance(run_id)` is one bounded decision/transition, not a permanently waiting graph.
- New `orchestration_version=supervisor-v1` Runs use a dedicated stream and SQL stage recovery; legacy defaults remain readable.
- Fast route uses the same candidate/prepare/grade contracts with `origin=fast_pipeline`, then deterministic acceptance or same-Run promotion.

- [x] Write end-to-end deterministic tests: fast lookup does not invoke planner/child agent; complex plan runs two workers concurrently and fans in; grade gaps resume only the missing task; analysis needs_input returns through retrieval.
- [x] Write recovery tests for supervisor disappearance after dispatch, candidates and grading, broker notification duplication, cancelled Runs and unsupported Run versions.
- [x] Observe failures, implement coordinator transitions, outbox/listener/recovery transport and role-specific deployment wiring.
- [x] Expose worker role/instance identity through CLI and validate new schema readiness; never open shared SQLite for new coordination.
- [x] Run worker/runtime tests and legacy Run/worker/config tests. Expected: both selected versions execute through their own boundaries.

## Task 5: Real model streaming and versioned public output

**Files:** runtime/model_gateway.py; orchestration streaming; API query endpoints/static client; tests/unit/orchestration/test_streaming.py; gateway/API/client tests.

**Interfaces:**
- `ModelGateway.stream(call) -> AsyncIterator[ModelStreamEvent]`: content deltas and terminal usage; no whole-response buffering or retry after emitted content.
- `AnswerPublisher` persists epoch/seq chunks under current lease before exposing them.
- Versioned public answer includes evidence/analysis assessments, pending/passed/failed/unavailable faithfulness and stream status.
- SSE cursor replay reads persisted chunks, respects user scope/cancellation, and closes only after bounded quality finalization.

- [x] Write fake-provider tests proving the first delta arrives before provider completion and prefix failure never retries into the same epoch.
- [x] Write tests for cancelled/stale owner chunk rejection, reconnect ordering/dedup, finalizing recovery and absent completion on EOF.
- [x] Write API/client tests for partial text display, quality updates, terminal statuses and legacy audited answers.
- [x] Observe failures, implement gateway streaming, frozen-input answer/quality stages, scoped endpoints and browser rendering.
- [x] Run stream, API and browser contract tests. Expected: raw planner/grader tokens never reach the answer channel.

## Task 6: Integration, operational documentation and final review

**Files:** integration/e2e orchestration tests; docs/local-operations.md; approved spec implementation status; this plan/ledger.

- [x] Exercise independent processes with disposable Redis and file SQL: kill/restart coordinator/worker, preserve budgets and reconstruct one join. Unit tests cover stale return, partial join and stream reconnect.
- [x] Execute these integration invariants against disposable MySQL. Completed in the 2026-10-31 follow-up using existing local MySQL; see the live acceptance record.
- [ ] Cross-host failure/load acceptance remains a separate deployment prerequisite, outside this local-process acceptance scope.
- [x] Run full offline tests, ruff and relevant mypy checks; compare failures to the recorded baseline rather than silently omitting them.
- [x] Document migration, role startup, feature selection, recovery and rollback. Record which tests need opt-in/live infrastructure.
- [x] Obtain one fresh whole-change review, fix material findings with failing-then-passing regression tests, and rerun affected checks.
- [x] Record completed scope and actual verification evidence; do not claim live model quality, production rollout or multi-host availability without those tests.

## 2026-10-30 completion record (offline stage)

Core implementation is complete in the working tree; no commit, deployment, live migration, index change or paid model evaluation was performed. The fresh review found delayed analysis reception and capacity/admission gaps; each was fixed and covered by offline regressions. See [operations and verification](../../supervisor-runtime.md) for measured results and remaining infrastructure validation. The initial design document and pointer were preserved.

## 2026-10-31 真实环境验收补充

已按后续授权完成真实 MySQL、Qwen、DeepSeek、ES 和 reranker 的隔离验收及运行时修复。历史离线计数与限制保留为当时记录；当前证据以[真实链路验收记录](../../supervisor-live-validation.md)为准。业务入口仍默认关闭。本次提交包含运行时、验收修复、Analysis / 最终答案单次 100,000 token 上限及配套测试和文档；原始模型日志、数据库备份与本地配置保留在 Git 忽略的本机证据目录。
