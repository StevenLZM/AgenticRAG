# Agentic RAG 开发进度快照

> 快照日期：2026-08-14
>
> 当前状态：路线图 Phase 1–5 已完成；Query Outbox/Worker、真实 Query E2E、Mem0 生产组合、真实评测和降级遥测的后续加固也已完成，当前实现位于独立 worktree，尚未合并到 `main`。
>
> 本文档是恢复开发时的首要状态入口；详细设计、接口约束和任务拆分以文末权威文档为准。

## 1. 当前开发位置

- 仓库根目录：`/Users/steven/LzmWorkSpace/AgenticRAG`
- 实现 worktree：`/Users/steven/LzmWorkSpace/AgenticRAG/.worktrees/agentic-rag-implementation`
- 实现分支：`sdd-agentic-rag-implementation`
- 分支基线：`main@1e02fbb`
- Phase 4 最终代码：`c3cf053 fix: preserve business terminal query outcomes`
- 当前 follow-up 最终代码：`67d492a feat: expose explicit degradation and circuit telemetry`（发布文档/门禁提交待本次完成）
- Conda 环境：`agentic-rag`

`main` 尚未合并当前实现。继续开发前先进入实现 worktree，并确认工作树干净：

```bash
cd /Users/steven/LzmWorkSpace/AgenticRAG/.worktrees/agentic-rag-implementation
git branch --show-current
git status --short
git log -5 --oneline
```

## 2. 总体进度

原始路线图共 32 个任务（Phase 1/2/3 各 6 个，Phase 4 为 9 个，Phase 5 为 5 个）；其后新增的 Query Runtime/Mem0/真实评测 follow-up 共 7 个任务。

| 阶段 | 状态 | 任务数 | 交付摘要 |
|---|---:|---:|---|
| Phase 1：Foundation and Persistence | 完成 | 6/6 | 领域契约、MySQL、Outbox/Redis、Checkpoint、Artifact Store、Bootstrap/Health |
| Phase 2：Document Ingestion | 完成 | 6/6 | 安全上传、Docling AST、Parent/Child、Embedding、发布/对账、可恢复 Worker |
| Phase 3：Retrieval and Evidence | 完成 | 6/6 | Dense/BM25、RRF/Rerank、Parent 聚合、降级检索图、EvidenceBuilder |
| Phase 4：Query and Agent Runtime | 完成 | 9/9 | ModelGateway、Memory、Fast/Research、Subagent、审计、QueryGraph、Worker、API |
| Phase 5：Evaluation and Operations | 完成 | 5/5 | Trace/在线指标、确定性评测、离线报告、对抗/负载/恢复演练、备份恢复/就绪检查/最终验收 |

**总体完成：原始路线图 `32/32`，follow-up `7/7`；实现任务 `100%`，当前只剩分支合并与生产鉴权/RBAC 审批。**

## 3. 已实现能力

### 3.1 Phase 1–2：基础设施与文档摄取

- Python 3.11、Pydantic Settings、pytest/Ruff/mypy 工具链，以及共享领域模型、运行时版本快照和 UUIDv7 标识。
- MySQL 持久化、Alembic 迁移、事务 Outbox、Redis Streams、SQLite Checkpoint、Artifact Store 和 FastAPI Bootstrap/Health。
- 上传安全门、MIME/大小/压缩炸弹检查、Docling Fragment/Canonical AST、结构优先 Parent-Child Chunking。
- Qwen embedding 适配、Elasticsearch Child/向量索引、Manifest 驱动发布、Publisher/Reconciler、删除 tombstone、Claim/Lease/Heartbeat 和 DLQ 恢复。

Phase 2 最终里程碑为 `55a132e`；Phase 1/2 的详细任务提交和约束见对应阶段计划。

### 3.2 Phase 3：检索与证据

- 服务端 `user_id`、scope、版本和 date/search selectors 过滤；Dense 与 BM25 适配器并发召回，单 lane 降级和双失败 fail-closed。
- 固定 RRF（`k=60`）融合、Cross-Encoder rerank、Child 去重、Parent 聚合/抓取及 provenance 保留。
- 确定性 Retrieval Graph，输入错误/取消透传，检索结果携带 snapshot/index generation。
- EvidenceBuilder 实现 scope/provenance 校验、DataEnvelope、coverage-first 选择、token 上限、普通检索每文档上限及可信单文档 selector 例外。

Phase 3 最终代码里程碑为 `c0d1305`，独立复审通过。

### 3.3 Phase 4：Query 与 Agent Runtime

| Task | 最终提交 | 交付 |
|---|---|---|
| 1 | `a4c999e` | OpenAI-compatible Structured ModelGateway、单次 repair/retry、usage/prompt hash 和不可变 Runtime Snapshot |
| 2 | `36e5e9d` | scoped Mem0 边界、轻量事实抽取、assistant confirmation 校验、tombstone 删除/重试/对账、租户 fail-closed |
| 3 | `55f253d` | JSON-safe QueryState、Memory-aware Router、Fast RAG、取消和非有限 JSON 值保护 |
| 4 | `31ad0b5` | 动态 Todo、calculator/research tools、受限 ResearchAgentLoop 和严格 action schema repair |
| 5 | `bfc96ea` | 有界并发 Subagent、隔离子状态、超时/取消清理、确定性 evidence reducer |
| 6 | `ec4920f` | Generation、Faithfulness/Citation 强制审计、授权/version resolver、单次修订闭环 |
| 7 | `11535b0` | 固定 QueryGraph、checkpoint namespace、真实 EvidenceBuilder 路径、可重放幂等事件 |
| 8 | `38ad905..7ecfb1f` | 原子 Run/Outbox、Query Worker lease/heartbeat/reclaim、批量投递清空、重试/DLQ/取消 |
| 9 | `8f44b93..c3cf053` | Query Run、SSE/cursor、sync wrapper、cancel、Memory、Feedback API；业务终止结果保留为 COMPLETED |

Phase 4 当前具备的关键边界：

- QueryGraph 固定编排 Fast/Research、EvidenceBuilder、Generation、Faithfulness、Citation 和 Finalize；缺证据、审计失败、无法回答等结果 fail-closed。
- Query Run 持久化 `question` 与最终 `answer`（迁移 `0006`、`0007`），Worker 在崩溃后可从 Run 重建状态；Redis reclaim 与 fresh delivery 按批次全部处理，避免消息滞留 PEL。
- 终止原因 `clarify`、`refuse`、`cannot_answer`、`audit_failed`、`research_round_limit`、`research_action_invalid` 等映射为结构化业务完成；未知或非字符串原因仍重试并最终进入失败/DLQ。
- API 覆盖 `/v1/query-runs`、SSE 重连、取消、同步 `/v1/query`、`/v1/memories` 和 `/v1/feedback`；未知 SSE 事件摘要脱敏，未配置 Memory provider 时读写均 fail-closed。

### 3.4 Phase 5：评测与运维（已完成 Task 1–5）

- Task 1 最终提交为 `8729171`（基线实现 `053e8e4`，后续安全/接入修复至 `b83b716`、生命周期与成本语义修复 `ecf2274`、队列重领取去重 `8729171`）。
- `TraceRecorder` 提供本地 OpenTelemetry-compatible 嵌套 span，绑定 `run_id`、快照和 parent/span 层级；取消、异常和跨 Run/Recorder 上下文均 fail-closed。
- `AgentEventEmitter` 仅持久化严格 allowlist 的有限枚举/数值字段，拒绝 prompt、隐藏推理、原始 Tool payload、凭据和不安全标识；事件 payload 使用内容寻址 Artifact，并校验 URI/hash 完整性。
- QueryGraph、QueryWorker、ModelGateway、Tool/Memory/Retrieval/Rerank/Audit 边界已接入共享快照遥测；Graph/Worker 生命周期事件使用稳定 event key，Redis 重领取不会重复队列指标，lease 丢失不会伪造终态事件。
- `MetricsProjector` 支持 cursor/reducer 跨页累计 queue/run/node latency、retrieval、token/call、citation、repair、degraded、feedback、outbox/lease/reconciler 和业务终态指标。无可信定价来源时 `estimated_cost_status=unavailable`，不把零值伪装成成本估算；只有带安全成本字段的观察事件才标记 `observed`。

Task 2（`7aac8d8`，基线实现 `fff7ea5`）已完成：

- 新增纯离线 `evals` 包和严格 `EvaluationCase`、`IngestionFidelityCase`、`SecurityCase` 模型；拒绝未知字段、非 JSON/非有限值、敏感/provider 字段、重复 ID、路径穿越和非严格类型转换。
- Recall@k、MRR、NDCG@k 使用 binary relevance，先按原始排名位置截取 top-k，再在窗口内去重；重复事件按稳定 `event_key` 去重，并按 `user_id` 与 `runtime_config_snapshot_id` fail-closed 过滤。
- 固定数据集已纳入 wheel：baseline 24（8 single-hop、8 multi-hop、4 scanned-PDF、4 Excel）、ingestion fidelity 13、security 12；CLI 为 `python -m evals.validate_datasets evals/datasets`。

Task 3（`15a9150`，基线实现 `6e603a4`）已完成：

- `EvalRunner` 通过注入 Query API/Graph client 执行离线案例，结果模型严格校验 snapshot、answer、evidence、route、events 引用和 deterministic/Ragas 指标；不构造或改写检索/审计逻辑。
- results JSONL 与 summary JSON 使用临时文件、fsync 和 `os.replace` 原子写入；完整/同 snapshot 行可断点续跑，损坏、重复或旧 snapshot 行隔离并重算。
- `RagasAdapter` 允许显式离线 backend；未安装/未配置时输出 `status=unavailable` 空指标，不伪造分数。报告默认拒绝混合 snapshot，仅接受命名 baseline 映射进行比较。
- CLI `python -m evals.run --dataset ... --output ... [--limit N]` 提供确定性 fixture smoke，产出 results/summary 且不访问外部服务。

Task 4（`2e99e00..1947806`，基线实现 `2e99e00`）已完成并通过独立复审：

- 新增安全回归 E2E：验证文档 prompt injection/filter override、隐藏 Unicode、Evidence ID 伪造、跨用户 evidence/memory/checkpoint/event 隔离、Mem0 不可用降级，以及 Artifact/Event payload 中的原始 prompt/tool/hidden reasoning 脱敏；安全扫描覆盖 durable payload 与 event type/node/summary。
- 新增背压 E2E：通过真实 `ConcurrencyManager` 观察 run/LLM/reranker 最大并发，独立压测至少 8 个 LLM slot，记录 queue wait 时间，并使用可注入 API/Worker health probes 验证活性。
- 新增 `scripts/run_recovery_drill.py`：七个固定故障场景使用隔离 in-memory fakes 和现有 `AgentEvent`、`UserScope`、`MemoryServiceImpl`、`LocalArtifactStore` 边界；覆盖 SSE 重连、Query/Ingestion 重放、Outbox Redis 故障、ES 激活中断、Artifact quarantine、Mem0 scope/outage。报告包含 scenario invariants、replay/duplicate/leak/quarantine 计数，使用 fsync+replace 原子写入，失败或泄漏时返回非零。
- Task 4 最终验证：E2E 10 passed；全量 importlib 测试 530 passed、38 skipped；Ruff、scoped mypy、diff-check 通过；直接脚本与 module CLI 均返回 0，报告确定性且无临时文件残留。

Task 5（Backup/Restore/Readiness/Final Acceptance）已完成：

- `scripts/backup_local.py` 对 SQLite checkpoint、Artifact、可选 MySQL dump、Elasticsearch generation/alias/template/document 以及显式 Redis key prefix 做内容寻址清单、SHA-256 完整性校验和原子发布；默认不触碰生产服务数据。
- `scripts/restore_local.py` 在验证 manifest、路径和每个文件 hash 后，仅恢复到不存在的目标；服务恢复要求显式空 MySQL 数据库、新 Elasticsearch generation 和空 Redis target prefix，并在导入后执行 Alembic、文档计数、mapping/alias/Redis payload 校验与 readiness。
- `scripts/run_api.py` 提供有限优雅退出；`ReadinessChecks.require_ready()` 对依赖不可用 fail-closed；`scripts/verify_acceptance.py` 严格要求泄漏为 0、citation coverage 为 1.0、无未审计答案、恢复演练和备份恢复均通过。
- 真实本地服务验证使用隔离资源：MySQL schema/API 15+4 项、Redis Streams 3 项、Elasticsearch 检索 1 项、备份恢复 E2E 15 项均通过；不修改默认 `agentic_rag` 数据库、Redis 默认数据或现有 ES generation。真实 DeepSeek/Qwen 模型 smoke 1 项通过（Qwen embedding 1024 维）。
- 最终 baseline 24 cases 离线评测产生 `user_leak_count=0`、`citation_coverage=1.0`、`unaudited_answer_count=0`、两个恢复 gate 均为 true；`verify_acceptance.py` 返回 `ACCEPTANCE PASSED`。

### 3.5 Query Runtime/Mem0/真实评测 follow-up（7/7）

| Task | 最终提交 | 交付 |
|---|---|---|
| 1 | `ae7c6fe` | Query Outbox 按 `aggregate_type` 隔离，查询 worker 不会消费 ingestion rows |
| 2 | `d543f31` / `513a0aa` | 生产 Query 依赖组合、Worker 启动与 API/Worker 快照绑定 |
| 3 | `2a117a3` | API → Outbox → Redis → Query Worker → QueryGraph → MySQL 的真实边界 E2E |
| 4 | `a4ececf` / `fa85aca` | Mem0 `AsyncMemory` 工厂、用户作用域/删除 tombstone、显式真实 provider 测试开关 |
| 5 | `2bb7f25` | Graph/API 评测模式、真实 client provenance、严格断点续跑与 `verify_acceptance` 门禁 |
| 6 | `67d492a` | 降级/熔断/重试/DLQ 的结构化 warning 与 durable event，安全字段 allowlist、指标去重 |
| 7 | 本次提交 | 发布门禁测试、真实服务运行顺序、文档和最终验收快照 |

这些 follow-up 共同回答了生产级系统的两个核心问题：Query Outbox 保证“数据库 Run 状态与待投递消息”在同一事务中可恢复；Query Worker 负责租约、重领取、心跳、重试/DLQ、取消和最终 ACK。任何降级、熔断或拒绝都会同时留下安全 warning 和可重放 durable event，API/SSE 只暴露有限枚举，不泄露 prompt、工具载荷或 provider 原文。

## 4. 最近验证证据

验证基于实现分支 follow-up 工作树，使用 `conda` 环境 `agentic-rag`：

```text
full importlib suite: 575 passed, 46 skipped
Task 1 focused observability/graph/worker/model suite: 70 passed
Task 2 focused evaluation suite: 23 passed
Task 3 focused runner/evaluation suite: 50 passed
Task 3 related eval/query/runtime/observability subset: 168 passed
full unit suite after Task 3: 459 passed
Task 4 focused E2E suite: 10 passed
Recovery drill CLI: direct/module invocation exit 0; duplicates=0, leaks=0
Task 5 backup/restore E2E with disposable MySQL/Redis/Elasticsearch: 15 passed
real Query composition/E2E integration: 2 passed (API deployment E2E is explicit opt-in)
Mem0 provider contract: explicit opt-in; missing provider skips, configured provider failures
  fail the test rather than silently using a no-op memory service
degradation/circuit focused suite: 46 passed
MySQL schema integration with module-scoped event loop: 15 passed
MySQL document API integration with module-scoped event loop: 4 passed
Redis Streams integration: 3 passed
Elasticsearch retrieval integration: 1 passed
Live DeepSeek/Qwen smoke: 1 passed
baseline evaluation: 24 cases; acceptance verifier: ACCEPTANCE PASSED
ruff check src tests evals scripts: All checks passed
mypy src evals scripts: no issues found
git diff --check b83b716..8729171: clean
git diff --check 2e99e00..1947806: clean
```

外部服务集成测试的 skip 是显式配置结果，未提供以下独立测试资源时不会伪造通过：

```bash
AGENTIC_RAG_TEST_MYSQL_DSN
AGENTIC_RAG_TEST_REDIS_DSN
AGENTIC_RAG_TEST_ELASTICSEARCH_URL
```

Mem0 真实服务测试同样需要本地 provider/服务配置；当前单元测试使用注入的 async client/fake 覆盖边界行为。

## 5. 本地运行与数据布局

项目不使用 Docker，默认本地组件如下：

| 组件 | 默认位置/端口 | 当前职责 |
|---|---|---|
| MySQL | `127.0.0.1:3306` | Document、Version、Job、Parent、Run/Event、Outbox、删除/DLQ |
| Redis | `127.0.0.1:6379/0` | Ingestion/Query Streams、consumer group、reclaim、dead stream |
| Elasticsearch 8 | `http://127.0.0.1:9200` | Child BM25、dense vector、结构化 filter |
| SQLite | `var/ingestion_checkpoints.sqlite` | IngestionGraph checkpoint |
| SQLite | `var/query_checkpoints.sqlite` | QueryGraph checkpoint |
| Local files | `var/artifacts` | 上传源文件、解析/分块/Manifest artifacts |

Phase 2 的 embedding/provider 环境变量仍按对应计划配置；Query API/Worker 还需配置 MySQL、Redis 以及注入的 ModelGateway/Memory provider。

## 6. 尚未实现与上线前注意事项

- 生产上线前仍需执行一次分支级发布审查，并把实现 worktree 合并到 `main`；本地最终验收不等同于生产鉴权/RBAC 审批。
- 当前 V1 只有 `user_id` 命名空间隔离，没有完整鉴权、RBAC 或用户身份解析；生产入口不能继续依赖 `default_user`。
- Memory 的真实 Mem0 provider 由部署注入；未配置时服务会 fail-closed，不应把 no-op 结果当作生产记忆。
- 本地 Elasticsearch、MySQL、Redis 已用隔离测试资源完成联调；Mem0 真实 provider 仍由部署注入，当前安全回归使用注入 fake/边界测试，不能把 Mem0 fake 结果当生产可用性证明。
- 当前实现仍在 `sdd-agentic-rag-implementation`，合并到 `main` 前需进行一次分支级回归和发布审查。

## 7. 下一次开发的准确起点

Follow-up Task 7 已完成；下一步是分支级发布审查和合并，不应重新实现 Phase 1–5 或 Query Runtime follow-up。

恢复步骤：

1. 进入实现 worktree，确认分支为 `sdd-agentic-rag-implementation`、工作树干净。
2. 运行 `docs/local-operations.md` 中的静态、真实服务、Graph/API 评测和最终验收命令。
3. 请求独立 reviewer 对 follow-up 与 Task 5 提交范围复审；若通过，执行分支级 diff、迁移和发布审查。
4. 将实现分支合并到 `main` 前，重新确认 `.env.local`、备份目录和隔离测试数据库未被纳入提交。

Phase 5 顺序：

```text
Trace Recorder + Online Metrics (complete)
  -> Deterministic Retrieval/AgentLoop Evaluation (complete)
  -> Offline Ragas Reports (complete)
  -> Adversarial/Load/Recovery Suites (complete)
  -> Backup/Restore/Readiness/Final Acceptance (complete)
```

## 8. 权威文档索引

- [生产级 Agentic RAG 总体设计](./superpowers/specs/2026-08-04-production-agentic-rag-design.md)
- [五阶段实现路线图](./superpowers/plans/2026-08-04-agentic-rag-implementation-roadmap.md)
- [Phase 1 计划](./superpowers/plans/2026-08-04-agentic-rag-phase-1-foundation.md)
- [Phase 2 计划](./superpowers/plans/2026-08-04-agentic-rag-phase-2-ingestion.md)
- [Phase 3 计划](./superpowers/plans/2026-08-04-agentic-rag-phase-3-retrieval.md)
- [Phase 4 计划](./superpowers/plans/2026-08-04-agentic-rag-phase-4-query-runtime.md)
- [Phase 5 计划](./superpowers/plans/2026-08-04-agentic-rag-phase-5-evaluation-operations.md)

## 9. 恢复开发时的第一条提示词建议

```text
继续开发 AgenticRAG。先完整阅读 docs/development-progress.md、总体设计、实现路线图、Phase 5 计划和 query-runtime follow-up 计划；确认当前 worktree/branch/HEAD 与进度快照一致，运行静态、全量、真实服务和 Graph/API 验收门禁。不要重做已完成阶段，不要把 fixture smoke 当作生产验收，也不要跳过 Mem0、备份恢复或降级/熔断遥测检查。
```
