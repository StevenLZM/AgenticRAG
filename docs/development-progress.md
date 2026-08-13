# Agentic RAG 开发进度快照

> 快照日期：2026-08-13
>
> 当前状态：Phase 1–4 已完成并通过独立审查；Phase 5 尚未开始。当前实现位于独立 worktree，尚未合并到 `main`。
>
> 本文档是恢复开发时的首要状态入口；详细设计、接口约束和任务拆分以文末权威文档为准。

## 1. 当前开发位置

- 仓库根目录：`/Users/steven/LzmWorkSpace/AgenticRAG`
- 实现 worktree：`/Users/steven/LzmWorkSpace/AgenticRAG/.worktrees/agentic-rag-implementation`
- 实现分支：`sdd-agentic-rag-implementation`
- 分支基线：`main@1e02fbb`
- Phase 4 最终代码：`c3cf053 fix: preserve business terminal query outcomes`
- Conda 环境：`agentic-rag`

`main` 尚未合并当前实现。继续开发前先进入实现 worktree，并确认工作树干净：

```bash
cd /Users/steven/LzmWorkSpace/AgenticRAG/.worktrees/agentic-rag-implementation
git branch --show-current
git status --short
git log -5 --oneline
```

## 2. 总体进度

路线图共 32 个任务（Phase 1/2/3 各 6 个，Phase 4 为 9 个，Phase 5 为 5 个）。

| 阶段 | 状态 | 任务数 | 交付摘要 |
|---|---:|---:|---|
| Phase 1：Foundation and Persistence | 完成 | 6/6 | 领域契约、MySQL、Outbox/Redis、Checkpoint、Artifact Store、Bootstrap/Health |
| Phase 2：Document Ingestion | 完成 | 6/6 | 安全上传、Docling AST、Parent/Child、Embedding、发布/对账、可恢复 Worker |
| Phase 3：Retrieval and Evidence | 完成 | 6/6 | Dense/BM25、RRF/Rerank、Parent 聚合、降级检索图、EvidenceBuilder |
| Phase 4：Query and Agent Runtime | 完成 | 9/9 | ModelGateway、Memory、Fast/Research、Subagent、审计、QueryGraph、Worker、API |
| Phase 5：Evaluation and Operations | 未开始 | 0/5 | 评测、负载/恢复、备份恢复、运行就绪和最终验收 |

**总体完成：`27/32` 个路线图任务，约 `84.4%`。**

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

## 4. 最近验证证据

验证基于实现分支最终提交 `c3cf053`，使用 `conda` 环境 `agentic-rag`：

```text
unit tests: 391 passed
runtime/api/memory integration: 27 passed, 5 skipped
ruff check src tests scripts: All checks passed
mypy src: no issues found in 80 source files
git diff --check c0d1305..HEAD: clean
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

- Phase 5 尚未开始，仍缺少五项评测与运维交付：在线 Trace/指标、确定性检索与 AgentLoop 评测、Ragas 离线报告、对抗/负载/恢复测试、备份恢复/就绪检查/最终验收。
- 当前 V1 只有 `user_id` 命名空间隔离，没有完整鉴权、RBAC 或用户身份解析；生产入口不能继续依赖 `default_user`。
- Memory 的真实 Mem0 provider 由部署注入；未配置时服务会 fail-closed，不应把 no-op 结果当作生产记忆。
- Elasticsearch、MySQL、Redis、Mem0 的真实联调尚未在本环境执行；上线前必须使用独立测试资源完成门禁。
- 当前实现仍在 `sdd-agentic-rag-implementation`，合并到 `main` 前需进行一次分支级回归和发布审查。

## 7. 下一次开发的准确起点

下一任务是 **Phase 5 Task 1：Trace Recorder and Online Metrics Projection**。

恢复步骤：

1. 进入实现 worktree，确认分支为 `sdd-agentic-rag-implementation`、工作树干净、HEAD 为 `c3cf053` 或其后续 docs-only 提交。
2. 运行 Phase 4 回归门禁，确认 Query Worker/API/审计没有回归。
3. 完整阅读 Phase 5 计划与全局约束，先为 Task 1 编写 RED 测试和独立 brief。
4. 配置可丢弃的 MySQL/Redis/Elasticsearch 测试资源；有 Mem0 provider 时再启用真实 Memory 集成。
5. 按 TDD、实现报告、独立 reviewer、fix/re-review 流程推进，不跳过评测的可重复性和运行时指标契约。

Phase 5 顺序：

```text
Trace Recorder + Online Metrics
  -> Deterministic Retrieval/AgentLoop Evaluation
  -> Offline Ragas Reports
  -> Adversarial/Load/Recovery Suites
  -> Backup/Restore/Readiness/Final Acceptance
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
继续开发 AgenticRAG。先完整阅读 docs/development-progress.md、总体设计、实现路线图和 Phase 5 计划；确认当前 worktree/branch/HEAD 与进度快照一致，运行 Phase 4 回归门禁，然后使用 Subagent-Driven Development 从 Phase 5 Task 1 开始。不要重做 Phase 1–4，不要跳过真实服务门禁、评测可重复性或恢复测试。
```
