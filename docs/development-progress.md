# Agentic RAG 开发进度快照

> 快照日期：2026-08-20
>
> 当前状态：原始五阶段路线图、Query Runtime/Mem0/真实评测后续任务，以及控制台与其运行时补强均已在当前 `main` 完成。不存在需要合并的实现 worktree 或分支；后续工作从仓库根目录的 `main` 开始。

> 本文档是恢复开发时的状态入口；设计约束和任务拆分以文末权威文档为准。

## 当前开发位置

- 仓库根目录：`/Users/steven/LzmWorkSpace/AgenticRAG`
- 当前分支：`main`
- 当前基线：`f3a0e53`（控制台最终边界加固；以 `git log -1 --oneline` 为准）
- Conda 环境：`agentic-rag`
- 启动前检查：在仓库根目录运行 `git status --short` 和 `git log -5 --oneline`，确认当前 `main` 工作树干净。

## 总体进度

| 阶段 | 状态 | 交付摘要 |
|---|---|---|
| 阶段 1：基础设施与持久化 | 完成 | 领域契约、MySQL、Outbox/Redis、Checkpoint、Artifact Store、Bootstrap/Health |
| 阶段 2：文档摄取 | 完成 | 安全上传、Docling AST、Parent/Child、Embedding、发布/对账、可恢复 Worker |
| 阶段 3：检索与证据 | 完成 | Dense/BM25、RRF/Rerank、Parent 聚合、降级检索图、EvidenceBuilder |
| 阶段 4：查询与 Agent 运行时 | 完成 | ModelGateway、Memory、Fast/Research、Subagent、审计、QueryGraph、Worker、API |
| 阶段 5：评测与运维 | 完成 | Trace/在线指标、确定性评测、离线报告、对抗/负载/恢复演练、备份恢复与最终验收 |
| 控制台与运行时补强 | 完成 | 全局研究预算、生产 Subagent、同源控制台、真实 Graph/API 验收 |

原始路线图的 `32/32` 项和后续 Query Runtime/Mem0/真实评测任务均已完成。当前 `main` 的剩余事项是上线审批，而不是代码分支合并。

## 已实现能力

### 持久化、摄取、检索与证据

- MySQL 持久化 Run、Event、Outbox、文档版本和审计状态；Elasticsearch 保存可检索 Child；SQLite 分别保存 Query 与 Ingestion checkpoint；Artifact 使用内容寻址存储。
- 文档摄取包含 MIME、大小和压缩炸弹防护、Docling Fragment/Canonical AST、Parent-Child 分块、Qwen embedding、索引发布、对账、租约、心跳与 DLQ 恢复。
- 检索在服务端执行 `user_id`、scope、版本和 selector 过滤，使用 Dense/BM25、固定 RRF、Cross-Encoder、Parent 聚合和 `EvidenceBuilder`。任一路检索故障会留下有限降级证据；双路不可用时 fail-closed。

### 查询运行时与生产投递

- Query Outbox 在 `POST /v1/query-runs` 的同一个 MySQL 事务内持久化 Run 与 `query_run` 投递意图，因而 Redis 短暂不可用不会丢失已提交查询。
- Query Worker 从 Redis Stream 领取和重新领取任务，维护租约与心跳，从 checkpoint/持久化 Run 恢复，执行 QueryGraph，实施重试/DLQ、取消和终态 ACK。答案或业务拒答只有在持久化且通过审计后才 ACK，避免重复 Run 和伪完成。
- QueryGraph 固定编排 Fast/Research、EvidenceBuilder、Generation、Faithfulness、Citation 和 Finalize。`clarify`、`refuse`、`cannot_answer`、`audit_failed`、`research_round_limit` 与 `research_action_invalid` 都是结构化、可观察的业务终态；未知故障仍按重试/DLQ 路径处理。
- `SubagentDispatcher` 已接入生产组合根，并与 QueryGraph/Query Worker 共享 `ConcurrencyManager`。研究路径会进行 Todo 初始创建；合法动作可 Todo 追加，服务器校验 Todo 所有权与动作 schema。
- `research_attempt_count` 是保存在 QueryState/SQLite checkpoint 的全局研究尝试数，跨 `ResearchAgentLoop` 重入累计；它不是 QueryRun DB 字段。达到 `RuntimeConfigSnapshot` 上限后，系统阻止未完成 Todo 并以 `research_round_limit` 终态停止，不再调用模型。

### 记忆、控制台与安全观察

- Mem0 默认启用，按用户命名空间工作，并使用 `infer=False`。没有专用 Mem0 embedding 配置时会复用 Qwen 配置。provider 故障只会产生 `memory_provider_degraded`，使 `/health/ready` 不就绪；不会把空记忆当作成功或放宽检索、引用和审计。
- `GET /` 提供同源的 Agentic RAG 控制台；页面经既有 Query Run、SSE、文档、Mem0 和 Health API 工作。它仅呈现服务端白名单事件与字段，可显示 `queued`、`running`、`completed`、`refuse`、`audit_failed`、`research_round_limit` 及安全的降级/熔断/重试/DLQ 提示。
- 事件与日志脱敏 prompt、隐藏推理、工具载荷和服务提供方响应，只保留 `component`、`reason`、`outcome`、`attempt`、`retryable` 等有限字段。`MODEL_REPAIR_EXHAUSTED`、`CIRCUIT_OPEN`、`OUTBOX_RETRY`、`WORKER_DLQ` 与 `memory_provider_degraded` 均可用于排查，不构成泄密通道。

## 真实验收与证据

- 真实服务测试在隔离 MySQL、Redis、Elasticsearch、Mem0 collection、SQLite checkpoint 和 Artifact 中执行，不触碰默认生产命名空间。
- `scripts/run_real_query_acceptance.py` 启动真实 Query Worker/API，使用当前 snapshot、真实 Graph/API、真实 client provenance 与 Mem0，生成 `evaluation_mode=api`、`client_provenance=real_query_api` 和 `runtime_config_snapshot_id` 证据。
- `scripts/run_real_query_acceptance.py` 先调用 `EvalRunner.run()`，再由 `console_acceptance_passed` 控制台门禁在 EvalRunner 后执行，校验当前 snapshot、精确 `client_provenance=real_query_api`、Mem0 读写、控制台/SSE 以及恢复和备份证据。`scripts/verify_acceptance.py` 是通用验证器，只校验其已有的 summary 字段：正数 `real_query_count`、`user_leak_count=0`、`citation_coverage=1.0`、`unaudited_answer_count=0`、恢复/备份、`evaluation_mode` 与非空且非 `fixture` 的 `client_provenance`；它不单独校验当前 snapshot、精确 provenance 或 Mem0。`fixture` 的 `SMOKE ONLY` 结果不能替代该验收。
- 已覆盖 MySQL schema/API、Redis Streams、Elasticsearch 检索、Mem0 provider、真实模型、恢复演练与备份恢复；外部服务变量缺失时对应 opt-in 测试会明确 skip，已配置但服务不健康时必须失败。

## 本地运行与验证入口

加载 `.env.local` 后，依赖和进程的启动顺序见 [本地运行手册](./local-operations.md)。以下三个命令都是阻塞进程，必须在三个独立终端（或受监督的后台进程）启动，不能写成串行命令；否则 API 不退出时两个 Worker 永远不会启动：

```sh
# 终端一
conda run -n agentic-rag python scripts/run_api.py
# 终端二
conda run -n agentic-rag python scripts/run_query_worker.py
# 终端三
conda run -n agentic-rag python scripts/run_ingestion_worker.py
```

控制台调用 `POST /v1/query` 同步 wrapper；`POST /v1/query-runs` 是同一持久化 Run 的异步 API，供不等待终态的客户端使用。三个进程就绪后再验证页面：

```sh
curl http://127.0.0.1:8000/
```

完整门禁必须额外运行真实 Graph/API 接受脚本及 `scripts/verify_acceptance.py`。所有静态检查使用：

```sh
conda run -n agentic-rag ruff check src tests evals scripts
MYPYPATH=src conda run -n agentic-rag mypy --explicit-package-bases src evals scripts
```

## 上线审批事项

以下事项明确保留为生产上线前的审批和验收工作，不应被本地完成状态掩盖：

1. **生产鉴权/RBAC：** 当前 V1 只有 `user_id` 命名空间隔离，尚无完整身份认证、授权或 RBAC；生产入口不能依赖 `default_user`。
2. **reranker 分数标定：** 必须区分 `retrieval_score`、`rrf_score` 和 `rerank_score`，按 Reranker 模型/revision、score activation、索引代际和评测切片标定版本化 `min_rerank_score`。验收应覆盖真实分数传播、排序、阈值边界、无候选路径、Recall/Precision/NDCG、误拒/误接收率和引用覆盖率。
3. **发布证据复跑：** 每次候选发布都要在隔离服务上重跑恢复、备份、真实 Graph/API、当前 snapshot、client provenance 与 Mem0 门禁，并保留脱敏的 summary 和遥测。

## 权威文档索引

- [生产级 Agentic RAG 总体设计](./superpowers/specs/2026-08-04-production-agentic-rag-design.md)
- [五阶段实现路线图](./superpowers/plans/2026-08-04-agentic-rag-implementation-roadmap.md)
- [查询运行时与 Mem0 设计](./superpowers/specs/2026-08-14-query-runtime-mem0-evaluation-design.md)
- [控制台设计与实施计划](./superpowers/plans/2026-08-15-agentic-rag-console.md)
- [本地运行手册](./local-operations.md)
