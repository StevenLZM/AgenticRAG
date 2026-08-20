# Agentic RAG 生产实现路线图

> **当前实现状态：** 五个阶段、Query Runtime/Mem0/真实评测后续加固，以及控制台运行时补强均已完成并已合并到 `main`。当前基线为 `fb1a5b5`；不存在待合并的实现分支。真实 Graph/API 验收使用当前 snapshot、真实 client provenance 与隔离服务资源。剩余工作仅为生产鉴权/RBAC 和 reranker 分数标定等上线审批事项。

> **使用说明：** 本路线图记录已交付的生产能力与继续开发的门禁。新增变更必须遵循 red-green-refactor，保持命令、环境变量、API 路径、事件名、状态枚举和文件路径可直接复制。

**目标：** 交付获批准的单主机生产级 Agentic RAG 系统，并以可恢复运行时、可审计证据和真实服务验收作为发布前提。

**架构：** 系统先持久化事实和投递意图，再执行摄取、检索、QueryGraph/Agent 运行时和评测。Query Outbox 与 Query Worker 分离：前者在 MySQL 同事务内保存 Run 与待投递意图，后者从 Redis 领取任务、维护租约/心跳、恢复、重试/DLQ、执行 Graph 并在终态持久化后 ACK。该边界避免 HTTP 成功与 Redis 发布之间的丢失窗口，也避免重启产生重复 Run。

**技术栈：** Python 3.11、FastAPI、LangGraph、DeepSeek、Qwen `text-embedding-v3`、Docling、Elasticsearch 8、MySQL、Redis Streams、SQLite Checkpointer、mem0ai 2.0.12、BGE Cross-Encoder、Ragas、pytest。

## 全局约束

- 本地运行不使用 Docker、Kubernetes、微服务、Celery、Dramatiq、RQ 或 ARQ。
- 只使用 LangGraph 作为 Graph 和 Agent 编排框架；Query 与 Ingestion checkpoint 分别使用启用 WAL 的 SQLite 文件，且每个文件只允许一个写入进程。
- 主模型为 DeepSeek `deepseek-v4-pro`，轻量模型为 `deepseek-v4-flash`；向量嵌入为 Qwen `text-embedding-v3`，维度为 1024。
- `user_id` 由服务端注入，V1 的 `default_user` 仅用于本地命名空间隔离，不构成生产身份认证。
- 文档、Memory 和 Tool 输入均为不可信数据，不能覆盖系统指令或 Tool 模式。
- Mem0 默认启用；provider 故障只能产生 `memory_provider_degraded` 和受控降级，不能伪造空记忆成功。
- 所有生产事件和控制台字段必须经白名单过滤，不能暴露 prompt、隐藏推理、工具载荷、密钥或原始服务提供方响应。

## 已完成的有序阶段

| 阶段 | 已交付能力 | 关键完成证据 |
|---|---|---|
| 阶段 1：基础设施与持久化 | 领域模型、MySQL/Alembic、Outbox/Redis、SQLite checkpoint、Artifact、Bootstrap/Health | 迁移、重放、重复通知与本地依赖检查 |
| 阶段 2：文档摄取 | 安全上传、Docling AST、Parent-Child 分块、Embedding/索引暂存、发布/对账、摄取恢复 | 多格式固定数据、活动版本和 Worker 恢复 |
| 阶段 3：检索与证据 | Dense/BM25、RRF、Cross-Encoder、Parent 聚合、检索图与 `EvidenceBuilder` | 跨用户隔离、固定 Recall@6、NDCG@10、MRR |
| 阶段 4：查询与 Agent 运行时 | ModelGateway、Memory、Fast/Research、Subagent、审计、QueryGraph、Query Worker、API/SSE | 取消、checkpoint 恢复、SSE 重连、终态审计 |
| 阶段 5：评测与运维 | Trace/Metrics、离线评测、对抗/负载/恢复、备份恢复、就绪与验收 | 恢复/备份演练和 `scripts/verify_acceptance.py` |

## 已完成的运行时与控制台补强

- `SubagentDispatcher` 已接入生产组合根，并和 QueryGraph、Query Worker 共用 `ConcurrencyManager`；超时、取消与 evidence reducer 保持有界。
- 研究路径会完成 Todo 初始创建，合法动作可 Todo 追加。Todo 的所有者、标题和动作 schema 均在服务端校验，页面只显示安全状态。
- `research_attempt_count` 是持久化、跨 Graph 重入的全局预算。达到 `RuntimeConfigSnapshot` 上限即阻止未完成 Todo，以 `research_round_limit` 终态结束，不继续调用模型。
- `GET /` 提供 Agentic RAG 控制台；它使用 Query Run、SSE、文档、Mem0 和 Health API，支持断线重连，仅显示服务端白名单字段和安全终态。
- `MODEL_REPAIR_EXHAUSTED`、`CIRCUIT_OPEN`、`WORKER_DLQ`、`OUTBOX_RETRY`、`memory_provider_degraded` 和检索降级均有有限日志与 durable event。日志字段固定为 `component`、`reason`、`outcome`、`attempt`、`retryable` 与 `degraded_components`。

## 当前验收门禁

常规静态与离线检查：

```sh
conda run -n agentic-rag ruff check src tests evals scripts
MYPYPATH=src conda run -n agentic-rag mypy --explicit-package-bases src evals scripts
conda run -n agentic-rag python -m pytest --import-mode=importlib \
  -m 'not integration and not e2e and not live_model' -q
conda run -n agentic-rag python -m evals.run --mode fixture \
  --dataset evals/datasets/baseline.jsonl --output var/artifacts/evals/fixture
```

真实验收必须启动 API 与 Query Worker，并使用隔离 MySQL、Redis、Elasticsearch、Mem0、SQLite checkpoint 和 Artifact。`fixture` 输出的 `SMOKE ONLY` 永远不能替代真实结果：

```sh
set -a; source .env.local; set +a
export HF_HOME=/tmp/agentic-rag-hf
export MEM0_TELEMETRY=0
export AGENTIC_RAG_RUN_REAL_QUERY_PROVIDER_E2E=1
conda run -n agentic-rag python scripts/run_real_query_acceptance.py \
  --output var/artifacts/evals/real-api-current
conda run -n agentic-rag python scripts/verify_acceptance.py \
  --report var/artifacts/evals/real-api-current/summary.json
```

最终 PASS 要求真实 Graph/API summary 同时证明：当前 snapshot 与 `runtime_config_snapshot_id` 一致、`client_provenance=real_query_api`、Mem0 可用或有明确受控的 provider degraded 证据、`real_query_count` 为正数、`user_leak_count=0`、`citation_coverage=1.0`、`unaudited_answer_count=0`，以及恢复和备份演练均通过。缺失真实来源、服务不健康或任一指标不满足时，`scripts/verify_acceptance.py` 必须非零退出。

## 上线前审批事项

1. **生产鉴权/RBAC：** 设计并验收真实身份认证、授权和 RBAC；不能把 `default_user` 或 `user_id` 命名空间隔离视为生产鉴权。
2. **reranker 分数标定：** 显式传播 `retrieval_score`、`rrf_score` 和 `rerank_score`；为版本化 `min_rerank_score` 建立按模型/revision、score activation、索引代际和评测切片的标定及回归门禁。
3. **发布复验：** 每个候选发布都要在隔离资源中重跑当前 snapshot、真实 client provenance、Mem0、恢复、备份和真实 Graph/API 验收；保存脱敏 summary 与遥测证据。

## 详细文档

- [阶段 1 计划](./2026-08-04-agentic-rag-phase-1-foundation.md)
- [阶段 2 计划](./2026-08-04-agentic-rag-phase-2-ingestion.md)
- [阶段 3 计划](./2026-08-04-agentic-rag-phase-3-retrieval.md)
- [阶段 4 计划](./2026-08-04-agentic-rag-phase-4-query-runtime.md)
- [阶段 5 计划](./2026-08-04-agentic-rag-phase-5-evaluation-operations.md)
- [控制台实施计划](./2026-08-15-agentic-rag-console.md)
- [本地运行手册](../../local-operations.md)
