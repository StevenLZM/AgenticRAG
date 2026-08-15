# Agentic RAG 生产实现路线图

> **当前实现状态：** 阶段 1–5 以及 Query Runtime/Mem0/真实评测后续加固均已在 `sdd-agentic-rag-implementation` 完成。继续工作前请阅读[开发进度快照](../../development-progress.md)；目前只剩分支合并和生产身份/RBAC 审批。

> **供 Agent 工作者使用：** 必须使用 `superpowers:subagent-driven-development`（推荐）或 `superpowers:executing-plans` 子技能，按任务逐项执行本路线图。步骤使用复选框（`- [ ]`）跟踪。

**目标：** 交付获批准的单主机生产级 Agentic RAG 系统，分为五个有序、可独立审查的实现阶段。

**架构：** 先构建持久化存储和运行时契约，再完成文档摄取、确定性检索、QueryGraph/Agent 运行时，最后完成评测和生产验收。每个阶段都有独立的详细计划，必须通过该阶段的验证门禁后才能进入下一阶段。

**技术栈：** Python 3.11、FastAPI、LangGraph、DeepSeek、Qwen `text-embedding-v3`、Docling、Elasticsearch 8、MySQL、Redis Streams、SQLite Checkpointer、mem0ai 2.0.12、BGE Cross-Encoder、Ragas、pytest。

## 全局约束

- 本地运行不使用 Docker、Kubernetes、微服务、Celery、Dramatiq、RQ 或 ARQ。
- 只使用 LangGraph 作为 Graph 和 Agent 编排框架。
- 主模型：DeepSeek `deepseek-v4-pro`；轻量模型：`deepseek-v4-flash`。
- 向量嵌入：Qwen `text-embedding-v3`，维度必须为 1024。
- Reranker：`BAAI/bge-reranker-v2-m3`，在 Query Worker 中只加载一次。
- Elasticsearch 默认端点为 `http://localhost:9200`；MySQL 和 Redis 端点来自环境配置；Redis 默认使用 `redis://127.0.0.1:6379/0`。
- MySQL 存储 Parent、Run、Job、Outbox、审计和删除状态；ES 存储活动 Child 检索记录；本地文件存储带版本的 Artifact。
- SQLite Checkpoint 分别使用 Query 和 Ingestion 文件，并启用 WAL；每个文件严格只允许一个写入进程。
- `user_id` 默认值为 `default_user`，由 API 注入，Agent 或检索请求不能提供或修改它。
- V1 假设调用方可信；`user_id` 只是数据作用域，不是身份认证。
- 检索到的文档和 Memory 都是不可信数据，不能覆盖系统指令或 Tool 模式。
- 不使用 Token 或成本预算做路由；Token 和 estimated cost 只作为观测字段。
- 单元测试不要求真实模型 API。需要本地 ES、MySQL 或 Redis 的集成测试必须使用明确的 pytest 标记；在未启动服务时显式选择这些标记应给出清晰的依赖错误。
- 开发和评测环境使用以下方式安装：`conda run -n agentic-rag python -m pip install -e '.[dev,eval]'`。
- 所有命令都在 `agentic-rag` Conda 环境中执行；非交互脚本使用 `conda run -n agentic-rag ...`。
- 每个任务都遵循 red-green-refactor，结束时运行聚焦测试，并产生一个可审查的提交。

---

## 有序阶段计划

1. [阶段 1 — 基础设施与持久化](./2026-08-04-agentic-rag-phase-1-foundation.md)
   创建 Python 包、类型化契约、配置、MySQL 模式、仓储、Outbox/Redis 接口、SQLite Checkpoint 和 Artifact Store。

2. [阶段 2 — 文档摄取](./2026-08-04-agentic-rag-phase-2-ingestion.md)
   交付上传安全、Docling Fragment/Canonical AST 组装、Parent-Child Chunking、向量嵌入/索引暂存、Publisher、Reconciler 和 Ingestion Worker 恢复。

3. [阶段 3 — 检索与证据](./2026-08-04-agentic-rag-phase-3-retrieval.md)
   交付服务端作用域 Dense/BM25 检索、RRF、Cross-Encoder 重排、Parent 获取、RetrievalPipelineGraph 和确定性 EvidenceBuilder。

4. [阶段 4 — Query 与 Agent 运行时](./2026-08-04-agentic-rag-phase-4-query-runtime.md)
   交付 ModelGateway、mem0ai MemoryService、QueryGraph、真正的 ResearchAgentLoop、Subagents、强制审计、持久化 Query Run、取消、SSE 和 Query API。

5. [阶段 5 — 评测与生产验收](./2026-08-04-agentic-rag-phase-5-evaluation-operations.md)
   交付 tracing 与 metrics、离线评测、对抗/负载/恢复套件、迁移、备份/恢复、进程脚本和最终验收门禁。

## 阶段门禁

| 阶段 | 进入下一阶段前必须具备的证据 |
|---|---|
| 1 | 单元测试通过；在干净本地数据库上完成迁移升级/降级；Redis 重复通知和 SQLite 重放测试通过。 |
| 2 | 文本 PDF、扫描 PDF、文本文件和 Excel 固定测试数据都能生成有效 Canonical AST、Parent/Child 记录和可恢复的活动版本。 |
| 3 | 混合检索集成测试通过且无用户泄漏；确定性 Recall@6、NDCG@10 和 MRR 固定测试数据产生预期值。 |
| 4 | Fast RAG、多跳循环、Subagent、审计、取消、checkpoint 恢复和 SSE 重连测试通过。 |
| 5 | 完整单元/集成/端到端/评测套件通过；备份恢复和 Worker 中断演练通过；质量硬门禁保持零违规。 |

## 设计覆盖关系

| 已批准的设计领域 | 负责的实现任务 |
|---|---|
| 本地三进程运行时、Run 生命周期、Outbox、Lease、取消和 Checkpoint | 阶段 1 任务 3–6；阶段 4 任务 8–9 |
| QueryGraph 路由、Fast RAG、ResearchAgentLoop、Todo、Subagent 和上下文压缩 | 阶段 4 任务 3–7 |
| Dense/BM25 过滤、RRF、Cross-Encoder、Parent 获取和 EvidenceBuilder | 阶段 3 任务 1–6 |
| Docling AST、跨页组装、Parent-Child Chunking、索引和对账 | 阶段 2 任务 1–6 |
| 工作记忆/长期记忆、Mem0 生命周期和删除 | 阶段 1 任务 5；阶段 4 任务 2 |
| Evidence、Faithfulness 和引用门禁 | 阶段 4 任务 6–7 |
| 上传、检索上下文、Memory 和 Tool 输入的安全边界 | 阶段 2 任务 1–2；阶段 3 任务 1 和 6；阶段 4 任务 2、4 和 6 |
| 运行时版本、链路追踪、在线指标和三层评测 | 阶段 1 任务 2；阶段 5 任务 1–3；Query Runtime 后续任务 5–6 |
| 重试/降级、负载/背压、恢复和本地运维 | 阶段 3 任务 3 和 5；阶段 4 任务 8；阶段 5 任务 4–5；Query Runtime 后续任务 1–4、6–7 |

## 最终验证命令

```bash
conda run -n agentic-rag ruff check src tests evals scripts
conda run -n agentic-rag mypy src
conda run -n agentic-rag python -m pytest --import-mode=importlib \
  -m "not integration and not e2e and not live_model" -q
conda run -n agentic-rag python -m pytest --import-mode=importlib -m integration -q
conda run -n agentic-rag python -m pytest --import-mode=importlib -m e2e -q
conda run -n agentic-rag python -m pytest --import-mode=importlib -m live_model tests/smoke -q
# fixture 仅用于冒烟测试；最终验收必须使用 graph 或 api 来源证明。
conda run -n agentic-rag python -m evals.run --mode fixture \
  --dataset evals/datasets/baseline.jsonl --output artifacts/evals/fixture
conda run -n agentic-rag python -m evals.run --mode graph \
  --dataset var/artifacts/evals/runtime-baseline.jsonl --output artifacts/evals/graph
conda run -n agentic-rag python scripts/verify_acceptance.py \
  --report artifacts/evals/graph/summary.json
```

只要 `user_leak_count != 0`、引用覆盖率低于 100%、返回了未审计答案、报告只有 fixture 模式、缺少真实 Query 来源证明、必要服务不健康，或恢复演练未完成，最终验证器就必须以非零状态退出。API 模式使用相同的运行时快照匹配数据集，并显式设置 `AGENTIC_RAG_EVAL_SNAPSHOT_ID`。
