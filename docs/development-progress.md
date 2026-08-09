# Agentic RAG 开发进度快照

> 快照日期：2026-08-09
>
> 当前状态：Phase 1、Phase 2 已完成并通过审查；按用户要求暂停，Phase 3 尚未开始。
>
> 本文档是恢复开发时的首要状态入口；详细设计和任务拆分以文末链接的设计文档与阶段计划为准。

## 1. 必须从这里继续

- 仓库根目录：`/Users/steven/LzmWorkSpace/AgenticRAG`
- 实现 worktree：`/Users/steven/LzmWorkSpace/AgenticRAG/.worktrees/agentic-rag-implementation`
- 实现分支：`sdd-agentic-rag-implementation`
- 最后代码里程碑：`55a132e6936337ed49066b2732b4ed6d2cd1e478`
- 进度快照是后续 docs-only 提交；恢复时以分支实际 HEAD 作为新任务 BASE
- 分支基线：`main@1e02fbb`
- Conda 环境：`agentic-rag`

`main` 尚未合并当前实现。恢复开发时不要直接在仓库根目录的 `main` 上继续，先进入实现 worktree：

```bash
cd /Users/steven/LzmWorkSpace/AgenticRAG/.worktrees/agentic-rag-implementation
git branch --show-current
git status --short
git log -5 --oneline
```

预期分支为 `sdd-agentic-rag-implementation`，工作树为空，且 `55a132e` 是当前 HEAD 的祖先提交。可以用下列命令验证：

```bash
git merge-base --is-ancestor 55a132e HEAD
```

## 2. 总体进度

| 阶段 | 状态 | 任务数 | 说明 |
|---|---:|---:|---|
| Phase 1：Foundation and Persistence | 完成 | 6/6 | 包、领域契约、MySQL、Redis Outbox、SQLite Checkpoint、Artifact Store、Bootstrap/Health |
| Phase 2：Document Ingestion | 完成 | 6/6 | 上传、Docling AST、Parent-Child Chunk、Embedding/Staging、Publisher/Reconciler、可恢复 Worker |
| Phase 3：Retrieval and Evidence | 未开始 | 0/6 | 下一阶段，从 Retrieval Contracts 与服务端 Filter Builder 开始 |
| Phase 4：Query and Agent Runtime | 未开始 | 0/9 | QueryGraph、Agent Loop、mem0ai、Subagent、审核、SSE |
| Phase 5：Evaluation and Operations | 未开始 | 0/5 | 三层评测、恢复/负载测试、备份恢复、最终验收 |

总体完成：`12/32` 个路线图任务，约 `37.5%`。

## 3. 已实现的系统边界

### 3.1 Phase 1：基础设施与持久化

- Python 3.11 项目、Pydantic Settings、pytest/Ruff/mypy 工具链。
- 共享领域模型、运行时版本快照与 UUIDv7 标识。
- MySQL Alembic 迁移 `0001` 至 `0005`：Document、Version、Job、Parent、Outbox、Run、Event、删除与 DLQ 状态。
- Async SQLAlchemy/asyncmy repository 与事务边界。
- Transactional Outbox 和 Redis Streams Broker：重复投递去重、ACK、XAUTOCLAIM、Dead Letter。
- SQLite Checkpoint：Query/Ingestion 独立文件，WAL 模式；当前约束为每个文件一个 writer 进程。
- 本地 Artifact Store：受信任路径、原子写入、校验和、版本化引用。
- FastAPI 应用骨架、依赖容器和 MySQL/Redis/Elasticsearch/SQLite 健康检查。

Phase 1 里程碑提交：

| Task | Commit | 结果 |
|---|---|---|
| Package/Settings | `fff1bc8` | review clean |
| Domain/Runtime contracts | `9edbc54` | review clean |
| MySQL schema/repositories | `51c12dc` | fix round 3 后 clean |
| Outbox/Redis | `b4f61b5` | fix round 2 后 clean |
| Checkpoint/Artifact Store | `68906ba` | fix round 1 后 clean |
| Bootstrap/Health | `d962af6` | review clean |

### 3.2 Phase 2：文档摄取

已经实现固定、可恢复的摄取流程：

```text
load_job
  -> upload_safety_gate
  -> parse_fragments
  -> assemble_canonical
  -> content_safety_gate
  -> validate_canonical
  -> chunk
  -> embed_and_stage
  -> publish
  -> finalize
```

主要能力：

- 上传 API、默认 `user_id`、用户数据隔离、文件大小/MIME/压缩炸弹等安全检查。
- Docling 按页/批次解析 PDF、扫描 PDF/OCR、文本、Excel，统一输出 Fragment AST。
- Global Assembler 合并跨页段落、标题、列表、表格与页面家具，生成 Canonical AST，并执行确定性去重。
- 结构优先 Parent-Child Chunking：Parent 保留推理上下文，Child 面向检索；使用递归边界作为 fallback，不使用纯语义切分。
- Qwen `text-embedding-v3` 适配器，默认 1024 维；Tokenizer 显式配置为 Qwen 家族。
- MySQL 保存 Parent 与生命周期，Elasticsearch 保存 Child 结构化字段和向量；向量存储保留可替换 port，当前实现为 ES。
- 默认可写索引 generation 为 `index-v2`；旧 `index-v1` fail-closed。
- Manifest 驱动的跨存储发布顺序、并发 winner、失败恢复与物理清理。
- Publisher/Reconciler 处理发布漂移、删除 tombstone、迟到写入、孤儿数据、状态指针修复和 Outbox 重投。
- LangGraph 固定摄取 Graph，SQLite Checkpoint 使用稳定 `thread_id=ingestion:{job_id}`。
- 单 Worker Redis Stream 消费、MySQL Claim/Lease/Heartbeat、终态 ACK、三次重试、持久化 DLQ、优雅停止和恢复。
- 人工 quarantine approve/reject CLI；批准会轮换 Outbox delivery generation，避免命中已 ACK 的旧消息。
- 每个 Artifact、Embedding、Parent/Child/Manifest、SQL attach 和 Publisher mutation 前重新校验 Claim fence。
- 原始消息与运行事件的持久化审计基础已由 Phase 1 schema 提供；Query 侧审计在 Phase 4 完成。

Phase 2 里程碑提交：

| Task | Commit | 结果 |
|---|---|---|
| Upload/Safety/Job | `7d737b6` | fix round 2 后 clean |
| Docling/Canonical AST | `4477f1b` | fix round 3 后 clean |
| Parent-Child Chunking | `4b8391a` | fix round 3 后 clean |
| Embedding/Staging/Manifest | `65c2ad3` | fix round 2 后 clean |
| Publisher/Reconciler/Delete | `a66fe0f` | fix round 2 后 clean |
| IngestionGraph/Worker Recovery | `55a132e` | fix round 2 后 Spec/Quality PASS |

## 4. 最近验证证据

在 `55a132e` 上最后一次控制端验证结果：

```text
pytest: 263 passed, 32 skipped, 18 warnings
ruff: All checks passed
mypy: no issues found in 49 source files
py_compile: clean
git diff --check: clean
```

对应命令：

```bash
conda run -n agentic-rag python -m pytest -q
conda run -n agentic-rag ruff check src scripts tests
conda run -n agentic-rag mypy src \
  scripts/run_ingestion_worker.py \
  scripts/review_quarantined_version.py \
  tests/unit/ingestion/test_worker.py \
  tests/unit/ingestion/test_worker_store.py \
  tests/unit/ingestion/test_worker_publication_recovery.py \
  tests/e2e/test_ingestion_pipeline.py \
  tests/e2e/test_ingestion_pipeline_real_services.py \
  tests/integration/ingestion/test_quarantine_approval_delivery.py
```

32 个 skip 主要是未提供显式测试 DSN 的真实 MySQL、Redis、Elasticsearch 测试。配置了测试 DSN 但服务不可达时，门控测试会失败，不会降级为 fake。

## 5. 本地运行与数据布局

项目不使用 Docker，所有组件本地启动：

| 组件 | 默认位置/端口 | 当前职责 |
|---|---|---|
| MySQL | `127.0.0.1:3306` | Document、Version、Job、Parent、Outbox、Run/Event、删除/DLQ 状态 |
| Redis | `127.0.0.1:6379/0` | Ingestion Stream、消费者组、pending reclaim、dead stream |
| Elasticsearch 8 | `http://127.0.0.1:9200` | Child BM25 字段、Dense Vector、结构化 filter 字段 |
| SQLite | `var/ingestion_checkpoints.sqlite` | IngestionGraph checkpoint |
| SQLite | `var/query_checkpoints.sqlite` | 预留给 Phase 4 QueryGraph |
| Local files | `var/artifacts` | 上传源文件、Fragment/Canonical/Chunk/Manifest artifacts |

运行环境变量：

```bash
export AGENTIC_RAG_MYSQL_DSN='mysql+asyncmy://rag:rag@127.0.0.1:3306/agentic_rag?charset=utf8mb4'
export AGENTIC_RAG_REDIS_URL='redis://127.0.0.1:6379/0'
export AGENTIC_RAG_ELASTICSEARCH_URL='http://127.0.0.1:9200'
export AGENTIC_RAG_INDEX_GENERATION='index-v2'

# Settings 当前要求这两个 URL；摄取 Worker 实际调用 Qwen embedding endpoint。
export AGENTIC_RAG_DEEPSEEK_BASE_URL='<deepseek-openai-compatible-base-url>'
export AGENTIC_RAG_QWEN_EMBEDDING_BASE_URL='<qwen-openai-compatible-base-url>'
export AGENTIC_RAG_QWEN_API_KEY='<secret>'
```

真实服务测试必须使用独立资源：

```bash
export AGENTIC_RAG_TEST_MYSQL_DSN='mysql+asyncmy://rag_test:rag_test@127.0.0.1:3306/agentic_rag_test?charset=utf8mb4'
export AGENTIC_RAG_TEST_REDIS_DSN='redis://127.0.0.1:6379/15'
export AGENTIC_RAG_TEST_ELASTICSEARCH_URL='http://127.0.0.1:9200'
```

警告：`AGENTIC_RAG_TEST_MYSQL_DSN` 必须指向可丢弃测试库。Schema 集成测试会执行 Alembic downgrade/upgrade，不能指向运行库。

初始化与启动：

```bash
cd /Users/steven/LzmWorkSpace/AgenticRAG/.worktrees/agentic-rag-implementation
conda run -n agentic-rag alembic upgrade head
conda run -n agentic-rag python scripts/run_ingestion_worker.py
```

## 6. 已确认但尚未实现的范围

以下内容不是缺陷，而是后续阶段计划：

- Phase 3 前尚无生产检索链：Dense/BM25、RRF、Cross-Encoder、Parent 聚合与 EvidenceBuilder 未实现。
- QueryGraph、Fast RAG、复杂问题 ResearchAgentLoop、动态 Todo、Tool Use、Subagent 和 Context Compact 在 Phase 4。
- `mem0ai==2.0.12` 已锁定依赖，但长期记忆边界、检索/写入策略在 Phase 4 Task 2 接入。
- Evidence Grader 与 Faithfulness Audit 必须是强制 Graph Node，在 Phase 4 完成。
- Query 原始消息、Tool Event、SSE、反馈 API 和完整审计链在 Phase 4。
- Recall@K、NDCG@K、MRR、AgentLoop 轨迹、Ragas、在线指标和恢复/负载验收在 Phase 5。
- Milvus 只有可替换向量存储边界，本轮仍使用 Elasticsearch，不做双写。
- V1 只有 `user_id` 数据隔离，没有鉴权系统，默认用户为 `default_user`。

## 7. 下一次开发的准确起点

下一任务是路线图 Task 13 / Phase 3 Task 1：

**Retrieval Contracts and Server-Side Filter Builder**

开始前：

1. 进入实现 worktree，确认分支正确、工作树干净，且 `55a132e` 是 HEAD 的祖先。
2. 激活或通过 `conda run -n agentic-rag` 使用项目环境。
3. 配置独立测试 MySQL/Redis/Elasticsearch，运行 Phase 2 真实服务门禁。
4. 完整阅读 Phase 3 计划与全局约束。
5. 使用 `superpowers:subagent-driven-development`，为 Phase 3 Task 1 生成独立 brief，记录 BASE=恢复时的当前 HEAD。
6. 严格执行 TDD、实现报告、独立 reviewer、最多五轮 fix/re-review；不要跳到 Agent Loop。

Phase 3 的实现顺序不得交换：

```text
Retrieval Contracts + Filter Builder
  -> ES Dense/BM25 Adapters
  -> RRF + Cross-Encoder Reranking
  -> Parent Aggregation/Fetch
  -> RetrievalPipelineGraph + Degradation
  -> EvidenceBuilder + Packed Context
```

Phase 3 必须继续遵守：

- `user_id`、`search_type` 等 metadata 只能作为 BM25/Dense 的服务端必要 filter，不单独形成 metadata retrieval tool。
- Dense、BM25、Hybrid、Rerank、Document Fetch 是原子工具/适配能力；固定检索执行链由 Retrieval Subgraph 强制编排。
- Parent 存 MySQL，Child 及向量存 ES；先召回 Child，再按 pointer/version gate 获取 Parent。
- Cross-Encoder 使用 `BAAI/bge-reranker-v2-m3`，进程内只加载一次。
- 保留 Milvus 替换 port，但不在当前迭代实现 Milvus。
- 确定性检索门禁包含 Recall@6、NDCG@10 和 MRR。

## 8. 权威文档索引

- [生产级 Agentic RAG 总体设计](./superpowers/specs/2026-08-04-production-agentic-rag-design.md)
- [五阶段实现路线图](./superpowers/plans/2026-08-04-agentic-rag-implementation-roadmap.md)
- [Phase 1 计划](./superpowers/plans/2026-08-04-agentic-rag-phase-1-foundation.md)
- [Phase 2 计划](./superpowers/plans/2026-08-04-agentic-rag-phase-2-ingestion.md)
- [Phase 3 计划](./superpowers/plans/2026-08-04-agentic-rag-phase-3-retrieval.md)
- [Phase 4 计划](./superpowers/plans/2026-08-04-agentic-rag-phase-4-query-runtime.md)
- [Phase 5 计划](./superpowers/plans/2026-08-04-agentic-rag-phase-5-evaluation-operations.md)
- [Phase 2 Task 5 实现报告](../.superpowers/sdd/2026-08-04-agentic-rag-phase-2-ingestion/task-5-report.md)
- [Phase 2 Task 6 实现报告](../.superpowers/sdd/2026-08-04-agentic-rag-phase-2-ingestion/task-6-report.md)

## 9. 恢复开发时的第一条提示词建议

```text
继续开发 AgenticRAG。先完整阅读 docs/development-progress.md、总体设计、
实现路线图和 Phase 3 计划；确认当前 worktree/branch/HEAD 与进度快照一致，
运行必要基线，然后使用 Subagent-Driven Development 从 Phase 3 Task 1 开始。
不要重做 Phase 1/2，不要提前实现 Phase 4 Agent Loop。
```
