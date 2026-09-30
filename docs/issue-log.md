# Agentic RAG 问题与修改日志

> 更新日期：2026-08-28
>
> 本文档是继续开发、复现故障和发布验收的事实入口。记录以当前代码、测试、MySQL 事件和实际运行输出为准；`已修复`表示代码和回归测试已经覆盖，`已诊断`表示根因已经确认但不一定改变代码，`待处理`表示不能被本地通过状态替代的生产门禁。

## 1. 当前交接快照

| 项目 | 当前值 |
|---|---|
| 仓库 | `/Users/steven/LzmWorkSpace/AgenticRAG` |
| 分支 | `main` |
| HEAD | `679f4f2` — `fix: respect Qwen embedding batch limits` |
| Python 环境 | Conda `agentic-rag`；项目不要求同时启用 `.venv` |
| 工作树 | 存在未提交改动，主要涉及 ES 活动 Alias、MySQL UTC、Parent/Child 审计、LLM 诊断和对应测试；继续开发前必须先审阅 `git diff` |
| 最近静态检查 | Ruff 通过；Mypy 通过（115 个源文件） |
| 最近单元/集成回归 | `642 passed, 39 skipped, 19 warnings`（E2E 未纳入该命令） |
| 最近服务检查 | `/health/live` 为 `live`；`/health/ready` 的 configuration、MySQL、Redis、Elasticsearch、artifacts、checkpoints、reranker、memory 均为 `available` |

五个阶段、Query Runtime/Mem0/真实评测后续任务和同源控制台已经实现；当前开发重心是未提交补强的整理、真实验收复跑以及生产上线审批，而不是重新实现阶段 1–5。

## 2. 已完成的主线能力

- 阶段 1–5 已覆盖持久化、Outbox/Redis、SQLite checkpoint、文档安全摄取、Parent/Child 切块、Dense/BM25/RRF/Rerank、EvidenceBuilder、QueryGraph、审计、备份恢复和离线/在线评测。
- Query Outbox 在 MySQL 同事务中保存 Run 与投递意图；Query Worker 负责 Redis 领取、租约、心跳、checkpoint 恢复、重试/DLQ 和终态 ACK。
- 生产组合根已注入 `SubagentDispatcher`，研究路径会 Todo 初始创建和 Todo 追加；`research_attempt_count` 跨 Graph 重入持久化并受当前 snapshot 的全局上限约束。
- Mem0 默认启用，使用用户命名空间和 `infer=False`；同源控制台支持上传、入库状态、查询/SSE、证据、审计、Memory 和降级提示。
- 真实验收脚本支持 Graph/API、当前 `RuntimeConfigSnapshot`、`client_provenance=real_query_api`、Mem0、泄漏/引用/审计/恢复/备份门禁；`fixture` 结果只用于冒烟。

## 3. 问题修改记录

### 3.1 已修复（Elasticsearch 活动 Alias、MySQL UTC、审计与 LLM 诊断字段）

| 编号 | 现象与根因 | 修改与证据 | 状态 |
|---|---|---|---|
| FIX-001 | 物理 Child 索引存在但 `agenticrag-children-active` 缺失，必须每次手工创建 Alias，导致查询无法命中。 | `ElasticsearchChildIndexStore.ensure_active_alias()` 在 Query Worker 启动时只修复缺失 Alias；Publisher 用一次原子 Alias 更新切换代际；指向其他代际时 fail-closed，并记录 `elasticsearch_active_alias_repaired/pending`。覆盖 `tests/unit/persistence/test_elasticsearch_staging.py`。 | 已修复（当前工作树未提交）；历史环境可重启验证 |
| FIX-002 | MySQL `DATETIME(6)` 无时区，实例/会话时区按本地配置解释，出现约 8 小时误差。 | `create_mysql_engine()` 为每个 asyncmy 连接注入 `SET time_zone = '+00:00'`；运维手册补充 `SET PERSIST time_zone = '+00:00'`、检查 SQL 和历史数据处理原则。覆盖 `tests/unit/persistence/test_mysql.py`。 | 已修复（当前工作树未提交）；历史值不做盲目加减 |
| FIX-003 | Qwen/DashScope embedding 单次提交过大时返回 batch size 错误，入库任务停在失败/重试。 | HEAD `679f4f2` 将 embedding 请求按最多 10 条拆批，并补充批量边界测试。 | 已修复 |
| FIX-004 | 生产组合根创建 `ResearchAgentLoop` 时没有注入 `SubagentDispatcher`，`delegate_research` 只有定义没有实际执行。 | 组合根创建真正的 `SubagentDispatcher`，与 QueryGraph/Query Worker 共享 `ConcurrencyManager`，子 Agent 使用受限 ResearchToolset 和 evidence reducer；覆盖 `tests/unit/runtime/test_query_composition.py` 及 Subagent 测试。 | 已修复 |
| FIX-005 | 正常研究链路只有 `update_todos`，没有服务器生成根 Todo，导致任务拆分无法开始。 | 研究开始时由服务端按问题创建根 Todo；合法动作可追加 Todo，所有者、标题、ID 和动作 schema 后端校验；控制台只显示校验后的状态。 | 已修复 |
| FIX-006 | “4 轮”原来是单次 `ResearchAgentLoop` 调用上限，Grader 重入后计数归零，整个 Query 没有全局预算。 | `research_attempt_count` 写入 QueryState/SQLite checkpoint，跨 Graph 重入累计；达到 snapshot 上限就阻止未完成 Todo、记录 `research_round_limit` 并停止模型调用。 | 已修复 |
| FIX-007 | Parent 授权记录保存父级 locator，而答案 Evidence 使用 Child locator；审计按字符串完全相等比较，合法 Parent/Child 引用被误拒。 | Citation Validator 增加结构化 AST locator 的范围、block、canonical path 和 span 包含关系校验，同时保留双方一致的旧 opaque locator；补充“子块在父块内通过/越界拒绝”测试。 | 已修复（当前工作树未提交）；历史 `audit_failed` 记录不回写 |
| FIX-008 | 用户查询“刘泽明教育经历”被提示检索降级且无答案，容易误判为 ES 没有索引。 | Trace `01a045b2-2bc0-73a9-abbc-c8dd9edd06af` 的 checkpoint 和 Artifact 实际包含 3 个已验证 Parent Evidence；失败发生在后续 LLM 重试耗尽，不是 ES 空结果。检索降级与模型不可用必须分开看。 | 已诊断；新事件字段补强在当前工作树 |
| FIX-009 | MySQL `agent_events` 中多次 `MODEL_RETRY` 的 attempt 看起来重复，无法区分同一 graph node 的不同逻辑调用。 | LLM 降级事件的安全 `operation` 和 scope 内递增事件序列加入稳定 event key，并将 operation 作为 `node_name`；避免 MySQL 去重覆盖不同调用。历史事件不重写。 | 已修复（当前工作树未提交） |
| FIX-010 | `provider_outage` 只有应用层总类，旧事件没有错误类型、HTTP 状态、客户端超时和 provider request id，无法还原 DeepSeek 的精确故障。 | `ModelGateway` 现在记录白名单字段 `requested_model`、`protocol`、`client_timeout_seconds`、`error_class`、`http_status`、`provider_request_id`；logging、SSE 和控制台同步白名单，拒绝原始异常文本和疑似凭证。 | 已修复（当前工作树未提交）；新 Trace 才具备完整诊断 |
| FIX-011 | DeepSeek 结构化响应偶尔不符合节点 schema；如果把格式错误当网络错误重试，会造成原因混淆或泄漏原始响应。 | `ModelGateway` 固定协议选择、Pydantic schema 校验和一次 repair；区分 `protocol_error`、`model_schema_invalid`、`provider_outage`、`model_unavailable`、`circuit_open`，最终 fail-closed。 | 已修复 |
| FIX-012 | 备份目录丢失后无法仅靠页面恢复；操作员不知道需要手工准备哪些目录和变量。 | 本地手册明确 SQLite checkpoint、Artifact、MySQL dump、ES 代际/别名、配置快照的备份边界，并要求停止 Worker 后校验清单哈希；恢复前检查版本和 dead stream。 | 已记录运行手册；仍需按发布门禁演练 |

### 3.2 运维与环境问题

| 编号 | 现象与根因 | 当前约定 | 状态 |
|---|---|---|---|
| OPS-001 | 入库/查询任务长期 `queued`、页面无进度，常见原因是只启动了 API，或把三个阻塞进程写在同一段串行命令中，Worker 根本没有启动。 | Elasticsearch、MySQL、Redis、Alembic 之后，API、Query Worker、Ingestion Worker 必须在三个独立终端启动；以 `/health/ready` 和 Worker 生命周期日志确认。 | 已记录并固化到运行手册 |
| OPS-002 | 终端同时出现 Conda `agentic-rag` 和 `.venv`，容易误以为需要双环境。 | 项目统一使用 Conda `agentic-rag`；`.venv/` 和 `.env.*` 已忽略。`src/agentic_rag/config.py` 是版本化代码，不应为隐藏密钥而加入忽略；秘密放在 `.env.local`。 | 已明确 |

### 3.3 已诊断但不应误判为代码故障

#### Trace `01a045b2-2bc0-73a9-abbc-c8dd9edd06af`

- 终态是业务 `completed`，答案载荷为 `cannot_answer`，并非“已审计通过的答案”。
- 事件顺序显示：`MODEL_RETRY`（attempt 1）→ `LLM_COMPLETED`（attempts 2）→ 再次 `MODEL_RETRY`（attempt 2）→ `LLM_COMPLETED`（attempts 4）→ `MODEL_RETRY_EXHAUSTED`（attempt 3，`model_unavailable`）。
- 当前 DeepSeek client 使用约 30 秒 timeout、`max_retries=0`，重试由 `ModelGateway` 统一执行；旧 telemetry 没有 `error_class/http_status/provider_request_id`，因此只能确认是瞬态 provider/transport 边界，不能声称是服务提供方“确定宕机”。
- 对照 Trace `01a045b2-0d33-7679-82db-1254b1bc8fbd` 时必须注意它是另一条独立 Run，不能用来替代当前 Trace 的事件序列。

#### `agent_events` 的定位边界

`agent_events` 是关键生命周期/Graph/降级事件时间线，不是完整应用日志，也不直接保存 prompt、隐藏推理、工具载荷或 provider 原文。事件属性通常通过 `payload_ref` 指向 Artifact；缺失单条事件时要同时检查 Worker 日志、Run、Outbox 和 checkpoint，而不能把“无事件”当成“节点未执行”。

## 4. 当前待处理问题

| 优先级 | 待处理项 | 完成定义 |
|---|---|---|
| P0 | 生产鉴权/RBAC | 接入真实身份认证、授权、角色/资源权限；不能把 `default_user` 或 `user_id` 命名空间隔离当作生产鉴权。 |
| P1 | reranker 分数标定 | 明确 `retrieval_score`、`rrf_score`、`rerank_score` 的传播；按模型/revision、score activation、索引代际和评测切片版本化 `min_rerank_score`，回归 Recall/Precision/NDCG、误拒/误接收和引用覆盖率。 |
| P1 | 每次发布复跑真实门禁 / Evaluation PASS | 使用当前 snapshot、真实 Graph/API、`real_query_api` provenance、Mem0、恢复和备份重新生成脱敏 summary，再运行 `scripts/verify_acceptance.py`；`Evaluation PASS` 不能用旧 summary 或 fixture 代替。 |
| P2 | 历史 Trace 诊断缺口 | 旧事件无法补回 provider 错误类型/状态/request id；只能保留原样并在新 Trace 验证字段，不对历史事件做猜测性回填。 |
| P2 | 跨进程 trace/span 关联 | 当前 `trace_id` 默认等于 `run_id`，Span 记录仍以进程内 TraceRecorder 为主；若需要跨 API、Outbox、Worker、Graph 的完整链路，再设计持久化 span/trace 方案。 |
| P2 | 当前工作树整理 | 对未提交改动逐组 review（Alias、UTC、审计、诊断、测试），拆分清晰提交并重新跑静态/回归/真实验收；在此之前不要声称 `main` 是干净发布基线。 |

## 5. 按 Trace ID 排查

1. 先确认 Run 和最终状态：

   ```sql
   SELECT id, user_id, thread_id, checkpoint_thread_id, status, route,
          attempt_count, error_code, termination_reason,
          runtime_config_snapshot_id, created_at, started_at, finished_at,
          result_ref, answer
   FROM agent_runs
   WHERE id = '<trace_id>';
   ```

2. 再按持久化顺序查看关键事件。通常 `trace_id` 默认等于 `run_id`；用 `run_id` 查询可以命中 `(user_id, run_id, id)` 索引：

   ```sql
   SELECT id, trace_id, run_id, event_type, node_name, summary,
          payload_ref, runtime_config_snapshot_id, created_at
   FROM agent_events
   WHERE run_id = '<run_id>'
   ORDER BY id;
   ```

3. 检查 Query Outbox 是否仍在投递、重试或死信：

   ```sql
   SELECT id, aggregate_id, status, attempt_count, next_attempt_at,
          created_at, dispatched_at
   FROM task_outbox
   WHERE aggregate_type = 'query_run' AND aggregate_id = '<run_id>';
   ```

4. 如果事件有 `payload_ref='artifact://...'`，把它解析到 `var/artifacts/` 后只读检查 JSON；不要把原始 prompt、Memory 原文或 provider 响应复制到日志/工单。随后用 `checkpoint_thread_id` 检查 `var/query_checkpoints.sqlite` 的 `checkpoints`/`writes`，必要时使用 LangGraph 解码器恢复 QueryState。
5. 时间按 UTC 解释；上海本地显示需要明确加 8 小时，不能直接修改 MySQL 历史 `DATETIME`。
6. 重点对照 `runtime_config_snapshot_id`、`event_type`、`node_name/operation`、`reason`、`attempt`、`error_class` 和 `http_status`。`provider_outage` 先按瞬态错误总类处理，只有安全字段支持时才下结论。

## 6. 后续开发顺序

1. 复核当前工作树并按功能边界提交；先保持现有 API、事件名、状态枚举和快照兼容。
2. 用隔离 MySQL、Redis、Elasticsearch、Mem0、SQLite checkpoint 和 Artifact 重跑真实 Graph/API 全链路，保留脱敏 summary、SSE 和事件证据。
3. 实施生产鉴权/RBAC；随后完成 reranker 标定和 Evaluation PASS 回归。
4. 根据运维需求决定是否增加跨进程 OpenTelemetry/span 持久化；不要为了补历史 Trace 而修改不可变事件。

## 7. 关联文档

- [开发进度快照](./development-progress.md)
- [本地运行手册](./local-operations.md)
- [总体实现路线图](./superpowers/plans/2026-08-04-agentic-rag-implementation-roadmap.md)
- [生产级总体设计](./superpowers/specs/2026-08-04-production-agentic-rag-design.md)
- [控制台与 Query Runtime 计划](./superpowers/plans/2026-08-15-agentic-rag-console.md)
