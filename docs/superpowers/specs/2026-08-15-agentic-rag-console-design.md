# Agentic RAG 控制台页面设计规格

## 1. 目标与范围

本规格定义一个由 FastAPI 直接托管的 Agentic RAG 控制台页面，用于本地运行、联调和发布前验收。页面不替代后端的租户、证据、审计和权限边界；它只展示后端已经确认的状态，并通过现有 API 发起操作。

本次范围包括：

- 查询输入、Query Run 状态、SSE 事件时间线和最终答案；
- 证据、引用覆盖率、审计结果、运行时快照和 client provenance；
- 文档上传、摄取任务状态和失败重试提示；
- 当前用户的 Mem0 记忆列表和删除操作；
- MySQL、Redis、Elasticsearch、Reranker、Checkpoint、Artifact 和 Mem0 健康状态；
- 降级、熔断、重试、模型 repair 耗尽、子 Agent 未接入、Todo 未创建和研究预算耗尽的明确提示。

不在本次范围：用户注册、完整身份认证、RBAC、跨用户管理后台、生产级多租户切换和复杂报表编辑器。

## 2. 实现方案

采用原生 HTML/CSS/JavaScript，由 FastAPI 提供同源静态页面。理由是当前项目没有前端构建链，原生页面可以直接在 `agentic-rag` Conda 环境启动，避免引入 Node 依赖和额外部署产物；页面状态只通过 JSON/SSE API 更新，未来仍可迁移为 React 客户端。

页面资源建议放在：

```text
src/agentic_rag/api/static/
  index.html
  app.css
  app.js
```

FastAPI 新增 `GET /` 和 `/static/*`，复用现有同源 API。页面不把 `user_id` 放入请求体，用户作用域仍由服务端 Settings 注入。

## 3. 页面结构

```text
┌────────────────────────────────────────────────────────────┐
│ Agentic RAG │ snapshot │ Mem0 │ DeepSeek │ 系统健康        │
├──────────────┬───────────────────────────────┬─────────────┤
│ 查询         │ 对话与 Query Run               │ 证据/审计   │
│ 文档         │ 问题输入                       │ 引用列表     │
│ 记忆         │ 流式事件时间线                 │ 覆盖率       │
│ 运维         │ 最终答案                       │ 审计状态     │
│              │ 降级/熔断提示                  │ provenance  │
└──────────────┴───────────────────────────────┴─────────────┘
```

### 3.1 查询工作区

- 使用 `POST /v1/query` 发起同步等待；超时后保留 `run_id`，不创建第二个 Run；
- 使用 `GET /v1/query-runs/{run_id}` 刷新状态；
- 使用 `GET /v1/query-runs/{run_id}/events` 建立 SSE，保存最后事件 ID 并支持断线重连；
- 展示 `queued`、`running`、`completed`、`cancelled`、`failed` 和业务拒答终态；
- 展示 `QUERY_ROUTED`、`RETRIEVAL_COMPLETED`、`MEMORY_LOADED`、`EVIDENCE_GRADED`、`FAITHFULNESS_AUDITED`、`CITATION_VALIDATED`、`ANSWER_FINALIZED`；
- 将 `RETRIEVAL_DEGRADED`、`CIRCUIT_OPEN`、`MODEL_REPAIR_EXHAUSTED`、`WORKER_DLQ`、`LEASE_LOST`、`COMPONENT_DEGRADED` 渲染为醒目的状态卡；
- 答案卡必须同时显示 `audited`、citation coverage、引用 Parent ID、route、snapshot ID 和 client provenance；
- `research_action_invalid`、`audit_failed`、`cannot_answer` 等拒答结果显示为安全拒答，不伪造答案。

### 3.2 文档工作区

- 拖拽或选择 PDF、TXT、Excel 文件；
- 调用 `POST /v1/documents`；
- 轮询 `GET /v1/ingestion-jobs/{job_id}`，显示 queued/running/completed/failed；
- 失败时展示有限错误代码和 retryable，不展示解析器原始异常或文件内容；
- 成功后提示用户等待活动索引代际可检索，再发起查询。

### 3.3 Mem0 工作区

- 调用 `GET /v1/memories` 展示当前服务端用户作用域的记忆；
- 调用 `DELETE /v1/memories/{memory_id}` 删除；
- Mem0 unavailable 时显示“记忆服务降级”，并指向 `/health/ready` 与 `memory_provider_degraded` 日志；
- 不把空列表解释为“没有记忆”：服务不可用必须显示错误状态。

### 3.4 运维工作区

- 调用 `/health/live` 和 `/health/ready`；
- 展示 MySQL、Redis、Elasticsearch、Artifact、Checkpoint、Reranker、Mem0；
- 展示当前 `runtime_config_snapshot_id`、DeepSeek protocol、模型 ID、索引代际；
- 展示最近安全事件的类型、组件、reason、outcome、retryable 和 attempt；
- 不展示 prompt、隐藏推理、原始工具载荷、Authorization、API key 或 provider 原文。

## 4. 数据流与后端联调

```text
页面输入
  -> POST /v1/query
  -> MySQL Run + Query Outbox
  -> Redis Query Stream
  -> Query Worker
  -> QueryGraph / Retrieval / Mem0 / ModelGateway / Audit
  -> MySQL Run + AgentEvent
  -> SSE / GET Run
  -> 页面答案、证据和状态
```

页面与后端使用同源部署，避免 CORS 和快照来源不一致。每个页面会话只保存 `run_id`、最后 SSE 事件 ID 和有限的 UI 状态；不会保存完整 prompt、memory text 或模型原文到 LocalStorage。

## 5. 降级和错误交互

| 后端结果 | 页面行为 |
|---|---|
| `202` | 显示“已排队”，进入 SSE/状态轮询 |
| `200 + completed` | 显示答案、证据、审计和 provenance |
| `503` | 显示不可用组件、retryable 和重试按钮 |
| `RETRIEVAL_DEGRADED` | 显示检索降级，不宣称答案可信度提高 |
| `CIRCUIT_OPEN` | 显示熔断冷却中，禁止高频自动重试 |
| `MODEL_REPAIR_EXHAUSTED` | 显示结构化响应失败，建议稍后重试或改写问题 |
| `research_action_invalid` | 显示研究动作不符合 schema 的安全拒答 |
| `audit_failed` | 显示答案未通过审计，不展示草稿答案 |
| `WORKER_DLQ` | 显示任务进入死信，需要运维处理 |

页面提示只能使用后端白名单字段；如果事件未知，统一显示“进度更新”。

## 6. 当前 Query Runtime 缺口的可视化要求

页面必须诚实呈现以下待解决状态，不能把它们显示为成功：

1. `SubagentDispatcher` 未注入时，委派按钮显示“子 Agent 尚未接入生产组合”，并保留 `retryable=false`；
2. Todo 集合为空且研究动作要求拆分时，显示“当前链路尚未创建 Todo”，而不是显示委派成功；
3. 全局 `research_attempt_count` 超过上限时，显示“研究预算耗尽”，并关联 Run 事件和终态原因。

这些状态在后端缺口修复后仍保留兼容枚举，避免页面依赖具体异常字符串。

## 7. 测试与验收

新增测试：

- API 页面路由和静态资源返回 200；
- 查询页面使用真实 `run_id`、SSE 事件和 `Last-Event-ID` 重连；
- 202、200、503、取消、业务拒答和未知事件脱敏；
- 文档上传与摄取状态轮询；
- Mem0 unavailable 不显示空成功；
- snapshot/provenance 在答案卡中一致；
- 降级、熔断、repair exhausted、子 Agent 未接入和研究预算耗尽提示可见；
- 真实 Graph/API 验收仍由 `scripts/run_real_query_acceptance.py` 和 `verify_acceptance.py` 完成，页面不能替代泄漏、引用、审计、恢复和备份门禁。

实现完成后必须运行：

```bash
conda run -n agentic-rag pytest --import-mode=importlib tests -q
conda run -n agentic-rag ruff check src tests evals scripts
conda run -n agentic-rag mypy src evals
```

