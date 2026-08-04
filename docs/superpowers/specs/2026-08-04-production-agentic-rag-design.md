# 生产级 Agentic RAG 设计文档

> 日期：2026-08-04  
> 状态：已完成交互式设计，等待书面审阅  
> 实现语言：Python  
> 编排框架：LangGraph

## 1. 摘要

本项目构建一个本地启动、面向企业文档问答和复杂研究任务的生产级 Agentic RAG 系统。系统同时支持：

- 简单问题的传统 RAG 快路径；
- 多跳、多轮检索和动态计划；
- 并行 Research Subagent；
- PDF、扫描 PDF、文本和 Excel 摄取；
- Parent-Child 混合检索；
- 长期记忆与会话工作记忆；
- 强制证据、忠实度和引用审核；
- Checkpoint 恢复、幂等重放、有限重试和降级；
- 原始消息、Tool Event 和审核事件审计；
- 请求内审核、在线质量监控与离线 RAG/AgentLoop 基准评测。

系统遵循三层职责边界：

1. QueryGraph 控制不能由模型随意决定的宏观流程。
2. ResearchAgentLoop 动态规划检索方向和 Tool 使用顺序。
3. Tools 与确定性 Subgraph 执行边界清晰的原子能力。

设计原则是“生产级，但不过度设计”：V1 实现当前正确性、隔离、恢复和评测所需能力；未来能力通过少量接口预留，不提前实现复杂基础设施。

## 2. 目标与非目标

### 2.1 V1 目标

- 用户少、无独立鉴权系统时，仍通过 `user_id` 实现数据和记忆隔离。
- 支持默认用户 `default_user`，但所有数据访问都必须显式携带用户范围。
- 使用 LangGraph 实现 QueryGraph、ResearchAgentLoop、RetrievalPipelineGraph 和 IngestionGraph。
- 使用 DeepSeek 主模型和轻量模型完成路由、Agent、生成和审核。
- 使用 ES 完成 Dense 与 BM25 检索，MySQL 保存 Parent 和审计数据。
- 使用 Docling 统一文档解析结果并建立 Canonical AST。
- 使用 mem0ai 作为长期记忆实现，不复制或重写 Mem0 核心。
- 使用 Redis Streams 实现简单、可恢复的本地摄取任务队列。
- 提供请求内质量保护、在线运行指标与反馈，以及可复现的离线检索、生成和 AgentLoop 评测。

### 2.2 V1 非目标

- 不使用 Docker、Kubernetes 或微服务拆分。
- 不使用知识图谱，也不实现 Knowledge Graph Tool。
- 不允许 Agent 直接执行 SQL，不提供 `query_mysql` Tool。
- 不将 Metadata Search 作为独立召回通道。
- 不实现 Token 或成本预算决策节点。
- 不实现 Milvus Adapter，只保留 `VectorIndex` 接口。
- 不实现 RBAC、完整 HITL UI、事件总线、缓存体系、自动扩缩容或复杂熔断平台。
- 不实现复杂的记忆衰减算法或重新实现 Mem0 的存储、检索与冲突处理。

## 3. 技术栈

| 能力 | V1 选择 |
|---|---|
| 语言 | Python |
| API | FastAPI |
| Graph / Agent 编排 | LangGraph |
| 主模型 | DeepSeek `deepseek-v4-pro` |
| 轻量模型 | DeepSeek `deepseek-v4-flash` |
| Embedding | Qwen `text-embedding-v3`，1024 维 |
| Reranker | `BAAI/bge-reranker-v2-m3` Cross-Encoder |
| 文档解析 | Docling |
| 向量与 BM25 | 本地 Elasticsearch，`localhost:9200` |
| Parent 与审计数据 | 本地 MySQL |
| Checkpointer | SQLite |
| 摄取任务通知 | 本地 Redis Streams，`127.0.0.1:6379` |
| 长期记忆 | `mem0ai==2.0.12`，进程内库接入 |
| RAG 评测 | Ragas + 确定性检索指标 |

ES 本地启动方式：

```bash
cd ~/Downloads/elasticsearch-8
./bin/elasticsearch
```

## 4. 总体架构

```mermaid
flowchart TB
    U["Client"] --> API["FastAPI"]

    API --> QG["QueryGraph<br/>宏观流程与强制审核"]
    QG --> RA["ResearchAgentLoop<br/>动态计划与 Tool Loop"]
    RA --> RT["retrieve_evidence"]
    RT --> RPG["RetrievalPipelineGraph<br/>确定性混合检索"]

    RPG --> ES["Elasticsearch<br/>Child + Dense + BM25"]
    RPG --> MYSQL["MySQL<br/>Parent + Audit"]

    API --> REDIS["Redis Streams<br/>摄取任务通知"]
    REDIS --> IW["Ingestion Worker"]
    IW --> IG["IngestionGraph"]
    IG --> DOCLING["Docling + Canonical AST"]
    IG --> ES
    IG --> MYSQL

    QG --> MEM["MemoryService<br/>mem0ai"]
    MEM --> MEMES["ES agent_memories_v1"]

    QG --> QCP["query_checkpoints.sqlite"]
    IG --> ICP["ingestion_checkpoints.sqlite"]
```

### 4.1 三层职责

#### 第一层：Graph

Graph 固定控制：

- 请求路由；
- Research 循环上限；
- 节点超时与有限重试；
- 强制审核；
- 错误降级与拒答路径；
- Checkpoint 持久化；
- 最终结束条件。

V1 不实现 Token 或成本预算判断，但保留固定安全上限。

#### 第二层：ResearchAgentLoop

AgentLoop 动态决定：

- 如何拆解问题；
- Todo 如何创建、修改、拆分和合并；
- 下一步缺少什么证据；
- 使用哪个检索请求；
- 是否并行委派 Research Subagent；
- 是否改变检索方向；
- 何时提交证据或无法回答。

#### 第三层：Tools 与确定性 Subgraph

Agent 可见的动作分为两类。

Tool 调用：

- `update_todos`
- `retrieve_evidence`
- `delegate_research`
- `calculator`

终止动作：

- `submit_evidence`
- `cannot_answer`

其中 `retrieve_evidence` 是 Agent 视角的高层 Tool，内部由 RetrievalPipelineGraph 实现。

Agent 不直接看到：

- `dense_search`
- `bm25_search`
- `rrf_fusion`
- `rerank`
- `parent_fetch`
- ES 或 MySQL 客户端

这些能力是 RetrievalPipelineGraph 的内部 Node 或 Port。

## 5. QueryGraph

```mermaid
flowchart TD
    S["START"] --> MC["MemoryContextLoader"]
    MC --> R["意图与复杂度路由"]

    R -->|简单问题| F["Fast RAG"]
    R -->|复杂问题| A["ResearchAgentLoop"]

    F --> EG["Evidence Grader<br/>强制 Node"]
    A --> EG

    EG -->|充分| G["Generate"]
    EG -->|Fast 路径证据不足| A
    EG -->|复杂路径证据不足| A
    EG -->|需要用户信息| C["Clarify"]
    EG -->|无法回答| N["Refuse"]

    G --> FA["Faithfulness Audit<br/>强制 Node"]
    FA -->|通过| CV["Citation Validator<br/>强制 Node"]
    FA -->|失败且未修正| G
    CV -->|通过| O["Finalize"]
    CV -->|失败且未修正| G
    G -->|已修正一次仍失败| N
```

Clarify 与 Refuse 路径不生成知识性答案，因此不进入 Faithfulness 与 Citation 审核；所有正常答案必须经过三个审核阶段。

### 5.1 路由

路由使用 `deepseek-v4-flash` 输出结构化结果：

```text
route: fast_rag | research
normalized_query: string
reason_code: string
```

V1 不设置单独的结构化 SQL 路由。结构化 Metadata 只作为 Dense 与 BM25 的 Filter；MySQL 只由内部 Repository 获取 Parent。

### 5.2 Fast RAG

Fast RAG 只执行一次 RetrievalPipelineGraph。Evidence Grader 判定不足时自动升级到 ResearchAgentLoop，而不是直接生成低质量答案。

### 5.3 状态

`QueryState` 只保存 JSON 可序列化数据：

```text
request
messages
memory_context
route
research
evidence
answer
audit_results
revision_count
errors
termination_reason
```

`ResearchState`：

```text
plan_revision
todos
discovered_entities
retrieval_history
evidence_ids
last_observation
submitted
```

State 不保存数据库连接、ES Client、DoclingDocument、Embedding 数组、Cross-Encoder 对象或完整 Parent 大文本。大对象写入存储或本地 Artifact，State 只保留引用。

## 6. ResearchAgentLoop 与 Multi-Agent

```mermaid
flowchart TD
    IN["Research Context"] --> AG["Research Agent<br/>deepseek-v4-pro"]
    AG -->|update_todos| TD["Todo Reducer"]
    AG -->|retrieve_evidence| RP["RetrievalPipelineGraph"]
    AG -->|delegate_research| SA["并行 Research Subagents"]
    AG -->|calculator| CA["Calculator"]
    AG -->|submit_evidence| OUT["Evidence Submission"]
    AG -->|cannot_answer| NA["Cannot Answer"]

    TD --> OBS["Observation"]
    RP --> OBS
    SA --> MERGE["Evidence Reducer"]
    CA --> OBS
    MERGE --> OBS
    OBS --> CC["Context Builder / Compact"]
    CC --> AG
```

这是一个真正的 Agent Loop：每一轮都由 LLM 根据当前观察选择动作，Tool 或 Subgraph 返回 Observation 后再次进入 Agent，而不是按固定检索步骤一次性结束。

### 6.1 动态 Todo

Todo 状态：

```text
pending | in_progress | completed | blocked | skipped
```

规则：

- Agent 可以在循环内首次创建计划，也可以根据证据动态修改计划。
- Todo 可以新增、拆分、合并、重排、跳过或标记阻塞。
- `completed` 必须包含 Evidence ID 或明确的非检索结果引用。
- 依赖项完成前不能进入 `in_progress`。
- Todo Reducer 拒绝依赖环和非法状态转换。
- Subagent 只能修改分配给自己的 Todo；Supervisor 管理全局 Todo。
- Todo 变化以 `TODO_UPDATED` Event 追加记录。

不在进入 AgentLoop 前设置独立静态 Planner Node，避免计划与检索观察脱节。

### 6.2 并行 Research Subagent

- Supervisor 仅在多个 Todo 相互独立时使用 LangGraph `Send` 并行启动 Subagent。
- Subagent 使用同一个受限 Research Loop，但只接收自己的问题、Filter、记忆摘要和 Evidence Manifest。
- Subagent 可以调用 `retrieve_evidence` 与 `calculator`，不能生成最终用户答案。
- Subagent 返回结构化 Evidence 和 Todo 结果，由 Evidence Reducer 去重合并。
- Query Multi-Agent 不经过 Redis；Redis 只用于后台摄取任务。

### 6.3 Context Compact

每次调用 Research Agent 前构建受控上下文，保留：

- 系统约束与原始问题；
- MemoryContext 摘要；
- 未完成 Todo；
- 最近 Observation；
- Evidence Manifest；
- Evidence Grader 返回的缺口。

较早的 Tool 输出、已完成 Todo 细节和重复检索结果由 `deepseek-v4-flash` 压缩。压缩是上下文窗口安全机制，不是成本预算决策；完整原始消息与 Tool Event 已写入 MySQL，可用于审计和重放。

## 7. RetrievalPipelineGraph

Agent 只提交结构化 `RetrievalRequest`：

```text
query
search_type
document_ids
content_types
date_range
top_k_override
```

`user_id` 与有效版本约束由服务端注入，Agent 不能提供或覆盖。`top_k_override` 只能在 Graph 配置的上下限内调整。

```mermaid
flowchart LR
    REQ["RetrievalRequest"] --> FB["Filter Builder / Validator"]
    FB --> DS["Dense Search"]
    FB --> BS["BM25 Search"]
    DS --> RRF["RRF Fusion"]
    BS --> RRF
    RRF --> CE["Cross-Encoder Rerank"]
    CE --> PA["Parent Aggregation"]
    PA --> PF["MySQL Parent Fetch"]
    PF --> EB["EvidenceBatch"]
```

### 7.1 Filter 规则

以下 Filter 同时作用于 Dense 和 BM25：

- 强制：`user_id`
- 强制：`is_active=true`
- 可选：`search_type`
- 可选：`document_id`
- 可选：`content_type`
- 可选：文档标签与日期范围

不存在第三路 Metadata Recall，也不允许两路召回使用不同的用户范围。

### 7.2 默认检索参数

```text
Dense Top K       = 40
BM25 Top K        = 40
RRF 保留          = 30
Cross-Encoder 输入 = 30
重排后 Child       = 10
每个 Parent Child  = 最多 2 个
最终 Parent Top K   = 6
```

这些是 V1 基线参数，后续通过离线评测调优，不实现在线自动调参。

### 7.3 Cross-Encoder

默认模型：`BAAI/bge-reranker-v2-m3`。

- 本地进程启动时加载一次。
- 通过 `Reranker.rerank(query, candidates)` 接口隔离模型实现。
- Reranker 超时或不可用时使用 RRF 顺序降级，并记录 `rerank_degraded=true`。
- 不单独部署 Reranker 服务。

### 7.4 向量数据库抽象

```text
VectorIndex.search(query_vector, filter, top_k)
LexicalIndex.search(query_text, filter, top_k)
```

V1 两个接口都由 ES Adapter 实现。未来 Milvus 只替换 `VectorIndex`，BM25 可继续留在 ES，Graph 和 Tool 契约保持不变。

## 8. Parent-Child Chunking

采用分层混合策略：

- Parent：Canonical AST 结构语义 Chunking。
- Child：Docling HybridChunker 的结构感知、Token 限制 Chunking。
- 递归或行级切分：仅用于超长原子块兜底。
- V1 不使用 Embedding 或 LLM 相似度驱动的语义边界判断。

### 8.1 Parent 规则

- 标题章节、表格、图片说明和 Excel 逻辑数据区是语义边界。
- 同一标题下相邻 AST 节点依次合并。
- 目标长度 `1200-1800 tokens`，最大约 `2400 tokens`。
- 不跨一级主题边界，不混合表格与正文。
- 跨页连续段落在 Global Assembler 中先合并，再参与 Parent 构建。
- 超大表格按行组拆 Parent，并重复表头。

### 8.2 Child 规则

- Child 不跨 Parent。
- 使用 Docling HybridChunker。
- Contextualized Child 最大 `384 tokens`，为 Query、标题路径和 Cross-Encoder 特殊 Token 留空间。
- 检索文本包含标题路径、表格标题或图片说明以及正文。
- 同标题下过小的相邻块可以合并。
- 表格分块重复表头。
- 默认不增加文本重叠；Parent Fetch 已恢复完整上下文。若离线评测证明存在明显边界召回问题，再为普通长文本增加小幅重叠。

### 8.3 超长原子块兜底

普通文本顺序：

```text
段落 → 换行 → 句末标点 → 分号 → 逗号 → Token 硬切
```

表格、代码与日志优先保持行完整，使用行级 Token Chunking。

## 9. 文档摄取架构

IngestionGraph 独立于 QueryGraph：

```mermaid
flowchart TD
    RAW["原始文档"] --> PARSE["按页或批次 Docling 解析"]
    PARSE --> FRAG["Fragment AST"]
    FRAG --> ASM["Global Assembler"]
    ASM --> CAN["Canonical AST"]
    CAN --> VAL["校验 / 跨页合并 / 去重"]
    VAL --> CHUNK["Parent-Child Chunking"]
    CHUNK --> EMB["Embedding"]
    EMB --> IDX["写 MySQL Parent 与 ES Child"]
    IDX --> PUB["版本发布 active"]
```

### 9.1 Canonical AST

Fragment AST 与 Canonical AST 都以 `DoclingDocument` 为核心数据模型，项目只增加薄 Envelope：

```text
document_id
document_version_id
user_id
source_type
source_uri
content_hash
parser_version
pipeline_version
created_at
```

Canonical AST 必须满足：

- 全局阅读顺序已确定；
- 跨页段落、表格和列表已合并；
- 重复页眉、页脚和 OCR 重复内容已处理；
- 标题层级、页码、Bounding Box 与来源可追踪；
- Schema 与引用完整性校验通过。

Canonical AST 以 JSON Artifact 保存在本地文件系统，MySQL 保存路径和版本信息。

### 9.2 文档类型

- 文本 PDF：Docling 文本与布局解析。
- 扫描 PDF：Docling OCR 管线。
- Excel：转换为工作表、逻辑区域、表格和行级 AST。
- 纯文本：转换为统一 DoclingDocument 结构。

不同文档类型从 Fragment AST 之后共用同一条 Assembler、Chunking 与 Indexing 管线。

### 9.3 Redis Streams 摄取队列

不使用 Celery、Dramatiq、RQ 或 ARQ。

```text
Stream: agenticrag:jobs:ingestion
Group:  agenticrag-ingestion-workers
```

消息只包含 `job_id` 和入队时间；MySQL `ingestion_jobs` 是任务真实状态来源。

Worker 流程：

1. `XREADGROUP` 读取通知。
2. 在 MySQL 中原子 Claim Job 并写 Lease。
3. 周期更新 Heartbeat。
4. 使用固定 `thread_id=ingestion:{job_id}` 运行或恢复 IngestionGraph。
5. 更新 Job 状态后 `XACK`。
6. Lease 过期任务可被其他 Worker 接管。

## 10. 数据存储

### 10.1 MySQL 核心表

#### `documents`

```text
id
user_id
source_type
filename
mime_type
content_hash
status
active_version_id
created_at
updated_at
```

#### `document_versions`

```text
id
document_id
version_no
parser_version
pipeline_version
canonical_ast_path
status: building | active | failed | inactive
created_at
```

#### `parent_chunks`

```text
id
user_id
document_id
document_version_id
ordinal
heading_path
content_type
content
page_from
page_to
ast_locator
content_hash
status
```

#### `ingestion_jobs`

```text
id
user_id
document_id
document_version_id
status
lease_owner
lease_expires_at
heartbeat_at
error_code
attempt_count
created_at
updated_at
```

#### `agent_runs`

保存 Query Run 的用户、Thread、开始结束时间、状态、路由和终止原因。

#### `messages`

保存用户原始消息、最终 Assistant 消息以及必要的公开 Tool Message，不保存隐藏 Chain-of-Thought。

#### `agent_events`

使用通用事件表保存：

```text
TOOL_STARTED
TOOL_COMPLETED
TODO_UPDATED
RETRIEVAL_COMPLETED
EVIDENCE_GRADED
FAITHFULNESS_AUDITED
CITATION_VALIDATED
CONTEXT_COMPACTED
NODE_RETRIED
COMPONENT_DEGRADED
USER_FEEDBACK
```

V1 不为每类事件建立独立表。大 Payload 写入本地 Artifact，事件表只保存摘要与引用。

### 10.2 ES Child 文档

```text
child_id
parent_id
user_id
document_id
document_version_id
version_no
search_type
content
embedding
heading_path
content_type
page_from
page_to
ordinal
content_hash
is_active
```

Keyword Filter 字段使用精确类型，Embedding 维度固定为 1024。

## 11. 记忆系统

### 11.1 四类记忆

| 类型 | V1 实现 |
|---|---|
| Working Memory | LangGraph State + SQLite Checkpoint |
| Semantic Memory | mem0ai |
| Episodic Memory | mem0ai |
| Procedural Memory | mem0ai |

### 11.2 接入方式

- 使用 `mem0ai` Python 开源库的 `AsyncMemory`，进程内接入。
- 通过项目 `MemoryService` 隔离调用方式，不暴露 mem0ai 到 Graph State。
- 使用独立 ES Index：`agent_memories_v1`。
- 使用 `text-embedding-v3` 和相同 `user_id` Namespace。
- 记忆抽取默认使用 `deepseek-v4-flash`。

### 11.3 读取与写入

- `MemoryContextLoader` 在请求开始、路由与检索之前执行一次。
- MemoryContext 以只读摘要传入 Fast RAG、ResearchAgentLoop 和 Subagent。
- V1 不在 AgentLoop 内提供 `search_memory` Tool，避免重复检索和循环行为不稳定。
- Finalize 完成后，由 MemoryService 从本轮公开消息中抽取稳定的用户事实、偏好和程序性信息。
- 文档检索证据、瞬时任务状态和 Tool 输出默认不写为长期记忆。
- 记忆写入失败不影响已经完成审核的答案，但记录降级事件。

### 11.4 生命周期

记忆写入、检索、排序、冲突处理、压缩归纳和删除主要使用 mem0ai 能力。项目薄策略层只负责：

- 用户 Namespace；
- 允许写入的记忆类型；
- Metadata 与来源；
- 对明显敏感或短期内容的过滤；
- 查看、单条删除和按用户清理接口。

不在 V1 重新实现复杂遗忘或衰减算法。

## 12. 强制审核

采用三个强制节点、一个共享修正回路。

### 12.1 Evidence Grader

使用 `deepseek-v4-flash`：

```text
decision: sufficient | insufficient | clarify | refuse
gaps: list[string]
```

- `sufficient`：进入 Generate。
- `insufficient`：将 Gaps 返回 ResearchAgentLoop。
- `clarify`：请求用户补充必要信息。
- `refuse`：知识库不能支持回答。

### 12.2 Faithfulness Audit

使用 `deepseek-v4-flash`：

```text
passed: boolean
issues: list[string]
```

只检查答案中的事实性主张是否能被 Evidence 支持。失败时将 Issues 返回 Generate，基于原 Evidence 删除或改写内容，不触发隐式检索。

### 12.3 Citation Validator

使用确定性代码检查：

- Evidence ID 是否存在；
- Evidence 是否属于当前用户和有效版本；
- 引用格式是否正确；
- 知识性答案是否包含有效引用。

Citation Validator 不重复判断语义支撑关系。

### 12.4 修正

```text
max_answer_revisions = 1
```

Faithfulness 或 Citation 失败统一返回 Generate，不建立独立 Repair Graph。修正一次仍未通过则拒绝返回未经验证的答案。

## 13. Checkpoint、幂等与恢复

Checkpoint、幂等和任务接管解决不同问题：

| 机制 | 责任 |
|---|---|
| SQLite Checkpoint | 保存 Graph 执行位置和 State |
| 幂等键 | 节点重复执行时不产生重复副作用 |
| Redis Lease / Heartbeat | Worker 崩溃后任务可被接管 |
| 文档版本状态 | 未完成版本不可检索 |

### 13.1 Checkpoint

```text
QueryGraph:     query_checkpoints.sqlite
IngestionGraph: ingestion_checkpoints.sqlite
```

Query 使用会话 `thread_id`；摄取使用 `thread_id=ingestion:{job_id}`。

外部传入的会话 ID 不能直接作为 Checkpoint Key。API 必须生成 `checkpoint_thread_id={user_id}:{thread_id}`，防止不同用户使用相同 `thread_id` 读取彼此状态。

### 13.2 幂等键

```text
ingestion_key = hash(user_id + document_id + content_hash + pipeline_version)
parent_id     = hash(document_version_id + ast_locator)
child_id      = hash(parent_id + child_ordinal + content_hash)
event_id      = hash(run_id + node_name + logical_event_key)
```

MySQL 使用唯一键或 Upsert，ES 使用确定性 ID。Checkpoint 落盘前发生的副作用可以安全重放。

### 13.3 文档版本发布

- 新版本在完整解析和索引前为 `building`。
- Retrieval 只能搜索 `is_active=true` 的 Child。
- 全部 Parent 和 Child 写入成功后再发布新版本：先写入不可见的新版本，再批量激活新 Child、失活旧 Child，最后更新 MySQL `active_version_id`。
- MySQL 与 ES 不实现分布式事务；失败由版本状态、幂等重试和补偿完成。
- 若发布过程中断导致新旧版本短暂同时可见，Parent 聚合按 `document_id` 只保留最高 `version_no`，重试继续完成旧版本失活。

## 14. 重试、降级与错误

### 14.1 默认安全上限

```text
max_research_rounds = 4
max_answer_revisions = 1
node_retry_attempts = 2
graph_recursion_limit = 50
query_node_timeout_seconds = 90
retrieval_timeout_seconds = 30
ingestion_stage_timeout_seconds = 600
```

### 14.2 错误分类

| 类型 | 处理 |
|---|---|
| 超时、429、5xx、临时连接错误 | 指数退避加随机抖动，最多重试 2 次 |
| 参数、Schema、用户范围、文件格式错误 | 不重试，直接返回明确错误 |
| 单个可选组件失败 | 按降级矩阵继续执行并记录事件 |

### 14.3 降级矩阵

- Dense 失败：只用 BM25。
- BM25 失败：只用 Dense。
- Dense 与 BM25 都失败：不生成答案。
- Cross-Encoder 失败：使用 RRF 排序。
- mem0ai 失败：无长期记忆继续查询。
- MySQL Parent Fetch 失败：不生成答案。
- 审核失败或最终不可用：Fail Closed，返回可重试错误。

统一错误响应：

```text
error_code
message
retryable
trace_id
degraded_components
```

## 15. API 与本地进程

V1 运行：

```text
FastAPI API Process
Ingestion Worker Process
```

最小 API：

```text
POST   /v1/query
POST   /v1/documents
GET    /v1/ingestion-jobs/{job_id}
DELETE /v1/documents/{document_id}
GET    /v1/memories
DELETE /v1/memories/{memory_id}
POST   /v1/feedback
```

约束：

- `user_id` 缺省为 `default_user`。
- `thread_id` 可由服务生成。
- 用户范围由 API 注入，Agent 不能覆盖。
- `/v1/query` 只返回审核后的最终答案，不流式输出未审核草稿。
- `/v1/feedback` 接收 `run_id`、`rating: up | down` 和可选 `comment`；服务端校验 Run 属于当前 `user_id` 后写入 `USER_FEEDBACK` Event。
- 文档删除使 MySQL Parent 与 ES Child 失效。长期记忆不保存文档证据，因此通过独立 Memory API 管理。

## 16. 推荐代码结构

```text
src/agentic_rag/
├── api/
├── query/
│   ├── graph.py
│   ├── state.py
│   ├── research_loop.py
│   ├── tools.py
│   └── audit.py
├── retrieval/
│   ├── graph.py
│   ├── models.py
│   ├── ports.py
│   └── adapters/
├── ingestion/
│   ├── graph.py
│   ├── parser.py
│   ├── assembler.py
│   ├── chunker.py
│   ├── indexer.py
│   └── worker.py
├── memory/
│   └── service.py
├── persistence/
│   ├── mysql.py
│   ├── checkpoint.py
│   └── redis_queue.py
├── models/
├── observability/
└── config.py

evals/
tests/
scripts/
```

按业务能力分包，不建立复杂 DDD、插件框架或多仓库结构。

## 17. 三层质量评测体系

生产质量闭环分为三层，职责不能混用：

| 层次 | 执行时机 | 责任 |
|---|---|---|
| 请求内强制审核 | 每次线上请求 | 阻止无证据、幻觉或无效引用答案返回 |
| 在线质量监控与反馈 | 每次线上请求 | 发现真实流量异常、退化和用户不满意 |
| 离线基准评测 | 发布前或人工触发 | 使用固定数据集可复现地比较系统版本 |

三层形成闭环：请求内审核保护单次回答，线上失败和反馈补充离线数据集，离线回归验证修复后再发布。

### 17.1 请求内审核

Evidence Grader、Faithfulness Audit 和 Citation Validator 是 QueryGraph 的固定质量门禁，定义见第 12 节。它们属于运行时 Guardrail，不用于比较不同版本的整体质量。

### 17.2 在线质量监控与反馈

每次请求基于 `agent_runs` 和 `agent_events` 记录：

```text
请求总延迟与节点延迟
路由类型
检索轮数
拒答与澄清率
审核失败与答案修正率
Loop 上限触发率
Dense / BM25 / Reranker 降级率
引用有效率
用户数据泄漏数
用户正向与负向反馈
```

V1 不为这些指标建设独立实时平台；通过 MySQL 审计数据进行查询和汇总。用户反馈通过 `/v1/feedback` 写入现有 `agent_events`，不新增反馈表。

V1 不对每个线上请求同步或异步运行完整 Ragas。线上问题通常没有参考答案，而且额外 LLM Judge 会增加延迟、非确定性和敏感数据处理范围。真实流量增加后，才考虑脱敏后的异步抽样语义评测。

### 17.3 离线基准 Eval Runner

离线评测不放入线上 QueryGraph，而是通过独立 Runner 调用真实 QueryGraph：

```mermaid
flowchart LR
    DS["JSONL Evaluation Dataset"] --> RUN["调用真实 QueryGraph"]
    RUN --> COL["收集 Answer / Evidence / Events"]
    COL --> DET["确定性检索指标"]
    COL --> RAGAS["Ragas 语义指标"]
    DET --> REP["JSONL + Summary"]
    RAGAS --> REP
```

### 17.4 离线数据集

```json
{
  "case_id": "case_001",
  "user_id": "eval_user",
  "question": "问题",
  "reference_answer": "参考答案",
  "reference_parent_ids": ["parent_1"],
  "expected_route": "fast_rag",
  "tags": ["single_hop", "pdf"]
}
```

V1 使用人工整理的小规模高质量数据集，不使用自动测试集生成。

### 17.5 离线指标

确定性检索指标：

```text
Parent Recall@6
NDCG@10
MRR
user_leak_count
```

Ragas 语义指标：

```text
Faithfulness
Answer Relevancy
Context Precision
```

离线 AgentLoop 运行指标：

```text
average_retrieval_rounds
loop_limit_hit_rate
```

不使用严格 Tool Call Sequence Accuracy，因为同一研究任务可能存在多条正确 Tool 路径。

### 17.6 线上与离线硬门禁

```text
user_leak_count = 0
有效引用率 = 100%
未通过审核的答案返回数 = 0
```

硬门禁在线请求和离线回归都必须满足。首次运行建立其他指标的基线；后续发布比较相对回归，不在设计阶段拍脑袋设置全部阈值。

## 18. 测试策略

### 18.1 单元测试

- Fragment AST 与 Global Assembler；
- 跨页合并和去重；
- ParentBuilder 与 Hybrid Child Chunking；
- Filter 注入与用户范围不可覆盖；
- Todo Reducer 和 Evidence Reducer；
- Router 与审核结构化 Schema；
- 幂等 ID 与事件 ID；
- Context Compact 保留项。

### 18.2 集成测试

- ES Dense、BM25、RRF、Cross-Encoder 和 Parent Fetch；
- MySQL Parent、审计事件和 Job Lease；
- Redis Streams 重复投递与接管；
- SQLite Checkpoint 恢复；
- mem0ai 用户隔离、读取和删除。

### 18.3 端到端测试

- PDF、扫描 PDF、Excel、文本上传并完成索引；
- Fast RAG；
- 多跳 AgentLoop；
- 并行 Subagent；
- 证据不足、澄清和拒答；
- 单路检索与 Reranker 降级；
- 进程中断后的摄取恢复；
- 不同 `user_id` 之间零数据泄漏。

## 19. 可观测性与审计

- 所有日志包含 `trace_id`、`run_id`、`thread_id`、`user_id` 和 `node_name`。
- 原始用户消息、最终回答和 Tool Event 写入 MySQL。
- 在线质量指标从 `agent_runs` 与 `agent_events` 汇总，不在请求路径中额外调用 LLM Judge。
- 用户反馈与对应 `run_id` 关联，负向反馈用于补充离线失败用例。
- 不保存隐藏 Chain-of-Thought 或模型内部推理 Token。
- 敏感值和密钥在日志及事件 Payload 中脱敏。
- V1 使用结构化应用日志和 MySQL 审计，不引入独立可观测性平台。

## 20. V1 验收标准

满足以下条件即认为 V1 设计目标完成：

1. 本地环境无需 Docker 即可启动 API、Worker、ES、MySQL 和 Redis。
2. 四类文档可以统一进入 Canonical AST、Parent-Child Chunking 和索引流程。
3. Fast RAG 与真正的 ResearchAgentLoop 都能运行。
4. Agent 可以动态修改 Todo，并在独立任务上并行运行 Subagent。
5. Dense、BM25、RRF、Cross-Encoder 和 Parent Fetch 边界清楚。
6. `user_id` 同时约束文档、Child、Parent、Memory、Checkpoint 调用上下文和审计查询。
7. 所有正常答案经过 Evidence、Faithfulness 和 Citation 三个审核阶段。
8. Checkpoint 恢复不会因节点重放产生重复 Parent、Child 或 Event。
9. Redis 重复投递和 Worker 崩溃不会丢失摄取任务。
10. mem0ai 故障可降级，审核与核心检索故障 Fail Closed。
11. 在线监控记录质量与运行指标，用户反馈可追溯到 Run。
12. 离线评测输出检索、语义和 AgentLoop 指标，用户泄漏数始终为零。

## 21. 后续迭代边界

在 V1 数据证明有必要后，才考虑：

- Milvus `VectorIndex` Adapter；
- Redis 查询缓存或 Embedding 缓存；
- 完整认证、RBAC 与知识库级权限；
- 人工确认 UI；
- 脱敏后的线上异步 Ragas 抽样、A/B 实验与独立评测结果平台；
- 更复杂的 Memory 遗忘和衰减策略；
- 流式进度事件；
- Kubernetes、微服务和自动扩缩容。

这些能力不改变 V1 已定义的 Graph、Tool、Retrieval Port 和数据契约。
