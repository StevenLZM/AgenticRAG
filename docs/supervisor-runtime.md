# Supervisor / Retrieval / Analysis 运行说明

更新：2026-10-31。代码已完成离线回归及隔离环境中的真实 MySQL、Qwen、DeepSeek、Elasticsearch、reranker 验证；新入口默认关闭。未迁移业务数据库或重启业务服务。详细证据与负例见[真实链路验收记录](supervisor-live-validation.md)。

设计依据：[设计文档](superpowers/specs/2026-10-30-supervisor-retrieval-analysis-design.md)。执行记录：[开发计划](superpowers/plans/2026-10-30-supervisor-runtime.md)。

## 已实现的执行边界

```mermaid
flowchart TD
    Q[Query Run] --> R[Route]
    R -->|单目标事实查询| F[快速检索至 rerank]
    R -->|研究 / 分析| S[Supervisor]
    S -->|TaskSpec + Group| T[一个或多个 Retrieval Agent]
    T --> P[Evidence Prepare：有界汇总 / 回源 / 去重]
    F --> P
    P --> G[独立 Evidence Grader]
    G --> C[Coordinator 接纳与反馈]
    C -->|缺口与剩余预算| S
    C -->|已接纳证据| A[Analysis Report Agent + Calculator]
    A --> AG[Analysis Assessment]
    AG --> C
    C -->|冻结完整或部分输入| O[最终模型流式生成]
    O --> V[独立文本忠实性检查]
```

- 简单事实查询不创建 AgentTask、不调用 Supervisor 规划；不足时保留同一 Run 的候选、评分、期限和预算。新的检索组可通过 `retained_fast_criteria` 复用快路径结果，仅派发缺口任务。
- `TaskSpec` 强制包含目标与验收项、输出合同、工具和来源指引、任务边界。来源硬过滤与 `required_document_ids` 验收由代码执行；偏好不能扩大授权。
- 子 Agent 的查询、摘要、结果和上下文引用按任务隔离。续派仅选择指定成员，沿用已耗预算与已完成成员；重复合同的新 ID、连续无新增证据或覆盖进展的补检会被拒绝。
- 检索计划按剩余查询数与剩余轮次限制本轮查询数量，避免首轮用尽补差额度；该上限同时进入模型 schema 和持久化预算校验。分析草稿的输出配额包含由执行器生成的计算 finding。
- 数学计算仅在 Analysis 中执行，核对操作数原文、主体、期间、单位和来源；用户直接给出的数值绑定原始问题及哈希。分析结果还需独立接纳。新增资料会使旧输入分析失效，迟到的旧评分不能重新接纳它。
- Analysis 的任务依赖只接受 Retrieval 任务，比较报告由一个 Analysis 汇总多路已接纳证据。当前不支持 Analysis 串接；派发时拒绝该合同并反馈 Supervisor，避免执行器收到不支持的结果类型后才失败。
- 检索组固定成员、必需成员、收集截止时间和汇总版本。完整或超时汇总会产生不可变输入快照；迟到结果不修改已冻结快照。同组依赖造成的 join 死锁在派发前拒绝。
- 冻结回答前限制最终证据和分析上下文。部分回答选择完整支持单元并注明遗漏，不截断原文后沿用完整覆盖结论。

## 模型额度

2026-10-31 按项目目标放宽默认模型额度，为多轮补差、结构修复和分析报告留出空间。额度是执行上限，不要求模型用满；持久化计量、调用前预留、完成后结算和超限停止仍由代码强制执行。

| 范围 | 模型调用次数 | 累计输出 token |
| --- | --- | --- |
| 单个子任务默认 | 32 | 262,144 |
| 单个子任务合同可设上限 | 128 | 1,048,576 |
| 单个 Run 全局上限 | 256 | 1,048,576 |

Analysis 与最终答案单次输出上限为 100,000 token，Supervisor 为 8,192，其他结构化阶段为 4,096。子任务累计额度为一次完整分析、一次结构修复及独立评估留出空间。预留按 Run 与任务两层剩余额度裁剪；重试和结构修复分别占用调用次数与 token。最终答案只预留一次调用，正文存储保护上限为 1,000,000 字符，单 delta 上限仍为 16,384 字符。

新 Run 另存 `answer_input_char_limit=1,000,000`，限制最终汇总的证据与已接纳分析的序列化字符数；它不含问题和状态元数据，也不等同于模型 token 上限。每个检索证据包仍按原预算打包。完整支持单元不可截断，汇总超限仍要求明确选择部分回答并记录遗漏。旧 Run 缺少该字段时保留原 snapshot 限制，不在恢复时回填。

新 Run 和未显式指定模型额度的新任务使用上述默认值。显式小预算仍生效；已持久化的 Run、任务在恢复时保留原额度与消耗。查询次数、补差轮次、无进展停止、并发限制和绝对期限继续独立生效。额度调整的真实复验记录见[验收记录](supervisor-live-validation.md)，各轮结果分别保留实际配置与证据。

## 持久化与进程

新增迁移 `0009_supervisor_runtime` 创建九张表：`supervisor_runs`、`agent_work_items`、`agent_task_results`、`supervisor_decisions`、`evidence_input_snapshots`、`supervisor_inbox`、`supervisor_outbox`、`answer_stream_chunks`、`agent_resource_leases`。

MySQL 保存执行事实；Redis Streams 只负责可重放通知。派发合同与 Outbox、结果与后继阶段、Inbox 与唤醒意图在短事务内提交。Coordinator 可重建，Worker 由租约和 fencing token 取得执行权。重投不会重置查询/模型预算；结果只在有效任务、输入、轮次和组版本上生效。

新协调链不打开 QueryGraph 的 SQLite checkpointer。`query_worker_count=1` 仍约束旧 Query Worker；新角色通过独立进程扩展。

| 角色 | 工作 |
| --- | --- |
| `coordinator` | Run 准入、Outbox、监听/Inbox、依赖接纳、组收集、恢复和 API 投影 |
| `supervisor` | 路由与 Supervisor 决策 |
| `retrieval` | 快速检索、Retrieval Agent；加载 embedding / reranker |
| `analysis` | 分析、验证操作数和计算 |
| `evidence` | Evidence Prepare、Evidence Grader、Analysis Assessment |
| `answer` | 正文 provider 流与后续忠实性检查 |
| `all` | 以上执行阶段的合并 Worker；仍需另启 `coordinator` |

## 部署步骤

先在可丢弃 MySQL 环境通过迁移和竞争测试，再安排业务环境迁移和启用。API、Coordinator 和所有新 Worker 必须使用一致的代码、数据库、索引代际、模型端点与并发配置。

`AGENTIC_RAG_EMBEDDING_MODEL` 控制文档与查询向量；`AGENTIC_RAG_MEM0_EMBEDDING_MODEL` 只控制 Mem0。OpenAI 兼容 embedding 客户端需要服务商的兼容接口地址。模型名写入 SQL 版本、Manifest 和 ES mapping；新 Retrieval Worker 启动时只读校验模型、物理 generation 与 active alias 的对应关系，不修复或切换业务 alias。即便维度同为 1024，不同 embedding 模型也不能共用已有向量空间，切换模型需要独立 generation 与重建安排。配置不匹配的旧待发布版本保留给匹配的 Worker，不按损坏数据 quarantine。

在项目 Python 环境和已配置的目标数据库上执行迁移：

```sh
python -m alembic upgrade 0009_supervisor_runtime
```

仅在 API 创建新 Run 时设置 `AGENTIC_RAG_SUPERVISOR_ENABLED=true`。其快照保存 `orchestration_version=supervisor-v1` 和 `answer_protocol_version=answer-v2`；已创建 Run 的版本不随开关改变。Worker 启动会检查新表是否存在。

开发环境可在两个终端分别运行：

```sh
python scripts/run_agent_worker.py --role coordinator
```

```sh
python scripts/run_agent_worker.py --role all
```

需要并行检索时独立启动多个 `--role retrieval` 进程，并分别启动 `supervisor`、`analysis`、`evidence`、`answer` Worker。`all` 进程自身一次执行一个阶段；它不代替多个并行执行进程。

新运行时使用 SQL 准入锁落实 `AGENTIC_RAG_MAX_CONCURRENT_QUERY_RUNS`；未准入的公共 Run 保持 queued，排队仍计入创建时确定的绝对期限。SQL 租约落实 `MAX_CONCURRENT_LLM_CALLS`、`MAX_CONCURRENT_RERANKS`、`SUPERVISOR_TENANT_MODEL_LIMIT`（均带 `AGENTIC_RAG_` 前缀）。`MAX_PARALLEL_SUBAGENTS_PER_RUN` 同时计算检索和分析子任务。租约续期必须仍持有全部所需容量槽位。

这些限制在新运行时的进程间共享。旧 QueryGraph 仍使用原有进程内限制；两套运行时并存时不能把它们的配额当作一个部署总上限。

## 流式协议与恢复

API 沿用 `/v1/query-runs/{run_id}` 和 `/v1/query-runs/{run_id}/events`，按 Run 固定的版本选择协议。`answer-v2` 使用 `epoch:seq` 作为 SSE `Last-Event-ID`，按 SQL 已提交的 chunk 顺序重放。

事件包括 `answer.delta`、`answer.citation`、`answer.text_completed`、`answer.completed`、`answer.interrupted` 和 `answer.cancelled`。只有完整且属于冻结 manifest 的引用标记产生 citation 事件。前端即时显示 delta，并按 epoch/seq 去重。

`faithfulness` 独立取值为 `pending/passed/failed/unavailable`；正文生成完成时先进入 finalizing，审核完成或明确不可用后结束。新答案不伪造旧协议的 `audited=true`。完整流的最终状态为 completed 也可能对应 faithfulness=failed，界面必须保留该质量状态。

必需的 Analysis 未被接纳，或因最终上下文预算被遗漏时，模型评分器的 `passed` 也不能使答案通过：确定性规则将其降为 `unavailable`；已有 `failed` 保持失败。流式正文仍是待验证文本，不能把收到 delta 当作计算已通过验证。

分析结论和最终答案还检查新增十进制数字：数值须来自授权证据正文/标题、原始问题或已验证计算。分析阶段的错误进入已有的一次结构修复；最终质量阶段发现无依据数字时，不能提交 `passed`。支持十进制四舍五入、规范千位分隔符和省略整数部分的小数，排除引用 ID 和列表/标题编号。此规则补充模型语义评估，不证明数字与实体、单位、公式的归属正确，也不解析中文数字。没有 `calculate` 权限的任务在结构验证时同样不能调用计算器。

来源在冻结、写入 delta、GET/历史投影及 SSE 重放发送时重新检查。撤权后停止新的披露；已发送的内容无法撤回。来源抽屉继续使用现有授权。

- Coordinator 重启：从 SQL Run、未消费结果、Inbox、Outbox 和组截止时间恢复，无需保留内存中的等待协程。
- Worker 崩溃：租约过期后重派，最多三次技术 attempt；旧 token 不能提交。已消费或状态不明的模型预留不免费返还。
- Provider 未报告 token 用量，或结构修复的多次用量无法逐次确认时，按已知调用次数的预留上限计费预算，并记录 `estimated_usage`；不把未知用量当作零。
- 剩余一次模型额度仍可执行一次调用，但禁止未预留的修复或重试；流式正文只预留一次实际调用。可确认的修复用量正常结算，失败和崩溃保守保留预留。
- Redis 不可用：执行/结果仍可由 SQL 扫描恢复，待恢复后重发通知。不要手工把通知 ACK 当成任务完成。
- 正文 Worker 丢失、取消或 provider 中断：保留已提交前缀，标记中断/取消，不把重新生成的文本拼到原 epoch。
- 只有质量阶段丢失或到期：保留完整正文，质量标记 unavailable。

数据库中 `work_items.stage/status/owner/lease_token/lease_until/attempt_count`、结果哈希、组快照、Run budget 与 Inbox 游标可用于排查。日志只记录阶段、工作 ID 和异常类型，不打印凭据或完整候选正文。当前实现逐 delta 持久化；高并发下的批量提交和性能调优尚需压测。

## 验证与回滚

本次验证使用已有 Python 环境，无新增安装。完整用例、模型标识、Run ID、负例及最终计数见[验收记录](supervisor-live-validation.md)。

- 真实 MySQL 9.4.0：实际迁移升级/降级/再升级，schema/repository 测试 15 项，REPEATABLE READ 竞争测试 2 项通过。旧 SQLite checkpoint 的 7 项恢复测试也通过；两者是不同持久化层。
- 真实 Qwen + ES + BGE reranker + DeepSeek：快速查询、两个独立进程并行检索、独立评分、仅缺口成员续派、Analysis 计算和最终流式答案均通过。
- 强制停止 Coordinator/Supervisor，保留子 Worker 完成结果；随后 mysqldump 恢复至另一可丢弃库，清空临时 Redis，重启全部测试进程。Run 状态、原期限、上下文与预算连续，已完成检索未重跑。
- HTTP 首 delta 在 provider 完成前到达；Last-Event-ID 重连的序列、事件正文与 SQL 记录一致。另行杀死 Answer Worker，已发送前缀保留且不会重生成拼接。
- 质量负例发现模型评分器误放行未被接纳的计算；补上必需 Analysis 缺失的确定性闸门，并以原始真实 verdict 在 MySQL 上重放验证：正例仍 passed，负例 unavailable。
- 全量离线回归、变动文件 Ruff、源代码 Mypy、差异检查通过。独立复审发现的边界问题均已修复并补回归。

可重复命令：

```sh
RAG_EVAL_PYTHON=/path/to/project/python /path/to/project/python -m pytest -q -m 'not integration and not e2e and not live_model'
python -m pytest -q tests/integration/api/test_chat_client.py
python -m pytest -q tests/integration/orchestration/test_process_recovery.py
AGENTIC_RAG_TEST_MYSQL_DSN='mysql+asyncmy://user:password@127.0.0.1/disposable_schema' \
  python -m pytest -q tests/integration/orchestration/test_store_mysql.py
```

MySQL 测试会创建新协调表，并删除自己生成的 Run 行；必须使用可丢弃 schema。本次真实链路使用少量合成文档与限定用例，并非业务语料答案质量基准、跨主机故障或高并发负载验收。

回滚入口时先将新 Run 创建开关关闭，继续运行新 Worker 直至已创建的 supervisor-v1 Run 排空或正常取消，保留版本化 API 和 SQL 数据供历史读取。旧 Query Worker 不接管新版本 Run。常规回滚无需 downgrade；删除新表会丢失任务与流重放记录，只能在明确备份与清理安排后单独处理。
