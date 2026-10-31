# Supervisor 真实链路验收记录

日期：2026-10-31。范围为已授权的真实 MySQL / 模型端到端验收与修复，使用少量合成文档。业务入口仍默认关闭；实现、运行说明和本验收记录一并版本化。

## 环境与隔离

使用本机已有 MySQL 9.4.0（REPEATABLE READ）、Elasticsearch，以及本次单独启动的临时 Redis 和独立角色进程。所有测试数据库均以 `agentic_rag_supervisor_test_` 开头；物理索引为 `agenticrag-children-sup-live-340939cdf6d0`，不切换业务 active alias。

实际模型为 Qwen `qwen3.7-text-embedding-flash`（1024 维）、`BAAI/bge-reranker-v2-m3`、DeepSeek `deepseek-v4-pro` / `deepseek-flash`。轻模型请求 ID 为 `deepseek-v4-flash`，后者是 provider 返回的实际模型标识。检索观察记录中的 dense、BM25、rerank 均有真实候选，无降级通道；没有注入假向量、模型回答或评分。

用户更新凭据后，实测定位到两项配置差异：OpenAI 兼容客户端需要 `/compatible-mode/v1`，而非 DashScope 原生 `/api/v1`；检索读取 `AGENTIC_RAG_EMBEDDING_MODEL`，仅修改 Mem0 模型名不会改变检索模型。仅修正这两项非秘密配置，未输出或复制密钥。

## 通过的真实场景

| 场景 | 实际观察 |
| --- | --- |
| 快速检索 | Run `01a18f92-bb94-7166-8cde-8f623b79347a`：返回凌星 2025 年收入 120 亿元及有效引用；无 AgentTask、无 Supervisor 规划阶段；full / passed。 |
| 并行检索 | Run `01a18f97-cb8f-7b07-86a0-5795473fed24`：两个不同 PID 的 Retrieval Worker 同时处理两家公司，执行区间重叠约 6.49 秒。 |
| 独立汇总与定向补差 | 首次独立评分判远海 2024 年数值及可比口径不足。只续派 `ret_yh`；`ret_lx` 的结果引用、上下文、轮次均保留。最终查询数分别为 1、2，轮次分别为 1、2；第二次独立评分全部 supported / sufficient。 |
| 独立 Analysis | 同一 Run 的已接纳分析产生两个经操作数回源验证的计算：`(120−100)/100×100=20%`、`(260−200)/200×100=30%`。分析评估 accepted，最终带引用的表格答案 full / passed。 |
| Supervisor 恢复 | 任务派发后 SIGKILL Coordinator 与 Supervisor。两个子 Worker 在父进程不在时各自持久化结果；恢复后消费这些结果，初始检索均为一次 attempt，没有重跑。 |
| SQL checkpoint 恢复 | 停止测试进程，将 mysqldump 导入另一可丢弃数据库，逐字段比较 Run 状态；清空本次 Redis 后启动新进程。原 deadline、预算、上下文和结果引用保持连续。最终并行用例全部走完。 |
| 原生流与重连 | 并行 Run 收到 307 个 delta，含 citation/终态共 312 个事件。HTTP 客户端首 delta 早于 provider 完成 1.795 秒；在 `1:1` 断线后用 Last-Event-ID 重连，ID、类型、事件内容和最终正文与 SQL 记录一致。 |
| 流中断 | Run `01a18f86-cb6f-7e68-9b3a-191aefeadfb1`：首段落库后杀死 Answer Worker，租约恢复将流标记 interrupted / unavailable，保留原前缀与 epoch，仅一个 answer attempt，没有拼接重新生成的文本。该用例的上游是 partial，不用于证明报告质量。 |

另一个真实用户算术 Run `01a18f85-9447-7f2a-80d7-c9ade5356a00` 独立通过 Analysis、父进程消失、SQL 备份恢复和流式重连；答案 20%，full / passed，共 95 个 delta，首段早于 provider 完成 0.639 秒。

补差用例先冻结两个初始检索结果，再发布补充文档，然后恢复协调进程。首次评分仍仅使用被冻结的原候选；后续缺口检索才能读到新资料。这是受控的数据到达时序，没有改写模型决策。新协调链使用 MySQL 状态恢复，不是把旧 LangGraph SQLite checkpoint 当作分布式任务存储。

## 发现并修复的问题

- DeepSeek 默认 thinking 将输出额度耗在推理中、导致 JSON 截断：新有界阶段显式配置 thinking，Supervisor 合同分配适当输出上限，截断不会重复同额度修复；旧调用默认不变。
- 派发字段、组 criterion 绑定、无效补差目标及用户算术工具权限存在边界不一致：修正合同和来源校验，并让错误决策形成持久化反馈。
- 首轮查询耗尽补检额度：按剩余轮次分配 QueryPlan 最大查询数量，模型 schema 与代码总预算共同限制。
- 已确认的结构修复用量被按整额收费，以及剩一次调用却要求预留两次：准确结算可确认用量；最后一次额度关闭额外重试/修复；未知消耗仍保守收费。
- 分析草稿未计算执行器新增 finding 的数量：动态 schema 校验说明与计算的合计配额，进入已有的有限结构修复。
- Embedding 模型名在 SQL staging / Manifest / ES 元数据中写死，且新 Worker 缺少向量空间校验：贯通配置并新增只读启动校验，拒绝同 generation 混用同维度不同模型。配置不匹配不再误 quarantine 合法旧版本。
- 质量模型可能误放行未验证的计算：新增必需 Analysis 缺失或被最终上下文省略的确定性规则，`passed` 降为 `unavailable`，已有 `failed` 不提升。可选辅助分析被省略不会误伤已完整的必需分析。

负例 `report3`（Run `01a18f99-7f74-7ed8-a8ba-811f6ae76ba8`）中，模型给操作数引文加上原句没有的公司名前缀，严格引用校验拒绝了分析；最终模型却自行算出 20%，评分模型还返回 passed。修复后，将该真实 verdict 和冻结文本原样在真实 MySQL 的新测试 Run 重放，提交状态为 unavailable；同样重放并行正例仍为 passed。此重放没有再次调用模型，也没有修改原始历史 Run，不能表述为该负例已经生成了合格报告。

## 自动化验证与证据

真实 MySQL schema / repository 测试 **15 passed**；REPEATABLE READ 竞争测试 **2 passed**；旧 SQLite checkpoint 恢复测试 **7 passed**。这些数字有不同测试范围，不与离线总数重复相加。

最终全量离线回归：**1589 passed，35 skipped，147 deselected**，30.05 秒；两个既有 pytest 模块重写 warning。变动的 **75 个 Python 文件 Ruff**、**44 个源文件 Mypy** 和 `git diff --check` 均通过。独立复审还以 fake provider 验证了调用预算上限与重试/修复的五种组合。

测试工具为 `scripts/supervisor_acceptance_worker.py`，可通过显式资源 manifest 启动 seed、补充数据、API 和各 Worker 角色。它校验本地 MySQL 与可丢弃名称，注入隔离物理索引而不改业务 alias。部署入口仍为 `scripts/run_agent_worker.py`。

本机证据目录：`.superpowers/sdd/2026-10-31-supervisor-live/`。其中 `verified-results.json` 保存流、并行、补差和恢复断言；`quality-guard-replay.json` 保存原始真实 verdict 的重放结果；`*-sql-final.json`、`*-faults.jsonl`、`*-sse.jsonl` 和角色日志保留时序与数据库证据。该目录被 Git 忽略，不把原始数据库备份或模型日志提交到仓库。

这些用例证明指定运行时行为，不代表业务语料质量基准、所有模型输出稳定通过、跨主机容灾或高并发压测。流式文本在质量状态终结前仍未验证。业务旧索引仍属于旧 embedding 模型，不能仅改模型配置便用新向量直接查询；业务切换需另行安排新 generation 与重建。

## 清理与业务核对

验收完成后已停止本次 8 个 API/Worker 进程和临时 Redis，删除 5 个可丢弃数据库与测试物理索引，保留本机日志/备份/断言文件。业务 MySQL 四项计数与开始时一致：documents 1007、document_versions 1009、parent_chunks 2258、agent_runs 177；业务 active alias 仍仅指向 `agenticrag-children-index-v4`。未执行业务数据库迁移或业务服务重启。

## 同日复验：单次输出上限 100,000 token

按用户的新要求，Analysis 和最终生成均传入 `max_tokens=100000`，测试审计同时记录应用层上限与实际 provider 请求参数。子任务累计输出默认提高到 262,144，为一次完整调用、一次修复和独立评估留出空间；Run 总额度仍为 1,048,576。正文存储保护上限提高为 1,000,000 字符，模型额度是上限，不要求用满。

本轮另建物理索引 `agenticrag-children-sup-live-cc851388f6d9`、独立 Redis 与可丢弃 MySQL 库，不复用上一轮已经清理的环境。Qwen、DeepSeek、BGE 均为真实调用，没有替换模型输出、向量或评分。主要证据位于 `.superpowers/sdd/2026-10-31-supervisor-100k-live/`。

| 场景 | 本轮实际结果 |
| --- | --- |
| 快速查询 | Run `01a18fa6-87b1-7f47-8be3-c5c19b2d5496`：full / passed，30 个原生 delta，首段早于 provider 完成 0.226 秒，最终请求上限 100,000。 |
| 并行、补差及父进程恢复 | Run `01a18fa7-cc99-7505-a59b-60b33a069f74`：两个 PID 的检索区间重叠 3.551 秒；两个结果均在父进程被杀后完成。只续派远海任务到第二轮，凌星结果原样保留，初始检索各一个 attempt。 |
| SQL checkpoint | 同一并行 Run 的完整 mysqldump 恢复到另一个测试库；Run 状态逐字段一致，清空临时 Redis 后用全新进程继续。原期限、预算、结果引用和私有上下文连续。 |
| 分析、完整答案与 SSE 重连 | 并行 Run 接纳的计算为 20% / 30%，最终 full / passed；Analysis 和最终请求均为 100,000。390 个正文 delta、395 个总事件，首段领先完成 2.212 秒。`Last-Event-ID` 重连事件与 SQL 记录及最终正文完全一致。 |
| 修复后的用户算术恢复 | Run `01a18fae-5100-772c-a7f0-2ce4822b1190`：父进程被杀后 Analysis 独立落库，再次备份恢复 MySQL 并清空 Redis；分析没有重跑，原输入额度也保留。最终 20%，full / passed，104 个 delta，首段领先完成 2.165 秒。 |
| 流中断 | Run `01a18faf-070f-7115-a7c6-0cde8dc741f9`：首段落库后杀死 Answer Worker，恢复后保持“根据”前缀与 epoch，只有一个 answer attempt，interrupted / unavailable。上游为 partial，不能用此例证明报告质量。 |
| 数值依据修复后的完整报告 | Run `01a18fbc-db60-7df1-beec-4056b9bd4bda`：两个 Retrieval 进程实际重叠约 6.148 秒，一个 Analysis 汇总两路已接纳证据并计算 20% / 30%，最终 full / passed。222 个 delta、227 个事件；首段领先完成 1.435 秒，重连与 SQL 记录一致，分析和最终请求上限均为 100,000，11 项断言通过。 |

### 本轮发现与修复

- 最终输入原先把已接纳的分析报告也算入每包证据的 12,000 字符额度，合法长报告会被拒绝或整体遗漏。新 Run 独立持久化 `answer_input_char_limit=1000000`，对汇总证据和报告做完整单元选择；旧 Run 缺字段时沿用旧限制，不在恢复中回填。新默认、旧 Run、显式小额度三个真实 MySQL 边界测试通过；这些边界测试使用确定性模型响应，与真实模型用例分开计数。
- 用户算术负例 `01a18faa-e9d1-71f7-bf1e-7f5ff21dac6b` 将“亿元人民币”写成“人民币亿元”，严格来源校验正确拒绝，但错误原先发生在模型修复流程之后。现在原文、哈希和元数据校验进入既有结构修复流程，执行器仍重复验证。
- 负例 `01a18fac-8aeb-72e7-9cc2-c031727ade4a` 暴露 `max_items=1` 被模型理解成“一份报告”。明确其实际计数为结论条目加计算条目，并同时反馈输出配额、引用和元数据错误，仍只允许一次已预留的修复。
- 中断用例的上游分析错误地用 `query` 引用文档计算结果。已将允许的 finding 来源检查前移到同一修复流程；计算及来源的后置强校验继续保留，无效来源不会被放行。
- 完整报告复验 `01a18fb1-b978-7da7-bb23-cd45ccde9e11` 的两个计算任务均被接纳，但 Supervisor 又派发了依赖这两个 Analysis 的比较任务，超出执行器支持的输入类型；最终 partial / unavailable，没有被质量闸门误放行。新增派发校验：Analysis 只能依赖 Retrieval；比较报告由一个 Analysis 汇总多路检索结果。覆盖同批及已有任务依赖两个回归，错误在派发前反馈。
- 后续负例 `01a18fb4-8bc6-7df8-b771-95993b4f3ab6` 的已验证计算只有 20% / 30%，分析和最终文本却新增“高出10个百分点”，质量模型也误判 passed。新增十进制数值依据规则，在分析的一次结构修复和最终质量提交中检查；没有 `calculate` 权限时，结构验证也不执行计算器。以该真实 verdict 和原始冻结文本在新 MySQL Run 重放，提交结果为 failed；并行正例与新的完整报告重放仍为 passed，所有原始 Run 和正文保持不变。此重放没有额外模型调用，不能将原负例记为生成成功。
- 数值规则的复审修复了十进制四舍五入、证据标题中的年份、Markdown 标题编号和 `.1` 小数的误报/漏检，并覆盖逗号列表不应变成千位数的情况。规则仅是数值字面量检查，实体、单位和语义关联仍由严格操作数校验及独立质量评估负责，不宣称完成所有数值主张的形式化证明。
- 首次故障注入因沙箱禁止读取 `ps` 而未执行停机，该 Run 已取消，不计作恢复成功。随后获准在明确测试资源范围运行。脚本现在对进程检查失败立即报错；成功用例还交叉核对实际 kill 记录与子任务完成时间，防止停机测试假成功。

最新离线回归 **1608 passed，35 skipped，147 deselected**，31.67 秒，两个既有 warning；Ruff 检查 77 个 Python 文件、Mypy 检查 46 个文件通过。实际 MySQL 迁移成功，竞争测试 2 项、旧 SQLite checkpoint 测试 7 项通过；这两类恢复分别对应不同运行时。新增长报告 MySQL 边界 3 项通过。独立复审验证了预算、旧 Run 兼容、依赖类型、有限修复和数值检查边界。数值检查最后的格式兼容修复完成后，再次将三个保存的真实 verdict 在 MySQL 重放，正反例结果保持正确。

上述证据验证真实请求采用 100,000 上限及指定恢复行为，没有要求模型实际生成十万 token，也不代表十万 token 满长度、长上下文质量或高并发压力测试。所有失败用例保留原始记录，没有改写历史状态为通过。

本轮清理已完成：8 个 API/Worker 进程均确认退出，临时 Redis 已停止，3 个可丢弃数据库和含 3 个测试块的物理索引已删除。本轮前后业务表计数与 active alias 一致，未迁移或重启业务环境。证据目录保留 `verified-results.json`、`quality-guard-replay.json`、进程退出核对、清理报告和测试日志；提交仅包含源码、测试、运行工具和文档。
