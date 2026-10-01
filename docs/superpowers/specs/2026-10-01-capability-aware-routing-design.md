# 信息来源路由与检索升级控制设计

> 日期：2026-10-01
>
> 状态：会话方案已获用户同意；书面设计待审阅，尚未实现或验收。
>
> 范围：Router、证据不足后的升级决策、真实模型路由回归。
>
> 主设计：[生产级 Agentic RAG 设计文档](./2026-08-04-production-agentic-rag-design.md)。

## 1. 目标与问题依据

本次修复三个问题：非知识库问题误入 RAG；明确缺少所需能力后仍进入 Research 重复检索；路由改动缺少正反例真实模型回归。成功标准不是仅让“天气”命中 chat，而是匹配问题所需信息来源和当前系统能力，同时保留文档问答、追问、研究与证据审计。

依据为 2026-09-28 已只读核对的 MySQL 事件与 SQLite checkpoint。thread 为 `01a0e72f-868c-7160-8723-8672db5f7372`，run/trace 为 `01a0e72f-868f-7009-b548-d4921ad3d0dd`，问题为“今天北京天气如何”。Router 保存的结果为 `fast_rag`，理由为 `Simple factual lookup requiring current weather retrieval.`。它识别了实时需求，却把外部天气检索映射成知识库检索。

第一次检索返回简历、项目计划、学校介绍等不相关内容。评估已经指出缺少实时气象数据，程序仍因 `insufficient` 转入 Research；改写为“今天北京天气 实时 温度 天气状况 风力 湿度”再次搜索同一知识库，最终 `cannot_answer`。创建至完成约 64 秒，Research 开始至判定无法回答约 51 秒。以上为历史诊断，不是本次重新运行结果，也不是优化后的性能承诺。

2026-10-01 代码复核确认：`RouteDecision` 仅包含 route、normalized_query、reason_code；Router 输入为 question 和 memory_context；Fast RAG 与主图 Evidence Grader 各自对 `insufficient` 无条件转 Research；Worker 的 messages 初始化只有当前用户消息，不能假定它已经是最近多轮对话。

## 2. 用户约束与范围

### 2.1 本次实现范围

- 一次 Router 语义分类调用，加确定性执行策略，不新增串行分类模型。
- 区分知识库、通用知识、会话信息和外部实时/外部查询需求。
- 将系统当前具备的执行能力作为服务端事实传入决策。
- 能力不足时产生受控说明，不编造实时事实或工具执行结果。
- 统一 Fast RAG 和 Research 后证据评估的升级策略。
- 增加代码、真实模型和端到端三层回归，并保留版本化结果。
- 在 checkpoint 和安全事件中区分初始路由、实际经过路径与终止原因。

### 2.2 明确不做与待解决事项

用户要求将 tool/MCP 权限问题留作后续处理。本次不实现工具级授权、RBAC、MCP credential/scope 管理、用户审批、工具权限动态发现或权限管理界面，不新增通用工具注册平台。

“能力是否存在”和“用户是否有权调用”是两个问题。本次只从当前组合根已经装配的执行器及配置生成最小能力描述：知识库检索已装配；实时天气和联网执行器未装配。因此后两者不可执行，而不是“用户无权执行”。不能把配置中一个未接线的 enable 标记当成可用能力。

不新增天气、网页搜索或其他外部执行器。未来接入这些能力时，必须同时补齐实际调用路径与权限治理；当前新增描述字段不能自行开启不存在的工具。现有 user_id、文档版本、检索范围、记忆隔离和安全检查保持不变。

不改 PDF 解析、chunker、ES 索引或 MySQL Parent，不重建任何数据；本次不需要 SQL 表结构迁移。不修改本次范围外的正式评测、预算、打分与成本实现，不自动重启运行服务。

## 3. 方案选择与职责

选择“单次语义分类 + 服务端执行策略 + 检索后止损”。仅补天气关键词无法处理“总结天气报告”等反例；仅改 Prompt 没有执行约束；新增第二个独立 Router 则增加延迟与职责重复。当前已有证据评估调用，扩展其输出即可形成第二道防线，无需增加一个专门的能力判断模型。

LLM 判断用户要什么，确定性策略决定当前能够执行什么。服务端能力描述本次只是组合根的只读派生数据，不是权限系统。模型、用户消息和记忆都不能修改能力描述或数据访问范围。

## 4. Router 输入和输出

### 4.1 输入

- 原始 question，保留用户措辞。
- 现有 memory_context，只作不可信上下文，不作为知识库证据。
- 最近对话：由服务端按同一 user_id 和业务 thread_id 读取当前 Run 创建前已经完成的对话；最多最近 6 条 user/assistant 消息，总计最多 8000 字符，保留时间顺序，不读取内部 tool 载荷或其他 thread。
- 当前请求已表达的文档约束及现有服务端 scope；不新增文档选择 UI/API，不允许模型扩大范围。
- 服务端实际能力描述及当前日期、时区；“今天”的解释依据请求时点，不依赖模型训练日期。

最近对话使用独立的 routing_context 字段，不把历史消息追加到用于本次记忆抽取的 messages，避免重复抽取旧事实。缺失历史时不猜指代，必要时澄清。正式独立用例评测采用隔离 thread，不混入用户历史和长期记忆；多轮评测显式构造同一隔离 thread 的历史。

### 4.2 语义分类

新增严格的 `RouteAssessment`，LLM 不直接决定最终执行分支：

```json
{
  "required_sources": ["external_realtime"],
  "retrieval_complexity": "none",
  "needs_clarification": false,
  "normalized_query": "今天北京天气如何",
  "reason_code": "realtime_information_required"
}
```

- required_sources 为去重的非空枚举集合：general、conversation、knowledge_base、external_realtime、external_lookup、unknown。unknown 只能单独使用。
- retrieval_complexity 为 none、single、multi。只有包含 knowledge_base 才能为 single/multi；包含 knowledge_base 时不能为 none。
- needs_clarification 表示缺少会影响执行的重要条件，不使用未经校准的模型置信度阈值。
- normalized_query 可结合可信来源标记下的对话上下文消解指代，不能改变任务、补造事实或修改租户范围。
- reason_code 使用有界枚举，不把自由文本理由当代码条件；矛盾组合按 schema-invalid 处理。

分类示例必须成对包含实时天气/上传天气报告、实时股价/财报历史数据、通用概念/文档中的定义，以及多轮指代和主题切换。不能因为用户以前讨论简历，就把当前天气问题强制归到简历；也不能因为没有“文档”二字就认定是通用问题。

现有 `RouteDecision` 保留为服务端派生的最终执行决定，继续提供 chat、fast_rag、research 和 normalized_query，避免把模型提议、实际执行和对外显示混为一谈。`RouteAssessment`、执行决定与能力版本分别进入 JSON-only checkpoint。

## 5. 确定性路由和受控响应

决策顺序为：处理结构无效或服务故障；处理来源不明及必要澄清；检查未满足的外部能力；最后决定普通 Chat 或知识库单次/多步检索。

| 条件 | 执行动作 |
|---|---|
| 仅 general/conversation，信息充分 | chat，正常非检索回复 |
| 仅外部实时/外部查询，且执行器未接入 | chat，固定能力限制说明，不再次调用回答模型 |
| knowledge_base，single，所需能力具备 | fast_rag |
| knowledge_base，multi，所需能力具备 | research |
| 知识库与不可执行外部需求混合 | clarify，说明可做文档部分，询问是否先分析该部分；保留完整原始请求，不自动宣称完成全部任务 |
| unknown 或缺少必要指代 | clarify，不默认 research |
| Router 超时、连接错误或结构化输出修复耗尽 | 受控 cannot_answer，记录 router_unavailable/router_schema_invalid；不谎称是用户需求不清，也不进入 Research |

“一次 Router 调用”指一次分类任务，保留 ModelGateway 已有有界 transport/格式修复行为，不宣称所有场景只有一次 HTTP 请求。禁止通过再次启动整个 QueryGraph 达成隐式重试。

明确能力不足的回复使用 `chat + response_mode=capability_unavailable`，status 为 cannot_answer；普通聊天为 conversation 模式；澄清为 clarify 模式及状态。技术故障响应使用独立的错误原因，不能伪装成能力缺失。

受控非知识性文本复用 Chat 的 public answer 约束：无证据引用、无 audited 声明。若已误检索，清空公开答案的 evidence_parent_ids 与引用，内部 trace/checkpoint 仍保留诊断证据。不放宽 `PublicAnswer` 对 RAG 知识性答案必须审核的要求。所选分支仍进入现有 Finalize，保留记忆策略和终态投递契约。

证据评估与 Research 获得相同的有界问题上下文；消解后的问题用于解释查询目标，原始 question 保留用于审计。仅改写检索词却让后续评估继续面对无指代上下文的“他呢”不算完成多轮修复。

## 6. 证据缺口分类与统一升级策略

扩展 `EvidenceGrade`，保留 decision 与 gaps，新增 gap_type：none、missing_facts、multi_step_required、query_ambiguous、external_realtime_required、external_lookup_required、irrelevant_results、unknown。gaps 为解释文本，不用于字符串匹配分支。`sufficient` 必须匹配 none；clarify/refuse 的原有语义保留。

Evidence Grader 获得问题上下文与实际可用来源描述，负责报告证据缺口，不负责授权或决定工具是否已接入。代码结合 gap_type、能力描述与现有全局研究预算决定下一步。

| 评估结果 | 统一动作 |
|---|---|
| sufficient | generate，并保留既有全部审计 |
| insufficient + missing_facts/multi_step_required | 预算内进入 Research |
| insufficient + irrelevant_results/unknown | 不推断全库没有答案，允许现有预算内的补充检索 |
| clarify/query_ambiguous | 受控澄清 |
| 明确缺外部实时/查询来源，且该来源不可执行 | 停止研究，输出能力限制说明；混合请求按第 5 节澄清 |
| refuse | 保留既有拒绝路径 |

refuse、clarify 和 sufficient 的合法结果优先按 decision 处理；只有 insufficient 再按 gap_type 分支，不能让缺口标签覆盖拒绝决定。Fast RAG 与主图 `evidence_grader` 共用一个纯决策函数，避免两处条件分叉。Research 入口也检查已保存的不可解决能力缺口，防止在后续图重入中重新创建相同任务；不新增一次 LLM 判断。不可解决标记只在当前 Run 内有效，新用户请求必须重新判断。

技术故障不写作语义 gap。检索/评估超时等在现有适用的底层有界重试耗尽后，输出可观察的 cannot_answer，不能仅为了重试同一失效服务而升级整个研究循环。保留取消、租约、Worker 任务重投与幂等机制，不在本次改写基础设施异常分类。

不增加全局相似度阈值，不靠检索低分断言问题域外；不把知识库证据缺失转成“用通用知识猜一个文档答案”。

## 7. 状态、可观测性与兼容

- 新 Run 保存 policy_version、RouteAssessment、执行决定、response_mode、最近对话来源 ID 和有界上下文；能力摘要来自组合根，执行时仍以实际依赖为准。
- 记录 initial_route、executed_path、gap_type、termination_reason。路径只在实际进入节点时追加，避免把计划执行的分支当成已执行；初始 fast_rag 后经过 research 应可明确辨认。
- 事件/SSE 只增加有界枚举与标识字段；不记录原始 Prompt、记忆、隐藏推理、工具载荷或完整分类自由文本。已有持久化 checkpoint 的访问边界保持不变。
- PublicAnswer.route 表示最终交付分支；initial_route 保存在独立状态与安全事件，不能用前者推断完整路径。现有 API route 枚举不增加新值。
- 新 schema/prompt 使用新版本；旧提示词保留供基线评测，不能用新 parser 解析旧 Prompt 输出。现有快照哈希规则不能因新增默认字段改变历史 snapshot_id。
- 旧已完成 Run、历史事件和答案只读保留，不重写诊断记录。旧 RouteDecision/EvidenceGrade 缺字段时按显式 legacy 适配读取，缺失 gap_type 视为 unknown，不推断成外部能力缺失。
- 旧 Run 恢复不追加入当前用户的新对话；缺少已捕获的历史上下文时采用原有有限上下文。部署时先排空活动查询，再以新 graph/prompt/policy 版本启动新 Run；历史记录可读不等于允许混用新旧图执行在途 checkpoint。
- 不新增数据库列，使用现有 JSON 状态/答案和事件契约。将策略与 Prompt 变更纳入现有实现指纹与配置快照，不破坏正式评测的隔离与预算归因。

## 8. 回归与验收

### 8.1 确定性测试

覆盖 RouteAssessment 枚举/组合校验、服务端能力不可被用户或记忆覆盖、混合请求澄清、非法输出/超时不启动研究；正常文档不足仍可研究；两个证据评估入口策略一致；旧 schema 可读；公开能力说明无引用；历史上下文按 user/thread 隔离且不重复记忆抽取；主题切换不被旧历史覆盖。对新策略分支断言节点与依赖调用次数，不只断言最终 route 字符串。

### 8.2 真实模型分类回归

新增独立路由数据集与 runner，不修改当前正式 RAG 评测的题集、金标或计分口径。固定 60 个用例：10 个通用/会话、10 个外部实时/外部查询、20 个知识库单次/多步、10 个多轮指代/主题切换、10 个混合需求/提示注入/澄清。

冻结数据集后，旧 Prompt + 旧 schema 与新 Prompt + 新 schema 各完整运行 3 次，保持同一实际模型、配置和测试上下文；产生逐例结果与聚合指标。模型服务失败单列为无效评测，不按正确路由计分；报告模型标识、配置与提示词哈希、调用耗时及数据集哈希。使用隔离评测身份/thread，禁用真实用户记忆写入；分类评测不执行检索或 Research。

验收门槛：8 个固定核心用例的三次运行全部通过；有效样本中整体路由正确率至少 95%；“必须依赖知识库却绕过检索”和“无需知识库却进入检索”的比例分别不高于 5%。澄清不算完成知识库正确路由，单独报告澄清率，不能靠全量澄清降低误检索率。以上是本次工程准入线，不是生产统计置信保证；同时报告分母、每次结果和相对旧版变化。

固定核心用例为：原始北京天气问题、天气口语改写、上传天气报告摘要、刘泽明京东职责、两份简历比较、在京东经历后追问“做了几年”、从简历切换到天气、文档与实时天气混合请求。除初始标签，还验证规范化问题未篡改意图。

### 8.3 端到端验收

- 原始天气问题：一次分类任务、0 次知识库检索、0 次 Research、无伪造温度/查询成功声明、受控能力说明。
- 强制模拟入口误分后，评估得到明确不可执行外部缺口：首次 Fast RAG 后停止，0 次 Research，不泄露无关引用。
- 上传天气报告、简历事实问题：走知识库，引用与审计正常；不要求所有简单请求一定停在快路径，证据确实不足时可研究。
- 多轮真实 API 请求：历史由服务端加载，指代可解；新主题不被旧话题挟持。
- 混合请求：先说明缺少的能力并澄清，不能静默丢弃实时部分。
- API/SSE、公开答案、持久化事件与 checkpoint 对实际执行路径的记录一致。

真实服务不可用时如实记录“未验收”，不能用 FakeGateway 通过代替模型或端到端结果。性能记录实际耗时和调用计数，不设未经基线验证的固定秒数承诺。

## 9. 预计影响范围与交付记录

修改范围预计包括 `models/schemas.py`、`query/router.py`、`query/state.py`、`query/fast_rag.py`、`query/audit.py`、`query/graph.py`、`query/chat.py`、相关 Prompt、组合根、最近消息的只读 Repository/Worker 接线、公开结果/安全事件投影与测试；新增独立的策略模块和路由评测资产。实际实施计划须逐项列出测试与兼容验证，不顺便重构其他未提交工作。

当前交付仅为书面设计及主设计索引更新。未修改生产代码、未执行真实模型评测、未重启服务。tool/MCP 权限治理仍为明确延期事项；完成实现后应追加实际测试、真实模型结果与部署状态，不把本节的计划描述当成完成记录。
