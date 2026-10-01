# Capability Aware Routing Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 修复非知识库请求误路由、不可解决的证据缺口仍升级研究，以及真实路由回归缺失三个问题。

**Architecture:** 一次轻量模型调用产出信息来源判断，服务端纯策略函数生成执行决定；已有 Evidence Grader 增加缺口分类并复用同一执行策略。独立只读上下文服务提供最近对话，受控 Chat 返回能力限制，保留 RAG 证据审计和现有运行隔离。

**Tech Stack:** Python 3.11、Pydantic、LangGraph、SQLAlchemy async、MySQL、SQLite checkpoint、pytest；复用已有 ModelGateway 和配置的真实模型，不新增框架。

**Spec:** [已确认设计](../specs/2026-10-01-capability-aware-routing-design.md)，用户于 2026-10-01 确认。本文待审阅与执行方式选择，所有未勾选项均未实施。

## Global Constraints

- 一次 Router 语义分类调用，加确定性执行策略，不新增串行分类模型。
- 本次不实现工具级授权、RBAC、MCP credential/scope 管理、用户审批、工具权限动态发现或权限管理界面，不新增通用工具注册平台。
- 不新增天气、网页搜索或其他外部执行器；现有 user_id、文档版本、检索范围、记忆隔离和安全检查保持不变。
- 最近对话最多最近 6 条 user/assistant 消息，总计最多 8000 字符，保留时间顺序，不读取内部 tool 载荷或其他 thread。
- 不改 PDF 解析、chunker、ES 索引或 MySQL Parent，不重建任何数据；本次不需要 SQL 表结构迁移。
- 不修改本次范围外的正式评测、预算、打分与成本实现，不自动重启运行服务。
- 新增字段不能改变历史 snapshot_id；旧 Prompt 保留，禁止新 schema 解析旧 Prompt 输出。
- “能力可用”是组合根已装配执行器的事实，不是权限授予；本次生产 external_realtime/external_lookup 均为 false。
- 固定 60 个用例：10 个通用/会话、10 个外部实时/外部查询、20 个知识库单次/多步、10 个多轮指代/主题切换、10 个混合需求/提示注入/澄清。
- 旧 Prompt + 旧 schema 与新 Prompt + 新 schema 各完整运行 3 次；8 个核心用例三次全过，整体正确率至少 95%，误检索率/漏检索率分别不高于 5%；失败服务调用单列，不作为分类通过。
- 真实服务不可用时如实记录“未验收”，不能用 FakeGateway 通过代替模型或端到端结果。

## Review Focus

- 同一业务 thread 开始第二个 Run 时，不得继承上个 Run 的 route、能力缺口或旧 memory_context；Task 3/6 断言新 Run 重新判断。
- 已检索后转能力说明时，Worker 不能从内部 evidence 再注入无关 parent 引用；Task 7 断言公开答案无引用、内部证据仍保留。
- 重投跨过午夜或对话变更时，解释“今天”和指代所用的上下文应稳定；Task 3/6 固定 run 创建时点与上下文。
- 评估器异常不能被吞成语义 insufficient；Task 5 断言耗尽底层重试后无 Research 额外调用。
- 评测全澄清、丢失难例或 provider 失败不能伪装成正确率提升；Task 8 对分母、缺样本和澄清计分设反例。

## 执行基线与文件职责

本计划基于当前工作树而非干净 HEAD；当前存在正式评测相关的未提交改动，尤其 `query/graph.py`、`runtime/models.py`、`runtime/query_composition.py`、`observability/logging.py`。开始实施前使用 using-git-worktrees 技能检查并选择安全工作位置，记录这些差异；不能在仅有 HEAD 的新 worktree 中假定它们已存在。不能覆盖、回滚或随本功能提交其他人的改动；重叠文件按本任务 hunk 暂存，无法区分时先协调。

验证命令中的 `python` 指 Conda 环境 `/Users/steven/miniconda3/envs/agentic-rag/bin/python`；pytest 已在 pyproject.toml 配置 pythonpath。执行前运行现有 query/runtime 相关测试记录基线，区分既有失败与新回归。以下“预期 FAIL”均指新增行为断言失败，环境/导入配置错误不算有效红灯。

| 文件或文件组 | 职责 |
|---|---|
| `models/schemas.py`、新增 `query/routing_policy.py` | 严格语义契约、能力描述、纯分支决策 |
| 新增 `query/routing_context.py`、`persistence/conversations.py` | 有界上下文类型、同租户同线程历史只读装载 |
| `query/router.py`、新增 Router/Chat v2 Prompt | 单次分类调用、原意保留和响应模式 |
| `query/audit.py`、`query/fast_rag.py`、`query/graph.py` | 缺口评估、共享升级策略、拓扑和错误接线 |
| `query/context.py`、`query/research_loop.py`、`query/state.py` | 后续问题上下文、研究重入保护、Run 级状态 |
| `runtime/query_composition.py`、`runtime/models.py`、`runtime/query_worker.py` | 服务端能力、快照版本、持久化输入/输出接线 |
| `query/chat.py`、`query/public_answer.py`、安全事件/API/UI | 无引用受控回复、最终路由与实际执行路径 |
| 新增 `evals/routing.py`、路由数据集与 runner | 与正式 RAG 评测隔离的分类回归 |

## Task 1: 语义契约及旧数据读取

**Files:** Modify `src/agentic_rag/models/schemas.py`; Create `tests/unit/query/test_routing_schemas.py`.

**Interfaces:** 新增 `InformationSource`、`GapType` Literal，与 Spec 枚举完全一致；`RouteAssessment` 含 required_sources、retrieval_complexity、needs_clarification、normalized_query、reason_code。现有 `EvidenceGrade` 增加可空 gap_type（默认 None，代表 legacy）；新增 `EvidenceGradeV2(EvidenceGrade)` 将 gap_type 设为必填非空，执行 v2 组合校验。`RouteDecision` 保留旧字段/枚举以读取历史。

- [ ] 写测试 `test_route_assessment_combinations`：参数化验证 unknown 只能单独出现、来源不得为空/重复、只有 knowledge_base 才允许 single/multi、需要知识库不能为 none；额外字段禁止、空 normalized_query 禁止。`test_v2_grade_and_legacy_read` 的关键断言：
  ```python
  assert EvidenceGrade.model_validate({"decision": "insufficient", "gaps": []}).gap_type is None
  assert RouteDecision.model_validate(old_route).route == "fast_rag"
  with pytest.raises(ValidationError):
      EvidenceGradeV2(decision="sufficient", gap_type="external_realtime_required")
  ```
  `old_route` 为包含原有三个字段的固定 JSON fixture，不来自本地用户数据。
- [ ] 运行 `python -m pytest tests/unit/query/test_routing_schemas.py -q`，确认新契约未实现时失败。
- [ ] 实现上述类型。RouteAssessment.reason_code 固定枚举 general_conversation、conversation_reference、knowledge_base_lookup、knowledge_base_research、realtime_information_required、external_lookup_required、mixed_sources、clarification_required；不能解析理由文本控制程序。v2 sufficient 要求 none，insufficient 不接受 none；旧缺口在后续策略读取时映射 unknown，不通过 v2 校验器伪造历史字段。
- [ ] 重跑该文件及 `tests/unit/query/test_router_fast_path.py`、`tests/unit/query/test_audit.py`，新增和旧构造测试均通过。
- [ ] 仅提交本任务的 schema 与测试差异：`feat: add versioned routing and evidence gap schemas`。

## Task 2: 能力描述及共享纯策略

**Files:** Create `src/agentic_rag/query/routing_policy.py`, `tests/unit/query/test_routing_policy.py`.

**Interfaces:** `RuntimeCapabilities` 为 frozen Pydantic 对象，version="capabilities-v1"、knowledge_base: bool、external_realtime: Literal[False]=False、external_lookup: Literal[False]=False。本次不暴露开启未来执行器的旗标。`PolicyDecision` 包含 next_node（chat/fast_rag/research_agent/generate/end）、route（chat/fast_rag/research 或 None 表示保持）、response_mode（conversation/capability_unavailable/clarify/technical_error 或 None）、termination_reason: str|None、reason_code: str、missing_sources: tuple[InformationSource,...]。

定义 `decide_route(assessment: RouteAssessment, capabilities: RuntimeCapabilities) -> PolicyDecision`、`decide_grade(grade: EvidenceGrade, assessment: RouteAssessment | None, capabilities: RuntimeCapabilities, *, research_attempts: int, max_research_rounds: int) -> PolicyDecision`、`technical_failure(code: str) -> PolicyDecision`；技术原因只接受 router_unavailable/router_schema_invalid/routing_context_unavailable/retrieval_unavailable/evidence_grader_unavailable。

- [ ] 写参数化测试，覆盖 Spec §5/§6 全部行；核心断言如下，weather/kb/mixed fixtures 为合法 RouteAssessment，capabilities 为 knowledge_base=True：
  ```python
  assert decide_route(weather, capabilities).response_mode == "capability_unavailable"
  assert decide_route(weather, capabilities).route == "chat"
  assert decide_route(kb, capabilities).next_node == "fast_rag"
  assert decide_route(mixed, capabilities).termination_reason == "clarify"
  assert technical_failure("router_unavailable").next_node == "chat"
  ```
  补测 refuse 不被 gap 覆盖、unknown/irrelevant_results 在预算内仍研究、预算耗尽终止、capabilities 无法通过 extra 字段或 bool true 开启外部工具。
- [ ] 运行 `python -m pytest tests/unit/query/test_routing_policy.py -q`，确认缺少策略时失败。
- [ ] 实现纯函数，不使用 gateway、数据库、关键词、相似度阈值或模型置信度。decision=refuse/clarify/sufficient 优先；其余才处理 gap。missing_sources+knowledge_base 混合时澄清；纯外部缺口能力说明；知识库执行器缺失属于技术配置不可用，不允许模型绕过。
- [ ] 重跑 Task 1/2 的测试文件，全部通过；所有语义分支均有单元测试。
- [ ] 仅提交新模块与测试：`feat: centralize capability-aware routing policy`。

## Task 3: 有界最近对话与稳定请求时点

**Files:** Create `src/agentic_rag/query/routing_context.py`, `src/agentic_rag/persistence/conversations.py`, `tests/unit/query/test_routing_context.py`, `tests/integration/persistence/test_conversation_context.py`; Modify `src/agentic_rag/query/state.py`.

**Interfaces:** `RoutingTurn(id: str, role: Literal['user','assistant'], content: str, truncated: bool=False)` 与 `RoutingContext(run_id: str, user_id: str, thread_id: str|None, requested_at: str, timezone: str, history: tuple[RoutingTurn,...], history_available: bool)` 为 frozen JSON-safe 模型。`ConversationReader.load(scope: UserScope, *, run_id: str, thread_id: str) -> Awaitable[RoutingContext]`；SQL 实现 `SqlAlchemyConversationReader(session_factory)`。`RoutingContextUnavailable(RuntimeError)` 表示装载失败，不能用空历史伪装成功。`bound_history(turns: Sequence[RoutingTurn]) -> tuple[RoutingTurn,...]`；`reasoning_question(state: QueryState) -> str` 返回明确标记为不可信数据的原问题、同 Run 规范化问题与有界历史 JSON，不修改 request.question。

- [ ] 写 `test_bound_history_limits`，断言最多 6 条、content 总字符 <=8000、保留时间顺序；预算自最新向旧分配，最后一个保留项只保留尾部并标记已裁剪。SQL 测试覆盖同 user/thread、只读已完成 Run、finished_at 严格早于当前 Run.created_at、跨午夜重读相同日期、零历史/不合法公开答案。`test_reasoning_context_is_run_scoped` 断言不同 run_id 的上下文和规范化问题不被采用，request.question 未改变。
- [ ] 运行新单元测试及新 SQLite/SQLAlchemy 集成测试，确认无 loader 时失败；此任务不连接用户生产 MySQL。
- [ ] 实现只读 SQL：先按 run_id/user_id/thread_id 校验并读取当前 Run.created_at，再查询同 scope 的最近 6 个已完成 Run，按 created_at/id 稳定排序。由 question 和经 PublicAnswer 投影的 answer.segments 重建 user/assistant 对话，生成稳定 `query:<run_id>:user/assistant` ID，再裁剪到 6 条/8000 字符；不依赖目前无生产写入接线的 messages 表，不回填历史数据。日期按显式 Asia/Shanghai 转换，不使用主机本地时区。
- [ ] 重跑新测试；确认读取过程中 SQL 只有 SELECT。新增 QueryState 可选字段 routing_context、route_assessment、routing_policy_version、capabilities、response_mode、initial_route、executed_path、last_evidence_grade、routing_owner_run_id；不把历史复制进 messages。
- [ ] 仅提交本任务差异：`feat: load bounded run-scoped routing context`。

## Task 4: 单次 Router 分类与受控 Chat

**Files:** Modify `src/agentic_rag/query/router.py`, `src/agentic_rag/query/chat.py`, `tests/unit/query/test_router_fast_path.py`; Create `src/agentic_rag/prompts/router_v2.md`, `src/agentic_rag/prompts/chat_v2.md`, `tests/unit/query/test_chat.py`.

**Interfaces:** 扩展 `route_query(state: QueryState, gateway: ModelGateway, *, capabilities: RuntimeCapabilities) -> dict[str, object]`；`run_chat(state: QueryState, gateway: ModelGateway) -> dict[str, object]` 保持外部签名，读取服务端 response_mode；新增 `controlled_reply(mode: str, reason_code: str, *, missing_sources: Sequence[InformationSource]) -> str`，只接受策略枚举。

- [ ] 写 `test_weather_routes_chat_once`，fake gateway 返回 weather assessment，断言一次 complete_structured、next_node=chat、无伪造实时事实；`test_controlled_reply_skips_model` 用会在调用时抛 AssertionError 的 gateway 断言能力说明/澄清/技术错误无需回答模型。补测天气报告、混合请求、记忆声称“已接入天气工具”、指代/切换主题、模型 timeout/schema-invalid、CancelledError 继续向外传播。
- [ ] 运行 `python -m pytest tests/unit/query/test_router_fast_path.py tests/unit/query/test_chat.py -q`，确认天气及错误兜底断言失败。
- [ ] 实现 v2 Prompt，按来源再复杂度分类；系统能力放在独立服务端消息中，用户/记忆/历史一律不可信数据。v2 仅使用 RouteAssessment，新策略派生 RouteDecision；保留 v1 函数路径只供明确 legacy/baseline 选择，不把新结构失败回退旧 Prompt。能力回复如“当前未接入实时天气查询，无法确认今天北京的天气”，其他领域用通用受控文案；缺历史的澄清说明所缺对象，混合请求说明仅可先分析文档。普通 Chat 使用 chat_v2，允许稳定通用知识、禁止伪造实时事实与文档结论。
- [ ] 重跑相关文件；断言所有受控回复 route=chat、无 audited/evidence_ids、status 为对应 clarify/cannot_answer，正常 conversation 无错误状态。路由故障原因保留在 errors，不误写为 capability_unavailable。
- [ ] 仅提交本任务差异：`feat: route by information source with controlled chat responses`。

## Task 5: 缺口评估与技术故障分离

**Files:** Modify `src/agentic_rag/query/audit.py`, `src/agentic_rag/query/fast_rag.py`, `tests/unit/query/test_audit.py`, `tests/unit/query/test_router_fast_path.py`; Create `src/agentic_rag/prompts/evidence_grader_v2.md`.

**Interfaces:** `EvidenceGradingUnavailable(RuntimeError)` 定义在 audit.py，异常只暴露安全 code，原异常作为 cause。EvidenceGrader 的 grade 和 fast_rag.py 中的 Protocol 增加 `routing_context: RoutingContext|None=None` 与 `capabilities: RuntimeCapabilities|None=None` keyword 参数；v2 返回 EvidenceGradeV2，legacy 返回 EvidenceGrade。FastRagDependencies 增加 RuntimeCapabilities。

- [ ] 写 `test_grader_outage_is_not_semantic_gap`：gateway timeout/修复耗尽抛 EvidenceGradingUnavailable，而不是 `insufficient`。`test_fast_rag_realtime_gap_stops_research` 断言首次检索一次后 next_node=chat、response_mode=capability_unavailable；`test_empty_evidence_stays_unknown` 断言无证据不推断“全库没有”或实时能力缺失；补测普通 missing_facts 仍研究、拒绝优先和底层取消传播。
- [ ] 运行上述现有测试文件，确认与旧吞异常/统一升级行为相比出现预期失败。
- [ ] v2 grader 使用完整的有界文本/manifest及 routing_context、能力摘要；空 evidence 的本地结果为 insufficient+unknown。去除把所有异常转换为语义不足的兜底，将非取消、非进程终止异常包成安全的 EvidenceGradingUnavailable。Fast RAG 调用 decide_grade；检索/评估技术错误调用 technical_failure，保留原有底层重试而不升级研究。旧 data/schema 的解析集中在一个适配点，测试 fake port 同步签名。
- [ ] 重跑 Task 1/2/4/5 测试；核对没有新增评估 LLM 调用，原 sufficient 路径与审核测试仍通过。
- [ ] 仅提交本任务差异：`fix: distinguish evidence gaps from unavailable capabilities and providers`。

## Task 6: 主图接线与快照版本隔离

**Files:** Modify `src/agentic_rag/query/graph.py`, `src/agentic_rag/query/context.py`, `src/agentic_rag/query/research_loop.py`, `src/agentic_rag/runtime/query_composition.py`, `src/agentic_rag/runtime/models.py`, `src/agentic_rag/runtime/query_worker.py`; Test `tests/unit/query/test_graph.py`, `tests/unit/query/test_research_loop.py`, `tests/unit/query/test_evaluation_isolation.py`, `tests/unit/runtime/test_models.py`, `tests/unit/runtime/test_query_composition.py`, `tests/integration/persistence/test_sqlite_checkpoint.py`.

**Interfaces:** QueryGraphDependencies 增加 capabilities 与可选 ConversationReader。新增 `prepare_routing_state(state: QueryState, *, capabilities: RuntimeCapabilities, conversations: ConversationReader|None) -> Awaitable[dict[str, object]]` 于 routing_context.py；仅接受当前 Run 的缓存上下文。RuntimeConfigSnapshot 增加 `routing_policy_version: str|None=None`，新组合根设置 routing-v2、graph_version=query-v2、prompt_version=prompt-v2；旧 None 不参与 snapshot_id。

- [ ] 写主图测试：weather 只经过 memory_loader/route/chat/finalize，retrieval 与 research 的 spy 次数均为 0；强制误路由后 v2 grader 修正不得进入研究。Research 后 grader 相同缺口得到相同动作；记录不可解决缺口后重入 research 前不调用 LLM；同业务 thread 新 run 不继承旧阻断/记忆/路由。SQLite 关闭重开后同 run 上下文不变；跨午夜 requested_at 不变；旧 JSON snapshot 哈希与硬编码基线一致。上下文应进入研究 Prompt 以及生成/审核的问题参数，用户原文不被覆盖。
- [ ] 运行这些文件，记录新增断言红灯；先保留已有正式评测隔离与预算测试基线。
- [ ] 在 route 节点开始准备上下文、重新绑定实际 capabilities；Worker 给 request 写入服务端 thread_id；同 Run 可复用 routing_context，不同 Run 显式重置全部 per-run 决策字段和缓存 memory_context，重置必须先于 memory_loader 读取缓存。不得改 Worker 为另一套 checkpoint 恢复算法。无数据库上下文的直接图测试注入稳定 requested_at fixture，生产 Reader 必须成功校验当前 Run；RoutingContextUnavailable 经 technical_failure("routing_context_unavailable") 终止，不能伪称无历史。reasoning_question 传给研究/生成/审核，contexts 不作为证据。
- [ ] 更新条件边允许 route、Fast RAG 与 grader 到 chat，technical_error 仍 finalize；两处评估共用 decide_grade，研究节点进入前重检同 Run 不可解决缺口和全局预算。更新组合根 Prompt 清单/指纹输入，不改现有评测身份逻辑。部署快照不兼容已有 guard 应阻止混用；新版本不执行旧图在途 checkpoint。
- [ ] 重跑本任务和 Task 1–5 测试。断言 routing-v2 缺失的历史快照可读取但不会被默默打上新版本；独立评测 memory disabled 仍生效，多轮历史只能来自同一隔离 thread。
- [ ] 仅暂存属于本任务的 hunk 后提交：`feat: integrate routing policy with graph state and versioned runtime`。

## Task 7: 公开答案与实际路径一致

**Files:** Modify `src/agentic_rag/query/graph.py`, `src/agentic_rag/runtime/query_worker.py`, `src/agentic_rag/query/public_answer.py`, `src/agentic_rag/observability/logging.py`, `src/agentic_rag/api/query_runs.py`, `src/agentic_rag/api/static/app.js`; Test `tests/unit/query/test_public_answer.py`, `tests/unit/observability/test_degradation_events.py`, `tests/integration/runtime/test_query_worker.py`, `tests/integration/api/test_query_runs.py`.

**Interfaces:** `routing_summary(state: QueryState) -> dict[str, object]` 放在 routing_policy.py，仅输出 initial_route、executed_path、gap_type、response_mode、termination_reason 等有界枚举。在 graph 节点实际进入时追加路径；同一个研究节点重入允许重复记录，长度受全局图/研究上限约束。

- [ ] 写 `test_chat_terminal_does_not_inherit_retrieval_citations`：result.evidence 包含 Parent，但最终 controlled Chat 的公开 evidence_parent_ids=[]、audited is None，status 不丢失；内部 result.evidence 未清空。写 `test_initial_route_differs_from_executed_path`：初始 fast_rag、实际 research、最终研究答案 route=research；能力说明则最终 chat。补测 API/SSE 白名单拒绝 Prompt/记忆/用户伪造工具字段，旧事件仍可渲染。
- [ ] 运行以上文件，确认 Worker 当前从 evidence 自动注入 parent_ids 的路径暴露预期失败。
- [ ] 修正 `_public_answer_projection` 使用服务端最终 RouteDecision，chat 时不传 evidence Parent IDs；保持 RAG require_audited 不变。必要时调整 public schema 投影而不放宽引用/审计要求。事件追加枚举摘要，UI 显示实际阶段而非仅最初 QUERY_ROUTED；持久化仍用现有 JSON/事件，不增加 SQL 列。
- [ ] 重跑本任务及主图测试；校验未引入未经审核的知识性回答通道。
- [ ] 仅提交本任务差异：`fix: expose final routing path without leaking unrelated citations`。

## Task 8: 独立真实路由评测入口

**Files:** Create `evals/routing.py`, `evals/datasets/routing_v2.jsonl`, `scripts/eval_routing.py`, `tests/unit/evals/test_routing.py`; Modify `pyproject.toml` 仅在需要时加入路由 JSONL 包资源，不改变正式评测依赖。

**Interfaces:** `RoutingCase` 含 id、group、question、history、required_source_set、expected_route、expected_response_mode、core: bool、normalized_query_must_include: tuple[str,...]；`RoutingSample` 含 case_id、variant、repeat、actual_route、response_mode、normalized_query、latency_ms、error_code、model_id、prompt_hash、dataset_sha256。`async run_routing_eval(cases: Sequence[RoutingCase], gateway: ModelGateway, snapshot: RuntimeConfigSnapshot, *, repeats: int=3) -> list[RoutingSample]`；`score_routing(samples: Sequence[RoutingSample], cases: Sequence[RoutingCase]) -> dict[str, object]`。

- [ ] 写数据/计分测试：精确 60 题及五组 10/10/20/10/10、恰好 8 个 core；每 variant 180 样本。断言全澄清对知识库题计错、遗漏样本/重复 case-repeat 不达标、provider 失败计 invalid 而非通过；误检索分母为应不检索题，漏检索分母为应检索题，并报告澄清率。已知答案中的实体不得在规范化 query 丢失。用 spy 确认 runner 不调用 retrieval/memory write。
- [ ] 运行 `python -m pytest tests/unit/evals/test_routing.py -q`，确认数据/runner 缺失时失败。
- [ ] 实现 CLI：`python scripts/eval_routing.py --dataset evals/datasets/routing_v2.jsonl --repeats 3 --output-dir <新建目录>`。每次创建唯一输出目录，不覆盖已有结果；写 samples.jsonl、report.json、report.md、manifest.json。旧 variant 用 router_v1+RouteDecision，新 variant 用 router_v2+RouteAssessment+decide_route，输入同样有界 question/memory/history/日期/能力；保持同一模型和模型参数，按 case/repeat 交错两 variant 以减少时间偏差。显式记录旧 Prompt 未指导使用新增上下文字段；可附旧生产输入诊断，但不混入受控对照指标。
- [ ] 测试 exact thresholds：8 core 三次全过且无缺样本、有效样本正确率 >=.95、两方向误差各 <=.05；完整验收需新 variant 全部预期调用均有有效结果，不能剔除 provider 失败后宣布通过。状态输出 PASS/FAIL/INCOMPLETE；退出码分别 0/1/2。runner 直接使用 ModelGateway，配置安全加载，报告不得打印 API Key/DSN/原始提供方载荷。
- [ ] 运行新单元测试并用 fake gateway 完整生成 360 条样本的测试报告验证结构；这是计分器测试，不叫真实模型评测。
- [ ] 仅提交新增评测资产及本任务 package-data hunk：`test: add isolated real-model routing regression runner`。

## Task 9: 端到端与实际验收

**Files:** Create `tests/e2e/test_capability_routing.py`, `tests/fixtures/routing_services.py`, `docs/capability-routing-validation.md`; Modify `docs/local-operations.md`、主设计/专项设计的交付状态。

**Interfaces:** `routing_runtime` 为新 async pytest fixture，复用现有 `real_query_runtime` 的隔离生命周期与真实组合根，不改变正式质量评测 fixtures 的口径；仅在隔离数据范围补充真实上传的天气报告、两份简历等测试资产并完成摄取。记忆策略 disabled、唯一 user/thread、独立 checkpoint/索引范围；确切服务环境变量沿用 `tests/fixtures/query_services.py`，缺失时明确 skip。已有直接种子 Parent fixture只能验证协议，不冒充文档上传验收。

- [ ] 写 E2E 断言：天气一次分类、0 检索/研究、能力说明；天气报告/简历有经过审计的文档引用；比较题可研究；两次真实 API 请求指代可解；切换天气不继承知识库路由；混合请求澄清。强制误分的检索后止损使用单独 contract 测试并清晰标记，不能把 fake 分类视为真实模型命中。
- [ ] 运行 contract 图/API 测试，确认在未接线版本上失败；随后用 Task 1–8 实现使其通过。执行 `python -m pytest tests/unit/query tests/unit/runtime tests/unit/observability tests/unit/evals/test_routing.py tests/integration/persistence/test_sqlite_checkpoint.py tests/integration/persistence/test_conversation_context.py tests/integration/runtime/test_query_worker.py tests/integration/api/test_query_runs.py -q`，另对本轮变更文件运行 Ruff 与相关包 Mypy；基线已有失败必须单列，不能静默忽略。
- [ ] 使用 `mktemp -d /private/tmp/agentic-rag-routing-eval.XXXXXX` 建立评测输出目录，运行 Task 8 CLI 的真实模型 360 次分类，保留逐例及版本信息；不要同时运行多个昂贵评测占用同一模型并发。未通过的用例按失败原因修复，数据集冻结后不能删除/改标签来过门槛。
- [ ] 在显式配置的隔离服务上运行 `python -m pytest tests/e2e/test_capability_routing.py -m 'e2e and live_model' -v`。真实服务不可用则保留未验收状态；不启动/重启用户生产 Worker，不清除现有数据以凑通过结果。
- [ ] 将真实命令、退出码、测试数量、模型/数据集/Prompt 哈希、每类指标、未验收项和产物位置写入 validation 文档；local-operations 写明正式切换前排空旧活动 Run、校验新 graph/prompt/policy 快照、显式授权后重启服务。完成代码不等于已部署；tool/MCP 权限治理仍保留延期状态。
- [ ] 仅提交本任务测试/文档：`test: verify routing boundaries end to end and document rollout`；按所选执行流程完成最终独立审查，再报告验收与部署各自状态。

## 顺序与自审记录

任务依赖为 1 → 2 → 3 → 4 → 5 → 6 → 7 → 8 → 9。接口共享较多，推荐当前会话由主代理顺序实施，最后独立审查；若选择 subagent-driven，仍按依赖顺序逐任务交付，避免并行修改相同图和快照文件。

覆盖核对：Spec §1–3 对应全局约束与 Task 1/2；§4 对应 Task 1/3/4；§5 对应 Task 2/4/6/7；§6 对应 Task 2/5/6；§7 对应 Task 3/6/7；§8 对应 Task 8/9；§9 的交付状态在 Task 9 更新。Review Focus 五项均已纳入对应测试。

当前只完成计划编写与设计确认状态更新。没有运行以上实现测试、真实模型或 E2E；没有开始生产代码修改。用户审阅此计划并选择执行方式后，才进入实现。
