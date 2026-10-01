# Persistent Chat Sessions Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [x]`) syntax for tracking.

**Goal:** 实现可持久化、可切换、可继续追问的聊天会话，将问题、完整回答和按需展开的来源放在同一消息流中。

**Architecture:** 新增轻量会话元数据，复用 `agent_runs` 作为每轮问答的唯一事实来源；同一个 session 固定复用同一个 thread，每次发送创建独立 Run。服务端统一处理归属、提交去重、单活动任务、历史分页和来源权限；原生 JavaScript 管理逐会话消息状态与当前视图连接，最终答案只在持久化终态后显示。

**Tech Stack:** 现有 Python 3.11–3.12、FastAPI、SQLAlchemy async/MySQL、Alembic、Redis Streams、LangGraph/SQLite checkpoint、HTML/CSS/JavaScript、pytest、Node `vm` 测试；不增加前端框架或模型依赖。

**Spec:** [持久化多轮聊天与会话管理设计](../specs/2026-10-01-persistent-chat-sessions-design.md)，用户已确认。源码基线 `b989c96`，设计文档提交 `2f71062`。

**Branch / status:** 按用户要求在现有 checkout 的 `main` 分支开发；不另建分支或 worktree。用户已选择当前会话顺序实施（A）；Task 1–8 已实现并验证，已完成一次独立审查及问题修复。

## Global Constraints

- 所有命令在仓库根目录执行；Python 检查统一使用 `conda run -n agentic-rag python -m ...`。Node 仅执行现有风格的本地前端测试。
- 不碰已有未跟踪 `tmp/`，每次提交只暂存本任务明确修改的文件；不推送、不部署、不对应用数据库执行迁移，不自动运行付费模型实验。
- `user_id` 由服务端 `UserScope` 注入；新 session 使用服务端 UUIDv7，`thread_id` 固定等于 session ID。知识库与 Mem0 继续按 `user_id` 共享和隔离。
- 不启用 `messages` 第二份问答存储；不导入旧散落 Run；旧查询/评测 API 保持兼容，聊天 session 不承载 evaluation。
- 每个 session 同时只有一个 `queued/running/cancel_requested` Run。Run 与 Outbox 同事务；答案、来源和会话终态活动时间同事务，保留 owner/generation/lease 检查。
- 同时操作 session 与 Run 时先锁 session 再锁 Run；不跨模型调用、SSE 等待或浏览器网络请求持有事务。
- 默认标题为首问折叠空白后前 30 个 Unicode 字符，超长加省略号；手动标题 1–100 字符；问题去首尾空白，1–32,000 字符，内部换行不折叠。
- 历史默认 30 轮、最大 100 轮；会话列表同样默认 30、最大 100。模型历史保持最多 6 条角色消息、8000 字符。
- 界面只显示简单阶段，不显示检索日志；整条正文只从已完成 Run 的 `PublicAnswer` 投影产生，绝不暴露草稿、prompt、tool payload 或隐藏推理。
- 来源仅取最终回答真正引用的当轮片段：每片段最多 2000 个 Unicode 字符、最多 64 条、整个 UTF-8 JSON 最多 256 KiB；省略必须显式标注。
- 删除为逻辑删除，活动 Run 阻止删除；不删除文档、Mem0、checkpoint。关联旧 Run URL 也不能绕过删除。
- 当前页面内存保存草稿；sessionStorage 只保存 session/request/Run ID 和 SSE 游标，不保存问题、回答、来源、记忆或凭证。
- 距离底部 80px 内自动跟随；SSE 连续三次重连失败后每 2 秒查询 Run；隐藏页面暂停轮询，重新可见立即核验。
- 本次页面恢复不修改 Worker 的进程级 checkpoint 恢复语义；后者已知计数重置问题另行处理。

## Review Focus

1. 发送已落库但响应丢失，刷新后初次查证又返回 404：继续核验原请求，不创建第二个 Run；Task 2、5、6 覆盖。
2. 排队任务取消直接进入终态、重复取消、旧租约 Worker 晚写：会话活动时间只随真实状态迁移更新，答案与来源不得部分落盘；Task 2、3 覆盖。
3. 切换会话后旧 SSE、来源请求、停止响应晚到，或未知事件紧跟终态：不得串线、回退终态或重新开放错误会话的输入；Task 4、6、7 覆盖。
4. 来源已展开后文档被删除/更新，或证据含多字节文本和伪造 AST：重新核验访问权，保留准确历史版本，遵守实际字节上限；Task 3、5、7 覆盖。
5. 同毫秒多条历史、手动标题先于首问、中文输入法 Enter、浏览器禁止 sessionStorage：顺序稳定、标题不覆盖、输入不误发，存储不可用时仍可从服务器列表恢复；Task 1、2、6、7 覆盖。

## File Structure and Interfaces

下列模块均位于 `src/agentic_rag/`；表定义继续集中在 `persistence/repositories.py`，避免创建第二份 metadata。

| 文件 | 新增/修改 | 单一职责 |
| --- | --- | --- |
| `domain/chat_sessions.py` | 新增 | ChatSession、摘要、通用分页值与领域异常 |
| `persistence/chat_sessions.py` | 新增 | 会话 CRUD、轮次/提交查询、来源文档授权查询 |
| `runtime/chat_sessions.py` | 新增 | 短事务编排、创建重试与归属检查；调用 RunManager 发问 |
| `persistence/repositories.py`、`runtime/run_manager.py` | 修改 | Run 扩展、已注册会话约束、幂等创建与终态原子更新 |
| `persistence/schema_readiness.py` | 新增 | 只读核验聊天表/列/索引，不自动迁移 |
| `query/answer_sources.py` | 新增 | 有界、版本化来源快照与校验 |
| `runtime/chat_sources.py` | 新增 | 每次展开的文档授权和安全来源视图 |
| `query/phases.py`、`runtime/query_phase_reader.py` | 新增 | 固定阶段事件的发出与持久化阶段读取 |
| `api/query_context.py` | 新增 | 将旧查询接口已有 scope/snapshot/dependency 获取逻辑原样共享 |
| `api/chat_sessions.py` | 新增 | 会话、轮次、提交查证、来源 HTTP 契约 |
| `api/static/chat-state.js` | 新增 | 可独立测试的状态与请求身份，不操作 DOM/网络 |
| `api/static/chat-api.js` | 新增 | 同源 HTTP、SSE 解码、重连/轮询与终止控制 |
| `api/static/chat-view.js` | 新增 | 消息/会话/来源渲染、滚动与键盘行为 |
| `api/static/console-tools.js` | 新增 | 从 app.js 移出的上传、记忆、健康/配置展示 |
| `api/static/app.js` | 修改 | 页面控制器，连接状态、API、视图和工具 |

前端继续使用有序 `defer` 脚本与 `window.AgenticRagChat` 命名空间，便于沿用 Node vm 测试；不同时引入 ES module 打包。加载顺序：state、api、view、tools、app。

依赖顺序为 **1 → 2 → 3 → 4 → 5 → 6 → 7 → 8**。Task 3/4 虽可分别审阅，执行时仍按此顺序，减少公共接口同时变化。

---

### Task 1: 会话持久化、元数据 CRUD 和游标分页

**Files:**
- Create: `alembic/versions/0008_chat_sessions.py`、`src/agentic_rag/domain/chat_sessions.py`、`src/agentic_rag/persistence/chat_sessions.py`、`src/agentic_rag/runtime/chat_sessions.py`。
- Create: `src/agentic_rag/persistence/schema_readiness.py`、`tests/unit/persistence/test_chat_session_schema.py`、`tests/integration/persistence/test_chat_sessions.py`、`tests/fixtures/chat_sessions.py`。
- Modify: `src/agentic_rag/persistence/repositories.py`、`src/agentic_rag/api/health.py`、`scripts/run_query_worker.py`、`tests/integration/persistence/test_mysql_schema.py`、`tests/unit/api/test_health.py`、`tests/unit/runtime/test_query_worker_entrypoint.py`。

**Interfaces:**
- `ChatSession` 为冻结 dataclass，字段严格对应 spec §4.2；`ChatSessionSummary(session: ChatSession, active_run_id: str | None, active_run_status: RunStatus | None)`；`Page[T](items: tuple[T, ...], next_cursor: str | None)` 使用 Python 3.11 `Generic[T]`。
- 领域异常：`SessionNotFound`、`SessionGone`、`SessionBusy(active_run_id: str)`、`IdempotencyConflict`。后两项供 Task 2 使用，不含原始问题或其他用户信息。
- `ChatSessionService(session_factory: async_sessionmaker[AsyncSession], run_manager: RunManager)`：`create(scope: UserScope, creation_request_id: str) -> tuple[ChatSession, bool]`、`get(scope, session_id) -> ChatSessionSummary`、`list(scope, *, cursor: str | None = None, limit: int = 30) -> Page[ChatSessionSummary]`、`rename(scope, session_id, title: str) -> ChatSession`、`delete(scope, session_id) -> None`。本文 Service/Repository 的 I/O 成员均为 async；未重复标出的 scope/session_id 类型均为 UserScope/str。
- `SqlAlchemyChatSessionRepository(session: AsyncSession)` 提供上述 CRUD 的 SQL 操作与 `get_owned(scope, session_id, *, for_update: bool = False, include_deleted: bool = False) -> ChatSession | None`；事务归 Service 所有，Repository 不自行 commit。
- `async check_chat_schema(engine: AsyncEngine) -> None` 缺少必需结构抛安全的 `ChatSchemaUnavailable`，运维错误明确指向缺少的迁移版本。

- [x] **Step 1: 写会话生命周期与真实 MySQL 竞争测试。** 在 `test_chat_sessions.py` 写 `test_create_replay_rename_delete_and_scope`，核心断言：

```python
first, created = await service.create(user_a, request_id)
same, replay_created = await service.create(user_a, request_id)
assert created and not replay_created and same.id == first.id
assert (await service.list(user_b)).items == ()
await service.rename(user_a, first.id, "手动标题")
assert (await service.get(user_a, first.id)).session.title_source == "manual"
```

另测并发同创建 ID 仅一行、同 ID 在另一用户下独立、删除后 replay 抛 `SessionGone`、重复删除幂等、存在活动任务不能删除。构造相同活动时间的 101 个会话，按 `(last_activity_at,id)` 验证翻页无重无漏；不把动态会话列表误当冻结快照。

- [x] **Step 2: 运行 RED。** `conda run -n agentic-rag python -m pytest tests/unit/persistence/test_chat_session_schema.py tests/integration/persistence/test_chat_sessions.py -q`。预期缺少模块/表而失败；缺少显式测试 DSN 时集成测试跳过必须单独记录，不能算 PASS。
- [x] **Step 3: 实现 schema 与迁移。** revision `0008_chat_sessions`，down_revision `0007_query_run_answer`；会话时间使用 `DateTime().with_variant(mysql.DATETIME(fsp=6), "mysql")`。增加 `(user_id, creation_request_id)` 唯一键、`(user_id, deleted_at, last_activity_at, id)` 索引；Run 增加 nullable `client_request_id`、`answer_sources` 和 `(user_id, thread_id, client_request_id)` 唯一键。保留 active-slot 原约束，不回填历史。迁移结构测试同时核对旧 NULL 请求键可多次存在。
- [x] **Step 4: 实现服务和仓储。** UUID 请求键规范化为标准小写格式；手动标题去首尾空白且 1–100 字符。创建唯一键竞争必须先回滚冲突事务，再用新事务读取原创建结果，不在 aborted transaction 中续查；仅处理对应约束的冲突。列表查询批量关联活动 Run，避免逐会话查询。游标使用有版本/列表类型/UTC 时间/ID 的有界 base64url JSON，非法格式拒绝，所有 SQL 仍独立加用户/未删除过滤。
- [x] **Step 5: 加入 schema 就绪核验。** `build_readiness_checks` 增加 `chat_schema`；Query Worker 在启动消费前调用同一检查。只读检查表、两列和必需唯一键；迁移旧版测试库时应 unready，升级后 ready。测试资源仅由显式 `AGENTIC_RAG_TEST_MYSQL_DSN` 提供；新 fixture 不导入旧 schema 测试模块的全库 downgrade 自动 fixture。
- [x] **Step 6: 验证并提交。** 重跑 Step 2，加上 `tests/unit/api/test_health.py tests/unit/runtime/test_query_worker_entrypoint.py tests/unit/test_migrations.py tests/integration/persistence/test_mysql_schema.py`；要求配置测试库时无跳过且通过。提交本任务 Files：`feat: persist chat sessions and scoped history metadata`。

### Task 2: 幂等发问、轮次读取与 Run 生命周期一致性

**Files:**
- Modify: `src/agentic_rag/persistence/repositories.py`、`src/agentic_rag/persistence/chat_sessions.py`、`src/agentic_rag/runtime/run_manager.py`、`src/agentic_rag/runtime/chat_sessions.py`。
- Create: `tests/integration/persistence/test_chat_turns.py`。
- Modify: `tests/integration/persistence/test_chat_sessions.py`、`tests/unit/persistence/test_repository_contracts.py`、`tests/integration/persistence/test_conversation_context.py`。

**Interfaces:**
- `QueryRun` 增加带默认值的 `client_request_id: str | None`、`created_at: datetime | None`、`finished_at: datetime | None`、`answer_sources: dict[str, Any] | None`；SQL 读出的真实记录必须有 created_at。
- 在 `runtime/run_manager.py` 定义 `RunSubmission(run: QueryRun, created: bool)`；新增 `RunManager.submit_chat(scope: UserScope, session_id: str, question: str, snapshot: RuntimeConfigSnapshot, *, client_request_id: str) -> RunSubmission`。
- `ChatSessionService.submit(...) -> RunSubmission` 同参数委托上述方法；`turns(scope, session_id, *, cursor=None, limit=30) -> Page[QueryRun]`；`find_submission(scope, session_id, client_request_id) -> QueryRun | None`。
- Repository 新增 `find_submission(scope, session_id, client_request_id) -> QueryRun | None` 与 `list_turns(scope, session_id, *, cursor, limit) -> Page[QueryRun]`；Task 5 负责 HTTP 投影，绝不直接序列化 QueryRun。
- `RunRepository.create_queued` 及两个 SQL/事务适配器增加 keyword `client_request_id: str | None = None`；`finish` 增加 keyword `answer_sources: dict[str, Any] | None = None`。所有实际和测试适配器同步，旧调用方不传时仍成立。

- [x] **Step 1: 写事务故障与竞争测试。** `test_duplicate_submission_is_one_run_and_outbox` 使用两条独立 MySQL 连接同时提交：

```python
results = await asyncio.gather(submit(sid, key, "首问"), submit(sid, key, "首问"))
assert len({r.run.id for r in results}) == 1
assert sorted(r.created for r in results) == [False, True]
assert await count_session_runs(sid) == await count_session_outbox(sid) == 1
```

这些测试辅助函数在本测试模块定义，`submit` 包装真实 Service，计数查询均按本测试用户和 session 过滤。补测同 key 不同问题、内部换行差异、终态后 replay、不同 key 竞争、Outbox 写失败回滚标题和 Run、删除/发问双向竞争。手动命名后再发首问必须保持手动标题；未命名会话的 emoji 首问按 Unicode 字符截取 30 字。重复执行竞争测试以显式 barrier 调度，不依赖固定 sleep。
- [x] **Step 2: 运行 RED。** `conda run -n agentic-rag python -m pytest tests/integration/persistence/test_chat_turns.py -q`，预期新方法/字段未实现；需使用显式测试库确认失败来自待实现行为。
- [x] **Step 3: 实现统一提交事务。** 先锁 session，校验归属/未删除/evaluation 禁止；先查原请求，再检查 active-slot，最后创建 Run/Outbox/首问标题/活动时间。旧 `RunManager.create` 签名保持不变；底层 `create_queued` 对已注册 thread 强制同样的 session 检查和标题/活动更新，不把旧未知 thread 自动建为 session。Foreign session 也不能当作“未知 thread”继续创建。
- [x] **Step 4: 统一生命周期与删除边界。** 在 SQL 层新增共享 session 锁辅助函数供创建、取消、finish 使用，避免跨模块循环导入。`get` 拒绝已删除关联会话，可信 Worker 的 `get_for_delivery` 仍能看到历史 Run 并 ACK 重复投递。queued→cancelled 在取消事务内更新活动时间；running→cancel_requested 不更新终态时间；重复取消/finish 不刷新活动时间。finish 的 lease fence 更新不成功时回滚全部写入，成功后才写 session 活动时间。
- [x] **Step 5: 实现完整轮次分页和查证。** SQL 按 `(created_at DESC,id DESC)` 取 limit+1，页内返回升序，游标取本页最旧项；传入其他 session 的游标仍不得改变查询归属。历史读取包含所有状态，模型 ConversationReader 继续只读既有已完成公开内容，固定 6 条消息/8000 字符；补跨用户、跨 session、同时间戳、失败/活动排除的回归断言。
- [x] **Step 6: 验证并提交。** 重跑 Step 2，以及 `tests/integration/persistence/test_chat_sessions.py tests/integration/persistence/test_conversation_context.py tests/unit/persistence/test_repository_contracts.py tests/integration/runtime/test_query_worker.py`。包括 queued cancel、重复 cancel 和过期 generation 不改变来源/活动时间的断言；提交 `feat: make chat turns idempotent and transactionally consistent`。

### Task 3: 最终答案的历史来源快照与展开授权

**Files:**
- Create: `src/agentic_rag/query/answer_sources.py`、`src/agentic_rag/runtime/chat_sources.py`、`tests/unit/query/test_answer_sources.py`、`tests/integration/persistence/test_chat_sources.py`。
- Modify: `src/agentic_rag/runtime/query_worker.py`、`src/agentic_rag/domain/chat_sessions.py`、`src/agentic_rag/persistence/chat_sessions.py`、`tests/integration/runtime/test_query_worker.py`。

**Interfaces:**
- `AnswerSource` 字段为 evidence/document/version/parent ID、heading_path、heading_truncated、可空 page_from/page_to、excerpt、truncated、excerpt_omitted；`AnswerSources` 为严格 Pydantic 模型，含 `schema_version=1`、run_id、runtime_config_snapshot_id、items、omitted_source_count。
- `build_answer_sources(answer: PublicAnswer, packed: PackedEvidence, *, run_id: str, snapshot_id: str) -> AnswerSources | None`；不安全/缺失 manifest 抛 `InvalidAnswerSources`，Worker 将该附加投影降级为 None，不改变通过审核的答案。
- 在 `domain/chat_sessions.py` 增加 `SourceDocumentAccess(document_id: str, document_version_id: str, filename: str, active_version_id: str | None)` 冻结 dataclass；`SqlAlchemyChatSessionRepository.source_documents(scope: UserScope, pairs: Sequence[tuple[str, str]]) -> dict[tuple[str, str], SourceDocumentAccess]` 以 document/version 对批量查授权，不返回无权限项。
- `ChatSourceService(session_factory).get(scope: UserScope, session_id: str, run_id: str) -> SourceView`；`SourceView` 含 `status: Literal['available','partial','unavailable','none']`、items、omitted_source_count；每项只含显示需要的文件名、章节/页码、片段、版本状态和截断标记。

- [x] **Step 1: 写投影边界测试。** `test_only_cited_packed_content_is_snapshotted`、`test_multibyte_snapshot_is_bounded_and_marks_omissions` 的关键断言：

```python
snapshot = build_answer_sources(answer, packed, run_id="r", snapshot_id="s")
assert [item.evidence_id for item in snapshot.items] == ["e2", "e1"]
assert all(len(item.excerpt) <= 2000 for item in snapshot.items)
assert len(snapshot.items) <= 64
assert len(snapshot.model_dump_json().encode("utf-8")) <= 256 * 1024
assert len(json.dumps(snapshot.model_dump(mode="json")).encode("utf-8")) <= 256 * 1024
```

fixture answer 首次引用顺序为 e2/e1，packed 另含未引用 e3。另造超过 64 条、四字节 Unicode、JSON 转义文本、ID/manifest 不一致、无正文/纯 Chat/未审核答案、损坏 locator。超过条数时 omitted_source_count 必须准确，预算耗尽有 excerpt_omitted 标志。
- [x] **Step 2: 运行 RED。** `conda run -n agentic-rag python -m pytest tests/unit/query/test_answer_sources.py tests/integration/persistence/test_chat_sources.py -q`。
- [x] **Step 3: 实现确定性来源投影。** ID 匹配 `[A-Za-z0-9_.:-]{1,255}`；heading 最多 16 段且每条来源的标题文字累计最多 128 个 Unicode 字符，截断设置 heading_truncated。先核验原始 item/manifest 一致，再做展示截断；先预留有界元数据再分配片段预算。字节检查同时覆盖 UTF-8 输出和 SQLAlchemy 默认 JSON ASCII 转义形式，不能只用字符数或紧凑输出估计持久化大小。页码仅从校验后的现有 `AstLocator.spans` 提取，无法验证则为 null，不暴露 locator 原文。内容只来自匹配 manifest 的 `PackedEvidence.items`，不从整个 parent 或后续 checkpoint 替换。
- [x] **Step 4: 接入 Worker 与终态事务。** 从本次 graph result 的 `packed_context` 解析 PackedEvidence，且仅在本轮合法公开答案完成时构建来源；将 answer 与 answer_sources 一同传入 Task 2 的 finish。迁移 Worker 测试端口，避免现有捕获 `TypeError` 的兼容分支静默丢掉生产来源写入。注入来源写入失败确认答案/状态也回滚；注入投影校验失败则确认已审核答案可完成且来源 unavailable。
- [x] **Step 5: 实现每次展开的授权视图。** 一次短事务核验未删除 session、Run 归属及快照绑定的 run/snapshot ID；批量 join 当前用户 documents 和对应 document_versions。`status='deleted'` 或任何 deletion_status、不存在/错属版本均隐藏片段；可访问的历史版本保留旧片段并标注 historical。文件名只从受控查询读取；无快照的有引用回答返回 unavailable，纯 Chat 返回 none。损坏版本/超限快照不退回 raw artifact。
- [x] **Step 6: 验证并提交。** 重跑 Step 2，加 `tests/integration/runtime/test_query_worker.py tests/integration/persistence/test_chat_turns.py`。验证更新文档仍显示旧片段、删除后不再返回片段、错误用户/Run/snapshot 不泄露；提交 `feat: preserve scoped source excerpts for each answer`。

### Task 4: 真实执行阶段及安全事件投影

**Files:**
- Create: `src/agentic_rag/query/phases.py`、`src/agentic_rag/runtime/query_phase_reader.py`、`tests/unit/query/test_phases.py`。
- Modify: `src/agentic_rag/query/graph.py`、`src/agentic_rag/query/fast_rag.py`、`src/agentic_rag/query/audit.py`、`src/agentic_rag/observability/logging.py`、`src/agentic_rag/api/query_runs.py`。
- Modify: `tests/unit/query/test_graph.py`、`tests/unit/query/test_audit.py`、`tests/unit/query/test_router_fast_path.py`、`tests/integration/api/test_query_runs.py`。

**Interfaces:**
- `QueryPhase = Literal['processing','retrieving','researching','auditing']`；`PhaseReporter = Callable[[QueryPhase], Awaitable[None]]`。
- `QueryPhaseEmitter(*, run_id: str, scope: UserScope, snapshot: RuntimeConfigSnapshot, event_emitter: AgentEventEmitter | None = None, event_repository: EventRepository | None = None)`；`async report(phase: QueryPhase) -> None` 发出 `QUERY_PHASE_CHANGED`。复用 graph 的 snapshot 重绑定规则，只吞普通事件写入错误，不吞 cancellation/process-exit。
- `run_fast_rag(state, dependencies, *, report_phase: PhaseReporter | None = None)` 与 `generate_with_mandatory_audits(..., report_phase: PhaseReporter | None = None)`，旧调用参数仍可用；reporter 不进入 checkpoint。
- `QueryPhaseReader(session_factory).latest(scope: UserScope, run_ids: Sequence[str]) -> dict[str, QueryPhase]` 批量读取每个 Run 最新有效阶段，无记录则调用方用 processing。

- [x] **Step 1: 写顺序和失败隔离测试。** `test_audit_phase_precedes_actual_audit_and_repeats_on_revision` 使用记录型 generator/auditor/reporter：

```python
assert calls == ["processing", "generate", "auditing", "audit_fail",
                 "processing", "generate", "auditing", "audit_pass"]
```

另测 fast retrieval 前为 retrieving、其内部证据 grader 前为 auditing；research 子检索不发 retrieving；reporter 故障不改变答案/路由/预算；取消照常传播。
- [x] **Step 2: 运行 RED。** `conda run -n agentic-rag python -m pytest tests/unit/query/test_phases.py tests/unit/query/test_audit.py tests/unit/query/test_router_fast_path.py -q`。
- [x] **Step 3: 实现阶段发出。** graph 路由/Chat/生成为 processing，fast 内部检索为 retrieving，research 循环为 researching，实际 grader/auditor 前为 auditing。重用事件行 summary 存固定枚举，并将这四个值加入安全 summary 白名单；不写原始日志、不依赖 artifact 读回 phase。每次转换使用独立事件 key，保留发生顺序，不能把第二次审核误去重掉。恢复重复事件由前端按 ID 去重，终态优先。
- [x] **Step 4: 实现阶段查询和 SSE 白名单。** 只接受固定事件类型和枚举；SSE 新事件只在既有安全 envelope 上添加 phase，不投影任意 attributes。批量 reader 过滤用户和 Run ID；`QueryRunResponse` 添加可选 phase，旧测试容器未注入 reader 时安全回退 processing；queued/cancel_requested/terminal 的 UI 状态仍由 Run status 决定。`ANSWER_FINALIZED` 只触发核验 Run，不直接发布内容。
- [x] **Step 5: 验证并提交。** 重跑 Step 2，加 `tests/unit/query/test_graph.py tests/integration/api/test_query_runs.py tests/unit/runtime/test_evaluation_identity.py`；未知 phase 不输出、晚阶段不改变终态、阶段缺失回退 processing。实现指纹按现有全源码 hash 自然更新，不改旧冻结报告；提交 `feat: expose safe live query phases for chat`。

### Task 5: 会话 HTTP 接口与生产依赖装配

**Files:**
- Create: `src/agentic_rag/api/chat_sessions.py`、`src/agentic_rag/api/query_context.py`、`tests/integration/api/test_chat_sessions.py`。
- Modify: `src/agentic_rag/api/app.py`、`src/agentic_rag/api/query_runs.py`、`src/agentic_rag/api/errors.py`、`src/agentic_rag/bootstrap.py`、`tests/integration/api/test_query_runs.py`、`tests/fixtures/query_services.py`。

**Interfaces:**
- 新 router 完整实现 spec §5 的九个 method/path，创建请求仅 creation_request_id；发问仅 query/client_request_id。所有请求 `extra='forbid'`，ID 为 UUID，分页 limit 1–100。
- `SessionResponse`：session_id、title、title_source、created_at、updated_at、last_activity_at、active_run_id、active_run_status、phase；公开响应不暴露 user_id/deleted_at。
- `TurnResponse`：run_id、client_request_id、created_at、finished_at、question、status、answer: PublicAnswer | None、phase、source_status、terminal_code；列表为 `{items, next_cursor}`。发问与 submissions 均返回 TurnResponse；前者 202/200 区分新建/重放。公开时间统一为带 Z 的 UTC RFC3339、六位小数；前端排序保留完整时间精度，不用 Date 毫秒截断替代游标顺序。
- `query_context.py` 提供 `request_scope(request: Request) -> UserScope`、`runtime_snapshot(request: Request) -> RuntimeConfigSnapshot`、`require_dependency(request: Request, name: str) -> Any`；从旧私有 helpers 提取，保留原行为，不重写 evaluation 策略。
- AppContainer 新增 chat_session_service、chat_source_service、query_phase_reader 三个可空字段；build_container 生产路径注入真实实现。

- [x] **Step 1: 写 API 契约测试。** `test_session_turn_replay_and_reload_contract` 关键断言：

```python
assert created.status_code == 201 and replay_session.status_code == 200
assert submitted.status_code == 202 and replay_turn.status_code == 200
assert submitted.json()["run_id"] == replay_turn.json()["run_id"]
assert history.json()["items"][0]["question"] == "首问"
assert sources.headers["cache-control"] == "no-store"
```

请求全部通过 ASGI client；补九个接口、两用户、410 创建重放、404 删除/未知、422 额外字段/非法 UUID/limit/长问题、409 busy 与 idempotency 区别。明确模拟“服务端已提交后响应丢失”的原键查证，不只测试 Fake 返回缓存。
- [x] **Step 2: 运行 RED。** `conda run -n agentic-rag python -m pytest tests/integration/api/test_chat_sessions.py -q`。
- [x] **Step 3: 实现 router/响应和错误映射。** `SessionNotFound`→404、`SessionGone`→410（仅创建重放）、`SessionBusy`→409 SESSION_BUSY + 指向已授权 Run 的 Location header、`IdempotencyConflict`→409 IDEMPOTENCY_CONFLICT。busy 不回显草稿；旧非聊天接口仍保持 ACTIVE_RUN_EXISTS。来源及含历史正文的读取设置 no-store；JSON 错误沿用安全信封，不把 exception detail 返回 UI。
- [x] **Step 4: 装配与旧入口回归。** 注册 router 和真实服务；轮次终态投影统一使用 project_public_answer，未完成不返回正文；safe terminal_code 从固定状态映射。`_owned_run`/取消/每轮 SSE 核验走 Task 2 删除检查，已发起的流也不能在删除后继续读历史。为运行中测试 fixture 注入相同 session_factory/outbox stream 的服务，禁止新聊天绕回默认生产队列。
- [x] **Step 5: 验证并提交。** 重跑 Step 2，加 `tests/integration/api/test_query_runs.py tests/unit/test_bootstrap_close.py tests/unit/api/test_runtime_summary.py`；旧非聊天 query/evaluation 正常，已删除/foreign 聊天 thread 无法通过旧入口创建或访问。提交 `feat: serve persistent chat session and source APIs`。

### Task 6: 前端逐会话状态、提交恢复与连接生命周期

**Files:**
- Create: `src/agentic_rag/api/static/chat-state.js`、`src/agentic_rag/api/static/chat-api.js`、`tests/integration/api/chat_state.cjs`、`tests/integration/api/chat_transport.cjs`、`tests/integration/api/test_chat_client.py`。

**Interfaces:**
- `createChatState() -> ChatState`，包含 selectedSessionId、viewGeneration、sessions Map；每个 SessionState 含 turns Map、orderedRunIds、draft、pendingSubmission、historyCursor；每轮独立 phase/cursor/terminal。
- `activateSession(state, sessionId) -> {sessionId, generation}`；`applyRun(state, token, turn) -> boolean` 仅在视图代次匹配且未降级终态时接受；`applyPhase(state, token, runId, eventId, phase) -> boolean` 同样有界去重。
- `beginSubmission(state, sessionId, question, requestId) -> PendingSubmission` 只保留一个未知操作；pending 绑定被提交的文本快照，不会随正在编辑的下一条草稿改变。
- `createChatApi({fetchImpl, timers})` 提供 createSession、listSessions、getSession、renameSession、deleteSession、listTurns、submitTurn、findSubmission、getSources、getRun、cancelRun、watchRun，参数/响应与 Task 5 一致；`watchRun({runId, cursor, signal, onPhase, onRun}) -> Promise<void>`。
- `saveResumeMetadata(storage, state) -> void`、`loadResumeMetadata(storage) -> ResumeMetadata`；仅固定白名单字段，版本化且每次异常可降级为空缓存。

- [x] **Step 1: 写 Node 状态/网络测试。** 在 chat_state.cjs 构造两个 session 与过期视图 token，断言：

```javascript
assert.equal(applyRun(state, oldToken, completedA), false);
assert.equal(state.selectedSessionId, "session-b");
assert.equal(applyPhase(state, currentToken, completedB.run_id, 99, "auditing"), false);
assert.equal(serializedResume.includes("问题秘密"), false);
```

补 lost-response→同 key、查证暂时 404→仍未知、SESSION_BUSY→保留草稿、重复页按 Run ID 去重、storage 抛 SecurityError 不影响聊天、切换保留草稿。chat_transport.cjs 用可控 fake clock/fetch 测试三次重连失败后 2 秒轮询、隐藏/恢复、UTF-8/SSE 跨 chunk、正常 EOF 后 GET、abort 不触发 cancel。
- [x] **Step 2: 运行 RED。** `conda run -n agentic-rag python -m pytest tests/integration/api/test_chat_client.py -q`；pytest wrapper 必须执行两个 Node 脚本，并在异常时包含有界 stderr。
- [x] **Step 3: 实现纯状态层。** Run 终态不可被旧事件回退，历史合并按 created_at/UUID 排序且不把重复页当新消息。重复发送同一 pending 操作用相同 request ID；明确受理后才清空与提交快照仍相同的草稿，避免晚响应删掉用户新输入。草稿仅内存；储存失败显示可恢复提示，从服务器列表仍可找回已受理轮次。
- [x] **Step 4: 实现 HTTP/SSE 层。** 使用 fetch + AbortController，延续 Last-Event-ID；重连延迟固定 0.5/1/2 秒，三次重连失败后轮询，成功连接重置次数。停止观察不等于取消任务；stop POST 成功后持续 GET/SSE 直到实际终态。404/410 会话已删除时关闭观察；瞬时网络错误只报告连接状态。请求响应均使用匹配 session/run/token 的回调，不写 DOM。
- [x] **Step 5: 验证并提交。** 重跑 Step 2，确认无真实网络/模型调用、fake timers 不遗留 pending 任务；提交 `feat: manage chat state and recover interrupted observations`。

### Task 7: 连续消息界面、来源展开与工具抽屉

**Files:**
- Create: `src/agentic_rag/api/static/chat-view.js`、`src/agentic_rag/api/static/console-tools.js`、`tests/integration/api/chat_view_flow.cjs`。
- Modify: `src/agentic_rag/api/static/index.html`、`src/agentic_rag/api/static/app.css`、`src/agentic_rag/api/static/app.js`、`src/agentic_rag/api/app.py`。
- Modify: `tests/integration/api/test_console.py`、`tests/integration/api/console_flow.cjs`、`tests/integration/api/console_terminal_flow.cjs`、`tests/integration/api/test_chat_client.py`。

**Interfaces:**
- `createChatView(document)`：`renderSessionList(items)`、`renderSession(sessionState)`、`updateTurn(turn)`、`prependTurns(turns)`、`renderSources(runId, sourceView)`、`setComposer({draft, canSend, canStop})`、`scrollToLatest()`；只接受 Task 5/6 的受控值。
- `createConsoleTools({document, fetchImpl, onDocumentsChanged})` 提供 `open(kind: 'documents' | 'memory' | 'system')`、`close()`；移用现有上传轮询、Mem0 管理、健康与 runtime summary 逻辑，不更改用户隔离。
- app.js 控制器负责初始化服务器列表、创建/切换/命名/删除、提交/查证/停止、可见性变化，以及来源重新授权；所有 view 写入前核验当前 token。

- [x] **Step 1: 写界面行为测试。** chat_view_flow.cjs 与现有 console flow 的断言覆盖：

```javascript
assert.equal(messageList.children.length, 6); // 三轮用户/系统消息
assert.equal(sendCountAfterCompositionEnter, 0);
assert.equal(scroller.scrollTop, beforeUpdateScrollTop); // 用户在上方阅读
assert.equal(rawLogPanelCount, 0);
assert.equal(answerA.textContent, originalAnswerA); // B 完成不覆盖 A
```

fixture 在实际事件监听器上触发 compositionstart/end、keydown、scroll，而非只测同名辅助函数。来源折叠/再次展开、旧来源响应晚到、工具删除文档后、页面重新聚焦时均测重新授权；文本含 `<img onerror>` 必须原样作为文本，无 DOM 注入。
- [x] **Step 2: 运行 RED。** `conda run -n agentic-rag python -m pytest tests/integration/api/test_console.py tests/integration/api/test_chat_client.py -q`。先让新交互断言失败，不能以删除安全回归测试换取通过。
- [x] **Step 3: 实现布局与渲染。** 左列表、中间消息流、底部 composer；窄屏断点 768px，列表/工具改抽屉且有返回焦点、Escape 和键盘关闭。用户/系统独立消息节点，进行中只更新对应占位，正文保留换行/段落；禁止 innerHTML 注入原始内容。初始最新 30 轮定位底部，上翻 prepend 以前后 scrollHeight 差补偿位置；≤80px 跟随，来源展开不强制跟随，按钮“回到最新”。
- [x] **Step 4: 接入所有会话操作。** 新建请求去重、标题前 30 字/手动命名、删除确认与 busy 提示；发送期间仍可编辑下一草稿，Enter/Shift+Enter 与 IME 正确；停任务后等终态。长度使用 Unicode code point 计数，与 Python 保持一致，不依赖 textarea 的 UTF-16 maxlength；测试 emoji 输入边界。未知提交刷新后查证同 key，界面提供“检查提交状态”，不把 404 宣称为未执行；旧待确认操作在内存草稿完整时可用原 key 重试。
- [x] **Step 5: 接入来源和工具。** 每条回答独立“查看来源”，仅在展开时请求；来源状态/省略/历史版本按 Task 3 文案呈现。切换会话、文档工具变更、重新聚焦时清除旧来源片段再核验；不借用任何全局 evidence 容器。将原工具功能迁入抽屉，移除检索时间线和全局答案/证据 DOM；安全终态说明保留，技术日志不进入聊天。
- [x] **Step 6: 更新资源交付与验证。** 静态白名单仅增加 chat-state.js/chat-api.js/chat-view.js/console-tools.js；HTML 按既定 defer 顺序加载，未知文件仍 404。新 DOM 用稳定 ID：session-list、new-chat、chat-messages、query-form、query-input、send-button、cancel-button、scroll-latest、tools-drawer。重跑 Step 2，并保留纯 Chat 澄清/拒答正文、未审核 RAG 草稿禁止展示等已有回归；提交 `feat: replace single-answer console with persistent chat UI`。

### Task 8: 集成验收、实际浏览器检查与运维说明

**Files:**
- Create: `tests/integration/runtime/test_chat_pipeline.py`、`docs/validation/2026-10-01-persistent-chat-sessions.md`。
- Modify: `tests/fixtures/query_services.py`、`tests/e2e/test_query_console_real_services.py`、`docs/local-operations.md`；仅在必要时更新 README 中旧控制台入口描述。

**Interfaces:**
- 新集成测试复用现有 `real_query_fixture`（真实 MySQL/Redis/ES + 确定性模型端口），不是会调用付费 provider 的 `real_query_runtime`。
- 新 fixture 的聊天服务与 RunManager 必须使用同一隔离 broker/outbox stream。清理只按本次测试 user/session/stream，增加 chat_sessions 清理，不扩大到其他数据。
- 验收记录逐项标明 PASS/FAIL/NOT RUN、命令、运行环境、必要证据和未解决问题；不把集成跳过、人工步骤未执行记作通过。

- [x] **Step 1: 写真实边界集成测试。** `test_three_turn_chat_survives_new_client_and_preserves_scope` 在一个 session 受理并完成三轮后销毁 HTTP client，再用新 client 读取历史：先断言 3 个不同 Run、同一个 thread、6 条界面消息的数据基础，然后提交第 4 问并检查仍使用该 thread 和限定的 ConversationReader 历史。`test_background_completion_and_source_versions` 切到 session B 后让 A 完成，再读取 A 的最终正文和专属来源。额外断言两个 session 的 Mem0 调用都是同一 user_id，另一用户不可读取。
- [x] **Step 2: 运行真实服务验收。** 显式准备隔离的 `AGENTIC_RAG_TEST_MYSQL_DSN`、`AGENTIC_RAG_TEST_REDIS_DSN`、`AGENTIC_RAG_TEST_ELASTICSEARCH_URL` 后运行 `conda run -n agentic-rag python -m pytest tests/integration/runtime/test_chat_pipeline.py tests/integration/runtime/test_query_pipeline.py -q`。预期生产 API→Run/Outbox→Worker→Graph→终态链路通过；没有可用隔离服务则记录 NOT RUN，保留结论限制。
- [x] **Step 3: 完成一次针对性回归。** 运行 `conda run -n agentic-rag python -m pytest tests/unit tests/integration/api tests/integration/persistence/test_chat_sessions.py tests/integration/persistence/test_chat_turns.py tests/integration/persistence/test_chat_sources.py tests/integration/persistence/test_conversation_context.py tests/integration/runtime/test_query_worker.py -q`。Ruff 只检查本次变更的 Python 文件；记录已有类型检查基线，检查触及模块新增错误，不顺手重构无关代码。只有新失败或修复后才追加针对性重跑。
- [x] **Step 4: 实际浏览器验收。** 使用隔离测试后端打开页面，在 1440px/390px 宽度检查三轮问答、列表折叠、重命名、刷新进行中任务、切换期间完成、停止、来源展开与文档删除、滚动/回到最新、中文输入、键盘焦点。用可控延迟/断网验证旧请求晚到和轮询回退；记录截图/结果，不只以 Node 虚拟 DOM 代替真实布局验证。
- [x] **Step 5: 更新运维文档。** 明确 main 开发、历史列表从新功能起、逻辑删除语义、完整历史与模型窗口差别、备份/排空 Run/增量迁移/协调 API+Worker+静态发布步骤。文档给出检查命令，不在此步执行部署；回滚应用保留新增表列。更新旧 live-provider smoke 的 DOM/接口断言，但不自动触发真实 provider。
- [x] **Step 6: 完成最终审查并提交。** 对照 spec §9 验收矩阵检查全部实现与证据，解决功能内发现的问题，再按选定执行方式完成独立代码审查；`git diff --check` 必须通过，确认当前分支仍为 main 且仅包含目标改动。提交 `test: verify persistent chat workflows and document rollout`，向用户报告实际通过项及任何未执行项。

## Coverage and Handoff

| Spec 章节 | 负责的任务 |
| --- | --- |
| §1–3 会话交互、布局、输入、滚动 | Task 1、6、7、8 |
| §4 身份、模型、去重和事务 | Task 1、2、5 |
| §5 HTTP 契约 | Task 5 |
| §6 阶段、断线/刷新恢复、多轮上下文 | Task 2、4、6、8 |
| §7 历史来源与授权 | Task 3、5、7 |
| §8 旧 API/部署/文件职责 | Task 1、2、5、7、8 |
| §9–10 验收与已确认边界 | Task 8 与 Global Constraints |

推荐由当前会话顺序实施（Native）：数据库事务、HTTP 契约与前端恢复状态相互依赖，连续持有上下文更容易保持接口一致；完成后再做一次独立整体审查。另一选择是逐任务分配实现与审查子代理，审查更频繁，但上下文和审查开销更高。两种方式都遵守用户指定的 `main`，不切回其他开发分支。

本计划已按用户选择顺序实施；各任务证据、实施裁决与最终审查记录见 docs/validation/2026-10-01-persistent-chat-sessions.md。
