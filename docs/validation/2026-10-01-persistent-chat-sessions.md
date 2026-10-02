# 持久化聊天验收记录

后续检查与修复见 [2026-10-02 二次检查](2026-10-02-persistent-chat-review.md)，其中修正了本文最后一项创建恢复测试与真实传输契约不一致的问题。

日期：2026-10-01。实现基线：`4b1983b`；按用户确认在 `main` 开发。功能已实现；已完成一次独立审查，并以失败复现驱动修复全部 3 个 Important 问题。没有推送、部署、重启应用服务或迁移应用数据库。

## 实际环境与边界

Python 3.11（conda `agentic-rag`）、Node 24.19.0、本机 Chrome；测试使用本次专用 MySQL 数据库 `agentic_rag_chat_test_89a527b6cf55`，已在该库验证 0008 迁移。Redis Stream、Elasticsearch 索引、用户 ID、checkpoint 和 artifacts 均由 fixture 随机隔离，清理限定本次资源；未使用应用库运行破坏性测试。验收结束后已关闭两次临时预览，确认两个预览索引不存在，并删除本次两个专用 MySQL 测试库。

真实服务集成使用确定性模型、向量和 reranker；实际经过 MySQL → Outbox → Redis → Query Worker → v2 Graph → 审计/来源快照 → MySQL 终态。Mem0 端口为测试替身，核验所有会话仍传递同一 user_id。**这些是协议与集成证据，不是模型质量、真实 Mem0 或 RAG 效果验收。**

## 执行结果

| 检查 | 状态 | 结果 |
| --- | --- | --- |
| 单元、API、会话持久化与 Worker 针对性回归 | PASS | 1190 passed，5 skipped；跳过是 SQLite 不支持的 MySQL 并发场景，对应 MySQL 参数项已执行 |
| 新多轮聊天及旧 Query 真实基础设施链路 | PASS | 5 passed；三轮完成后新 client 读取 6 条消息基础，第 4 问仍用原 thread；每轮 v2 上下文分别 0/2/4/6 条 |
| 会话单活动 Run、提交幂等、并发创建与软删除 | PASS | SQLite 和隔离 MySQL 测试；同键同问复用，冲突不创建额外 Run，历史按微秒时间+ID 稳定排序 |
| 取消与过期 Worker 写入 | PASS | 排队取消、重复取消、租约 owner/generation 失效不更新答案、来源或会话活动时间 |
| 来源快照与授权 | PASS | 引用顺序、256 KiB 实际序列化预算、中文/emoji、伪造 AST、跨用户、跨轮、更新版本与删除均覆盖 |
| 阶段与终态公开投影 | PASS | 白名单阶段；中间模型结果不展示；终态事件触发重新 GET 核验；安全拒答保留解释 |
| Chrome 1440×960 / 390×844 | PASS | 12 组真实 DOM/布局检查，见下文及截图 |
| 全仓 `pytest -q`（启用隔离 MySQL） | BASELINE FAIL | 1255 passed、38 skipped、9 failed；同样 9 项在未修改的 4b1983b 复现，详见下文 |
| Ruff（本次全部变更 Python 文件） | PASS | 退出码 0 |
| Mypy（全 `src/agentic_rag`） | BASELINE FAIL | 基线与当前均 10 个错误，位于 ingestion/models.py、retrieval/graph.py、testing/e2e_harness.py；本次新增错误已修复，未改无关模块 |
| `git diff --check` | PASS | 无空白错误 |
| live-provider console smoke | NOT RUN | 已更新 DOM、会话、轮次和来源断言；没有开启付费模型/真实 Mem0 opt-in |
| 应用发布、应用数据库迁移、实体手机软键盘与 OS 中文输入法人工测试 | NOT RUN | 浏览器覆盖真实 composition/key 事件及窄视口，不等同于真机输入法测试 |

回归命令（预先显式配置测试 DSN，不输出凭据）：

```sh
conda run -n agentic-rag python -m pytest tests/unit tests/integration/api \
  tests/integration/persistence/test_chat_sessions.py \
  tests/integration/persistence/test_chat_turns.py \
  tests/integration/persistence/test_chat_sources.py \
  tests/integration/persistence/test_conversation_context.py \
  tests/integration/runtime/test_query_worker.py -q

AGENTIC_RAG_RUN_REAL_QUERY_E2E=1 AGENTIC_RAG_MEM0_ENABLED=false \
  conda run -n agentic-rag python -m pytest \
  tests/integration/runtime/test_chat_pipeline.py \
  tests/integration/runtime/test_query_pipeline.py -q

conda run -n agentic-rag python -m mypy --follow-imports=silent src/agentic_rag
```

本次在忽略的计划工作目录中使用包装脚本设置专用数据库和显式测试 DSN。Mypy 基线从 `git archive 4b1983b src pyproject.toml` 解包至临时目录检查，相同参数的错误列表与当前逐行一致；错误数不能记作类型检查全通过。Alembic 的 path_separator 弃用和 pytest fixture 已导入警告不影响通过项。

### 全仓已存在的失败

全仓运行显式关闭 real-provider、real-query、real-Mem0、backup/restore opt-in，只开启本次隔离 MySQL；结果为 **9 failed / 1255 passed / 38 skipped**，不能记作全仓通过。将未修改的 `4b1983b` 源码、Alembic 与以下测试解包到单独临时目录，使用第二个专用库 `agentic_rag_chat_test_baseline_89a527b6cf55`，执行：

```sh
conda run -n agentic-rag python -m pytest \
  tests/integration/retrieval/test_parent_fetch.py \
  tests/unit/memory/test_service.py \
  tests/unit/observability/test_degradation_events.py -q
```

基线结果 **相同 9 failed / 21 passed**。其中 3 项直接在 `ParentFetchFixture` 上调用不存在的 `.fetch()`；另外 6 项是先运行迁移时 Alembic `fileConfig` 禁用了既有 logger，导致后续 memory/observability 的 caplog 断言为空。这些文件在本次 diff 中没有修改。本次按已确认范围保留问题，未用跳过或放宽断言把全仓结果改为通过；聊天专项按计划顺序执行为绿色。

## 真实浏览器证据

验收脚本：[chat_browser_acceptance.cjs](../../tests/e2e/chat_browser_acceptance.cjs)。使用 Playwright Core 驱动独立无头 Chrome 临时配置，只访问隔离后端；Playwright 安装在临时目录，没有新增产品依赖。可用 `[显式测试 DSN] conda run -n agentic-rag python -m tests.e2e.chat_preview --output /private/tmp/chat-preview-新目录` 启动有 3 秒生成延迟的专用后端。该工具要求 MySQL 库名以 `agentic_rag_chat_test_` 开头，退出时清理 fixture 资源。

```sh
# NODE_PATH 指向临时安装 playwright-core 的 node_modules；不使用真实用户账号。
CHAT_PREVIEW_URL=http://127.0.0.1:8766 \
CHAT_SEEDED_DOCUMENT_ID="$(cat /private/tmp/chat-preview-新目录/seeded-document-id)" \
CHAT_BROWSER_OUTPUT=/private/tmp/chat-browser-results \
node tests/e2e/chat_browser_acceptance.cjs
```

实际通过：三轮消息和按需来源、重命名、切换期间完成、刷新同一 Run、停止、向上阅读/回到最新、composition Enter 防误发/Shift+Enter 换行、延迟来源响应不跨会话、连续 4 次 SSE 失败后 GET 轮询、已展开来源在文档删除后失焦/聚焦重新核验并撤回片段、390px 无横向溢出/隐藏侧栏不可聚焦/Tab 循环/Escape 焦点返回、无未捕获 JavaScript 错误。

| 桌面 | 手机 | 手机会话列表 |
| --- | --- | --- |
| [1440px 截图](assets/chat-20261001/desktop-1440.png) | [390px 截图](assets/chat-20261001/mobile-390.png) | [390px 侧栏](assets/chat-20261001/mobile-sidebar-390.png) |

机器记录：[results.json](assets/chat-20261001/results.json)。截图中的英文回答来自确定性测试替身，不代表真实模型语言表现。

浏览器验收发现并修复了移动端隐藏侧栏仍可聚焦的问题：真实 Chrome 重现失败后，在折叠状态增加 visibility 隔离并复测通过。刷新恢复还补齐了 sessionStorage 默认存储测试（正文/草稿不进入存储）。旧 fixture 按消息下标解析模型输入、没有接入 v2 路由上下文，已改为按角色读取并使用当前组合边界；先重现拒答和只加载一次记忆，再验证通过，未放宽实际 schema 或审计。

## 实施裁决

1. 按明确要求在 main 开发，覆盖默认隔离策略；代价是修改直接位于 main，以小提交和不推送限制影响。
2. 旧 schema 集成 fixture 的异步连接池跨测试事件循环，引发失败；改为每测试创建连接池，代价是更多测试连接，无生产行为变化。
3. 32,000 个中文/emoji 超出 MySQL TEXT 容量，已用真实 MySQL 复现；在尚未发布的 0008 将 question 扩为 MEDIUMTEXT，downgrade 保留更宽类型防止截断。代价是 DDL 维护窗口和降级后保留额外容量。
4. 旧 document API fixture 有同类跨循环连接池问题，同样改为每测试创建；代价是更多测试连接，无生产行为变化。
5. 确定性真实服务 fixture 的消息位置/v1 路由已落后于生产 v2；更新角色解析和能力/历史依赖。代价是测试适配变化，生产路径与审计约束未变。

6. 全仓 9 个已有失败已在开发前提交复现；保留本轮范围，不修改无关的 ParentFetcher 测试 API 和全局 Alembic 日志配置。代价是全仓门禁仍需单独修复，不能声称全仓测试全通过。

7. 审查将 Worker 进程级恢复计数重置排除在判断外；继续遵守用户确认的排除范围。代价是该旧恢复限制保留。
8. 审查没有判断真实 provider/Mem0 质量、生产负载与部署；维持 NOT RUN，确定性测试只证明接口和事务行为。代价是生产质量/发布结论仍需另行验收。
9. 审查将 9 个全仓失败视为既有问题；接受已在未修改基线独立复现的归因，沿用第 6 项裁决。代价是相同的全仓门禁限制保留。

## 最终独立审查与修复

使用全新上下文的 gpt-6-astra 对 `4b1983b..200763a` 进行一次只读整体审查，涵盖计划的五项 Review Focus。结论：无 Critical，3 个 Important，无 Minor。实现者按实际用户影响维持这三个问题的 Important 分级，一次修复，不再分配重复审查。

- **晚到的停止响应中断下一轮观察。** R1 已由 SSE 进入终态并开始 R2 后，R1 的停止 HTTP 响应会误 abort R2。观察连接现在绑定 Run ID，只停止对应 Run；取消按钮的忙碌状态同样按 Run 管理，切换会话后不会沿用旧禁用状态。
- **分页请求期间切换会话导致按钮永久禁用。** 历史加载状态现在绑定视图 generation，每次切换重新计算按钮状态；旧请求的 finally 不解锁新视图正在加载的按钮。
- **被删除会话的创建键永久重试。** 只有明确 `410 SESSION_GONE` 才清除已确认删除的创建键并持久化；不自动创建替代会话，下一次明确点击“新建”才生成新键。网络未知结果继续保留原键。

`test_chat_controller_lifecycle` 的 lateCancel、crossSessionCancel、historySwitch、deletedCreation 四个场景均先实际失败后通过。lateCancel 进一步核验 R2 最终回答仍展示且输入恢复，不只检查 AbortSignal。前端 51 项测试通过；修复后使用新隔离后端复跑 Chrome 12 组检查全部通过，并重新检查截图。完整聊天专项 1190 passed、5 intentional SQLite skips。延期 Minor：**无**。

已知 Worker 进程级恢复计数重置问题按确认范围另行处理。本轮不增加认证/RBAC，不迁入旧独立 Run；应用上线按[运维说明](../local-operations.md#持久化聊天升级与回滚2026-10-01)执行。
