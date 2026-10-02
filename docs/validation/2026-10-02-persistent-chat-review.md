# 持久化聊天二次检查

日期：2026-10-02。检查起点：`a6d500f`。按用户要求继续在 `main` 检查并修复，未推送、部署或迁移应用数据库。原验收记录见 [2026-10-01](2026-10-01-persistent-chat-sessions.md)。

## 修复内容

| 场景 | 根因和修复 |
| --- | --- |
| 恢复已删除会话的创建请求 | 控制器读取 `error.code`，真实适配器提供 `error.errorCode`。改用真实 `Response` → `HttpError` 测试复现并修正；明确 410 后清键，下次用户点击才新建。原测试替身字段不符合实际契约，已替换。 |
| 旧提交查证覆盖下一轮 | 查证成功/失败响应没有验证 pending 请求归属。现在只处理仍属于当前 pending 的结果；观察连接同时校验活动 Run，旧回调不能停止新 Run。 |
| 发问后切走再返回 | 旧视图的 POST 先清除 pending，但没有把已受理 Run 投影到重新打开的视图。统一受理入口按会话身份写入当前视图，并防止更早会话快照清除新 Run。 |
| 聚焦恢复与提交交错 | 旧 GET 能覆盖刚受理的 Run；连续 focus/visibility 请求还会保存暂时的 `loaded=false`。增加 Run 修订计数，合并同视图刷新，独立维护刷新忙碌状态，不覆盖稳定的加载状态。 |
| 删除完成后抢走导航 | 删除后的列表请求结束时没有检查视图代次。现在立即清空已删除视图，只在用户尚未切换时自动选择下一会话；确定会话失效时清空消息、来源和恢复标识并禁用发送。 |
| 历史补齐后乱序或阅读位置跳动 | 分页无条件前插，普通同步无条件后插。统一按状态层的时间/ID 顺序插入，保留既有 DOM、展开来源和可视消息锚点。Run 观察接口补齐微秒级 `created_at`，保证通过 SESSION_BUSY 接管其他标签页任务时也能排序。 |
| 停止响应擦除最终答案 | 取消接口把新状态套在取消前的旧 Run 上，可能返回 `completed` 但 `answer=null`。取消后重新读取持久化 Run 并再次检查访问权限。HTTP 回归验证该响应与最终 GET 一致。 |
| 工具抽屉异常体验 | readiness 的正常 503 过去被当作无法读取状态；现在展示“服务尚未就绪”和依赖状态。手机抽屉关闭时将焦点返回可见的会话列表按钮。 |

新增回归均有实际失败复现。包括旧创建测试的真实传输复现、控制器时序测试、取消 HTTP 契约测试和真实 Chrome 焦点失败；没有用放宽断言或跳过新问题来通过检查。

## 独立复核

按 requesting-code-review / receiving-code-review 工作流进行一次全新上下文的只读复核。审查给出 4 个 Important，无 Critical、无 Minor：连续聚焦禁用发送、重新打开丢失受理结果、普通同步消息乱序、取消响应覆盖最终答案。四项均由执行者独立复现、修复并验证；未安排重复整体审查。

接受的审查边界：既有 Worker 进程恢复计数问题、认证/RBAC 扩展不属于已确认范围；真实模型/Mem0 质量、生产负载与发布没有由确定性测试替代。9 项全仓失败沿用前次在未修改基线的复现证据。真机软键盘和系统中文输入法仍需人工走查，浏览器已覆盖 composition 事件和窄视口。

## 验证结果

- 最终聊天专项：**1203 passed / 5 skipped**，覆盖单元、API、SQLite/MySQL 会话/轮次/来源持久化、上下文和 Worker。
- 实际 MySQL → Outbox → Redis → Worker → v2 Graph → 审计/来源 → MySQL：**5 passed**。模型、向量和 reranker 使用确定性替身，没有调用付费模型或真实 Mem0。
- 全仓（显式隔离 MySQL，沿用前次 opt-in 配置）：**9 failed / 1267 passed / 38 skipped**。失败名称与前次完全一致：3 项 ParentFetchFixture API 错用，6 项迁移禁用 logger 后的 caplog 失败。该次运行后增加了 Run 创建时间 HTTP 契约用例；最终专项再次覆盖这项补充。
- 全 src Mypy：仍是原有 **10 errors / 3 files**，位于 ingestion/models.py、retrieval/graph.py、testing/e2e_harness.py；本次改动文件无新增类型错误。
- 变更 Python 文件 Ruff、变更 JS/CJS Prettier、`git diff --check`：通过。
- Chrome 桌面 1440×960、手机 390×844：原有 **12 组通过**；本轮新增 **5 组通过**（总计 17 组），无未捕获页面 JavaScript 错误。

一次额外的全基础设施探测复用了已有聊天专项数据的测试库，得到 `12 failed / 1274 passed / 23 skipped`：除上述 9 项外，入库/发布 fixture 出现 3 项跨测试数据冲突（例如同一测试用户文档数为 58，断言期待 4；reconciler 扫描到其他测试遗留记录）。这不是绿色全仓证据，也未修改或放宽这几个测试。正式对照运行新建独立数据库，仅启用前次相同的 MySQL opt-in；全基础设施入库验收仍需要其自己的独占干净环境。

所有数据库均为本轮创建的 `agentic_rag_chat_test_*`，Redis stream / Elasticsearch index / 用户 / checkpoint 由 fixture 隔离；未修改应用数据库。清理结果：两座本轮专用数据库均已删除；三次预览已停止，逐一确认三座预览索引不存在，fixture 执行了隔离 Redis 清理。用户原有 `tmp/` 保留。

## 浏览器证据与复跑

原流程脚本：[chat_browser_acceptance.cjs](../../tests/e2e/chat_browser_acceptance.cjs)。新增脚本：[chat_review_browser.cjs](../../tests/e2e/chat_review_browser.cjs)，覆盖缓存历史缺口补齐与滚动锚点、真实 410 创建恢复、503 系统状态、实际 SESSION_BUSY 跨标签页补齐，以及手机工具抽屉焦点恢复。

按前次记录启动独立 [chat_preview.py](../../tests/e2e/chat_preview.py)，为脚本配置 `NODE_PATH`（临时 playwright-core）、`CHAT_PREVIEW_URL`、`CHAT_BROWSER_OUTPUT`；原脚本额外要求 fixture 的 `CHAT_SEEDED_DOCUMENT_ID`。原脚本会删除这个专用文档，因此两组脚本分别使用新 fixture。Playwright/Prettier 仅临时安装，没有新增产品依赖。

[桌面截图](assets/chat-20261002/desktop-1440.png) · [手机截图](assets/chat-20261002/mobile-390.png) · [手机侧栏](assets/chat-20261002/mobile-sidebar-390.png) · [原流程结果](assets/chat-20261002/results.json) · [新增检查结果](assets/chat-20261002/review-results.json)

## 用户走查清单

1. 同一会话连续提问至少三轮，检查上下文、滚动展示、底部输入和来源按需展开。
2. 回答处理中切到另一会话再返回，刷新页面，确认不重复提交且能继续展示最终结果。
3. 向上阅读、加载更早消息、返回最新；展开来源后切换或删除对应文档，确认按最新权限重新展示。
4. 停止后继续提问；同一会话在两个标签页操作，检查忙碌提示、草稿保留和消息顺序。
5. 重命名、删除、新建会话；手机宽度检查侧栏、工具抽屉、焦点以及真机输入法和软键盘。

本轮没有升级或重启应用服务。实际环境走查前按 [持久化聊天升级说明](../local-operations.md#持久化聊天升级与回滚2026-10-01) 应用 0008 并启动当前 main 版本；测试截图中的英文回答来自确定性模型。
