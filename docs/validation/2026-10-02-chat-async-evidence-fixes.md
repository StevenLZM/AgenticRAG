# 聊天异步状态与研究证据预算修复

日期：2026-10-02。开发分支：`main`。基线：`a831020`。

## 修复行为

| 问题 | 处理与保护 |
| --- | --- |
| 收到检索阶段仍显示等待中 | `RUN_STARTED` 和有效阶段事件推进 `queued → running`；阶段变更计入状态修订。保留事件去重、取消状态及终态保护，迟到的 queued 快照不能倒退状态。 |
| 第一问结束时发现第二问，却未监听 | 历史刷新、会话恢复、提交确认及前台恢复统一协调活动任务监听。新任务使用自身游标；旧任务回调不能停止新监听。 |
| A→B→A 后迟到拒绝残留提交状态 | 响应归属仍由 session/request ID 判定，界面更新使用同一会话的当前视图。拒绝清理 pending，未知结果保留幂等键；SESSION_BUSY 接续活动任务，不覆盖更新的任务，也不重新激活已知终态任务。 |
| 合法累计证据超预算提前拒答 | 直接检索和委派检索先校验输入包与原始来源，再以全部原始批次统一筛选、去重、裁剪。最终包校验通过后才进入检查点和后续模型上下文；Todo 引用仍按最终保留证据核对。 |

计数继续采用包含包装文本的 Unicode 字符数；最终预算未提高。索引版本、manifest、输入包预算和缺失批次等校验保留。

独立复核补充发现：全局裁剪后再次检索同一原文，裁剪长度差异仍可能被旧比较逻辑误判为损坏。现以每个输入包自己的原始批次核对 parent/version、全文、文档身份、标题路径和 child locator；只有双方都有一致原始来源时才允许不同裁剪。缺少原始批次的旧检查点继续严格比较内容。

旧设计的合并预算约定已同步更新：[研究链路设计](../superpowers/specs/2026-08-31-fast-rag-research-rebuild-repair-design.md)。

## 回归证据

- 最初新增的 7 个前端时序用例、3 个累计证据用例在原实现上全部出现预期失败；5 个损坏证据保护用例当时通过。
- 额外复现并修复：迟到 busy 响应重新激活已完成任务；2000/12000 预算下裁剪后重复检索误拒答；同一 parent/version 的原始文档身份冲突。
- 最终默认全仓：**1251 passed / 89 skipped / 20 warnings**。显式移除了真实基础设施、真实提供商及外部评估的测试启用变量；跳过项不代表已经验证真实 MySQL/Redis/Elasticsearch 或模型链路。警告来自现有 pytest fixture 导入及解析依赖。
- 最终研究循环与聊天客户端专项：**63 passed**。另有独立复核的 12 项裁剪与来源完整性测试全部通过，复核未发现剩余可操作问题。
- 独立 Chrome：**3 组通过，0 个未捕获页面异常**。运行生产静态资源、控制器、HTTP/SSE 传输；测试自带隔离的内存 HTTP 服务，控制接口时序，不连接应用数据库或付费模型。
- 变更 Python 文件 Ruff、变更 JS/CJS Prettier、`git diff --check` 通过。
- 后续按用户要求修复 `retrieval/graph.py` 两项既有 Mypy 错误：使用明确的阶段 Literal 类型及固定字段访问，保留原有列表默认值、顺序和观测字段。检索图与研究循环联合 Mypy 检查通过（2 个文件，0 错误）；检索与研究循环相关测试 **87 passed / 5 skipped**，Ruff 通过。

## 浏览器场景与复跑

[浏览器回归脚本](../../tests/e2e/chat_async_recovery_browser.cjs) 创建随机本地端口和独立 Chrome 配置，完成后关闭浏览器、SSE 和服务。

1. 首次 HTTP 返回 queued，保持 SSE 开启；收到开始与检索事件后依次显示处理中、检索中，无需刷新。
2. A 的第一问终态历史请求挂起，B 提交第二问；放行 A 的历史请求后，A 接续第二问的研究、审核和最终答案，并恢复发送。
3. A 提交后切到 B 会话再返回；另一个标签页启动第三问后，原提交才返回 SESSION_BUSY。A 清理 pending、监听第三问，最终保留草稿并恢复发送，没有重复发问。

专项测试：

```sh
python -m pytest tests/integration/api/test_chat_client.py tests/unit/query/test_research_loop.py -q
```

浏览器脚本需要可用的 `playwright-core` 和 Chrome；通过 `NODE_PATH` 指定临时工具依赖目录即可，不增加产品依赖。`CHAT_CHROME_PATH` 可覆盖 Chrome 路径；`CHAT_BROWSER_OUTPUT` 可选，用于保存截图和结果 JSON。

```sh
node tests/e2e/chat_async_recovery_browser.cjs
```

本轮未推送或重启应用服务，未更改应用数据或 schema。后端研究修复需查询 Worker 重新加载代码后生效。
