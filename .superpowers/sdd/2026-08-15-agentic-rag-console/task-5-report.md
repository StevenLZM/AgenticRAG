# Task 5 报告：中文运行手册、路线图与发布状态收敛

## 交付内容

- 将 `docs/local-operations.md` 的运行说明、控制台操作、故障解释和验收步骤统一为中文；命令、环境变量、API 路径、事件名、状态枚举和文件路径保持可复制的英文标识。
- 明确区分 Query Outbox 与 Query Worker：前者在 MySQL 同事务内保存 Run 和待投递意图，后者负责 Redis 领取、租约、心跳、恢复、重试/DLQ、QueryGraph 执行和终态 ACK；文档说明该分离避免丢失窗口和重复 Run。
- 补充 Agentic RAG 控制台启动、SSE 重连、服务端白名单展示和日志排障说明，覆盖 `memory_provider_degraded`、`MODEL_REPAIR_EXHAUSTED`、`CIRCUIT_OPEN`、`WORKER_DLQ` 及安全字段 `component`、`reason`、`outcome`、`attempt`、`retryable`。
- 记录 `SubagentDispatcher` 已接入、Todo 初始创建/追加、持久化全局 `research_attempt_count`、Mem0 默认启用，以及真实 Graph/API + 当前 snapshot + client provenance + Mem0 + 恢复/备份门禁。
- 将进度快照和总体路线图改为当前 `main`/`fb1a5b5` 状态，移除旧实现 worktree、未合并分支和待实现控制台的陈旧描述；生产鉴权/RBAC 和 reranker 分数标定保留为独立上线审批事项。
- 新增 `tests/unit/test_documentation_contracts.py`，只验证稳定的业务术语、英文接口标识和上线门禁，不检查自然语言比例。

## TDD 证据

先新增文档契约测试并运行：初始结果为 `2 failed`。失败原因是运行手册缺少控制台与 Subagent/Todo/全局研究预算契约，开发快照仍使用旧的 worktree/未合并叙述。改写文档后同一测试转绿。

## 验证

```text
conda run -n agentic-rag pytest --import-mode=importlib \
  tests/unit/test_documentation_contracts.py -q
2 passed

conda run -n agentic-rag ruff check src tests evals scripts
All checks passed!

MYPYPATH=src conda run -n agentic-rag mypy --explicit-package-bases src evals scripts
Success: no issues found in 113 source files

git diff --check
无输出（通过）
```

## 修复轮 2：控制台门禁归属更正

- `run_real_query_acceptance.py` 的顺序为先调用 `EvalRunner.run()`，再单独执行 `console_acceptance_passed` 独立控制台门禁；它在 EvalRunner 后执行，不属于 EvalRunner 本身。
- 三份运维/状态文档统一为“`console_acceptance_passed` 控制台门禁在 EvalRunner 后执行”，并保留通用 `scripts/verify_acceptance.py` 的职责边界。
- 文档契约改为要求 `console_acceptance_passed`、`在 EvalRunner 后执行`，并拒绝把控制台门禁归属给 EvalRunner 的陈旧表述。

## 注意事项

- 真实 Graph/API 与 Mem0 验收仍是显式 opt-in；缺少隔离服务变量时测试必须清晰 skip，不能生成 PASS summary。
- 本任务只收敛文档与文档契约测试，不修改页面或运行时代码。

## 修复轮 1：运行拓扑与验收职责更正

- 本地运行手册和开发进度明确 `run_api.py`、`run_query_worker.py`、`run_ingestion_worker.py` 均为阻塞进程，必须在三个独立终端或受监督后台进程运行，避免 API 阻塞后 Worker 永远不启动。
- 控制台实际调用 `POST /v1/query` 同步 wrapper；`POST /v1/query-runs` 是同一持久化 Run 的异步 API。文档保留 SSE 路径，并明确页面不使用异步创建路径替代同步调用。
- 真实接受脚本先运行 `EvalRunner.run()`，再由 `console_acceptance_passed` 独立控制台门禁在 EvalRunner 后执行，负责当前 snapshot、精确 `client_provenance=real_query_api`、Mem0 读写、控制台和 SSE 的专用验证。`scripts/verify_acceptance.py` 仅检查其已有通用 summary 字段，不能单独声称校验这些专用事实。
- `research_attempt_count` 的持久化范围更正为 QueryState/SQLite checkpoint；它不是 QueryRun DB 字段。

### 修复轮 TDD 与验证

- 新增文档回归后先运行：`2 passed, 1 failed`；失败原因是旧文档没有“三个独立终端”说明。
- 修正三份文档后，最终验证为：

```text
conda run -n agentic-rag pytest --import-mode=importlib \
  tests/unit/test_documentation_contracts.py -q
3 passed

conda run -n agentic-rag ruff check src tests evals scripts
All checks passed!

MYPYPATH=src conda run -n agentic-rag mypy --explicit-package-bases src evals scripts
Success: no issues found in 113 source files

git diff --check
无输出（通过）
```
