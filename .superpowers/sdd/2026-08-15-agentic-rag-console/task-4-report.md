# Task 4 报告：真实 Graph/API、Mem0 与控制台验收

## 状态

已完成代码、离线回归与验收门禁实现，并闭合独立复审指出的 stream 隔离、fail-closed teardown 与 opt-in 配置分类问题。真实 provider 验收已按设计执行 opt-in 预检，但当前环境没有显式的隔离服务变量，因此测试清晰跳过；未把该状态记为通过，也没有生成虚假的 PASS 报告。

## 交付内容

- 新增 `tests/e2e/test_query_console_real_services.py`：复用同一个 production `AppContainer` 给 `create_app(settings, container=container)` 与 `QueryWorker`，通过真实 `HttpQueryClient`/API 路径验证页面、SSE、当前 snapshot、已审计回答、服务端 Evidence Parent、EvalRunner provenance、引用覆盖、泄漏门禁、恢复和备份门禁。
- `real_query_runtime` 只在显式设置 `AGENTIC_RAG_RUN_REAL_QUERY_PROVIDER_E2E=1`、三个 `AGENTIC_RAG_TEST_*` 本地服务变量以及 DeepSeek/Qwen/Mem0 配置后运行。缺少变量会 `skip`；一旦已配置的 provider、Mem0、MySQL、Redis、ES、Worker 或模型调用失败，则测试失败而不是退回 fixture 成功。
- 真实验收脚本使用独立 user、ES generation、Mem0 collection、artifact/checkpoint 根目录，并在 `finally` 中按所属 Run/Outbox ID 清理 MySQL 行、Redis stream/dedupe 字段、查询 index 和 Mem0 collection。它先在临时目录写 EvalRunner 中间结果，只有所有 console 门禁通过后才在指定 `--output` 写最终 `summary.json`。
- API 侧的 `container.run_manager` 使用带私有 stream 的 `TransactionalRunRepository`，因此本次 acceptance 的 Run 与 Outbox 在**同一 MySQL 事务**写入该私有 stream；不是等 dispatcher publish 时再映射。`TransactionalQueryOutboxAdapter` 的 list/claim/mark/retry 同时带 `user_id + stream_name` 条件，且 `agenticrag:e2e:*` 私有 stream 不允许省略 user scope，普通 worker 的全局 stream 无法 claim acceptance 行。
- acceptance teardown 会累计 Worker、HTTP client、API lifespan、checkpoint、dependencies、MySQL/Redis/ES 清理、container 和 temporary root 的全部异常；`CancelledError` 等控制流也会在其余边界清理后原样重抛。任一 teardown 失败都发生在 summary promotion 前，旧 `summary.json` 已移为 `summary.stale-*`，不会保留给 `verify_acceptance` 读取的 PASS。
- 缺少 opt-in 服务变量或 provider credentials 会明确 skip；一旦显式配置了 placeholder/disabled provider 或非 loopback 的测试服务地址，fixture/acceptance 会 fail，而不是把无效配置伪装为 skip。
- 最终 summary 新增/严格校验：`console_page_status`、`sse_last_event_id`、`console_run_snapshot_id`、`memory_provider_available`、`memory_boundary`、`degradation_events`，并保留 `client_provenance=real_query_api`、`runtime_config_snapshot_id`、引用、泄漏、审计、恢复和备份字段。
- 健康路径以生产 `MemoryService` 完成真实的 scoped Mem0 read/write，再通过 `/v1/memories` 验证写入可见；Mem0 不可用、模型结构化输出错误或 provider 异常不会被伪造为健康成功。

## 降级与 Mem0 生产修复

- `MemoryServiceImpl.list` 和显式 `delete` 不再把 Mem0 的 `OSError`、`TimeoutError`、`ConnectionError` 吞成空列表或 HTTP 204。它们记录已有的 `memory_provider_degraded` 结构化降级日志/事件并抛出安全的 `OSError` 子类，`/v1/memories` 因而返回既有的 `503 MEMORY_UNAVAILABLE`。Query-time memory load/capture 仍保持降级但不中断有效答案的既有语义。
- SSE 只从已脱敏的持久化事件 artifact 中读取 `attempt`、`component`、`reason`、`outcome`、`retryable` 五个 allowlist 字段；不会转发 prompt、provider response、raw payload 或 tool input。控制台以同一 allowlist 再次验证后才显示这些字段。
- 真实验收在已完成的 Run 上发出一个受控 `CIRCUIT_OPEN` 降级：这会产生结构化 warning 与 durable event，并由 SSE/控制台断言其安全字段。该做法不注入 query provider client，也不会掩盖真实 provider 失败。

## 验证摘要

- RED：新增 teardown cancellation 回归最初从 `_record_teardown_error` 直接抛出 `CancelledError`，导致后续 client cleanup 不执行；私有 stream 缺 user scope 的 worker adapter 回归最初也未报错；显式非 loopback 服务配置最初被 skip。三者均已最小修复并转绿。
- 相关回归：`conda run -n agentic-rag pytest --import-mode=importlib tests/unit/test_real_query_acceptance.py tests/unit/test_query_worker_scope.py tests/unit/test_real_provider_fixture_config.py tests/unit/test_bootstrap_close.py tests/unit/testing/test_isolated_query_broker.py tests/e2e/test_query_console_real_services.py tests/e2e/test_query_pipeline_real_services.py tests/integration/api/test_query_runs.py tests/integration/api/test_console.py tests/unit/memory/test_service.py tests/unit/persistence/test_repository_contracts.py tests/integration/runtime/test_query_worker.py -q`：`107 passed, 3 skipped`。
- `conda run -n agentic-rag pytest --import-mode=importlib -q`：`631 passed, 47 skipped`。仅有既有的 Docling/Torch deprecation warnings。
- `conda run -n agentic-rag ruff check .`：通过。
- `MYPYPATH=src conda run -n agentic-rag mypy --explicit-package-bases ...`（覆盖 Task4 production、fixture、script 与新增回归文件）：通过。
- `git diff --check` 与 `node --check src/agentic_rag/api/static/app.js`：通过。

## 真实服务预检

执行：

```bash
set -a; source .env.local; set +a
AGENTIC_RAG_RUN_REAL_QUERY_PROVIDER_E2E=1 \
conda run -n agentic-rag pytest --import-mode=importlib \
  tests/e2e/test_query_console_real_services.py -q -rs
```

结果：清晰跳过，原因是缺少 `AGENTIC_RAG_TEST_MYSQL_DSN`、`AGENTIC_RAG_TEST_REDIS_DSN`、`AGENTIC_RAG_TEST_ELASTICSEARCH_URL`。没有运行真实 provider，也没有在 `var/artifacts/evals/real-api-current` 创建 PASS summary。

## Concerns

- 要完成真实 provider 放行，操作者需要提供以上三个指向 loopback 测试服务的显式变量，并保留有效 DeepSeek、Qwen 与 Mem0 配置；随后运行 brief 中的 `run_real_query_acceptance.py` 和 opt-in E2E 命令。已配置后任一 provider outage 或 schema 失败都会 fail closed。
- 对 `tests/unit/memory/test_service.py` 运行独立 mypy 仍报告 12 个既有 fake/模型构造注解问题（行 144–299，早于本任务新增的 outage assertions）；本任务新增/修改的 production、fixture、script 和 gate 文件的 scoped mypy 已通过。

## Follow-up review round 1

- `run_real_query_acceptance.py` 不再把 MySQL、Redis、查询 index 与 Mem0 index 清理串在单个 coroutine 中。它们现在是四个独立 guarded teardown boundary；MySQL 失败后仍会尝试删除 private Redis keys、当前 generation child index 和 Mem0 collection，并把所有失败累计到 fail-closed gate。
- 真实 E2E 明确断言 `answer.evidence_parent_ids` 包含 fixture seed 生成的 server Parent ID，而不只检查字段非空。`RealQueryRuntime` 只暴露该 server-created ID 给该断言。
- `RealQueryFixture` 与 provider fixture 都不再吞 Elasticsearch index 删除错误；cleanup failure 会传播，避免测试把未清理的服务状态误判为成功。
- 新增 RED/GREEN 回归分别覆盖：MySQL cleanup 失败后其他 durable backend 仍执行、两个 fixture 的 ES cleanup error 传播、以及 seeded Parent ID 的 API evidence 断言。
- 验证：相关 `111 passed, 3 skipped`；全量 `635 passed, 47 skipped`；`ruff check .`、Task4 scoped `mypy`、`git diff --check` 均通过。真实 E2E 仍因缺少 `AGENTIC_RAG_TEST_MYSQL_DSN`、`AGENTIC_RAG_TEST_REDIS_DSN`、`AGENTIC_RAG_TEST_ELASTICSEARCH_URL` 明确 skip，未创建 PASS summary。
