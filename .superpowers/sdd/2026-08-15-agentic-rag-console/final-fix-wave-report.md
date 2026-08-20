# 最终 Fix Wave 报告

基线：`a3acd71`。本波次只处理最终 review 的四项 Important，没有重做 Task 1–5。

## 修复结果

1. 真实验收恢复/备份门禁
   - `run_real_query_acceptance.py` 与真实 provider fixture 不再调用本地 `run_recovery_drill()` / `run_backup_restore_drill()` 并把布尔结果写成真实 PASS。
   - 恢复证据来自本次隔离 API、Query Worker、Run、snapshot 与私有 Redis Stream：Worker 停止后通过 API 创建 Run，重启同一生产组合 Worker，注入重复投递，并验证 audited 终态、私有 group ACK 及唯一终态事件。
   - 备份/恢复覆盖本次隔离 MySQL database、Redis prefix、Elasticsearch generation、Artifact root 与 Query SQLite checkpoint，并恢复到不同的 database/prefix/generation 后逐项查询验证。
   - 真实备份必须显式设置 `AGENTIC_RAG_RUN_REAL_BACKUP_RESTORE=1` 与 loopback `AGENTIC_RAG_TEST_MYSQL_ADMIN_DSN`；缺少配置时 fixture 明确 skip/unavailable，不产生 PASS；配置存在而建库、provider、backup、restore、验证或清理失败时直接失败。
   - `console_acceptance_passed()` 现在要求结构化 recovery/backup evidence；仅有两个旧布尔值不能通过。

2. 共享 fixture 隔离
   - `real_query_fixture` 改用 `IsolatedQueryBroker` 的私有 stream/group，并让 API Outbox 在创建 Run 的同一事务中写入私有 stream。
   - Outbox dispatcher 同时受 `user_id` 与 `stream_name` 约束；MySQL 清理只删除 fixture user/run IDs 对应的行，Outbox 还要求私有 stream；Redis 只删除 broker 明确列出的私有 keys。
   - 回归覆盖并行 fixture stream/group/cleanup key 不相交，以及 foreign MySQL/Redis 数据不会落入清理谓词。

3. 公共 Answer 边界
   - 新增 `PublicAnswer` / `PublicAnswerSegment`，均为 `extra="forbid"` 的严格公共模型；公共投影只允许 audited segments、evidence IDs/parent IDs、route、snapshot、client provenance、citation coverage 与受控终止状态。
   - Worker 持久化前使用真实 graph evidence、route 与 claimed snapshot 重建投影；completed 回答必须 audited 且含 segment。API 读取历史 Run 时再次投影，并以 Run 的 snapshot 覆盖存储值。
   - Console 只渲染明确的 segment text、evidence、audit 与 provenance 字段，不再序列化整个 answer 或兼容任意别名。
   - API/Worker/UI 恶意注入测试确认 `prompt`、`provider_response`、`tool_input`、`raw` 与未知嵌套字段均不泄漏。

4. Research checkpoint 脱敏
   - schema/provider/retrieval/delegate/calculator 失败只写受控 `reason` / `error_code`、有限 `retryable` 与 `attempt`；不再把 `str(error)` 写入 QueryState。
   - provider schema 与异常分别映射到 `research_action_invalid`、`model_unavailable`；retrieval 与 delegate 分别映射到 `retrieval_unavailable`、`subagent_unavailable`。
   - 敏感异常回归检查 graph State、durable event/Artifact、Run API 与 SSE，确认 Authorization、URL 与 provider response 原文均不存在。

## TDD 证据

- Answer/research 初始 RED：`12 failed`；实现后相关集合 GREEN：`66 passed`。
- Fixture 隔离初始 RED：构造器缺少私有 broker；实现后相关集合 GREEN：`16 passed`。
- 真实门禁初始 RED：裸 recovery/backup 布尔仍可通过、broker 缺少 evidence 属性；实现后 focused 总计：`85 passed`。

## 最终验证

```text
conda run -n agentic-rag pytest --import-mode=importlib -q
654 passed, 47 skipped, 18 warnings in 24.65s

conda run -n agentic-rag ruff check src tests evals scripts
All checks passed!

MYPYPATH=src conda run -n agentic-rag mypy --explicit-package-bases src evals scripts
Success: no issues found in 115 source files

conda run -n agentic-rag python -m evals.validate_datasets evals/datasets
baseline: 24 cases
ingestion_fidelity: 13 cases
security: 12 cases
dataset validation passed

git diff --check
无输出（通过）
```

真实 provider E2E 在当前环境明确跳过：未设置 `AGENTIC_RAG_RUN_REAL_QUERY_PROVIDER_E2E=1`。没有生成真实 PASS summary，也没有用本地 drill 替代。剩余 concern 仅为需要在具备显式隔离 MySQL/Redis/Elasticsearch、provider/Mem0 凭据及 MySQL CLI 的批准环境中执行一次真实门禁；任何配置齐全后的服务或恢复失败都会 fail closed。
