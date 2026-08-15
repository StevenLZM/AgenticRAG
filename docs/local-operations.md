# 本地运行手册

以下流程仅适用于本地 V1 部署。每次执行命令前先加载本地配置：

```sh
set -a; source .env.local; set +a
```

## 启动与升级

按以下顺序启动依赖：Elasticsearch、MySQL、Redis。确认三者都指向本地端点后，再按以下顺序执行迁移并启动进程：

```sh
conda run -n agentic-rag python scripts/check_local_dependencies.py
conda run -n agentic-rag alembic upgrade head
conda run -n agentic-rag python scripts/run_api.py --grace-seconds 30
conda run -n agentic-rag python scripts/run_query_worker.py
conda run -n agentic-rag python scripts/run_ingestion_worker.py
```

API 启动后可访问 `/health/live`；只有在所有依赖都报告 `available` 后，才使用 `/health/ready`。升级时先停止 Worker，等待当前图节点到达 SQLite checkpoint，再执行迁移，并按相同顺序重新启动。

## Query Outbox 与 Query Worker

`POST /v1/query-runs` 会在同一个 MySQL 事务中写入持久化 Run 和
`query_run` Outbox 行。Query Worker 只把 `query_run` 行分发到 Query Redis
Stream，通过租约领取/重新领取消息，在图运行期间发送心跳，并且只在审计后的答案（或终态业务拒答）持久化之后 ACK。Redis 发布失败会保留为可重试状态；连续失败后进入有界重试/DLQ 流程。这样可以避免请求已提交但任务在 MySQL 与 Redis 之间丢失，并允许重启后的 Worker 继续处理而不创建第二个 Run。

所有降级边界都会产生可观察信号。进程日志会写入有界警告，事件仓储会写入安全的持久化事件，例如
`RETRIEVAL_DEGRADED`、`CIRCUIT_OPEN`、`COMPONENT_DEGRADED`、`OUTBOX_RETRY` 或
`WORKER_DLQ`；事件只包含白名单中的 component/reason/outcome/attempt 字段。SSE 会暴露相同的运行事件名称，但始终脱敏原始 prompt、记忆文本、隐藏推理、工具载荷和服务提供方响应。熔断打开或服务降级时，API 也会明确保留 `retryable`/`degraded_components` 字段。

## DeepSeek 结构化调用与故障诊断

`ModelGateway` 是唯一的 DeepSeek 重试和 schema repair 边界。`AGENTIC_RAG_DEEPSEEK_PROTOCOL=auto` 时，如果客户端同时暴露两种接口，优先使用 Chat Completions；只有明确设置 `responses` 才使用 Responses API。结构化调用会请求 JSON object，但最终仍必须通过应用的 Pydantic schema；Markdown JSON 围栏只会被移除，不会放宽字段、枚举、证据 ID 或非空约束。

常见信号含义如下：

- `provider_outage`：超时、连接错误、429 或 5xx，网关按上限重试。
- `model_unavailable`：瞬态故障耗尽重试次数，查询进入明确降级/拒答路径。
- `protocol_error`：选择的 Chat/Responses 接口不可用或协议参数不被客户端支持，不会伪装成网络重试。
- `model_schema_invalid`：供应商返回了内容，但一次 repair 后仍不符合节点 schema；系统 fail-closed。
- `circuit_open`：连续失败触发熔断，后续调用在冷却窗口内快速失败。

模型诊断事件只保留 schema 名、协议、请求/实际模型、attempt、错误类型、输出长度和输出 SHA-256，不保存 prompt、隐藏推理、原始工具载荷或原始模型响应。排查时先按 `snapshot_id`、`event_type` 和 `reason` 聚合，不要把 repair 内容复制到日志或工单。

## Mem0 长期记忆

Mem0 默认启用。请在 `agentic-rag` Conda 环境中安装 `mem0ai==2.0.12`。应用负责维护用户命名空间，调用 Mem0 时使用 `infer=False`，并在写入持久化事实前先执行轻量模型抽取。Mem0 embedding 优先读取专用变量；没有专用变量时自动复用 Qwen embedding 的 URL 和密钥：

```dotenv
AGENTIC_RAG_MEM0_ENABLED=1
AGENTIC_RAG_MEM0_COLLECTION=agent_memories_v1
AGENTIC_RAG_MEM0_EMBEDDING_BASE_URL=https://<qwen-endpoint>/v1
AGENTIC_RAG_MEM0_EMBEDDING_API_KEY=<qwen-key>
AGENTIC_RAG_MEM0_EMBEDDING_MODEL=text-embedding-v3
AGENTIC_RAG_MEM0_ELASTICSEARCH_API_KEY=<es-api-key>
AGENTIC_RAG_MEM0_HISTORY_DB_PATH=var/mem0/history.db
```

本地 Elasticsearch（`localhost`、`127.0.0.1` 或 `::1`）可以不设置认证；远程 Elasticsearch 必须设置 API key 或用户名/密码。显式的 `AGENTIC_RAG_MEM0_EMBEDDING_BASE_URL` 和 `AGENTIC_RAG_MEM0_EMBEDDING_API_KEY` 会覆盖 Qwen fallback。若要临时关闭 Mem0 进行故障隔离，可设置 `AGENTIC_RAG_MEM0_ENABLED=0`；这会把记忆标记为 disabled，而不是伪装成可用。

如果要使用 Mem0 自己管理的 LLM（通常不需要，因为抽取由应用的 ModelGateway 完成），再设置 `AGENTIC_RAG_MEM0_LLM_ENABLED=1`，并同时设置 `AGENTIC_RAG_MEM0_LLM_MODEL`、`AGENTIC_RAG_MEM0_LLM_BASE_URL` 和 `AGENTIC_RAG_MEM0_LLM_API_KEY`。如果 Mem0 已启用但安装、配置或服务提供方构造失败，查询 API 仍可继续运行，但会明确进入 memory degraded 状态：`/health/ready` 报告 `memory=unavailable`，并写入包含 `component=mem0`、`reason`、`outcome=degraded`、`retryable` 的有界 `memory_provider_degraded` 日志。查询证据、审计和租户隔离不会因此放宽；`GET/DELETE /v1/memories` 会返回记忆服务不可用，而不是返回空的成功结果。

真实服务提供方合约测试只能使用明确指定的临时命名空间。缺少变量时测试会跳过；已配置但服务提供方不健康时测试会失败：

```sh
export AGENTIC_RAG_TEST_MEM0_ENABLED=1
export AGENTIC_RAG_TEST_MYSQL_DSN='mysql+asyncmy://.../agentic_rag_test'
export AGENTIC_RAG_TEST_ELASTICSEARCH_URL='http://127.0.0.1:9200'
export AGENTIC_RAG_TEST_MEM0_EMBEDDING_BASE_URL='https://<qwen-endpoint>/v1'
export AGENTIC_RAG_TEST_MEM0_EMBEDDING_API_KEY='<qwen-key>'
export AGENTIC_RAG_TEST_MEM0_ELASTICSEARCH_API_KEY='<es-api-key>'
conda run -n agentic-rag python -m pytest --import-mode=importlib \
  tests/e2e/test_mem0_real_services.py -q -s
```

## 优雅停止与故障恢复

向 API 和 Worker 发送 `SIGTERM`。API 会立即拒绝新任务，并在配置的优雅退出时间内等待正在执行的 HTTP 请求完成。Worker 的图状态保存在配置的 query 与 ingestion SQLite checkpoint 文件中；重启两个单 Worker 进程后会继续处理可恢复任务。不要强制杀死进程，也不要复制处于打开状态的 checkpoint 数据库。

任何重试前都要先检查隔离区中的版本：

```sh
conda run -n agentic-rag python scripts/review_quarantined_version.py --help
```

检查 Redis dead stream（死信流）；只有在故障原因修复后，才重试失败的 Run/Job。重试具有幂等性：不要为了清空 pending 消息而手动 ACK，也不要重试已经完成的 Run/Job。

## 备份

备份前先干净地停止 API 和 Worker。默认命令只备份本地 SQLite checkpoint 与 Artifact；如果输出路径已存在会拒绝执行，随后通过 SQLite backup API 固化 WAL 状态，记录应用/架构/索引代际，写入规范化哈希清单，并校验每个文件的哈希。清单使用完整性哈希而非加密签名；请把备份保存到受信任且权限受控的介质上。

```sh
conda run -n agentic-rag python scripts/backup_local.py --output var/backups/backup-001
```

如需包含配置中的 MySQL 数据库和当前 Elasticsearch 代际，使用显式服务开关。该命令会执行 `mysqldump --single-transaction`，并导出受控索引、别名和模板：

```sh
conda run -n agentic-rag python scripts/backup_local.py --output var/backups/backup-001 --include-services
```

如需包含 Redis，请提供一个明确的键命名空间。备份不会扫描配置 Redis 数据库中的全部键：

```sh
conda run -n agentic-rag python scripts/backup_local.py \
  --output var/backups/backup-001 --include-services \
  --redis-key-prefix 'agentic-rag:backup:'
```

至少在不同本地介质上保留三份已验证的备份。删除旧备份前，先对每份备份执行恢复演练。备份目录属于不可变的运维证据：不要编辑 manifest、哈希文件、dump 或导出文件。

## 恢复与 Elasticsearch 回滚

只能恢复到不存在的路径。恢复流程会在创建目标并原子发布之前，校验哈希清单和每个内容哈希；不会替换已存在的目录。

```sh
conda run -n agentic-rag python scripts/restore_local.py \
  --backup var/backups/backup-001 --target var/restore-drill/backup-001
```

包含服务状态的备份必须显式提供空的服务目标。MySQL 目标必须是独立的空数据库；Elasticsearch 必须使用新的索引代际。命令会拒绝非空 MySQL 数据库或已存在的目标索引，导入 dump，执行 Alembic 迁移，恢复并校验目标索引/模板/别名，同时保持生产当前代际不变。

```sh
conda run -n agentic-rag python scripts/restore_local.py \
  --backup var/backups/backup-001 --target var/restore-drill/backup-001 \
  --mysql-dsn 'mysql+asyncmy://.../agentic_rag_restore_001' \
  --elasticsearch-url http://127.0.0.1:9200 --index-generation restore-001 \
  --redis-dsn 'redis://127.0.0.1:6379/15' \
  --redis-key-prefix 'agentic-rag:restore:'
```

如果备份包含 Redis，恢复必须提供不同的明确目标前缀。恢复会把源前缀映射到目标前缀；写入前会扫描并确认目标命名空间为空。

要回滚搜索，只有在检查目标代际的映射和文档数量后，才把受控活动别名指向此前已验证的代际。在回滚通过就绪检查和查询冒烟测试前，不要删除当前索引。

### 临时真实服务恢复演练

真实服务测试不会猜测 admin DSN，也不会使用配置中的 `agentic_rag` 数据库。设置一个明确的 MySQL admin DSN（仅允许创建/删除临时测试数据库）、一个本地 Redis 数据库和本地 Elasticsearch 端点。不要把 admin 密码写入 shell 历史或提交到仓库：

```sh
set -a; source .env.local; set +a
export AGENTIC_RAG_RUN_REAL_BACKUP_RESTORE=1
export AGENTIC_RAG_TEST_MYSQL_ADMIN_DSN='mysql+asyncmy://<admin>:<password>@127.0.0.1:3306/mysql'
export AGENTIC_RAG_TEST_REDIS_DSN='redis://127.0.0.1:6379/15'
export AGENTIC_RAG_TEST_ELASTICSEARCH_URL='http://127.0.0.1:9200'
conda run -n agentic-rag python -m pytest --import-mode=importlib \
  tests/e2e/test_backup_restore.py -q -s
```

固定测试会随机生成 `agentic_rag_backup_*` 和 `agentic_rag_restore_*` 数据库、一个代际/别名以及两个用于源到目标映射的 Redis 键前缀，并在 `finally` 中清理。如果缺少任一显式变量，真实测试会跳过，不会触碰推断出的服务目标。

## 最终门禁

使用显式开关执行最终验收序列，包括基础设施和模型测试。`live_model` 没有凭据缺失跳过逻辑：选择该标记但未提供凭据时必须失败。真实备份测试需要生成隔离目标，必须单独启用。

```sh
conda run -n agentic-rag ruff check src tests evals scripts
conda run -n agentic-rag mypy src
conda run -n agentic-rag python -m pytest -m 'not integration and not e2e and not live_model' -q
conda run -n agentic-rag python -m pytest -m integration -q
conda run -n agentic-rag python -m pytest -m e2e -q
AGENTIC_RAG_RUN_REAL_BACKUP_RESTORE=1 conda run -n agentic-rag python -m pytest -m e2e tests/e2e/test_backup_restore.py -q
conda run -n agentic-rag python -m pytest -m live_model tests/smoke -q
conda run -n agentic-rag python -m evals.run --mode fixture \
  --dataset evals/datasets/baseline.jsonl --output var/artifacts/evals/fixture
# 最终验收必须使用与运行时快照匹配的数据集和真实客户端：
conda run -n agentic-rag python -m evals.run --mode graph \
  --dataset var/artifacts/evals/runtime-baseline.jsonl --output var/artifacts/evals/graph
# 或评估已部署的 API（设置数据集使用的快照 ID）：
AGENTIC_RAG_EVAL_SNAPSHOT_ID='<runtime snapshot id>' \
conda run -n agentic-rag python -m evals.run --mode api \
  --base-url http://127.0.0.1:8000 \
  --dataset var/artifacts/evals/runtime-baseline.jsonl \
  --output var/artifacts/evals/api
conda run -n agentic-rag python scripts/verify_acceptance.py \
  --report var/artifacts/evals/graph/summary.json
```

发布门禁还要求提供以下证据：Worker 已启动、真实 Graph/API 查询已完成、Mem0 可用，并且观测到至少一个安全的降级/熔断事件。请使用隔离发布运行生成的文件路径；普通本地测试不要设置这些变量：

```sh
export AGENTIC_RAG_RELEASE_WORKER_STARTED=1
export AGENTIC_RAG_RELEASE_QUERY_E2E=1
export AGENTIC_RAG_RELEASE_MEM0_AVAILABLE=1
export AGENTIC_RAG_RELEASE_SUMMARY=var/artifacts/evals/graph/summary.json
export AGENTIC_RAG_RELEASE_TELEMETRY=var/artifacts/observability/release.jsonl
conda run -n agentic-rag python -m pytest --import-mode=importlib \
  tests/e2e/test_release_query_gate.py -q
```

默认的 `fixture` 模式仅用于离线冒烟测试，并会打印 `SMOKE ONLY`；它永远不能满足最终验收。Graph/API 模式会持久化真实客户端来源证明，并拒绝运行时快照与组合出的 QueryGraph/API Run 不一致的案例。请基于已播种的文档和当前 `RuntimeConfigSnapshot` 准备 `runtime-baseline.jsonl`，不要事后修改数据集的 snapshot ID。`verify_acceptance.py` 必须针对 Graph/API 摘要（而不是 fixture 摘要）运行。只要泄漏不为零、引用覆盖率不等于 1.0、存在未审计答案、恢复或备份演练失败，或 `real_query_count` 不为正数，验证器就会失败。未配置 Ragas 后端时，Ragas 会明确保持 `unavailable`，不会伪造分数。
