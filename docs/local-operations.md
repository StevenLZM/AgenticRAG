# 本地运行手册

以下流程仅适用于本地 V1 部署。每次执行命令前先加载本地配置：

```sh
set -a; source .env.local; set +a
```

## 启动、控制台与查询

查询入口使用同一次 Router 调用选择 `chat`、`fast_rag` 或 `research`。
问候、闲聊和不带检索请求的偏好陈述走 `chat → finalize`；包含文档查询或分析的
混合输入仍走检索路径。三条路径均在原 `finalize` 位置从原始用户消息提取长期记忆。
聊天回复不携带文档引用，也不标记为已通过文档审计；回复时不承诺记忆已经写入。
Router 与聊天 Prompt 修改后需重启 API 和 Query Worker，以加载一致的配置快照。

按以下顺序启动依赖：Elasticsearch、MySQL、Redis。确认三者都指向本地端点后，先执行依赖检查和迁移：

```sh
conda run -n agentic-rag python scripts/check_local_dependencies.py
conda run -n agentic-rag alembic upgrade head
```

`python scripts/run_api.py`、`python scripts/run_query_worker.py` 和 `python scripts/run_ingestion_worker.py` 都是阻塞进程。必须在三个独立终端（或三个受监督的后台进程）分别启动，不能把它们放入一段串行命令：第一个进程会持续运行，后面的 Worker 永远不会启动。

终端一启动 API：

```sh
conda run -n agentic-rag python scripts/run_api.py --grace-seconds 30
```

终端二启动 Query Worker：

```sh
conda run -n agentic-rag python scripts/run_query_worker.py
```

终端三启动 Ingestion Worker：

```sh
conda run -n agentic-rag python scripts/run_ingestion_worker.py
```

API 启动后可访问 `/health/live`；只有在所有依赖都报告 `available` 后，才使用 `/health/ready`。在浏览器打开 Agentic RAG 控制台，或用以下命令确认同源页面已提供：

```sh
curl http://127.0.0.1:8000/
```

### MySQL 时区约定

应用将 MySQL `DATETIME(6)` 字段统一解释为 UTC。`DATETIME` 本身不保存时区信息，因此应用创建的每个 MySQL 连接都会自动执行 `SET time_zone = '+00:00'`；这样由 `CURRENT_TIMESTAMP(6)` 生成的默认值也与应用写入的 UTC 时间保持一致。已有记录不需要加 8 小时，也不要直接修改历史数据。

检查实例的时区和当前时间：

```sql
SELECT
  @@global.time_zone AS global_time_zone,
  @@session.time_zone AS session_time_zone,
  @@system_time_zone AS system_time_zone,
  NOW(6) AS mysql_now,
  UTC_TIMESTAMP(6) AS utc_now,
  TIMEDIFF(NOW(6), UTC_TIMESTAMP(6)) AS offset_from_utc;
```

为使其他客户端和重启后的实例也默认使用 UTC，MySQL 8+ 可使用管理员账号执行持久化配置（会立即应用，并写入 `mysqld-auto.cnf`）：

```sql
SET PERSIST time_zone = '+00:00';
```

也可以在 `my.cnf` 的 `[mysqld]` 段加入（适用于不使用 `SET PERSIST` 的环境）：

```ini
default-time-zone = '+00:00'
```

若只能执行临时配置，可先执行 `SET GLOBAL time_zone = '+00:00'`；它只影响之后新建的连接，重启后会失效。修改配置文件后按本机 MySQL 安装方式重启服务；API 和两个 Worker 重启后会自动建立 UTC 会话。

### Elasticsearch 活动 Alias

Child 检索始终访问稳定 Alias `agenticrag-children-active`，而不是把物理索引名写入查询代码。当前修复发布的默认契约是 `ingestion_pipeline_version=ingestion-v2`、`index_generation=index-v3`，对应的物理索引只能是 `agenticrag-children-index-v3`。Query Worker 启动时会校验当前 `AGENTIC_RAG_INDEX_GENERATION` 对应的物理索引；如果索引已存在但 Alias 缺失，会自动创建并记录 `elasticsearch_active_alias_repaired`。Ingestion Worker 发布新代际时通过 Elasticsearch 原子 Alias 操作切换目标；Alias 已指向其他代际时会 fail-closed 并记录异常，避免静默覆盖正在使用的搜索代际。

因此正常启动不需要手工执行 `PUT /_alias`。排查时可只读检查 Alias：

```sh
curl http://127.0.0.1:9200/_alias/agenticrag-children-active
```

如果当前代际物理索引尚未创建，Query Worker 会记录 `elasticsearch_active_alias_pending`；首个成功入库发布完成后，Publisher 会自动创建或切换 Alias。

### v2/v3 重建前置检查（仅 dry-run）

重建前必须先完成备份校验、停止并确认 API/Worker 已 quiescent，以及把源 Artifact 放入隔离 staging。`scripts/rebuild_preflight.py` 只读取操作员提供的 dry-run 清单并写出报告，不删除索引、不停服务、不软删除文档，也不执行重传；它拒绝通配符、未解析的物理索引和没有设计触发原因的非触发源重排：

```sh
conda run -n agentic-rag python scripts/rebuild_preflight.py \
  --index-generation index-v3 \
  --physical-index agenticrag-children-index-v3 \
  --backup-verified --workers-quiescent --sources-staged \
  --report var/artifacts/rebuild-preflight-v3.json
```

只有报告 `ready=true` 且逐项记录了重排原因、Parent 数量、token 分布和 source-span 覆盖后，才能由另一个经审批的变更执行备份—删除—重建 runbook。不要把 `*`、未展开的环境变量或 Alias 名称作为删除目标。

控制台实际调用 `POST /v1/query` 这个同步 wrapper：它创建与异步 API 相同的持久化 Run，并在 `wait_seconds` 内等待终态；未完成时返回现有 Run 的 `202` 响应，不会创建第二个 Run。`POST /v1/query-runs` 是直接创建 Run 后立即返回的异步 API，供其他客户端或显式异步流程使用；页面不会把它误当成同步路径。两种入口都通过 `GET /v1/query-runs/{run_id}/events` 的 SSE 接收进度。断线后页面会使用最后收到的 event ID 重连；运维排查也可传入 `Last-Event-ID` 重新连接同一路径。页面只呈现服务端白名单内的运行状态、证据、审计、`runtime_config_snapshot_id` 与 `client_provenance`，绝不呈现 prompt、记忆原文、隐藏推理、工具载荷或服务提供方原始响应。升级时先停止 Worker，等待当前图节点到达 SQLite checkpoint，再执行迁移，并按相同顺序重新启动。

## Query Outbox 与 Query Worker

Query Outbox 与 Query Worker 是两个不同的生产职责，缺一不可。`POST /v1/query-runs` 的 Query Outbox 在 MySQL 同事务中写入持久化 Run 和待投递的 `query_run` 意图；它不执行 Graph，也不依赖 Redis 当时可用。Query Worker 再将 `query_run` 投递到 Query Redis Stream，通过 Redis 领取/重新领取消息、维护租约和心跳、恢复中断的 Run、执行 QueryGraph、处理有界重试/DLQ，并且只在审计后的答案或终态业务拒答已持久化后执行终态 ACK。

这种拆分避免了“HTTP 已成功但 Redis 未发布”的丢失窗口，也避免 Worker 重启时重复创建 Run。Redis 发布故障保持为可重试 Outbox 状态；Worker 失联后由租约和重新领取恢复；超过重试上限才进入 DLQ。不得手动 ACK pending 消息来制造完成状态。

## 子 Agent、Todo 与研究预算

`SubagentDispatcher` 已接入生产组合根，并与 QueryGraph 和 Query Worker 共用 `ConcurrencyManager`。Todo 初始创建由 Research Agent 首次通过 `create_todos.items` 提交计划，服务器分配 ID，不再自动添加根 Todo；后续 Todo 追加也由 Agent 显式提交。依赖使用 `blocked_by`；旧 checkpoint 的 `dependencies` 仍可读取。只有所有前置任务已完成的 pending Todo 才能执行，失败任务由 Agent 显式重试或跳过，完成状态与输出引用由服务端维护。

每个 AgentLoop 节点只执行一个动作，然后通过同节点自边进入下一轮，动作间由 SQLite checkpoint 持久化。新 Run 的 `AGENTIC_RAG_MAX_RESEARCH_ROUNDS` 默认 6、最大 8，创建计划和提交也计入动作预算；单任务最多执行 2 次。计划最多 12 个任务、每项 8 个直接依赖、依赖路径最多 4 个节点。修改环境变量需重启 API/Worker，新设置不覆盖已创建 Run 的运行时快照。

委派使用 asyncio 并行检索 ready Todo，不增加独立调度节点。带依赖的委派任务必须在 `delegate_research.queries` 中提供根据上游事实解析后的检索词；worker 同时接收当轮证据清单、依赖证据和计算结果引用。成功兄弟任务的结果不会因另一任务普通执行异常而丢失。证据充分性仍由 Evidence Grader 判断；返回新 gaps 后必须新增工作再提交。

`research_attempt_count` 是跨 QueryGraph 重入且持久化在 QueryState/SQLite checkpoint 的全局研究尝试计数，不是单次 `ResearchAgentLoop` 的局部循环变量，也不是 QueryRun DB 字段。达到当前 `RuntimeConfigSnapshot` 中的上限时，未完成 Todo 会标记为 blocked，查询以 `research_round_limit` 的可观察终态停止，且不再调用模型。这个预算、子 Agent 的超时/取消及证据归并共同防止递归研究消耗失控。

## 降级、熔断与重试日志

所有降级边界都会产生可观察信号。进程日志会写入有界 warning，事件仓储会写入安全的持久化事件，例如 `memory_provider_degraded`、`RETRIEVAL_DEGRADED`、`MODEL_REPAIR_EXHAUSTED`、`CIRCUIT_OPEN`、`COMPONENT_DEGRADED`、`OUTBOX_RETRY` 或 `WORKER_DLQ`。日志和 SSE 只保留白名单字段；通用降级事件包括 `component`、`reason`、`outcome`、`attempt`、`retryable` 与 `degraded_components`，LLM 事件还可包括 `operation`、`requested_model`、`protocol`、`client_timeout_seconds`、`error_class`、`http_status` 和 `provider_request_id`。这些字段只用于定位调用边界，不保存原始 prompt、记忆文本、隐藏推理、工具载荷或服务提供方响应。

- `memory_provider_degraded`：检查 Mem0 配置、provider 连通性与 `/health/ready`；查询可继续，但记忆接口会 fail-closed。
- `MODEL_REPAIR_EXHAUSTED`：模型结构化输出在允许的 repair 后仍不符合 schema；保留 fail-closed 拒答，不要复制原始模型内容。
- `CIRCUIT_OPEN`：连续依赖失败已打开熔断；检查对应 `component`、`reason` 和冷却窗口，而不是强制重试。
- `WORKER_DLQ`：修复根因后检查 Run、租约和 Redis dead stream，再走受控重试；不得直接 ACK。

## DeepSeek 结构化调用与故障诊断

`ModelGateway` 是唯一的 DeepSeek 重试和 schema repair 边界。`AGENTIC_RAG_DEEPSEEK_PROTOCOL=auto` 时，如果客户端同时暴露两种接口，优先使用 Chat Completions；只有明确设置 `responses` 才使用 Responses API。结构化调用会请求 JSON object，但最终仍必须通过应用的 Pydantic schema；Markdown JSON 围栏只会被移除，不会放宽字段、枚举、证据 ID 或非空约束。

常见信号含义如下：

- `provider_outage`：超时、连接错误、429 或 5xx，网关按上限重试。
- `model_unavailable`：瞬态故障耗尽重试次数，查询进入明确降级/拒答路径。
- `protocol_error`：选择的 Chat/Responses 接口不可用或协议参数不被客户端支持，不会伪装成网络重试。
- `model_schema_invalid`：供应商返回了内容，但一次 repair 后仍不符合节点 schema；系统 fail-closed。
- `circuit_open`：连续失败触发熔断，后续调用在冷却窗口内快速失败。

模型诊断事件只保留 schema 名、协议、请求/实际模型、调用 operation、客户端超时、attempt、错误类型、HTTP 状态、服务提供方 request id、输出长度和输出 SHA-256，不保存 prompt、隐藏推理、原始工具载荷或原始模型响应。`provider_request_id` 仅保留服务提供方生成的安全标识；若 SDK 没有提供它，字段会省略。排查时先按 `snapshot_id`、`event_type`、`operation` 和 `reason` 聚合；同一 graph node 内多个逻辑调用按事件序列区分，不要把 repair 内容复制到日志或工单。

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

本地回环 Elasticsearch 的 `local-no-auth` 只是 mem0ai 2.0.12 的配置校验哨兵，不会作为 `Authorization` 头发送；远程端点不得使用该值。Mem0 可选的 PostHog 遥测在本地或 CI 中建议关闭：

```sh
export MEM0_TELEMETRY=0
```

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
MYPYPATH=src conda run -n agentic-rag mypy --explicit-package-bases src evals scripts
conda run -n agentic-rag python -m pytest -m 'not integration and not e2e and not live_model' -q
conda run -n agentic-rag python -m pytest -m integration -q
conda run -n agentic-rag python -m pytest -m e2e -q
AGENTIC_RAG_RUN_REAL_BACKUP_RESTORE=1 conda run -n agentic-rag python -m pytest -m e2e tests/e2e/test_backup_restore.py -q
conda run -n agentic-rag python -m pytest -m live_model tests/smoke -q
# 真实 Graph/API、当前 snapshot 和 client provenance 的正式评测入口。
# 先按 docs/rag-evaluation-progress.md 核对隔离资源，不使用业务8000/9200。
# 已完成的24题不要重复提交；优先运行下方只读核验。
PYTHONPATH=src conda run -n agentic-rag python -m scripts.verify_acceptance \
  --report var/artifacts/evals/real-corpus-v1-20260916/evaluation/summary.json \
  --run-dir var/artifacts/evals/real-corpus-v1-20260916 --quality-only
# 只有需要恢复未完成测评时才运行；复用原Run和已评分结果。
NO_PROXY=127.0.0.1,localhost PYTHONPATH=src RAGAS_DO_NOT_TRACK=true \
var/eval-venv-ragas042/bin/python -m scripts.run_real_rag_evaluation \
  --run-dir var/artifacts/evals/real-corpus-v1-20260916 --stage evaluate
# 只读真实运行E2E，不启动Query或judge。
AGENTIC_RAG_EVAL_RUN_DIR=var/artifacts/evals/real-corpus-v1-20260916 \
conda run -n agentic-rag pytest --import-mode=importlib \
  tests/e2e/test_query_evaluation_api.py -q
```

发布门禁还要求提供以下证据：Worker 已启动、真实 Graph/API 查询已完成、Mem0 可用，并且观测到至少一个安全的降级/熔断事件。请使用隔离发布运行生成的文件路径；普通本地测试不要设置这些变量：

```sh
export AGENTIC_RAG_RELEASE_WORKER_STARTED=1
export AGENTIC_RAG_RELEASE_QUERY_E2E=1
export AGENTIC_RAG_RELEASE_MEM0_AVAILABLE=1
export AGENTIC_RAG_EVAL_RUN_DIR=var/artifacts/evals/real-corpus-v1-20260916
export AGENTIC_RAG_RELEASE_SUMMARY=var/artifacts/evals/real-corpus-v1-20260916/evaluation/summary.json
export AGENTIC_RAG_RELEASE_TELEMETRY=var/artifacts/observability/release.jsonl
conda run -n agentic-rag python -m pytest --import-mode=importlib \
  tests/e2e/test_release_query_gate.py -q
```

`fixture`/`graph` 不再是业务评测CLI模式。`scripts/run_real_query_acceptance.py` 已停用，退出码2，不创建资源或生成PASS；原先由 `console_acceptance_passed` 在 EvalRunner 后执行的旧控制台验收流程不再作为RAG质量入口。确定性模型/固定向量只留在明确标记的协议测试中，其real_query_count为0。正式评测使用真实上传文件、已冻结金标、隔离HTTP API和真实Ragas；不要事后修改gold或snapshot来适配结果。

`scripts/verify_acceptance.py` 通用验证器现在必须提供`--run-dir`，重新读取scoped MySQL/checkpoint，验证24题、8份文件、上传任务、源事实映射、真实排名、评分输入及汇总；它不单独校验生产Mem0容量、安全对抗和恢复/备份演练。`--quality-only`表示“测评执行完整”，不是“质量达标”。省略该参数仍执行发布硬门禁；本轮拒答率低且未做恢复/备份演练，因此不能发布通过。Ragas失败必须显式记录，禁止补0或用参考答案替代。模型tokens未逐Run持久化，报告明确不可用。
# 一条命令运行既有语料的真实 RAG 测评

信息来源路由的独立回归与服务切换说明位于本文末尾，不与本节正式 RAG 质量评测混合计分。

在项目根目录运行：

```sh
./scripts/eval_rag.sh
```

默认使用 `var/artifacts/evals/real-corpus-v1-20260916/corpus` 中的 8 份文档、同一运行目录的 `gold-mapping.json`（24 题）和已经发布到 ES9201 的索引。调用真实 Query API/Worker/Graph 和真实 Ragas；不会生成语料、上传、重新切分/embedding 入库、创建索引或启动/重启服务。查询向量和 Ragas 评分仍会调用模型，产生费用。

每次默认创建独立的 `var/artifacts/evals/evaluation-时间-随机码/`，不覆盖首轮成绩。终端显示输出目录；其中 `report.md` 为中文报告，`report.json` 为完整核验报告，`summary.json`/`results.jsonl`/`failures.jsonl` 为汇总、逐题结果和失败明细，`queries/` 与 `runtime-evidence/` 保存续跑账本和运行证据。

```sh
# 仅只读检查，不请求模型、不生成报告文件
./scripts/eval_rag.sh --check

# 指定一个尚不存在的输出目录
./scripts/eval_rag.sh --output-dir var/artifacts/evals/my-rag-eval

# 中断后显式续跑；不重新提交已记录的查询，不重复有效的已完成评分
./scripts/eval_rag.sh --resume var/artifacts/evals/my-rag-eval

# 指定既有输入路径；必须与上传账本/冻结金标/现有索引匹配
./scripts/eval_rag.sh \
  --source-run var/artifacts/evals/real-corpus-v1-20260916 \
  --docs-dir var/artifacts/evals/real-corpus-v1-20260916/corpus \
  --gold var/artifacts/evals/real-corpus-v1-20260916/gold-mapping.json
```

当前入口是本项目冻结语料的重测入口，不是任意目录自动建库工具：目录内所有受支持文档必须与 manifest 完全匹配，辅助 `qa/` 不参与测评；新增/缺失/变更文档、错误版本或未入库内容直接报错，绝不静默跳过。更换数据集需先另行完成上传和冻结金标。服务必须事先可用（默认 API8001、ES9201、隔离 MySQL/Redis、Query Worker）。默认解释器为 `var/eval-venv-ragas042/bin/python`，可用 `RAG_EVAL_PYTHON` 指定已配置的环境；脚本不会安装依赖。

退出码：0 表示全量测评及运行证据核验完成，**不是生产发布通过**；1 表示本轮未完成/评分或核验失败，查看报告并修复后续跑；2 表示前置检查或参数错误。快照或输入变化须开启新实验；judge 配置变化时底层 runner 可能重新评分，但会复用查询结果。并发续跑同一目录会被文件锁拒绝；不确定的 POST 提交结果不会自动重提。

## 信息来源路由回归与切换

设计与实际结果见[路由验收记录](capability-routing-validation.md)。在项目根目录，用已配置模型凭据的环境运行：

```sh
python scripts/eval_routing.py --dataset evals/datasets/routing_v2.jsonl \
  --repeats 3 --output-dir /private/tmp/routing-eval-unique-new-directory
```

目录必须为空，不能覆盖旧结果。该入口只调用分类模型，60 题各运行新旧版 3 次共 360 次逻辑调用；不读写业务记忆、不检索、不接入天气工具。输出 samples.jsonl、manifest.json、report.json、report.md；退出码 0/1/2 分别表示 PASS/FAIL/INCOMPLETE。分类 PASS 不等于端到端上线通过。

真实服务回归使用 `python -m pytest tests/e2e/test_capability_routing.py -m 'e2e and live_model' -v -rs`。沿用现有真实 Query fixture 的显式本地测试配置：`AGENTIC_RAG_RUN_REAL_QUERY_PROVIDER_E2E=1`、测试 MySQL/Redis/ES DSN、测试 MySQL admin DSN 与 backup/restore opt-in；缺失时 skip。仅创建和清理随机命名的测试数据库、索引、Redis namespace 与 checkpoint；上传小型文本天气报告和简历，经真实 parser/assembler/chunker/embedding 发布，查询记忆策略 disabled，不触碰生产数据或 active alias。

正式切换前先完成隔离端到端验收，再停止接收新请求并排空旧活动 Run；不要让旧 prompt/schema checkpoint 被新图恢复。验证新快照包含 graph=query-v2、prompt=prompt-v2、routing_policy_version=routing-v2 及 v2 Prompt 哈希。得到用户明确授权后才重启 API/Query Worker，核对 API 与 Worker 的 snapshot_id 一致，再进行天气与文档问答 smoke。回滚也须排空活动 Run 并恢复匹配的图/Prompt 版本。本轮没有执行切换，不需要业务数据库迁移、重建索引或重新切块。
