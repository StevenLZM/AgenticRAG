# AgenticRAG Docker Deployment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 交付独立、可验证、可安全停止的容器化应用，先证明 Mem0 与主检索的真实 ES 认证，再验证完整业务和失败回退。

**Architecture:** 一个应用镜像运行 API 和两个独立 Worker，Compose 管理独立 MySQL、Redis、带 IK 的 ES、模型缓存和本地状态。ES 连接统一解析，容器使用真实账号密码及可信 CA；初始化与业务权限分离。验证环境与当前项目完全分开，不在本次执行真实数据迁移或上线切换。

**Tech Stack:** Python 3.11、Docker Compose、Elasticsearch 与 IK 8.19.0、Mem0 2.0.12、MySQL 9.4.0、Redis 7.4.9、Docling、LangGraph、pytest。

**Spec:** `docs/superpowers/specs/2026-10-03-docker-deployment-design.md`，用户于 2026 年 10 月 3 日确认；本实现计划仍待审核及选择执行方式。

## Global Constraints

- API、Query Worker、Ingestion Worker 分别运行；两个 Worker 各一个实例。
- 保持 RAG 路由、分块、Embedding 维度、检索排序、记忆策略和当前数据格式不变。
- 现有 ES `9200`、评测 ES `9201`、业务 API、评测 API、共享 MySQL 和 Redis 不停机、不修改配置、不改变认证。
- 不因测试失败关闭 Mem0、伪造凭证、扩大匿名访问范围或忽略证书错误。
- 应用使用 Python 3.11 的 Linux CPU 镜像；Mem0 保持 `2.0.12`；ES 和 IK 均保持 `8.19.0`。
- MySQL 和 Redis 以当前 `9.4.0`、`7.4.9` 为兼容基线；验证镜像可用性和架构后锁定 digest，不静默换主版本。
- 知识库使用 `index-v4` 和 IK，Embedding 保持 1024 维；仅 Compose 显式设置，不改变宿主机代码默认的 `index-v3` 和 `standard`。
- Linux arm64 和 amd64 分别记录构建及运行证据；未测架构不能标为通过。
- 不自动替换现有业务服务，不迁移真实数据，不合并或推送实现分支。正式切换是后续单独授权的操作。
- 实现位于独立 worktree 和 `codex/` 分支；现有未提交文件、评测文档和 `tmp/` 不是可丢弃内容。

## Review Focus

1. Secret 中含 `$`、`@`、空格或末尾换行，以及新旧配置冲突：不能损坏凭证、静默降级或泄漏到 URL/日志。由任务 1、2、5 验证。
2. 恢复目标缺失索引或 SQLite：不能通过自动建空存储伪装成功，也不能把新安装流程当作恢复。由任务 5、8 验证。
3. 测试项目名碰撞、被篡改的 manifest、宿主路径挂载：拒绝启动或停止不属于本次任务的资源。由任务 3、9 验证。
4. 长时间 OCR、批量 Redis delivery、终态落库前崩溃：停止有时间上界，恢复不重复已完成业务效果。由任务 7、9 验证。
5. 冷模型缓存、错误 UID、第二次初始化和容器重建：应明确失败或保留原有状态，不能在第一份用户文档才暴露问题。由任务 4、6、8、9 验证。

## 执行入口与证据规则

- [ ] 使用 using-git-worktrees 技能和原生 worktree 工具，从包含本设计和计划的已确认提交创建隔离工作区。`main` 保持原样；不复制 `.env.local`、`var/` 或 `tmp/`。记录实现基线 SHA 及主工作区的已有改动清单，未提交文档不由 Git SHA 代替备份。
- [ ] 在隔离工作区运行 `python -m pytest tests/unit -q`、`python -m ruff check src scripts tests` 和 `python -m mypy src/agentic_rag --ignore-missing-imports`，记录总数和退出码；若基线失败，先报告并确定处理方式，不把原有失败算作容器化回归。
- [ ] 所有任务先写失败测试，确认失败原因，再实现、复测。Git 提交只包含本任务明确列出的文件，禁止 `git add .`。
- [ ] 执行 live 测试前，任务 3 的资源身份、端口和资源预算检查必须通过。只发送合成数据给已配置的模型服务，不复制真实业务文档。
- [ ] 下文命令均在隔离工作区运行。宿主机使用现有 Python 环境；镜像测试使用镜像内 Python。规划阶段不执行任何 `up`、`build`、`run` 或清理命令。

## 文件组织与依赖顺序

`persistence/es_connection.py` 只管连接参数及客户端构造；`deployment/secrets.py` 管 Secret 文件输入；`deployment/isolation.py` 与 `scripts/docker_stack.py` 管资源归属及测试栈生命周期。`runtime/model_assets.py` 管本地模型定位，`runtime/worker_lifecycle.py` 管进程就绪/停止，`runtime/query_resume.py` 管生产图的 checkpoint 恢复选择。初始化和验收使用窄入口，不能通过构造整个 AppContainer 偷跑迁移或建索引。

任务顺序为 `1 → 2 → 3 → 4 → 5 → 6 → 7 → 8 → 9`。任务 5 是认证硬门槛，失败时保留隔离证据并停止后续全栈接线，不修改现有 ES。任务 9 是交付门槛，失败时不合并、不上线。

### Task 1: 定义统一 ES 配置与 Secret 输入

**Files:**

- Create: `src/agentic_rag/persistence/es_connection.py`、`src/agentic_rag/deployment/__init__.py`、`src/agentic_rag/deployment/secrets.py`。
- Modify: `src/agentic_rag/config.py`。
- Test: `tests/unit/persistence/test_es_connection.py`、`tests/unit/deployment/test_secrets.py`、`tests/unit/test_config.py`。

**Interfaces:**

- `ElasticsearchConnection` 为 frozen dataclass：`url: str`、`username: str | None`、`password: SecretStr | None`、`verify_certs: bool`、`ca_certs: Path | None`；repr 不暴露密码。
- `resolve_elasticsearch_connection(settings: object) -> ElasticsearchConnection`；保留 SimpleNamespace 单测兼容，新增 `ElasticsearchConfigurationError`。
- `create_async_es(connection: ElasticsearchConnection, *, timeout_seconds: float = 30.0) -> AsyncElasticsearch`、`create_sync_es(connection: ElasticsearchConnection, *, timeout_seconds: float = 30.0) -> Elasticsearch`、`create_es_http_client(connection: ElasticsearchConnection, *, timeout_seconds: float = 30.0) -> httpx.AsyncClient`。
- `mem0_es_config(connection: ElasticsearchConnection, *, collection: str, dimensions: int, auto_create_index: bool) -> dict[str, object]`。
- `resolve_secret_files(environ: Mapping[str, str]) -> dict[str, str]` 返回进程环境副本，不直接修改全局环境；新增 `SecretConfigurationError`。

- [ ] **Step 1: 写配置和文件输入失败测试。** 断言 `https://elasticsearch:9200` 加用户名密码可以构造、缺失或半套凭证被拒绝；新旧同值可兼容，冲突被拒绝；无认证 loopback 仍生成 `local-no-auth`；普通 host 的真实 API key 配置明确报错。断言通用字段默认不改变现有 index/lexical 默认值。
- [ ] **Step 2: 运行红测。** `python -m pytest tests/unit/persistence/test_es_connection.py tests/unit/deployment/test_secrets.py tests/unit/test_config.py -q`；必须因待实现接口失败，而非依赖缺失或连接真实服务。
- [ ] **Step 3: 实现连接解析与客户端工厂。** 新增通用 `elasticsearch_user/password/ca_certs/verify_certs`，`verify_certs` 用 `None` 表示未显式设置，最终默认为 True；用 `model_fields_set` 区分旧字段的默认值与显式配置，SimpleNamespace 则以实际提供的属性判断，不能把旧默认 True 误判为配置冲突。禁止 URL userinfo/query/fragment 中夹带参数及非预期路径，Mem0 的 IPv6 host 保留方括号。客户端关闭由调用者负责。HTTPX 使用独立 client、`trust_env=False`、CA 对应的 SSLContext，不给业务 API 请求附带 ES 凭证。
- [ ] **Step 4: 实现并验证 Secret 输入。** 只接受明确白名单的 `AGENTIC_RAG_*_FILE`：MYSQL_DSN、ELASTICSEARCH_PASSWORD、DEEPSEEK_API_KEY、QWEN_API_KEY、MEM0_EMBEDDING_API_KEY、MEM0_LLM_API_KEY、DASHSCOPE_API_KEY；原值和文件同时设置即拒绝。文件必须可读、非空，最多去掉一个末尾 LF 或 CRLF，不裁剪其他空格；错误只包含配置名。测试断言 `b' p@$ ss \r\n'` 解析为 `' p@$ ss '`，日志和 repr 不包含测试密码。运行 Step 2 命令至全绿。
- [ ] **Step 5: 提交上述文件。** `git commit -m "feat: centralize Elasticsearch connection and secret resolution"`。

关键测试名与断言：`test_mem0_basic_auth_preserves_raw_password` 使用 `assert config['password'] == ' p@$ ss '`；`test_loopback_sentinel_is_not_sent_as_authorization` 使用 `assert 'Authorization' not in request.headers`；`test_ipv6_host_round_trips` 使用 `assert config['host'] == 'http://[::1]'`。缺失文件、冲突及真实 API key 使用 `pytest.raises(ElasticsearchConfigurationError)` 或 `pytest.raises(SecretConfigurationError)`，异常文本不能包含测试密码。

### Task 2: 给所有受支持的 ES 调用接入同一配置

**Files:**

- Modify: `src/agentic_rag/bootstrap.py`、`src/agentic_rag/memory/factory.py`、`scripts/backup_local.py`、`scripts/restore_local.py`、`scripts/migrate_ik_index.py`、`scripts/real_acceptance_evidence.py`、`scripts/export_formal_eval_snapshot.py`、`scripts/upload_bulk_corpus.py`、`scripts/run_real_rag_evaluation.py`、`evals/real_stack.py`。
- Test: `tests/unit/memory/test_mem0_factory.py`、`tests/unit/persistence/test_es_client_wiring.py`、`tests/e2e/test_backup_restore.py`、`tests/unit/test_ik_migration.py`、`tests/unit/evals/test_real_stack.py`、`tests/integration/memory/test_mem0_adapter.py`。

**Interfaces:**

- 消费任务 1 的连接工厂，`build_container(settings)` 和 `build_mem0_config(settings)` 的外部签名不变。
- `BackupSpec`、`ServiceRestoreSpec` 和 `scripts/real_acceptance_evidence.py` 的 `ServiceBackupResources` 新增 `elasticsearch_connection: ElasticsearchConnection | None = None`，只在进程内传递，不能序列化进备份 manifest；同时有 URL 时必须一致。
- 配置新增 `mem0_auto_create_index: bool = True`；宿主机默认不变，恢复校验和 Docker 运行期使用 False，建索引归 app-init。

- [ ] **Step 1: 写接线红测。** patch 客户端构造器捕获调用，断言主 AsyncElasticsearch、Mem0 普通 host 和 HTTPX 收到相同 basic auth/CA；脚本中的模型与业务 API client 没有 ES Authorization；备份与评测 JSON 不包含密码；Mem0 自动建索引配置可关闭。
- [ ] **Step 2: 运行红测。** `python -m pytest tests/unit/persistence/test_es_client_wiring.py tests/unit/memory/test_mem0_factory.py tests/e2e/test_backup_restore.py -q`，确认新契约失败，现有独立临时备份测试不访问业务环境。
- [ ] **Step 3: 替换直接构造。** bootstrap 在 Mem0 降级 catch 之前解析统一连接，非法通用配置不能被吞成降级。backup、restore 的导入和恢复后 readiness 都接入连接对象；CLI 从同一 Settings 解析，库调用未给连接对象时维持旧 URL-only loopback 用法。`real_acceptance_evidence.py` 两处构造与三个 HTTPX ES 消费者一起覆盖。修正隔离评测配置使业务 ES 凭证及 Secret 路径不会继承到不同实例，不放宽其 dedicated loopback 校验；更新真实 Mem0 测试入口为显式 basic auth/CA，而非旧 API key 前提。只改连接，不改备份格式、恢复目标检查或数据迁移语义。
- [ ] **Step 4: 复测并搜索遗漏。** 运行 Step 2 命令；`rg -n 'AsyncElasticsearch\(|Elasticsearch\(|base_url=settings.elasticsearch_url' src scripts`，除统一工厂或明确的测试构造外不得残留新的配置绕行点。调整原 Mem0 单测的虚构 API key fixture，增加明确拒绝案例，而非继续假定 API key 有效。
- [ ] **Step 5: 提交上述接线文件。** `git commit -m "fix: share Elasticsearch authentication across application and tooling"`。

关键测试名与断言：`test_all_clients_receive_same_basic_auth` 使用 `assert captured_auth == ('rag_app', 'test-password')`；`test_backup_manifest_does_not_serialize_connection` 使用 `assert 'test-password' not in manifest_json`；`test_eval_target_does_not_inherit_business_credentials` 使用 `assert evaluation_connection.username is None`。运行 Step 2 时补跑上面新增的 unit 测试；真实 Mem0 fixture 在任务 5 的隔离环境执行，不对现有评测实例运行。

### Task 3: 建立独立测试栈的安全边界和回退命令

**Files:**

- Create: `src/agentic_rag/deployment/isolation.py`、`scripts/docker_stack.py`、`tests/unit/deployment/test_isolation.py`。
- Modify: `.gitignore`。

**Interfaces:**

- `StackManifest` 记录 `project_name`、随机 `ownership_token`、实现 `source_commit`、源工作区基线、Compose 文件哈希、明确的容器/网络/卷 ID、配置哈希、测试端口与资源限额；不存 Secret 内容。
- `capture_baseline(source_root: Path, output: Path) -> StackManifest`、`validate_stack_config(config: Mapping[str, object], manifest: StackManifest) -> None`、`stop_owned_stack(manifest: StackManifest) -> None`。
- CLI：`python -m scripts.docker_stack prepare --source-root ABS --output ABS` 创建独占的状态目录并输出其 manifest 路径；`up --manifest ABS --stage auth|full`；`down --manifest ABS`；`verify-baseline --manifest ABS`。后续任务扩展实现 up，不在本任务启动容器。

- [ ] **Step 1: 写所有权红测。** 伪造 Docker 输出，断言标签 token 不一致、项目名已占用、ID 不符、Compose 哈希改变、external volume、host network、挂载源工作区 `.env.local/var/tmp` 或其他未批准宿主路径时 fail closed，且没有执行 stop/remove。断言 `down` 没有 prune、volume rm、`-v`、`--remove-orphans` 或 Git reset。
- [ ] **Step 2: 运行红测。** `python -m pytest tests/unit/deployment/test_isolation.py -q`。
- [ ] **Step 3: 实现 manifest 与预检。** 项目名为 `agenticrag-docker-` 加随机后缀，所有资源有项目和 token 标签。每阶段生成不可覆盖的解析后 Compose 配置并记录哈希与镜像 ID，源文件后续修改不篡改已部署阶段的记录；新增阶段原子追加 manifest，创建中断后先按 token 和原始基线核对实际资源再恢复操作。生成配置必须显式禁用继承宿主 `.env.local`，使用绝对容器路径；API 默认 loopback 随机空闲端口，拒绝保护端口 `8000/8001/9200/9201/3306/6379`。使用当前 Docker 上下文/daemon 身份并记录，身份改变时拒绝操作。资源限额显式写入配置，内存总预算不超过 Docker 总内存的 70%，启动前另检查现有占用和磁盘余量；不足则停止，不停其他服务。
- [ ] **Step 4: 实现安全停止和基线核对并复测。** 只按重新核对标签后的具体 ID 停止本次容器；先导出脱敏日志，再移除本次空网络；默认不删除卷。基线核对检查源代码与配置哈希、既有服务身份和原来可用的健康端点，不要求仍在处理业务的数据库计数保持静止。报告其他任务并发变动，不自动覆盖。重跑 Step 2 并检查无真实 Docker 调用。
- [ ] **Step 5: 提交安全工具。** `git commit -m "feat: guard disposable Docker stacks and preserve rollback baseline"`。忽略 `.docker-state/`、`.docker-secrets/`，不改变其他任务文件的跟踪状态。

关键测试名与断言：`test_foreign_token_prevents_any_cleanup` 使用 `assert destructive_calls == []`；`test_failed_stack_keeps_volumes` 使用 `assert volume_delete_calls == []`；`test_protected_ports_rejected` 对六个保护端口逐个断言 `pytest.raises(ValueError)`。启动预检失败同样必须满足 `assert docker_create_calls == []`。

### Task 4: 构建可复现应用镜像和 ES IK 镜像

**Files:**

- Create: `Dockerfile`、`.dockerignore`、`docker/elasticsearch/Dockerfile`、`docker/images.lock.json`、`requirements/docker-linux-arm64.lock`、`requirements/docker-linux-amd64.lock`、`scripts/docker_entrypoint.py`、`tests/unit/deployment/test_image_contract.py`。
- Modify: `pyproject.toml`，仅补充构建/测试依赖边界，不升级业务算法。

**Interfaces:**

- 应用镜像工作目录 `/app`、非 root UID/GID `10001:10001`、ENTRYPOINT `python -m scripts.docker_entrypoint`；任意业务命令作为参数，以 `os.execvpe` 直接替换进程。
- entrypoint 消费任务 1 的 Secret 环境转换，不自行连接数据库；`MEM0_TELEMETRY=false` 必须在导入 Mem0 前设置。
- images.lock 保存具体版本、可用架构、镜像 digest 与 IK 下载 SHA256；两份 requirements lock 固定所有传递依赖和哈希。

- [ ] **Step 1: 写镜像契约红测。** 检查 `/app` 中不包含宿主 `.env.local`、Secret、`var/`、`tmp/`、`.git/`；这不要求删除 Linux 基础镜像本身的 `/tmp` 或 `/var`。镜像必须包含静态页面、prompts、`alembic.ini` 和 `alembic/`；entrypoint 不经 shell 拼接参数，能向 Python 传递 SIGTERM，敏感配置不落命令行。
- [ ] **Step 2: 运行红测。** `python -m pytest tests/unit/deployment/test_image_contract.py -q`。
- [ ] **Step 3: 解析 Linux 依赖与镜像。** 在受任务 3 管理的临时构建环境中验证 Python 3.11、ES/IK 8.19.0、MySQL 9.4.0、Redis 7.4.9 的镜像及架构；记录实际 digest，找不到精确兼容版本时报告而非替换。以当前宿主机相关包版本为候选解析 Linux wheel，固定 CPU Torch、Docling、OCR、sentence-transformers 和 ES 8.x 客户端。安装仅发生于构建环境，不能改宿主 Conda。
- [ ] **Step 4: 实现并构建。** 应用采用 Debian bookworm slim 基础，安装 Docling/OCR 必需运行库；IK 安装在镜像构建期并核验 SHA256。运行 Step 2；使用 `docker buildx build --platform linux/arm64 --load -t agenticrag:docker-arm64 .` 构建本机候选，amd64 独立构建验证；镜像内运行 `pip check`、导入实际依赖并验证 Mem0 版本。测试镜像文件系统与 `docker history` 不含合成 Secret。构建前执行任务 3 资源预检。
- [ ] **Step 5: 提交镜像与锁文件。** `git commit -m "build: add pinned CPU application and Elasticsearch IK images"`。构建记录保存到本次状态目录，不能把只构建成功写成业务验收通过。

关键测试名与断言：`test_image_excludes_host_state` 使用 `assert forbidden_paths.isdisjoint(image_paths)`；`test_entrypoint_preserves_argv` 使用 `assert exec_argv == requested_argv`；`test_runtime_uid_is_nonroot` 使用 `assert runtime_uid == 10001`。后续任务修改代码后必须重建候选应用镜像，最终认证与全栈报告指向同一个最终 digest。

### Task 5: 在真实容器网络完成认证硬门槛

**Files:**

- Create: `compose.infra.yaml`、`compose.auth.yaml`、`docker/elasticsearch/elasticsearch.yml`、`docker/elasticsearch/roles.json`、`scripts/initialize_volumes.py`、`scripts/initialize_es_security.py`、`scripts/check_docker_auth.py`、`tests/unit/deployment/test_es_security.py`、`tests/integration/deployment/test_container_auth.py`。
- Modify: `scripts/docker_stack.py`。

**Interfaces:**

- `initialize_es_security(settings: Settings) -> None` 仅由一次性管理容器调用，使用 bootstrap Secret/CA；配置生成无凭证日志。
- `initialize_volumes(*, role: Literal['app', 'es']) -> None` 只处理本次初始化容器挂载的固定白名单路径。一次性 root 初始化容器将指定原始 Secret 复制到对应专用卷，设置应用 UID 10001 或 ES UID 1000，文件 0400、目录 0700；应用只读挂载其专用 Secret 和公共 CA，不挂载 CA 私钥或管理员 Secret。不能假定 Compose 的 bind-mounted Secret 自动按声明的 uid/mode 改属主。
- `async run_auth_probe(settings: Settings, *, phase: Literal['write', 'verify_after_restart', 'delete']) -> dict[str, object]` 使用任务 2 的真实工厂与 Mem0Adapter，输出脱敏的逐项结果及合成记忆 ID。
- `up --stage auth` 启动 cert-init、volume-init、ES、security-init、auth-probe；应用探针使用任务 4 的同一应用镜像，以 `https://elasticsearch:9200` 访问，不使用 host networking 或宿主映射端口。

- [ ] **Step 1: 写权限和认证红测。** 验证初始化账号不挂给 auth-probe，应用账号不是 elastic；正确凭证成功，错误密码/未带认证的真实 HTTP 请求为 401，跨索引请求为 403，错误 CA 握手失败。项目配置缺少凭证的提前失败与 ES 实际 401 分开记录。role 不允许 `*` 全索引或 `manage_security`。
- [ ] **Step 2: 运行红测。** `python -m pytest tests/unit/deployment/test_es_security.py -q`；live 用例默认跳过，没有任务 3 的有效 manifest 时必须拒绝执行，而非回退到 `.env.local`。
- [ ] **Step 3: 建立认证基础设施。** cert-init 在专用卷生成 CA 和包含 `elasticsearch` SAN 的证书；ES 开启 HTTPS 和安全认证。管理容器幂等创建应用用户及探针所需知识库、Mem0 正式和 entities 索引，探针设置 auto_create_index=False；沿用现有 ChildIndexStore 与已锁定 Mem0 适配器的索引定义。运行角色 cluster 只给 `monitor`，index 权限限于 `agenticrag-children-index-v4`、`agenticrag-children-active`、`agent_memories_v1` 和 `agent_memories_v1_entities` 的 `read/write/create_index/manage`；负向测试索引由管理员创建，不属于应用角色。关闭遥测，不授予 mem0migrations 权限。证书、Secret 和状态卷先校验 UID/权限。重复初始化不轮换已有凭证或清空存储。auth 阶段使用明确服务列表，只带起所需依赖，不意外启动尚未验证的完整业务服务。
- [ ] **Step 4: 通过真实认证门槛。** `python -m scripts.docker_stack up --manifest ABS --stage auth` 后运行 `AGENTIC_RAG_DOCKER_TEST_MANIFEST=ABS python -m pytest tests/integration/deployment/test_container_auth.py -q`。主 sync/async/HTTPX 客户端与 Mem0 都必须测试；Mem0 写入合成记忆、等待 refresh、按正确用户读回、真实 Embedding 搜索命中、另一用户看不到。保留 ID 后重建 ES/probe 容器但复用卷，用 phase=verify_after_restart 验证，再 phase=delete 删除并确认。初始化错误、任何必测项跳过、或返回空结果均不得算通过。失败运行任务 3 的安全停止并报告，不进入任务 6。
- [ ] **Step 5: 提交认证栈和测试。** `git commit -m "feat: verify container Elasticsearch authentication and Mem0 persistence"`，Secret、证书私钥、测试状态与真实凭证不入 Git。

关键测试名与断言：`test_wrong_password_is_rejected_by_es` 使用 `assert response.status_code == 401`；`test_unrelated_index_is_forbidden` 使用 `assert response.status_code == 403`；`test_mem0_search_returns_saved_id` 使用 `assert saved_id in result_ids`；`test_mem0_survives_container_recreation` 使用 `assert restored_id == saved_id`。这些断言必须来自容器中真实请求，不能只断言预期异常的 mock。

### Task 6: 固定模型资源并验证离线加载

**Files:**

- Create: `src/agentic_rag/runtime/model_assets.py`、`scripts/initialize_models.py`、`docker/models.lock.json`、`tests/unit/runtime/test_model_assets.py`、`tests/integration/deployment/test_model_cache.py`。
- Modify: `src/agentic_rag/runtime/query_composition.py`、`scripts/run_ingestion_worker.py`、`src/agentic_rag/config.py`。

**Interfaces:**

- `ModelAssets` 保存 tokenizer、reranker、Docling/OCR 的本地路径和 revision；`load_model_assets(manifest_path: Path) -> ModelAssets` 校验完整性。
- `build_docling_converter(assets: ModelAssets) -> object` 使用显式 RapidOCR 后端和本地模型，供现有 `DocumentParser(converter=...)` 注入。
- `initialize_models(lock_path: Path, cache_root: Path) -> Path` 预下载、实际加载后原子发布 manifest；新增可选 `model_assets_manifest: Path | None` 配置，None 保持宿主机原路径。

- [ ] **Step 1: 写缓存和身份红测。** manifest 缺模型、哈希不符、逻辑模型 ID 与 Settings 不符时启动失败；重复初始化不重写完整模型。指定本地路径加载后，Settings 的 `embedding_tokenizer_model/reranker_model`、snapshot.reranker_version 和 Reranker.model_version 不变。
- [ ] **Step 2: 运行红测。** `python -m pytest tests/unit/runtime/test_model_assets.py tests/unit/runtime/test_query_composition.py -q`。
- [ ] **Step 3: 实现本地资源接线。** lock 固定 `Qwen/Qwen3-Embedding-0.6B`、`BAAI/bge-reranker-v2-m3` 及当前 Docling/OCR 所需资源的实际 revision/hash。Query 的 CrossEncoder 和 Ingestion 的 tokenizer 使用 manifest 路径，但不覆写模型 ID；Ingestion 在宣布 ready 前构造并预热注入的 converter。不能改变全局 Chunker 或 AST 规则。
- [ ] **Step 4: 实测冷缓存和离线复用。** 扩展 Compose model-init，一次性可联网预热；后续在禁止外部网络的测试容器中完成 tokenizer、reranker 和一份合成扫描 PDF 的 OCR。执行 `AGENTIC_RAG_DOCKER_TEST_MANIFEST=ABS python -m pytest tests/integration/deployment/test_model_cache.py -q`；额外断言缺缓存时立即明确失败，不转为运行期下载。两种架构分别记录证据。
- [ ] **Step 5: 提交模型资源接线。** `git commit -m "feat: preload and verify pinned local model assets"`。只提交 manifest 格式和 lock，模型二进制保留在缓存卷。

关键测试名与断言：`test_local_assets_preserve_logical_model_ids` 使用 `assert snapshot.reranker_version == 'BAAI/bge-reranker-v2-m3'`；`test_cold_offline_cache_fails_before_ready` 使用 `assert worker_ready is False`；`test_scanned_pdf_uses_preloaded_ocr` 使用 `assert synthetic_marker in parsed_text` 并确认测试容器无外网访问能力。

### Task 7: 给 Worker 增加可验证的就绪和有界停止

**Files:**

- Create: `src/agentic_rag/runtime/worker_lifecycle.py`、`src/agentic_rag/runtime/query_resume.py`、`scripts/check_worker_health.py`、`tests/unit/runtime/test_worker_lifecycle.py`、`tests/integration/runtime/test_query_checkpoint_resume.py`。
- Modify: `src/agentic_rag/runtime/query_worker.py`、`src/agentic_rag/ingestion/worker.py`、`scripts/run_query_worker.py`、`scripts/run_ingestion_worker.py`、`src/agentic_rag/config.py`、`tests/unit/ingestion/test_worker.py`、`tests/unit/runtime/test_query_worker_entrypoint.py`。

**Interfaces:**

- `WorkerLifecycle(state_path: Path, *, worker_kind: Literal['query', 'ingestion'], instance_id: str)` 提供 `starting()/ready()/stopping()/heartbeat()`，原子写入 PID、实例 ID、状态和 monotonic 时间；`worker_is_ready(state_path: Path, *, max_age_seconds: float = 15.0) -> bool` 不构造整个应用。
- 两个 `run_forever` 新增可选 `shutdown_grace_seconds: float | None = None`，None 保持宿主机兼容；容器设置 30 秒，Compose `stop_grace_period` 为 60 秒。
- Settings 新增 `worker_shutdown_grace_seconds: float | None = None` 和 `worker_health_path: Path | None = None`，由两个入口传递；每次进程启动生成新 instance_id，不复用上一次的 ready 标记。
- `CheckpointedQueryRunner(compiled_graph)` 保留现有 `ainvoke(state: dict[str, object], config: dict[str, object]) -> dict[str, object]` 边界；仅生产 `build_graph_factory` 返回此包装，内部检查持久化 snapshot 后选择首次执行、继续 pending 节点或复用已完成结果。

- [ ] **Step 1: 写停止与恢复红测。** Query 停止后不处理同批第二条消息，未处理消息不 ACK；SIGTERM 立即 unready，当前任务在 grace 内完成可 ACK，超时不假报成功。用真实 SQLite saver 编译最小多节点图，中断后关闭并重新打开 saver，断言已完成节点计数仍为 1、pending 节点继续、终态 checkpoint 与 SQL 终态间崩溃不重复图效果。snapshot 的 run/user/config 身份不符时拒绝复用。
- [ ] **Step 2: 运行红测。** `python -m pytest tests/unit/runtime/test_worker_lifecycle.py tests/unit/runtime/test_query_worker_entrypoint.py tests/unit/ingestion/test_worker.py tests/integration/runtime/test_query_checkpoint_resume.py -q`。不能仅验证 checkpoint thread ID 相同。
- [ ] **Step 3: 实现生命周期接线。** 启动后标 starting；模型、converter、checkpoint、存储和 Mem0 初始化均通过后标 ready；5 秒更新一次进程心跳，检测过期及旧实例记录。停止事件与当前任务完成并行等待，Query 在每条消息前检查 stop，停止后不领取新 batch；取消时保留现有 lease fencing、PEL 与 Outbox 语义。模型/解析原生线程不可取消时，由容器 60 秒退出上界终止进程，不能声称 asyncio.cancel 已终止 OCR。
- [ ] **Step 4: 按失败用例补齐恢复选择并复测。** 现有 Query 路径总是传入新 state，不能假设自动续跑；包装器借鉴现有 IngestionGraph 的 aget_state/resume 模式，仅在身份匹配时使用 pending checkpoint。重跑 Step 2、`tests/integration/runtime/test_query_worker.py` 和 `tests/e2e/test_ingestion_pipeline.py`，保留原有取消和重复投递契约。此处只修复部署验收所需恢复语义，不调整 RAG 节点策略。
- [ ] **Step 5: 提交 Worker 生命周期。** `git commit -m "fix: bound worker shutdown and verify durable checkpoint resume"`。

关键测试名与断言：`test_shutdown_does_not_start_second_delivery` 使用 `assert processed_ids == [first_id]`；`test_stale_worker_is_unready` 使用 `assert worker_is_ready(state_path) is False`；`test_query_reopens_checkpoint_without_replaying_completed_node` 使用 `assert completed_node_calls == 1` 和 `assert result['run_id'] == original_run_id`，沿用现有 QueryState 的顶层 run_id。

### Task 8: 接入完整 Compose 和幂等初始化

**Files:**

- Create: `compose.yaml`、`.env.docker.example`、`scripts/initialize_application.py`、`scripts/check_container_health.py`、`tests/unit/deployment/test_application_init.py`、`tests/unit/deployment/test_compose_contract.py`。
- Modify: `.gitignore`、`scripts/docker_stack.py`、`src/agentic_rag/api/health.py`、`tests/unit/api/test_health.py`。

**Interfaces:**

- `initialize_application(settings: Settings, *, mode: Literal['fresh', 'validate-existing']) -> None` 使用窄 MySQL/ES client，不调用 build_container；只有 fresh 可执行 Alembic/创建索引。
- Compose 共用任务 5 的 infra 定义：`docker compose -f compose.infra.yaml -f compose.yaml ...`；auth 配置不能自动带起 full 服务。
- 容器路径固定：`/data/artifacts`、`/data/checkpoints/query.sqlite`、`/data/checkpoints/ingestion.sqlite`、`/data/mem0/history.db`、`/data/tools/invocations.sqlite`、`/models`、`/run/agentic-rag`；每类状态为独立项目卷或独立子目录，不共享宿主状态。

- [ ] **Step 1: 写初始化和 Compose 红测。** fresh 两次执行不改变已存文档/记忆；validate-existing 缺 SQL schema、索引、alias 或 history 时零创建并失败；错误 alias 不自动切换。API/Worker 均 UID 10001，artifacts 所有者匹配且 mode 0700；SQLite 挂目录而非文件；app-init 成功前 API/Worker 不启动，model-init 成功前 Worker 不 ready。Secret 和管理凭证的挂载按服务明确列出。
- [ ] **Step 2: 运行红测。** `python -m pytest tests/unit/deployment/test_application_init.py tests/unit/deployment/test_compose_contract.py tests/unit/api/test_health.py -q`。
- [ ] **Step 3: 实现新安装与只读已有状态检查。** fresh 使用显式的独立 MySQL DSN 跑 Alembic 到 head，复用 ChildIndexStore.ensure_index/ensure_active_alias 并验证最终 alias；初始化 Mem0 正式及 entities 索引，持久化 history schema。validate-existing 在任何 Mem0 构造之前检查全部必要状态，只用 validate_index 和只读 alias 查询。不能把 ensure_active_alias 返回 False 单独视为失败或成功；数据库迁移 head 与 chat schema 都检查。全栈恢复数据导入仍不在本任务范围。
- [ ] **Step 4: 实现运行与健康检查并复测。** API 绑定容器 `0.0.0.0:8000`，仅宿主 loopback 随机端口；query/ingestion 各一份，`init: true` 和 60 秒 stop grace。ES/Redis/MySQL 无宿主端口；Redis AOF。API health 不把配置有模型名当成 Worker 模型已加载；Worker probe 检查本容器 ready/heartbeat，并做轻量存储联通检查，不重复载入模型。Mem0 降级必须使 readiness 不通过。重跑 Step 2，执行 `up --stage full`，记录三个业务进程真实就绪后才进入任务 9。
- [ ] **Step 5: 提交完整编排。** `git commit -m "feat: compose application services with idempotent initialization"`。`.env.docker.example` 只含非敏感示例和 Secret 文件路径；不生成或提交 `.env.local` 改动。

关键测试名与断言：`test_validate_existing_never_creates_missing_state` 使用 `assert write_calls == []`；`test_fresh_init_is_idempotent` 使用 `assert second_document_ids == first_document_ids`；`test_ready_rejects_degraded_memory` 使用 `assert readiness['memory'] == 'unavailable'`。`.gitignore` 必须显式允许跟踪 `.env.docker.example`，不放开真实环境文件。

### Task 9: 完整链路验收与失败回退演练

**Files:**

- Create: `tests/integration/deployment/conftest.py`、`tests/e2e/test_docker_application.py`、`tests/e2e/test_docker_recovery.py`、`tests/e2e/test_docker_rollback.py`、`scripts/verify_docker_deployment.py`、`docs/docker-deployment.md`。
- Modify: `scripts/docker_stack.py`，只补充完整验收及报告收集。

**Interfaces:**

- 所有真实部署测试必须给 `AGENTIC_RAG_DOCKER_TEST_MANIFEST`，fixture 从该文件和已验证 Docker 身份获得端点，不读取当前业务 `.env.local`；缺失时明确 skip，验收汇总将任何必测 skip 判为未通过。
- `python -m scripts.verify_docker_deployment --manifest ABS` 收集配置、镜像和模型版本、用例结果、日志脱敏检查、重启前后数据 ID、回滚证据，生成 `report.json` 和 `report.md`，失败返回非零。

- [ ] **Step 1: 写端到端失败用例。** 合成文本 PDF、扫描 PDF、XLSX 经公开上传接口入库并经真实 Query API 问答；断言父子数据存在、1024 维、`.zh` analyzer 正确、中文/英文检索命中目标、Mem0 用户隔离。可复用旧测试的合成文件生成器，但不能复用会连接宿主服务或使用固定 Embedding/伪 reranker 的 fixture。
- [ ] **Step 2: 运行红测并核对前提。** `AGENTIC_RAG_DOCKER_TEST_MANIFEST=ABS python -m pytest tests/e2e/test_docker_application.py tests/e2e/test_docker_recovery.py tests/e2e/test_docker_rollback.py -q`；失败必须可定位到待完成验收链路，任何测试都不能落到业务/评测端点。
- [ ] **Step 3: 补齐真实中断与持久化证据。** 先证明有非终态 run/job、SQL lease、Redis PEL 和持久化 checkpoint，再终止并重建相应 Worker，复用卷，等待 lease/reclaim 条件而非固定 sleep。断言任务 ID 不变、claim generation 增长、终态才 ACK、只有一份 publication/answer/turn/记忆效果。至少覆盖 Query、Ingestion，以及终态写入和 ACK 之间的中断。保留文档/记忆 ID 与内容哈希，另重建整个业务容器和 ES 后验证原 ID 仍可读，不能用重启后的新上传替代。
- [ ] **Step 4: 执行失败回退演练并复测。** 在本次栈内用错误凭证启动探针，确认认证失败、业务未就绪，然后 `down --manifest ABS`；核对源工作区、原配置、业务/评测服务身份及原先健康端点，确认卷和脱敏日志保留。故意篡改 manifest token 的清理请求必须拒绝。重新创建本次栈、恢复正确 Secret 后，前述数据仍在；重跑 Step 2 和任务 5 的完整认证测试。全量 unit、Ruff、mypy 按仓库基线命令通过；禁止运行会默认写当前业务库的其他 live fixture。
- [ ] **Step 5: 生成报告和部署文档并提交。** 文档给出 Secret 准备、一次性 model-init、组合 Compose 启动、日志/健康检查、安全停止、保留卷重建的精确命令，说明宿主端口与容器服务名的区别。分开报告 arm64/amd64 实测状态，不声称任何未运行测试通过。明确本次回退是撤掉隔离试验环境，未来真实切换必须另行备份和授权；不把现有 backup/restore 当作已完成的完整迁移工具。`git commit -m "test: prove Docker deployment recovery and rollback safety"`。

关键测试名与断言：`test_pending_job_recovers_after_worker_recreation` 使用 `assert final_task_id == initial_task_id`、`assert final_claim_generation > initial_claim_generation`、`assert pending_count == 0`、`assert published_result_count == 1`；`test_failed_experiment_preserves_source_environment` 使用 `assert source_config_hash_after == source_config_hash_before`、`assert original_service_identity_after == original_service_identity_before`。这不是断言真实业务数据在测试期间没有正常增长。

## 最终复核和交付

- [ ] 对照 Spec 的每一项验收门槛核对 report；所有必测项通过且无必测 skip，才标记隔离容器化完成。
- [ ] 在独立分支进行一次完整代码审查；重点检查 Secret 泄漏、初始化顺序、角色权限、恢复身份隔离和清理资源归属。
- [ ] 向用户交付实现分支、启动文档、版本锁、测试报告及失败回退命令，说明现有业务仍未切换；不自动合并、推送、迁移或删除测试数据卷。
- [ ] 如果环境资源、精确版本镜像或某架构不可用，保存已有实现与证据，报告具体未通过项和后续选择；不能用跳过测试来达成验收。

## 参考

- [Compose 项目名与隔离](https://docs.docker.com/compose/how-tos/project-name/)。
- [Compose down 默认保留命名数据卷](https://docs.docker.com/reference/cli/docker/compose/down/)。
- [ES 8.19 权限定义](https://www.elastic.co/guide/en/elasticsearch/reference/8.19/security-privileges.html)。
