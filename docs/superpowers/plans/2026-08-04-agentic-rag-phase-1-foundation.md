# Agentic RAG Phase 1 Foundation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Establish the executable Python project, typed cross-phase contracts, durable persistence primitives and local service bootstrap required by all later phases.

**Architecture:** Use a `src` package, Pydantic contracts and small async Port/Adapter boundaries. MySQL is the durable state authority, Redis carries duplicate-safe notifications, SQLite persists LangGraph checkpoints, and the local Artifact Store owns large immutable payloads.

**Tech Stack:** Python 3.11, Pydantic 2, FastAPI, SQLAlchemy 2 async, Alembic, asyncmy, redis-py asyncio, Elasticsearch async client, LangGraph SQLite saver, pytest, Ruff, mypy.

## Global Constraints

- Preserve the package boundary `src/agentic_rag`; do not create framework, plugin or DDD layers beyond the approved design.
- All repository and broker operations are async; pure domain functions remain synchronous.
- Generate IDs in application code with UUID7-compatible sortable strings or deterministic SHA-256 hashes where the design requires idempotency.
- Never put clients, sessions, embeddings, model objects or large Parent text in Graph state.
- Local integration tests are marked `integration`; the default unit suite uses fakes and temporary files.

---

### Task 1: Python Package, Tooling and Settings

**Files:**
- Create: `pyproject.toml`
- Create: `src/agentic_rag/__init__.py`
- Create: `src/agentic_rag/api/__init__.py`
- Create: `src/agentic_rag/domain/__init__.py`
- Create: `src/agentic_rag/runtime/__init__.py`
- Create: `src/agentic_rag/persistence/__init__.py`
- Create: `src/agentic_rag/config.py`
- Create: `tests/unit/test_config.py`
- Create: `tests/conftest.py`

**Interfaces:**
- Produces: `Settings`, `get_settings() -> Settings`, pytest markers `integration` and `e2e`.
- Consumes: Environment variables prefixed with `AGENTIC_RAG_`.

- [ ] **Step 1: Write the failing settings test**

```python
def test_settings_use_local_defaults(monkeypatch):
    monkeypatch.setenv("AGENTIC_RAG_MYSQL_DSN", "mysql+asyncmy://rag:rag@127.0.0.1/rag")
    monkeypatch.setenv("AGENTIC_RAG_DEEPSEEK_BASE_URL", "https://models.example.invalid/v1")
    monkeypatch.setenv("AGENTIC_RAG_QWEN_EMBEDDING_BASE_URL", "https://embeddings.example.invalid/v1")
    settings = Settings()
    assert settings.redis_url == "redis://127.0.0.1:6379/0"
    assert settings.elasticsearch_url == "http://localhost:9200"
    assert settings.embedding_dimensions == 1024
    assert settings.query_worker_count == 1
    assert settings.ingestion_worker_count == 1
```

- [ ] **Step 2: Run the test and verify the missing package failure**

Run: `pytest tests/unit/test_config.py -q`
Expected: FAIL because `agentic_rag.config` does not exist.

- [ ] **Step 3: Create packaging, dependencies and the typed settings object**

```toml
[project]
name = "agentic-rag"
version = "0.1.0"
requires-python = ">=3.11,<3.13"
dependencies = [
  "alembic",
  "asyncmy",
  "docling",
  "elasticsearch[async]",
  "fastapi",
  "langgraph",
  "langgraph-checkpoint-sqlite",
  "mem0ai==2.0.12",
  "openai",
  "orjson",
  "pydantic>=2",
  "pydantic-settings",
  "redis",
  "sentence-transformers",
  "sqlalchemy[asyncio]>=2",
  "tenacity",
  "uvicorn",
  "uuid6"
]

[project.optional-dependencies]
dev = ["httpx", "mypy", "pytest", "pytest-asyncio", "ruff"]
eval = ["datasets", "ragas"]

[tool.pytest.ini_options]
asyncio_mode = "auto"
markers = [
  "integration: requires local infrastructure",
  "e2e: full system scenario",
  "live_model: calls configured external model APIs"
]
```

```python
class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="AGENTIC_RAG_", extra="forbid")
    mysql_dsn: str
    redis_url: str = "redis://127.0.0.1:6379/0"
    elasticsearch_url: str = "http://localhost:9200"
    deepseek_base_url: str
    deepseek_api_key: SecretStr | None = None
    qwen_embedding_base_url: str
    qwen_api_key: SecretStr | None = None
    default_user_id: str = "default_user"
    main_model: str = "deepseek-v4-pro"
    light_model: str = "deepseek-v4-flash"
    embedding_model: str = "text-embedding-v3"
    embedding_dimensions: int = 1024
    reranker_model: str = "BAAI/bge-reranker-v2-m3"
    max_concurrent_query_runs: int = 4
    max_concurrent_llm_calls: int = 8
    max_concurrent_reranks: int = 1
    max_parallel_subagents_per_run: int = 3
    max_research_rounds: int = 4
    max_answer_revisions: int = 1
    query_run_timeout_seconds: int = 300
    max_evidence_tokens: int = 12_000
    research_context_soft_limit_tokens: int = 16_000
    query_worker_count: Literal[1] = 1
    ingestion_worker_count: Literal[1] = 1
    query_checkpoint_path: Path = Path("var/query_checkpoints.sqlite")
    ingestion_checkpoint_path: Path = Path("var/ingestion_checkpoints.sqlite")
    artifact_root: Path = Path("var/artifacts")
```

- [ ] **Step 4: Run static and unit checks**

Run: `ruff check src/agentic_rag/config.py tests/unit/test_config.py && mypy src/agentic_rag/config.py && pytest tests/unit/test_config.py -q`
Expected: all commands exit 0.

- [ ] **Step 5: Commit the project scaffold**

```bash
git add pyproject.toml src/agentic_rag tests
git commit -m "build: scaffold agentic rag package"
```

### Task 2: Shared Domain and Runtime Contracts

**Files:**
- Create: `src/agentic_rag/domain/models.py`
- Create: `src/agentic_rag/runtime/models.py`
- Create: `src/agentic_rag/runtime/ids.py`
- Create: `tests/unit/runtime/test_models.py`

**Interfaces:**
- Produces: `UserScope`, `RunStatus`, `JobStatus`, `DocumentStatus`, `DocumentVersionStatus`, `RuntimeConfigSnapshot`, `new_id()`, `content_id(*parts)`.
- Consumes: `Settings` from Task 1.

- [ ] **Step 1: Write failing tests for immutable scope and deterministic snapshots**

```python
def test_runtime_snapshot_id_is_content_addressed():
    left = RuntimeConfigSnapshot(**SNAPSHOT_DATA)
    right = RuntimeConfigSnapshot.model_validate(left.model_dump())
    assert left.snapshot_id == right.snapshot_id

def test_user_scope_rejects_blank_user():
    with pytest.raises(ValidationError):
        UserScope(user_id=" ")
```

- [ ] **Step 2: Verify red state**

Run: `pytest tests/unit/runtime/test_models.py -q`
Expected: FAIL because the contract modules do not exist.

- [ ] **Step 3: Implement exact enums and immutable models**

```python
class RunStatus(StrEnum):
    QUEUED = "queued"
    RUNNING = "running"
    CANCEL_REQUESTED = "cancel_requested"
    CANCELLED = "cancelled"
    COMPLETED = "completed"
    FAILED = "failed"

class JobStatus(StrEnum):
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    QUARANTINED = "quarantined"
    FAILED = "failed"

class DocumentStatus(StrEnum):
    PROCESSING = "processing"
    ACTIVE = "active"
    FAILED = "failed"
    DELETED = "deleted"

class DocumentVersionStatus(StrEnum):
    UPLOADED = "uploaded"
    BUILDING = "building"
    ACTIVE = "active"
    QUARANTINED = "quarantined"
    FAILED = "failed"
    INACTIVE = "inactive"

class UserScope(BaseModel):
    model_config = ConfigDict(frozen=True)
    user_id: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]

class RuntimeConfigSnapshot(BaseModel):
    model_config = ConfigDict(frozen=True)
    app_version: str
    graph_version: str
    prompt_version: str
    main_model_id: str
    light_model_id: str
    embedding_model: str
    embedding_dimensions: Literal[1024]
    reranker_version: str
    retrieval_config_version: str
    index_generation: str
    memory_config_version: str
    max_research_rounds: int = Field(default=4, ge=1, le=4)
    max_answer_revisions: int = Field(default=1, ge=0, le=1)
    query_run_timeout_seconds: int = Field(default=300, ge=30, le=300)
    max_evidence_tokens: int = Field(default=12_000, ge=1_000, le=12_000)
    research_context_soft_limit_tokens: int = Field(default=16_000, ge=4_000)
    max_parallel_subagents_per_run: int = Field(default=3, ge=1, le=3)

    @property
    def snapshot_id(self) -> str:
        payload = self.model_dump_json(exclude_none=True)
        return hashlib.sha256(payload.encode()).hexdigest()

def new_id() -> str:
    return str(uuid7())

def content_id(*parts: str) -> str:
    digest = hashlib.sha256()
    for part in parts:
        encoded = part.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
    return digest.hexdigest()
```

Use `content_id(*parts)` for the design's ingestion, Parent, Child and Event idempotency keys; never use ambiguous string concatenation.

- [ ] **Step 4: Run contract tests and type checks**

Run: `pytest tests/unit/runtime/test_models.py -q && mypy src/agentic_rag/domain src/agentic_rag/runtime`
Expected: PASS.

- [ ] **Step 5: Commit contracts**

```bash
git add src/agentic_rag/domain src/agentic_rag/runtime tests/unit/runtime
git commit -m "feat: define shared runtime contracts"
```

### Task 3: MySQL Schema and Repository Ports

**Files:**
- Create: `alembic.ini`
- Create: `alembic/env.py`
- Create: `alembic/versions/0001_initial_schema.py`
- Create: `src/agentic_rag/persistence/mysql.py`
- Create: `src/agentic_rag/persistence/repositories.py`
- Create: `tests/integration/persistence/test_mysql_schema.py`
- Create: `tests/unit/persistence/test_repository_contracts.py`

**Interfaces:**
- Produces: `RunRepository`, `IngestionJobRepository`, `DocumentRepository`, `ParentRepository`, `EventRepository`, `OutboxRepository`, `MemoryTombstoneRepository` protocols and SQLAlchemy adapters.
- Consumes: IDs and status enums from Task 2.

- [ ] **Step 1: Write repository contract and active-run exclusivity tests**

```python
@pytest.mark.integration
async def test_only_one_active_run_per_user_thread(run_repo):
    first = await run_repo.create_queued(scope=UserScope(user_id="u1"), thread_id="t1", snapshot=SNAPSHOT)
    with pytest.raises(ActiveRunConflict):
        await run_repo.create_queued(scope=UserScope(user_id="u1"), thread_id="t1", snapshot=SNAPSHOT)
    await run_repo.mark_completed(first.id, result_ref="artifact://answer")
    second = await run_repo.create_queued(scope=UserScope(user_id="u1"), thread_id="t1", snapshot=SNAPSHOT)
    assert second.id != first.id
```

- [ ] **Step 2: Verify migration and repository tests fail**

Run: `pytest tests/unit/persistence tests/integration/persistence/test_mysql_schema.py -q`
Expected: FAIL because migrations and adapters are absent.

- [ ] **Step 3: Implement the approved schema and transactional repository methods**

The migration must create exactly these tables with foreign keys and indexes from the design: `documents`, `document_versions`, `parent_chunks`, `ingestion_jobs`, `agent_runs`, `task_outbox`, `messages`, `agent_events`, `memory_tombstones`.

```python
class RunRepository(Protocol):
    async def create_queued(self, scope: UserScope, thread_id: str, snapshot: RuntimeConfigSnapshot) -> QueryRun: ...
    async def claim(self, run_id: str, owner: str, lease_seconds: int) -> QueryRun | None: ...
    async def heartbeat(self, run_id: str, owner: str, lease_seconds: int) -> None: ...
    async def request_cancel(self, run_id: str, scope: UserScope) -> RunStatus: ...
    async def finish(self, run_id: str, status: RunStatus, result_ref: str | None, error_code: str | None) -> None: ...

class EventRepository(Protocol):
    async def append(self, event: AgentEvent) -> int: ...
    async def list_after(self, run_id: str, scope: UserScope, after_id: int, limit: int) -> list[AgentEvent]: ...
```

Use `active_slot=1` for active Run states and `NULL` for terminal states, with unique key `(user_id, thread_id, active_slot)`. Repository methods that create a Run/Job must accept an SQLAlchemy transaction so the matching Outbox record is inserted atomically.

- [ ] **Step 4: Apply, downgrade and reapply the migration, then run tests**

Run: `alembic upgrade head && pytest tests/integration/persistence/test_mysql_schema.py -q && alembic downgrade base && alembic upgrade head && pytest tests/unit/persistence -q`
Expected: every command exits 0; all required indexes and tables exist after reapply.

- [ ] **Step 5: Commit persistence schema**

```bash
git add alembic.ini alembic src/agentic_rag/persistence tests/unit/persistence tests/integration/persistence
git commit -m "feat: add durable mysql repositories"
```

### Task 4: Outbox Dispatcher and Redis Stream Broker

**Files:**
- Create: `src/agentic_rag/persistence/outbox.py`
- Create: `src/agentic_rag/persistence/redis_queue.py`
- Create: `tests/unit/persistence/test_outbox_dispatcher.py`
- Create: `tests/integration/persistence/test_redis_streams.py`

**Interfaces:**
- Produces: `StreamBroker.publish()`, `StreamBroker.consume()`, `StreamBroker.ack()`, `StreamBroker.reclaim()`, `OutboxDispatcher.dispatch_once()`.
- Consumes: `OutboxRepository` from Task 3.

- [ ] **Step 1: Write a duplicate-safe dispatcher test**

```python
async def test_dispatcher_marks_outbox_only_after_publish(fake_outbox, fake_broker):
    row = fake_outbox.pending("query_run", "run-1", "agenticrag:jobs:query")
    fake_broker.fail_once = True
    assert await OutboxDispatcher(fake_outbox, fake_broker).dispatch_once() == 0
    assert row.status == "pending"
    assert await OutboxDispatcher(fake_outbox, fake_broker).dispatch_once() == 1
    assert row.status == "dispatched"
```

- [ ] **Step 2: Verify the test fails**

Run: `pytest tests/unit/persistence/test_outbox_dispatcher.py -q`
Expected: FAIL because dispatcher and broker do not exist.

- [ ] **Step 3: Implement the thin broker and dispatcher**

```python
class StreamBroker(Protocol):
    async def publish(self, stream: str, aggregate_id: str, enqueued_at: datetime) -> str: ...
    async def consume(self, stream: str, group: str, consumer: str, block_ms: int) -> list[StreamMessage]: ...
    async def ack(self, stream: str, group: str, message_id: str) -> None: ...
    async def reclaim(self, stream: str, group: str, consumer: str, min_idle_ms: int) -> list[StreamMessage]: ...
    async def dead_letter(self, dead_stream: str, message: StreamMessage, reason: str) -> None: ...

class OutboxDispatcher:
    async def dispatch_once(self, limit: int = 100) -> int:
        rows = await self.outbox.claim_pending(limit=limit)
        dispatched = 0
        for row in rows:
            try:
                await self.broker.publish(row.stream_name, row.aggregate_id, row.created_at)
            except RedisError:
                await self.outbox.schedule_retry(row.id)
            else:
                await self.outbox.mark_dispatched(row.id)
                dispatched += 1
        return dispatched
```

- [ ] **Step 4: Run unit and local Redis integration tests**

Run: `pytest tests/unit/persistence/test_outbox_dispatcher.py -q && pytest -m integration tests/integration/persistence/test_redis_streams.py -q`
Expected: duplicate delivery returns the same aggregate ID and can be safely acknowledged after repository Claim.

- [ ] **Step 5: Commit queue primitives**

```bash
git add src/agentic_rag/persistence/outbox.py src/agentic_rag/persistence/redis_queue.py tests
git commit -m "feat: add redis outbox delivery"
```

### Task 5: SQLite Checkpoint and Local Artifact Store

**Files:**
- Create: `src/agentic_rag/persistence/checkpoint.py`
- Create: `src/agentic_rag/persistence/artifacts.py`
- Create: `tests/unit/persistence/test_artifacts.py`
- Create: `tests/integration/persistence/test_sqlite_checkpoint.py`

**Interfaces:**
- Produces: `CheckpointBackend.open_query()`, `CheckpointBackend.open_ingestion()`, `ArtifactStore.put_json()`, `put_bytes()`, `read_json()`, `verify()`.
- Consumes: Checkpoint and Artifact paths from `Settings`.

- [ ] **Step 1: Write atomic artifact and checkpoint replay tests**

```python
def test_artifact_write_is_content_verified(tmp_path):
    store = LocalArtifactStore(tmp_path)
    ref = store.put_json("runs/r1/evidence.json", {"evidence_ids": ["e1"]})
    assert store.verify(ref)
    assert store.read_json(ref) == {"evidence_ids": ["e1"]}
```

- [ ] **Step 2: Verify red state**

Run: `pytest tests/unit/persistence/test_artifacts.py tests/integration/persistence/test_sqlite_checkpoint.py -q`
Expected: FAIL because the adapters do not exist.

- [ ] **Step 3: Implement atomic files and single-writer SQLite configuration**

```python
@dataclass(frozen=True)
class ArtifactRef:
    uri: str
    sha256: str
    size_bytes: int

class ArtifactStore(Protocol):
    def put_json(self, relative_path: str, value: Mapping[str, Any]) -> ArtifactRef: ...
    def put_bytes(self, relative_path: str, value: bytes) -> ArtifactRef: ...
    def read_json(self, ref: ArtifactRef) -> Any: ...
    def verify(self, ref: ArtifactRef) -> bool: ...
```

Write to a sibling temporary file, `fsync`, verify SHA-256, then `os.replace` into the versioned destination. The Checkpoint adapter must create parent directories, enable WAL and `busy_timeout`, expose separate query/ingestion savers, and reject configured worker counts other than one.

- [ ] **Step 4: Run persistence tests**

Run: `pytest tests/unit/persistence/test_artifacts.py -q && pytest -m integration tests/integration/persistence/test_sqlite_checkpoint.py -q`
Expected: atomic replacement, Hash mismatch detection and Graph checkpoint resume all pass.

- [ ] **Step 5: Commit storage adapters**

```bash
git add src/agentic_rag/persistence/checkpoint.py src/agentic_rag/persistence/artifacts.py tests
git commit -m "feat: add checkpoint and artifact storage"
```

### Task 6: Application Bootstrap and Health Contracts

**Files:**
- Create: `src/agentic_rag/api/app.py`
- Create: `src/agentic_rag/api/health.py`
- Create: `src/agentic_rag/api/errors.py`
- Create: `src/agentic_rag/bootstrap.py`
- Create: `tests/unit/api/test_health.py`
- Create: `scripts/check_local_dependencies.py`

**Interfaces:**
- Produces: `create_app(settings: Settings) -> FastAPI`, `/health/live`, `/health/ready`, `build_container(settings) -> AppContainer`, unified `ApiError` response handling.
- Consumes: MySQL engine, Redis broker, ES client, Checkpoint backend and Artifact Store from Tasks 3–5.

- [ ] **Step 1: Write liveness and readiness tests**

```python
async def test_readiness_reports_failed_dependency(app_client, fake_dependencies):
    fake_dependencies.elasticsearch.healthy = False
    response = await app_client.get("/health/ready")
    assert response.status_code == 503
    assert response.json()["dependencies"]["elasticsearch"] == "unavailable"
```

- [ ] **Step 2: Verify red state**

Run: `pytest tests/unit/api/test_health.py -q`
Expected: FAIL because the application factory is absent.

- [ ] **Step 3: Implement explicit bootstrap and health behavior**

`/health/live` returns 200 if the event loop serves the request. `/health/ready` checks configuration including required model credentials, MySQL `SELECT 1`, Redis `PING`, ES cluster reachability, Artifact write/read/delete in a health namespace, Checkpoint initialization and Reranker initialization flag. It must not call a paid model API.

```python
class ApiError(BaseModel):
    error_code: str
    message: str
    retryable: bool
    trace_id: str
    degraded_components: tuple[str, ...] = ()
```

Map validation/scope/format errors to non-retryable 4xx responses, active-Run conflict to 409, and temporary dependency failures to retryable 503. Never return provider exceptions or stack traces to clients.

```python
@dataclass
class AppContainer:
    settings: Settings
    repositories: Repositories
    broker: StreamBroker
    artifacts: ArtifactStore
    checkpoints: CheckpointBackend
    elasticsearch: AsyncElasticsearch

def create_app(settings: Settings) -> FastAPI:
    app = FastAPI(title="Agentic RAG", version="0.1.0")
    app.state.container = build_container(settings)
    app.include_router(health_router)
    return app
```

- [ ] **Step 4: Run Phase 1 verification**

Run: `ruff check src tests scripts && mypy src && pytest -m "not integration and not e2e and not live_model" -q && python scripts/check_local_dependencies.py`
Expected: static/unit checks pass; dependency script reports each configured service with a non-zero exit if any required service is unavailable.

- [ ] **Step 5: Commit bootstrap**

```bash
git add src/agentic_rag/api src/agentic_rag/bootstrap.py tests/unit/api scripts/check_local_dependencies.py
git commit -m "feat: add local application bootstrap"
```
