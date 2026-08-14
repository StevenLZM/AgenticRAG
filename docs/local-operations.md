# Local operations

These procedures operate only on this local V1 installation. Load local configuration before every command:

```sh
set -a; source .env.local; set +a
```

## Start and upgrade

Start dependencies in this order: Elasticsearch, MySQL, then Redis. Confirm all three are local endpoints, then apply migrations and start processes in the following order:

```sh
conda run -n agentic-rag python scripts/check_local_dependencies.py
conda run -n agentic-rag alembic upgrade head
conda run -n agentic-rag python scripts/run_api.py --grace-seconds 30
conda run -n agentic-rag python scripts/run_query_worker.py
conda run -n agentic-rag python scripts/run_ingestion_worker.py
```

The API is live at `/health/live`; use `/health/ready` only after all dependencies report `available`. For an upgrade, stop workers first, let the current graph node reach its SQLite checkpoint, apply the migration, and restart in the same order.

## Mem0 long-term memory

Mem0 is disabled by default so a local Query API can start without the optional
provider. Enable it only after installing `mem0ai==2.0.12` in the `agentic-rag`
Conda environment and supplying an Elasticsearch authentication method. The
application owns the user namespace, calls Mem0 with `infer=False`, and uses
the light-model extractor before writing durable facts:

```dotenv
AGENTIC_RAG_MEM0_ENABLED=1
AGENTIC_RAG_MEM0_COLLECTION=agent_memories_v1
AGENTIC_RAG_MEM0_EMBEDDING_BASE_URL=https://<qwen-endpoint>/v1
AGENTIC_RAG_MEM0_EMBEDDING_API_KEY=<qwen-key>
AGENTIC_RAG_MEM0_EMBEDDING_MODEL=text-embedding-v3
AGENTIC_RAG_MEM0_ELASTICSEARCH_API_KEY=<es-api-key>
AGENTIC_RAG_MEM0_HISTORY_DB_PATH=var/mem0/history.db
```

For a Mem0-managed LLM (normally unnecessary because extraction remains an
application ModelGateway call), set `AGENTIC_RAG_MEM0_LLM_ENABLED=1` together
with `AGENTIC_RAG_MEM0_LLM_MODEL`, `AGENTIC_RAG_MEM0_LLM_BASE_URL`, and
`AGENTIC_RAG_MEM0_LLM_API_KEY`. If Mem0 is enabled but its configuration or
provider construction fails, the API keeps memory degraded, `/health/ready`
reports `memory=unavailable`, and a bounded `memory_provider_degraded` log is
emitted; query evidence and tenant isolation do not silently broaden.

Run the real provider contract only with an explicit disposable namespace. It
skips when variables are absent and fails when a configured provider is
unhealthy:

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

## Graceful stop and recovery work

Send `SIGTERM` to the API and workers. The API immediately refuses new work and allows in-flight HTTP work to drain for the configured grace period. Worker graph state is stored in the configured query and ingestion SQLite checkpoint files; restarting the one-worker processes resumes recoverable work. Do not kill or copy an open checkpoint database.

Review quarantined versions before any retry:

```sh
conda run -n agentic-rag python scripts/review_quarantined_version.py --help
```

Inspect the Redis dead stream and retry only a failed Run/Job after its cause is fixed. Retries are idempotent: never manually acknowledge a pending message merely to clear it, and do not retry a completed Run/Job.

## Backup

Stop API and workers cleanly before taking a backup. The default command backs up only local SQLite checkpoints and Artifacts; it refuses an existing output path, checkpoints SQLite WAL state through SQLite's backup API, records app/schema/index generations, writes a canonical hash manifest, and verifies every file hash. The manifest is integrity-hashed, not a cryptographic signature; store backups on trusted/permissioned media.

```sh
conda run -n agentic-rag python scripts/backup_local.py --output var/backups/backup-001
```

To include the configured MySQL database and active Elasticsearch generation, use the explicit service flag. This performs a `mysqldump --single-transaction` and an Elasticsearch export of the controlled index, aliases, and templates.

```sh
conda run -n agentic-rag python scripts/backup_local.py --output var/backups/backup-001 --include-services
```

To include Redis, supply one explicit key namespace. The backup never scans
the whole configured Redis database:

```sh
conda run -n agentic-rag python scripts/backup_local.py \
  --output var/backups/backup-001 --include-services \
  --redis-key-prefix 'agentic-rag:backup:'
```

Keep at least three verified backups on separate local media. Test each backup with a restore drill before deleting an older backup. The backup directory is immutable operational evidence: never edit its manifest, hash file, dump, or exports.

## Restore and Elasticsearch rollback

Restore only into a path that does not yet exist. The restore verifies the hash manifest and every content hash before it creates the target and publishes the target atomically. It never replaces an existing directory.

```sh
conda run -n agentic-rag python scripts/restore_local.py \
  --backup var/backups/backup-001 --target var/restore-drill/backup-001
```

A backup that includes service state requires deliberately supplied, empty service targets. The MySQL target must be a separate empty database; Elasticsearch must use a fresh index generation. The command refuses a non-empty MySQL database or existing target index, imports the dump, runs Alembic migrations, restores and verifies the target index/template/alias, and leaves the active production generation unchanged.

```sh
conda run -n agentic-rag python scripts/restore_local.py \
  --backup var/backups/backup-001 --target var/restore-drill/backup-001 \
  --mysql-dsn 'mysql+asyncmy://.../agentic_rag_restore_001' \
  --elasticsearch-url http://127.0.0.1:9200 --index-generation restore-001 \
  --redis-dsn 'redis://127.0.0.1:6379/15' \
  --redis-key-prefix 'agentic-rag:restore:'
```

When Redis is present in the backup, restore requires a distinct explicit
target prefix. Keys are restored by remapping the source prefix to that target;
the target namespace must be empty and is scanned before any write.

To roll back search, point the controlled active alias to the previous verified generation only after checking that generation's mapping and document count. Do not delete the current index until the rollback has passed readiness and a query smoke test.

### Disposable real-service restore drill

The real-service test never guesses an admin DSN and never uses the configured
`agentic_rag` database. Set an explicit MySQL admin DSN (with permission to
create/drop only disposable test databases), a local Redis database, and the
local Elasticsearch endpoint. Do not put the admin password in shell history
or commit it:

```sh
set -a; source .env.local; set +a
export AGENTIC_RAG_RUN_REAL_BACKUP_RESTORE=1
export AGENTIC_RAG_TEST_MYSQL_ADMIN_DSN='mysql+asyncmy://<admin>:<password>@127.0.0.1:3306/mysql'
export AGENTIC_RAG_TEST_REDIS_DSN='redis://127.0.0.1:6379/15'
export AGENTIC_RAG_TEST_ELASTICSEARCH_URL='http://127.0.0.1:9200'
conda run -n agentic-rag python -m pytest --import-mode=importlib \
  tests/e2e/test_backup_restore.py -q -s
```

The fixture generates random `agentic_rag_backup_*` and
`agentic_rag_restore_*` databases, one generation/alias, and two Redis key
prefixes for source-to-target remapping, then removes them in `finally`. If any
explicit variable is missing, the real test skips rather than touching an
inferred service target.

## Final gate

Run the final acceptance sequence with explicit opt-in infrastructure/model tests. `live_model` has no credential skip: selecting it with missing credentials is a failure. The real backup test requires generated isolated targets and must be enabled separately.

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
# Final acceptance must use a runtime-snapshot-matched dataset and a real client:
conda run -n agentic-rag python -m evals.run --mode graph \
  --dataset var/artifacts/evals/runtime-baseline.jsonl --output var/artifacts/evals/graph
# Or evaluate a deployed API (set the snapshot ID used by the dataset):
AGENTIC_RAG_EVAL_SNAPSHOT_ID='<runtime snapshot id>' \
conda run -n agentic-rag python -m evals.run --mode api \
  --base-url http://127.0.0.1:8000 \
  --dataset var/artifacts/evals/runtime-baseline.jsonl \
  --output var/artifacts/evals/api
conda run -n agentic-rag python scripts/verify_acceptance.py \
  --report var/artifacts/evals/graph/summary.json
```

The default `fixture` mode is an offline smoke test only and prints `SMOKE
ONLY`; it can never satisfy final acceptance. Graph/API mode persists the real
client provenance and rejects a case whose runtime snapshot differs from the
composed QueryGraph/API Run. Prepare `runtime-baseline.jsonl` from the seeded
documents and the current `RuntimeConfigSnapshot` rather than changing a
dataset's snapshot ID after the fact. Run `verify_acceptance.py` against the
Graph/API summary (not the fixture summary). The verifier returns failure
unless leakage is zero, citation coverage is exactly 1.0, unaudited answers
are zero, recovery and backup/restore drills pass, and `real_query_count` is
positive. Ragas remains explicitly `unavailable` when no backend is
configured; it never fabricates a score.
