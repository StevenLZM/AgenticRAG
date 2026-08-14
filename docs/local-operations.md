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
conda run -n agentic-rag python -m evals.run --dataset evals/datasets/baseline.jsonl --output var/artifacts/evals/final
conda run -n agentic-rag python scripts/verify_acceptance.py --report var/artifacts/evals/final/summary.json
```

`evals.run` executes the deterministic 24-case fixture and isolated local
recovery/backup drills, so the final report contains all five hard-gate fields.
The acceptance command returns failure unless leakage is zero, citation
coverage is exactly 1.0, unaudited answers are zero, and both recovery and
backup/restore drills pass. Ragas remains explicitly `unavailable` when no
backend is configured; it never fabricates a score.
