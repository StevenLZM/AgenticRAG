# Task 5 report: local operations, backup/restore, acceptance

## RED to GREEN evidence

RED was recorded before implementation:

```sh
conda run -n agentic-rag python -m pytest -m e2e tests/e2e/test_backup_restore.py -q
```

It failed during collection with `ModuleNotFoundError: No module named 'scripts.backup_local'`. The new local backup/restore test then passed after the deterministic hash-verified manifest implementation:

```sh
conda run -n agentic-rag python -m pytest --import-mode=importlib tests/e2e/test_backup_restore.py -q
# 14 passed, 1 skipped (before explicit real-service variables)
```

The readiness RED test failed with `AttributeError: 'ReadinessChecks' object has no attribute 'require_ready'`; it is green after adding fail-closed `require_ready()`:

```sh
conda run -n agentic-rag python -m pytest tests/unit/api/test_health.py -q
# 13 passed
```

The live-model RED test failed with `ModuleNotFoundError: No module named 'scripts.live_model_smoke'`; the selected provider gate is green:

```sh
conda run -n agentic-rag python -m pytest -m live_model tests/smoke -q
# 1 passed
```

The test made actual DeepSeek `deepseek-v4-flash` structured routing and `deepseek-v4-pro` completion calls and an actual Qwen `text-embedding-v3` call. It asserted provider-returned model IDs and exactly 1024 embedding dimensions. It has no credential skip when `live_model` is selected.

## Implementation

- `backup_local.py` writes a new directory atomically, SQLite-backups checkpoint files after WAL checkpointing, copies Artifact files without symlink traversal, records app/schema/index generations, and hashes a sorted file inventory. `manifest.sha256` is an integrity hash, not a cryptographic signature.
- Service backup is explicit: MySQL uses a consistent `mysqldump --single-transaction`; Elasticsearch exports the controlled generation plus aliases/templates and all documents in stable ID order; Redis exports only an explicit safe key prefix with DUMP/TTL data and supports source-to-target prefix remapping on restore.
- `restore_local.py` validates the manifest and every hash before creating a target, rejects existing targets, uses a staging directory and atomic publish, validates Artifact paths, and requires explicit service targets when a service export is present. MySQL restores only to a verified-empty database, imports the dump, and runs Alembic. Elasticsearch creates a fresh generation, restores documents/template/alias, and checks index count/mapping/aliases.
- `run_api.py` provides a Uvicorn launcher with bounded SIGTERM/SIGINT draining. Existing workers already use `stop_event` signal handling and durable SQLite checkpoints.
- `verify_acceptance.py` is strict: any missing, wrong-typed, or non-passing hard gate fails.
- `docs/local-operations.md` documents start/upgrade/stop/quarantine/dead-stream/retry/backup/retention/restore/rollback and final gates.

## Real-service evidence and explicit skips

```sh
conda run -n agentic-rag python scripts/check_local_dependencies.py
# configuration/mysql/redis/elasticsearch/artifacts/checkpoints/reranker: available
```

The isolated MySQL/Redis/Elasticsearch restore drill is opt-in and only uses generated `agentic_rag_backup_*` and `agentic_rag_restore_*` databases, a generated `agenticrag-children-e2e-*` index and two generated Redis prefixes for source-to-target remapping. It never connects to or deletes the configured `agentic_rag` application data.

```sh
AGENTIC_RAG_RUN_REAL_BACKUP_RESTORE=1 \
AGENTIC_RAG_TEST_MYSQL_ADMIN_DSN='mysql+asyncmy://<admin>:<password>@127.0.0.1:3306/mysql' \
AGENTIC_RAG_TEST_REDIS_DSN='redis://127.0.0.1:6379/15' \
AGENTIC_RAG_TEST_ELASTICSEARCH_URL='http://127.0.0.1:9200' \
  conda run -n agentic-rag python -m pytest --import-mode=importlib \
  tests/e2e/test_backup_restore.py -q -s
# 15 passed (4 prefix-safety checks + isolated MySQL/ES/Redis restore)
```

The real run created random disposable MySQL source/restore databases, applied
Alembic head, inserted and read a marker row, exported/restored one ES index
and alias with one document, and round-tripped one Redis key from a source
prefix into a distinct restore prefix. Cleanup
removed the generated databases, indices, and key. The configured
`agentic_rag` database and default Redis/ES namespaces were not touched.

An earlier direct attempt to create a temporary database through asyncmy's `mysql` system schema was rejected with MySQL 1045. The fixture no longer uses that path: it requires the explicit test admin DSN and invokes only the TCP MySQL CLI for generated test database creation/drop. Mem0 has no configured isolated test provider variable (`AGENTIC_RAG_TEST_MEM0_CONFIG`), so no Mem0 state was created or mutated.

## Verification

```sh
conda run -n agentic-rag ruff check src tests evals scripts
# All checks passed
conda run -n agentic-rag mypy src
# Success: no issues found in 87 source files
conda run -n agentic-rag python -m pytest --import-mode=importlib \
  -m "not integration and not e2e and not live_model" -q
# 486 passed, 96 deselected

The final full importlib suite (including opt-in skips) passed 553 tests with
39 skipped tests and 18 expected provider/parser warnings.
conda run -n agentic-rag python -m pytest --import-mode=importlib \
  -o asyncio_default_fixture_loop_scope=module \
  -o asyncio_default_test_loop_scope=module \
  tests/integration/persistence/test_mysql_schema.py -q
# 15 passed
conda run -n agentic-rag python -m pytest --import-mode=importlib \
  -o asyncio_default_fixture_loop_scope=module \
  -o asyncio_default_test_loop_scope=module \
  tests/integration/api/test_documents.py -q
# 4 passed
conda run -n agentic-rag python -m pytest --import-mode=importlib \
  tests/integration/persistence/test_redis_streams.py -q
# 3 passed
conda run -n agentic-rag python -m pytest --import-mode=importlib \
  tests/integration/retrieval/test_elasticsearch_search.py -q
# 1 passed
conda run -n agentic-rag python -m pytest --import-mode=importlib \
  -m live_model tests/smoke -q
# 1 passed
```

`pyproject.toml` now uses pytest's `--import-mode=importlib`, which resolves existing same-basename test collection conflicts without deleting workspace cache files.

## Final acceptance

`evals.run` now writes the five required hard-gate fields from strict case
projections plus isolated local recovery/backup drills. The exact final run
covered all 24 baseline cases:

```text
completed_cases=24
user_leak_count=0
citation_coverage=1.0
unaudited_answer_count=0
recovery_drill_passed=true
backup_restore_passed=true
ACCEPTANCE PASSED
```

Ragas is reported as `status=unavailable` with empty metrics when no backend is
configured; this is explicit and does not affect the deterministic hard gates.
