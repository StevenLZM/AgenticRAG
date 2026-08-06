# Phase 2 Task 5 implementation report

## Status

DONE

## Scope implemented

- Added `VersionPublisher.publish(version_id)` with the required fixed sequence:
  verify the scoped Manifest and Canonical AST Artifacts plus exact Parent/Child
  counts; activate new Parents; activate new Children; deactivate old Children;
  deactivate old Parents; atomically update the MySQL document pointer and
  document/version statuses.
- Added replaceable lifecycle ports and idempotent, exact-scope MySQL and
  Elasticsearch implementations. Elasticsearch lifecycle writes abort and fail
  closed on timeouts, version conflicts, or item failures.
- Added `IngestionReconciler.run_once()` and explicit `ReconcileReport` IDs for
  claimed pending outbox notifications, expired Jobs, stale BUILDING versions,
  classified pointer/status mismatches, and deleted Documents.
- Added atomic Redis delivery-attempt deduplication around `XADD`, reused a durable
  SQL claim lease for reconciler dispatch, and isolated Redis failures per item so
  unrelated lifecycle repair continues. Claim/transport retries keep one stable
  dedupe generation; expired-Job reclaim intentionally rotates it.
- Added pointer-gated Parent hydration. During the required activate-new-before-
  deactivate-old sequence, only Parents belonging to the durable
  `active_version_id` are visible.
- Added explicit deactivation repair for obsolete/concurrent-loser versions and
  invalid pointers. Stale scans exclude queued Jobs and RUNNING Jobs with a live
  lease, preventing active workers from being quarantined by age alone.
- Added convergent deletion cleanup for every trusted version scope. Child and
  Parent removal is idempotent; SQL deletion completion is recorded only after
  both physical stores succeed.
- No ingestion graph or Task 6 worker behavior was added.

## TDD evidence

Initial RED:

```text
conda run -n agentic-rag python -m pytest -m integration tests/integration/ingestion/test_publisher_reconciler.py -q

ModuleNotFoundError: No module named 'agentic_rag.ingestion.publisher'
```

The first GREEN fake-store lifecycle run was `4 passed`. Review-driven RED cases
then reproduced three silently accepted Elasticsearch lifecycle conflicts, the
unclaimed outbox API, and non-deterministic Redis publication:

```text
3 x Failed: DID NOT RAISE ChildIndexWriteError
AttributeError: reconciliation repository has no list_pending_outbox
AttributeError: FakeRedis has no xadd
```

The strengthened offline suite injects a failure after every publication boundary
(including an acknowledged final commit), verifies canonical/Manifest corruption
quarantine, classified obsolete-version deactivation, Redis failure isolation,
deduplicated publication, and repeat-safe deletion.

## Verification

```text
conda run -n agentic-rag python -m pytest tests/unit/ingestion/test_publisher_reconciler_unit.py tests/unit/persistence/test_lifecycle.py tests/unit/persistence/test_elasticsearch_staging.py tests/unit/persistence/test_redis_queue.py tests/unit/persistence/test_outbox_dispatcher.py -q
28 passed

conda run -n agentic-rag python -m pytest -m integration tests/integration/ingestion/test_publisher_reconciler.py -q
1 skipped (explicit disposable MySQL and Elasticsearch DSNs absent)

conda run -n agentic-rag python -m pytest -q
216 passed, 21 skipped, 18 warnings

conda run -n agentic-rag ruff check src tests
All checks passed!

conda run -n agentic-rag mypy src
Success: no issues found in 38 source files

conda run -n agentic-rag env MYPYPATH=src mypy --explicit-package-bases tests/unit/ingestion/test_publisher_reconciler_unit.py tests/unit/persistence/test_lifecycle.py tests/unit/persistence/test_elasticsearch_staging.py tests/unit/persistence/test_redis_queue.py tests/integration/ingestion/test_publisher_reconciler.py
Success: no issues found in 5 source files

git diff --check
clean
```

## Integration and safety notes

- The required real-store integration test is gated on both explicit disposable
  endpoints. When configured it stages and publishes v1, stages v2, injects a
  failure after real Child activation, verifies pointer-gated Parent retrieval,
  repairs through the real SQL reconciler and real ES/MySQL stores, and then
  reconciles deletion.
- Those services were not configured in this environment, so the real-store path
  was collected but skipped rather than guessing or mutating developer services.
- Ephemeral SQL tests execute the live/reclaimed Job exclusion, concurrent older
  loser classification and durable deactivation, pointed-INACTIVE repair,
  ingestion-only outbox claim, stable claim generation, and intentional reclaim
  generation rotation branches without external infrastructure.
- All lifecycle queries and mutations carry trusted user, document, version,
  search-type, and Index Generation boundaries. The ES adapter remains replaceable
  by another `VersionLifecycleStore` such as Milvus.

## Review

The mandatory internal review initially reported ES partial-write handling,
pointer/concurrency convergence, live-job staleness, outbox claim/deduplication,
Redis failure isolation, canonical Artifact verification, and integration-depth
findings. Each was reproduced or covered by a focused regression and fixed. Final
re-review reported no remaining Critical or Important findings.

## Official review fix round 1

The follow-up review of commit `026422df45d48c736e8b803164f86328646e9935`
identified eight Important convergence and integration-depth gaps. This fix round
addresses all eight without adding Task 6 behavior:

- Document deletion now has a durable `pending`/`fenced`/`completed` marker introduced by
  Alembic revision `0004_document_deletion_status`. Selection no longer infers
  cleanup completion from Version status, including deleted Documents with only
  inactive or zero Versions.
- Deletion is a durable two-phase fence: the first pass atomically transitions
  `pending` to `fenced`, records `deletion_fenced_at`, and terminates queued or
  running Job leases. Only a later pass after the configured quiescence interval
  may sweep physical stores and transition to `completed`. A controlled
  interleaving test writes late Parent, Child, and Artifact data after the fence
  and proves the later sweep removes it before completion.
- Deletion now removes the exact trusted `documents/{user}/{document}` Artifact
  tree before recording completion, and atomically terminates queued/running Job
  leases so a stale worker cannot commit after deletion.
- Publication, mismatch repair, and deletion are isolated per item. A permanently
  failing first candidate no longer starves unrelated work in the same bounded
  pass, and reports include only fully repaired IDs.
- `PublicationObsoleteError` distinguishes a newer durable winner from integrity
  corruption. Obsolete candidates are physically deactivated in Child then Parent
  stores before their SQL Version becomes terminal.
- Pointer scans now classify a pointed active Version with a drifted Document
  status as `restore`; the repair rechecks pointer and Version status in a locked
  transaction before restoring Document visibility.
- RUNNING Jobs whose expiry is missing are treated as malformed expired claims and
  reclaimed. Deletion completion clears live worker leases and records
  `document_deleted`.
- The explicit-DSN MySQL/Elasticsearch integration is parameterized over all five
  publication boundaries (`new_parent`, `new_child`, `old_child`, `old_parent`,
  acknowledged `finalize`), follows active Child search results through
  pointer-gated Parent hydration, and adds concurrent v2/v3 finalizers converging
  to the newest winner with the loser physically inactive.
- Real MySQL coverage adds two concurrent `SKIP LOCKED` reconciliation claims and
  a fresh `FOR UPDATE` lock-release assertion. Real Redis coverage executes the
  Lua dedupe path, including publish-success/SQL-mark-failure retry and reclaim
  generation rotation.

Review-fix RED/GREEN evidence included:

```text
inactive deletion / per-item deletion: 12 failed, 1 passed -> 13 passed
publication/deactivation isolation and obsolete loser: 3 failed -> 3 passed
status drift / missing lease / deletion-worker lease: 2 failed, 2 passed -> 4 passed
deletion fence / late in-flight physical writes: review finding -> 3 focused passed
focused lifecycle, Artifact, publisher/reconciler units: 40 passed
```

Fresh verification after the fix round:

```text
conda run -n agentic-rag python -m pytest --collect-only -q \
  tests/integration/ingestion/test_publisher_reconciler.py \
  tests/integration/persistence/test_redis_streams.py \
  tests/integration/persistence/test_mysql_schema.py
23 tests collected

conda run -n agentic-rag python -m pytest -q
229 passed, 28 skipped, 18 warnings

conda run -n agentic-rag ruff check .
All checks passed!

conda run -n agentic-rag mypy src
Success: no issues found in 38 source files

git diff --check
clean
```

The real-service tests were collected but skipped because explicit disposable
MySQL, Elasticsearch, and Redis endpoints were not configured. When a DSN is
configured, the new MySQL/Elasticsearch fixture and Redis Lua test allow
connectivity failures to fail rather than silently substituting guessed services.
