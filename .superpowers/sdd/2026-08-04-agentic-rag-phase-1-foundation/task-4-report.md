# Task 4 Report: Outbox Dispatcher and Redis Stream Broker

## TDD evidence

The dispatcher contract was added before its production modules. The required
red command was run with the requested Conda environment:

```text
conda run -n agentic-rag pytest tests/unit/persistence/test_outbox_dispatcher.py -q
E   ModuleNotFoundError: No module named 'agentic_rag.persistence.outbox'
1 error in 0.04s
```

That failure was the expected absent-dispatcher failure. The minimal
implementation then made the dispatcher test pass. A repository-boundary test
was added for the `claim_pending` and `schedule_retry` extension. That test was
also run red before adding the claim lease; it failed with an `IndexError` while
expecting the missing claim-lease update statement.

## Implementation

- `OutboxDispatcher.dispatch_once()` uses rows claimed through a
  caller-owned transaction, publishes to Redis first, then records dispatch.
  `claim_pending` locks due rows and advances `next_attempt_at` by a 30-second
  lease before Redis is called; after commit, a competing dispatcher cannot
  select those rows until the lease expires. A `RedisError` increments
  `attempt_count`, moves `next_attempt_at` five seconds forward, and leaves the
  row pending.
- `RedisStreamsBroker` uses redis-py asyncio `XADD`, `XREADGROUP`, `XACK`,
  `XAUTOCLAIM`, `XGROUP CREATE`, and `XADD` for dead letters. Notifications
  contain the aggregate ID, so re-published/redelivered messages remain safe
  for a worker to fence through its durable aggregate claim before acknowledgement.
- `OutboxRepository` now exposes `claim_pending` and `schedule_retry`. The
  SQLAlchemy adapter claims with `SELECT ... FOR UPDATE SKIP LOCKED` plus the
  lease update under the caller-owned transaction; it intentionally does not
  call `commit()`. `list_pending` remains a read-only compatibility query.

## Changed files

- `src/agentic_rag/persistence/outbox.py`
- `src/agentic_rag/persistence/redis_queue.py`
- `src/agentic_rag/persistence/repositories.py`
- `tests/unit/persistence/test_outbox_dispatcher.py`
- `tests/unit/persistence/test_repository_contracts.py`
- `tests/integration/persistence/test_redis_streams.py`

## Verification

```text
conda run -n agentic-rag ruff check src tests
All checks passed!

conda run -n agentic-rag mypy src/agentic_rag/persistence --ignore-missing-imports
Success: no issues found in 5 source files

conda run -n agentic-rag pytest -q
30 passed, 13 skipped in 0.28s

conda run -n agentic-rag pytest -m integration tests/integration/persistence/test_redis_streams.py -q
1 skipped in 0.02s
```

The integration test requires an explicit `AGENTIC_RAG_TEST_REDIS_DSN` with a
loopback Redis host. Without it, it skips clearly and touches no Redis
database. It uses unique test stream names and removes only those streams.

## Environment and dependency notes

`redis>=5,<7` was installed into the `agentic-rag` Conda environment because
the project-declared `redis` dependency was not present there. No Docker or
Redis service was started. The dependency is already declared in
`pyproject.toml`; the environment installation is not a source change.

## Commit

`feat: add redis outbox delivery` (this commit includes the implementation,
tests, and this report).

## Concern

The transactional outbox remains at-least-once by design: a crash after Redis
publishes but before the caller commits `mark_dispatched` can create a second
stream message. Consumers must claim/process the aggregate durably before
calling `ack`; the broker exposes the aggregate ID specifically for that fence.
Callers must commit or roll back the session after `dispatch_once`; this task
intentionally preserves Task 3's no-auto-commit repository rule.

## Fix round 1

### Review fixes

- `RedisStreamsBroker` now accepts both redis-py decoded (`str`) stream fields
  and its default byte (`bytes`) stream fields in `consume` and `reclaim`.
  Focused tests exercise both response shapes.
- `OutboxRepository.list_pending` is restored as a side-effect-free due-row
  query. It does not lock rows or move `next_attempt_at`; only
  `claim_pending`, used by the dispatcher, creates the caller-owned lock and
  30-second lease.
- The durable acknowledgement test now calls the Task 3
  `SqlAlchemyRunRepository.claim` boundary. A rejected claim records no ACK;
  a successful claim performs its durable update/read before the ACK recorder
  observes acknowledgement. This replaces the previous in-memory set as the
  safety assertion.

### Fix-round TDD evidence

The new byte/string and read-only-list tests were run before production edits:

```text
conda run -n agentic-rag pytest tests/unit/persistence/test_redis_queue.py \
  tests/unit/persistence/test_repository_contracts.py::test_list_pending_does_not_claim_or_lease_outbox_rows -q

2 failed, 1 passed
KeyError: 'aggregate_id'  # byte-keyed Redis fields
assert 2 == 1             # list_pending performed a lease update
```

### Fix-round verification

```text
conda run -n agentic-rag ruff check src tests
All checks passed!

conda run -n agentic-rag mypy src/agentic_rag/persistence --ignore-missing-imports
Success: no issues found in 5 source files

conda run -n agentic-rag pytest -q
34 passed, 13 skipped in 0.21s

conda run -n agentic-rag pytest -m integration tests/integration/persistence/test_redis_streams.py -q
1 skipped in 0.02s
```

The explicit loopback-only `AGENTIC_RAG_TEST_REDIS_DSN` gate remains unchanged;
no Redis instance was started for this fix round.

## Fix round 2

### Durable integration coverage

- `test_worker_commits_run_claim_before_redis_ack` reuses the existing explicit
  disposable-MySQL migration fixture and requires the existing loopback-only
  `AGENTIC_RAG_TEST_REDIS_DSN`. It commits a successful
  `SqlAlchemyRunRepository.claim` before `XACK`, verifies the committed running
  state while the stream entry is still pending, and then acknowledges it. It
  also verifies rejected and deliberately rolled-back claims leave their Redis
  entries pending and leave the rolled-back run queued.
- `test_outbox_claim_lease_excludes_an_independent_session` opens two separate
  MySQL sessions. It confirms that the first session's `FOR UPDATE SKIP LOCKED`
  claim hides the controlled due row from the second session, then confirms the
  committed lease continues to prevent a later claim. Neither repository method
  commits on behalf of its caller.

### Verification

```text
conda run -n agentic-rag ruff check src tests
All checks passed!

conda run -n agentic-rag mypy src/agentic_rag/persistence --ignore-missing-imports
Success: no issues found in 5 source files

conda run -n agentic-rag pytest tests/unit -q
34 passed in 0.14s

conda run -n agentic-rag pytest -m integration tests/integration/persistence/test_mysql_schema.py -q
14 skipped in 0.15s
```

No `AGENTIC_RAG_TEST_MYSQL_DSN` or `AGENTIC_RAG_TEST_REDIS_DSN` was supplied in
this environment, so the new end-to-end tests correctly skipped rather than
guessing at or mutating a database. No Docker or local service was started.
