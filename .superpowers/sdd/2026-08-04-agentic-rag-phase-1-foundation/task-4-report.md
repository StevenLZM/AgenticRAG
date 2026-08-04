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
  call `commit()`. `list_pending` remains as a compatibility alias.

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
