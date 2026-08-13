"""Durability contracts for the Phase 4 query-run worker.

These tests deliberately use in-memory ports: MySQL/Redis integration coverage
belongs in the service suite, while the lifecycle fencing must also be
executable in the default local test environment.
"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from typing import AsyncIterator

import pytest

from agentic_rag.domain.models import RunStatus, UserScope
from agentic_rag.persistence.redis_queue import StreamMessage
from agentic_rag.persistence.repositories import ActiveRunConflict, QueryRun
from agentic_rag.runtime.models import RuntimeConfigSnapshot


SCOPE = UserScope(user_id="user-1")
SNAPSHOT = RuntimeConfigSnapshot(
    app_version="test", graph_version="query-v1", prompt_version="prompt-v1",
    main_model_id="main", light_model_id="light", embedding_model="embedding",
    embedding_dimensions=1024, reranker_version="reranker",
    retrieval_config_version="retrieval", index_generation="index-v1",
    memory_config_version="memory-v1",
)


@dataclass
class RecordingTransaction:
    committed: bool = False


@dataclass
class RecordingSessions:
    transaction: RecordingTransaction = field(default_factory=RecordingTransaction)

    @asynccontextmanager
    async def begin(self) -> AsyncIterator[RecordingTransaction]:
        yield self.transaction
        self.transaction.committed = True


@dataclass
class Runs:
    run: QueryRun | None = None
    runs: dict[str, QueryRun] = field(default_factory=dict)
    creates: int = 0
    outbox_created: list[str] = field(default_factory=list)
    finishes: list[RunStatus] = field(default_factory=list)
    answers: list[dict[str, object] | None] = field(default_factory=list)
    claimed: int = 0

    async def create_queued(
        self, scope: UserScope, thread_id: str, snapshot: RuntimeConfigSnapshot, *, question: str = "", transaction: object | None = None
    ) -> QueryRun:
        del transaction
        self.creates += 1
        if any(current.status in {RunStatus.QUEUED, RunStatus.RUNNING, RunStatus.CANCEL_REQUESTED} for current in self.runs.values()):
            raise ActiveRunConflict("active run")
        run_id = f"run-{self.creates}"
        self.run = QueryRun(
            id=run_id, user_id=scope.user_id, thread_id=thread_id, question=question,
            checkpoint_thread_id=f"query:{scope.user_id}:{thread_id}", status=RunStatus.QUEUED,
            active_slot=1, runtime_config_snapshot_id=snapshot.snapshot_id,
            runtime_config_snapshot=snapshot.model_dump(mode="json"),
        )
        self.runs[self.run.id] = self.run
        self.outbox_created.append(self.run.id)
        return self.run

    async def get(self, run_id: str, scope: UserScope) -> QueryRun | None:
        run = self.runs.get(run_id)
        return run if run is not None and run.user_id == scope.user_id else None

    async def get_for_delivery(self, run_id: str) -> QueryRun | None:
        return self.runs.get(run_id)

    async def claim(self, run_id: str, owner: str, lease_seconds: int) -> QueryRun | None:
        del owner, lease_seconds
        current = self.runs.get(run_id)
        if current is None or current.status is not RunStatus.QUEUED:
            return None
        self.claimed += 1
        self.run = replace(current, status=RunStatus.RUNNING, claim_generation=self.claimed)
        self.runs[run_id] = self.run
        return self.run

    async def heartbeat(self, *args: object, **kwargs: object) -> None:
        del args, kwargs

    async def request_cancel(self, run_id: str, scope: UserScope) -> RunStatus:
        current = self.runs[run_id]
        assert current.user_id == scope.user_id
        self.run = replace(current, status=RunStatus.CANCEL_REQUESTED)
        self.runs[run_id] = self.run
        return self.run.status

    async def finish(self, run_id: str, status: RunStatus, result_ref: str | None, error_code: str | None, *, owner: str, claim_generation: int, answer: dict[str, object] | None = None) -> None:
        del result_ref, error_code, owner, claim_generation
        assert run_id in self.runs
        self.finishes.append(status)
        self.answers.append(answer)
        self.run = replace(self.runs[run_id], status=status, active_slot=None)
        self.runs[run_id] = self.run


@dataclass
class Broker:
    messages: list[StreamMessage] = field(default_factory=list)
    reclaimed: list[StreamMessage] = field(default_factory=list)
    acknowledged: list[str] = field(default_factory=list)
    dead: list[str] = field(default_factory=list)
    consume_error: Exception | None = None

    async def consume(self, *args: object, **kwargs: object) -> list[StreamMessage]:
        del args, kwargs
        if self.consume_error is not None:
            raise self.consume_error
        result, self.messages = self.messages, []
        return result

    async def reclaim(self, *args: object, **kwargs: object) -> list[StreamMessage]:
        del args, kwargs
        result, self.reclaimed = self.reclaimed, []
        return result

    async def ack(self, stream: str, group: str, message_id: str) -> None:
        del stream, group
        self.acknowledged.append(message_id)

    async def dead_letter(self, dead_stream: str, message: StreamMessage, reason: str, dedupe_key: str | None = None) -> None:
        del dead_stream, reason, dedupe_key
        self.dead.append(message.id)


@dataclass
class GraphFactory:
    calls: list[str] = field(default_factory=list)

    def __call__(self, *, checkpoint_thread_id: str) -> "_Graph":
        self.calls.append(checkpoint_thread_id)
        return _Graph()


async def test_create_stages_queued_run_and_outbox_in_one_transaction() -> None:
    from agentic_rag.runtime.run_manager import RunManager

    sessions, runs = RecordingSessions(), Runs()
    manager = RunManager(session_factory=sessions, runs=runs)

    run = await manager.create(SCOPE, "thread-1", "What notice applies?", SNAPSHOT)

    assert run.status is RunStatus.QUEUED
    assert runs.outbox_created == [run.id]
    assert sessions.transaction.committed is True


async def test_sql_run_repository_stages_the_outbox_in_manager_transaction() -> None:
    from agentic_rag.persistence.repositories import SqlAlchemyRunRepository
    from agentic_rag.runtime.run_manager import RunManager

    class Session:
        statements: list[object] = []

        async def execute(self, statement: object) -> object:
            self.statements.append(statement)
            return object()

    session = Session()

    @asynccontextmanager
    async def transaction() -> AsyncIterator[object]:
        yield session

    class Sessions:
        def begin(self) -> object:
            return transaction()

    manager = RunManager(session_factory=Sessions(), runs=SqlAlchemyRunRepository())  # type: ignore[arg-type]
    await manager.create(SCOPE, "thread-1", "What notice applies?", SNAPSHOT)

    assert [statement.table.name for statement in session.statements] == ["agent_runs", "task_outbox"]


async def test_active_thread_conflict_does_not_create_a_second_outbox() -> None:
    from agentic_rag.runtime.run_manager import RunManager

    manager = RunManager(session_factory=RecordingSessions(), runs=Runs())
    await manager.create(SCOPE, "thread-1", "first", SNAPSHOT)

    with pytest.raises(ActiveRunConflict):
        await manager.create(SCOPE, "thread-1", "second", SNAPSHOT)

    assert manager.runs.outbox_created == ["run-1"]


async def test_worker_claims_invokes_stable_checkpoint_and_acks_only_terminal_run() -> None:
    from agentic_rag.runtime.query_worker import QueryWorker

    runs = Runs()
    run = await runs.create_queued(
        SCOPE, "thread-1", SNAPSHOT, question="What notice applies?"
    )
    broker = Broker(messages=[StreamMessage("1-0", run.id, datetime.now(UTC))])
    factory = GraphFactory()
    worker = QueryWorker(runs=runs, broker=broker, graph_factory=factory, worker_id="worker-1")
    await worker.run_one()

    assert factory.calls == ["query:user-1:thread-1"]
    assert runs.finishes == [RunStatus.COMPLETED]
    assert runs.answers == [{"status": "audited"}]
    assert broker.acknowledged == ["1-0"]


@pytest.mark.parametrize(
    "termination_reason",
    [
        "clarify",
        "refuse",
        "cannot_answer",
        "research_action_invalid",
        "audit_failed",
        "research_round_limit",
    ],
)
async def test_worker_persists_business_terminal_reasons_as_completed(
    termination_reason: str,
) -> None:
    """A user-facing refusal/clarification is a completed, audited Run outcome."""
    from agentic_rag.runtime.query_worker import QueryWorker

    runs = Runs()
    run = await runs.create_queued(
        SCOPE, "thread-1", SNAPSHOT, question="What notice applies?"
    )
    broker = Broker(messages=[StreamMessage("1-0", run.id, datetime.now(UTC))])

    class TerminalGraph:
        async def ainvoke(
            self, state: dict[str, object], config: dict[str, object]
        ) -> dict[str, object]:
            del state, config
            return {"termination_reason": termination_reason}

    worker = QueryWorker(
        runs=runs,
        broker=broker,
        graph_factory=lambda **_: TerminalGraph(),
        worker_id="worker-1",
    )
    await worker.run_one()

    assert runs.finishes == [RunStatus.COMPLETED]
    assert runs.answers == [{"status": termination_reason}]
    assert broker.dead == []
    assert broker.acknowledged == ["1-0"]


class _Graph:
    async def ainvoke(self, state: dict[str, object], config: dict[str, object]) -> dict[str, object]:
        del state, config
        return {"termination_reason": "completed", "answer": {"status": "audited"}}


def _queued_peer(run: QueryRun, run_id: str = "run-2") -> QueryRun:
    return replace(
        run,
        id=run_id,
        thread_id=f"{run.thread_id}-{run_id}",
        checkpoint_thread_id=f"query:{run.user_id}:{run.thread_id}-{run_id}",
        status=RunStatus.QUEUED,
        active_slot=1,
        claim_generation=0,
    )


async def test_worker_drains_every_fresh_message_in_one_delivery_batch() -> None:
    from agentic_rag.runtime.query_worker import QueryWorker

    runs = Runs()
    first = await runs.create_queued(SCOPE, "thread-1", SNAPSHOT, question="first")
    second = _queued_peer(first)
    runs.runs[second.id] = second
    broker = Broker(messages=[
        StreamMessage("1-0", first.id, datetime.now(UTC)),
        StreamMessage("2-0", second.id, datetime.now(UTC)),
    ])
    worker = QueryWorker(runs=runs, broker=broker, graph_factory=lambda **_: _Graph(), worker_id="worker-1")

    processed = await worker.run_one()

    assert processed is True
    assert runs.finishes == [RunStatus.COMPLETED, RunStatus.COMPLETED]
    assert broker.acknowledged == ["1-0", "2-0"]


async def test_worker_drains_reclaimed_before_fresh_without_losing_fresh_batch() -> None:
    from agentic_rag.runtime.query_worker import QueryWorker

    runs = Runs()
    reclaimed_run = await runs.create_queued(SCOPE, "thread-1", SNAPSHOT, question="reclaimed")
    fresh_run = _queued_peer(reclaimed_run)
    runs.runs[fresh_run.id] = fresh_run
    broker = Broker(
        reclaimed=[StreamMessage("1-0", reclaimed_run.id, datetime.now(UTC))],
        messages=[StreamMessage("2-0", fresh_run.id, datetime.now(UTC))],
    )
    worker = QueryWorker(runs=runs, broker=broker, graph_factory=lambda **_: _Graph(), worker_id="worker-1")

    await worker.run_one()

    assert broker.acknowledged == ["1-0", "2-0"]


async def test_duplicate_delivery_that_cannot_claim_is_not_executed_or_acked() -> None:
    from agentic_rag.runtime.query_worker import QueryWorker

    runs = Runs()
    run = await runs.create_queued(SCOPE, "thread-1", SNAPSHOT, question="What notice applies?")
    broker = Broker(messages=[StreamMessage("1-0", run.id, datetime.now(UTC))])
    await runs.claim(run.id, "other-worker", 30)

    worker = QueryWorker(runs=runs, broker=broker, graph_factory=lambda **_: _Graph(), worker_id="worker-1")
    await worker.run_one()

    assert runs.finishes == []
    assert broker.acknowledged == []


async def test_duplicate_terminal_delivery_is_acked_without_graph_execution() -> None:
    from agentic_rag.runtime.query_worker import QueryWorker

    runs = Runs()
    run = await runs.create_queued(SCOPE, "thread-1", SNAPSHOT, question="What notice applies?")
    runs.run = replace(run, status=RunStatus.COMPLETED, active_slot=None)
    runs.runs[run.id] = runs.run
    broker = Broker(messages=[StreamMessage("1-0", run.id, datetime.now(UTC))])
    worker = QueryWorker(runs=runs, broker=broker, graph_factory=lambda **_: _Graph(), worker_id="worker-1")

    await worker.run_one()

    assert runs.finishes == []
    assert broker.acknowledged == ["1-0"]


async def test_worker_cancels_run_when_cancellation_is_observed_during_execution() -> None:
    from agentic_rag.runtime.query_worker import QueryWorker

    runs = Runs()
    run = await runs.create_queued(SCOPE, "thread-1", SNAPSHOT, question="What notice applies?")
    broker = Broker(messages=[StreamMessage("1-0", run.id, datetime.now(UTC))])
    started = asyncio.Event()

    class BlockingGraph:
        async def ainvoke(self, state: dict[str, object], config: dict[str, object]) -> dict[str, object]:
            del state, config
            started.set()
            await asyncio.Event().wait()
            return {}

    worker = QueryWorker(
        runs=runs, broker=broker, graph_factory=lambda **_: BlockingGraph(), worker_id="worker-1",
        lease_seconds=2, heartbeat_interval_seconds=0.01,
    )
    task = asyncio.create_task(worker.run_one())
    await started.wait()
    await runs.request_cancel(run.id, SCOPE)
    await task

    assert runs.finishes == [RunStatus.CANCELLED]
    assert broker.acknowledged == ["1-0"]


async def test_worker_fails_and_acks_after_run_timeout() -> None:
    from agentic_rag.runtime.query_worker import QueryWorker

    runs = Runs()
    run = await runs.create_queued(SCOPE, "thread-1", SNAPSHOT, question="What notice applies?")
    broker = Broker(messages=[StreamMessage("1-0", run.id, datetime.now(UTC))])

    class BlockingGraph:
        async def ainvoke(self, state: dict[str, object], config: dict[str, object]) -> dict[str, object]:
            del state, config
            await asyncio.Event().wait()
            return {}

    worker = QueryWorker(
        runs=runs, broker=broker, graph_factory=lambda **_: BlockingGraph(), worker_id="worker-1",
        run_timeout_seconds=0.01,
    )
    await worker.run_one()

    assert runs.finishes == [RunStatus.FAILED]
    assert broker.acknowledged == ["1-0"]


async def test_third_claim_failure_dead_letters_then_acks() -> None:
    from agentic_rag.runtime.query_worker import QueryWorker

    runs = Runs()
    run = await runs.create_queued(SCOPE, "thread-1", SNAPSHOT, question="What notice applies?")
    runs.claimed = 2
    broker = Broker(messages=[StreamMessage("1-0", run.id, datetime.now(UTC))])

    class BrokenGraph:
        async def ainvoke(self, state: dict[str, object], config: dict[str, object]) -> dict[str, object]:
            del state, config
            raise RuntimeError("boom")

    worker = QueryWorker(runs=runs, broker=broker, graph_factory=lambda **_: BrokenGraph(), worker_id="worker-1")
    await worker.run_one()

    assert runs.finishes == [RunStatus.FAILED]
    assert broker.dead == ["1-0"]
    assert broker.acknowledged == ["1-0"]


async def test_invalid_graph_result_fails_closed_after_maximum_attempts() -> None:
    from agentic_rag.runtime.query_worker import QueryWorker

    runs = Runs(claimed=2)
    run = await runs.create_queued(SCOPE, "thread-1", SNAPSHOT, question="What notice applies?")
    broker = Broker(messages=[StreamMessage("1-0", run.id, datetime.now(UTC))])

    class InvalidGraph:
        async def ainvoke(self, state: dict[str, object], config: dict[str, object]) -> dict[str, object]:
            del state, config
            return {"termination_reason": "unexpected"}

    worker = QueryWorker(runs=runs, broker=broker, graph_factory=lambda **_: InvalidGraph(), worker_id="worker-1")
    await worker.run_one()

    assert runs.finishes == [RunStatus.FAILED]
    assert broker.dead == ["1-0"]


async def test_missing_graph_termination_fails_closed_after_maximum_attempts() -> None:
    from agentic_rag.runtime.query_worker import QueryWorker

    runs = Runs(claimed=2)
    run = await runs.create_queued(SCOPE, "thread-1", SNAPSHOT, question="What notice applies?")
    broker = Broker(messages=[StreamMessage("1-0", run.id, datetime.now(UTC))])

    class IncompleteGraph:
        async def ainvoke(self, state: dict[str, object], config: dict[str, object]) -> dict[str, object]:
            del state, config
            return {"answer": {"status": "unverified"}}

    worker = QueryWorker(runs=runs, broker=broker, graph_factory=lambda **_: IncompleteGraph(), worker_id="worker-1")
    await worker.run_one()

    assert runs.finishes == [RunStatus.FAILED]
    assert broker.dead == ["1-0"]
