"""Lease-fenced Redis Stream worker for durable checkpointed query runs."""

from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable
from contextlib import suppress
from datetime import UTC, datetime
from typing import Protocol, cast

from langgraph.checkpoint.base import BaseCheckpointSaver

from agentic_rag.domain.models import RunStatus, UserScope
from agentic_rag.observability.logging import (
    AgentEventEmitter,
    event_emission_scope,
    stable_event_key,
)
from agentic_rag.observability.tracing import TraceRecorder
from agentic_rag.persistence.redis_queue import StreamBroker, StreamMessage
from agentic_rag.persistence.repositories import LeaseLost, QueryRun, RunRepository
from agentic_rag.query.state import new_query_state
from agentic_rag.runtime.concurrency import ConcurrencyManager
from agentic_rag.runtime.models import RuntimeConfigSnapshot


QUERY_STREAM = "agenticrag:jobs:query"
QUERY_GROUP = "agenticrag-query-workers"
QUERY_DEAD_STREAM = "agenticrag:jobs:query:dead"
TERMINAL_RUN_STATUSES = {RunStatus.CANCELLED, RunStatus.COMPLETED, RunStatus.FAILED}
BUSINESS_TERMINAL_REASONS = frozenset(
    {
        "completed",
        "clarify",
        "refuse",
        "cannot_answer",
        "research_action_invalid",
        "audit_failed",
        "research_round_limit",
    }
)


class QueryGraph(Protocol):
    async def ainvoke(self, state: dict[str, object], config: dict[str, object]) -> dict[str, object]: ...


QueryGraphFactory = Callable[..., Awaitable[QueryGraph] | QueryGraph]


class RunCancelled(RuntimeError):
    """The durable cancellation flag was observed by the worker heartbeat."""


def build_graph_factory(
    dependencies: object, checkpointer: BaseCheckpointSaver[str]
) -> QueryGraphFactory:
    """Bind process-owned graph services to one durable checkpoint backend.

    The message only selects its server-derived checkpoint thread id; it never
    supplies services or a checkpointer as part of Redis-delivered state.
    """
    from agentic_rag.query.graph import QueryGraphDependencies, build_query_graph

    if not isinstance(dependencies, QueryGraphDependencies):
        raise TypeError("dependencies must be QueryGraphDependencies")

    def factory(*, checkpoint_thread_id: str) -> QueryGraph:
        if not checkpoint_thread_id.startswith("query:"):
            raise ValueError("query checkpoint thread_id must use query namespace")
        return cast(QueryGraph, build_query_graph(dependencies, checkpointer))

    return factory


class QueryWorker:
    """Claim once, resume the fixed graph checkpoint, then ACK terminal runs only."""

    def __init__(
        self,
        *,
        runs: RunRepository,
        broker: StreamBroker,
        graph_factory: QueryGraphFactory,
        worker_id: str,
        concurrency: ConcurrencyManager | None = None,
        lease_seconds: int = 30,
        heartbeat_interval_seconds: float = 10,
        run_timeout_seconds: float = 300,
        reclaim_idle_ms: int = 30_000,
        block_ms: int = 1_000,
        max_attempts: int = 3,
        trace_recorder: TraceRecorder | None = None,
        event_emitter: AgentEventEmitter | None = None,
    ) -> None:
        if not worker_id.strip():
            raise ValueError("worker_id must not be blank")
        if lease_seconds <= 0 or heartbeat_interval_seconds <= 0:
            raise ValueError("lease intervals must be positive")
        if heartbeat_interval_seconds >= lease_seconds:
            raise ValueError("heartbeat interval must be shorter than lease")
        if run_timeout_seconds <= 0 or reclaim_idle_ms < 0 or block_ms < 0:
            raise ValueError("worker timeouts must be bounded")
        if max_attempts < 1:
            raise ValueError("max_attempts must be positive")
        self._runs = runs
        self._broker = broker
        self._graph_factory = graph_factory
        self._worker_id = worker_id
        self._concurrency = concurrency or ConcurrencyManager()
        self._lease_seconds = lease_seconds
        self._heartbeat_interval = heartbeat_interval_seconds
        self._run_timeout = run_timeout_seconds
        self._reclaim_idle_ms = reclaim_idle_ms
        self._block_ms = block_ms
        self._max_attempts = max_attempts
        self._trace_recorder = trace_recorder
        self._event_emitter = event_emitter

    async def run_one(self) -> bool:
        """Drain one Redis delivery batch, preferring reclaimed messages first."""
        reclaimed = await self._broker.reclaim(
            QUERY_STREAM, QUERY_GROUP, self._worker_id, self._reclaim_idle_ms
        )
        fresh = await self._broker.consume(
            QUERY_STREAM, QUERY_GROUP, self._worker_id, self._block_ms
        )
        messages = (*reclaimed, *fresh)
        if not messages:
            return False
        for message in messages:
            await self.process_message(message)
        return True

    async def process_message(self, message: StreamMessage) -> None:
        claim = await self._runs.claim(message.aggregate_id, self._worker_id, self._lease_seconds)
        if claim is None:
            existing = await self._runs.get_for_delivery(message.aggregate_id)
            if existing is not None and existing.status in TERMINAL_RUN_STATUSES:
                await self._broker.ack(QUERY_STREAM, QUERY_GROUP, message.id)
            return
        snapshot = RuntimeConfigSnapshot.model_validate(claim.runtime_config_snapshot)
        run_started = time.perf_counter()
        await self._emit_run_lifecycle(
            claim,
            "RUN_STARTED",
            started_monotonic=run_started,
        )
        queue_wait = max(0.0, (datetime.now(UTC) - message.enqueued_at).total_seconds())
        if (
            self._event_emitter is not None
            and self._event_emitter.runtime_config_snapshot_id == snapshot.snapshot_id
        ):
            try:
                await self._event_emitter.emit(
                    run_id=claim.id,
                    user_id=claim.user_id,
                    event_type="QUEUE_WAITED",
                    summary="completed",
                    event_key=stable_event_key(
                        claim.id, message.id, "QUEUE_WAITED"
                    ),
                    attributes={"queue_wait_seconds": queue_wait},
                )
            except (asyncio.CancelledError, KeyboardInterrupt, SystemExit):
                raise
            except Exception:
                pass
        recorder = (
            self._trace_recorder
            if self._trace_recorder is not None
            and self._trace_recorder.runtime_config_snapshot_id == snapshot.snapshot_id
            else None
        )
        emitter = (
            self._event_emitter
            if self._event_emitter is not None
            and self._event_emitter.runtime_config_snapshot_id == snapshot.snapshot_id
            else None
        )
        if emitter is not None:
            async with event_emission_scope(emitter, claim.id, "queue", user_id=claim.user_id):
                if recorder is not None:
                    async with recorder.span(
                        "queue", run_id=claim.id, attributes={"queue_wait_seconds": queue_wait}
                    ):
                        await self._process_claimed_message(
                            message, claim, started_monotonic=run_started
                        )
                else:
                    await self._process_claimed_message(
                        message, claim, started_monotonic=run_started
                    )
            return
        if recorder is not None:
            async with recorder.span(
                "queue", run_id=claim.id, attributes={"queue_wait_seconds": queue_wait}
            ):
                await self._process_claimed_message(
                    message, claim, started_monotonic=run_started
                )
            return
        await self._process_claimed_message(message, claim, started_monotonic=run_started)

    async def _process_claimed_message(
        self,
        message: StreamMessage,
        claim: QueryRun,
        *,
        started_monotonic: float,
    ) -> None:
        if claim.status is RunStatus.CANCEL_REQUESTED:
            await self._finish_and_ack(
                message,
                claim,
                RunStatus.CANCELLED,
                None,
                started_monotonic=started_monotonic,
                termination_reason="cancelled",
            )
            return
        if not claim.question.strip():
            await self._finish_and_ack(
                message,
                claim,
                RunStatus.FAILED,
                "missing_query",
                started_monotonic=started_monotonic,
                termination_reason="failed",
            )
            return
        try:
            async with self._concurrency.run_slot():
                async with asyncio.timeout(
                    min(self._run_timeout, float(RuntimeConfigSnapshot.model_validate(claim.runtime_config_snapshot).query_run_timeout_seconds))
                ):
                    result = await self._invoke_with_heartbeat(claim)
        except RunCancelled:
            await self._finish_and_ack(
                message,
                claim,
                RunStatus.CANCELLED,
                None,
                started_monotonic=started_monotonic,
                termination_reason="cancelled",
            )
            return
        except asyncio.TimeoutError:
            await self._finish_and_ack(
                message,
                claim,
                RunStatus.FAILED,
                "run_timeout",
                started_monotonic=started_monotonic,
                termination_reason="failed",
            )
            return
        except LeaseLost:
            return
        except asyncio.CancelledError:
            raise
        except Exception as error:
            await self._handle_execution_failure(
                message,
                claim,
                type(error).__name__,
                started_monotonic=started_monotonic,
            )
            return

        if not isinstance(result, dict):
            await self._handle_execution_failure(
                message,
                claim,
                "invalid_graph_result",
                started_monotonic=started_monotonic,
            )
            return
        termination = result.get("termination_reason")
        if not isinstance(termination, str):
            await self._handle_execution_failure(
                message,
                claim,
                "invalid_termination",
                started_monotonic=started_monotonic,
            )
            return
        if termination == "cancelled":
            terminal = RunStatus.CANCELLED
        elif termination in BUSINESS_TERMINAL_REASONS:
            terminal = RunStatus.COMPLETED
        else:
            await self._handle_execution_failure(
                message,
                claim,
                "invalid_termination",
                started_monotonic=started_monotonic,
            )
            return
        answer = result.get("answer")
        if answer is None and terminal is RunStatus.COMPLETED:
            answer = {"status": termination}
        if not isinstance(answer, dict) and answer is not None:
            await self._handle_execution_failure(
                message,
                claim,
                "invalid_graph_result",
                started_monotonic=started_monotonic,
            )
            return
        await self._finish_and_ack(
            message,
            claim,
            terminal,
            None,
            started_monotonic=started_monotonic,
            termination_reason=termination,
            answer=cast(dict[str, object] | None, answer),
        )

    async def run_forever(self, *, stop_event: asyncio.Event | None = None) -> None:
        stop = stop_event or asyncio.Event()
        current: asyncio.Task[bool] | None = None
        failures = 0
        try:
            while not stop.is_set():
                current = asyncio.create_task(self.run_one())
                try:
                    await asyncio.shield(current)
                    failures = 0
                except asyncio.CancelledError:
                    stop.set()
                    with suppress(asyncio.CancelledError):
                        await current
                    raise
                except Exception:
                    failures += 1
                    await asyncio.sleep(min(0.25 * (2 ** min(failures - 1, 4)), 5.0))
                finally:
                    current = None
        finally:
            if current is not None:
                with suppress(asyncio.CancelledError):
                    await current

    async def _invoke_with_heartbeat(self, claim: QueryRun) -> dict[str, object]:
        graph = self._graph_factory(checkpoint_thread_id=claim.checkpoint_thread_id)
        if hasattr(graph, "__await__"):
            graph = await graph  # type: ignore[assignment,misc]
        state = new_query_state(
            run_id=claim.id, question=claim.question, scope=UserScope(user_id=claim.user_id),
            snapshot=RuntimeConfigSnapshot.model_validate(claim.runtime_config_snapshot),
        )
        config: dict[str, object] = {"configurable": {"thread_id": claim.checkpoint_thread_id}}
        task = asyncio.create_task(graph.ainvoke(dict(state), config))  # type: ignore[union-attr]
        try:
            while True:
                done, _ = await asyncio.wait({task}, timeout=self._heartbeat_interval)
                if done:
                    return await task
                current = await self._runs.get(
                    claim.id, UserScope(user_id=claim.user_id)
                )
                if current is None:
                    raise LeaseLost(claim.id)
                if current.status in {RunStatus.CANCEL_REQUESTED, RunStatus.CANCELLED}:
                    task.cancel()
                    with suppress(asyncio.CancelledError):
                        await task
                    raise RunCancelled(claim.id)
                await self._runs.heartbeat(
                    claim.id, self._worker_id, self._lease_seconds,
                    claim_generation=claim.claim_generation,
                )
        finally:
            if not task.done():
                task.cancel()
                with suppress(asyncio.CancelledError):
                    await task

    async def _finish_and_ack(
        self,
        message: StreamMessage,
        claim: QueryRun,
        status: RunStatus,
        error_code: str | None,
        *,
        started_monotonic: float,
        termination_reason: str,
        answer: dict[str, object] | None = None,
    ) -> None:
        try:
            try:
                await self._runs.finish(
                    claim.id,
                    status,
                    None,
                    error_code,
                    owner=self._worker_id,
                    claim_generation=claim.claim_generation,
                    answer=answer,
                )
            except TypeError as error:
                # Keep deployment-owned lightweight Run ports source-compatible
                # while they migrate to the answer projection.  Do not swallow
                # unrelated TypeErrors raised inside repository implementations.
                if "unexpected keyword argument 'answer'" not in str(error):
                    raise
                await self._runs.finish(
                    claim.id,
                    status,
                    None,
                    error_code,
                    owner=self._worker_id,
                    claim_generation=claim.claim_generation,
                )
        except LeaseLost:
            # The lease fence is authoritative.  Leave the Redis delivery
            # pending for reclaim and do not append or acknowledge a terminal
            # event owned by a different worker.
            return
        await self._emit_run_lifecycle(
            claim,
            _run_event_type(status),
            started_monotonic=started_monotonic,
            termination_reason=termination_reason,
        )
        await self._broker.ack(QUERY_STREAM, QUERY_GROUP, message.id)

    async def _dead_letter_and_fail(
        self,
        message: StreamMessage,
        claim: QueryRun,
        reason: str,
        *,
        started_monotonic: float,
    ) -> None:
        await self._broker.dead_letter(
            QUERY_DEAD_STREAM, message, reason,
            dedupe_key=f"query-dead:{claim.id}:{claim.claim_generation}",
        )
        await self._finish_and_ack(
            message,
            claim,
            RunStatus.FAILED,
            reason,
            started_monotonic=started_monotonic,
            termination_reason="failed",
        )

    async def _handle_execution_failure(
        self,
        message: StreamMessage,
        claim: QueryRun,
        reason: str,
        *,
        started_monotonic: float,
    ) -> None:
        if claim.claim_generation >= self._max_attempts:
            await self._dead_letter_and_fail(
                message,
                claim,
                reason,
                started_monotonic=started_monotonic,
            )

    async def _emit_run_lifecycle(
        self,
        claim: QueryRun,
        event_type: str,
        *,
        started_monotonic: float,
        termination_reason: str | None = None,
    ) -> None:
        """Emit authoritative run lifecycle metadata after the lease boundary.

        ``RUN_STARTED`` is written immediately after a successful claim.  A
        terminal event is written only after ``RunRepository.finish`` returns,
        so a stale owner that loses its lease cannot publish an authoritative
        outcome.  The event key is stable across Redis redelivery and the
        measured duration is carried as a trusted, server-observed attribute.
        """
        emitter = self._event_emitter
        if emitter is None:
            return
        try:
            snapshot_id = RuntimeConfigSnapshot.model_validate(
                claim.runtime_config_snapshot
            ).snapshot_id
        except (TypeError, ValueError):
            return
        if emitter.runtime_config_snapshot_id != snapshot_id:
            return
        attributes: dict[str, object] = {}
        if event_type != "RUN_STARTED":
            attributes["run_latency_seconds"] = max(
                0.0, time.perf_counter() - started_monotonic
            )
            if termination_reason in {
                "completed",
                "clarify",
                "refuse",
                "cannot_answer",
                "research_action_invalid",
                "audit_failed",
                "research_round_limit",
                "cancelled",
                "failed",
            }:
                attributes["termination_reason"] = termination_reason
        summary = {
            "RUN_STARTED": "started",
            "RUN_COMPLETED": "completed",
            "RUN_FAILED": "failed",
            "RUN_CANCELLED": "cancelled",
        }.get(event_type, "telemetry event")
        try:
            await emitter.emit(
                run_id=claim.id,
                user_id=claim.user_id,
                event_type=event_type,
                summary=summary,
                # A Run has one durable lifecycle even when Redis reclaims a
                # delivery and increments the attempt/claim generation.  The
                # run-level key therefore deduplicates starts and terminal
                # outcomes across all ownership attempts.
                event_key=stable_event_key(claim.id, event_type),
                attributes=attributes,
            )
        except (asyncio.CancelledError, KeyboardInterrupt, SystemExit):
            raise
        except Exception:
            # Telemetry is informational.  A broken event sink must not turn a
            # durable terminal Run into a retry or a duplicate execution.
            return


def _run_event_type(status: RunStatus) -> str:
    return {
        RunStatus.COMPLETED: "RUN_COMPLETED",
        RunStatus.FAILED: "RUN_FAILED",
        RunStatus.CANCELLED: "RUN_CANCELLED",
    }[status]
